"""Linear mixed-effects model — lme4-style profiled deviance.

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

from dataclasses import dataclass, field
from enum import IntEnum
import warnings
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.sparse import csc_array, eye_array

from .family import Family, Gaussian
from .formula import (
    BinOp,
    ExpandedFormula,
    ReTerms,
    _bar_lhs_to_ef,
    _eval_atom,
    _eval_group,
    _flatten_nested_group,
    _materialize_re_lhs,
    materialize,
    materialize_bars,
)
from .design import prepare_design
from .lm import _label_top_n, _lowess, _qq_plot
from .utils import format_df, format_signif, format_signif_jointly

__all__ = ["lme", "Profile"]


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
    _SKSPARSE_WARNED = True


def cho_factor(M):
    if _HAS_SKSPARSE:
        return _SksparseFactor(M)
    _warn_no_sksparse_once()
    return _SpluFactor(M)


@dataclass(slots=True)
class _FitInputs:
    """Pre-assembled inputs for :meth:`lme._fit_from_components`.

    Built by the public formula-based ``lme()`` constructor, or assembled
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
    other families raise :class:`NotImplementedError` until Phase 2-5 of
    ``lme-family-port.md`` land."""

    reml: bool
    """``True`` for REML, ``False`` for ML."""

    # Optional inputs ---------------------------------------------------
    weights: Optional[np.ndarray] = None
    """Prior weights (``None`` ≡ unit weights)."""

    mustart: Optional[np.ndarray] = None
    """Starting μ for GLMM PIRLS (Phase 2)."""

    etastart: Optional[np.ndarray] = None
    """Starting η for GLMM PIRLS (Phase 2)."""

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

    # Phase 8 plumbing --------------------------------------------------
    nAGQ: int = 1
    """Number of adaptive Gauss-Hermite quadrature points per group. Default
    1 ≡ Laplace approximation. ``0`` skips Stage 1 (LMM-style θ-only fit);
    ``>1`` is reserved for Phase 9 (AGQ) and currently raises."""

    tol_pwrss: float = 1e-7
    """PIRLS convergence tolerance — ``glmerControl(tolPwrss=)``."""

    maxit_pwrss: int = 100
    """PIRLS iteration cap — ``glmerControl(maxit=)`` (n.b. lme4 hard-codes
    this at 30 in pp_internal but exposes ``maxit`` via control)."""

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
    """Optimizer-specific control options. Currently only Nelder_Mead keys
    are recognised (``maxfun``, ``xtol_rel``, ``ftol_abs``, ``ftol_rel``).
    Mirrors ``glmerControl(optCtrl=)``."""

    # Diagnostic carries ------------------------------------------------
    # These follow the data through the fit so the resulting ``lme`` instance
    # can produce diagnostics, predict on new data, and round-trip formulas.
    expanded: Optional[ExpandedFormula] = None
    """The parsed/expanded formula, used by ``predict`` and ``profile``."""

    data: Optional[pl.DataFrame] = None
    """Post-NA-omit row set, in row-aligned order with X/Z/y/offset."""


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

    f_add = np.empty(n)
    f_sub = np.empty(n)
    for j in range(n):
        xj = x.copy(); xj[j] = xadd[j]; f_add[j] = float(fn(xj))
        xj = x.copy(); xj[j] = xsub[j]; f_sub[j] = float(fn(xj))

    g = (f_add - f_sub) / (udelta + ldelta)
    H = np.empty((n, n))
    for j in range(n):
        udj, ldj = udelta[j], ldelta[j]
        H[j, j] = f_add[j] / udj**2 - 2.0 * fx / (udj * ldj) + f_sub[j] / ldj**2
        for i in range(j):
            udi, ldi = udelta[i], ldelta[i]
            x_aa = x.copy(); x_aa[i] = xadd[i]; x_aa[j] = xadd[j]
            x_as = x.copy(); x_as[i] = xadd[i]; x_as[j] = xsub[j]
            x_sa = x.copy(); x_sa[i] = xsub[i]; x_sa[j] = xadd[j]
            x_ss = x.copy(); x_ss[i] = xsub[i]; x_ss[j] = xsub[j]
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
        idx = 0
        for i in range(c):
            for j in range(i, c):
                Lt[i, j] = theta[off + idx]
                idx += 1
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
    Laplace path (Phase 4); Phase 1's profiled-deviance code does not go
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
        """
        return float(np.sum(self.deviance_residuals()))

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
    :meth:`lme._fit_from_components` already operates. Mathematically
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

        Port of ``merPredD::sqrL(f)`` (predModule.cpp:140).
        """
        u_f = self.u(f)
        return float(np.einsum("i,i->", u_f, u_f))


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


def _glmm_devfun_factory(
    pred: _PredState,
    resp: _GlmResponse,
    *,
    nagq: int,
    tol_pwrss: float = 1e-7,
    maxit_pwrss: int = 100,
    verbose: int = 0,
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
        (θ, β) closure. ``nagq > 1`` (adaptive Gauss-Hermite) is implemented
        in Phase 9 — pass through this factory unchanged once the
        ``_pwrss_update`` AGQ path is added.
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
        return resp.laplace(
            pred.log_det_l_sq, pred.log_det_rx_sq, pred.sqr_l_u(1.0),
        )
    return devfun_theta_beta


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


class lme:
    """Linear mixed-effects model, fit by ML or REML profiled deviance.

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
        family: Optional[Family] = None,
        REML: bool = True,
        weights: Optional[np.ndarray] = None,
        start=None,
        nAGQ0initStep: bool = True,
    ):
        self.formula = formula
        self.REML = REML

        if family is None:
            family = Gaussian()

        d = prepare_design(formula, data)
        if not d.expanded.bars:
            raise ValueError(
                f"lme requires at least one random-effect bar; got formula={formula!r}"
            )
        # materialize_bars is called on d.data (response-NA-cleaned) so it
        # applies the same NA-omit policy as materialize() did for X — the
        # resulting Z stays row-aligned with X.
        re = materialize_bars(d.expanded, d.data)
        y = d.y.to_numpy().astype(float)

        # Sum any `offset(...)` atoms from the formula. β̂, û and the
        # variance components are all unchanged by the offset; only the
        # fitted/residual scale shifts. ``y`` here is the *original* response
        # (response scale); ``_fit_from_components`` builds ``y_solve = y -
        # offset`` internally for the Gaussian fit.
        n = len(y)
        off = np.zeros(n)
        for off_node in d.expanded.offsets:
            off = off + _eval_atom(off_node, d.data).values.flatten().astype(float)

        self._fit_from_components(_FitInputs(
            X_df=d.X,
            y=y,
            re_terms=re,
            offset=off,
            family=family,
            reml=REML,
            weights=weights,
            start=start,
            nagq0_init_step=nAGQ0initStep,
            expanded=d.expanded,
            data=d.data,
        ))

    def _fit_from_components(self, inputs: _FitInputs) -> None:
        """Fit the model given pre-assembled design pieces.

        Public ``lme()`` calls this after running ``prepare_design`` and
        ``materialize_bars``. External callers (``hea.gamm``) call it
        directly after composing smooth random-effect blocks via
        ``smooth2random`` — bypassing the formula parser entirely.

        Dispatches on ``inputs.family``: Gaussian-identity uses the
        profiled-deviance + CHOLMOD path implemented here. Other families
        raise :class:`NotImplementedError` until Phase 2-5 of
        ``lme-family-port.md`` add the Laplace approximation.
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
                "path does not yet honour non-unit weights; Phase 8 adds this."
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

        theta0 = re.theta.astype(float).copy()
        res = minimize(
            lambda th: self._ml_deviance(th) if not REML else self._reml_deviance(th),
            theta0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )
        theta_hat = res.x
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
        ``_pred``/``_resp`` for downstream phases. Full post-fit attributes
        (``fitted``, ``residuals``, ``AIC``, ``logLik``, ``sigma`` for
        unknown-scale families, σ-component summary tables, plotting hooks)
        land in Phase 6 of ``.claude/plans/lme-family-port.md``.
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
        bounds_beta = [(None, None)] * p
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
        # PIRLS inner-loop control. Match lme4's glmerControl defaults
        # (tolPwrss=1e-7, maxit=100). Phase 8 will plumb user overrides.
        tol_pwrss = 1e-7
        maxit_pwrss = 100

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

        if nagq0_init_step:
            # Stage 0 — joint PIRLS at θ₀, then optimize devfun over θ only.
            _pwrss_update(pred, resp, u_only=False, tol=tol_pwrss, maxit=maxit_pwrss)
            devfun_stage0 = _glmm_devfun_factory(
                pred, resp, nagq=0, tol_pwrss=tol_pwrss, maxit_pwrss=maxit_pwrss,
            )
            # Stage 0 step sizes: lme4's R wrapper default
            # (optimizer.R:5) ``xst = rep(0.02, n)``. ``xt = xst·5e-4``.
            xst0 = np.full(n_theta, 0.02)
            xt0 = xst0 * 5e-4
            nm0 = NelderMead(lb_theta, ub_theta, xst0, theta0, xtol_abs=xt0)
            status0 = nm0.minimize(devfun_stage0)
            theta_stage0 = nm0.xpos().copy()
            # Re-anchor pred/resp at the Stage 0 optimum. ``minimize`` may
            # have last evaluated at a probe — recall at the best point.
            devfun_stage0(theta_stage0)
            # lme4 uses pp.beta(1) (= the converged β from Stage 0 joint
            # PIRLS) as the Stage 1 starting β. modular.R:475 (`fixef0 <-
            # rho$pp$delb`). A user-supplied ``start["beta"]`` overrides.
            beta_start = beta_user_start if beta_user_start is not None else pred.beta(1.0).copy()
            self._optim_stage0 = {
                "par": theta_stage0, "fval": nm0.value(),
                "feval": nm0.nevals, "status": int(status0),
            }
        else:
            # No Stage 0 — go straight to Stage 1 with θ₀ and β=0 (or
            # user-supplied β).
            _pwrss_update(pred, resp, u_only=True, tol=tol_pwrss, maxit=maxit_pwrss)
            theta_stage0 = theta0
            beta_start = beta_user_start if beta_user_start is not None else np.zeros(p)
            self._optim_stage0 = None

        # Stage 1 — optimize over (θ, β) jointly. β is folded into the
        # offset; PIRLS uses u_only=True. The factory snapshots lp0 at the
        # current (post-Stage-0) state and base_offset = resp.offset.
        devfun_stage1 = _glmm_devfun_factory(
            pred, resp, nagq=1, tol_pwrss=tol_pwrss, maxit_pwrss=maxit_pwrss,
        )
        start_par = np.concatenate([theta_stage0, beta_start])
        lb_par = np.concatenate([lb_theta, lb_beta])
        ub_par = np.concatenate([ub_theta, ub_beta])
        # Stage 1 step sizes — lme4's ``adj=TRUE`` tweak at lmer.R:2533-2540:
        # θ block uses 0.1, β block uses ``min(βSD, 10)``, all scaled by 0.2.
        # ``βSD`` is sqrt(diag(unsc())); unsc = (RX·RX')⁻¹ with RX from the
        # current weighted decomposition (the Stage-0 converged state).
        beta_sd = _beta_sd_from_RX(pred.RX) if p > 0 else np.zeros(0)
        xst1 = 0.2 * np.concatenate([
            np.full(n_theta, 0.1),
            np.minimum(beta_sd, 10.0),
        ])
        xt1 = xst1 * 5e-4
        nm1 = NelderMead(lb_par, ub_par, xst1, start_par, xtol_abs=xt1)
        status1 = nm1.minimize(devfun_stage1)
        theta_hat = nm1.xpos()[:n_theta].copy()
        beta_hat = nm1.xpos()[n_theta:].copy()
        # Re-anchor at the Stage 1 optimum.
        devfun_stage1(nm1.xpos())
        self._optim = {
            "par": nm1.xpos().copy(), "fval": nm1.value(),
            "feval": nm1.nevals, "status": int(status1),
        }

        self.theta = theta_hat
        self._beta = beta_hat
        self._pred = pred
        self._resp = resp
        self.method = "glmer.ML"   # lme4's @resp$family != gaussian path

        # ----- post-fit attributes (Phase 6) ------------------------------

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

        # ----- SE(β̂) / vcov_beta via the Schur-complement RX --------------
        # ``pred.RX`` is the lower-Cholesky factor of the Schur-complement
        # Hessian of the conditional log-likelihood w.r.t. β at the
        # converged state. So Var(β̂) = σ² · (RX·RX')⁻¹ = σ² · RX⁻ᵀ·RX⁻¹.
        if p > 0:
            Rx_inv = solve_triangular(pred.RX, np.eye(p), lower=True)
            vcov_beta = self.sigma_squared * np.einsum("ij,ik->jk", Rx_inv, Rx_inv)
            se_beta = np.sqrt(np.diag(vcov_beta))
        else:
            vcov_beta = np.zeros((0, 0))
            se_beta = np.zeros(0)
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
        other = [k for k in range(len(self._theta_bounds)) if k != slot_i]
        theta_rest0 = np.array([theta_start[k] for k in other])

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
        if self.REML:
            return lme(self.formula, self.data, REML=False).profile(
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

    def confint(self, level: float = 0.95) -> pl.DataFrame:
        """R: ``confint.merMod`` — profile-likelihood CIs at ``level``.

        Mirrors lme4's default ``method="profile"``: runs :meth:`profile`
        and inverts each ζ-curve at ``±Φ⁻¹((1+level)/2)``. Returns one row
        per variance-component SD (``.sig01``, …, ``.sigma``) and one row
        per fixed effect.
        """
        return self.profile().confint(level=level)

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

    def _header(self) -> str:
        if self.REML:
            return "Linear mixed model fit by REML"
        return "Linear mixed model fit by maximum likelihood"

    def _fit_criterion_lines(self) -> list[str]:
        if self.REML:
            return [f"REML criterion at convergence: {self.REML_criterion:.4f}"]
        labels = ["AIC", "BIC", "logLik", "-2*log(L)", "df.resid"]
        vals = [
            f"{self.AIC:.4f}",
            f"{self.BIC:.4f}",
            f"{self.loglike:.4f}",
            f"{self.deviance:.4f}",
            f"{self.df_resid}",
        ]
        widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
        hdr = " ".join(l.rjust(w) for l, w in zip(labels, widths))
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
        scaled = self.residuals / self.sigma
        qs = np.quantile(scaled, [0.0, 0.25, 0.5, 0.75, 1.0])
        labels = ["Min", "1Q", "Median", "3Q", "Max"]
        vals = [f"{v:.4f}" for v in qs]
        widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
        hdr = " ".join(l.rjust(w) for l, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return ["Scaled residuals:", hdr, row]

    def summary(self, digits: int = 4) -> None:
        out = [self._header(), f"Formula: {self.formula}", ""]
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
        tbl = pl.DataFrame({
            "":           raw[""].to_list(),
            "Estimate":   est_s,
            "Std. Error": se_s,
            "t value":    format_signif(tval, digits=digits),
        })
        out.append(format_df(
            tbl,
            align={c: "right" for c in ("Estimate", "Std. Error", "t value")},
        ))
        corr_lines = self._fixef_corr_lines()
        if corr_lines:
            out.append("")
            out.extend(corr_lines)
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


class Profile:
    """Profile-likelihood output from :meth:`lme.profile`.

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
            from .plot._util import resolve_ax
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
