"""GLM family + link abstraction — mirrors R's ``family()`` augmented with
mgcv's ``fix.family.{link,var,ls}`` derivative fields.

Each :class:`Family` exposes the variance function ``V(μ)`` and its first
two derivatives, the deviance residuals ``dev_resids``, the saturated
log-likelihood ``ls(y, w, scale)`` (with first/second derivatives wrt
``log scale`` for unknown-scale REML), an ``initialize`` for starting
values, ``validmu``, and the AIC contribution.

Each :class:`Link` exposes ``link(μ)``, ``linkinv(η)``, ``mu_eta(η) =
dμ/dη``, plus second-through-fourth derivatives ``d²g/dμ²``, ``d³g/dμ³``,
``d⁴g/dμ⁴`` (with respect to μ, not η — matching mgcv's ``$d2link``
naming).

For a non-canonical link the PIRLS Newton step uses

    αᵢ = 1 + (yᵢ − μᵢ)·(V'/V + g''·dμ/dη)ᵢ
    wᵢ = αᵢ · (dμᵢ/dηᵢ)² / V(μᵢ)
    zᵢ = ηᵢ + (yᵢ − μᵢ) / ((dμᵢ/dηᵢ) · αᵢ)

so that the converged ``H = X'WX + Sλ`` is the **observed** penalized
Hessian, not the Fisher one. That makes ``∂β̂/∂ρ_k = -exp(ρ_k) H⁻¹ S_k β̂``
valid even for non-canonical links — the same identity that drives the
Gaussian REML derivatives in :mod:`hea.gam`.
"""

from __future__ import annotations

import contextlib
import itertools
import math

import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.special import (
    digamma, expit, gamma as _gamma_fn, gammaln, log_ndtr, logit, polygamma,
)

from .R import nmath as _nmath
from .R.nmath import (
    _dpois_raw,
    _dbinom_raw,
    _lgammafn_arr,
    dnorm5_vec,
    pnorm5_vec,
)
from ._dispatch import rs_fn as _rs_fn


def _polygamma(deriv, x):
    """``scipy.special.polygamma``-signature shim over R's ``dpsifn``
    (nmath/polygamma.c) — mgcv-faithful and rust-accelerated. scipy's
    Hurwitz-zeta ``polygamma`` is ~2.2x slower than R's ``psigamma`` for
    deriv>=1 (its digamma is faster, so deriv 0 stays on scipy)."""
    return _nmath.psigamma_vec(x, deriv)

# The GLM/GLMM aic hooks evaluate the saddlepoint log-density primitives
# (_dpois_raw / _dbinom_raw) on n-vectors every objective eval. Route them to the
# Rust kernels when present (bit-identical to the pure-Python ones — verified by
# the T1 parity gate — so the cumsum reduction stays bit-for-bit); the scalar
# Python path was a measured hot spot (≈16% of a cbpp glmer fit via _bd0/
# _stirlerr). See plans/rust-port-implementation.md.
_rs_dbinom_raw = _rs_fn("dbinom_raw")
_rs_dpois_raw = _rs_fn("dpois_raw")
# mgcv coxlpl (coxph.c:141) single-pass risk-set sweeps (deriv 0 → l/lb/lbb,
# deriv 1/2 → +d1H, deriv 3 → d2H); the numpy `_coxlpl` below is the bit-close
# oracle + HEA_NO_RS fallback. None when the extension is absent.
_rs_cox_l = _rs_fn("cox_l")
_rs_cox_lpl0 = _rs_fn("cox_lpl0")
_rs_cox_lpl_d1 = _rs_fn("cox_lpl_d1")
_rs_cox_d2h = _rs_fn("cox_d2h")
# mgcv tweedious (misc.c:170) per-row series sweep for the scalar-p Tweedie
# saturated-likelihood moments; the dense-matrix `_tweedie_log_a_vec` below is
# the numpy oracle + HEA_NO_RS fallback. None when the extension is absent.
_rs_tweedie_series = _rs_fn("tweedie_series")
# mgcv tweedious2 (misc.c:513) per-row sweep for the VECTOR-p Tweedie moments
# (α per row ⇒ no shared tables); the dense-matrix `_tweedie_log_a_vec_pv`
# below is the numpy oracle + HEA_NO_RS fallback. None when the extension is
# absent.
_rs_tweedie_series_pv = _rs_fn("tweedie_series_pv")
# gamlss.gH Hessian block crossprod (gamlss.r:653-660), the deterministic,
# row/col-consistent `crossprod(X_i, WX_j)` used only under `deterministic_xwx()`
# (gam.fit5's rank check) — where numpy `@`'s BLAS GEMM can give a rank-deficient
# duplicate column an asymmetric Hessian row that flips the QR rank-check drop
# platform-dependently. ~1.5x the `np.einsum` fallback; einsum is the HEA_NO_RS
# oracle (also row/col-consistent). None when the extension is absent.
_rs_gamlss_xwx = _rs_fn("gamlss_xwx")

# When set (by `deterministic_xwx()`), gamlss_gH assembles its Hessian blocks
# with the fixed-order `_xwx` reduction instead of numpy `@` — see gamlss_gH.
_GAMLSS_XWX_DETERMINISTIC = False


@contextlib.contextmanager
def deterministic_xwx():
    """Within this block, gamlss_gH's Hessian-block crossprod uses the
    row/col-consistent `_xwx` reduction (rust, else einsum) rather than the
    alignment-sensitive numpy `@`. gam.fit5 wraps its rank-check Hessian
    recompute in it so the dropped unidentifiable column is platform-stable."""
    global _GAMLSS_XWX_DETERMINISTIC
    prev = _GAMLSS_XWX_DETERMINISTIC
    _GAMLSS_XWX_DETERMINISTIC = True
    try:
        yield
    finally:
        _GAMLSS_XWX_DETERMINISTIC = prev


def _xwx(xi, wxj):
    """`crossprod(xi, wxj) = Σ_k xi[k,r]·wxj[k,c]` via a fixed per-entry
    reduction (identical output rows/cols for identical input columns): rust
    `gamlss_xwx`, else `np.einsum`. Both are within ~n·eps of `@` but
    construction-deterministic, unlike `@`."""
    if _rs_gamlss_xwx is not None:
        return np.asarray(_rs_gamlss_xwx(np.ascontiguousarray(xi),
                                         np.ascontiguousarray(wxj)))
    return np.einsum("kr,kc->rc", xi, wxj)


def _dbinom_raw_disp(x, n, p, q, give_log=True):
    """``_dbinom_raw`` via Rust when available, else the pure-Python kernel."""
    if _rs_dbinom_raw is None:
        return _dbinom_raw(x, n, p, q, give_log)
    shape = np.broadcast_shapes(np.shape(x), np.shape(n), np.shape(p), np.shape(q))
    a = [np.ascontiguousarray(np.broadcast_to(v, shape), dtype=float).ravel()
         for v in (x, n, p, q)]
    return np.asarray(_rs_dbinom_raw(a[0], a[1], a[2], a[3], give_log)).reshape(shape)


def _dpois_raw_disp(x, lam, give_log=True):
    """``_dpois_raw`` via Rust when available, else the pure-Python kernel."""
    if _rs_dpois_raw is None:
        return _dpois_raw(x, lam, give_log)
    shape = np.broadcast_shapes(np.shape(x), np.shape(lam))
    bx = np.ascontiguousarray(np.broadcast_to(x, shape), dtype=float).ravel()
    bl = np.ascontiguousarray(np.broadcast_to(lam, shape), dtype=float).ravel()
    return np.asarray(_rs_dpois_raw(bx, bl, give_log)).reshape(shape)


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


class Link:
    """Base class. Subclasses must implement ``link``, ``linkinv``,
    ``mu_eta``, ``d2link``, ``d3link``, ``d4link``."""
    name: str

    def link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def linkinv(self, eta: np.ndarray) -> np.ndarray: raise NotImplementedError
    def mu_eta(self, eta: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d2link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d3link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError
    def d4link(self, mu: np.ndarray) -> np.ndarray: raise NotImplementedError

    # Grouped link-derivative accessors for the REML weight-derivative chain
    # (_dw_deta / _d2w_deta2). The default just calls the individual methods;
    # links whose derivatives share an expensive transcantal (probit: η=Φ⁻¹(μ)
    # and φ(η)) override to compute it ONCE — bit-identical to the separate
    # calls, only the recomputed qnorm/dnorm are saved.
    def d23link(self, mu: np.ndarray):
        return self.d2link(mu), self.d3link(mu)

    def d234link(self, mu: np.ndarray):
        return self.d2link(mu), self.d3link(mu), self.d4link(mu)

    def valideta(self, eta: np.ndarray) -> bool: return True

    # mgcv ``link$g2g``, ``g3g``, ``g4g`` (R/efam.r): higher-order link
    # curvature ratios needed by ``Family.dDeta`` for extended families
    # under non-identity links. ``g2g(μ) = g″(μ)/g′(μ) · μ_η`` etc; we
    # use the equivalent form ``g″(μ)·μ_η = g2g`` direct from mgcv's
    # source. Identity link has all-zero curvature → IdentityLink
    # overrides to return zeros without computing.
    def g2g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g2g() is not implemented; needed for "
            "extended families under this non-identity link."
        )
    def g3g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g3g() is not implemented; needed for "
            "extended families under this non-identity link (level≥1)."
        )
    def g4g(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.g4g() is not implemented; needed for "
            "extended families under this non-identity link (level≥2)."
        )

    def __repr__(self) -> str:
        return self.name


class IdentityLink(Link):
    name = "identity"
    def g2g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def g3g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def g4g(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def link(self, mu): return np.asarray(mu, dtype=float)
    def linkinv(self, eta): return np.asarray(eta, dtype=float)
    def mu_eta(self, eta): return np.ones_like(np.asarray(eta, dtype=float))
    def d2link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d4link(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))


class LogLink(Link):
    name = "log"
    def link(self, mu):
        # R's log(0) is a silent -Inf (only negatives warn) — numpy's
        # divide RuntimeWarning has no R counterpart. Live at the η start
        # for censored count families, whose mustart may contain exact
        # zeros (cpois initialize's pmax(y, min(y>0)) quirk).
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # mgcv clamps to .Machine$double.eps to avoid 0 — replicate so divisions
        # by μ in PIRLS / V'(μ) etc. don't blow up at extreme negative η.
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    def mu_eta(self, eta):
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    # R-level ^ is R_pow (sequential multiplies for ^2/^3/^4 at |μ|≤11,
    # not numpy **) — last-ulp parity for the μ-derivative tables.
    def d2link(self, mu): return -1.0 / _rpow_int(
        np.asarray(mu, dtype=float), 2)
    def d3link(self, mu): return 2.0 / _rpow_int(
        np.asarray(mu, dtype=float), 3)
    def d4link(self, mu): return -6.0 / _rpow_int(
        np.asarray(mu, dtype=float), 4)
    # log link: g'(μ)=1/μ, g''(μ)=-1/μ², g'''(μ)=2/μ³, g''''(μ)=-6/μ⁴ →
    # g2g=g''/g'²=-1, g3g=g'''/g'³=2, g4g=g''''/g'⁴=-6.
    # mgcv gam.fit3.r:2229-2231.
    def g2g(self, mu): return -np.ones_like(np.asarray(mu, dtype=float))
    def g3g(self, mu): return 2.0 * np.ones_like(np.asarray(mu, dtype=float))
    def g4g(self, mu): return -6.0 * np.ones_like(np.asarray(mu, dtype=float))


class InverseLink(Link):
    name = "inverse"
    def link(self, mu): return 1.0 / np.asarray(mu, dtype=float)
    def linkinv(self, eta): return 1.0 / np.asarray(eta, dtype=float)
    def mu_eta(self, eta): return -1.0 / _rpow_int(
        np.asarray(eta, dtype=float), 2)
    # R-level ^ = R_pow: ^2/^3/^4 sequential multiplies at |μ|≤11,
    # ^5 always libm pow (arithmetic.c:217-221) — not numpy **.
    def d2link(self, mu): return 2.0 / _rpow_int(
        np.asarray(mu, dtype=float), 3)
    def d3link(self, mu): return -6.0 / _rpow_int(
        np.asarray(mu, dtype=float), 4)
    def d4link(self, mu): return 24.0 / _rpow(
        np.asarray(mu, dtype=float), 5.0)
    # inverse link: g'=-1/μ², g''=2/μ³, g'''=-6/μ⁴, g''''=24/μ⁵ →
    # g2g = g''/g'² = (2/μ³)·μ⁴ = 2μ;  g3g = g'''/g'³ = (-6/μ⁴)·(-μ⁶) = 6μ²;
    # g4g = g''''/g'⁴ = (24/μ⁵)·μ⁸ = 24μ³.
    # mgcv gam.fit3.r:2234-2236.
    def g2g(self, mu): return 2.0 * np.asarray(mu, dtype=float)
    def g3g(self, mu): return 6.0 * _rpow_int(np.asarray(mu, dtype=float), 2)
    def g4g(self, mu): return 24.0 * _rpow_int(np.asarray(mu, dtype=float), 3)
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(eta != 0))


class SqrtLink(Link):
    """``g(μ) = √μ`` — alternate poisson link."""
    name = "sqrt"
    def link(self, mu): return np.sqrt(np.asarray(mu, dtype=float))
    def linkinv(self, eta): return np.asarray(eta, dtype=float) ** 2
    def mu_eta(self, eta): return 2.0 * np.asarray(eta, dtype=float)
    def d2link(self, mu): return -0.25 * np.asarray(mu, dtype=float) ** -1.5
    def d3link(self, mu): return 0.375 * np.asarray(mu, dtype=float) ** -2.5
    def d4link(self, mu): return -0.9375 * np.asarray(mu, dtype=float) ** -3.5
    # fix.family.link's extended-family ratios (gam.fit3.r:2243-2247):
    # g' = ½μ^-½ ⇒ g2g = g″/g′² = -μ^-½, g3g = g‴/g′³ = 3/μ,
    # g4g = g⁗/g′⁴ = -15·μ^-1.5. μ = 0 rows (censored count families'
    # zero counts) give R's silent ±Inf — numpy would warn.
    def g2g(self, mu):
        with np.errstate(divide="ignore"):
            return -np.asarray(mu, dtype=float) ** -0.5
    def g3g(self, mu):
        with np.errstate(divide="ignore"):
            return 3.0 / np.asarray(mu, dtype=float)
    def g4g(self, mu):
        with np.errstate(divide="ignore"):
            return -15.0 * np.asarray(mu, dtype=float) ** -1.5
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class PowerLink(Link):
    """R's ``power(λ)`` link for 0 < λ ≠ 1: ``g(μ) = μ^λ``.

    Use the :func:`power` factory, which mirrors R exactly — ``λ ≤ 0``
    returns the log link and ``λ = 1`` the identity, so only genuine
    powers reach this class. ``linkinv``/``mu_eta`` carry R's
    ``.Machine$double.eps`` floor; the d2link..d4link table is
    fix.family.link's power branch (gam.fit3.r:2329-2335 quasi
    vector-link form ≡ the "mu^" name branch :2415-2421).
    """
    def __init__(self, lam: float):
        self.lam = float(lam)
        # R: link name is paste0("mu^", round(lambda, 3)).
        self.name = f"mu^{round(self.lam, 3):g}"
    def link(self, mu):
        return np.asarray(mu, dtype=float) ** self.lam
    def linkinv(self, eta):
        eps = np.finfo(float).eps
        return np.maximum(
            np.asarray(eta, dtype=float) ** (1.0 / self.lam), eps,
        )
    def mu_eta(self, eta):
        eps = np.finfo(float).eps
        return np.maximum(
            np.asarray(eta, dtype=float) ** (1.0 / self.lam - 1.0)
            / self.lam, eps,
        )
    def d2link(self, mu):
        lam = self.lam
        return lam * (lam - 1.0) * np.asarray(mu, dtype=float) ** (lam - 2.0)
    def d3link(self, mu):
        lam = self.lam
        return (lam * (lam - 1.0) * (lam - 2.0)
                * np.asarray(mu, dtype=float) ** (lam - 3.0))
    def d4link(self, mu):
        lam = self.lam
        return (lam * (lam - 1.0) * (lam - 2.0) * (lam - 3.0)
                * np.asarray(mu, dtype=float) ** (lam - 4.0))
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


def power(lam: float = 1.0) -> Link:
    """R ``stats::power(lambda)``: the ``μ^λ`` link-glm object.

    Exact R semantics: ``λ ≤ 0`` → the log link, ``λ = 1`` → identity,
    otherwise :class:`PowerLink`. Pass the OBJECT to a family —
    ``quasi(link=power(1/3))`` — exactly as in R (R's ``make.link``
    does not accept a "power(...)" string and neither does hea).
    """
    lam = float(lam)
    if not np.isfinite(lam):
        raise ValueError("invalid argument 'lambda'")
    if lam <= 0.0:
        return LogLink()
    if lam == 1.0:
        return IdentityLink()
    return PowerLink(lam)


class LogitLink(Link):
    """``g(μ) = log(μ/(1-μ))`` — canonical binomial link."""
    name = "logit"
    def link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return np.log(mu / (1.0 - mu))
    def linkinv(self, eta):
        # stats C ``logit_linkinv`` (family.c:73-90) verbatim:
        # tmp = η<-30 ? eps : (η>30 ? 1/eps : e^η), return tmp/(1+tmp) —
        # the thresholds are what keep PIRLS off μ = 0/1 where
        # V(μ) = μ(1-μ) collapses. exp(η) for η ∈ (30, 710) is finite
        # and unused; beyond that numpy would warn where C silently
        # overflows, so compute it only where selected.
        eta = np.asarray(eta, dtype=float)
        eps = np.finfo(float).eps
        tmp = np.exp(np.where(np.abs(eta) > 30.0, 0.0, eta))
        tmp = np.where(eta < -30.0, eps,
                       np.where(eta > 30.0, 1.0 / eps, tmp))
        return tmp / (1.0 + tmp)
    def mu_eta(self, eta):
        # stats C ``logit_mu_eta`` (family.c:92-108) verbatim:
        # |η|>30 → eps (a hard drop below the true value — R's own
        # guard); else e^η/((1+e^η)·(1+e^η)).
        eta = np.asarray(eta, dtype=float)
        eps = np.finfo(float).eps
        expE = np.exp(np.where(np.abs(eta) > 30.0, 0.0, eta))
        opexp = 1.0 + expE
        return np.where(np.abs(eta) > 30.0, eps, expE / (opexp * opexp))
    # μ-derivative table (fix.family.link's logit rows) and the
    # extended-family ratios (gam.fit3.r:2237-2241): R-level ``^`` is
    # R_pow — sequential multiplies for ^2/^3/^4 at |x| ≤ 11 — not
    # numpy ``**`` (last-ulp drift for ^3/^4).
    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 1.0 / _rpow_int(1.0 - mu, 2) - 1.0 / _rpow_int(mu, 2)
    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 / _rpow_int(1.0 - mu, 3) + 2.0 / _rpow_int(mu, 3)
    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 6.0 / _rpow_int(1.0 - mu, 4) - 6.0 / _rpow_int(mu, 4)
    def g2g(self, mu):
        mu = np.asarray(mu, dtype=float)
        return _rpow_int(mu, 2) - _rpow_int(1.0 - mu, 2)
    def g3g(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 * _rpow_int(mu, 3) + 2.0 * _rpow_int(1.0 - mu, 3)
    def g4g(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 6.0 * _rpow_int(mu, 4) - 6.0 * _rpow_int(1.0 - mu, 4)


def _dnorm(x):
    # R's dnorm (nmath/dnorm.c) — bit-exact, incl. the |x|>=5 split path that
    # the naive exp(-x²/2)/√(2π) misses when probit η drifts into the tail.
    return _nmath.dnorm5_vec(np.asarray(x, dtype=float))


class ProbitLink(Link):
    """``g(μ) = Φ⁻¹(μ)`` — probit binomial link."""
    name = "probit"
    def link(self, mu): return _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # R: clamp η to ±qnorm(eps); pnorm of clamped η.
        eta = np.asarray(eta, dtype=float)
        thresh = -_nmath.qnorm5(np.finfo(float).eps)
        return _nmath.pnorm5_vec(np.clip(eta, -thresh, thresh))
    def mu_eta(self, eta):
        # dnorm(η), lower-clamped.
        eps = np.finfo(float).eps
        return np.maximum(_dnorm(np.asarray(eta, dtype=float)), eps)
    def d2link(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return eta / d ** 2
    def d3link(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (1.0 + 2.0 * eta * eta) / d ** 3
    def d4link(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (7.0 * eta + 6.0 * eta ** 3) / d ** 4

    def d23link(self, mu):
        # η=Φ⁻¹(μ) and φ(η) computed once, shared by d2/d3 — bit-identical to
        # d2link(mu), d3link(mu) (same η, same d, same expressions).
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return eta / d ** 2, (1.0 + 2.0 * eta * eta) / d ** 3

    def d234link(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (eta / d ** 2, (1.0 + 2.0 * eta * eta) / d ** 3,
                (7.0 * eta + 6.0 * eta ** 3) / d ** 4)
    # extended-family ratios (gam.fit3.r:2249-2266): with η=Φ⁻¹(μ) and
    # g'=1/φ(η), the g″/g'ᵏ ratios collapse to polynomials in η.
    def g2g(self, mu):
        return _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
    def g3g(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        return 1.0 + 2.0 * eta * eta
    def g4g(self, mu):
        eta = _nmath.qnorm5_vec(np.asarray(mu, dtype=float))
        return 7.0 * eta + 6.0 * eta ** 3


class CauchitLink(Link):
    """``g(μ) = tan(π(μ-½))`` — Cauchy-quantile binomial link.

    Heavier-tailed than probit/logit; fits well when a fraction of obs are
    far from the (logit) decision boundary.
    """
    name = "cauchit"
    def link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return np.tan(np.pi * (mu - 0.5))
    def linkinv(self, eta):
        # R: clamp η to ±qcauchy(eps); pcauchy(η) = ½ + atan(η)/π.
        eps = np.finfo(float).eps
        thresh = -np.tan(np.pi * (eps - 0.5))
        eta_c = np.clip(np.asarray(eta, dtype=float), -thresh, thresh)
        return 0.5 + np.arctan(eta_c) / np.pi
    def mu_eta(self, eta):
        eps = np.finfo(float).eps
        eta = np.asarray(eta, dtype=float)
        return np.maximum(1.0 / (np.pi * (1.0 + eta * eta)), eps)
    def d2link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        return 2.0 * np.pi ** 2 * eta * (1.0 + eta * eta)
    def d3link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return 2.0 * np.pi ** 3 * (1.0 + 3.0 * eta2) * (1.0 + eta2)
    def d4link(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return 2.0 * np.pi ** 4 * (8.0 * eta + 12.0 * eta2 * eta) * (1.0 + eta2)
    # extended-family ratios (gam.fit3.r:2272-2291): η=qcauchy(μ),
    # g'=1/f(η) with f the Cauchy density — the g″/g'ᵏ ratios in η.
    def g2g(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        return eta / (1.0 + eta * eta)
    def g3g(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return (1.0 + 3.0 * eta2) / (1.0 + eta2) ** 2
    def g4g(self, mu):
        eta = np.tan(np.pi * (np.asarray(mu, dtype=float) - 0.5))
        eta2 = eta * eta
        return ((8.0 + 12.0 * eta2) / (1.0 + eta2) ** 2) * (eta / (1.0 + eta2))


class CloglogLink(Link):
    """``g(μ) = log(-log(1-μ))`` — complementary log-log binomial link."""
    name = "cloglog"
    def link(self, mu):
        return np.log(-np.log1p(-np.asarray(mu, dtype=float)))
    def linkinv(self, eta):
        # 1 - exp(-exp(η)), clamped to [eps, 1-eps] (R: avoid mu=0,1 boundary).
        eps = np.finfo(float).eps
        eta = np.asarray(eta, dtype=float)
        return np.clip(-np.expm1(-np.exp(eta)), eps, 1.0 - eps)
    def mu_eta(self, eta):
        # exp(η - exp(η)); R clamps η at 700 (to keep exp(η) finite) and
        # lower-clamps the result at eps.
        eps = np.finfo(float).eps
        eta = np.minimum(np.asarray(eta, dtype=float), 700.0)
        return np.maximum(np.exp(eta) * np.exp(-np.exp(eta)), eps)
    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return -1.0 / ((1.0 - mu) ** 2 * l1m) * (1.0 + 1.0 / l1m)
    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return (-2.0 - 3.0 * l1m - 2.0 * l1m ** 2) / (1.0 - mu) ** 3 / l1m ** 3
    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        l1m = np.log1p(-mu)
        return (-12.0 - 11.0 * l1m - 6.0 * l1m ** 2 - 6.0 / l1m) / (1.0 - mu) ** 4 / l1m ** 3
    # extended-family ratios (gam.fit3.r:2293-2303), l1m = log1p(−μ):
    # g'=−1/(l1m·(1−μ)) ⇒ g2g=−l1m−1, g3g=l1m(2·l1m+3)+2,
    # g4g=−l1m(l1m(6·l1m+11)+12)−6.
    def g2g(self, mu):
        l1m = np.log1p(-np.asarray(mu, dtype=float))
        return -l1m - 1.0
    def g3g(self, mu):
        l1m = np.log1p(-np.asarray(mu, dtype=float))
        return l1m * (2.0 * l1m + 3.0) + 2.0
    def g4g(self, mu):
        l1m = np.log1p(-np.asarray(mu, dtype=float))
        return -l1m * (l1m * (6.0 * l1m + 11.0) + 12.0) - 6.0


class InverseSquareLink(Link):
    """``g(μ) = 1/μ²`` — canonical inverse-Gaussian link."""
    name = "1/mu^2"
    def link(self, mu): return 1.0 / np.asarray(mu, dtype=float) ** 2
    def linkinv(self, eta):
        # PIRLS step-halving may transiently call us with eta<0 entries;
        # the caller checks valideta() and rejects them. Silence the
        # sqrt-of-negative warning so strict warning modes (pytest's
        # `np.errstate(invalid="raise")`) don't trip over a recoverable
        # halving step.
        with np.errstate(invalid="ignore"):
            return 1.0 / np.sqrt(np.asarray(eta, dtype=float))
    def mu_eta(self, eta):
        with np.errstate(invalid="ignore"):
            return -0.5 * np.asarray(eta, dtype=float) ** -1.5
    def d2link(self, mu): return 6.0 * np.asarray(mu, dtype=float) ** -4
    def d3link(self, mu): return -24.0 * np.asarray(mu, dtype=float) ** -5
    def d4link(self, mu): return 120.0 * np.asarray(mu, dtype=float) ** -6
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


class SoftplusLink(Link):
    """``μ = softplus(η) = log(1 + e^η)`` — the smooth-rectifier link.

    Not an mgcv/`make.link` built-in (its variance-function home is the
    transcendental ``V(μ)=1−e^{−μ}``, outside the Morris NEF-QVF families), but
    standard in the neural-GLM literature (Paninski 2004; Pillow et al.) as a
    numerically gentle, concavity-preserving alternative to the canonical log
    link for `Poisson()` point-process / RF models: log-link-like (μ≈e^η) at
    low rates, identity-like (μ≈η) at high rates, so no exponential blow-up.

    ``g(μ) = log(e^μ − 1)`` (μ>0); ``g′(μ) = 1/(1−e^{−μ})``. Writing
    ``u = e^{−μ}``, ``s = 1−u``:

    * ``g″  = −u/s²``            ``g‴  =  u(1+u)/s³``      ``g⁗ = −u(1+4u+u²)/s⁴``
    * ``g2g = g″/g′²  = −u``     ``g3g = g‴/g′³ = u(1+u)`` ``g4g = g⁗/g′⁴ = −u(1+4u+u²)``

    PIRLS handles this as an ordinary non-canonical link (full-Newton inner
    steps); pairing with `Poisson()` forfeits only the canonical-log
    convenience, not global concavity. Use ``s = -expm1(-μ)`` for ``1−u`` so the
    small-μ (log-like) regime keeps full precision.
    """
    name = "softplus"

    def link(self, mu):
        # g(μ) = log(e^μ − 1); expm1 keeps small-μ accurate.
        mu = np.maximum(np.asarray(mu, dtype=float), np.finfo(float).eps)
        return np.log(np.expm1(mu))

    def linkinv(self, eta):
        # μ = log1p(e^η) = logaddexp(0, η); eps-floored like LogLink so μ>0
        # feeds V(μ) / divisions safely at extreme negative η.
        return np.maximum(np.logaddexp(0.0, np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)

    def mu_eta(self, eta):
        # dμ/dη = σ(η); expit is the stable logistic. Lower-clamp like mgcv.
        return np.maximum(expit(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)

    def _u_s(self, mu):
        mu = np.maximum(np.asarray(mu, dtype=float), np.finfo(float).eps)
        u = np.exp(-mu)
        s = -np.expm1(-mu)        # 1 − e^{−μ}, accurate as μ→0⁺
        return u, s

    def d2link(self, mu):
        u, s = self._u_s(mu)
        return -u / s ** 2

    def d3link(self, mu):
        u, s = self._u_s(mu)
        return u * (1.0 + u) / s ** 3

    def d4link(self, mu):
        u, s = self._u_s(mu)
        return -u * (1.0 + 4.0 * u + u * u) / s ** 4

    def g2g(self, mu):
        u, _ = self._u_s(mu)
        return -u

    def g3g(self, mu):
        u, _ = self._u_s(mu)
        return u * (1.0 + u)

    def g4g(self, mu):
        u, _ = self._u_s(mu)
        return -u * (1.0 + 4.0 * u + u * u)


_LINKS = {
    "identity": IdentityLink,
    "log": LogLink,
    "inverse": InverseLink,
    "sqrt": SqrtLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "cauchit": CauchitLink,
    "cloglog": CloglogLink,
    "1/mu^2": InverseSquareLink,
    "softplus": SoftplusLink,
}


def _resolve_link(link, default: str) -> Link:
    if link is None:
        return _LINKS[default]()
    if isinstance(link, Link):
        return link
    if isinstance(link, str):
        if link not in _LINKS:
            raise ValueError(f"unknown link {link!r}; supported: {list(_LINKS)}")
        return _LINKS[link]()
    # Allow `link=log` (the function reference) the way R's `Gamma(link=log)` does.
    name = getattr(link, "__name__", None)
    if name in _LINKS:
        return _LINKS[name]()
    raise ValueError(f"unknown link {link!r}")


def _brent_fmin(f, ax: float, bx: float, tol: float) -> tuple[float, float]:
    """R's ``Brent_fmin`` (src/library/stats/src/optimize.c) — the exact
    golden-section + successive-parabolic-interpolation loop behind
    ``stats::optimize``, ported operation-for-operation so mgcv code
    built on ``optimize`` (``find.null.dev``) reproduces R's stop points.
    Returns ``(x_min, f(x_min))``.
    """
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(np.finfo(float).eps)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:                       # fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if (abs(p) >= abs(q * 0.5 * r)
                or p <= q * (a - x) or p >= q * (b - x)):
            # golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0.0 else -tol1)
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


def find_null_dev(family: "Family", y, eta, offset, weights) -> float:
    """mgcv ``find.null.dev`` (efam.r:98-117): the null deviance of an
    extended family — deviance of the best single-constant model on the
    link scale, found by 1-D ``optimize`` over the constant with mgcv's
    interval-doubling protocol (double the half-width until the minimum
    is interior). Replaces the standard weighted-mean null deviance in
    the extended postprocs (nb efam.r:283, tw efam.r:3239,
    scat efam.r:3742) — for non-canonical-ish links the optimal constant
    is NOT the weighted mean, so the two differ at 1e-3 level.

    ``eta`` is the converged linear predictor INCLUDING the offset
    (mgcv's ``linear.predictors``); the initial constant comes from the
    weighted mean of ``linkinv(eta − offset)``, while the candidate
    models are ``μ = linkinv(γ + offset)``.
    """
    y = np.asarray(y, dtype=float)
    eta = np.asarray(eta, dtype=float)
    offset = np.zeros_like(eta) if offset is None else np.asarray(
        offset, dtype=float)
    weights = np.asarray(weights, dtype=float)
    link = family.link

    def fnull(gamma: float) -> float:
        # 3-arg dev.resids like mgcv's fnull — extended families read
        # their current θ when ``theta=None``.
        mu = link.linkinv(gamma + offset)
        return float(np.sum(family.dev_resids(y, mu, weights)))

    mu0 = link.linkinv(eta - offset)
    mum = float(np.mean(mu0 * weights) / np.mean(weights))
    eta0 = float(link.link(mum))
    deta = abs(eta0) * 0.1 + 1.0       # search interval half width
    tol = float(np.finfo(float).eps) ** 0.25   # optimize's default tol
    while True:
        lo, hi = eta0 - deta, eta0 + deta
        x_min, f_min = _brent_fmin(fnull, lo, hi, tol)
        if lo < x_min < hi:
            return f_min
        deta *= 2.0


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


class Family:
    """Base class for GLM families."""
    name: str
    canonical_link_name: str
    scale_known: bool
    # Number of "extra" family parameters that the GAM outer Newton should
    # estimate jointly with (ρ, log φ). Default 0 (Gaussian, Gamma, Poisson,
    # Binomial, IG, Quasi); ``tw`` overrides to 1 (its θ_tw → p
    # reparametrisation). The GAM hooks read ``n_theta`` to size the outer
    # vector and call ``set_theta(values)`` before each criterion eval; they
    # call ``dscore_extra(...)`` to obtain the score-side ∂(2·V_R)/∂θ_extra
    # contributions for the gradient.
    n_theta: int = 0
    # Mirrors mgcv ``inherits(family, "extended.family")``. Standard
    # exponential families (Gaussian, Poisson, ...) leave it ``False``;
    # extended families (Scat, ziP, ocat, gevlss, ...) flip to ``True``
    # so the bam(discrete=TRUE) PIRLS path uses the ``Dd → dDeta`` Newton
    # weights (``w = Deta2/2``, ``z = (η-off) - Deta/Deta2``) instead of
    # the standard Fisher weights ``w = w_prior · μ_η²/V(μ)``.
    is_extended: bool = False
    # Whether the bam outer loop should call ``_estimate_theta`` between
    # PIRLS iters. Set ``True`` only on extended families with free θ
    # (Scat with both θ free, nb with k free, etc). Standard families and
    # extended families with all θ user-locked leave it ``False``.
    estimate_theta_callback: bool = False

    # mgcv's canonical link for PIRLS's full-Newton/Fisher switch
    # (fix.family.link's table, gam.fit3.r:2316-2323). ``None`` means
    # "same as canonical_link_name" (the table's gaussian/poisson/
    # binomial/Gamma/IG rows). Families outside that table set "none"
    # explicitly — quasi (table fallback :2322), Tweedie
    # (gam.fit3.r:3105), tw (efam.r:3262), scat/nb — so the inner loop
    # never takes the Fisher shortcut whatever the link. Distinct from
    # ``canonical_link_name``, which also resolves the *default* link.
    _newton_canonical: str | None = None

    def __init__(self, link=None):
        self.link = _resolve_link(link, self.canonical_link_name)

    @property
    def is_canonical(self) -> bool:
        canon = self._newton_canonical
        if canon is None:
            canon = self.canonical_link_name
        return self.link.name == canon

    def set_theta(self, values) -> None:
        """Mutate the family's extra parameters from a length-``n_theta``
        array. Default is a no-op (consistent with ``n_theta = 0``);
        :class:`tw` overrides to update ``self.theta`` and ``self.p``.
        """
        if self.n_theta != 0:
            raise NotImplementedError(
                f"{type(self).__name__} declares n_theta={self.n_theta} "
                f"but did not override set_theta()."
            )

    def get_theta(self) -> np.ndarray:
        """Return the current extra parameters as a length-``n_theta`` array.
        Default empty; :class:`tw` returns ``[θ_tw]``."""
        return np.zeros(0)

    def variance(self, mu): raise NotImplementedError
    def dvar(self, mu): raise NotImplementedError
    def d2var(self, mu): raise NotImplementedError
    def d3var(self, mu): raise NotImplementedError

    def dev_resids(self, y, mu, wt, theta=None) -> np.ndarray:
        """Per-observation deviance contributions; sum is the deviance D.

        ``theta`` is accepted but ignored for standard exponential
        families. Extended families (``is_extended=True``) read it to
        compute deviance at a probe θ during inner-Newton θ estimation.
        """
        raise NotImplementedError

    # ----- extended-family hooks (no-ops for standard families) ---------
    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        """Mirrors mgcv ``family$Dd``. Returns a dict of derivatives of
        ``-logL`` wrt μ and θ at fixed (y, μ, θ, w):

        * level 0: ``Dmu``, ``Dmu2``, ``EDmu2`` (all length-n).
        * level ≥ 1: + ``Dth``, ``Dmuth``, ``Dmu2th``, ``EDmu2th``,
          ``Dmu3``, ``EDmu3``. ``D*th`` shape ``(n, n_theta)``.
        * level ≥ 2: + ``Dmu4``, ``Dth2``, ``Dmuth2``, ``Dmu2th2``,
          ``Dmu3th``. ``D*th2`` packed column-major upper-triangle of
          shape ``(n, n_theta·(n_theta+1)/2)``.

        Standard families don't implement ``Dd`` — bam's PIRLS path uses
        the Fisher branch for them. Only extended families override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.Dd() is not implemented; this family "
            "uses the standard-Fisher PIRLS path. Set is_extended=True "
            "and implement Dd() to use the extended-family Newton path."
        )

    def dDeta(self, y, mu, wt, theta, level: int = 0, dd: dict | None = None) -> dict:
        """Convert ``Dd`` (μ-space derivatives) to η-space via the link
        chain rule. Mirrors mgcv ``dDeta`` (R/efam.r). For identity link
        it copies ``Dmu → Deta``, ``Dmu2 → Deta2``, ...; for non-identity
        it applies ``Deta = Dmu · μ_η`` etc with the ``g2g``/``g3g``/
        ``g4g`` link curvature terms.

        Returns a dict with at minimum ``Deta``, ``Deta2``, ``EDeta2``
        (level 0). ``Deta.Deta2 = Dmu/(Dmu2·μ_η - Dmu·g2g)`` is the
        Newton-step working-response numerator that bam's PIRLS reads.

        ``dd`` lets a caller pass a precomputed ``Dd`` table (≥ ``level``) to
        share the per-obs deviance derivatives with raw-``Dd`` consumers (the
        gam.fit4 ``_Dd``/``_dDeta`` caches) instead of recomputing them.
        """
        r = dd if dd is not None else self.Dd(y, mu, theta, wt, level=level)
        link = self.link
        if link.name == "identity":
            # Unguarded divisions as in mgcv (gam.fit4.r:17-19): vanishing
            # Dmu2 rows (extreme-z clog) give R's silent Inf/NaN.
            with np.errstate(divide="ignore", invalid="ignore"):
                d = {
                    "Deta": r["Dmu"],
                    "Deta2": r["Dmu2"],
                    "EDeta2": r["EDmu2"],
                    "Deta.Deta2": r["Dmu"] / r["Dmu2"],
                    "Deta.EDeta2": r["Dmu"] / r["EDmu2"],
                }
            if level > 0:
                # θ-derivative keys use R's NULL list-read semantics: a
                # family with no parameters (cpois) emits no D*th at all —
                # propagate the absence as None. Nothing downstream reads
                # them when the θ block is empty (mgcv's nth==0 guard,
                # gam.fit4.r:66-75; hea gates on family.n_theta > 0).
                d.update({
                    "Dth": r.get("Dth"),
                    "Detath": r.get("Dmuth"),
                    "Deta3": r["Dmu3"],
                    "Deta2th": r.get("Dmu2th"),
                    "EDeta3": r.get("EDmu3"),
                })
                # EDmu2th is optional (ziP omits it; mgcv's R-NULL list
                # access silently skips it, gam.fit4.r:23) — mirror that.
                if r.get("EDmu2th") is not None:
                    d["EDeta2th"] = r["EDmu2th"]
            if level > 1:
                d.update({
                    "Deta4": r["Dmu4"],
                    "Dth2": r.get("Dth2"),
                    "Detath2": r.get("Dmuth2"),
                    "Deta2th2": r.get("Dmu2th2"),
                    "Deta3th": r.get("Dmu3th"),
                })
            return d
        # Non-identity link path. mgcv ``dDeta`` expects ``link.g2g(μ)``,
        # ``g3g``, ``g4g`` to be implemented on the link object. R computes
        # the whole table silently — μ = 0 rows (censored count families)
        # make ±Inf·0 → NaN products that the caller-side ``good`` mask
        # then drops (gam.fit4.r:62) — so keep numpy quiet too.
        ig1 = link.mu_eta(link.link(np.asarray(mu, dtype=float)))
        ig12 = ig1 * ig1

        def _cb(a, v):
            # R recycles a length-n vector down the columns of an (n, k)
            # θ-derivative matrix (gfam's D*th blocks; single-θ families
            # keep 1-D arrays where * is already elementwise).
            a = np.asarray(a, dtype=float)
            return a * (v[:, None] if a.ndim == 2 else v)

        g2g = link.g2g(mu)
        with np.errstate(invalid="ignore"):
            d = {
                "Deta": r["Dmu"] * ig1,
                "Deta2": r["Dmu2"] * ig12 - r["Dmu"] * g2g * ig1,
                "EDeta2": r["EDmu2"] * ig12,
            }
        # Unguarded divisions, mirroring mgcv gam.fit4.r:39-40: where the
        # denominator vanishes R yields Inf silently, so ignore the FP flag.
        with np.errstate(divide="ignore", invalid="ignore"):
            d["Deta.Deta2"] = r["Dmu"] / (r["Dmu2"] * ig1 - r["Dmu"] * g2g)
            d["Deta.EDeta2"] = r["Dmu"] / (r["EDmu2"] * ig1)
        if level > 0:
            ig13 = ig12 * ig1
            # θ keys: R NULL semantics for parameter-free families
            # (cpois) — see the identity-branch note.
            has_th = r.get("Dth") is not None
            d["Dth"] = r.get("Dth")
            g3g = link.g3g(mu)
            with np.errstate(invalid="ignore"):
                d["Detath"] = _cb(r["Dmuth"], ig1) if has_th else None
                d["Deta3"] = (r["Dmu3"] * ig13
                              - 3.0 * r["Dmu2"] * g2g * ig12
                              + r["Dmu"] * (3.0 * g2g * g2g - g3g) * ig1)
                EDmu3 = r.get("EDmu3")
                if EDmu3 is not None:
                    d["EDeta3"] = (EDmu3 * ig13
                                   - 3.0 * r["EDmu2"] * g2g * ig12)
                d["Deta2th"] = (_cb(r["Dmu2th"], ig12)
                                - _cb(_cb(r["Dmuth"], g2g), ig1)
                                if has_th else None)
                EDmu2th = r.get("EDmu2th")
                if EDmu2th is not None:
                    d["EDeta2th"] = _cb(EDmu2th, ig12)
        if level > 1:
            g4g = link.g4g(mu)
            ig14 = ig12 * ig12
            has_th2 = r.get("Dth2") is not None
            d["Dth2"] = r.get("Dth2")
            with np.errstate(invalid="ignore"):
                d["Deta4"] = (ig14 * r["Dmu4"]
                              - 6.0 * r["Dmu3"] * ig13 * g2g
                              + r["Dmu2"] * (15.0 * g2g * g2g - 4.0 * g3g)
                              * ig12
                              - r["Dmu"]
                              * (15.0 * g2g ** 3 - 10.0 * g2g * g3g + g4g)
                              * ig1)
                d["Detath2"] = _cb(r["Dmuth2"], ig1) if has_th2 else None
                d["Deta2th2"] = (_cb(r["Dmu2th2"], ig12)
                                 - _cb(_cb(r["Dmuth2"], g2g), ig1)
                                 if has_th2 else None)
                d["Deta3th"] = ((_cb(r["Dmu3th"], ig13)
                                 - _cb(_cb(3.0 * np.asarray(r["Dmu2th"],
                                                            dtype=float),
                                           g2g), ig12)
                                 + _cb(_cb(r["Dmuth"],
                                           3.0 * g2g * g2g - g3g), ig1))
                                if has_th2 else None)
        return d

    def preinitialize(self, y) -> dict | None:
        """One-shot pre-fit hook. mgcv ``family$preinitialize(y, family)``
        runs once before the first PIRLS iter and may return ``{"Theta":
        ...}`` (initial θ override) and/or ``{"y": ...}`` (transformed
        response). Default: no-op. Extended families with data-dependent
        θ start (Scat: ``c(1.5, log(0.8·sd(y)))``) override.
        """
        return None

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """One-shot post-fit hook — mgcv ``family$postproc(family, y,
        prior.weights, fitted, linear.predictors, offset, intercept)``.
        Extended families return ``{"null_deviance": ...}`` (via
        :func:`find_null_dev`, replacing the standard weighted-mean null
        deviance) and ``{"family_name": ...}`` (the θ-embedding relabel
        mgcv writes into ``family$family`` — "Scaled t(ν,σ)",
        "Negative Binomial(Θ)", "Tweedie(p=…)"). Default: empty dict
        (standard families keep estimate.gam's generics).
        """
        return {}

    def set_ind(self, ind) -> None:
        """Per-chunk row-subset hook for bam's chunked evaluation.

        mgcv's bgam.fit windows two kinds of per-row family state to the
        chunk rows: ``family$setInd(ind)`` (bam.r:1063, NULL-guarded —
        gfam's ``fi`` index) and ``subsety(y, ind)`` (bam.r:1068 — the
        censored families' ``attr(y,"censor")``, which rides the y being
        passed in mgcv but lives on the family in hea). Families carrying
        such state override; ``ind=None`` restores the full view. Default:
        no per-row state — no-op (mgcv's ``is.null(family$setInd)``).
        """
        return None

    # ----- qq.gam hooks (mgcv fix.family.qf / fix.family.rd,
    # plots.r:31-91). ``None`` means unavailable: the qq machinery then
    # tries simulation (rd) and finally falls back to a normal QQ plot.
    # Subclasses override with methods qf(p, mu, wt, scale) — the
    # response quantile function — and rd(rng, mu, wt, scale) — random
    # deviates (rng is a numpy Generator; mgcv uses R's global RNG).
    qf = None
    rd = None

    # mgcv residuals.gam dispatches to ``family$residuals(object, type)``
    # when the family supplies one (mgcv.r:3429) — general families
    # (gaulss & co) define their own residuals this way. hea's signature
    # is ``residuals(y, fitted, type)`` (the only pieces mgcv's hooks
    # read off the object). ``None`` means use the standard
    # deviance/pearson/working/response computations.
    residuals = None

    def initialize(self, y, wt) -> np.ndarray:
        """Starting μ̂ for PIRLS. Return a length-n positive (or family-valid)
        vector. Default: y; subclasses override when y can be at the boundary.
        """
        return np.asarray(y, dtype=float).copy()

    def gam_initialize(self, y, wt, n=None) -> np.ndarray:
        """Starting μ̂ for gam/bam PIRLS — mgcv patches some families'
        ``initialize`` before fitting (``fix.family``, gam.fit3.r:2550),
        making starts valid where glm's would refuse (e.g. gaussian-log
        with y ≤ 0). Default: same as ``initialize``; Gaussian overrides.

        ``n`` is the binomial trials vector from a ``cbind(succ, fail)``
        response (R's initialize keeps it distinct from the prior
        weights); only forwarded when given so ``initialize`` overrides
        without an ``n`` parameter stay valid.
        """
        if n is not None:
            return self.initialize(y, wt, n=n)
        return self.initialize(y, wt)

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(mu)))

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        """``-2·loglik + 2·k_overhead``. Returned without smoothing penalty;
        the caller adds ``+2·edf`` (or whatever df rule it uses).

        ``theta`` is accepted but ignored for standard families.
        Extended families read it for the AIC contribution from θ.
        """
        raise NotImplementedError

    def _aic_dev1(self, dev, scale, wt) -> float:
        """The ``dev1`` argument that ``aic(y, μ, dev1, wt, n)`` consumes.

        Mirrors ``gam.fit3.r:848-849``. For unknown-scale non-Gaussian families
        (Gamma, IG) and scale-known families (Poisson, binomial), this is
        ``scale · Σwt`` so the AIC uses the Pearson/REML scale estimator (or
        the fixed scale=1). Gaussian overrides this to return ``dev`` directly
        because the MLE σ² = dev/n has a closed form and mgcv prefers it
        over the moment estimator for the AIC.
        """
        return float(scale) * float(np.sum(np.asarray(wt, dtype=float)))

    def ls(self, y, wt, scale) -> np.ndarray:
        """Saturated log-likelihood at μ=y, plus its 1st/2nd derivative
        wrt ``log φ`` (φ = scale) — used by REML when scale is unknown.

        Returns a length-3 ``(ls0, d_ls/d_log_φ, d²_ls/d_log_φ²)`` array
        summed over observations. mgcv's ``family$ls`` returns ``d/dφ``
        and ``d²/dφ²``; we apply the chain rule internally so the caller
        works directly in the ρ = log φ parametrisation that REML and
        gam.fit3's outer optimiser use. For scale-known families
        (Poisson, binomial) ``d1 = d2 = 0``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}(link={self.link.name})"


class Gaussian(Family):
    """``y ~ N(μ, σ²)``; scale σ² is unknown."""
    name = "gaussian"
    canonical_link_name = "identity"
    scale_known = False

    def variance(self, mu): return np.ones_like(np.asarray(mu, dtype=float))
    def dvar(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d2var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))

    def gam_initialize(self, y, wt):
        # mgcv fix.family (gam.fit3.r:2550-2561): link-aware starting μ̂ so
        # gaussian fits with log/inverse links start inside the valid
        # region (glm's initialize refuses y ≤ 0 under a log link).
        y = np.asarray(y, dtype=float)
        if self.link.name == "inverse":
            return y + (y == 0.0) * np.std(y, ddof=1) * 0.01
        if self.link.name == "log":
            return np.maximum(y, 0.01 * np.std(y, ddof=1))
        return y.copy()

    def qf(self, p, mu, wt, scale):
        sd = np.sqrt(scale / np.asarray(wt, dtype=float))
        return _nmath.qnorm5_vec(p, mu, sd, True, False)

    def rd(self, rng, mu, wt, scale):
        return rng.normal(mu, np.sqrt(scale / np.asarray(wt, dtype=float)))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2

    def aic(self, y, mu, dev, wt, n, theta=None):
        # R's gaussian()$aic verbatim: nobs·(log(2π·dev/nobs)+1) + 2
        # − Σ log(wt), with nobs = length(y) (NOT Σwt — prior weights are
        # precision multipliers on σ², not extra observations; they enter
        # through the −Σlog(wt) Jacobian term instead). A zero weight makes
        # this Inf, exactly as in R. The +2 is the "+1 family df"
        # placeholder; downstream adds 2·edf for the model.
        wt = np.asarray(wt, dtype=float)
        nobs = float(np.asarray(y).shape[0])
        sigma2 = dev / nobs
        with np.errstate(divide="ignore"):
            log_wt_sum = float(np.sum(np.log(wt)))
        return nobs * (np.log(2.0 * np.pi * sigma2) + 1.0) + 2.0 - log_wt_sum

    def _aic_dev1(self, dev, scale, wt):
        # Gaussian MLE σ² = dev/n is closed-form, so mgcv passes dev directly
        # (gam.fit3.r:848). Caller's `dev` is the family deviance = RSS for
        # Gaussian. n_eff = Σwt and dev/n_eff = MLE σ².
        return float(dev)

    def ls(self, y, wt, scale):
        # mgcv: ls = -½·nobs·log(2π·φ) + ½·Σ log w[w>0]
        # so d/d(log φ) = -nobs/2, d²/d(log φ²) = 0. (Same algebraic shape
        # as InverseGaussian — neither family has a y-term involving φ.)
        # `nobs` here is the *count* of w>0 obs, not Σw — mgcv weights act
        # as a precision multiplier on σ², not as a sample-size multiplier.
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * nobs * np.log(2.0 * np.pi * scale)
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)


class Gamma(Family):
    """``y ~ Gamma(shape=1/φ, scale=μ·φ)``; mean μ, variance φ·μ²."""
    name = "Gamma"
    canonical_link_name = "inverse"
    scale_known = False

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float)
        return mu * mu
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 * mu
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        # mgcv: -2 wt (log(y/μ) - (y-μ)/μ); use ifelse(y==0, 1, y/μ) so
        # log(0) doesn't propagate when an observation is exactly zero.
        ratio = np.where(y == 0, 1.0, y / mu)
        return -2.0 * wt * (np.log(ratio) - (y - mu) / mu)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("Gamma family requires strictly positive responses")
        return y.copy()

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def aic(self, y, mu, dev, wt, n, theta=None):
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        disp = dev / n_eff
        # R's Gamma()$aic: -2·Σ wt·log dgamma(y; 1/disp, scale=μ·disp) + 2.
        # +2 mirrors mgcv (one "extra" df for the dispersion).
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _nmath._disp(
                "dgamma", _nmath.dgamma,
                [y, 1.0 / disp, np.asarray(mu, dtype=float) * disp], (True,))
        return -2.0 * float(np.sum(logp * wt)) + 2.0

    def ls(self, y, wt, scale):
        # Direct port of mgcv:::fix.family.ls's Gamma branch (raw d/dφ form),
        # then a log-scale chain rule to match the hea convention:
        #   d/dlogφ  = φ · d/dφ
        #   d²/dlogφ² = φ · d/dφ + φ² · d²/dφ²
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        y = y[good]
        w = wt[good]
        sw = scale / w                                     # per-obs scale
        # k1/k2/k3 depend on the observation only through sw, and lgamma/digamma/
        # trigamma(1/sw) (trigamma=zeta is the gamma-REML hot spot) are the cost.
        # With constant prior weights (the usual case) sw is constant, so
        # evaluate those on the UNIQUE sw values and index back: byte-identical
        # to the per-obs form (each k·[i] = the same scalar ops on sw[i]) but
        # O(unique) special-function calls instead of O(n).
        usw, inv = np.unique(sw, return_inverse=True)
        isw = 1.0 / usw
        lsw = np.log(usw)
        u_lg = gammaln(isw)
        u_dg = digamma(isw)
        u_tg = polygamma(1, isw)
        # k1 = -lgamma(1/sw) - log(sw)/sw - 1/sw
        k1 = (-u_lg - lsw / usw - isw)[inv]
        ls0 = float(np.sum(k1 - np.log(y)))
        # k2 = (digamma(1/sw) + log(sw)) / sw²       (mgcv's d/dφ)
        k2 = ((u_dg + lsw) / (usw * usw))[inv]
        d1_phi = float(np.sum(k2 / w))
        # k3 = (-trigamma(1/sw)/sw + 1 - 2 log(sw) - 2 digamma(1/sw)) / sw³
        k3 = ((-u_tg / usw + 1.0 - 2.0 * lsw - 2.0 * u_dg) / (usw ** 3))[inv]
        d2_phi = float(np.sum(k3 / (w * w)))             # mgcv's d²/dφ²
        d1 = scale * d1_phi
        d2 = scale * d1_phi + scale * scale * d2_phi
        return np.array([ls0, d1, d2], dtype=float)

    def qf(self, p, mu, wt, scale):
        # mgcv fix.family.qf: qgamma(p, shape=1/scale, scale=mu*scale) —
        # prior weights are ignored (as in mgcv).
        sc = np.asarray(mu, dtype=float) * scale
        return _nmath._disp(
            "qgamma", _nmath.qgamma, [p, 1.0 / scale, sc], (True, False))

    def rd(self, rng, mu, wt, scale):
        mu = np.asarray(mu, dtype=float)
        return rng.gamma(shape=1.0 / scale, scale=mu * scale)


class Poisson(Family):
    """``y ~ Poisson(μ)``; mean = variance = μ; scale fixed at 1."""
    name = "poisson"
    canonical_link_name = "log"
    scale_known = True

    def variance(self, mu): return np.asarray(mu, dtype=float).copy()
    def dvar(self, mu): return np.ones_like(np.asarray(mu, dtype=float))
    def d2var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))
    def d3var(self, mu): return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: 2 wt (y log(y/μ) - (y-μ)); with the convention 0·log(0/μ) = 0
        # so a y=0 row contributes 2 wt μ.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        positive = y > 0
        # avoid log(0) on y=0 rows by substituting μ inside the log (the
        # whole y·log term is then masked to 0 anyway).
        ratio = np.where(positive, y / np.where(positive, mu, 1.0), 1.0)
        contrib = np.where(positive,
                           wt * (y * np.log(ratio) - (y - mu)),
                           wt * mu)
        return 2.0 * contrib

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("negative values not allowed for the 'Poisson' family")
        # mgcv/R: mustart = y + 0.1 to keep log(μ) finite when y=0.
        return y + 0.1

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # Port of lme4's ``PoissonDist::aic`` (glmFamily.cpp:321-326):
        # ``-2 · Σ wt[i] · Rf_dpois(y[i], mu[i], TRUE)`` with sequential
        # reduction. :func:`_dpois_raw` is vectorized; the final sum uses
        # ``np.cumsum(...)[-1]`` for sequential bit-match to Eigen3.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        logp = _dpois_raw_disp(y, mu, True)
        return -2.0 * float(np.cumsum(logp * wt)[-1])

    def ls(self, y, wt, scale):
        # Saturated log-lik at μ=y; scale-known so d/dlogφ = d²/dlogφ² = 0.
        # mgcv: sum(dpois(y, y, log=TRUE) · w).
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _nmath._disp("dpois", _nmath.dpois, [y, y], (True,))
        ls0 = float(np.sum(logp * wt))
        return np.array([ls0, 0.0, 0.0], dtype=float)

    def qf(self, p, mu, wt, scale):
        return _nmath._disp("qpois", _nmath.qpois, [p, np.asarray(mu, dtype=float)],
                            (True, False))

    def rd(self, rng, mu, wt, scale):
        return rng.poisson(np.asarray(mu, dtype=float)).astype(float)


class Binomial(Family):
    """``y·m ~ Binomial(m, μ)``; ``y`` is the success proportion in [0,1],
    ``wt`` is the binomial size ``m`` (= 1 for Bernoulli).

    The cbind(success, failure) response form is handled by the *model*
    front ends (``gam``, ``glm``), which convert it to (proportion,
    weights·trials) before fitting — R's binomial ``initialize`` does the
    same. The trials vector ``n`` stays distinct from the prior weights
    in ``aic``/``ls``/``initialize`` (R keeps them separate whenever the
    caller also supplies its own ``weights=``); when ``n`` is omitted,
    the prior weights play both roles exactly as before.
    """
    name = "binomial"
    canonical_link_name = "logit"
    scale_known = True

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float)
        return mu * (1.0 - mu)
    def dvar(self, mu):
        return 1.0 - 2.0 * np.asarray(mu, dtype=float)
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), -2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv (C_binomial_dev_resids): 2 wt [ y_log_y(y, μ) + y_log_y(1-y, 1-μ) ]
        # where y_log_y(y, μ) = y log(y/μ) for y>0, else 0.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)

        def yly(a, b):
            # 0·log(0/0) := 0; mask both arguments inside the log so numpy
            # doesn't evaluate log(0) on the dead branch and emit warnings.
            pos = a > 0
            safe_a = np.where(pos, a, 1.0)
            safe_b = np.where(pos, b, 1.0)
            return np.where(pos, a * np.log(safe_a / safe_b), 0.0)

        return 2.0 * wt * (yly(y, mu) + yly(1.0 - y, 1.0 - mu))

    def initialize(self, y, wt, n=None, warn_non_integer=True):
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y values must be 0 <= y <= 1 for the 'binomial' family")
        if n is not None:
            # R binomial initialize, NCOL(y)==2 branch: mustart =
            # (n·y + 0.5)/(n + 1) — the trials vector, NOT the (possibly
            # prior-weight-scaled) wt. Only the starting point differs;
            # the converged fit is identical either way. (That branch's
            # non-integer-counts warning fired at the cbind intake.)
            n = np.asarray(n, dtype=float)
            return (n * y + 0.5) / (n + 1.0)
        # R's NCOL(y)==1 branch: m = weights·y must be integral counts.
        # The warning is gated on the family template being literally
        # "binomial" (quasibinomial's initialize is the same expression
        # with %s = "quasibinomial", so its guard is false → silent;
        # QuasiBinomial delegates here with warn_non_integer=False).
        if warn_non_integer:
            m = wt * y
            if np.any(np.abs(m - np.rint(m)) > 0.001):
                import warnings as _w
                _w.warn("non-integer #successes in a binomial glm!",
                        stacklevel=2)
        # mgcv/R: mustart = (wt·y + 0.5) / (wt + 1) keeps μ in (0,1) so the
        # logit link starts finite even when y is exactly 0 or 1.
        return (wt * y + 0.5) / (wt + 1.0)

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0) and np.all(mu < 1))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # Port of lme4's ``binomialDist::aic`` (glmFamily.cpp:204-213):
        # ``-2 · Σ (wt[i]/m[i]) · Rf_dbinom(round(m·y), round(m), μ, TRUE)``
        # with sequential reduction. :func:`_dbinom_raw` is vectorized;
        # final sum uses ``np.cumsum(...)[-1]`` for bit-match to Eigen3.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        # R binomial()$aic: ``m <- if (any(n > 1)) n else wt`` — with a
        # cbind(succ, fail) response, ``n`` is the trials vector kept by
        # initialize and ``wt`` carries any extra prior weights on top
        # (wt = pw·n), so the density is evaluated at the true counts
        # with coefficient wt/m = pw. Callers passing a scalar n (nobs)
        # or ones keep the historical wt-only path bit-for-bit.
        n_arr = None if n is None else np.asarray(n, dtype=float)
        if (n_arr is not None and n_arr.ndim == 1
                and n_arr.shape == y.shape and np.any(n_arr > 1.0)):
            good = n_arr > 0
            weight = np.where(good, wt / np.where(good, n_arr, 1.0), 0.0)
            s_arr = np.rint(np.where(good, n_arr * y, 0.0))
            size = np.rint(n_arr)
            logp = _dbinom_raw_disp(s_arr, size, mu, 1.0 - mu, True)
            terms = np.where(good & np.isfinite(logp), weight * logp, 0.0)
            return -2.0 * float(np.cumsum(terms)[-1])
        m = np.rint(wt)
        # Mask out m<=0; for those, contribution is 0.
        good = m > 0
        if not np.any(good):
            return 0.0
        s_arr = np.rint(np.where(good, m * y, 0.0))
        weight = np.where(good, wt / np.where(good, m, 1.0), 0.0)
        logp = _dbinom_raw_disp(s_arr, m, mu, 1.0 - mu, True)
        terms = weight * logp
        # Replace -inf entries (oob) by 0 so they don't contaminate the
        # sum (lme4 filters via the m<=0 branch which sets contribution
        # to 0; oob cases shouldn't occur for valid data anyway).
        terms = np.where(good & np.isfinite(logp), terms, 0.0)
        return -2.0 * float(np.cumsum(terms)[-1])

    def ls(self, y, wt, scale, n=None):
        # mgcv: ls = -binomial$aic(y, n, y, w, 0) / 2; scale-known.
        # ``n`` (trials, cbind responses) flows into the aic exactly as in
        # fix.family.ls (gam.fit3.r:2516) — None keeps the wt-only path.
        ls0 = -0.5 * self.aic(y, y, 0.0, wt, n)
        return np.array([ls0, 0.0, 0.0], dtype=float)

    def qf(self, p, mu, wt, scale):
        # mgcv fix.family.qf: ceiling non-integer denominators with a
        # warning; qbinom(p, wt, mu)/(wt + (wt==0)).
        wt = np.asarray(wt, dtype=float)
        if not np.allclose(wt, np.ceil(wt)):
            wt = np.ceil(wt)
            import warnings as _w
            _w.warn("non-integer binomial denominator: quantiles "
                    "incorrect", stacklevel=2)
        q = _nmath._disp("qbinom", _nmath.qbinom, [p, wt, np.asarray(mu, dtype=float)],
                         (True, False))
        return q / (wt + (wt == 0))

    def rd(self, rng, mu, wt, scale):
        wt = np.asarray(wt, dtype=float)
        d = rng.binomial(np.rint(wt).astype(np.int64),
                         np.asarray(mu, dtype=float))
        return d / (wt + (wt == 0))


class InverseGaussian(Family):
    """``y ~ IG(μ, φ)``; mean μ, variance φ·μ³; scale φ unknown."""
    name = "inverse.gaussian"
    canonical_link_name = "1/mu^2"
    scale_known = False

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float)
        return mu ** 3
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 3.0 * mu * mu
    def d2var(self, mu):
        return 6.0 * np.asarray(mu, dtype=float)
    def d3var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 6.0)

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: wt · (y - μ)² / (y · μ²).
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2 / (y * mu * mu)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError(
                "positive values only are allowed for the 'inverse.gaussian' family"
            )
        return y.copy()

    def validmu(self, mu):
        # R/stats: TRUE — boundary handling is via the link's valideta.
        return bool(np.all(np.isfinite(np.asarray(mu, dtype=float))))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv: sum(wt) · (1 + log(dev/sum(wt) · 2π)) + 3 · Σ wt · log(y) + 2.
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        sw = float(wt.sum())
        return (sw * (1.0 + np.log(dev / sw * 2.0 * np.pi))
                + 3.0 * float(np.sum(np.log(y) * wt)) + 2.0)

    def ls(self, y, wt, scale):
        # mgcv (raw φ form):
        #   ls0 = -½ · Σ log(2π φ y³) + ½ · Σ log w[w>0]
        #   d/dφ ls = -nobs/(2φ),  d²/dφ² ls = +nobs/(2φ²)
        # Chain rule to log-scale: d/dlogφ = -nobs/2, d²/dlogφ² = 0
        # (same algebraic cancellation as Gaussian — the y³ term has no φ).
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * float(np.sum(np.log(2.0 * np.pi * scale * y[good] ** 3)))
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)

    def rd(self, rng, mu, wt, scale):
        # mgcv rig(n, mu, scale) (Michael-Schucany-Haas): inverse Gaussian with
        # variance scale·μ³, drawn as n rnorm (squared) then n runif — that draw
        # order makes it bit-exact to R's rig on the hea.R.rng stream.
        mu = np.asarray(mu, dtype=float)
        n = mu.shape[0]
        y = np.asarray(rng.normal(size=n)) ** 2
        mys = mu * scale * y
        x = np.empty(n)
        big = mys < np.finfo(float).eps ** -0.5
        x[big] = mu[big] * (1.0 + 0.5 * (mys[big]
                            - np.sqrt(4.0 * mys[big] + mys[big] ** 2)))
        x[~big] = mu[~big] / mys[~big]
        swap = np.asarray(rng.uniform(size=n)) > mu / (mu + x)
        x[swap] = mu[swap] ** 2 / x[swap]
        return x


# ---------------------------------------------------------------------------
# Quasi: pure quasi-likelihood (no full likelihood, dispersion always
# estimated). Variance functions and deviances coincide with the matching
# parametric families, so we delegate to them rather than re-derive.
# ---------------------------------------------------------------------------


_QUASI_VARIANCE_FAMILIES = {
    "constant": Gaussian,         # V(μ) = 1
    "mu":       Poisson,          # V(μ) = μ
    "mu^2":     Gamma,             # V(μ) = μ²
    "mu^3":     InverseGaussian,  # V(μ) = μ³
    "mu(1-mu)": Binomial,         # V(μ) = μ(1-μ)
}


class Quasi(Family):
    """R's ``quasi(link, variance)``: pure quasi-likelihood.

    The mean–variance relation is set by ``variance=`` (one of
    ``"constant"``, ``"mu"``, ``"mu^2"``, ``"mu^3"``, ``"mu(1-mu)"``).
    Dispersion is always estimated from the Pearson χ²/df_resid; there is
    no proper likelihood, so ``aic`` and ``ls`` return NaN — Wald inference
    uses the t-distribution because the scale is unknown.

    Variance functions and deviances coincide with the matching parametric
    families, so this class delegates ``variance/dvar/dev_resids/validmu``
    to them. ``initialize`` matches R's ``quasi()`` (which differs from
    Binomial's precision-weighted start when ``variance='mu(1-mu)'``).
    """
    name = "quasi"
    canonical_link_name = "identity"  # R's quasi() default, regardless of variance
    scale_known = False
    # fix.family.link's table fallback (gam.fit3.r:2322): plain quasi →
    # "none"; quasipoisson/quasibinomial override with log/logit.
    _newton_canonical = "none"

    def __init__(self, link=None, variance: str = "constant"):
        if variance not in _QUASI_VARIANCE_FAMILIES:
            raise ValueError(
                f"quasi(): variance must be one of {list(_QUASI_VARIANCE_FAMILIES)}; "
                f"got {variance!r}"
            )
        self.variance_name = variance
        self._shadow = _QUASI_VARIANCE_FAMILIES[variance]()
        super().__init__(link=link)

    def variance(self, mu): return self._shadow.variance(mu)
    def dvar(self, mu):     return self._shadow.dvar(mu)
    def d2var(self, mu):    return self._shadow.d2var(mu)
    def d3var(self, mu):    return self._shadow.d3var(mu)

    def dev_resids(self, y, mu, wt, theta=None):
        return self._shadow.dev_resids(y, mu, wt)

    def initialize(self, y, wt):
        # R's quasi(variance='mu(1-mu)') initialize is
        # ``pmax(0.001, pmin(0.999, y))`` — clip y into the open
        # interval (0, 1). Different from binomial's
        # ``(wt·y + 0.5) / (wt + 1)`` smoothing.
        if self.variance_name == "mu(1-mu)":
            y = np.asarray(y, dtype=float)
            if np.any(y < 0) or np.any(y > 1):
                raise ValueError(
                    "y values must be 0 <= y <= 1 for quasi(variance='mu(1-mu)')"
                )
            return np.clip(y, 0.001, 0.999)
        return self._shadow.initialize(y, wt)

    def validmu(self, mu):
        return self._shadow.validmu(mu)

    def aic(self, y, mu, dev, wt, n, theta=None):
        return float("nan")

    def ls(self, y, wt, scale, n=None):
        # ``n`` (trials, quasibinomial cbind responses) accepted and
        # ignored — mgcv's quasi ls(y,w,n,scale) never reads it.
        # Extended quasi-likelihood saturated piece (Nelder & Pregibon 1987;
        # McCullagh & Nelder 1989, §9.6). mgcv's ``quasi$ls`` drops both the
        # log(2π) and log V(y) constants — neither depends on φ or ρ, so they
        # don't affect REML's argmin; dropping log V(y) also sidesteps log 0
        # when y is at the support boundary (e.g. count zeros under
        # variance='mu'). What's left is the Gaussian φ-shape:
        #
        #     ls0 = -n_obs/2 · log φ + ½·Σ_{w>0} log w
        #     d/dφ ls = -n_obs/(2φ),  d²/dφ² ls = n_obs/(2φ²)
        #
        # Chain-ruled to log φ (hea's convention):
        #     d/dlog φ  = -n_obs/2
        #     d²/dlog φ² = -n_obs/2 + n_obs/2 = 0
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * nobs * np.log(scale)
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)

    def __repr__(self) -> str:
        return f"quasi(link={self.link.name}, variance={self.variance_name!r})"


class QuasiPoisson(Quasi):
    """R's ``quasipoisson(link="log")``: Poisson variance/deviance with
    estimated dispersion (no likelihood — AIC/logLik are NaN, EQL ls).

    Differs from ``Quasi(variance="mu")`` exactly where R differs:
    default link log, poisson's ``initialize`` (μ₀ = y + 0.1 with the
    negative-y check), canonical log for the Newton/Fisher switch
    (gam.fit3.r:2318), and the family name printers show.
    """
    name = "quasipoisson"
    canonical_link_name = "log"
    _newton_canonical = "log"

    def __init__(self, link=None):
        super().__init__(link=link, variance="mu")

    def initialize(self, y, wt):
        # R quasipoisson shares poisson's initialize verbatim.
        return self._shadow.initialize(y, wt)

    __repr__ = Family.__repr__


class QuasiBinomial(Quasi):
    """R's ``quasibinomial(link="logit")``: binomial variance/deviance
    with estimated dispersion (no likelihood — AIC/logLik are NaN).

    Shares binomial's ``initialize`` verbatim — the proportion-smoothing
    mustart and the ``cbind(succ, fail)`` trials form (which warns on
    non-integer counts, like R) — unlike ``Quasi(variance="mu(1-mu)")``'s
    clip-style start. Canonical logit (gam.fit3.r:2319).
    """
    name = "quasibinomial"
    canonical_link_name = "logit"
    _newton_canonical = "logit"

    def __init__(self, link=None):
        super().__init__(link=link, variance="mu(1-mu)")

    def initialize(self, y, wt, n=None):
        # R quasibinomial shares binomial's initialize verbatim (incl.
        # the n-form mustart for cbind responses) — minus the
        # non-integer-#successes warning, whose template guard
        # ("quasibinomial" == "binomial") is false in R.
        return self._shadow.initialize(y, wt, n=n, warn_non_integer=False)

    __repr__ = Family.__repr__


# ---------------------------------------------------------------------------
# Tweedie / tw — Dunn-Smyth (2005) series implementation.
#
# Tweedie EDF for ``1 < p < 2`` is the compound Poisson-Gamma: a Poisson(λ)
# count of Gamma jumps. Mean μ, variance ``φ·μ^p``; the density mixes a
# point mass at 0 with a continuous part on ``y > 0``. With ``α = (2-p)/(1-p)``
# (negative for 1<p<2):
#
#     y = 0:  log f(0; μ, φ, p) = -μ^(2-p) / (φ·(2-p))
#     y > 0:  log f(y; μ, φ, p) = -log y + log a(y, φ, p)
#                                + y·μ^(1-p)/(φ·(1-p)) - μ^(2-p)/(φ·(2-p))
#
# where ``a(y, φ, p) = Σ_{j≥1} W_j``,
#
#     log W_j = j·log z - log Γ(j+1) - log Γ(-j·α),
#     log z   = -α·log y + α·log(p-1) - (1-α)·log φ - log(2-p).
#
# We sum log-W_j outward from the dominant index ``j*`` (where d_j log W_j = 0)
# until terms drop ``≥ ld_eps`` below the running max, then log-sum-exp. The
# moments E_p[j] and Var_p[j] under ``p_j = W_j / Σ W_k`` give the φ-derivatives
# of log a:  d/dlog φ  log a = -(1-α)·E[j] ;  d²/dlog φ² log a = (1-α)²·Var[j].
# Direct port of mgcv's ``tweedious.c`` / ``ldTweedie``.
# ---------------------------------------------------------------------------


# Series tail tolerance: terms log W_j < log W_max - LD_EPS are dropped. mgcv
# uses ~36 (≈ -log(eps^½)); a touch tighter than the .Machine$double.eps
# threshold used in tweedious.c, but well past where summands matter.
_LD_EPS = 36.0
# Hard cap on series length to bound worst-case latency at extreme (y, φ, p).
# In practice the series is centred near j* with width ~√j*, so the loop
# exits via the LD_EPS gate long before this; the cap is purely a safety net.
_LD_J_MAX = 100000


def _tweedie_log_a_one(y_i: float, phi_i: float, p: float):
    """Series approximation log a(y, φ, p) = log Σ_{j≥1} W_j for one y > 0.

    Per-row reference for :func:`_tweedie_log_a_vec`. Returns
    ``(log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)`` under
    ``p_j = W_j/Σ W_k``: E[j], Var[j], E[j·ψ(-j·α)], and mgcv's three
    p-parameterisation working-derivative accumulators E[∂logW/∂p],
    E[(∂logW/∂p)²+∂²logW/∂p²], E[∂logW/∂p·j/(1−p)+∂²logW/∂p∂logφ]
    (tweedious, misc.c:346-503). The first two feed the φ-derivatives of
    log a; E[jψ] the p-derivative (Tweedie.dls_dp); the last three the
    p-second-derivatives (Tweedie._d2ls_dp). nmath (R's Rmath) special
    functions; ``wp1²+wp2`` combined per term (no moment-split cancellation).
    """
    om1 = 1.0 - p                  # negative
    onep2 = om1 * om1
    onep3 = onep2 * om1
    tm = 2.0 - p                   # positive
    alpha = tm / om1               # negative
    one_minus_alpha = 1.0 - alpha  # > 1; equals 1/(p-1)

    ly = np.log(y_i)
    rho = np.log(phi_i)
    # log W_j = j·log_z - lgamma(j+1) - lgamma(-j·α).
    log_z = (-alpha * ly + alpha * np.log(p - 1.0)
             - one_minus_alpha * rho - np.log(tm))
    log_neg = np.log(-om1) + rho
    wp_base = log_neg / onep2 - alpha / om1 + 1.0 / tm
    wp2_base = (2.0 * log_neg / onep3 - (3.0 * alpha - 2.0) / onep2
                + 1.0 / (tm * tm))

    j_star = np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha)
    j_star = max(j_star, 1.0)
    j_int = max(1, int(round(j_star)))

    def _lw(j):
        return (j * log_z - _nmath._lgammafn(j + 1.0)
                - _nmath._lgammafn(-j * alpha))

    # Walk outward from j_int both ways (mgcv pure-eps break, no `near` band).
    log_max = _lw(j_int)
    j_list = [float(j_int)]
    lw_list = [log_max]
    j = j_int + 1
    while j < _LD_J_MAX:
        v = _lw(j)
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        if v - log_max < -_LD_EPS:
            break
        j += 1
    j = j_int - 1
    while j >= 1:
        v = _lw(j)
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        if v - log_max < -_LD_EPS:
            break
        j -= 1

    j_arr = np.array(j_list, dtype=float)
    lw_arr = np.array(lw_list, dtype=float)
    weights = np.exp(lw_arr - log_max)
    sum_w = float(np.sum(weights))
    log_a = log_max + float(np.log(sum_w))

    p_w = weights / sum_w
    j_bar = float(np.sum(p_w * j_arr))
    j_var = float(np.sum(p_w * j_arr * j_arr) - j_bar * j_bar)
    # ψ(-j·α) well-defined for α<0, j≥1 (so -j·α > 0); same j-grid as the sum.
    psi_arr = _nmath.psigamma_vec(-j_arr * alpha, 0.0)
    trig_arr = _nmath.psigamma_vec(-j_arr * alpha, 1.0)
    j_psi_bar = float(np.sum(p_w * j_arr * psi_arr))
    # mgcv p-param working derivatives (misc.c:289-293,333-334).
    xj = (j_arr / onep2) * psi_arr
    wp1 = j_arr * wp_base + xj - j_arr * (ly / onep2)
    wp2 = (j_arr * wp2_base + 2.0 * xj / om1
           - trig_arr * (j_arr / onep2) ** 2 - 2.0 * j_arr * (ly / onep3))
    m_wp1 = float(np.sum(p_w * wp1))
    m_comb = float(np.sum(p_w * (wp1 * wp1 + wp2)))
    m_dwpp = float(np.sum(p_w * (wp1 * j_arr / om1 + j_arr / onep2)))
    return (log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)


def _tweedie_series_rs(ly, j_int, alpha, w_base, wp_base, wp2_base, onep, J):
    """Rust ``tweedious`` sweep for the scalar-p saturated-likelihood series.

    The rust kernel builds the shared length-``J`` nmath (R's Rmath) tables
    internally (α constant) and accumulates the moments + mgcv's working
    derivatives; this wrapper just verifies ``J`` covers every row's eps window
    (a cheap O(n) right-edge check via the nmath lgamma at the boundary and each
    row's peak — never the n×J matrix), growing it if needed. Returns the seven
    ``(n_active,)`` arrays ``(log_a, E[j], Var[j], E[jψ], m_wp1, m_comb,
    m_dwpp)`` in :func:`_tweedie_log_a_vec`'s order."""
    ly = np.ascontiguousarray(np.asarray(ly, dtype=float))
    j_int = np.ascontiguousarray(np.asarray(j_int, dtype=np.int64))
    log_z = w_base - alpha * ly  # lw at j=1 base (only for the J-coverage check)
    J = int(J)
    while J < _LD_J_MAX:
        # eps-window right-edge check: lw at j=J vs each row's integer peak.
        lw_right = J * log_z - _nmath._lgammafn(J + 1.0) - _nmath._lgammafn(-J * alpha)
        lw_peak = (j_int * log_z - _nmath._lgammafn_arr(j_int + 1.0)
                   - _nmath._lgammafn_arr(-j_int * alpha))
        if not bool(np.any(lw_right >= lw_peak - _LD_EPS)):
            break
        J = min(J * 2, _LD_J_MAX)
    res = _rs_tweedie_series(
        ly, j_int, float(alpha),
        np.ascontiguousarray(np.asarray(w_base, dtype=float)),
        np.ascontiguousarray(np.asarray(wp_base, dtype=float)),
        np.ascontiguousarray(np.asarray(wp2_base, dtype=float)),
        float(onep), int(J), float(_LD_EPS))
    return (res[:, 0], res[:, 1], res[:, 2], res[:, 3],
            res[:, 4], res[:, 5], res[:, 6])


def _tweedie_log_a_vec(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Vectorised over y (and per-obs phi). Returns seven arrays of shape
    ``y.shape``: ``log_a``, ``j_bar`` (E[j]), ``j_var`` (Var[j]),
    ``j_psi_bar`` (E[jψ]) and mgcv's THREE p-parameterisation working-derivative
    accumulators ``m_wp1`` (E[wp1]), ``m_comb`` (E[wp1²+wp2]),
    ``m_dwpp`` (E[wp1·j/(1−p)+wpp]) — the same set as :func:`_tweedie_log_a_one`,
    matching mgcv ``tweedious`` (`wdlogwdp/wi`, `wdW2d2W/wi`, `dWpp/wi`,
    misc.c:346-503) BEFORE the θ-chain. Combining ``wp1²+wp2`` per term (rather
    than the old separate ``E[(jψ)²]``/``E[j²ψ']`` moments + subtraction) avoids
    the ~1e-11 catastrophic cancellation in the 2nd derivatives. Entries with
    y==0 are 0 (the y=0 row uses the closed-form point mass, not the series).
    Per-obs phi handles weights via ``φ_i = φ/wt_i``.

    Builds a fixed ``j`` grid wide enough to cover every active row's
    eps-truncated series tail, then evaluates the (n_active, J) matrix
    of ``log W_j`` and reduces along ``j`` in one pass. ``J`` is sized
    so the eps gate fires within the grid for every row — agrees with
    the per-row :func:`_tweedie_log_a_one` walk to ~1e-13 absolute on
    log_a / moments (well below mgcv-oracle test tolerances).
    """
    y = np.asarray(y, dtype=float)
    phi_arr = np.broadcast_to(np.asarray(phi, dtype=float), y.shape).astype(float, copy=True)
    log_a = np.zeros_like(y)
    j_bar = np.zeros_like(y)
    j_var = np.zeros_like(y)
    j_psi_bar = np.zeros_like(y)
    m_wp1 = np.zeros_like(y)
    m_comb = np.zeros_like(y)
    m_dwpp = np.zeros_like(y)
    flat_y = y.ravel()
    flat_phi = phi_arr.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)
    ya = flat_y[active]
    pha = flat_phi[active]

    om1 = 1.0 - p
    tm = 2.0 - p
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha
    onep2 = om1 * om1
    onep3 = onep2 * om1

    ly = np.log(ya)
    rho = np.log(pha)
    log_z = (-alpha * ly + alpha * np.log(p - 1.0)
             - one_minus_alpha * rho - np.log(tm))
    # mgcv's per-row p-bases (misc.c:230-232): w_base folds in rho (so per-obs
    # phi makes them per-row even though α is constant); wp_base/wp2_base are the
    # p-derivative bases multiplied by j inside the sweep.
    w_base = alpha * np.log(p - 1.0) + rho / om1 - np.log(tm)
    log_neg = np.log(-om1) + rho
    wp_base = log_neg / onep2 - alpha / om1 + 1.0 / tm
    wp2_base = (2.0 * log_neg / onep3 - (3.0 * alpha - 2.0) / onep2
                + 1.0 / (tm * tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())
    n_active = ya.size

    # Window width from the LOCAL curvature of log W_j, NOT the old
    # conservative ``1/|alpha|`` bound (which over-allocated ~26× at p→2:
    # J≈3300 where the true eps-window is ~125 wide, then masked the excess
    # — mgcv's C `tweedious` sweeps a per-row window with early break,
    # misc.c:301). Near its peak (j ≈ j_int) the log series term is locally
    # Gaussian: d²logW_j/dj² = −(ψ'(j+1) + α²ψ'(−jα)), so it falls _LD_EPS
    # below the peak within ±√(2·_LD_EPS·var) (var = −1/that 2nd deriv).
    # Size J to the widest row's right edge; the grow-loop below is the
    # bit-exactness GUARANTEE — it stops only once the eps gate has provably
    # fired inside the window for every row (the rightmost column is below
    # `log_max − _LD_EPS`), so by unimodality no retained term is ever
    # truncated and the sum is identical to any wider J. `j_int + half` is
    # monotone increasing in j_int (ψ' decreasing ⇒ peak_var increasing), so
    # max_i(j_int_i + half_i) is attained at j_int_max — evaluate the curvature
    # at that one scalar instead of ψ' over the whole n_active j_int array
    # (which was a REML-loop hot spot via scipy zeta). Identical J, far cheaper.
    peak_var = 1.0 / (
        polygamma(1, j_int_max + 1.0)
        + alpha * alpha * polygamma(1, -j_int_max * alpha)
    )
    half = np.sqrt(2.0 * _LD_EPS * max(float(peak_var), 0.0))
    right = int(np.ceil(j_int_max + half)) + 16
    J = min(max(right, j_int_max + 1), _LD_J_MAX)

    if _rs_tweedie_series is not None:
        # Rust per-row sweep (mgcv tweedious): builds the shared length-J nmath
        # tables internally (α constant) and accumulates the moments + mgcv's
        # working derivatives. Python passes the p-bases; J sizes the table.
        (out_la, out_jb, out_jv, out_jpb,
         out_m1, out_mc, out_md) = _tweedie_series_rs(
            ly, j_int, alpha, w_base, wp_base, wp2_base, om1, J)
    else:
        out_la = np.empty(n_active)
        out_jb = np.empty(n_active)
        out_jv = np.empty(n_active)
        out_jpb = np.empty(n_active)
        out_m1 = np.empty(n_active)
        out_mc = np.empty(n_active)
        out_md = np.empty(n_active)
        while True:
            j_grid = np.arange(1, J + 1, dtype=float)
            # nmath (R's Rmath) special functions — NOT scipy — so the series
            # matches the arm64 R build to the libm floor.
            lgamma_jp1 = _nmath._lgammafn_arr(j_grid + 1.0)
            lgamma_neg_ja = _nmath._lgammafn_arr(-j_grid * alpha)
            psi_arr = _nmath.psigamma_vec(-j_grid * alpha, 0.0)
            trig_arr = _nmath.psigamma_vec(-j_grid * alpha, 1.0)

            # Chunk on the n_active axis to bound the (chunk, J) working set.
            chunk = max(1, _chunk_bytes // (48 * J))
            grow = False
            for s in range(0, n_active, chunk):
                e = min(s + chunk, n_active)
                lz_c = log_z[s:e]
                lw = (j_grid[None, :] * lz_c[:, None]
                      - lgamma_jp1[None, :] - lgamma_neg_ja[None, :])
                log_max = np.max(lw, axis=1)
                above_eps = lw >= (log_max[:, None] - _LD_EPS)
                # Right boundary still carrying weight ⇒ the eps-window extends
                # past J for some row ⇒ grow and recompute (rare — the analytic
                # width above covers it first-try in practice).
                if J < _LD_J_MAX and above_eps[:, -1].any():
                    grow = True
                    break
                w = np.where(above_eps, np.exp(lw - log_max[:, None]), 0.0)
                sum_w = np.sum(w, axis=1)
                out_la[s:e] = log_max + np.log(sum_w)
                p_w = w / sum_w[:, None]
                jb_c = np.sum(p_w * j_grid[None, :], axis=1)
                out_jb[s:e] = jb_c
                out_jv[s:e] = (
                    np.sum(p_w * j_grid[None, :] ** 2, axis=1) - jb_c * jb_c)
                out_jpb[s:e] = np.sum(
                    p_w * j_grid[None, :] * psi_arr[None, :], axis=1)
                # mgcv p-param working derivatives of log W_j (misc.c:289-293,
                # 333-334); combine wp1²+wp2 per term, then reduce.
                xj = (j_grid / onep2)[None, :] * psi_arr[None, :]
                wp1 = (j_grid[None, :] * wp_base[s:e, None] + xj
                       - j_grid[None, :] * (ly[s:e] / onep2)[:, None])
                wp2 = (j_grid[None, :] * wp2_base[s:e, None]
                       + 2.0 * xj / om1
                       - trig_arr[None, :] * ((j_grid / onep2) ** 2)[None, :]
                       - 2.0 * j_grid[None, :] * (ly[s:e] / onep3)[:, None])
                out_m1[s:e] = np.sum(p_w * wp1, axis=1)
                out_mc[s:e] = np.sum(p_w * (wp1 * wp1 + wp2), axis=1)
                out_md[s:e] = np.sum(
                    p_w * (wp1 * j_grid[None, :] / om1
                           + (j_grid / onep2)[None, :]), axis=1)
            if not grow:
                break
            J = min(J * 2, _LD_J_MAX)

    flat_la = log_a.ravel()
    flat_jb = j_bar.ravel()
    flat_jv = j_var.ravel()
    flat_jpb = j_psi_bar.ravel()
    flat_la[active] = out_la
    flat_jb[active] = out_jb
    flat_jv[active] = out_jv
    flat_jpb[active] = out_jpb
    m_wp1.ravel()[active] = out_m1
    m_comb.ravel()[active] = out_mc
    m_dwpp.ravel()[active] = out_md
    return (log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)


def _tweedie_log_a_vec_pv(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Per-observation-``p`` variant of :func:`_tweedie_log_a_vec`
    (mgcv's ``C_tweedious2`` case — ldTweedie called with vector
    ``theta``/``rho``, gam.fit3.r:2952-2956). Same seven return arrays;
    the special-function tables become (rows, J) matrices because α
    varies by row. Kept separate from the scalar-``p`` function so its
    existing consumers stay byte-identical."""
    y = np.asarray(y, dtype=float)
    phi_arr = np.broadcast_to(
        np.asarray(phi, dtype=float), y.shape).astype(float, copy=True)
    p_arr = np.broadcast_to(
        np.asarray(p, dtype=float), y.shape).astype(float, copy=True)
    log_a = np.zeros_like(y)
    j_bar = np.zeros_like(y)
    j_var = np.zeros_like(y)
    j_psi_bar = np.zeros_like(y)
    m_wp1 = np.zeros_like(y)
    m_comb = np.zeros_like(y)
    m_dwpp = np.zeros_like(y)
    flat_y = y.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)
    ya = flat_y[active]
    pha = phi_arr.ravel()[active]
    pa = p_arr.ravel()[active]

    om1 = 1.0 - pa
    tm = 2.0 - pa
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha
    onep2 = om1 * om1
    onep3 = onep2 * om1

    ly = np.log(ya)
    rho = np.log(pha)
    log_z = (-alpha * ly + alpha * np.log(pa - 1.0)
             - one_minus_alpha * rho - np.log(tm))
    # Per-row p-bases (mgcv tweedious2, misc.c:230-232 per observation).
    w_base = alpha * np.log(pa - 1.0) + rho / om1 - np.log(tm)
    log_neg = np.log(-om1) + rho
    wp_base = log_neg / onep2 - alpha / om1 + 1.0 / tm
    wp2_base = (2.0 * log_neg / onep3 - (3.0 * alpha - 2.0) / onep2
                + 1.0 / (tm * tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())
    n_active = ya.size

    # Per-row eps-window from the local curvature of log W_j (α is per-row
    # here), NOT the old ``1/min|alpha|`` worst-row bound: see the scalar
    # :func:`_tweedie_log_a_vec` for the derivation and the bit-exactness
    # guarantee (the grow-loop only stops once the eps gate has fired inside
    # the window for every row). Size the shared grid to the widest row.
    peak_var = 1.0 / (
        polygamma(1, j_int + 1.0)
        + alpha * alpha * polygamma(1, -j_int * alpha)
    )
    half = np.sqrt(2.0 * _LD_EPS * np.maximum(peak_var, 0.0))
    right = int(np.ceil(float(np.max(j_int + half)))) + 16
    J = min(max(right, j_int_max + 1), _LD_J_MAX)

    if _rs_tweedie_series_pv is not None:
        # Rust per-row windowed sweep (mgcv tweedious2): recomputes the −jα
        # special functions inside each row's eps-window via the nmath ports
        # (same C source as R/mgcv) and accumulates the moments + mgcv's working
        # derivatives — never the dense (n, J) matrix. The kernel caps the
        # up-sweep at `_LD_J_MAX` and stops per-row once the eps gate fires.
        res = _rs_tweedie_series_pv(
            np.ascontiguousarray(ly),
            np.ascontiguousarray(j_int.astype(np.int64)),
            np.ascontiguousarray(alpha),
            np.ascontiguousarray(w_base),
            np.ascontiguousarray(wp_base),
            np.ascontiguousarray(wp2_base),
            np.ascontiguousarray(om1),
            float(_LD_EPS), int(_LD_J_MAX))
        out_la = res[:, 0]
        out_jb = res[:, 1]
        out_jv = res[:, 2]
        out_jpb = res[:, 3]
        out_m1 = res[:, 4]
        out_mc = res[:, 5]
        out_md = res[:, 6]
    else:
        out_la = np.empty(n_active)
        out_jb = np.empty(n_active)
        out_jv = np.empty(n_active)
        out_jpb = np.empty(n_active)
        out_m1 = np.empty(n_active)
        out_mc = np.empty(n_active)
        out_md = np.empty(n_active)
        while True:
            j_grid = np.arange(1, J + 1, dtype=float)
            lgamma_jp1 = _nmath._lgammafn_arr(j_grid + 1.0)

            # per-row α ⇒ the -jα tables are (chunk, J); budget ~12 J-wide
            # doubles per row in flight → 96 J bytes per row.
            chunk = max(1, _chunk_bytes // (96 * J))
            grow = False
            for s in range(0, n_active, chunk):
                e = min(s + chunk, n_active)
                lz_c = log_z[s:e]
                op_c = om1[s:e]
                op2_c = onep2[s:e]
                op3_c = onep3[s:e]
                nja = -j_grid[None, :] * alpha[s:e, None]      # (c, J), > 0
                lw = (j_grid[None, :] * lz_c[:, None]
                      - lgamma_jp1[None, :] - _nmath._lgammafn_arr(nja))
                log_max = np.max(lw, axis=1)
                above_eps = lw >= (log_max[:, None] - _LD_EPS)
                if J < _LD_J_MAX and above_eps[:, -1].any():
                    grow = True
                    break
                w = np.where(above_eps, np.exp(lw - log_max[:, None]), 0.0)
                sum_w = np.sum(w, axis=1)
                out_la[s:e] = log_max + np.log(sum_w)
                p_w = w / sum_w[:, None]
                jb_c = np.sum(p_w * j_grid[None, :], axis=1)
                out_jb[s:e] = jb_c
                out_jv[s:e] = (
                    np.sum(p_w * j_grid[None, :] ** 2, axis=1) - jb_c * jb_c)
                psi_c = _nmath.psigamma_vec(nja, 0.0)
                trig_c = _nmath.psigamma_vec(nja, 1.0)
                out_jpb[s:e] = np.sum(p_w * j_grid[None, :] * psi_c, axis=1)
                # mgcv p-param working derivatives (per-row α), wp1²+wp2 combined.
                jo2 = j_grid[None, :] / op2_c[:, None]
                xj = jo2 * psi_c
                wp1 = (j_grid[None, :] * wp_base[s:e, None] + xj
                       - j_grid[None, :] * (ly[s:e] / op2_c)[:, None])
                wp2 = (j_grid[None, :] * wp2_base[s:e, None]
                       + 2.0 * xj / op_c[:, None] - trig_c * jo2 * jo2
                       - 2.0 * j_grid[None, :] * (ly[s:e] / op3_c)[:, None])
                out_m1[s:e] = np.sum(p_w * wp1, axis=1)
                out_mc[s:e] = np.sum(p_w * (wp1 * wp1 + wp2), axis=1)
                out_md[s:e] = np.sum(
                    p_w * (wp1 * j_grid[None, :] / op_c[:, None] + jo2), axis=1)
            if not grow:
                break
            J = min(J * 2, _LD_J_MAX)

    log_a.ravel()[active] = out_la
    j_bar.ravel()[active] = out_jb
    j_var.ravel()[active] = out_jv
    j_psi_bar.ravel()[active] = out_jpb
    m_wp1.ravel()[active] = out_m1
    m_comb.ravel()[active] = out_mc
    m_dwpp.ravel()[active] = out_md
    return (log_a, j_bar, j_var, j_psi_bar, m_wp1, m_comb, m_dwpp)


def _ld_tweedie_work(y, mu, theta, rho, a: float = 1.001,
                     b: float = 1.999) -> np.ndarray:
    """mgcv ``ldTweedie`` in the working (ρ, θ) parameterization with
    ``all.derivs=TRUE`` (gam.fit3.r:2838-3035): log Tweedie density and
    derivatives for vector ``mu``/``theta``/``rho``, with
    p = (a + b·e^θ)/(1 + e^θ) ∈ (a, b) and φ = e^ρ.

    Returns the (n, 10) array in mgcv's column order
    ``[l, ρ, ρρ, θ, θθ, θρ, μ, μμ, μθ, μρ]`` — exactly what twlss's ll
    consumes (gamlss.r:2575-2580). The closed-form saddle/zero parts
    and the (p, φ) → (θ, ρ) chain are line-by-line ports; the series
    part (the C ``tweedious2`` call) runs the Dunn-Smyth moment
    machinery (:func:`_tweedie_log_a_vec_pv`) with the same eps gate,
    converted to working-parameter derivatives via

        ∂log W_j/∂ρ = −(1−α)·j,   ∂log W_j/∂p = j·L′ + α′·j·ψ(−jα),

    L = log z, α = (2−p)/(1−p), and the chain p(θ).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    mu = np.ascontiguousarray(
        np.broadcast_to(np.asarray(mu, dtype=float), y.shape))
    theta = np.ascontiguousarray(
        np.broadcast_to(np.asarray(theta, dtype=float), y.shape))
    rho = np.ascontiguousarray(
        np.broadcast_to(np.asarray(rho, dtype=float), y.shape))
    if not (1.0 < a < b < 2.0):
        raise ValueError("1<a<b<2 (strict) required")

    # p(θ) and its θ-derivatives, the ±θ-stable branches
    # (gam.fit3.r:2849-2858)
    pos = theta > 0
    eth = np.exp(-np.abs(theta))
    p = np.where(pos, (b + a * eth) / (1.0 + eth),
                 (b * eth + a) / (eth + 1.0))
    # R-level ^ is R_pow per element: ^2 → x·x (numpy's **2 square fast
    # path matches) but ^3 → SEQUENTIAL x·x·x for |x|≤11 — numpy **3
    # takes the libm-pow loop and drifts the last ulp (receipt:
    # 26k/100k elements differ on the (1,2] range).
    d1eth = 1.0 + eth
    d3 = d1eth * d1eth * d1eth
    dpth1 = eth * (b - a) / (d1eth * d1eth)
    dpth2 = np.where(
        pos,
        ((a - b) * eth + (b - a) * eth * eth) / d3,
        ((a - b) * eth * eth + (b - a) * eth) / d3,
    )
    phi = np.exp(rho)

    ld = np.zeros((n, 10))

    # y == 0 rows: closed forms (gam.fit3.r:2920-2937), mu > 0 gate
    zm = (y == 0.0) & (mu > 0.0)
    if np.any(zm):
        mu_z = mu[zm]
        p_z = p[zm]
        phi_z = phi[zm]
        lmu_z = np.log(mu_z)
        ld[zm, 0] = -mu_z ** (2.0 - p_z) / (phi_z * (2.0 - p_z))
        ld[zm, 1] = -ld[zm, 0] / phi_z
        ld[zm, 2] = -2.0 * ld[zm, 1] / phi_z
        ld[zm, 3] = -ld[zm, 0] * (lmu_z - 1.0 / (2.0 - p_z))
        ld[zm, 4] = (2.0 * ld[zm, 3] / (2.0 - p_z)
                     + ld[zm, 0] * lmu_z ** 2)
        ld[zm, 5] = -ld[zm, 3] / phi_z
        mup = mu_z ** p_z
        ld[zm, 6] = -mu_z / (mup * phi_z)
        ld[zm, 7] = -(1.0 - p_z) / (mup * phi_z)
        ld[zm, 8] = lmu_z * mu_z / (mup * phi_z)
        ld[zm, 9] = -ld[zm, 6] / phi_z

    # y > 0 rows: saddle part in (p, φ) (gam.fit3.r:2974-2989)
    ind = y > 0.0
    any_pos = bool(np.any(ind))
    if any_pos:
        y_i = y[ind]
        mu_i = mu[ind]
        p_i = p[ind]
        phii = phi[ind]
        log_mu = np.log(mu_i)
        onep = 1.0 - p_i
        twop = 2.0 - p_i
        mu1p = mu_i ** onep
        k_theta = mu_i * mu1p / twop          # mu^(2-p)/(2-p)
        theta_s = mu1p / onep                 # mu^(1-p)/(1-p)
        a1 = y_i / onep - mu_i / twop
        l_base = mu1p * a1 / phii
        ld[ind, 0] = l_base - np.log(y_i)
        ld[ind, 1] = -l_base / phii
        ld[ind, 2] = 2.0 * l_base / phii ** 2
        x_ = (theta_s * y_i * (1.0 / onep - log_mu) / phii
              + k_theta * (log_mu - 1.0 / twop) / phii)
        ld[ind, 3] = x_
        ld[ind, 4] = (theta_s * y_i
                      * (log_mu ** 2 - 2.0 * log_mu / onep
                         + 2.0 / onep ** 2) / phii
                      - k_theta * (log_mu ** 2 - 2.0 * log_mu / twop
                                   + 2.0 / twop ** 2) / phii)
        ld[ind, 5] = -x_ / phii

    # transform (p, φ) derivatives to working (θ, ρ)
    # (gam.fit3.r:2990-2997) — all rows, zeros included
    ld[:, 2] = ld[:, 2] * phi ** 2 + ld[:, 1] * phi
    ld[:, 1] = ld[:, 1] * phi
    ld[:, 4] = ld[:, 4] * dpth1 ** 2 + ld[:, 3] * dpth2
    ld[:, 3] = ld[:, 3] * dpth1
    ld[:, 5] = ld[:, 5] * dpth1 * phi

    # all.derivs μ-columns for y > 0 (gam.fit3.r:2999-3009)
    if any_pos:
        a2 = mu1p / (mu_i * phii)             # 1/(mu^p · φ)
        ld[ind, 6] = a2 * (onep * a1 - mu_i / twop)
        ld[ind, 7] = -a2 * (onep * p_i * a1 / mu_i
                            + 2.0 * onep / twop)
        ld[ind, 8] = a2 * (-log_mu * onep * a1 - a1
                           + onep * (y_i / onep ** 2 - mu_i / twop ** 2)
                           + mu_i * log_mu / twop - mu_i / twop ** 2)
        ld[ind, 9] = a2 * (mu_i / (phii * twop) - onep * a1 / phii)
    ld[:, 9] = ld[:, 9] * phi
    ld[:, 8] = ld[:, 8] * dpth1

    # series part — added AFTER the transform: like the C code, it is
    # computed natively in (θ, ρ) (gam.fit3.r:3013-3020)
    if any_pos:
        # mgcv's ldTweedie dispatch (gam.fit3.r:2847,2942): when θ AND ρ are
        # constant across rows it sets `buffer=TRUE` and calls the scalar-p
        # C_tweedious, whose lgamma/digamma/trigamma(-jα) tables are shared
        # across rows (α is then a single value); only genuinely per-row θ/ρ
        # take the C_tweedious2 vector path. tw.null.fit's Newton evaluates a
        # CONSTANT (μ,θ,ρ) on every row (~1200×), so the shared-table path is
        # the difference between matching mgcv and a ~13× null-fit blow-up.
        # `np.ptp == 0` ⇔ mgcv's `length(unique(·))==1` (ULP-exact).
        buffer = (theta.size <= 1 or float(np.ptp(theta)) == 0.0) and \
                 (rho.size <= 1 or float(np.ptp(rho)) == 0.0)
        if buffer:
            la, jb, jv, jpb, m_wp1, m_comb, m_dwpp = _tweedie_log_a_vec(
                y_i, phii, float(p_i.flat[0]))
        else:
            la, jb, jv, jpb, m_wp1, m_comb, m_dwpp = _tweedie_log_a_vec_pv(
                y_i, phii, p_i)
        al = twop / onep                      # α
        one_m_al = 1.0 - al
        # Series derivatives in mgcv's well-conditioned working-parameter form
        # (tweedious, misc.c:498-503): the kernel returns the p-param accumulators
        # m_wp1=E[wp1], m_comb=E[wp1²+wp2], m_dwpp=E[wp1·j/(1−p)+wpp]; the θ-chain
        # (dpth1/dpth2) is reapplied here. Combining wp1²+wp2 per term (inside the
        # kernel) avoids the ~1e-11 cancellation the old moment split incurred.
        d1 = dpth1[ind]
        d2_ = dpth2[ind]
        # Form the WELL-CONDITIONED p-param 2nd derivatives first (the
        # m_comb−m_wp1² / m_dwpp−(jb/onep)·m_wp1 subtractions have no
        # cancellation, mgcv misc.c:500-501), THEN apply the θ-chain — doing
        # the chain inside the subtraction would re-introduce the cancellation.
        d2logS_dp2 = m_comb - m_wp1 ** 2              # ∂²log a/∂p²
        d2logS_dpdrho = m_dwpp - (jb / onep) * m_wp1  # ∂²log a/∂p∂ρ
        ld[ind, 0] += la
        ld[ind, 1] += -one_m_al * jb
        ld[ind, 2] += one_m_al ** 2 * jv
        ld[ind, 3] += d1 * m_wp1
        ld[ind, 4] += d1 ** 2 * d2logS_dp2 + d2_ * m_wp1
        ld[ind, 5] += d1 * d2logS_dpdrho
    return ld


def _tw_null_fit(y, a: float = 1.001, b: float = 1.999):
    """mgcv ``tw.null.fit`` (gamlss.r:2454-2490): stabilized,
    step-controlled Newton MLE of (μ, p, φ) for a plain Tweedie sample,
    iterating on the working scale (log μ, θ, ρ). Returns
    ``(mu, p, phi)`` — R's ``c(mu, p, sigma)``. The Hessian's log-μ
    chain and the negative-definite eigenvalue clamp are ported
    literally (the gradient stop test is exact, so the approximate
    chain only shapes the path)."""
    y = np.asarray(y, dtype=float)
    th = np.zeros(3)                     # log mu, theta, rho
    ones = np.ones_like(y)

    def _ld_sums(t):
        ld = _ld_tweedie_work(y, np.exp(t[0]) * ones, t[1] * ones,
                              t[2] * ones, a=a, b=b)
        # mgcv reduces ldTweedie with R's colSums, which accumulates in LDOUBLE
        # (80-bit on x86). Near the flat MLE the step-halving accept test
        # `Σl(θ₁) < Σl(θ)` resolves a ~1e-9 change inside a ~1e3 sum; a plain
        # float64 reduction's ~1e-10 noise stalls the Newton (gradient frozen
        # ~3e-5, well above the 1e-9·|l| break ⇒ all 50 iters × deep halving,
        # ~1200 evals vs mgcv's ~60). Accumulate in long double to match R's
        # colSums precision, then round to float64. (On ARM np.longdouble ≡
        # float64 — same as R's LDOUBLE there, so the parity is preserved.)
        return ld.sum(axis=0, dtype=np.longdouble).astype(np.float64)

    lds = _ld_sums(th)
    # The log-μ Hessian uses mgcv's approximate chain (no g·∂²μ term), so the
    # Newton converges only LINEARLY near the flat MLE — the μ-gradient drops
    # one ~0.4 ratio per step, oscillating in sign. mgcv reaches its 1e-9·|l|
    # gradient break before the step it accepts shrinks to FP noise; whether it
    # does is decided by sub-1e-9 likelihood comparisons (hence the LDOUBLE
    # colSums above). For datasets where the residual numpy-vs-R summation
    # difference freezes the gradient just shy of the break, detect the stall
    # (max|g| no longer falling) and stop — the (μ,p,φ) there already matches
    # mgcv to ~6 sig figs. Without it the loop burns all 50 iters at ~20
    # halvings each (~1000 evals vs mgcv's ~60-130).
    gmag_prev = float("inf")
    n_stall = 0
    for _ in range(50):
        g = lds[[6, 3, 1]].copy()
        if np.sum(np.abs(g) > 1e-9 * abs(lds[0])) == 0:
            break
        gmag = float(np.max(np.abs(g)))
        if gmag >= gmag_prev:
            n_stall += 1
            if n_stall >= 2:
                break
        else:
            n_stall = 0
        gmag_prev = gmag
        g[0] = g[0] * np.exp(th[0])      # work on log scale for mu
        H = np.zeros((3, 3))             # mu, th, rh
        H[0, 0] = lds[7]
        H[1, 1] = lds[4]
        H[2, 2] = lds[2]
        H[0, 1] = H[1, 0] = lds[8]
        H[0, 2] = H[2, 0] = lds[9]
        H[1, 2] = H[2, 1] = lds[5]
        H[:, 0] = H[:, 0] * np.exp(th[0])
        H[0, 1:] = H[0, 1:] * np.exp(th[0])
        ev, V = np.linalg.eigh(0.5 * (H + H.T))
        tol = float(np.max(np.abs(ev))) * 1e-7
        ev[ev > -tol] = -tol
        step = V @ ((V.T @ g) / ev)
        ms = float(np.max(np.abs(step)))
        if ms > 3.0:
            step = step * 3.0 / ms
        # Bounded step-halving (mgcv's tw.null.fit uses an unbounded while, but
        # relies on the gradient break firing first; cap it so the flat-MLE
        # stall can't spin — 40 halvings ⇒ a 1e-12 step, deep past any useful
        # move).
        accepted = False
        for _h in range(40):
            th1 = th - step
            lds1 = _ld_sums(th1)
            if lds1[0] < lds[0]:
                step = step / 2.0
            else:
                th = th1
                lds = lds1
                accepted = True
                break
        if not accepted:
            break
    t2 = th[1]
    if t2 > 0:
        p = (b + a * np.exp(-t2)) / (1.0 + np.exp(-t2))
    else:
        p = (b * np.exp(t2) + a) / (np.exp(t2) + 1.0)
    return float(np.exp(th[0])), float(p), float(np.exp(th[2]))


def _shash_log1pexp(x):
    """shash's ``.log1pexp`` (gamlss.r:3431-3441): log(1 + e^x) with
    R's binned stabilization. The x = −Inf corner (z = 0 exactly)
    falls in the first bin here and returns 0 — R's ``.bincode``
    NA-drops that boundary and would propagate the −Inf."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    m1 = x <= -37.0
    m2 = (x > -37.0) & (x <= 18.0)
    m3 = (x > 18.0) & (x <= 33.3)
    out[m1] = np.exp(x[m1])
    out[m2] = np.log1p(np.exp(x[m2]))
    out[m3] = x[m3] + np.exp(-x[m3])
    return out


def _sqrt_x2pm(x, m):
    """shash's ``.sqrtX2pm`` (gamlss.r:3444-3451): sqrt(x² + m),
    passing |x| through unchanged once |x| ≥ 1e8."""
    x = np.abs(np.asarray(x, dtype=float))
    out = x.copy()
    kk = x < 1e8
    out[kk] = np.sqrt(x[kk] ** 2 + m)
    return out


def _ax2m1_div_x2m2_sq(x, m1, m2, a=1.0):
    """shash's ``.ax2m1DivX2m2SQ`` (gamlss.r:3454-3466):
    (a·x² + m1)/(x² + m2)² computed stably for large |x|."""
    if a < 0:
        raise ValueError("'a' has to be positive")
    x = np.abs(np.asarray(x, dtype=float))
    kk = (a * x ** 2 + m1) < 0.0
    out = np.zeros_like(x)
    if np.any(kk):
        out[kk] = (a * x[kk] ** 2 + m1) / (x[kk] ** 2 + m2) ** 2
    nk = ~kk
    if np.any(nk):
        out[nk] = ((_sqrt_x2pm(np.sqrt(a) * x[nk], m1)
                    / _sqrt_x2pm(x[nk], m2)) / _sqrt_x2pm(x[nk], m2)) ** 2
    return out


def _sech(x):
    return 1.0 / np.cosh(x)


def _shash_derivs(y, mu, tau, eps, phi, phi_pen, deriv):
    """shash log-density and packed parameter-space derivatives —
    mgcv's shash ``ll`` body (gamlss.r:3487-3950) up to the etamu
    hand-off. Returns ``(l0, L1, L2, L3, L4)`` with the (μ, τ, ε, φ)
    packing; L3/L4 are None below the requesting deriv level. The
    third/fourth-derivative blocks are mechanical transcriptions of
    mgcv's auto-generated maxima code (sequencing and groupings kept
    line-for-line; only `del`→`delta` renamed).
    """
    y = np.asarray(y, dtype=float)
    sig = np.exp(tau)
    delta = np.exp(phi)
    z = (y - mu) / (sig * delta)
    dTasMe = delta * np.arcsinh(z) - eps
    g = -dTasMe
    CC = np.cosh(dTasMe)
    SS = np.sinh(dTasMe)
    with np.errstate(divide="ignore"):
        log_abs_z2 = 2.0 * np.log(np.abs(z))
    l0 = (-tau - 0.5 * np.log(2.0 * np.pi) + np.log(CC)
          - 0.5 * _shash_log1pexp(log_abs_z2) - 0.5 * SS ** 2
          - phi_pen * phi ** 2)
    L1 = L2 = L3 = L4 = None
    if deriv >= 1:
        zsd = z * sig * delta
        sSp1 = _sqrt_x2pm(z, 1.0)               # sqrt(z² + 1)
        asinhZ = np.arcsinh(z)

        # first derivatives (gamlss.r:3513-3519)
        De = np.tanh(g) - 0.5 * np.sinh(2.0 * g)
        Dm = 1.0 / (delta * sig * sSp1) * (delta * De + z / sSp1)
        Dt = zsd * Dm - 1.0
        Dp = Dt + 1.0 - delta * asinhZ * De - 2.0 * phi_pen * phi
        L1 = np.column_stack([Dm, Dt, De, Dp])

        # second derivatives, packed mm,mt,me,mp,tt,te,tp,ee,ep,pp
        # (gamlss.r:3522-3535)
        Dme = (_sech(g) ** 2 - np.cosh(2.0 * g)) / (sig * sSp1)
        Dte = zsd * Dme
        Dmm = (Dme / (sig * sSp1) + z * De / (sig ** 2 * delta * sSp1 ** 3)
               + _ax2m1_div_x2m2_sq(z, -1.0, 1.0) / (delta * sig * delta
                                                     * sig))
        Dmt = zsd * Dmm - Dm
        Dee = -2.0 * np.cosh(g) ** 2 + _sech(g) ** 2 + 1.0
        Dtt = zsd * Dmt
        Dep = Dte - delta * asinhZ * Dee
        Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
        Dtp = zsd * Dmp
        Dpp = (Dtp - delta * asinhZ * Dep
               + delta * (z / sSp1 - asinhZ) * De - 2.0 * phi_pen)
        L2 = np.column_stack([Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee,
                              Dep, Dpp])
    if deriv > 1:
        # third derivatives (gamlss.r:3545-3567)
        Deee = -2 * (np.sinh(2 * g) + _sech(g) ** 2 * np.tanh(g))
        Dmee = Deee / (sig * sSp1)
        Dmme = Dmee / (sig * sSp1) + z * Dee / (sig * sig * delta * sSp1 ** 3)
        Dmmm = (
            2 * z * Dme / (sig * sig * delta * sSp1 ** 3) + Dmme /
            (sig * sSp1) + _ax2m1_div_x2m2_sq(z, -1, 1, 2) * De /
            (sig ** 3 * delta ** 2 * sSp1) + 2 * (z / sSp1) *
            _ax2m1_div_x2m2_sq(z, -3, 1) / ((sig * delta) ** 3 * sSp1)
        )
        Dmmt = zsd * Dmmm - 2 * Dmm
        Dtee = zsd * Dmee
        Dmte = zsd * Dmme - Dme
        Dtte = zsd * Dmte
        Dmtt = zsd * Dmmt - Dmt
        Dttt = zsd * Dmtt
        Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
        Dtep = zsd * Dmep
        Deep = Dtee - delta * asinhZ * Deee
        Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
        Dmmp = (
            Dmmt + 2 * Dme / (sig * sSp1) + z * De /
            (delta * sig * sig * sSp1 ** 3) - delta * asinhZ * Dmme
        )
        Dmtp = zsd * Dmmp - Dmp
        Dttp = zsd * Dmtp
        Dmpp = (
            Dmtp + Dep / (sig * sSp1) + z ** 2 * De / (sig * sSp1 ** 3) -
            delta * asinhZ * Dmep + delta * Dme * (z / sSp1 - asinhZ)
        )
        Dtpp = zsd * Dmpp
        Dppp = (
            Dtpp - delta * asinhZ * Depp + delta * (z / sSp1 - asinhZ) *
            (2 * Dep + De) + delta * (z / sSp1) ** 3 * De
        )

        L3 = np.column_stack([Dmmm, Dmmt, Dmme, Dmmp, Dmtt, Dmte, Dmtp,
                              Dmee, Dmep, Dmpp, Dttt, Dtte, Dttp, Dtee,
                              Dtep, Dtpp, Deee, Deep, Depp, Dppp])
    if deriv > 3:
        # fourth derivatives — mgcv's auto-generated block
        # (gamlss.r:3586-3941); 35 columns in the packed order
        # mmmm..pppp listed at gamlss.r:3579-3582
        m = mu
        t = tau
        p = phi
        e = eps
        exp1 = np.e
        aaa1 = -t
        aaa2 = y - m
        aaa3 = exp1 ** p * np.asinh(exp1 ** (aaa1 - p) * aaa2) - e
        abb8 = np.cosh(aaa3)
        abb9 = np.sinh(aaa3)
        abb1 = exp1 ** ((-2 * t) - 2 * p)
        abb3 = aaa2 ** 2
        abb4 = 1 / exp1 ** t
        abb5 = -t - p
        abb7 = exp1 ** (2 * abb5) * abb3 + 1
        abb6 = 1 / np.sqrt(abb7)
        aee5 = aaa3 + e
        aff04 = abb1 * abb3 + 1
        aff05 = abb4 ** 2
        aff08 = 2 * abb5
        aff10 = 1 / abb7
        aff13 = abb8 ** 2
        aff14 = exp1 ** (aaa1 + aff08)
        aff15 = abb6 ** 3
        aff17 = abb9 ** 2
        agg15 = 1 / abb6
        agg17 = 1 / abb8
        aii11 = aaa3 + e
        aii12 = aii11 - abb4 * aaa2 * abb6
        aii17 = abb6 ** 3
        ajj15 = aaa2 ** 3
        ann05 = exp1 ** p
        ann06 = np.asinh(exp1 ** abb5 * aaa2)
        aoo09 = -aaa2 / (exp1 ** t * agg15)
        app02 = -2 * t
        app04 = exp1 ** (app02 - 2 * p) * abb3 + 1
        app08 = exp1 ** (app02 + aff08)
        app10 = 1 / abb7 ** 2
        app14 = exp1 ** (aaa1 + 4 * abb5)
        app16 = 1 / agg15 ** 5
        app21 = 1 / exp1 ** (3 * t)
        aqq03 = exp1 ** (app02 - 2 * p)
        aqq05 = aqq03 * abb3 + 1
        aqq27 = 1 / aff13
        arr06 = exp1 ** aff08 * aaa2 ** 2 + 1
        arr07 = 1 / np.sqrt(arr06) ** 3
        arr12 = 1 / arr06
        ass16 = aii11 - aaa2 / (exp1 ** t * agg15)
        ass23 = 1 / abb8
        ass28 = 1 / aff13
        att19 = aaa2 ** 4
        avv19 = aii11 - abb4 * aaa2 * abb6
        ayy14 = -abb4 * aaa2 * abb6
        ayy16 = aii11 + ayy14
        ayy17 = aii11 + ayy14 - aff14 * ajj15 * aii17
        ayy24 = ayy16 ** 2
        azz19 = aaa2 ** 5
        bdd07 = np.sqrt(exp1 ** aff08 * aaa2 ** 2 + 1)
        bdd08 = 1 / bdd07 ** 3
        bdd14 = 1 / bdd07
        bdd15 = aii11 - abb4 * aaa2 * bdd14
        bgg4 = (
            aee5 - aaa2 /
            (exp1 ** t * np.sqrt(exp1 ** (2 * abb5) * aaa2 ** 2 + 1))
        )
        bhh13 = -abb4 * aaa2 * bdd14
        bhh14 = ann05 * ann06
        bii11 = aii11 + aoo09
        bii15 = aii11 + aoo09 - aff14 * ajj15 * aii17
        bjj07 = 4 * abb5
        bjj08 = exp1 ** (app02 + bjj07)
        bjj11 = 1 / abb7 ** 3
        bjj14 = 1 / exp1 ** (4 * t)
        bjj18 = exp1 ** (aaa1 + 6 * abb5)
        bjj21 = 1 / agg15 ** 7
        bjj24 = exp1 ** (aff08 - 3 * t)
        bjj26 = exp1 ** (aaa1 + bjj07)
        j2 = (
            (-(6 * bjj14 * app10 * abb9 ** 4) / abb8 ** 4) -
            (12 * bjj24 * aaa2 * app16 * abb9 ** 3) / abb8 ** 3 + 8 * bjj14 *
            app10 * aqq27 * aff17 + 4 * app08 * app10 * aqq27 * aff17 - 15 *
            bjj08 * abb3 * bjj11 * aqq27 * aff17 - 4 * bjj14 * app10 * aff17 +
            4 * app08 * app10 * aff17 - 15 * bjj08 * abb3 * bjj11 * aff17 - 9
            * bjj26 * aaa2 * app16 * abb8 * abb9 + 24 * bjj24 * aaa2 * app16 *
            abb8 * abb9 + 15 * bjj18 * ajj15 * bjj21 * abb8 * abb9 + 9 * bjj26
            * aaa2 * app16 * agg17 * abb9 + 12 * bjj24 * aaa2 * app16 * agg17
            * abb9 - 15 * bjj18 * ajj15 * bjj21 * agg17 * abb9 - 4 * bjj14 *
            app10 * aff13 + 4 * app08 * app10 * aff13 - 15 * bjj08 * abb3 *
            bjj11 * aff13 - 2 * bjj14 * app10 - 4 * app08 * app10 + 15 * bjj08
            * abb3 * bjj11 + (6 * exp1 ** ((-4 * t) - 4 * p)) / app04 ** 2 -
            (48 * exp1 ** ((-6 * t) - 6 * p) * abb3) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 4) / app04 ** 4
        )
        bkk33 = 1 / abb8 ** 3
        bkk34 = abb9 ** 3
        k2 = (
            (-(6 * bjj14 * aaa2 * app10 * abb9 ** 4) / abb8 ** 4) + 6 * app21
            * aff15 * bkk33 * bkk34 - 12 * bjj24 * abb3 * app16 * bkk33 *
            bkk34 + 8 * bjj14 * aaa2 * app10 * aqq27 * aff17 + 13 * app08 *
            aaa2 * app10 * aqq27 * aff17 - 15 * bjj08 * ajj15 * bjj11 * aqq27
            * aff17 - 4 * bjj14 * aaa2 * app10 * aff17 + 13 * app08 * aaa2 *
            app10 * aff17 - 15 * bjj08 * ajj15 * bjj11 * aff17 - 12 * app21 *
            aff15 * abb8 * abb9 + 3 * aff14 * aff15 * abb8 * abb9 - 18 * bjj26
            * abb3 * app16 * abb8 * abb9 + 24 * bjj24 * abb3 * app16 * abb8 *
            abb9 + 15 * bjj18 * att19 * bjj21 * abb8 * abb9 - 6 * app21 *
            aff15 * agg17 * abb9 - 3 * aff14 * aff15 * agg17 * abb9 + 18 *
            bjj26 * abb3 * app16 * agg17 * abb9 + 12 * bjj24 * abb3 * app16 *
            agg17 * abb9 - 15 * bjj18 * att19 * bjj21 * agg17 * abb9 - 4 *
            bjj14 * aaa2 * app10 * aff13 + 13 * app08 * aaa2 * app10 * aff13 -
            15 * bjj08 * ajj15 * bjj11 * aff13 - 2 * bjj14 * aaa2 * app10 - 13
            * app08 * aaa2 * app10 + 15 * bjj08 * ajj15 * bjj11 +
            (24 * exp1 ** ((-4 * t) - 4 * p) * aaa2) / app04 ** 2 -
            (72 * exp1 ** ((-6 * t) - 6 * p) * ajj15) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 5) / app04 ** 4
        )
        bll16 = exp1 ** (aff08 - 2 * t)
        l2 = (
            (-(6 * app21 * aff15 * abb9 ** 4) / abb8 ** 4) -
            (6 * bll16 * aaa2 * app10 * abb9 ** 3) / abb8 ** 3 + 8 * app21 *
            aff15 * aqq27 * aff17 + aff14 * aff15 * aqq27 * aff17 - 3 * app14
            * abb3 * app16 * aqq27 * aff17 - 4 * app21 * aff15 * aff17 + aff14
            * aff15 * aff17 - 3 * app14 * abb3 * app16 * aff17 + 12 * bll16 *
            aaa2 * app10 * abb8 * abb9 + (6 * bll16 * aaa2 * app10 * abb9) /
            abb8 - 4 * app21 * aff15 * aff13 + aff14 * aff15 * aff13 - 3 *
            app14 * abb3 * app16 * aff13 - 2 * app21 * aff15 - aff14 * aff15 +
            3 * app14 * abb3 * app16
        )
        bmm34 = 1 / abb8 ** 3
        bmm35 = abb9 ** 3
        m2 = (
            (6 * app21 * aff15 * ass16 * abb9 ** 4) / abb8 ** 4 + 6 * app08 *
            aaa2 * app10 * ass16 * bmm34 * bmm35 - 6 * bjj24 * abb3 * app16 *
            bmm34 * bmm35 - 8 * app21 * aff15 * ass16 * ass28 * aff17 - aff14
            * aff15 * ass16 * ass28 * aff17 + 3 * bjj26 * abb3 * app16 * ass16
            * ass28 * aff17 + 6 * app08 * aaa2 * app10 * ass28 * aff17 - 12 *
            bjj08 * ajj15 * bjj11 * ass28 * aff17 + 4 * app21 * aff15 * ass16
            * aff17 - aff14 * aff15 * ass16 * aff17 + 3 * bjj26 * abb3 * app16
            * ass16 * aff17 + 6 * app08 * aaa2 * app10 * aff17 - 12 * bjj08 *
            ajj15 * bjj11 * aff17 - 12 * app08 * aaa2 * app10 * ass16 * abb8 *
            abb9 + 2 * aff14 * aff15 * abb8 * abb9 - 15 * bjj26 * abb3 * app16
            * abb8 * abb9 + 12 * bjj24 * abb3 * app16 * abb8 * abb9 + 15 *
            bjj18 * att19 * bjj21 * abb8 * abb9 - 6 * app08 * aaa2 * app10 *
            ass16 * ass23 * abb9 - 2 * aff14 * aff15 * ass23 * abb9 + 15 *
            bjj26 * abb3 * app16 * ass23 * abb9 + 6 * bjj24 * abb3 * app16 *
            ass23 * abb9 - 15 * bjj18 * att19 * bjj21 * ass23 * abb9 + 4 *
            app21 * aff15 * ass16 * aff13 - aff14 * aff15 * ass16 * aff13 + 3
            * bjj26 * abb3 * app16 * ass16 * aff13 + 6 * app08 * aaa2 * app10
            * aff13 - 12 * bjj08 * ajj15 * bjj11 * aff13 + 2 * app21 * aff15 *
            ass16 + aff14 * aff15 * ass16 - 3 * bjj26 * abb3 * app16 * ass16 -
            6 * app08 * aaa2 * app10 + 12 * bjj08 * ajj15 * bjj11 +
            (24 * exp1 ** ((-4 * t) - 4 * p) * aaa2) / app04 ** 2 -
            (72 * exp1 ** ((-6 * t) - 6 * p) * ajj15) / app04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 5) / app04 ** 4
        )
        n2 = (
            (-(6 * bjj14 * abb3 * app10 * abb9 ** 4) / abb8 ** 4) + 10 * app21
            * aaa2 * aff15 * bkk33 * bkk34 - 12 * bjj24 * ajj15 * app16 *
            bkk33 * bkk34 - 4 * aff05 * aff10 * aqq27 * aff17 + 8 * bjj14 *
            abb3 * app10 * aqq27 * aff17 + 19 * app08 * abb3 * app10 * aqq27 *
            aff17 - 15 * bjj08 * att19 * bjj11 * aqq27 * aff17 - 4 * aff05 *
            aff10 * aff17 - 4 * bjj14 * abb3 * app10 * aff17 + 19 * app08 *
            abb3 * app10 * aff17 - 15 * bjj08 * att19 * bjj11 * aff17 - 20 *
            app21 * aaa2 * aff15 * abb8 * abb9 + 9 * aff14 * aaa2 * aff15 *
            abb8 * abb9 - 24 * bjj26 * ajj15 * app16 * abb8 * abb9 + 24 *
            bjj24 * ajj15 * app16 * abb8 * abb9 + 15 * bjj18 * azz19 * bjj21 *
            abb8 * abb9 - 10 * app21 * aaa2 * aff15 * agg17 * abb9 - 9 * aff14
            * aaa2 * aff15 * agg17 * abb9 + 24 * bjj26 * ajj15 * app16 * agg17
            * abb9 + 12 * bjj24 * ajj15 * app16 * agg17 * abb9 - 15 * bjj18 *
            azz19 * bjj21 * agg17 * abb9 - 4 * aff05 * aff10 * aff13 - 4 *
            bjj14 * abb3 * app10 * aff13 + 19 * app08 * abb3 * app10 * aff13 -
            15 * bjj08 * att19 * bjj11 * aff13 + 4 * aff05 * aff10 - 2 * bjj14
            * abb3 * app10 - 19 * app08 * abb3 * app10 + 15 * bjj08 * att19 *
            bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        o2 = (
            (-(6 * app21 * aaa2 * aff15 * abb9 ** 4) / abb8 ** 4) + 4 * aff05
            * aff10 * bkk33 * bkk34 - 6 * bll16 * abb3 * app10 * bkk33 * bkk34
            + 8 * app21 * aaa2 * aff15 * aqq27 * aff17 + 3 * aff14 * aaa2 *
            aff15 * aqq27 * aff17 - 3 * app14 * ajj15 * app16 * aqq27 * aff17
            - 4 * app21 * aaa2 * aff15 * aff17 + 3 * aff14 * aaa2 * aff15 *
            aff17 - 3 * app14 * ajj15 * app16 * aff17 - 8 * aff05 * aff10 *
            abb8 * abb9 + 12 * bll16 * abb3 * app10 * abb8 * abb9 - 4 * aff05
            * aff10 * agg17 * abb9 + 6 * bll16 * abb3 * app10 * agg17 * abb9 -
            4 * app21 * aaa2 * aff15 * aff13 + 3 * aff14 * aaa2 * aff15 *
            aff13 - 3 * app14 * ajj15 * app16 * aff13 - 2 * app21 * aaa2 *
            aff15 - 3 * aff14 * aaa2 * aff15 + 3 * app14 * ajj15 * app16
        )
        p2 = (
            (6 * app21 * aaa2 * aff15 * ass16 * abb9 ** 4) / abb8 ** 4 - 4 *
            aff05 * aff10 * ass16 * bmm34 * bmm35 + 6 * app08 * abb3 * app10 *
            ass16 * bmm34 * bmm35 - 6 * bjj24 * ajj15 * app16 * bmm34 * bmm35
            - 8 * app21 * aaa2 * aff15 * ass16 * ass28 * aff17 - 3 * aff14 *
            aaa2 * aff15 * ass16 * ass28 * aff17 + 3 * bjj26 * ajj15 * app16 *
            ass16 * ass28 * aff17 + 10 * app08 * abb3 * app10 * ass28 * aff17
            - 12 * bjj08 * att19 * bjj11 * ass28 * aff17 + 4 * app21 * aaa2 *
            aff15 * ass16 * aff17 - 3 * aff14 * aaa2 * aff15 * ass16 * aff17 +
            3 * bjj26 * ajj15 * app16 * ass16 * aff17 + 10 * app08 * abb3 *
            app10 * aff17 - 12 * bjj08 * att19 * bjj11 * aff17 + 8 * aff05 *
            aff10 * ass16 * abb8 * abb9 - 12 * app08 * abb3 * app10 * ass16 *
            abb8 * abb9 + 6 * aff14 * aaa2 * aff15 * abb8 * abb9 - 21 * bjj26
            * ajj15 * app16 * abb8 * abb9 + 12 * bjj24 * ajj15 * app16 * abb8
            * abb9 + 15 * bjj18 * azz19 * bjj21 * abb8 * abb9 + 4 * aff05 *
            aff10 * ass16 * ass23 * abb9 - 6 * app08 * abb3 * app10 * ass16 *
            ass23 * abb9 - 6 * aff14 * aaa2 * aff15 * ass23 * abb9 + 21 *
            bjj26 * ajj15 * app16 * ass23 * abb9 + 6 * bjj24 * ajj15 * app16 *
            ass23 * abb9 - 15 * bjj18 * azz19 * bjj21 * ass23 * abb9 + 4 *
            app21 * aaa2 * aff15 * ass16 * aff13 - 3 * aff14 * aaa2 * aff15 *
            ass16 * aff13 + 3 * bjj26 * ajj15 * app16 * ass16 * aff13 + 10 *
            app08 * abb3 * app10 * aff13 - 12 * bjj08 * att19 * bjj11 * aff13
            + 2 * app21 * aaa2 * aff15 * ass16 + 3 * aff14 * aaa2 * aff15 *
            ass16 - 3 * bjj26 * ajj15 * app16 * ass16 - 10 * app08 * abb3 *
            app10 + 12 * bjj08 * att19 * bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        q2 = (
            (-(6 * aff05 * arr12 * abb9 ** 4) / abb8 ** 4) -
            (2 * aff14 * aaa2 * arr07 * abb9 ** 3) / abb8 ** 3 +
            (8 * aff05 * arr12 * aff17) / aff13 - 4 * aff05 * arr12 * aff17 +
            4 * aff14 * aaa2 * arr07 * abb8 * abb9 +
            (2 * aff14 * aaa2 * arr07 * abb9) / abb8 - 4 * aff05 * arr12 *
            aff13 - 2 * aff05 * arr12
        )
        r2 = (
            (6 * aff05 * aff10 * ass16 * abb9 ** 4) / abb8 ** 4 + 2 * aff14 *
            aaa2 * aff15 * ass16 * bmm34 * bmm35 - 4 * bll16 * abb3 * app10 *
            bmm34 * bmm35 - 8 * aff05 * aff10 * ass16 * ass28 * aff17 + 2 *
            aff14 * aaa2 * aff15 * ass28 * aff17 - 3 * app14 * ajj15 * app16 *
            ass28 * aff17 + 4 * aff05 * aff10 * ass16 * aff17 + 2 * aff14 *
            aaa2 * aff15 * aff17 - 3 * app14 * ajj15 * app16 * aff17 - 4 *
            aff14 * aaa2 * aff15 * ass16 * abb8 * abb9 + 8 * bll16 * abb3 *
            app10 * abb8 * abb9 - 2 * aff14 * aaa2 * aff15 * ass16 * ass23 *
            abb9 + 4 * bll16 * abb3 * app10 * ass23 * abb9 + 4 * aff05 * aff10
            * ass16 * aff13 + 2 * aff14 * aaa2 * aff15 * aff13 - 3 * app14 *
            ajj15 * app16 * aff13 + 2 * aff05 * aff10 * ass16 - 2 * aff14 *
            aaa2 * aff15 + 3 * app14 * ajj15 * app16
        )
        bss21 = 2 * aff14 * abb3 * aff15 - 3 * bjj26 * att19 * app16
        bss23 = -abb4 * aaa2 * abb6
        bss25 = aii11 + bss23
        bss26 = aii11 + bss23 - aff14 * ajj15 * aff15
        bss29 = bss25 ** 2
        bss33 = (
            (-4 * aff14 * aaa2 * aff15) + 18 * bjj26 * ajj15 * app16 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 5) / agg15 ** 7
        )
        s2 = (
            (-(6 * aff05 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) - 2 * aff14
            * aaa2 * aff15 * bss29 * bmm34 * bmm35 + 2 * aff05 * aff10 * bss26
            * bmm34 * bmm35 + 8 * app08 * abb3 * app10 * bss25 * bmm34 * bmm35
            + 8 * aff05 * aff10 * bss29 * ass28 * aff17 + aff14 * aaa2 * aff15
            * bss26 * ass28 * aff17 - 4 * aff14 * aaa2 * aff15 * bss25 * ass28
            * aff17 + 6 * bjj26 * ajj15 * app16 * bss25 * ass28 * aff17 + 2 *
            abb4 * abb6 * bss21 * ass28 * aff17 - 2 * bjj08 * att19 * bjj11 *
            ass28 * aff17 - 4 * aff05 * aff10 * bss29 * aff17 + aff14 * aaa2 *
            aff15 * bss26 * aff17 - 4 * aff14 * aaa2 * aff15 * bss25 * aff17 +
            6 * bjj26 * ajj15 * app16 * bss25 * aff17 + 2 * abb4 * abb6 *
            bss21 * aff17 - 2 * bjj08 * att19 * bjj11 * aff17 + 4 * aff14 *
            aaa2 * aff15 * bss29 * abb8 * abb9 - 4 * aff05 * aff10 * bss26 *
            abb8 * abb9 - 16 * app08 * abb3 * app10 * bss25 * abb8 * abb9 -
            bss33 * abb8 * abb9 + 2 * aff14 * aaa2 * aff15 * bss29 * ass23 *
            abb9 - 2 * aff05 * aff10 * bss26 * ass23 * abb9 - 8 * app08 * abb3
            * app10 * bss25 * ass23 * abb9 + bss33 * ass23 * abb9 - 4 * aff05
            * aff10 * bss29 * aff13 + aff14 * aaa2 * aff15 * bss26 * aff13 - 4
            * aff14 * aaa2 * aff15 * bss25 * aff13 + 6 * bjj26 * ajj15 * app16
            * bss25 * aff13 + 2 * abb4 * abb6 * bss21 * aff13 - 2 * bjj08 *
            att19 * bjj11 * aff13 - 2 * aff05 * aff10 * bss29 - aff14 * aaa2 *
            aff15 * bss26 + 4 * aff14 * aaa2 * aff15 * bss25 - 6 * bjj26 *
            ajj15 * app16 * bss25 - 2 * abb4 * abb6 * bss21 + 2 * bjj08 *
            att19 * bjj11 - (4 * aqq03) / aqq05 +
            (44 * exp1 ** ((-4 * t) - 4 * p) * abb3) / aqq05 ** 2 -
            (88 * exp1 ** ((-6 * t) - 6 * p) * att19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 6) / aqq05 ** 4
        )
        btt24 = aaa2 ** 6
        t2 = (
            (-(6 * bjj14 * ajj15 * app10 * abb9 ** 4) / abb8 ** 4) + 12 *
            app21 * abb3 * aff15 * bkk33 * bkk34 - 12 * bjj24 * att19 * app16
            * bkk33 * bkk34 - 7 * aff05 * aaa2 * aff10 * aqq27 * aff17 + 8 *
            bjj14 * ajj15 * app10 * aqq27 * aff17 + 22 * app08 * ajj15 * app10
            * aqq27 * aff17 - 15 * bjj08 * azz19 * bjj11 * aqq27 * aff17 - 7 *
            aff05 * aaa2 * aff10 * aff17 - 4 * bjj14 * ajj15 * app10 * aff17 +
            22 * app08 * ajj15 * app10 * aff17 - 15 * bjj08 * azz19 * bjj11 *
            aff17 - abb4 * abb6 * abb8 * abb9 - 24 * app21 * abb3 * aff15 *
            abb8 * abb9 + 13 * aff14 * abb3 * aff15 * abb8 * abb9 - 27 * bjj26
            * att19 * app16 * abb8 * abb9 + 24 * bjj24 * att19 * app16 * abb8
            * abb9 + 15 * bjj18 * btt24 * bjj21 * abb8 * abb9 + abb4 * abb6 *
            agg17 * abb9 - 12 * app21 * abb3 * aff15 * agg17 * abb9 - 13 *
            aff14 * abb3 * aff15 * agg17 * abb9 + 27 * bjj26 * att19 * app16 *
            agg17 * abb9 + 12 * bjj24 * att19 * app16 * agg17 * abb9 - 15 *
            bjj18 * btt24 * bjj21 * agg17 * abb9 - 7 * aff05 * aaa2 * aff10 *
            aff13 - 4 * bjj14 * ajj15 * app10 * aff13 + 22 * app08 * ajj15 *
            app10 * aff13 - 15 * bjj08 * azz19 * bjj11 * aff13 + 7 * aff05 *
            aaa2 * aff10 - 2 * bjj14 * ajj15 * app10 - 22 * app08 * ajj15 *
            app10 + 15 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        u2 = (
            (-(6 * app21 * abb3 * aff15 * abb9 ** 4) / abb8 ** 4) + 6 * aff05
            * aaa2 * aff10 * bkk33 * bkk34 - 6 * bll16 * ajj15 * app10 * bkk33
            * bkk34 - abb4 * abb6 * aqq27 * aff17 + 8 * app21 * abb3 * aff15 *
            aqq27 * aff17 + 4 * aff14 * abb3 * aff15 * aqq27 * aff17 - 3 *
            app14 * att19 * app16 * aqq27 * aff17 - abb4 * abb6 * aff17 - 4 *
            app21 * abb3 * aff15 * aff17 + 4 * aff14 * abb3 * aff15 * aff17 -
            3 * app14 * att19 * app16 * aff17 - 12 * aff05 * aaa2 * aff10 *
            abb8 * abb9 + 12 * bll16 * ajj15 * app10 * abb8 * abb9 - 6 * aff05
            * aaa2 * aff10 * agg17 * abb9 + 6 * bll16 * ajj15 * app10 * agg17
            * abb9 - abb4 * abb6 * aff13 - 4 * app21 * abb3 * aff15 * aff13 +
            4 * aff14 * abb3 * aff15 * aff13 - 3 * app14 * att19 * app16 *
            aff13 + abb4 * abb6 - 2 * app21 * abb3 * aff15 - 4 * aff14 * abb3
            * aff15 + 3 * app14 * att19 * app16
        )
        v2 = (
            (6 * app21 * abb3 * aff15 * avv19 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff05 * aaa2 * aff10 * avv19 * bmm34 * bmm35 + 6 * app08 * ajj15 *
            app10 * avv19 * bmm34 * bmm35 - 6 * bjj24 * att19 * app16 * bmm34
            * bmm35 + abb4 * abb6 * avv19 * ass28 * aff17 - 8 * app21 * abb3 *
            aff15 * avv19 * ass28 * aff17 - 4 * aff14 * abb3 * aff15 * avv19 *
            ass28 * aff17 + 3 * bjj26 * att19 * app16 * avv19 * ass28 * aff17
            + 12 * app08 * ajj15 * app10 * ass28 * aff17 - 12 * bjj08 * azz19
            * bjj11 * ass28 * aff17 + abb4 * abb6 * avv19 * aff17 + 4 * app21
            * abb3 * aff15 * avv19 * aff17 - 4 * aff14 * abb3 * aff15 * avv19
            * aff17 + 3 * bjj26 * att19 * app16 * avv19 * aff17 + 12 * app08 *
            ajj15 * app10 * aff17 - 12 * bjj08 * azz19 * bjj11 * aff17 + 12 *
            aff05 * aaa2 * aff10 * avv19 * abb8 * abb9 - 12 * app08 * ajj15 *
            app10 * avv19 * abb8 * abb9 + 9 * aff14 * abb3 * aff15 * abb8 *
            abb9 - 24 * bjj26 * att19 * app16 * abb8 * abb9 + 12 * bjj24 *
            att19 * app16 * abb8 * abb9 + 15 * bjj18 * btt24 * bjj21 * abb8 *
            abb9 + 6 * aff05 * aaa2 * aff10 * avv19 * ass23 * abb9 - 6 * app08
            * ajj15 * app10 * avv19 * ass23 * abb9 - 9 * aff14 * abb3 * aff15
            * ass23 * abb9 + 24 * bjj26 * att19 * app16 * ass23 * abb9 + 6 *
            bjj24 * att19 * app16 * ass23 * abb9 - 15 * bjj18 * btt24 * bjj21
            * ass23 * abb9 + abb4 * abb6 * avv19 * aff13 + 4 * app21 * abb3 *
            aff15 * avv19 * aff13 - 4 * aff14 * abb3 * aff15 * avv19 * aff13 +
            3 * bjj26 * att19 * app16 * avv19 * aff13 + 12 * app08 * ajj15 *
            app10 * aff13 - 12 * bjj08 * azz19 * bjj11 * aff13 - abb4 * abb6 *
            avv19 + 2 * app21 * abb3 * aff15 * avv19 + 4 * aff14 * abb3 *
            aff15 * avv19 - 3 * bjj26 * att19 * app16 * avv19 - 12 * app08 *
            ajj15 * app10 + 12 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) /
            aqq05 + (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        w2 = (
            (-(6 * aff05 * aaa2 * aff10 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            abb6 * bkk33 * bkk34 - 2 * aff14 * abb3 * aff15 * bkk33 * bkk34 +
            (8 * aff05 * aaa2 * aff10 * aff17) / aff13 - 4 * aff05 * aaa2 *
            aff10 * aff17 - 4 * abb4 * abb6 * abb8 * abb9 + 4 * aff14 * abb3 *
            aff15 * abb8 * abb9 - 2 * abb4 * abb6 * agg17 * abb9 + 2 * aff14 *
            abb3 * aff15 * agg17 * abb9 - 4 * aff05 * aaa2 * aff10 * aff13 - 2
            * aff05 * aaa2 * aff10
        )
        x2 = (
            (6 * aff05 * aaa2 * aff10 * avv19 * abb9 ** 4) / abb8 ** 4 - 2 *
            abb4 * abb6 * avv19 * bmm34 * bmm35 + 2 * aff14 * abb3 * aff15 *
            avv19 * bmm34 * bmm35 - 4 * bll16 * ajj15 * app10 * bmm34 * bmm35
            - 8 * aff05 * aaa2 * aff10 * avv19 * ass28 * aff17 + 3 * aff14 *
            abb3 * aff15 * ass28 * aff17 - 3 * app14 * att19 * app16 * ass28 *
            aff17 + 4 * aff05 * aaa2 * aff10 * avv19 * aff17 + 3 * aff14 *
            abb3 * aff15 * aff17 - 3 * app14 * att19 * app16 * aff17 + 4 *
            abb4 * abb6 * avv19 * abb8 * abb9 - 4 * aff14 * abb3 * aff15 *
            avv19 * abb8 * abb9 + 8 * bll16 * ajj15 * app10 * abb8 * abb9 + 2
            * abb4 * abb6 * avv19 * ass23 * abb9 - 2 * aff14 * abb3 * aff15 *
            avv19 * ass23 * abb9 + 4 * bll16 * ajj15 * app10 * ass23 * abb9 +
            4 * aff05 * aaa2 * aff10 * avv19 * aff13 + 3 * aff14 * abb3 *
            aff15 * aff13 - 3 * app14 * att19 * app16 * aff13 + 2 * aff05 *
            aaa2 * aff10 * avv19 - 3 * aff14 * abb3 * aff15 + 3 * app14 *
            att19 * app16
        )
        byy24 = 2 * aff14 * ajj15 * aff15 - 3 * bjj26 * azz19 * app16
        byy35 = (
            (-6 * aff14 * abb3 * aff15) + 21 * bjj26 * att19 * app16 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 6) / agg15 ** 7
        )
        y2 = (
            (-(6 * aff05 * aaa2 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) + 2
            * abb4 * abb6 * bss29 * bmm34 * bmm35 - 2 * aff14 * abb3 * aff15 *
            bss29 * bmm34 * bmm35 + 2 * aff05 * aaa2 * aff10 * bss26 * bmm34 *
            bmm35 + 8 * app08 * ajj15 * app10 * bss25 * bmm34 * bmm35 + 8 *
            aff05 * aaa2 * aff10 * bss29 * ass28 * aff17 - abb4 * abb6 * bss26
            * ass28 * aff17 + aff14 * abb3 * aff15 * bss26 * ass28 * aff17 - 6
            * aff14 * abb3 * aff15 * bss25 * ass28 * aff17 + 6 * bjj26 * att19
            * app16 * bss25 * ass28 * aff17 + abb4 * abb6 * byy24 * ass28 *
            aff17 + abb4 * aaa2 * abb6 * bss21 * ass28 * aff17 - 2 * bjj08 *
            azz19 * bjj11 * ass28 * aff17 - 4 * aff05 * aaa2 * aff10 * bss29 *
            aff17 - abb4 * abb6 * bss26 * aff17 + aff14 * abb3 * aff15 * bss26
            * aff17 - 6 * aff14 * abb3 * aff15 * bss25 * aff17 + 6 * bjj26 *
            att19 * app16 * bss25 * aff17 + abb4 * abb6 * byy24 * aff17 + abb4
            * aaa2 * abb6 * bss21 * aff17 - 2 * bjj08 * azz19 * bjj11 * aff17
            - 4 * abb4 * abb6 * bss29 * abb8 * abb9 + 4 * aff14 * abb3 * aff15
            * bss29 * abb8 * abb9 - 4 * aff05 * aaa2 * aff10 * bss26 * abb8 *
            abb9 - 16 * app08 * ajj15 * app10 * bss25 * abb8 * abb9 - byy35 *
            abb8 * abb9 - 2 * abb4 * abb6 * bss29 * ass23 * abb9 + 2 * aff14 *
            abb3 * aff15 * bss29 * ass23 * abb9 - 2 * aff05 * aaa2 * aff10 *
            bss26 * ass23 * abb9 - 8 * app08 * ajj15 * app10 * bss25 * ass23 *
            abb9 + byy35 * ass23 * abb9 - 4 * aff05 * aaa2 * aff10 * bss29 *
            aff13 - abb4 * abb6 * bss26 * aff13 + aff14 * abb3 * aff15 * bss26
            * aff13 - 6 * aff14 * abb3 * aff15 * bss25 * aff13 + 6 * bjj26 *
            att19 * app16 * bss25 * aff13 + abb4 * abb6 * byy24 * aff13 + abb4
            * aaa2 * abb6 * bss21 * aff13 - 2 * bjj08 * azz19 * bjj11 * aff13
            - 2 * aff05 * aaa2 * aff10 * bss29 + abb4 * abb6 * bss26 - aff14 *
            abb3 * aff15 * bss26 + 6 * aff14 * abb3 * aff15 * bss25 - 6 *
            bjj26 * att19 * app16 * bss25 - abb4 * abb6 * byy24 - abb4 * aaa2
            * abb6 * bss21 + 2 * bjj08 * azz19 * bjj11 - (8 * aqq03 * aaa2) /
            aqq05 + (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aqq05 ** 4
        )
        bzz7 = abb8 ** 2
        bzz9 = abb9 ** 2
        z2 = (
            (-(6 * abb4 * abb6 * abb9 ** 4) / abb8 ** 4) +
            (8 * abb4 * abb6 * bzz9) / bzz7 - 4 * abb4 * abb6 * bzz9 - 4 *
            abb4 * abb6 * bzz7 - 2 * abb4 * abb6
        )
        a3 = (
            (6 * abb4 * abb6 * aii12 * abb9 ** 4) / abb8 ** 4 -
            (2 * aff14 * abb3 * aii17 * abb9 ** 3) / abb8 ** 3 -
            (8 * abb4 * abb6 * aii12 * aff17) / aff13 + 4 * abb4 * abb6 *
            aii12 * aff17 + 4 * aff14 * abb3 * aii17 * abb8 * abb9 +
            (2 * aff14 * abb3 * aii17 * abb9) / abb8 + 4 * abb4 * abb6 * aii12
            * aff13 + 2 * abb4 * abb6 * aii12
        )
        cbb09 = 1 / agg15 ** 5
        cbb18 = 2 * aff14 * abb3 * aii17 - 3 * app14 * att19 * cbb09
        cbb24 = aii11 + ayy14 - aff14 * aaa2 ** 3 * aii17
        b3 = (
            (-(6 * abb4 * abb6 * ayy24 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            abb6 * cbb24 * bmm34 * bmm35 + 4 * aff14 * abb3 * aii17 * ayy16 *
            bmm34 * bmm35 + 8 * abb4 * abb6 * ayy24 * ass28 * aff17 + cbb18 *
            ass28 * aff17 - 4 * abb4 * abb6 * ayy24 * aff17 + cbb18 * aff17 -
            4 * abb4 * abb6 * cbb24 * abb8 * abb9 - 8 * aff14 * abb3 * aii17 *
            ayy16 * abb8 * abb9 - 2 * abb4 * abb6 * cbb24 * ass23 * abb9 - 4 *
            aff14 * abb3 * aii17 * ayy16 * ass23 * abb9 - 4 * abb4 * abb6 *
            ayy24 * aff13 + cbb18 * aff13 - 2 * abb4 * abb6 * ayy24 - 2 *
            aff14 * abb3 * aii17 + 3 * app14 * att19 * cbb09
        )
        ccc23 = (
            aii11 + ayy14 + aff14 * ajj15 * aii17 - 3 * app14 * azz19 * cbb09
        )
        ccc24 = ayy16 ** 3
        ccc28 = (
            (-4 * aff14 * abb3 * aii17) + 18 * app14 * att19 * cbb09 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 6) / agg15 ** 7
        )
        c3 = (
            (6 * abb4 * abb6 * ccc24 * abb9 ** 4) / abb8 ** 4 - 6 * aff14 *
            abb3 * aii17 * ayy24 * bmm34 * bmm35 - 6 * abb4 * abb6 * ayy16 *
            ayy17 * bmm34 * bmm35 - 8 * abb4 * abb6 * ccc24 * ass28 * aff17 +
            abb4 * abb6 * ccc23 * ass28 * aff17 + 3 * aff14 * abb3 * aii17 *
            ayy17 * ass28 * aff17 - 3 * cbb18 * ayy16 * ass28 * aff17 + 4 *
            abb4 * abb6 * ccc24 * aff17 + abb4 * abb6 * ccc23 * aff17 + 3 *
            aff14 * abb3 * aii17 * ayy17 * aff17 - 3 * cbb18 * ayy16 * aff17 +
            12 * aff14 * abb3 * aii17 * ayy24 * abb8 * abb9 + 12 * abb4 * abb6
            * ayy16 * ayy17 * abb8 * abb9 - ccc28 * abb8 * abb9 + 6 * aff14 *
            abb3 * aii17 * ayy24 * ass23 * abb9 + 6 * abb4 * abb6 * ayy16 *
            ayy17 * ass23 * abb9 + ccc28 * ass23 * abb9 + 4 * abb4 * abb6 *
            ccc24 * aff13 + abb4 * abb6 * ccc23 * aff13 + 3 * aff14 * abb3 *
            aii17 * ayy17 * aff13 - 3 * cbb18 * ayy16 * aff13 + 2 * abb4 *
            abb6 * ccc24 - abb4 * abb6 * ccc23 - 3 * aff14 * abb3 * aii17 *
            ayy17 + 3 * cbb18 * ayy16 - (8 * abb1 * aaa2) / aff04 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * ajj15) / aff04 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * azz19) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 7) / aff04 ** 4
        )
        cdd24 = aaa2 ** 7
        d3 = (
            (-(6 * bjj14 * att19 * app10 * abb9 ** 4) / abb8 ** 4) + 12 *
            app21 * ajj15 * aff15 * bkk33 * bkk34 - 12 * bjj24 * azz19 * app16
            * bkk33 * bkk34 - 7 * aff05 * abb3 * aff10 * aqq27 * aff17 + 8 *
            bjj14 * att19 * app10 * aqq27 * aff17 + 22 * app08 * att19 * app10
            * aqq27 * aff17 - 15 * bjj08 * btt24 * bjj11 * aqq27 * aff17 - 7 *
            aff05 * abb3 * aff10 * aff17 - 4 * bjj14 * att19 * app10 * aff17 +
            22 * app08 * att19 * app10 * aff17 - 15 * bjj08 * btt24 * bjj11 *
            aff17 - abb4 * aaa2 * abb6 * abb8 * abb9 - 24 * app21 * ajj15 *
            aff15 * abb8 * abb9 + 13 * aff14 * ajj15 * aff15 * abb8 * abb9 -
            27 * bjj26 * azz19 * app16 * abb8 * abb9 + 24 * bjj24 * azz19 *
            app16 * abb8 * abb9 + 15 * bjj18 * cdd24 * bjj21 * abb8 * abb9 +
            abb4 * aaa2 * abb6 * agg17 * abb9 - 12 * app21 * ajj15 * aff15 *
            agg17 * abb9 - 13 * aff14 * ajj15 * aff15 * agg17 * abb9 + 27 *
            bjj26 * azz19 * app16 * agg17 * abb9 + 12 * bjj24 * azz19 * app16
            * agg17 * abb9 - 15 * bjj18 * cdd24 * bjj21 * agg17 * abb9 - 7 *
            aff05 * abb3 * aff10 * aff13 - 4 * bjj14 * att19 * app10 * aff13 +
            22 * app08 * att19 * app10 * aff13 - 15 * bjj08 * btt24 * bjj11 *
            aff13 + 7 * aff05 * abb3 * aff10 - 2 * bjj14 * att19 * app10 - 22
            * app08 * att19 * app10 + 15 * bjj08 * btt24 * bjj11 -
            (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        e3 = (
            (-(6 * app21 * ajj15 * aff15 * abb9 ** 4) / abb8 ** 4) + 6 * aff05
            * abb3 * aff10 * bkk33 * bkk34 - 6 * bll16 * att19 * app10 * bkk33
            * bkk34 - abb4 * aaa2 * abb6 * aqq27 * aff17 + 8 * app21 * ajj15 *
            aff15 * aqq27 * aff17 + 4 * aff14 * ajj15 * aff15 * aqq27 * aff17
            - 3 * app14 * azz19 * app16 * aqq27 * aff17 - abb4 * aaa2 * abb6 *
            aff17 - 4 * app21 * ajj15 * aff15 * aff17 + 4 * aff14 * ajj15 *
            aff15 * aff17 - 3 * app14 * azz19 * app16 * aff17 - 12 * aff05 *
            abb3 * aff10 * abb8 * abb9 + 12 * bll16 * att19 * app10 * abb8 *
            abb9 - 6 * aff05 * abb3 * aff10 * agg17 * abb9 + 6 * bll16 * att19
            * app10 * agg17 * abb9 - abb4 * aaa2 * abb6 * aff13 - 4 * app21 *
            ajj15 * aff15 * aff13 + 4 * aff14 * ajj15 * aff15 * aff13 - 3 *
            app14 * azz19 * app16 * aff13 + abb4 * aaa2 * abb6 - 2 * app21 *
            ajj15 * aff15 - 4 * aff14 * ajj15 * aff15 + 3 * app14 * azz19 *
            app16
        )
        f3 = (
            (6 * app21 * ajj15 * aff15 * avv19 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff05 * abb3 * aff10 * avv19 * bmm34 * bmm35 + 6 * app08 * att19 *
            app10 * avv19 * bmm34 * bmm35 - 6 * bjj24 * azz19 * app16 * bmm34
            * bmm35 + abb4 * aaa2 * abb6 * avv19 * ass28 * aff17 - 8 * app21 *
            ajj15 * aff15 * avv19 * ass28 * aff17 - 4 * aff14 * ajj15 * aff15
            * avv19 * ass28 * aff17 + 3 * bjj26 * azz19 * app16 * avv19 *
            ass28 * aff17 + 12 * app08 * att19 * app10 * ass28 * aff17 - 12 *
            bjj08 * btt24 * bjj11 * ass28 * aff17 + abb4 * aaa2 * abb6 * avv19
            * aff17 + 4 * app21 * ajj15 * aff15 * avv19 * aff17 - 4 * aff14 *
            ajj15 * aff15 * avv19 * aff17 + 3 * bjj26 * azz19 * app16 * avv19
            * aff17 + 12 * app08 * att19 * app10 * aff17 - 12 * bjj08 * btt24
            * bjj11 * aff17 + 12 * aff05 * abb3 * aff10 * avv19 * abb8 * abb9
            - 12 * app08 * att19 * app10 * avv19 * abb8 * abb9 + 9 * aff14 *
            ajj15 * aff15 * abb8 * abb9 - 24 * bjj26 * azz19 * app16 * abb8 *
            abb9 + 12 * bjj24 * azz19 * app16 * abb8 * abb9 + 15 * bjj18 *
            cdd24 * bjj21 * abb8 * abb9 + 6 * aff05 * abb3 * aff10 * avv19 *
            ass23 * abb9 - 6 * app08 * att19 * app10 * avv19 * ass23 * abb9 -
            9 * aff14 * ajj15 * aff15 * ass23 * abb9 + 24 * bjj26 * azz19 *
            app16 * ass23 * abb9 + 6 * bjj24 * azz19 * app16 * ass23 * abb9 -
            15 * bjj18 * cdd24 * bjj21 * ass23 * abb9 + abb4 * aaa2 * abb6 *
            avv19 * aff13 + 4 * app21 * ajj15 * aff15 * avv19 * aff13 - 4 *
            aff14 * ajj15 * aff15 * avv19 * aff13 + 3 * bjj26 * azz19 * app16
            * avv19 * aff13 + 12 * app08 * att19 * app10 * aff13 - 12 * bjj08
            * btt24 * bjj11 * aff13 - abb4 * aaa2 * abb6 * avv19 + 2 * app21 *
            ajj15 * aff15 * avv19 + 4 * aff14 * ajj15 * aff15 * avv19 - 3 *
            bjj26 * azz19 * app16 * avv19 - 12 * app08 * att19 * app10 + 12 *
            bjj08 * btt24 * bjj11 - (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        g3 = (
            (-(6 * aff05 * abb3 * aff10 * abb9 ** 4) / abb8 ** 4) + 2 * abb4 *
            aaa2 * abb6 * bkk33 * bkk34 - 2 * aff14 * ajj15 * aff15 * bkk33 *
            bkk34 + (8 * aff05 * abb3 * aff10 * aff17) / aff13 - 4 * aff05 *
            abb3 * aff10 * aff17 - 4 * abb4 * aaa2 * abb6 * abb8 * abb9 + 4 *
            aff14 * ajj15 * aff15 * abb8 * abb9 - 2 * abb4 * aaa2 * abb6 *
            agg17 * abb9 + 2 * aff14 * ajj15 * aff15 * agg17 * abb9 - 4 *
            aff05 * abb3 * aff10 * aff13 - 2 * aff05 * abb3 * aff10
        )
        h3 = (
            (6 * aff05 * abb3 * aff10 * avv19 * abb9 ** 4) / abb8 ** 4 - 2 *
            abb4 * aaa2 * abb6 * avv19 * bmm34 * bmm35 + 2 * aff14 * ajj15 *
            aff15 * avv19 * bmm34 * bmm35 - 4 * bll16 * att19 * app10 * bmm34
            * bmm35 - 8 * aff05 * abb3 * aff10 * avv19 * ass28 * aff17 + 3 *
            aff14 * ajj15 * aff15 * ass28 * aff17 - 3 * app14 * azz19 * app16
            * ass28 * aff17 + 4 * aff05 * abb3 * aff10 * avv19 * aff17 + 3 *
            aff14 * ajj15 * aff15 * aff17 - 3 * app14 * azz19 * app16 * aff17
            + 4 * abb4 * aaa2 * abb6 * avv19 * abb8 * abb9 - 4 * aff14 * ajj15
            * aff15 * avv19 * abb8 * abb9 + 8 * bll16 * att19 * app10 * abb8 *
            abb9 + 2 * abb4 * aaa2 * abb6 * avv19 * ass23 * abb9 - 2 * aff14 *
            ajj15 * aff15 * avv19 * ass23 * abb9 + 4 * bll16 * att19 * app10 *
            ass23 * abb9 + 4 * aff05 * abb3 * aff10 * avv19 * aff13 + 3 *
            aff14 * ajj15 * aff15 * aff13 - 3 * app14 * azz19 * app16 * aff13
            + 2 * aff05 * abb3 * aff10 * avv19 - 3 * aff14 * ajj15 * aff15 + 3
            * app14 * azz19 * app16
        )
        i3 = (
            (-(6 * aff05 * abb3 * aff10 * bss29 * abb9 ** 4) / abb8 ** 4) + 2
            * abb4 * aaa2 * abb6 * bss29 * bmm34 * bmm35 - 2 * aff14 * ajj15 *
            aff15 * bss29 * bmm34 * bmm35 + 2 * aff05 * abb3 * aff10 * bss26 *
            bmm34 * bmm35 + 8 * app08 * att19 * app10 * bss25 * bmm34 * bmm35
            + 8 * aff05 * abb3 * aff10 * bss29 * ass28 * aff17 - abb4 * aaa2 *
            abb6 * bss26 * ass28 * aff17 + aff14 * ajj15 * aff15 * bss26 *
            ass28 * aff17 - 6 * aff14 * ajj15 * aff15 * bss25 * ass28 * aff17
            + 6 * bjj26 * azz19 * app16 * bss25 * ass28 * aff17 + 4 * app08 *
            att19 * app10 * ass28 * aff17 - 8 * bjj08 * btt24 * bjj11 * ass28
            * aff17 - 4 * aff05 * abb3 * aff10 * bss29 * aff17 - abb4 * aaa2 *
            abb6 * bss26 * aff17 + aff14 * ajj15 * aff15 * bss26 * aff17 - 6 *
            aff14 * ajj15 * aff15 * bss25 * aff17 + 6 * bjj26 * azz19 * app16
            * bss25 * aff17 + 4 * app08 * att19 * app10 * aff17 - 8 * bjj08 *
            btt24 * bjj11 * aff17 - 4 * abb4 * aaa2 * abb6 * bss29 * abb8 *
            abb9 + 4 * aff14 * ajj15 * aff15 * bss29 * abb8 * abb9 - 4 * aff05
            * abb3 * aff10 * bss26 * abb8 * abb9 - 16 * app08 * att19 * app10
            * bss25 * abb8 * abb9 + 6 * aff14 * ajj15 * aff15 * abb8 * abb9 -
            21 * bjj26 * azz19 * app16 * abb8 * abb9 + 15 * bjj18 * cdd24 *
            bjj21 * abb8 * abb9 - 2 * abb4 * aaa2 * abb6 * bss29 * ass23 *
            abb9 + 2 * aff14 * ajj15 * aff15 * bss29 * ass23 * abb9 - 2 *
            aff05 * abb3 * aff10 * bss26 * ass23 * abb9 - 8 * app08 * att19 *
            app10 * bss25 * ass23 * abb9 - 6 * aff14 * ajj15 * aff15 * ass23 *
            abb9 + 21 * bjj26 * azz19 * app16 * ass23 * abb9 - 15 * bjj18 *
            cdd24 * bjj21 * ass23 * abb9 - 4 * aff05 * abb3 * aff10 * bss29 *
            aff13 - abb4 * aaa2 * abb6 * bss26 * aff13 + aff14 * ajj15 * aff15
            * bss26 * aff13 - 6 * aff14 * ajj15 * aff15 * bss25 * aff13 + 6 *
            bjj26 * azz19 * app16 * bss25 * aff13 + 4 * app08 * att19 * app10
            * aff13 - 8 * bjj08 * btt24 * bjj11 * aff13 - 2 * aff05 * abb3 *
            aff10 * bss29 + abb4 * aaa2 * abb6 * bss26 - aff14 * ajj15 * aff15
            * bss26 + 6 * aff14 * ajj15 * aff15 * bss25 - 6 * bjj26 * azz19 *
            app16 * bss25 - 4 * app08 * att19 * app10 + 8 * bjj08 * btt24 *
            bjj11 - (8 * aqq03 * abb3) / aqq05 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * att19) / aqq05 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * btt24) / aqq05 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aqq05 ** 4
        )
        j3 = (
            (-(6 * abb4 * aaa2 * abb6 * abb9 ** 4) / abb8 ** 4) +
            (8 * abb4 * aaa2 * abb6 * bzz9) / bzz7 - 4 * abb4 * aaa2 * abb6 *
            bzz9 - 4 * abb4 * aaa2 * abb6 * bzz7 - 2 * abb4 * aaa2 * abb6
        )
        k3 = (
            (6 * abb4 * aaa2 * bdd14 * bdd15 * abb9 ** 4) / abb8 ** 4 -
            (2 * aff14 * ajj15 * bdd08 * abb9 ** 3) / abb8 ** 3 -
            (8 * abb4 * aaa2 * bdd14 * bdd15 * aff17) / aff13 + 4 * abb4 *
            aaa2 * bdd14 * bdd15 * aff17 + 4 * aff14 * ajj15 * bdd08 * abb8 *
            abb9 + (2 * aff14 * ajj15 * bdd08 * abb9) / abb8 + 4 * abb4 * aaa2
            * bdd14 * bdd15 * aff13 + 2 * abb4 * aaa2 * bdd14 * bdd15
        )
        cll08 = 1 / bdd07 ** 5
        cll16 = aii11 + bhh13
        cll17 = cll16 ** 2
        cll18 = 2 * aff14 * ajj15 * bdd08 - 3 * app14 * azz19 * cll08
        cll24 = aii11 + bhh13 - aff14 * ajj15 * bdd08
        l3 = (
            (-(6 * abb4 * aaa2 * bdd14 * cll17 * abb9 ** 4) / abb8 ** 4) + 2 *
            abb4 * aaa2 * bdd14 * cll24 * bmm34 * bmm35 + 4 * aff14 * ajj15 *
            bdd08 * cll16 * bmm34 * bmm35 + 8 * abb4 * aaa2 * bdd14 * cll17 *
            ass28 * aff17 + cll18 * ass28 * aff17 - 4 * abb4 * aaa2 * bdd14 *
            cll17 * aff17 + cll18 * aff17 - 4 * abb4 * aaa2 * bdd14 * cll24 *
            abb8 * abb9 - 8 * aff14 * ajj15 * bdd08 * cll16 * abb8 * abb9 - 2
            * abb4 * aaa2 * bdd14 * cll24 * ass23 * abb9 - 4 * aff14 * ajj15 *
            bdd08 * cll16 * ass23 * abb9 - 4 * abb4 * aaa2 * bdd14 * cll17 *
            aff13 + cll18 * aff13 - 2 * abb4 * aaa2 * bdd14 * cll17 - 2 *
            aff14 * ajj15 * bdd08 + 3 * app14 * azz19 * cll08
        )
        cmm12 = -3 * app14 * azz19 * cbb09
        cmm16 = 2 * aff14 * ajj15 * aii17 + cmm12
        cmm23 = aii11 + ayy14 + aff14 * ajj15 * aii17 + cmm12
        cmm28 = (
            (-4 * aff14 * ajj15 * aii17) + 18 * app14 * azz19 * cbb09 -
            (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 7) / agg15 ** 7
        )
        m3 = (
            (6 * abb4 * aaa2 * abb6 * ccc24 * abb9 ** 4) / abb8 ** 4 - 6 *
            aff14 * ajj15 * aii17 * ayy24 * bmm34 * bmm35 - 6 * abb4 * aaa2 *
            abb6 * ayy16 * ayy17 * bmm34 * bmm35 - 8 * abb4 * aaa2 * abb6 *
            ccc24 * ass28 * aff17 + abb4 * aaa2 * abb6 * cmm23 * ass28 * aff17
            + 3 * aff14 * ajj15 * aii17 * ayy17 * ass28 * aff17 - 3 * cmm16 *
            ayy16 * ass28 * aff17 + 4 * abb4 * aaa2 * abb6 * ccc24 * aff17 +
            abb4 * aaa2 * abb6 * cmm23 * aff17 + 3 * aff14 * ajj15 * aii17 *
            ayy17 * aff17 - 3 * cmm16 * ayy16 * aff17 + 12 * aff14 * ajj15 *
            aii17 * ayy24 * abb8 * abb9 + 12 * abb4 * aaa2 * abb6 * ayy16 *
            ayy17 * abb8 * abb9 - cmm28 * abb8 * abb9 + 6 * aff14 * ajj15 *
            aii17 * ayy24 * ass23 * abb9 + 6 * abb4 * aaa2 * abb6 * ayy16 *
            ayy17 * ass23 * abb9 + cmm28 * ass23 * abb9 + 4 * abb4 * aaa2 *
            abb6 * ccc24 * aff13 + abb4 * aaa2 * abb6 * cmm23 * aff13 + 3 *
            aff14 * ajj15 * aii17 * ayy17 * aff13 - 3 * cmm16 * ayy16 * aff13
            + 2 * abb4 * aaa2 * abb6 * ccc24 - abb4 * aaa2 * abb6 * cmm23 - 3
            * aff14 * ajj15 * aii17 * ayy17 + 3 * cmm16 * ayy16 -
            (8 * abb1 * abb3) / aff04 +
            (56 * exp1 ** ((-4 * t) - 4 * p) * aaa2 ** 4) / aff04 ** 2 -
            (96 * exp1 ** ((-6 * t) - 6 * p) * aaa2 ** 6) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aff04 ** 4
        )
        cnn3 = abb8 ** 2
        cnn5 = abb9 ** 2
        n3 = (
            (-(6 * abb9 ** 4) / abb8 ** 4) + (8 * cnn5) / cnn3 - 4 * cnn5 - 4
            * cnn3 - 2
        )
        coo7 = abb8 ** 2
        coo9 = abb9 ** 2
        o3 = (
            (6 * bgg4 * abb9 ** 4) / abb8 ** 4 - (8 * bgg4 * coo9) / coo7 + 4
            * bgg4 * coo9 + 4 * bgg4 * coo7 + 2 * bgg4
        )
        cpp06 = -aaa2 / (exp1 ** t * bdd07)
        cpp08 = (cpp06 + aii11) ** 2
        cpp12 = (
            aii11 + cpp06 - (exp1 ** (aaa1 + aff08) * aaa2 ** 3) / bdd07 ** 3
        )
        p3 = (
            (-(6 * cpp08 * abb9 ** 4) / abb8 ** 4) + (2 * cpp12 * abb9 ** 3) /
            abb8 ** 3 + (8 * cpp08 * aff17) / aff13 - 4 * cpp08 * aff17 - 4 *
            cpp12 * abb8 * abb9 - (2 * cpp12 * abb9) / abb8 - 4 * cpp08 *
            aff13 - 2 * cpp08
        )
        cqq12 = -aff14 * ajj15 * bdd08
        cqq19 = bhh14 + bhh13
        cqq20 = cqq19 ** 3
        cqq21 = (
            bhh14 + bhh13 + aff14 * ajj15 * bdd08 - 3 * app14 * azz19 * cll08
        )
        cqq25 = bhh14 + bhh13 + cqq12
        cqq28 = 1 / aff13
        q3 = (
            (6 * cqq20 * abb9 ** 4) / abb8 ** 4 -
            (6 * cqq19 * cqq25 * abb9 ** 3) / abb8 ** 3 - 8 * cqq20 * cqq28 *
            aff17 + cqq21 * cqq28 * aff17 + 4 * cqq20 * aff17 + cqq21 * aff17
            + 12 * cqq19 * cqq25 * abb8 * abb9 + (6 * cqq19 * cqq25 * abb9) /
            abb8 + 4 * cqq20 * aff13 + cqq21 * aff13 + 2 * cqq20 - ann05 *
            ann06 + abb4 * aaa2 * bdd14 + cqq12 + 3 * app14 * azz19 * cll08
        )
        crr18 = (
            aii11 + aoo09 + aff14 * ajj15 * aii17 - 3 * app14 * azz19 * cbb09
        )
        crr19 = bii11 ** 4
        crr21 = bii15 ** 2
        crr25 = (
            aii11 + aoo09 - 3 * aff14 * ajj15 * aii17 + 15 * app14 * azz19 *
            cbb09 - (15 * exp1 ** (aaa1 + 6 * abb5) * aaa2 ** 7) / agg15 ** 7
        )
        crr28 = bii11 ** 2
        r3 = (
            (-(6 * crr19 * abb9 ** 4) / abb8 ** 4) +
            (12 * crr28 * bii15 * abb9 ** 3) / abb8 ** 3 - 3 * crr21 * ass28 *
            aff17 + 8 * crr19 * ass28 * aff17 - 4 * bii11 * crr18 * ass28 *
            aff17 - 3 * crr21 * aff17 - 4 * crr19 * aff17 - 4 * bii11 * crr18
            * aff17 - 24 * crr28 * bii15 * abb8 * abb9 - crr25 * abb8 * abb9 -
            12 * crr28 * bii15 * ass23 * abb9 + crr25 * ass23 * abb9 - 3 *
            crr21 * aff13 - 4 * crr19 * aff13 - 4 * bii11 * crr18 * aff13 + 3
            * crr21 - 2 * crr19 + 4 * bii11 * crr18 - (8 * abb1 * abb3) /
            aff04 + (56 * exp1 ** ((-4 * t) - 4 * p) * aaa2 ** 4) / aff04 ** 2
            - (96 * exp1 ** ((-6 * t) - 6 * p) * aaa2 ** 6) / aff04 ** 3 +
            (48 * exp1 ** ((-8 * t) - 8 * p) * aaa2 ** 8) / aff04 ** 4
        )

        L4 = np.column_stack([j2, k2, l2, m2, n2, o2, p2, q2, r2, s2,
                              t2, u2, v2, w2, x2, y2, z2, a3, b3, c3,
                              d3, e3, f3, g3, h3, i3, j3, k3, l3, m3,
                              n3, o3, p3, q3, r3])
    return l0, L1, L2, L3, L4


def _r_tweedie(rng, mu, p: float, phi: float) -> np.ndarray:
    """mgcv ``rTweedie``: compound Poisson-Gamma deviates for 1 < p < 2 —
    ``N_i ~ Poisson(λ_i)`` jumps per row, each ``Gamma(shape, scale_i)``, summed
    (mgcv's ``C_psum``). R draws all ``rpois`` first, then every individual gamma
    jump in row order; reproducing that order makes this bit-exact via
    ``hea.R.rng`` (the earlier collapsed one-gamma-per-row form was Monte-Carlo).
    """
    mu = np.asarray(mu, dtype=float)
    if not (1.0 < p < 2.0):
        raise ValueError("p must be in (1,2)")
    if np.any(mu < 0):
        raise ValueError("mean, mu, must be non negative")
    if phi <= 0:
        raise ValueError("scale parameter must be positive")
    lam = mu ** (2.0 - p) / ((2.0 - p) * phi)
    shape = (2.0 - p) / (p - 1.0)
    scale = phi * (p - 1.0) * mu ** (p - 1.0)
    n = mu.shape[0]
    N = np.asarray(rng.poisson(lam)).astype(np.int64)   # n Poisson jump counts
    gs = np.repeat(scale, N)                             # scale_i repeated N_i×
    jumps = rng.gamma(shape, gs)                         # Σ N_i gamma jumps
    y = np.zeros(n)
    np.add.at(y, np.repeat(np.arange(n), N), jumps)      # C_psum per row
    return y


class Tweedie(Family):
    """Tweedie EDF with fixed power ``p ∈ (1, 2)`` — compound Poisson-Gamma.

    Mean ``μ``, variance ``φ·μ^p``. The density mixes an exact point mass at
    ``y = 0`` with a continuous part on ``y > 0``; ``ls`` and ``aic`` evaluate
    it via the Dunn-Smyth series (see :func:`_tweedie_log_a_one`). For joint
    estimation of ``p`` with the smoothing parameters, use :class:`tw`.

    Default link is ``log``. Scale ``φ`` is unknown (Pearson/REML estimated).
    """
    name = "Tweedie"
    canonical_link_name = "log"  # mgcv's default; no canonical link in the strict
                                  # EDF sense for non-integer p.
    # mgcv sets canonical="none" explicitly (gam.fit3.r:3105; tw
    # efam.r:3262): PIRLS runs full Newton even at the default log link.
    _newton_canonical = "none"
    scale_known = False

    def __init__(self, p: float, link=None):
        if not (1.0 < p < 2.0):
            raise ValueError(f"Tweedie requires 1 < p < 2; got p={p!r}")
        self.p = float(p)
        # (φ, p, y-fingerprint) → 7-moment Dunn-Smyth bundle. The saturated
        # series a(y, φ, p) is μ-INDEPENDENT, so ls/dls_dp/_d2ls_dp at one
        # (φ, p) point — and every PIRLS iter / repeated outer eval at that
        # point — recompute the same arrays. See _saturated_series.
        self._sat_series_cache: dict = {}
        super().__init__(link=link)

    def variance(self, mu):
        return np.asarray(mu, dtype=float) ** self.p

    def dvar(self, mu):
        return self.p * np.asarray(mu, dtype=float) ** (self.p - 1.0)

    def d2var(self, mu):
        return (self.p * (self.p - 1.0)
                * np.asarray(mu, dtype=float) ** (self.p - 2.0))

    def d3var(self, mu):
        return (self.p * (self.p - 1.0) * (self.p - 2.0)
                * np.asarray(mu, dtype=float) ** (self.p - 3.0))

    def dev_resids(self, y, mu, wt, theta=None):
        # 1<p<2 form (Jorgensen 1987):
        #   y > 0:  d_i = 2·[ y·(y^(1-p) - μ^(1-p))/(1-p) - (y^(2-p) - μ^(2-p))/(2-p) ]
        #   y = 0:  d_i = 2·μ^(2-p)/(2-p)
        # Both pieces are non-negative for 1<p<2, μ>0, y≥0; minimised at y=μ.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        # Mask y inside the y^(...) so y=0 rows don't generate spurious 0**neg.
        y_safe = np.where(zero, 1.0, y)
        d_pos = 2.0 * (y * (y_safe ** om1 - mu ** om1) / om1
                       - (y_safe ** tm - mu ** tm) / tm)
        d_zero = 2.0 * mu ** tm / tm
        return wt * np.where(zero, d_zero, d_pos)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the 'Tweedie' family"
            )
        # mgcv: mustart = y + 0.1·(y==0) — bump only the zeros so log(μ)
        # stays finite (Tweedie gam.fit3.r:3078, tw efam.r:3234).
        return y + 0.1 * (y == 0.0)

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def rd(self, rng, mu, wt, scale):
        # Tweedie rd (gam.fit3.r:3097-3099) / tw rd (efam.r:3245-3254,
        # inherited): rTweedie(mu, p, phi=scale). ``wt`` is in mgcv's
        # signature but unread — prior weights don't enter, bug-for-bug.
        # (mgcv's p==2 rgamma branch is unreachable here: hea requires
        # 1 < p < 2.)
        return _r_tweedie(rng, mu, self.p, float(scale))

    def _log_density(self, y, mu, phi, p=None):
        """Per-obs log f(y_i; μ_i, φ, p), shape (n,) — one unmodified φ for
        every row (mgcv's ``ldTweedie(y, mu, p, phi=scale)``; prior weights
        multiply the summed log-density at the call site, they never divide
        the dispersion — same convention as ``ls``). ``p`` defaults to the
        family's own power; ``tw.aic`` passes the power implied by its θ
        argument."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        phi_i = np.full_like(y, float(phi))
        if p is None:
            p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        # cumulant_i = y_i·μ_i^(1-p)/(1-p) - μ_i^(2-p)/(2-p) (the y-only term
        # vanishes at y=0; the rest is the y=0 closed form's exponent).
        cumulant = y * mu ** om1 / om1 - mu ** tm / tm
        out = np.empty_like(y)
        out[zero] = cumulant[zero] / phi_i[zero]
        if np.any(~zero):
            la = _tweedie_log_a_vec(y[~zero], phi_i[~zero], p)[0]
            out[~zero] = -np.log(y[~zero]) + la + cumulant[~zero] / phi_i[~zero]
        return out

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv's ``Tweedie()$aic`` (gam.fit3.r:3086) and ``tw()$aic``
        # (efam.r:3212), identical math: scale = dev/Σwt — the caller's
        # dev1 is scale·Σwt (gam.fit3.r:848 / gam.fit4.r:794), so this
        # recovers the REML/Pearson scale — then
        # -2·Σ wt·ldTweedie(y, μ, p, φ=scale) + 2.
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        phi = max(float(dev) / max(n_eff, 1e-300), 1e-12)
        log_f = self._log_density(y, mu, phi)
        return -2.0 * float(np.sum(log_f * wt)) + 2.0

    def _saturated_series(self, y_nz, phi_nz):
        """Memoised :func:`_tweedie_log_a_vec` at the saturated point (μ = y).

        ``ls``/``dls_dp``/``_d2ls_dp`` each evaluate the Dunn-Smyth series at
        the SAME ``(y, φ=scale, p)`` and extract different moments; the series
        is μ-independent, so within a score-eval (and wherever the joint outer
        Newton revisits a ``(φ, p)``) it is otherwise recomputed redundantly
        (a cliff fit makes 189 calls at only 33 distinct ``(φ, p)`` — 5.7×).
        Mirrors mgcv's ``buffer=TRUE`` reuse in ``ldTweedie``. Bit-identical:
        returns the very arrays ``_tweedie_log_a_vec`` would (the callers only
        read them). Keyed on ``(φ, p)`` + a cheap y fingerprint (size/ends/sum
        — exact within a fit, collision-proof across datasets); ``self.p`` is
        the live power so a ``tw`` p-update invalidates stale entries."""
        p = self.p
        scale = float(phi_nz[0]) if y_nz.size else 0.0
        key = (scale, p, y_nz.size,
               float(y_nz[0]) if y_nz.size else 0.0,
               float(y_nz[-1]) if y_nz.size else 0.0,
               float(y_nz.sum()) if y_nz.size else 0.0)
        cache = self._sat_series_cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        res = _tweedie_log_a_vec(y_nz, phi_nz, p)
        if len(cache) >= 64:
            cache.clear()
        cache[key] = res
        return res

    def ls(self, y, wt, scale):
        """Saturated log-lik Σ w_i·log f(y_i; y_i, φ, p) and its 1st/2nd
        derivatives wrt log φ (hea log-scale convention).

        mgcv's Tweedie convention (BOTH variants): the prior weight
        multiplies the per-obs log-density at *unmodified* φ —
        ``colSums(w·ldTweedie(y, y, phi=scale))`` (fix.family.ls,
        gam.fit3.r:3083) and ``w·ldTweedie(y, y, rho=log(scale))``
        (tw()$ls, efam.r:3224). This deliberately differs from the
        Gamma/exponential-family ``φ_i = φ/w_i`` convention. For y_i = 0
        with μ_i = y_i = 0 the cumulant is 0 and log f = 0; the entry
        contributes nothing to ls or its derivatives. For y_i > 0:

            log f_sat = -log y + log a(y, φ_i, p) + y^(2-p)/((1-p)(2-p)·φ_i)

        and using d/dlog φ_i log a = -(1-α)·E[j], d²/dlog φ_i² log a =
        (1-α)²·Var[j] (Dunn-Smyth moments under p_j = W_j/Σ W_k):

            d ls / dlog φ   = Σ w_i · (-(1-α)·E[j_i] - c_i/φ_i)
            d² ls / dlog φ² = Σ w_i · ( (1-α)²·Var[j_i] + c_i/φ_i )

        with c_i = y_i^(2-p)/((1-p)(2-p)) the saturated cumulant (negative
        for 1<p<2).
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return np.array([0.0, 0.0, 0.0], dtype=float)
        y_g = y[good]
        w_g = wt[good]
        phi_i = np.full_like(w_g, float(scale))
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        # one_minus_alpha = 1 - (2-p)/(1-p) = -1/(1-p) = 1/(p-1)
        one_minus_alpha = 1.0 / (p - 1.0)

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        # Saturated cumulant c_i = y^(2-p)/((1-p)(2-p)) for y>0; 0 at y=0.
        cum = np.where(zero, 0.0, y_safe ** tm / (om1 * tm))

        # Series moments at μ=y; only computed for y>0 rows. ``ls`` only
        # needs (log a, E[j], Var[j]); the j_psi_bar moment is consumed by
        # ``dls_dp`` for the p-derivative path.
        log_a = np.zeros_like(y_g)
        j_bar = np.zeros_like(y_g)
        j_var = np.zeros_like(y_g)
        if np.any(~zero):
            la_, jb_, jv_ = self._saturated_series(
                y_g[~zero], phi_i[~zero])[:3]
            log_a[~zero] = la_
            j_bar[~zero] = jb_
            j_var[~zero] = jv_

        # log f_sat per observation; y=0 row is 0 by the closed form.
        log_f_sat = np.where(zero, 0.0,
                             -np.log(y_safe) + log_a + cum / phi_i)
        ls0 = float(np.sum(w_g * log_f_sat))

        d1_per = np.where(zero, 0.0, -one_minus_alpha * j_bar - cum / phi_i)
        d2_per = np.where(zero, 0.0,
                          one_minus_alpha * one_minus_alpha * j_var
                          + cum / phi_i)
        d1 = float(np.sum(w_g * d1_per))
        d2 = float(np.sum(w_g * d2_per))
        return np.array([ls0, d1, d2], dtype=float)

    # ---- analytical p-derivatives (used by joint outer Newton in tw()) ----

    def dvar_dp(self, mu):
        """``∂V(μ)/∂p = log(μ)·μ^p`` (since V = μ^p ⇒ log V = p·log μ)."""
        mu = np.asarray(mu, dtype=float)
        return np.log(mu) * mu ** self.p

    def dD_dp(self, y, mu, wt):
        """Σ_i wt_i · ∂d_i/∂p at fixed (y, μ). Used by the joint outer
        Newton when ``family.n_theta > 0`` to evaluate ``∂Dp/∂p`` (the
        envelope theorem at PIRLS-converged β̂ kills the β-coupled chain).

        For y > 0:
            d_i = 2·[y·u/om1 - v/tm]   with u = y^om1 - μ^om1, v = y^tm - μ^tm,
                                            om1 = 1-p, tm = 2-p.
            ∂d_i/∂p = 2·[ y·(μ^om1·log μ - y^om1·log y)/om1 + y·u/om1²
                         - (μ^tm·log μ - y^tm·log y)/tm - v/tm² ]
        For y = 0:
            d_i = 2·μ^tm/tm,  ∂d_i/∂p = 2·μ^tm·[1/tm² - log μ/tm].
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        zero = (y == 0.0)
        log_mu = np.log(mu)
        # y_safe is only used inside masked branches; log_y substitutes 0 for
        # y=0 so y·log y = 0 (limit of y·log y as y→0⁺).
        y_safe = np.where(zero, 1.0, y)
        log_y = np.where(zero, 0.0, np.log(y_safe))

        # y > 0 branch
        y_om1 = y_safe ** om1
        mu_om1 = mu ** om1
        y_tm = y_safe ** tm
        mu_tm = mu ** tm
        u = y_om1 - mu_om1
        v = y_tm - mu_tm
        # ∂[y·u/om1]/∂p:  y·∂u/∂p / om1 + y·u/om1²
        #   ∂u/∂p = -y^om1·log y + μ^om1·log μ
        dA1 = (y * (mu_om1 * log_mu - y_om1 * log_y) / om1
               + y * u / (om1 * om1))
        # ∂[v/tm]/∂p:    ∂v/∂p / tm + v/tm²
        #   ∂v/∂p = -y^tm·log y + μ^tm·log μ
        dA2 = ((mu_tm * log_mu - y_tm * log_y) / tm
               + v / (tm * tm))
        d_dp_pos = 2.0 * (dA1 - dA2)

        # y = 0 branch
        d_dp_zero = 2.0 * mu_tm * (1.0 / (tm * tm) - log_mu / tm)

        return float(np.sum(wt * np.where(zero, d_dp_zero, d_dp_pos)))

    def dls_dp(self, y, wt, scale):
        """``∂ls/∂p`` (saturated log-lik). Companion to ``ls`` for the
        joint-outer-Newton p-direction.

        For y_i > 0:
            log f_sat = -log y + log a(y, φ_i, p) + cum_sat(y, p)/φ_i
            ∂log f_sat/∂p = ∂log a/∂p + ∂cum_sat/∂p / φ_i
        For y_i = 0: log f_sat ≡ 0 ⇒ ∂/∂p = 0.

        Series-moment piece (Dunn-Smyth + chain rule on log W_j = j·log z
        - lgamma(j+1) - lgamma(-j·α)):

            ∂log W_j/∂p = j·K_j/(1-p)² + j/(2-p)
            K_j         = log φ + log(p-1) + ψ(-j·α) - log y - (2-p)
            ∂log a/∂p   = E[j·K_j]/(1-p)² + E[j]/(2-p)

        ``E[j]`` and ``E[j·ψ(-j·α)]`` are returned by
        :func:`_tweedie_log_a_one` (see j_bar, j_psi_bar).

        Saturated cumulant cum_sat = y^(2-p)/((1-p)(2-p)); its p-derivative is
            ∂cum_sat/∂p = y^(2-p) · [(3 - 2p) - log(y)·(1-p)·(2-p)]
                          / [(1-p)·(2-p)]²
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return 0.0
        y_g = y[good]
        w_g = wt[good]
        # Same mgcv convention as ``ls``: weight outside, φ unmodified.
        phi_i = np.full_like(w_g, float(scale))
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p
        om1_tm = om1 * tm

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        log_y = np.where(zero, 0.0, np.log(y_safe))
        log_phi = np.log(phi_i)

        # ∂cum_sat/∂p (per-obs)
        y_tm = y_safe ** tm
        dcum_dp = np.where(
            zero, 0.0,
            y_tm * ((3.0 - 2.0 * p) - log_y * om1_tm) / (om1_tm * om1_tm)
        )

        # ∂log a/∂p via series moments. Need (j_bar, j_psi_bar) over y>0 rows.
        j_bar = np.zeros_like(y_g)
        j_psi_bar = np.zeros_like(y_g)
        if np.any(~zero):
            _, jb_, _, jpb_, *_rest2 = self._saturated_series(
                y_g[~zero], phi_i[~zero]
            )
            j_bar[~zero] = jb_
            j_psi_bar[~zero] = jpb_
        # K_const_i = log φ_i + log(p-1) - log y_i - (2-p)
        # E[j·K_j] = j_bar · K_const + j_psi_bar (since ψ has E[j·ψ(-jα)])
        K_const = log_phi + np.log(p - 1.0) - log_y - tm
        E_jK = j_bar * K_const + j_psi_bar
        dlog_a_dp = np.where(zero, 0.0, E_jK / (om1 * om1) + j_bar / tm)

        dlog_f_dp = np.where(zero, 0.0, dlog_a_dp + dcum_dp / phi_i)
        return float(np.sum(w_g * dlog_f_dp))

    def _d2ls_dp(self, y, wt, scale):
        """``(∂²ls/∂p², ∂²ls/∂p∂log φ)`` at the saturated point — the
        p-space second derivatives behind tw's analytic ``lsth2``
        (ldTweedie's columns 5/6 in the (θ,ρ) form before the p(θ)
        chain: gam.fit3.r:2802-2806 density part + the C_tweedious
        series part; family-review B4).

        Density part at μ = y (mgcv's ld[,5]/ld[,6] closed forms with
        θ_y·y = y^(2−p)/(1−p), k_y = y^(2−p)/(2−p), L = log y):

            d²/dp²   = [θ_y·y(L² − 2L/(1−p) + 2/(1−p)²)
                        − k_y(L² − 2L/(2−p) + 2/(2−p)²)]/φ
            d²/dp∂φ  = −x/φ  ⇒  d²/dp∂logφ = −x   (x = density ∂/∂p)

        Series part via Dunn-Smyth moments of log a = log Σ_j W_j with
        log W_j = j·log z − lgamma(j+1) − lgamma(−jα), α = (2−p)/(1−p),
        α′ = 1/(1−p)², α″ = 2/(1−p)³, K_j = C + ψ(−jα),
        C = log φ + log(p−1) − log y − (2−p):

            ∂logW_j/∂p       = j·α′·K_j + j/(2−p)              (=: G_j)
            ∂²logW_j/∂p²     = j[α″K_j + α′(1/(p−1) + 1
                                − jα′ψ′(−jα)) + 1/(2−p)²]
            ∂²logW_j/∂p∂logφ = j·α′

            ∂²log a/∂p²      = E[∂²logW/∂p²] + Var[G]
            ∂²log a/∂p∂logφ  = α′E[j] − (1/(p−1))·[(α′C + 1/(2−p))Var[j]
                                + α′(E[j²ψ] − E[jψ]E[j])]

        y = 0 rows contribute nothing (log f_sat ≡ 0 there, matching
        ldTweedie(y, y)'s all-zero rows at y = 0).
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        if not np.any(good):
            return 0.0, 0.0
        y_g = y[good]
        w_g = wt[good]
        phi_i = np.full_like(w_g, float(scale))
        p = self.p
        om1 = 1.0 - p
        tm = 2.0 - p

        zero = (y_g == 0.0)
        y_safe = np.where(zero, 1.0, y_g)
        L = np.where(zero, 0.0, np.log(y_safe))

        # --- density part (μ = y) ---------------------------------------
        y_tm = y_safe ** tm
        th_y = y_tm / om1                  # θ_y·y = y^(2-p)/(1-p)
        k_y = y_tm / tm
        x_dens = (th_y * (1.0 / om1 - L) + k_y * (L - 1.0 / tm)) / phi_i
        d2p_dens = (th_y * (L * L - 2.0 * L / om1 + 2.0 / (om1 * om1))
                    - k_y * (L * L - 2.0 * L / tm + 2.0 / (tm * tm))) / phi_i
        cross_dens = -x_dens               # already in log φ form

        # --- series part -------------------------------------------------
        d2p_ser = np.zeros_like(y_g)
        cross_ser = np.zeros_like(y_g)
        if np.any(~zero):
            (_, jb, _jv, _jpb, m_wp1, m_comb, m_dwpp) = self._saturated_series(
                y_g[~zero], phi_i[~zero]
            )
            # mgcv tweedious p-param 2nd derivatives (misc.c:500-501) from the
            # well-conditioned working accumulators: m_wp1 = E[∂logW/∂p],
            # m_comb = E[(∂logW/∂p)² + ∂²logW/∂p²], m_dwpp = E[∂logW/∂p·j/(1−p)
            # + ∂²logW/∂p∂logφ]. Combining (∂logW/∂p)²+∂²logW/∂p² PER TERM avoids
            # the ~1e-11 cancellation the old separate-moment split incurred.
            d2p_ser[~zero] = m_comb - m_wp1 ** 2
            cross_ser[~zero] = m_dwpp - (jb / om1) * m_wp1

        d2p = np.where(zero, 0.0, d2p_ser + d2p_dens)
        cross = np.where(zero, 0.0, cross_ser + cross_dens)
        return (float(np.sum(w_g * d2p)),
                float(np.sum(w_g * cross)))

    def __repr__(self):
        return f"Tweedie(p={self.p:.4g}, link={self.link.name})"


class tw(Tweedie):
    """Tweedie family with the power parameter ``p`` estimated jointly with
    the smoothing parameters — mgcv's ``tw()`` extended family.

    ``p`` is reparametrised through a scalar ``θ`` to keep the optimisation
    unconstrained:

        p(θ) = (a + b·exp(θ)) / (1 + exp(θ))    ⇒ p ∈ (a, b) as θ ∈ ℝ

    with default ``a = 1.01``, ``b = 1.99``. Initial p defaults to 1.5
    (mgcv's start) unless ``theta`` is passed (sets p = p(theta)).

    ``hea.gam`` estimates θ jointly with (ρ, log φ) in the analytical
    outer Newton (the family-generic Dd chain supplies the θ gradient
    and the analytic θ rows/cols of the REML Hessian — mgcv's gdi2
    ``D2``/``P2``/``ldet2`` blocks). The fitted ``p̂`` is stored on
    ``family.p``; the converged θ̂ on ``family.theta``.
    """
    name = "Tweedie"
    n_theta = 1

    # mgcv tw() okLinks (efam.r:3098-3101) — tw validates strictly,
    # UNLIKE fixed-p Tweedie() whose is.character fallback
    # (gam.fit3.r:3042-3045) accepts any make.link name (R-verified:
    # Tweedie(1.5, link="logit") constructs, tw(link="logit") errors).
    _OK_LINKS = ("log", "identity", "sqrt", "inverse")

    def __init__(self, theta: float | None = None, link=None,
                 a: float = 1.01, b: float = 1.99):
        if not (1.0 <= a < b <= 2.0):
            raise ValueError(
                f"tw() requires 1 ≤ a < b ≤ 2; got a={a!r}, b={b!r}"
            )
        self.a = float(a)
        self.b = float(b)
        if theta is None:
            # mgcv's tw() starts at p=1.5; θ such that p(θ)=1.5 is
            # θ = log((1.5 - a)/(b - 1.5)).
            p_init = 1.5
            theta_init = float(np.log((p_init - self.a) / (self.b - p_init)))
        else:
            theta_init = float(theta)
            p_init = self._p_of_theta(theta_init)
        self.theta = theta_init
        # Tweedie.__init__ validates 1 < p < 2 and sets p, link.
        super().__init__(p=p_init, link=link)
        if self.link.name not in self._OK_LINKS:
            raise ValueError(
                f'link "{self.link.name}" not available for tw family; '
                f'available links are {self._OK_LINKS}'
            )

    def _p_of_theta(self, theta: float) -> float:
        # mgcv's literal branch expressions (tw dev.resids/variance/Dd,
        # efam.r:3141-3149):
        #   p <- if (th>0) (b+a*exp(-th))/(1+exp(-th))
        #        else      (b*exp(th)+a)/(exp(th)+1)
        # An earlier expit-based rewrite ("sigmoid form for stability")
        # drifted p by an ulp, which the cancelling y^(2-p) − mu^(2-p)
        # deviance pieces amplified to ~4e-13 vs live R — the audit-2
        # B14 open follow-up. (The numpy-**-vs-libm-pow hypothesis
        # recorded there was WRONG: numpy ** ≡ math.pow ≡ libm pow,
        # 0/80k receipt.)
        th = float(theta)
        a, b = self.a, self.b
        if th > 0:
            e = math.exp(-th)
            return (b + a * e) / (1.0 + e)
        e = math.exp(th)
        return (b * e + a) / (e + 1.0)

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv tw dev.resids (efam.r tw:58-66): unlike the fixed-p
        # parent, a passed working θ is honored by mapping it to p —
        # estimate.theta probes the deviance at trial θ without
        # touching the family state. theta=None reads the state
        # (mgcv's get(".Theta")).
        if theta is None:
            p = self.p
        else:
            th = float(np.atleast_1d(np.asarray(theta, dtype=float))[0])
            p = self._p_of_theta(th)
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        # y1 <- y + (y==0); theta/kappa algebra with the p==1/p==2
        # branches structurally unreachable (1 < a ≤ p ≤ b < 2), and
        # mgcv's pmax(…, 0) clamp on rounding negatives.
        y1 = y + (y == 0.0)
        om1 = 1.0 - p
        tm = 2.0 - p
        t_term = (y1 ** om1 - mu ** om1) / om1
        kappa = (y ** tm - mu ** tm) / tm
        return np.maximum(2.0 * (y * t_term - kappa) * wt, 0.0)

    def dp_dtheta(self) -> float:
        """``dp/dθ`` — mgcv's ``dpth1`` literal branches (tw Dd,
        efam.r:3158-3159). Used by the outer Newton chain rule when
        joint-estimating θ_tw. R's ``^2`` is R_pow's ``x·x``."""
        th = float(self.theta)
        a, b = self.a, self.b
        if th > 0:
            e = math.exp(-th)
            d = 1.0 + e
            return e * (b - a) / (d * d)
        e = math.exp(th)
        d = e + 1.0
        return e * (b - a) / (d * d)

    def d2p_dtheta2(self) -> float:
        """``d²p/dθ²`` — mgcv's ``dpth2`` literal branches
        (efam.r:3160-3161); ``^3`` is R_pow's sequential ``x·x·x``."""
        th = float(self.theta)
        a, b = self.a, self.b
        if th > 0:
            e = math.exp(-th)
            d = e + 1.0
            return ((a - b) * e + (b - a) * math.exp(-2.0 * th)) / (d * d * d)
        e = math.exp(th)
        d = e + 1.0
        return ((a - b) * math.exp(2.0 * th) + (b - a) * e) / (d * d * d)

    def set_theta(self, theta) -> None:
        """Update θ (and the implied ``p``). Accepts a scalar or a 1-element
        array (consistent with the Family base ``n_theta``-array signature).
        """
        if hasattr(theta, "__len__"):
            if len(theta) != 1:
                raise ValueError(
                    f"tw expects a single theta; got length {len(theta)}"
                )
            theta = theta[0]
        self.theta = float(theta)
        self.p = self._p_of_theta(self.theta)

    def get_theta(self) -> np.ndarray:
        return np.array([self.theta], dtype=float)

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # tw postproc (efam.r:3237-3243): find.null.dev + "Tweedie(p=…)"
        # relabel with the fitted power rounded to 3 decimals.
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Tweedie(p={np.round(self.p, 3):g})",
        }

    def aic(self, y, mu, dev, wt, n, theta=None):
        # tw()$aic (efam.r:3211-3219): unlike the inherited Tweedie form,
        # the power comes from the θ ARGUMENT when one is given (gfam's
        # grouped aic passes each member its θ slice) — mgcv's ±θ-stable
        # expression verbatim; θ=None keeps the family's own power.
        if theta is None:
            p = None
        else:
            th = float(np.asarray(theta, dtype=float).reshape(-1)[0])
            a, b = self.a, self.b
            p = ((b + a * math.exp(-th)) / (1.0 + math.exp(-th)) if th > 0
                 else (b * math.exp(th) + a) / (math.exp(th) + 1.0))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        phi = max(float(dev) / max(n_eff, 1e-300), 1e-12)
        log_f = self._log_density(y, mu, phi, p=p)
        return -2.0 * float(np.sum(log_f * wt)) + 2.0

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        """mgcv ``tw()$ls`` in dict form (efam.r:3221-3230): saturated
        log-likelihood and its full first/second derivatives wrt the
        working parameters (θ, log φ) — ldTweedie's columns
        (1,4,2,5,6,3) summed with weight w:

            lsth1 = (LS₄, LS₂)
            lsth2 = [[LS₅, LS₆], [LS₆, LS₃]]

        The θ entries chain the p-space derivatives through p(θ):
        ∂/∂θ = (∂/∂p)·p′, ∂²/∂θ² = (∂²/∂p²)·p′² + (∂/∂p)·p″,
        ∂²/∂θ∂logφ = (∂²/∂p∂logφ)·p′ — exactly ldTweedie's work.param
        transform (gam.fit3.r:2808-2814). The p-space second
        derivatives come from :meth:`Tweedie._d2ls_dp` (family-review
        B4; previously NaN-poisoned).

        ``lsth2`` feeds the analytic θ rows/cols of the REML Hessian
        (gam.py ``_reml_hessian``): the ``−2·lsth2/γ`` ``ls2`` block for
        θ-θ and the ``−2·lsth2[θ,logφ]/γ`` cross for θ-log φ, matching
        mgcv's gam.fit4.r:746,757.
        """
        saved = None
        if theta is not None:
            th = np.asarray(theta, dtype=float).reshape(-1)
            # Skip the set/restore only on a bit-identical θ. An
            # approximate (allclose) skip evaluated the chain rule at a
            # θ up to ~1e-5 away from the one requested — Dd and
            # dev_resids honor the passed θ exactly, and that
            # g-vs-deviance inconsistency stalled estimate.theta's
            # Newton endgame (step halved to nothing at the optimum).
            if not np.array_equal(th, self.get_theta()):
                saved = self.get_theta().copy()
                self.set_theta(th)
        try:
            # Mechanical form (efam.r:3224-3229): ONE ldTweedie(y, y,
            # rho=log φ, theta=θ) evaluation, weighted, column-summed —
            # ls = ΣLs₁, lsth1 = (ΣLs₄, ΣLs₂), lsth2 = [[ΣLs₅, ΣLs₆],
            # [ΣLs₆, ΣLs₃]], LSTH1 = Ls[, c(4, 2)]. (An earlier detour
            # assembled the θ-blocks through dls_dp/_d2ls_dp — different
            # algebra, drifting lsth1/lsth2 up to ~1.5e-13 from R.)
            # R colSums accumulates left-to-right in double on arm64 —
            # `_rsum`, not pairwise np.sum.
            yv = np.asarray(y, dtype=float)
            wv = np.asarray(wt, dtype=float)
            ld = _ld_tweedie_work(
                yv, yv, np.full(yv.shape, self.theta),
                np.full(yv.shape, math.log(scale)), self.a, self.b)
            Ls = wv[:, None] * ld
            LS = np.array([_rsum(Ls[:, j]) for j in range(6)])
            lsth2 = np.array([[LS[4], LS[5]],
                              [LS[5], LS[2]]])
            return {
                "ls": float(LS[0]),
                "lsth1": np.array([LS[3], LS[1]]),
                "lsth2": lsth2,
                "LSTH1": Ls[:, [3, 1]],
            }
        finally:
            if saved is not None:
                self.set_theta(saved)

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        """Tweedie deviance derivatives wrt μ and θ — full port of mgcv
        tw()$Dd (efam.r:3155-3210). Level 0 feeds ``initial.spg``; level
        1's ``Dmuth`` feeds the family-θ column of ``db.drho``
        (∂β̂/∂θ for the Vc/edf2 sp-uncertainty correction)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        th = float(np.asarray(theta, dtype=float).ravel()[0])
        a, b = self.a, self.b
        # R-level ^2/^3 on these scalars is R_pow's sequential multiply
        # (|x| ≤ 11), NOT libm pow — pow(x,3.0) drifted dpth2 by an ulp
        # at θ>0, which every level-2 θ² row inherits.
        if th > 0:
            p = (b + a * np.exp(-th)) / (1 + np.exp(-th))
            d1 = 1 + np.exp(-th)
            dpth1 = np.exp(-th) * (b - a) / (d1 * d1)
            d2 = np.exp(-th) + 1
            dpth2 = (((a - b) * np.exp(-th) + (b - a) * np.exp(-2 * th))
                     / (d2 * d2 * d2))
        else:
            p = (b * np.exp(th) + a) / (np.exp(th) + 1)
            d1 = np.exp(th) + 1
            dpth1 = np.exp(th) * (b - a) / (d1 * d1)
            d2 = np.exp(th) + 1
            dpth2 = (((a - b) * np.exp(2 * th) + (b - a) * np.exp(th))
                     / (d2 * d2 * d2))
        mu1p = mu ** (1 - p)
        mup = mu ** p
        # mu**(-1-p) == mu**(-p-1) bit-for-bit (commutative exponent add); the
        # mgcv source recomputes it as `mup1` at level>0 — share the one pow.
        mupm1 = mu ** (-1 - p)
        r = {}
        ymupi = y / mup
        r["Dmu"] = 2 * wt * (mu1p - ymupi)
        r["Dmu2"] = 2 * wt * (mupm1 * p * y + (1 - p) / mup)
        r["EDmu2"] = (2 * wt) / mup
        if level > 0:
            i1p = 1 / (1 - p)
            y1 = y + (y == 0)
            logmu = np.log(mu)
            # Hoist sub-expressions the verbatim mgcv source recomputes (R's
            # `^`/`log` re-evaluate each time): y**(2-p), log(y1), their product,
            # and mu**(-p-1) — all byte-identical to the inline forms, just once.
            logy1 = np.log(y1)
            y2p = y ** (2 - p)
            y2plogy = y2p * logy1
            mu2p = mu * mu1p
            mup1 = mupm1
            # (2-p)^2 / i1p^2: R_pow x·x on the scalar (arithmetic.c:204).
            tmp2 = (2 - p) * (2 - p)
            i1p2 = i1p * i1p
            r["Dth"] = 2 * wt * (
                (y2plogy - mu2p * logmu) / (2 - p)
                + (y * mu1p * logmu - y2plogy) / (1 - p)
                - (y2p - mu2p) / tmp2
                + (y2p - y * mu1p) * i1p2
            ) * dpth1
            r["Dmuth"] = 2 * wt * logmu * (ymupi - mu1p) * dpth1
            r["Dmu3"] = -2 * wt * mup1 * p * (y / mu * (p + 1) + 1 - p)
            r["Dmu2th"] = 2 * wt * (
                mup1 * y * (1 - p * logmu) - (logmu * (1 - p) + 1) / mup
            ) * dpth1
            r["EDmu3"] = -2 * wt * p * mup1
            r["EDmu2th"] = -2 * wt * logmu / mup * dpth1
        if level > 1:
            logmu2 = logmu ** 2
            mup2 = mup1 / mu
            r["Dmu4"] = 2 * wt * mup2 * p * (p + 1) * (y * (p + 2) / mu + 1 - p)
            y2plog2y = y2plogy * logy1
            # R_pow scalars: (2-p)^2/^3, (1-p)^2/^3, dpth1^2 are all
            # sequential multiplies in R; and mgcv's parenthesization
            # multiplies the 6-term sum by dpth1² BEFORE 2·wt —
            # `2*wt*((…)*dpth1^2)` — the association order matters at
            # the last ulp.
            tm2 = (2 - p) * (2 - p)
            tm3 = tm2 * (2 - p)
            om2 = (1 - p) * (1 - p)
            om3 = om2 * (1 - p)
            dpth1_2 = dpth1 * dpth1
            r["Dth2"] = 2 * wt * ((
                (mu2p * logmu2 - y2plog2y) / (2 - p)
                + (y2plog2y - y * mu1p * logmu2) / (1 - p)
                + 2 * (y2plogy - mu2p * logmu) / tm2
                + 2 * (y * mu1p * logmu - y2plogy) / om2
                + 2 * (mu2p - y2p) / tm3
                + 2 * (y2p - y * mu1p) / om3
            ) * dpth1_2) + r["Dth"] * dpth2 / dpth1
            r["Dmuth2"] = (2 * wt * ((mu1p * logmu2
                                      - logmu2 * ymupi) * dpth1_2)
                           + r["Dmuth"] * dpth2 / dpth1)
            r["Dmu2th2"] = (2 * wt * ((mup1 * logmu * y * (logmu * p - 2)
                            + logmu / mup * (logmu * (1 - p) + 2)) * dpth1_2)
                            + r["Dmu2th"] * dpth2 / dpth1)
            r["Dmu3th"] = 2 * wt * mup1 * (
                y / mu * (logmu * (1 + p) * p - p - p - 1)
                + logmu * (1 - p) * p + p - 1 + p
            ) * dpth1
        return r

    def __repr__(self):
        return (f"tw(p={self.p:.4g}, link={self.link.name}, "
                f"a={self.a!r}, b={self.b!r})")


# ---------------------------------------------------------------------------
# Scaled-t — mgcv's ``scat()`` extended family
# ---------------------------------------------------------------------------


class Scat(Family):
    """Scaled-t extended family — direct port of mgcv ``scat()``
    (efam.r:3552-3768).

    Likelihood (with location ``μ``, scale ``σ``, dof ``ν``):

        f(y | μ, ν, σ) ∝ σ⁻¹ · (1 + ((y-μ)/σ)² / ν)^{-(ν+1)/2}

    Parameters ν and σ are estimated jointly with the smoothing
    parameters (mgcv ``estimate.theta``). Internally stored in log-form
    with a lower-bound shift on ν:

        θ₀ = log(ν − min_df)        ⇒  ν = exp(θ₀) + min_df > min_df
        θ₁ = log(σ)                  ⇒  σ = exp(θ₁) > 0

    ``min_df`` (default 3) prevents degenerate ν → 2 where the variance
    blows up. Set higher when the data clearly aren't very heavy-tailed.

    Default link ``identity``; ``log`` and ``inverse`` are also accepted
    (mgcv ``okLinks``).
    """
    name = "scat"
    canonical_link_name = "identity"
    _newton_canonical = "none"  # efam.r:2641 (canonical=""); extended
                                # path is always full Newton anyway.
    # mgcv treats scat as a fixed-scale family (``family$scale = 1``):
    # σ is in θ, not in φ. The bam/gam outer Newton therefore has no
    # log-φ slot for scat.
    scale_known = True
    is_extended = True
    n_theta = 2

    _OK_LINKS = ("identity", "log", "inverse")

    def __init__(self, theta=None, link: str = "identity",
                 min_df: float = 3.0):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for scat family; available '
                f'links are {self._OK_LINKS}'
            )
        # Match mgcv's ``min.df`` clamp + theta-sign decoding (efam.r:3576-3587):
        # * theta=None  → free θ, log-internal start (-2, -1)  → (ν=min_df+e⁻², σ=e⁻¹)
        # * theta given, all positive → fixed θ, n_theta=0
        # * theta given, any negative → free θ at |theta| as start
        # * if |theta[0]| ≤ min_df, lower min_df to 0.9·|theta[0]| with a warning.
        n_theta = 2
        if theta is not None and not np.any(np.asarray(theta) == 0.0):
            t = np.asarray(theta, dtype=float)
            if t.shape != (2,):
                raise ValueError(
                    f"scat theta must be a length-2 array (ν, σ); got "
                    f"shape {t.shape}"
                )
            if abs(t[0]) <= min_df:
                import warnings
                min_df = 0.9 * abs(t[0])
                warnings.warn(
                    "Supplied df below min.df. min.df reset",
                    stacklevel=2,
                )
            if np.any(t < 0):
                ini = np.array([np.log(abs(t[0]) - min_df),
                                np.log(abs(t[1]))], dtype=float)
            else:
                ini = np.array([np.log(t[0] - min_df),
                                np.log(t[1])], dtype=float)
                n_theta = 0
        else:
            ini = np.array([-2.0, -1.0], dtype=float)
        # Apply the actual instance settings.
        self.n_theta = int(n_theta)
        self.estimate_theta_callback = bool(n_theta > 0)
        self._min_df = float(min_df)
        self._theta = ini.copy()
        super().__init__(link=link)

    # ----- θ accessors (mgcv getTheta/putTheta) -------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float)
        if v.shape != (2,):
            raise ValueError(
                f"Scat.set_theta expects length-2 array (log θ); got "
                f"shape {v.shape}"
            )
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        """Return current θ. ``trans=True`` returns ``(ν, σ)`` on the
        original scale; ``trans=False`` returns the log-internal storage.
        Mirrors mgcv ``getTheta(trans=)``.
        """
        if trans:
            out = np.exp(self._theta).copy()
            out[0] += self._min_df
            return out
        return self._theta.copy()

    @property
    def min_df(self) -> float:
        return self._min_df

    # ----- variance / dev_resids / aic / ls -----------------------------

    def variance(self, mu):
        # Marginal var of σ·T(ν): σ²·ν/(ν-2). Used for sp init / Pearson.
        nu = np.float64(np.exp(self._theta[0]) + self._min_df)
        sig = np.float64(np.exp(self._theta[1]))
        return np.full(np.shape(mu), sig * sig * nu / max(nu - 2.0, 1e-10),
                       dtype=float)

    def dvar(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def d2var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: wt * (ν+1) * log1p((1/ν) * ((y-μ)/σ)²)  (efam.r:3609-3614)
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (nu + 1.0) * np.log1p((1.0 / nu) * ((y - mu) / sig) ** 2)

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(np.isnan(y)):
            raise ValueError("NA values not allowed for the scaled t family")
        # mgcv: mustart <- y + (y == 0) * 0.1   (efam.r:3736-3740)
        return y + (y == 0.0).astype(float) * 0.1

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(mu)))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv: -2·logL = 2·Σ wt·[ -lgamma((ν+1)/2) + lgamma(ν/2)
        #                          + log(σ·sqrt(πν))
        #                          + (ν+1)·log1p(((y-μ)/σ)²/ν)/2 ]
        # (efam.r:3690-3697)
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = (-gammaln((nu + 1.0) / 2.0)
                + gammaln(nu / 2.0)
                + np.log(sig * np.sqrt(np.pi * nu))
                + (nu + 1.0) * np.log1p(((y - mu) / sig) ** 2 / nu) / 2.0)
        return 2.0 * float(np.sum(term * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        """Saturated log-likelihood and θ-derivatives — mgcv ``ls`` for
        scat (efam.r:3699-3723). Returns a dict matching mgcv's shape:

            ls    : scalar saturated log-lik, Σᵢ wᵢ · ls_i(θ)
            lsth1 : (2,)   first derivatives wrt θ summed over i
            LSTH1 : (n,2)  per-obs first-derivative matrix
            lsth2 : (2,2)  Hessian wrt θ

        Used by ``_estimate_theta`` (Phase D). The base
        ``Family.ls(y, wt, scale)`` 3-vector signature is preserved for
        the standard families; extended-family callers test
        ``family.is_extended`` and dispatch here.
        """
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(wt, dtype=float)
        if w.size == 1:
            w = np.full(y.shape, float(w))
        nu = np.float64(np.exp(th[0]) + self._min_df)
        sig = np.float64(np.exp(th[1]))
        nu2 = nu - self._min_df       # = exp(th[0])
        nu2nu = nu2 / nu
        nu12 = (nu + 1.0) / 2.0
        # ls_i = lgamma((ν+1)/2) - lgamma(ν/2) - log(σ·sqrt(π·ν))
        term0 = (gammaln(nu12) - gammaln(nu / 2.0)
                 - np.log(sig * np.sqrt(np.pi * nu)))
        ls0 = float(np.sum(term0 * w))
        # First derivatives (per-obs, then summed):
        #   ∂ls/∂θ₀ per-obs = nu2 · ψ((ν+1)/2)/2 − nu2 · ψ(ν/2)/2 − 0.5·nu2nu
        #   ∂ls/∂θ₁ per-obs = -1   (constant)
        col0 = nu2 * digamma(nu12) / 2.0 - nu2 * digamma(nu / 2.0) / 2.0 \
            - 0.5 * nu2nu
        LSTH = np.column_stack([w * col0, -1.0 * w])
        lsth = LSTH.sum(axis=0)
        # Hessian (only [1,1] is nonzero per mgcv's ls):
        #   ∂²ls/∂θ₀² per-obs = nu2² · ψ′((ν+1)/2)/4 + nu2 · ψ((ν+1)/2)/2
        #                       − nu2² · ψ′(ν/2)/4 − nu2 · ψ(ν/2)/2
        #                       + 0.5·nu2nu² − 0.5·nu2nu
        d11 = (nu2 * nu2 * polygamma(1, nu12) / 4.0
               + nu2 * digamma(nu12) / 2.0
               - nu2 * nu2 * polygamma(1, nu / 2.0) / 4.0
               - nu2 * digamma(nu / 2.0) / 2.0
               + 0.5 * nu2nu * nu2nu - 0.5 * nu2nu)
        lsth2 = np.zeros((2, 2), dtype=float)
        lsth2[0, 0] = float(np.sum(d11 * w))
        return {"ls": ls0, "lsth1": lsth, "LSTH1": LSTH, "lsth2": lsth2}

    def ls(self, y, wt, scale):
        """Standard 3-vector ``ls`` contract: ``(ls0, d/dlogφ, d²/dlogφ²)``.

        Scat is ``scale_known = True`` — σ lives in θ, not φ — so the
        log-φ derivatives are identically zero, mirroring Poisson and
        Binomial. ``ls0`` is the saturated log-lik at μ=y under the
        current internal θ:

            ls0 = Σᵢ wᵢ · [lgamma((ν+1)/2) − lgamma(ν/2) − log(σ·√(πν))]

        The (y-μ)²/(σ²ν) term vanishes at μ=y so the saturated form
        carries only the normalising constants. ``_estimate_theta``
        (Phase D) reads the richer θ-derivative shape via
        :meth:`ls_extended` instead.
        """
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        nu = np.float64(np.exp(self._theta[0]) + self._min_df)
        sig = np.float64(np.exp(self._theta[1]))
        term = (gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - np.log(sig * np.sqrt(np.pi * nu)))
        ls0 = float(np.sum(term * wt))
        return np.array([ls0, 0.0, 0.0], dtype=float)

    # ----- Dd: μ- and θ-derivatives of −logL  (mgcv efam.r:3616-3687) ---

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        # Direct line-by-line port of mgcv ``scat$Dd``. Every variable
        # name and bracketing matches the source so future diffs against
        # mgcv stay mechanical.
        #
        # Note: nu/sig are kept as ``np.float64`` (not Python ``float``)
        # so divisions by zero in the σ→0 / ν→∞ extremes propagate as
        # ``inf``/``nan`` instead of raising ``ZeroDivisionError``. The
        # ``_estimate_theta`` Newton then sees a non-finite ``nll1`` and
        # step-halves naturally — mirroring mgcv R, which silently
        # produces ``Inf`` here.
        min_df = self._min_df
        th = np.asarray(theta, dtype=float)
        nu = np.float64(np.exp(th[0]) + min_df)
        sig = np.float64(np.exp(th[1]))
        nu1 = nu + 1.0
        nu2 = nu - min_df
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        w = np.asarray(wt, dtype=float)
        # mgcv broadcasts ``wt`` if scalar; when w is scalar, multiply
        # against length-n arrays via numpy broadcasting (works as-is).
        ym = y - mu
        a = 1.0 + (ym / sig) ** 2 / nu
        nu1ym = nu1 * ym
        sig2a = sig * sig * a
        nusig2a = nu * sig2a
        f = nu1ym / nusig2a
        f1 = ym / nusig2a
        n = y.shape[0]

        oo: dict = {}
        oo["Dmu"] = -2.0 * w * f
        oo["Dmu2"] = 2.0 * w * nu1 * (1.0 / nusig2a - 2.0 * f1 ** 2)
        # E[Dmu2] is the Fisher information per-obs at expected (y-μ)²:
        # 2·(ν+1) / (σ²·(ν+3)). Vectorised to length n.
        EDmu2_scalar = 2.0 * nu1 / (sig * sig) / (nu + 3.0)
        oo["EDmu2"] = np.full(n, EDmu2_scalar, dtype=float)

        if level > 0:
            nu1nusig2a = nu1 / nusig2a
            nu2nu = nu2 / nu
            fym = f * ym
            ff1 = f * f1
            f1ym = f1 * ym
            fymf1 = fym * f1
            ymsig2a = ym / sig2a

            Dth = np.zeros((n, 2), dtype=float)
            Dmuth = np.zeros((n, 2), dtype=float)
            Dmu2th = np.zeros((n, 2), dtype=float)
            EDmu2th = np.zeros((n, 2), dtype=float)
            Dth[:, 0] = w * nu2 * (np.log(a) - fym / nu)
            Dth[:, 1] = -2.0 * w * fym
            Dmuth[:, 0] = 2.0 * w * (f - ymsig2a - fymf1) * nu2nu
            Dmuth[:, 1] = 4.0 * w * f * (1.0 - f1ym)
            Dmu3 = 4.0 * w * f * (3.0 / nusig2a - 4.0 * f1 ** 2)
            Dmu2th[:, 0] = 2.0 * w * (
                -nu1nusig2a + 1.0 / sig2a + 5.0 * ff1
                - 2.0 * f1ym / sig2a - 4.0 * fymf1 * f1
            ) * nu2nu
            Dmu2th[:, 1] = 4.0 * w * (
                -nu1nusig2a + ff1 * 5.0 - 4.0 * ff1 * f1ym
            )
            EDmu3 = np.zeros(n, dtype=float)
            EDmu2th[:, 0] = (4.0 / (sig * sig * (nu + 3.0) ** 2)
                             * np.float64(np.exp(th[0])))
            EDmu2th[:, 1] = -2.0 * oo["EDmu2"]

            oo["Dth"] = Dth
            oo["Dmuth"] = Dmuth
            oo["Dmu3"] = Dmu3
            oo["Dmu2th"] = Dmu2th
            oo["EDmu3"] = EDmu3
            oo["EDmu2th"] = EDmu2th

        if level > 1:
            nu1nu = nu1 / nu
            fymf1ym = fym * f1ym
            f1ymf1 = f1ym * f1

            Dmu4 = 12.0 * w * (
                -nu1nusig2a / nusig2a + 8.0 * ff1 / nusig2a
                - 8.0 * ff1 * f1 ** 2
            )
            n2d = 3
            Dmu3th = np.zeros((n, 2), dtype=float)
            Dmu2th2 = np.zeros((n, n2d), dtype=float)
            Dmuth2 = np.zeros((n, n2d), dtype=float)
            Dth2 = np.zeros((n, n2d), dtype=float)

            Dmu3th[:, 0] = 4.0 * w * (
                -6.0 * f / nusig2a + 3.0 * f1 / sig2a
                + 18.0 * ff1 * f1 - 4.0 * f1ymf1 / sig2a
                - 12.0 * nu1ym * f1 ** 4
            ) * nu2nu
            Dmu3th[:, 1] = 48.0 * w * f * (
                -1.0 / nusig2a + 3.0 * f1 ** 2 - 2.0 * f1ymf1 * f1
            )

            Dth2[:, 0] = w * (
                nu2 * np.log(a)
                + nu2nu * ym ** 2
                * (-2.0 * nu2 - nu1 + 2.0 * nu1 * nu2nu
                   - nu1 * nu2nu * f1ym) / nusig2a
            )
            Dth2[:, 1] = 2.0 * w * (fym - ym * ymsig2a - fymf1ym) * nu2nu
            Dth2[:, 2] = 4.0 * w * fym * (1.0 - f1ym)

            term_a = 2.0 * nu2nu - 2.0 * nu1nu * nu2nu - 1.0 + nu1nu
            Dmuth2[:, 0] = 2.0 * w * f1 * nu2 * (
                term_a - 2.0 * nu2nu * f1ym + 4.0 * fym * nu2nu / nu
                - fym / nu - 2.0 * fymf1ym * nu2nu / nu
            )
            Dmuth2[:, 1] = 4.0 * w * (
                -f + ymsig2a + 3.0 * fymf1
                - ymsig2a * f1ym - 2.0 * fymf1 * f1ym
            ) * nu2nu
            Dmuth2[:, 2] = 8.0 * w * f * (-1.0 + 3.0 * f1ym - 2.0 * f1ym ** 2)

            Dmu2th2[:, 0] = 2.0 * w * nu2 * (
                -term_a + 10.0 * nu2nu * f1ym - 16.0 * fym * nu2nu / nu
                - 2.0 * f1ym + 5.0 * nu1nu * f1ym
                - 8.0 * nu2nu * f1ym ** 2
                + 26.0 * fymf1ym * nu2nu / nu
                - 4.0 * nu1nu * f1ym ** 2
                - 12.0 * nu1nu * nu2nu * f1ym ** 3
            ) / nusig2a
            Dmu2th2[:, 1] = 4.0 * w * (
                nu1nusig2a - 1.0 / sig2a - 11.0 * nu1 * f1 ** 2
                + 5.0 * f1ym / sig2a + 22.0 * nu1 * f1ymf1 * f1
                - 4.0 * f1ym ** 2 / sig2a - 12.0 * nu1 * f1ymf1 ** 2
            ) * nu2nu
            Dmu2th2[:, 2] = 8.0 * w * (
                nu1nusig2a - 11.0 * nu1 * f1 ** 2
                + 22.0 * nu1 * f1ymf1 * f1 - 12.0 * nu1 * f1ymf1 ** 2
            )

            oo["Dmu4"] = Dmu4
            oo["Dmu3th"] = Dmu3th
            oo["Dmu2th2"] = Dmu2th2
            oo["Dmuth2"] = Dmuth2
            oo["Dth2"] = Dth2

        return oo

    # ----- preinitialize / postproc / rd  (mgcv efam.r:3725-3757) -------

    def preinitialize(self, y) -> dict | None:
        # mgcv: when n.theta > 0, start with moderate ν and high σ:
        #   Theta <- c(1.5, log(0.8 * sd(y)))  (efam.r:3725-3734)
        # When all θ are user-fixed (n_theta = 0), no override.
        if self.n_theta > 0:
            y = np.asarray(y, dtype=float)
            sd_y = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
            sd_y = max(sd_y, 1e-10)  # guard against constant y
            return {"Theta": np.array([1.5, np.log(0.8 * sd_y)],
                                      dtype=float)}
        return None

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # scat postproc (efam.r:3742-3749): find.null.dev null deviance
        # + "Scaled t(ν,σ)" relabel, θ rounded to 3 decimals, ν > 999
        # reported as Inf.
        nu, sig = self.get_theta(trans=True)
        nu_disp = float(np.round(nu, 3))
        sig_disp = float(np.round(sig, 3))
        if nu_disp > 999.0:
            nu_disp_str = "Inf"
        else:
            nu_disp_str = f"{nu_disp:g}"
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Scaled t({nu_disp_str},{sig_disp:g})",
        }

    def rd(self, rng, mu, wt, scale):
        nu, sig = self.get_theta(trans=True)
        n = np.asarray(mu, dtype=float).shape[0]
        return rng.standard_t(nu, size=n) * sig + np.asarray(mu, dtype=float)

    def __repr__(self):
        nu, sig = self.get_theta(trans=True)
        return (f"Scat(theta=({nu:.4g}, {sig:.4g}), "
                f"link={self.link.name}, min_df={self._min_df:g})")


class nb(Family):
    """Negative binomial extended family — direct port of mgcv ``nb()``
    (efam.r:161-306).

    ``Var(y) = μ + μ²/Θ`` with the size parameter Θ estimated jointly
    with the smoothing parameters (θ = log Θ internally; scale fixed
    at 1 like Poisson).

    Constructor ``theta`` follows mgcv's sign convention:
    ``None``/``0`` → free θ starting at Θ=1; ``theta > 0`` → Θ fixed
    (``n_theta = 0``); ``theta < 0`` → free θ starting at ``|theta|``.
    Links: log (default), identity, sqrt.
    """
    name = "negative binomial"
    canonical_link_name = "log"
    _newton_canonical = "none"  # extended family: no Fisher shortcut.
    scale_known = True
    is_extended = True
    n_theta = 1

    _OK_LINKS = ("log", "identity", "sqrt")

    def __init__(self, theta: float | None = None, link: str = "log"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for nb family; available '
                f'links are {self._OK_LINKS}'
            )
        n_theta = 1
        if theta is not None and theta != 0.0:
            if theta > 0:
                ini = float(np.log(theta))
                n_theta = 0
            else:
                ini = float(np.log(-theta))
        else:
            ini = 0.0
        self.n_theta = int(n_theta)
        self._theta = np.array([ini], dtype=float)
        super().__init__(link=link)

    # ----- θ accessors ---------------------------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape != (1,):
            raise ValueError(
                f"nb.set_theta expects a single log Θ; got shape {v.shape}"
            )
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        if trans:
            return np.exp(self._theta).copy()
        return self._theta.copy()

    # ----- variance ------------------------------------------------------

    def variance(self, mu):
        Th = float(np.exp(self._theta[0]))
        mu = np.asarray(mu, dtype=float)
        return mu + mu * mu / Th

    def dvar(self, mu):
        Th = float(np.exp(self._theta[0]))
        return 1.0 + 2.0 * np.asarray(mu, dtype=float) / Th

    def d2var(self, mu):
        Th = float(np.exp(self._theta[0]))
        return np.full_like(np.asarray(mu, dtype=float), 2.0 / Th)

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    # ----- deviance / likelihood ----------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv (efam.r:199-205): 2·wt·[y·log(max(1,y)/μ)
        #                              − (y+Θ)·log((y+Θ)/(μ+Θ))]
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        Th = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return 2.0 * wt * (
            y * np.log(np.maximum(1.0, y) / mu)
            - (y + Th) * np.log((y + Th) / (mu + Th))
        )

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        # mgcv nb()$Dd verbatim (efam.r:207-237); θ = log Θ supplied.
        Th = float(np.exp(np.asarray(theta, dtype=float).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        yth = y + Th
        muth = mu + Th
        r = {}
        r["Dmu"] = 2.0 * wt * (yth / muth - y / mu)
        r["Dmu2"] = -2.0 * wt * (yth / muth ** 2 - y / mu ** 2)
        r["EDmu2"] = 2.0 * wt * (1.0 / mu - 1.0 / muth)
        if level > 0:
            r["Dth"] = -2.0 * wt * Th * (np.log(yth / muth)
                                         + (1.0 - yth / muth))
            r["Dmuth"] = 2.0 * wt * Th * (1.0 - yth / muth) / muth
            # mu^3/mu^4 are R_pow: sequential multiplies for |x|≤11,
            # libm pow above — numpy ** takes the pow loop everywhere
            # and drifts the last ulp (`_rpow_int` is the port).
            r["Dmu3"] = 4.0 * wt * (yth / _rpow_int(muth, 3)
                                    - y / _rpow_int(mu, 3))
            r["Dmu2th"] = 2.0 * wt * Th * (2.0 * yth / muth - 1.0) / muth ** 2
            r["EDmu2th"] = 2.0 * wt / muth ** 2
        if level > 1:
            r["Dmu4"] = 2.0 * wt * (6.0 * y / _rpow_int(mu, 4)
                                    - 6.0 * yth / _rpow_int(muth, 4))
            r["Dth2"] = -2.0 * wt * Th * (
                np.log(yth / muth) + Th * yth / muth ** 2 - yth / muth
                - 2.0 * Th / muth + 1.0 + Th / yth
            )
            r["Dmuth2"] = 2.0 * wt * Th * (
                2.0 * Th * yth / muth ** 2 - yth / muth
                - 2.0 * Th / muth + 1.0
            ) / muth
            r["Dmu2th2"] = 2.0 * wt * Th * (
                -6.0 * yth * Th / muth ** 2 + 2.0 * yth / muth
                + 4.0 * Th / muth - 1.0
            ) / muth ** 2
            r["Dmu3th"] = (4.0 * wt * Th * (1.0 - 3.0 * yth / muth)
                           / _rpow_int(muth, 3))
        return r

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv nb()$aic (efam.r:239-246); `dev` is unused (Θ-form direct).
        # R-level lgamma is nmath lgammafn (see ls_extended note).
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        Th = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = ((y + Th) * np.log(mu + Th) - y * np.log(mu)
                + _lgammafn_arr(y + 1.0) - Th * np.log(Th)
                + _nmath._lgammafn(Th)
                - _lgammafn_arr(Th + y))
        return 2.0 * float(np.sum(term * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        # mgcv nb()$ls (efam.r:248-275). scale is fixed at 1, so lsth1 is
        # the single θ derivative (no scale slot).
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        th0 = float(np.asarray(th).reshape(-1)[0])
        Th = float(np.exp(th0))
        y = np.asarray(y, dtype=float)
        w = np.asarray(wt, dtype=float)
        # The REML grad/Hessian/score each evaluate this saturated-likelihood
        # at the SAME (θ, scale) within an outer step (profiled 17 calls / 5
        # distinct = 71% redundant). It's a pure function of (y, wt, θ); memoise
        # like Tweedie._saturated_series (bit-identical — callers only read).
        cache = getattr(self, "_ls_cache", None)
        if cache is None:
            cache = self._ls_cache = {}
        key = (th0, float(scale), y.size,
               float(y[0]) if y.size else 0.0,
               float(y[-1]) if y.size else 0.0,
               float(y.sum()), float(w.sum()) if w.size else 0.0)
        hit = cache.get(key)
        if hit is not None:
            return hit
        # R-level lgamma/digamma are nmath lgammafn/dpsifn — scipy's
        # gammaln/digamma drift the last ulp (the nb rows of the audit-2
        # B14 census residue).
        ylogy = np.where(y > 0, y * np.log(np.maximum(y, 1e-300)), 0.0)
        term = ((y + Th) * np.log(y + Th) - ylogy
                + _lgammafn_arr(y + 1.0) - Th * np.log(Th)
                + _nmath._lgammafn(Th)
                - _lgammafn_arr(Th + y))
        # R sum() accumulates left-to-right in double on arm64 — _rsum.
        ls0 = -float(_rsum(term * w))
        yth = y + Th
        lyth = np.log(yth)
        psi0_yth = _nmath.psigamma_vec(yth, 0.0)
        psi0_th = _nmath.psigamma5(Th, 0.0)
        term1 = Th * (lyth - psi0_yth + psi0_th - th0)
        LSTH = (-term1 * w)[:, None]
        lsth = float(_rsum(LSTH.ravel()))
        psi1_yth = _polygamma(1, yth)
        psi1_th = _polygamma(1, Th)
        term2 = Th * (lyth - Th * psi1_yth - psi0_yth + Th / yth
                      + Th * psi1_th + psi0_th - th0 - 1.0)
        lsth2 = -float(_rsum(term2 * w))
        res = {
            "ls": ls0,
            "lsth1": np.array([lsth]),
            "lsth2": np.array([[lsth2]]),
            "LSTH1": LSTH,
        }
        if len(cache) >= 64:
            cache.clear()
        cache[key] = res
        return res

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the negative binomial "
                "family"
            )
        # mgcv: mustart <- y + (y == 0)/6
        return y + (y == 0.0) / 6.0

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # nb postproc (efam.r:283-289): find.null.dev + "Negative
        # Binomial(Θ)" relabel, Θ rounded to 3 decimals.
        Th = float(self.get_theta(trans=True)[0])
        return {
            "null_deviance": find_null_dev(
                self, y, eta=linear_predictors, offset=offset,
                weights=prior_weights,
            ),
            "family_name": f"Negative Binomial({np.round(Th, 3):g})",
        }

    def rd(self, rng, mu, wt, scale):
        Th = float(self.get_theta(trans=True)[0])
        mu = np.asarray(mu, dtype=float)
        # R's rnbinom(n, size=Θ, mu): per-element rpois(rgamma(Θ, μ/Θ)),
        # interleaved (gamma then poisson per draw, not block) so the draw
        # order matches R's stream — required for set.seed bit-exactness.
        out = np.empty(mu.shape[0])
        for i in range(mu.shape[0]):
            out[i] = rng.poisson(rng.gamma(shape=Th, scale=mu[i] / Th))
        return out

    def __repr__(self):
        Th = float(self.get_theta(trans=True)[0])
        return f"nb(theta={Th:.4g}, link={self.link.name})"


class negbin(Family):
    """Fixed-θ negative binomial — direct port of mgcv ``negbin()``
    (gam.fit3.r:2564-2642, "modified from Venables and Ripley's MASS
    library to work with gam.fit3").

    Unlike :class:`nb` (mgcv's extended family, θ estimated jointly with
    the smoothing parameters), ``negbin`` is a PLAIN exponential family:
    ``Var(y) = μ + μ²/θ`` at a fixed, user-supplied θ. Under ``gam``,
    estimate.gam forces φ = 1 whatever ``scale=`` says and turns
    GCV.Cp/GACV.Cp into UBRE (mgcv.r:1963-1966 + 1975-1979); under
    ``bam`` the scale is ESTIMATED (bam.r:2206 keys its known-scale
    list on famname ∈ {poisson, binomial} only — verified live:
    ``bam(negbin(2))`` reports ``scale.estimated=TRUE``).

    ``theta`` may be a vector (the legacy θ-range/θ-set search
    interface), but the only live mgcv path for ``len(θ) > 1`` is
    gam.outer's stop "Please provide a single value for theta or use nb
    to estimate it" (mgcv.r:1649-1650) — the search itself is
    deprecated.r-only (dead performance iteration). Every computation
    uses θ[0], mirroring mgcv's ``get(".Theta")[1]``.

    Links: any standard link name resolves — mgcv's ``negbin`` falls
    through to ``make.link(link)`` for character input
    (gam.fit3.r:2577-2579), so e.g. ``"inverse"`` is accepted despite
    the nominal okLinks of log/identity/sqrt (verified live).
    """
    name = "Negative Binomial"      # __init__ overrides with the θ-form
    canonical_link_name = "log"
    # canonical="" (gam.fit3.r:2641): never equal to the link name, so
    # PIRLS always takes the full-Newton branch (gam.fit3.r:118).
    _newton_canonical = "none"
    scale_known = True

    def __init__(self, theta=None, link: str = "log"):
        if theta is None:
            # mgcv: theta = stop("'theta' must be specified") — the lazy
            # default fires on first access (assign(".Theta", theta, env)).
            raise ValueError("'theta' must be specified")
        th = np.asarray(theta, dtype=float).reshape(-1)
        if th.size < 1 or not np.all(np.isfinite(th)) or np.any(th <= 0):
            raise ValueError(
                "negbin theta must be positive and finite; use nb() to "
                "estimate theta"
            )
        self._theta_all = th.copy()
        # famname: paste("Negative Binomial(", format(round(theta,3)), ")")
        # — a VECTOR in R for multi-θ; mgcv only ever reads family[1].
        self.name = f"Negative Binomial({np.round(th[0], 3):g})"
        super().__init__(link=link)

    @property
    def _th(self) -> float:
        """``get(".Theta")[1]`` — all computations use the first θ."""
        return float(self._theta_all[0])

    def get_theta(self) -> np.ndarray:
        """mgcv ``negbin()$getTheta()`` — the full θ vector on the
        NATURAL scale (unlike :class:`nb`, whose getTheta is log-scale)."""
        return self._theta_all.copy()

    # ----- variance (gam.fit3.r:2590-2595) -------------------------------

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float)
        return mu + mu * mu / self._th

    def dvar(self, mu):
        return 1.0 + 2.0 * np.asarray(mu, dtype=float) / self._th

    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 2.0 / self._th)

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    # ----- deviance / likelihood (gam.fit3.r:2599-2617) ------------------

    def dev_resids(self, y, mu, wt, theta=None):
        # 2·wt·[y·log(pmax(1,y)/μ) − (y+Θ)·log((y+Θ)/(μ+Θ))]
        Th = self._th
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return 2.0 * wt * (
            y * np.log(np.maximum(1.0, y) / mu)
            - (y + Th) * np.log((y + Th) / (mu + Th))
        )

    def aic(self, y, mu, dev, wt, n, theta=None):
        # (y+Θ)·log(μ+Θ) − y·log(μ) + lΓ(y+1) − Θ·log(Θ) + lΓ(Θ) − lΓ(Θ+y);
        # 2·Σ term·wt. ``dev`` is unused (Θ-form direct).
        Th = self._th
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = ((y + Th) * np.log(mu + Th) - y * np.log(mu)
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
        return 2.0 * float(np.sum(term * wt))

    def ls(self, y, wt, scale):
        # Saturated log-lik at μ=y; scale plays no role (φ ≡ 1), so both
        # log-φ derivatives are 0 — mgcv returns c(-sum(term*w), 0, 0).
        Th = self._th
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        # ylogy <- y; ylogy[y>0] <- y·log(y)  (y=0 row keeps 0).
        ylogy = np.where(y > 0, y * np.log(np.maximum(y, 1e-300)), y)
        term = ((y + Th) * np.log(y + Th) - ylogy
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
        return np.array([-float(np.sum(term * wt)), 0.0, 0.0])

    # ----- initialization / validity (gam.fit3.r:2597, 2618-2622) --------

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the negative binomial "
                "family"
            )
        # mustart <- y + (y == 0)/6
        return y + (y == 0.0) / 6.0

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    # ----- qq hooks (gam.fit3.r:2624-2632) --------------------------------

    def qf(self, p, mu, wt, scale):
        # qnbinom(p, size=Θ, mu=μ) — R dispatches the mu-parametrization.
        return _nmath._disp(
            "qnbinom_mu", _nmath.qnbinom_mu,
            [p, self._th, np.asarray(mu, dtype=float)], (True, False))

    def rd(self, rng, mu, wt, scale):
        # rnbinom(n=length(mu), size=Θ, mu=μ): per-element
        # rpois(rgamma(Θ, μ/Θ)), interleaved to match R's stream order.
        Th = self._th
        mu = np.asarray(mu, dtype=float)
        out = np.empty(mu.shape[0])
        for i in range(mu.shape[0]):
            out[i] = rng.poisson(rng.gamma(shape=Th, scale=mu[i] / Th))
        return out

    def __repr__(self):
        th = self._theta_all
        inner = (f"{th[0]:.4g}" if th.size == 1
                 else "[" + ", ".join(f"{t:.4g}" for t in th) + "]")
        return f"negbin(theta={inner}, link={self.link.name})"


class betar(Family):
    """Beta regression extended family — direct port of mgcv ``betar()``
    (efam.r:3269-3546).

    The response lies in (0, 1); ``μ`` is the mean and the single
    parameter ``θ`` (log-precision internally, ``φ = e^θ``) controls
    dispersion: ``Var(y) = μ(1−μ)/(1+φ)``. betar is mgcv's prototype for
    "−2logLik as deviance": :meth:`dev_resids` returns ``−2 logLik`` (the
    saturated term is omitted), ``ls`` is identically 0, and the true
    deviance (with its saturated reference) is assembled in
    :meth:`postproc` / :meth:`residuals` via the :meth:`saturated_ll`
    Newton solver. φ is estimated jointly with the smoothing parameters.

    Constructor ``theta`` follows mgcv's sign convention: ``None``/``0``
    → free θ starting at φ=1; ``theta > 0`` → φ fixed (``n_theta = 0``);
    ``theta < 0`` → free θ starting at ``|theta|``. Links: logit
    (default), probit, cloglog, cauchit.
    """
    name = "Beta regression"
    canonical_link_name = "logit"
    _newton_canonical = "none"  # extended family: no Fisher shortcut.
    scale_known = True
    is_extended = True
    n_theta = 1

    _OK_LINKS = ("logit", "probit", "cloglog", "cauchit")

    def __init__(self, theta: float | None = None, link: str = "logit",
                 eps: float | None = None):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for betar family; available '
                f'links are {self._OK_LINKS}'
            )
        n_theta = 1
        if theta is not None and theta != 0.0:
            if theta > 0:
                ini = float(np.log(theta))
                n_theta = 0
            else:
                ini = float(np.log(-theta))
        else:
            ini = 0.0
        self.n_theta = int(n_theta)
        self._theta = np.array([ini], dtype=float)
        # mgcv default eps = .Machine$double.eps*100
        self._eps = float(np.finfo(float).eps * 100) if eps is None \
            else float(eps)
        super().__init__(link=link)

    # ----- θ accessors ---------------------------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape != (1,):
            raise ValueError(
                f"betar.set_theta expects a single log φ; got shape "
                f"{v.shape}")
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        if trans:
            return np.exp(self._theta).copy()
        return self._theta.copy()

    # ----- variance ------------------------------------------------------

    def variance(self, mu):
        th = float(self._theta[0])
        mu = np.asarray(mu, dtype=float)
        return mu * (1.0 - mu) / (1.0 + np.exp(th))

    def dvar(self, mu):
        th = float(self._theta[0])
        return (1.0 - 2.0 * np.asarray(mu, dtype=float)) / (1.0 + np.exp(th))

    def d2var(self, mu):
        th = float(self._theta[0])
        return np.full_like(np.asarray(mu, dtype=float),
                            -2.0 / (1.0 + np.exp(th)))

    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    # ----- deviance (−2logLik) / Dd / aic --------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv betar dev.resids (efam.r:3316-3324): −2logLik per datum
        # (the saturated reference is added later via saturated_ll).
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        theta = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        muth = mu * theta
        return 2.0 * wt * (
            -gammaln(theta) + gammaln(muth) + gammaln(theta - muth)
            - muth * (np.log(y) - np.log1p(-y)) - theta * np.log1p(-y)
            + np.log(y) + np.log1p(-y))

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        # mgcv betar()$Dd verbatim (efam.r:3326-3367); θ = log φ supplied.
        theta = float(np.exp(np.asarray(theta, dtype=float).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        onemu = 1.0 - mu
        muth = mu * theta
        onemuth = onemu * theta
        psi0_th = digamma(theta)
        psi1_th = _polygamma(1, theta)
        psi0_muth = digamma(muth)
        psi0_onemuth = digamma(onemuth)
        psi1_muth = _polygamma(1, muth)
        psi1_onemuth = _polygamma(1, onemuth)
        psi2_muth = _polygamma(2, muth)
        psi2_onemuth = _polygamma(2, onemuth)
        psi3_muth = _polygamma(3, muth)
        psi3_onemuth = _polygamma(3, onemuth)
        log_yoney = np.log(y) - np.log1p(-y)
        r: dict = {}
        r["Dmu"] = 2.0 * wt * theta * (psi0_muth - psi0_onemuth - log_yoney)
        r["Dmu2"] = 2.0 * wt * theta ** 2 * (psi1_muth + psi1_onemuth)
        r["EDmu2"] = r["Dmu2"]
        if level > 0:
            r["Dth"] = 2.0 * wt * theta * (
                -mu * log_yoney - np.log1p(-y) + mu * psi0_muth
                + onemu * psi0_onemuth - psi0_th)
            r["Dmuth"] = r["Dmu"] + 2.0 * wt * theta ** 2 * (
                mu * psi1_muth - onemu * psi1_onemuth)
            r["Dmu3"] = 2.0 * wt * theta ** 3 * (psi2_muth - psi2_onemuth)
            r["Dmu2th"] = 2.0 * r["Dmu2"] + 2.0 * wt * theta ** 3 * (
                mu * psi2_muth + onemu * psi2_onemuth)
            r["EDmu2th"] = r["Dmu2th"]
        if level > 1:
            r["Dmu4"] = 2.0 * wt * theta ** 4 * (psi3_muth + psi3_onemuth)
            r["Dth2"] = r["Dth"] + 2.0 * wt * theta ** 2 * (
                mu ** 2 * psi1_muth + onemu ** 2 * psi1_onemuth - psi1_th)
            r["Dmuth2"] = r["Dmuth"] + 2.0 * wt * theta ** 2 * (
                mu ** 2 * theta * psi2_muth + 2.0 * mu * psi1_muth
                - theta * onemu ** 2 * psi2_onemuth
                - 2.0 * onemu * psi1_onemuth)
            r["Dmu2th2"] = 2.0 * r["Dmu2th"] + 2.0 * wt * theta ** 3 * (
                mu ** 2 * theta * psi3_muth + 3.0 * mu * psi2_muth
                + onemu ** 2 * theta * psi3_onemuth
                + 3.0 * onemu * psi2_onemuth)
            r["Dmu3th"] = 3.0 * r["Dmu3"] + 2.0 * wt * theta ** 4 * (
                mu * psi3_muth - onemu * psi3_onemuth)
        return r

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv betar()$aic (efam.r:3369-3376); `dev` unused.
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        theta = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        muth = mu * theta
        term = (-gammaln(theta) + gammaln(muth) + gammaln(theta - muth)
                - (muth - 1.0) * np.log(y)
                - (theta - muth - 1.0) * np.log1p(-y))
        return 2.0 * float(np.sum(term * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        # betar ls ≡ 0 (efam.r:3378-3385): deviance is −2logLik, the
        # saturated reference is folded in via saturated_ll instead.
        y = np.asarray(y, dtype=float)
        return {"ls": 0.0, "lsth1": np.array([0.0]),
                "lsth2": np.array([[0.0]]),
                "LSTH1": np.zeros((y.shape[0], 1))}

    def ls(self, y, wt, scale):
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # ----- saturated likelihood (Newton) ---------------------------------

    def saturated_ll(self, y, wt, theta):
        """Saturated log-lik by per-datum Newton — mgcv ``saturated.ll``
        (efam.r:3393-3462). ``theta`` is the precision φ (already
        exp'd). Returns ``{"f", "term", "mu"}``: the wt-summed saturated
        log-lik, the per-datum saturated log-lik, and the maximizing μ."""
        eps = self._eps
        y = np.asarray(y, dtype=float).copy()
        wt = np.asarray(wt, dtype=float)
        phi = float(theta)

        def gbh(yy, eta, deriv=False, a=1e-8):
            b = 1.0 - a
            ind = eta > 0
            expeta = np.where(ind, np.exp(-np.where(ind, eta, 0.0)),
                              np.exp(np.where(ind, 0.0, eta)))
            mu = np.where(ind, (a * expeta + b) / (1.0 + expeta),
                          (a + b * expeta) / (1.0 + expeta))
            la = phi * mu
            lb = phi * (1.0 - mu)
            ll = ((la - 1.0) * np.log(yy) + (lb - 1.0) * np.log1p(-yy)
                  - gammaln(la) - gammaln(lb) + gammaln(la + lb))
            g = h = None
            if deriv:
                g = (phi * np.log(yy) - phi * np.log1p(-yy)
                     - phi * digamma(mu * phi) + phi * digamma((1.0 - mu) * phi))
                h = -phi ** 2 * (_polygamma(1, mu * phi)
                                 + _polygamma(1, (1.0 - mu) * phi))
                dmueta1 = expeta * (b - a) / (1.0 + expeta) ** 2
                dmueta2 = (np.sign(eta) * ((a - b) * expeta
                           + (b - a) * expeta ** 2) / (expeta + 1.0) ** 3)
                h = h * dmueta1 ** 2 + g * dmueta2
                g = g * dmueta1
            return ll, g, h, mu

        n = y.shape[0]
        a = eps
        b = 1.0 - eps
        eta = y.copy()
        yc = y.copy()
        yc[yc < eps] = eps
        yc[yc > 1.0 - eps] = 1.0 - eps
        eta[yc <= eps * 1.2] = eps * 1.2
        eta[yc >= 1.0 - eps * 1.2] = 1.0 - eps * 1.2
        eta = np.log((eta - a) / (b - eta))
        yw = yc
        LS = np.zeros(n)
        muout = np.zeros(n)
        ii = np.arange(n)
        ls_l = ls_g = ls_h = None
        for _ in range(200):
            ls_l, ls_g, ls_h, ls_mu = gbh(yw, eta, True, a=eps / 10.0)
            conv = np.abs(ls_g) < np.mean(np.abs(ls_l) + 0.1) * 1e-8
            if np.sum(conv) > 0:
                LS[ii[conv]] = ls_l[conv]
                muout[ii[conv]] = ls_mu[conv]
                ii = ii[~conv]
                if ii.size > 0:
                    yw = yw[~conv]
                    eta = eta[~conv]
                    ls_l = ls_l[~conv]
                    ls_g = ls_g[~conv]
                    ls_h = ls_h[~conv]
                else:
                    break
            h = -ls_h
            if h.size:
                hmin = np.max(h) * 1e-4
                h[h < hmin] = hmin
            delta = ls_g / h
            big = np.abs(delta) > 2.0
            delta[big] = np.sign(delta[big]) * 2.0
            ls1_l = gbh(yw, eta + delta, False, a=eps / 10.0)[0]
            fail = ls1_l < ls_l
            k = 0
            while np.sum(fail) > 0 and k < 20:
                k += 1
                delta[fail] = delta[fail] / 2.0
                ls1_l[fail] = gbh(yw[fail], eta[fail] + delta[fail],
                                  False, a=eps / 10.0)[0]
                fail = ls1_l < ls_l
            eta = eta + delta
        if ii.size > 0:
            LS[ii] = ls_l
            import warnings
            warnings.warn("saturated likelihood may be inaccurate",
                          stacklevel=2)
        return {"f": float(np.sum(wt * LS)), "term": LS, "mu": muout}

    # ----- initialization / validity -------------------------------------

    def preinitialize(self, y) -> dict | None:
        # mgcv betar preinitialize (efam.r:3387-3391): clamp y into
        # (eps, 1−eps) so the log-lik is finite at the boundaries.
        eps = self._eps
        y = np.asarray(y, dtype=float).copy()
        y[y >= 1.0 - eps] = 1.0 - eps
        y[y <= eps] = eps
        return {"y": y}

    def initialize(self, y, wt):
        # mgcv: mustart <- y
        return np.asarray(y, dtype=float).copy()

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu)
        return bool(np.all(mu > 0.0) and np.all(mu < 1.0))

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # betar postproc (efam.r:3468-3486): the TRUE deviance and null
        # deviance fold in the saturated log-lik (2·f) on top of the
        # −2logLik dev_resids; relabel "Beta regression(φ)".
        theta = float(self.get_theta(trans=True)[0])
        y = np.asarray(y, dtype=float)
        wt = np.asarray(prior_weights, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        lf = self.saturated_ll(y, wt, theta)
        dev = 2.0 * lf["f"] + float(np.sum(self.dev_resids(y, fitted, wt)))
        if intercept:
            wtdmu = float(np.sum(wt * y) / np.sum(wt))
            mu_null = np.full(y.shape, wtdmu)
        else:
            mu_null = self.link.linkinv(np.asarray(offset, dtype=float))
        null_dev = 2.0 * lf["f"] + float(
            np.sum(self.dev_resids(y, mu_null, wt)))
        return {"deviance": dev, "null_deviance": null_dev,
                "family_name": f"Beta regression({np.round(theta, 3):g})"}

    def residuals_extended(self, y, mu, wt, type: str = "deviance"):
        """betar deviance residuals (efam.r:3493-3513): the saturated
        reference makes ``2·ls_term + dev_resids`` a proper (≥0) squared
        deviance residual, signed by ``sign(y−μ)``."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        if type == "response":
            return y - mu
        if type == "pearson":
            return (y - mu) / np.sqrt(self.variance(mu))
        if type != "deviance":
            raise ValueError(
                "betar residuals are 'deviance', 'response' or 'pearson'; "
                f"got {type!r}")
        theta = float(self.get_theta(trans=True)[0])
        lf = self.saturated_ll(y, wt, theta)
        res = 2.0 * lf["term"] + self.dev_resids(y, mu, wt)
        res = np.maximum(res, 0.0)
        return np.sqrt(res) * np.sign(y - mu)

    def rd(self, rng, mu, wt, scale):
        # mgcv betar rd (efam.r:3515-3523): Beta(φμ, φ(1−μ)) draws,
        # clamped into (eps, 1−eps).
        theta = float(self.get_theta(trans=True)[0])
        mu = np.asarray(mu, dtype=float)
        r = rng.beta(theta * mu, theta * (1.0 - mu))
        eps = self._eps
        r = np.where(r >= 1.0 - eps, 1.0 - eps, r)
        r = np.where(r < eps, eps, r)
        return r

    def qf(self, p, mu, wt, scale):
        # mgcv betar qf (efam.r:3525-3532): Beta quantile, clamped.
        from scipy.stats import beta as _beta
        theta = float(self.get_theta(trans=True)[0])
        mu = np.asarray(mu, dtype=float)
        q = _beta.ppf(p, theta * mu, theta * (1.0 - mu))
        eps = self._eps
        q = np.where(q >= 1.0 - eps, 1.0 - eps, q)
        q = np.where(q < eps, eps, q)
        return q

    def __repr__(self):
        phi = float(self.get_theta(trans=True)[0])
        return f"betar(theta={phi:.4g}, link={self.link.name})"


# ---------------------------------------------------------------------------
# Ordered categorical — mgcv ocat() (efam.r:2618-3081). The response is one
# of R ordered classes; a single latent variable μ (identity link, the only
# okLink) is split by R−1 cut points into class probabilities. The cut
# points are α = [−∞, −1, −1+cumsum(e^θ), +∞] with θ the n_theta = R−2 free
# log-step parameters. mgcv labels classes 1..R; hea is 0-based everywhere
# user-facing (and ``multinom`` already uses 0..K), so ``ocat`` exposes
# classes 0..R−1. The verbatim-transcribed Dd/dev/aic helpers below work in
# mgcv's 1-based convention (so they oracle-pin directly against mgcv); the
# ``ocat`` class converts 0↔1 only at the engine boundary.
# ---------------------------------------------------------------------------


def _ocat_alpha_full(theta: np.ndarray) -> tuple[np.ndarray, int]:
    """mgcv's ``alpha`` cut-point vector for Dd/dev.resids/aic: length R+1,
    ``alpha = [−∞, −1, −1+cumsum(e^θ), +∞]`` (0-based: alpha[0]=−∞,
    alpha[1]=−1, alpha[2:R]=…, alpha[R]=+∞). Returns (alpha, R)."""
    th = np.asarray(theta, dtype=float).reshape(-1)
    R = th.shape[0] + 2
    alpha = np.zeros(R + 1)
    alpha[0] = -np.inf
    alpha[R] = np.inf
    alpha[1] = -1.0
    if R > 2:
        alpha[2:R] = alpha[1] + np.cumsum(np.exp(th))
    return alpha, R


def _ocat_Fdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cancellation-resistant ``F(b) − F(a)`` for the logistic CDF F,
    with ``b > a`` (mgcv's inner ``Fdiff``, efam.r:2685-2696)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    hb = np.ones_like(b)
    hb[b > 0] = -1.0
    eb = np.exp(b * hb)
    ha = np.ones_like(a)
    ha[a > 0] = -1.0
    ea = np.exp(a * ha)
    out = np.empty_like(b)
    indb = b < 0
    out[indb] = eb[indb] / (1.0 + eb[indb]) - ea[indb] / (1.0 + ea[indb])
    inda = a > 0
    out[inda] = ((ea[inda] - eb[inda])
                 / ((ea[inda] + 1.0) * (eb[inda] + 1.0)))
    indm = (~indb) & (~inda)
    out[indm] = ((1.0 - ea[indm] * eb[indm])
                 / ((eb[indm] + 1.0) * (ea[indm] + 1.0)))
    return out


def _ocat_abcd(x: np.ndarray, level: int) -> tuple:
    """mgcv's ``abcd`` (efam.r:2736-2759): cancellation-resistant
    ``a_j = f_j²−f_j``, ``b_j = f_j−3f_j²+2f_j³``, ``c_j``, ``d_j`` for the
    logistic CDF, returned up to the requested level (None past it)."""
    x = np.asarray(x, dtype=float)
    h = np.ones_like(x)
    h[x > 0] = -1.0
    ex = np.exp(x * h)
    ex1 = ex + 1.0
    ex1k = ex1 ** 2
    aj = -ex / ex1k
    bj = cj = dj = None
    if level >= 0:
        ex1k = ex1k * ex1
        ex2 = ex ** 2
        bj = h * (ex - ex2) / ex1k
        if level > 0:
            ex1k = ex1k * ex1
            ex3 = ex2 * ex
            cj = (-ex3 + 4.0 * ex2 - ex) / ex1k
            if level > 1:
                ex1k = ex1k * ex1
                ex4 = ex3 * ex
                dj = h * (-ex4 + 11.0 * ex3 - 11.0 * ex2 + ex) / ex1k
    return aj, bj, cj, dj


def _ocat_Dd(y1, mu, theta, wt, level: int = 0) -> dict:
    """mgcv ``ocat()$Dd`` (efam.r:2721-2887) verbatim. ``y1`` is the class
    label in mgcv's 1-based convention (1..R)."""
    y1 = np.asarray(y1).astype(int)
    mu = np.asarray(mu, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    wt = np.ones_like(mu) if wt is None else np.asarray(wt, dtype=float)
    alpha, R = _ocat_alpha_full(theta)
    al1 = alpha[y1]
    al0 = alpha[y1 - 1]
    al1mu = al1 - mu
    al0mu = al0 - mu
    f = np.maximum(_ocat_Fdiff(al0mu, al1mu), np.finfo(float).tiny)
    a1, b1, c1, d1 = _ocat_abcd(al1mu, level)
    a0, b0, c0, d0 = _ocat_abcd(al0mu, level)
    a = a1 - a0
    if level >= 0:
        b = b1 - b0
    if level > 0:
        c = c1 - c0
    if level > 1:
        d = d1 - d0
    n = y1.shape[0]
    oo: dict = {}
    oo["D"] = -2.0 * wt * np.log(f)
    if level >= 0:
        oo["Dmu"] = -2.0 * wt * a / f
        a2 = a ** 2
        oo["Dmu2"] = oo["EDmu2"] = 2.0 * wt * (a2 / f - b) / f
    if R < 3:
        level = 0
    if level > 0:
        f2 = f ** 2
        a3 = a2 * a
        oo["Dmu3"] = 2.0 * wt * (-c - 2.0 * a3 / f2 + 3.0 * a * b / f) / f
        Dmua0 = 2.0 * (a0 * a / f - b0) / f
        Dmua1 = -2.0 * (a1 * a / f - b1) / f
        Dmu2a0 = -2.0 * (c0 + (a0 * (2.0 * a2 / f - b) - 2.0 * b0 * a) / f) / f
        Dmu2a1 = 2.0 * (c1 + (2.0 * (a1 * a2 / f - b1 * a) - a1 * b) / f) / f
        Da0 = -2.0 * a0 / f
        Da1 = 2.0 * a1 / f
        Dth = np.zeros((n, R - 2))
        Dmuth = np.zeros((n, R - 2))
        Dmu2th = np.zeros((n, R - 2))
        for kk in range(R - 2):
            etk = np.exp(theta[kk])
            ind = y1 == kk + 2
            Dth[ind, kk] = wt[ind] * Da1[ind] * etk
            Dmuth[ind, kk] = wt[ind] * Dmua1[ind] * etk
            Dmu2th[ind, kk] = wt[ind] * Dmu2a1[ind] * etk
            if R > kk + 3:
                ind = (y1 > kk + 2) & (y1 < R)
                Dth[ind, kk] = wt[ind] * (Da1[ind] + Da0[ind]) * etk
                Dmuth[ind, kk] = wt[ind] * (Dmua1[ind] + Dmua0[ind]) * etk
                Dmu2th[ind, kk] = wt[ind] * (Dmu2a1[ind] + Dmu2a0[ind]) * etk
            ind = y1 == R
            Dth[ind, kk] = wt[ind] * Da0[ind] * etk
            Dmuth[ind, kk] = wt[ind] * Dmua0[ind] * etk
            Dmu2th[ind, kk] = wt[ind] * Dmu2a0[ind] * etk
        oo["Dth"] = Dth
        oo["Dmuth"] = Dmuth
        oo["Dmu2th"] = oo["EDmu2th"] = Dmu2th
    if level > 1:
        oo["Dmu4"] = 2.0 * wt * ((3.0 * b ** 2 + 4.0 * a * c) / f
                                 + a2 * (6.0 * a2 / f - 12.0 * b) / f2 - d) / f
        Dmu3a0 = 2.0 * ((a0 * c + 3.0 * c0 * a + 3.0 * b0 * b) / f - d0
                        + 6.0 * a * (a0 * a2 / f - b0 * a - a0 * b) / f2) / f
        Dmu3a1 = 2.0 * (d1 - (a1 * c + 3.0 * (c1 * a + b1 * b)) / f
                        + 6.0 * a * (b1 * a - a1 * a2 / f + a1 * b) / f2) / f
        Dmua0a0 = 2.0 * (c0 + (2.0 * a0 * (b0 - a0 * a / f) - b0 * a) / f) / f
        Dmua1a1 = 2.0 * ((b1 * a + 2.0 * a1 * (b1 - a1 * a / f)) / f - c1) / f
        Dmua0a1 = 2.0 * (a0 * (2.0 * a1 * a / f - b1) - b0 * a1) / f2
        Dmu2a0a0 = 2.0 * (d0 + (b0 * (2.0 * b0 - b) + 2.0 * c0 * (a0 - a)) / f
                          + 2.0 * (b0 * a2 + a0 * (3.0 * a0 * a2 / f
                                                   - 4.0 * b0 * a
                                                   - a0 * b)) / f2) / f
        Dmu2a1a1 = 2.0 * ((2.0 * c1 * (a + a1) + b1 * (2.0 * b1 + b)) / f
                          + 2.0 * (a1 * (3.0 * a1 * a2 / f - a1 * b)
                                   - b1 * a * (a + 4.0 * a1)) / f2 - d1) / f
        Dmu2a0a1 = 0.0
        Da0a0 = 2.0 * (b0 + a0 ** 2 / f) / f
        Da1a1 = -2.0 * (b1 - a1 ** 2 / f) / f
        Da0a1 = -2.0 * a0 * a1 / f2
        n2d = (R - 2) * (R - 1) // 2
        Dmu3th = np.zeros((n, R - 2))
        Dth2 = np.zeros((n, n2d))
        Dmuth2 = np.zeros((n, n2d))
        Dmu2th2 = np.zeros((n, n2d))
        i = -1
        for jj in range(R - 2):
            for kk in range(jj, R - 2):
                i += 1
                ind = y1 >= jj + 1
                ar_k = np.full(n, np.exp(theta[kk]))
                ar1_k = ar_k.copy()
                ar_k[(y1 == R) | (y1 <= kk + 1)] = 0.0
                ar1_k[y1 < kk + 3] = 0.0
                ar_j = np.full(n, np.exp(theta[jj]))
                ar1_j = ar_j.copy()
                ar_j[(y1 == R) | (y1 <= jj + 1)] = 0.0
                ar1_j[y1 < jj + 3] = 0.0
                ar_kj = np.zeros(n)
                ar1_kj = np.zeros(n)
                if kk == jj:
                    ar_kj[(y1 > kk + 1) & (y1 < R)] = np.exp(theta[kk])
                    ar1_kj[y1 > kk + 2] = np.exp(theta[kk])
                    Dmu3th[ind, kk] = wt[ind] * (Dmu3a1[ind] * ar_k[ind]
                                                 + Dmu3a0[ind] * ar1_k[ind])
                Dth2[:, i] = wt * (Da1a1 * ar_k * ar_j + Da0a1 * ar_k * ar1_j
                                   + Da1 * ar_kj + Da0a0 * ar1_k * ar1_j
                                   + Da0a1 * ar1_k * ar_j + Da0 * ar1_kj)
                Dmuth2[:, i] = wt * (Dmua1a1 * ar_k * ar_j
                                     + Dmua0a1 * ar_k * ar1_j + Dmua1 * ar_kj
                                     + Dmua0a0 * ar1_k * ar1_j
                                     + Dmua0a1 * ar1_k * ar_j + Dmua0 * ar1_kj)
                Dmu2th2[:, i] = wt * (Dmu2a1a1 * ar_k * ar_j
                                      + Dmu2a0a1 * ar_k * ar1_j
                                      + Dmu2a1 * ar_kj
                                      + Dmu2a0a0 * ar1_k * ar1_j
                                      + Dmu2a0a1 * ar1_k * ar_j
                                      + Dmu2a0 * ar1_kj)
        oo["Dmu3th"] = Dmu3th
        oo["Dth2"] = Dth2
        oo["Dmuth2"] = Dmuth2
        oo["Dmu2th2"] = Dmu2th2
    return oo


def _ocat_dev_signed(y1, mu, wt, theta) -> tuple[np.ndarray, np.ndarray]:
    """ocat deviance residuals ``−2 wt log f`` plus the latent-midpoint
    sign (efam.r:2683-2719). ``y1`` 1-based."""
    y1 = np.asarray(y1).astype(int)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    alpha, R = _ocat_alpha_full(theta)
    al1 = alpha[y1]
    al0 = alpha[y1 - 1]
    s = np.sign((al1 + al0) / 2.0 - mu)
    f = _ocat_Fdiff(al0 - mu, al1 - mu)
    return -2.0 * wt * np.log(f), s


def _ocat_ini(R: int, y0) -> np.ndarray | None:
    """mgcv ``ocat.ini`` (efam.r:2927-2938): seed the R−2 log-step θ from
    the empirical cumulative class proportions. ``y0`` 0-based labels."""
    if R < 3:
        return None
    yy = np.concatenate([np.arange(1, R + 1),
                         np.asarray(y0).astype(int) + 1]).astype(float)
    yy = yy[np.isfinite(yy)].astype(int)
    counts = np.bincount(yy, minlength=R + 1)[1:R + 1].astype(float)
    p = np.cumsum(counts / yy.shape[0])
    eta = 5.0 if p[0] == 0 else -1.0 - np.log(p[0] / (1.0 - p[0]))
    theta = np.full(R - 1, -1.0)
    for i in range(1, R - 1):
        theta[i] = np.log(p[i] / (1.0 - p[i])) + eta
    theta = np.diff(theta)
    theta[theta <= 0.01] = 0.01
    return np.log(theta)


def _ocat_prob(theta, lp, se=None) -> tuple:
    """mgcv ``ocat.prob`` (efam.r:3002-3021): per-class probabilities (and
    optional delta-method SE) from the finite cut points ``theta`` (length
    R−1) and the latent linear predictor ``lp``."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    lp = np.asarray(lp, dtype=float)
    R = theta.shape[0]
    n = lp.shape[0]
    prob = np.zeros((n, R + 2))
    dp = np.zeros((n, R + 2))
    prob[:, R + 1] = 1.0
    for i in range(R):
        p = expit(theta[i] - lp)
        prob[:, i + 1] = p
        dp[:, i + 1] = p * (p - 1.0)
    prob = np.diff(prob, axis=1)
    dp = np.diff(dp, axis=1)
    if se is not None:
        se = np.asarray(se, dtype=float).reshape(-1, 1) * np.abs(dp)
    return prob, se


class ocat(Family):
    """Ordered categorical extended family — port of mgcv ``ocat()``
    (efam.r:2618-3081).

    The R ordered response classes (hea: 0..R−1, mgcv: 1..R) arise from a
    single latent variable with mean ``μ`` (identity link — the only
    okLink) split by R−1 cut points ``[−1, −1+cumsum(e^θ)]``. The
    ``n_theta = R−2`` free log-step parameters ``θ`` are estimated jointly
    with the smoothing parameters (the first cut point is fixed at −1 for
    identifiability). ``ls ≡ 0``; the deviance is the standard
    ``Σ −2 wt log f`` (no saturated fold, unlike :class:`betar`). Construct
    with ``ocat(R=k)`` (free θ) or ``ocat(theta=…)`` (mgcv's sign
    convention: positive → fixed, ``n_theta = 0``).
    """
    name = "Ordered Categorical"
    canonical_link_name = "identity"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    no_r_sq = True          # mgcv ocat sets no.r.sq=TRUE (efam.r:3080)
    _OK_LINKS = ("identity",)

    def __init__(self, theta=None, R: int | None = None,
                 link: str = "identity"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for ocat family; available '
                f'links are {self._OK_LINKS}')
        if theta is None and R is None:
            raise ValueError("Must supply theta or R to ocat")
        if theta is not None:
            theta = np.asarray(theta, dtype=float).reshape(-1)
            R = theta.shape[0] + 2
        R = int(R)
        if R < 2:
            raise ValueError(f"ocat requires R >= 2 categories; got R={R}")
        self._R = R
        n_theta = R - 2
        if theta is not None and np.sum(theta == 0.0) == 0:
            if np.sum(theta < 0.0):
                ini = np.log(np.abs(theta))           # initial θ supplied
            else:
                ini = np.log(theta)                   # fixed θ
                n_theta = 0
        else:
            ini = np.full(R - 2, -1.0)
        self.n_theta = int(n_theta)
        self._theta = np.asarray(ini, dtype=float).reshape(-1)
        super().__init__(link=link)

    # ----- θ accessors ---------------------------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape[0] != self._R - 2:
            raise ValueError(
                f"ocat.set_theta expects {self._R - 2} log-step params; got "
                f"shape {v.shape}")
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        th = self._theta.copy()
        if not trans:
            return th
        # Finite cut points (R−1 of them): [−1, −1+cumsum(e^θ)].
        R = th.shape[0] + 2
        alpha = np.zeros(R - 1)
        alpha[0] = -1.0
        if R > 2:
            alpha[1:] = alpha[0] + np.cumsum(np.exp(th))
        return alpha

    # ----- deviance / Dd / aic -------------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        y1 = np.asarray(y).astype(int) + 1
        rsd, _ = _ocat_dev_signed(y1, mu, wt, th)
        return rsd

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        y1 = np.asarray(y).astype(int) + 1
        return _ocat_Dd(y1, mu, theta, wt, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None):
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        y1 = np.asarray(y).astype(int) + 1
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        alpha, _ = _ocat_alpha_full(th)
        al1 = alpha[y1]
        al0 = alpha[y1 - 1]
        f = _ocat_Fdiff(al0 - mu, al1 - mu)
        return -2.0 * float(np.sum(np.log(f) * wt))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        # ocat ls ≡ 0 (efam.r:2918-2921).
        n = np.asarray(y).shape[0]
        nt = self._R - 2
        return {"ls": 0.0, "lsth1": np.zeros(nt),
                "lsth2": np.zeros((nt, nt)), "LSTH1": np.zeros((n, nt))}

    def ls(self, y, wt, scale):
        # Scale-known: the log-φ ls path is never taken; provide the stub.
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # ----- initialization / validity -------------------------------------

    def preinitialize(self, y) -> dict | None:
        # mgcv ocat preinitialize (efam.r:2926-2945): integer-class check +
        # seed θ from the empirical class proportions.
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Response should be integer class labels")
        if self._R > 2 and self.n_theta > 0:
            theta = _ocat_ini(self._R, y)
            if theta is not None:
                return {"Theta": theta}
        return None

    def initialize(self, y, wt):
        # mgcv ocat initialize (efam.r:2947-2960): mustart is the midpoint
        # of the (finite, init-only) cut interval bracketing each class.
        R = self._theta.shape[0] + 2
        y0 = np.asarray(y).astype(int)
        if np.any(y0 < 0) or np.any(y0 > R - 1):
            raise ValueError("values out of range")
        alpha = np.zeros(R + 1)
        alpha[0] = -2.0
        alpha[1] = -1.0
        if R > 2:
            alpha[2:R] = alpha[1] + np.cumsum(np.exp(self._theta))
        alpha[R] = alpha[R - 1] + 1.0
        y1 = y0 + 1
        return (alpha[y1] + alpha[y1 - 1]) / 2.0

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(np.asarray(mu))))

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # ocat postproc (efam.r:2672-2679): null deviance via find.null.dev
        # (the optimal latent constant ≠ weighted mean), and the cut-point
        # relabel "Ordered Categorical(c1,c2,…)".
        null_dev = find_null_dev(self, y, eta=linear_predictors,
                                 offset=offset, weights=prior_weights)
        cuts = ",".join(f"{c:g}" for c in np.round(self.get_theta(True), 2))
        return {"null_deviance": null_dev,
                "family_name": f"Ordered Categorical({cuts})"}

    def residuals_extended(self, y, mu, wt, type: str = "deviance"):
        """ocat residuals (efam.r:2962-2993). ``deviance``: signed
        ``√(−2 wt log f)``. ``response``: ``y − ŷ`` with ŷ the class implied
        by the latent ``mu`` (both 0-based). ``working`` is the engine's."""
        y0 = np.asarray(y).astype(int)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        if type == "response":
            alpha, R = _ocat_alpha_full(self._theta)
            fv = np.zeros(mu.shape[0], dtype=float)
            for i in range(R):                # 0-based class i ⇔ (α_i, α_{i+1}]
                fv[(mu > alpha[i]) & (mu <= alpha[i + 1])] = i
            return y0.astype(float) - fv
        if type != "deviance":
            raise ValueError(
                f"ocat residuals are 'deviance' or 'response'; got {type!r}")
        rsd, s = _ocat_dev_signed(y0 + 1, mu, wt, self._theta)
        return np.sqrt(np.maximum(rsd, 0.0)) * s

    def predict(self, se=False, X=None, beta=None, off=None, Vb=None,
                eta=None, y=None, lpi=None) -> dict:
        """ocat ``predict`` hook (efam.r:2996-3049): ``type="response"``
        returns the per-class probability matrix (n × R) with optional
        delta-method SE."""
        cuts = self.get_theta(trans=True)        # finite cut points (R−1)
        if eta is None:
            mu = X @ beta
            if off is not None:
                mu = mu + np.asarray(off, dtype=float)
            se_v = None
            if se:
                se_v = np.sqrt(np.maximum(
                    0.0, np.einsum("ij,jk,ik->i", X, Vb, X)))
            prob, sep = _ocat_prob(cuts, mu, se_v)
            return {"fit": prob, "se_fit": sep} if se else {"fit": prob}
        # Category implied by the latent η (mean of the latent variable).
        eta = np.asarray(eta, dtype=float)
        alpha = np.concatenate([[-np.inf], cuts, [np.inf]])
        fv = np.zeros(eta.shape[0], dtype=float)
        for i in range(alpha.shape[0] - 1):
            fv[(eta > alpha[i]) & (eta <= alpha[i + 1])] = i
        return {"fit": fv}

    def rd(self, rng, mu, wt, scale):
        # mgcv ocat rd (efam.r:3051-3070): latent = mu + logit(U), allocate
        # to classes by the [−∞,−1,…,+∞] cut points. Returns 0-based labels.
        alpha, R = _ocat_alpha_full(self._theta)
        mu = np.asarray(mu, dtype=float)
        u = rng.uniform(size=mu.shape[0])
        lat = mu + np.log(u / (1.0 - u))
        y = np.zeros(mu.shape[0], dtype=float)
        for i in range(R):                        # 0-based class i
            y[(lat > alpha[i]) & (lat <= alpha[i + 1])] = i
        return y

    def __repr__(self):
        return f"ocat(R={self._R}, link={self.link.name})"


# ---------------------------------------------------------------------------
# ziP — single-formula zero-inflated Poisson (mgcv ziP(), efam.r:3848-4147).
# The ONE linear predictor μ is the log Poisson mean γ (E(Poisson)=e^γ);
# presence has probability p = 1 − exp(−exp(η)) with η = θ₁ + (b+e^θ₂)·γ a
# fixed affine map of γ (so presence rises with the mean). n_theta = 2.
# The log-lik kernel `zipll` is shared with the 2-LP `ziplss` GeneralFamily;
# the affine map's derivatives come from `lind`. mgcv's "−2logLik as
# deviance" (like betar): dev_resids omit the saturated reference, folded
# back via `saturated_ll` for the reported deviance/residuals.
# ---------------------------------------------------------------------------


def _zip_lind(mu, theta, deriv, k=0.0):
    """mgcv ``lind`` (efam.r:3774-3792): the affine presence map
    ``p = θ₁ + (k+e^θ₂)·μ`` and its μ/θ derivatives. Linear in μ, so
    ``p_ll = p_lll = p_llll = 0``."""
    mu = np.asarray(mu, dtype=float)
    th2 = np.exp(theta[1])
    n = mu.shape[0]
    r = {"p": theta[0] + (k + th2) * mu, "p_l": k + th2, "p_ll": 0.0,
         "p_lll": 0.0, "p_llll": 0.0}
    if deriv:
        r["p_th"] = np.zeros((n, 2))
        r["p_th"][:, 0] = 1.0
        r["p_th"][:, 1] = th2 * mu
        r["p_lth"] = np.zeros((n, 2))
        r["p_lth"][:, 1] = th2
        r["p_llth"] = np.zeros((n, 2))
        r["p_lllth"] = np.zeros((n, 2))
        r["p_th2"] = np.zeros((n, 3))     # ordered th1th1, th1th2, th2th2
        r["p_th2"][:, 2] = mu * th2
        r["p_lth2"] = np.zeros((n, 3))
        r["p_lth2"][:, 2] = th2
        r["p_llth2"] = np.zeros((n, 3))
    return r


def _zip_Dd(y, mu, theta, wt, b, level: int = 0) -> dict:
    """mgcv ``ziP()$Dd`` (efam.r:3892-3949): the ZIP deviance derivatives in
    μ (= log Poisson mean γ) and θ, assembled from ``zipll`` (derivs w.r.t.
    γ and the presence LP) chained through ``lind`` (presence LP w.r.t. μ,θ).
    ``mu`` is the Poisson-mean linear predictor."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.ones_like(mu) if wt is None else np.asarray(wt, dtype=float)
    deriv = 1
    if level == 1:
        deriv = 2
    elif level > 1:
        deriv = 4
    g = _zip_lind(mu, theta, level, k=b)
    z = _zipll(y, mu, g["p"], deriv)
    pL = g["p_l"]
    pll = g["p_ll"]
    l1, l2, El2 = z["l1"], z["l2"], z["El2"]
    w2 = wt[:, None]
    oo: dict = {}
    oo["Dmu"] = -2.0 * wt * (l1[:, 0] + l1[:, 1] * pL)
    oo["Dmu2"] = -2.0 * wt * (l2[:, 0] + 2.0 * l2[:, 1] * pL
                             + l2[:, 2] * pL ** 2 + l1[:, 1] * pll)
    oo["EDmu2"] = -2.0 * wt * (El2[:, 0] + 2.0 * El2[:, 1] * pL
                              + El2[:, 2] * pL ** 2)
    if level > 0:
        l3 = z["l3"]
        pth, plth, pllth = g["p_th"], g["p_lth"], g["p_llth"]
        plll = g["p_lll"]
        c1 = l1[:, 1][:, None]
        c2g, c2e = l2[:, 1][:, None], l2[:, 2][:, None]
        oo["Dth"] = -2.0 * w2 * (c1 * pth)
        oo["Dmuth"] = -2.0 * w2 * (c2g * pth + c2e * pL * pth + c1 * plth)
        oo["Dmu2th"] = -2.0 * w2 * (
            l3[:, 1][:, None] * pth + 2.0 * l3[:, 2][:, None] * pL * pth
            + 2.0 * c2g * plth + l3[:, 3][:, None] * pL ** 2 * pth
            + c2e * (2.0 * pL * plth + pth * pll) + c1 * pllth)
        oo["Dmu3"] = -2.0 * wt * (
            l3[:, 0] + 3.0 * l3[:, 1] * pL + 3.0 * l3[:, 2] * pL ** 2
            + 3.0 * l2[:, 1] * pll + l3[:, 3] * pL ** 3
            + 3.0 * l2[:, 2] * pL * pll + l1[:, 1] * plll)
    if level > 1:
        l4 = z["l4"]
        pth, plth = g["p_th"], g["p_lth"]
        pllth, plllth = g["p_llth"], g["p_lllth"]
        pth2, plth2, pllth2 = g["p_th2"], g["p_lth2"], g["p_llth2"]
        plll, pllll = g["p_lll"], g["p_llll"]
        # p.thth, p.lthth, p.lthlth, p.llthth (ordered th1th1, th1th2, th2th2)
        pthth = np.zeros((y.shape[0], 3))
        pthth[:, 0] = pth[:, 0] ** 2
        pthth[:, 1] = pth[:, 0] * pth[:, 1]
        pthth[:, 2] = pth[:, 1] ** 2
        plthth = np.zeros((y.shape[0], 3))
        plthth[:, 0] = pth[:, 0] * plth[:, 0] * 2.0
        plthth[:, 1] = pth[:, 0] * plth[:, 1] + pth[:, 1] * plth[:, 0]
        plthth[:, 2] = pth[:, 1] * plth[:, 1] * 2.0
        plthlth = np.zeros((y.shape[0], 3))
        plthlth[:, 0] = plth[:, 0] * plth[:, 0] * 2.0
        plthlth[:, 1] = plth[:, 0] * plth[:, 1] + plth[:, 1] * plth[:, 0]
        plthlth[:, 2] = plth[:, 1] * plth[:, 1] * 2.0
        pllthth = np.zeros((y.shape[0], 3))
        pllthth[:, 0] = pth[:, 0] * pllth[:, 0] * 2.0
        pllthth[:, 1] = pth[:, 0] * pllth[:, 1] + pth[:, 1] * pllth[:, 0]
        pllthth[:, 2] = pth[:, 1] * pllth[:, 1] * 2.0
        c1 = l1[:, 1][:, None]
        c2g, c2e = l2[:, 1][:, None], l2[:, 2][:, None]
        c3 = [l3[:, j][:, None] for j in range(4)]
        c4 = [l4[:, j][:, None] for j in range(5)]
        oo["Dth2"] = -2.0 * w2 * (c2e * pthth + c1 * pth2)
        oo["Dmuth2"] = -2.0 * w2 * (
            c3[2] * pthth + c2g * pth2 + c3[3] * pL * pthth
            + c2e * (pth2 * pL + plthth) + c1 * plth2)
        oo["Dmu2th2"] = -2.0 * w2 * (
            c4[2] * pthth + c3[1] * pth2 + 2.0 * c4[3] * pthth * pL
            + 2.0 * c3[2] * (pth2 * pL + plthth) + 2.0 * c2g * plth2
            + c4[4] * pthth * pL ** 2
            + c3[3] * (pth2 * pL ** 2 + 2.0 * plthth * pL + pthth * pll)
            + c2e * (plthlth + 2.0 * pL * plth2 + pllthth + pth2 * pll)
            + c1 * pllth2)
        oo["Dmu3th"] = -2.0 * w2 * (
            c4[1] * pth + 3.0 * c4[2] * pth * pL + 3.0 * c3[1] * plth
            + 2.0 * c4[3] * pth * pL ** 2
            + c3[2] * (6.0 * plth * pL + 3.0 * pth * pll) + 3.0 * c2g * pllth
            + c4[3] * pth * pL ** 2 + c4[4] * pth * pL ** 3
            + 3.0 * c3[3] * (pL ** 2 * plth + pth * pL * pll)
            + c2e * (3.0 * plth * pll + 3.0 * pL * pllth + pth * plll)
            + c1 * plllth)
        oo["Dmu4"] = -2.0 * wt * (
            l4[:, 0] + 4.0 * l4[:, 1] * pL + 6.0 * l4[:, 2] * pL ** 2
            + 6.0 * l3[:, 1] * pll + 4.0 * l4[:, 3] * pL ** 3
            + 12.0 * l3[:, 2] * pL * pll + 4.0 * l2[:, 1] * plll
            + l4[:, 4] * pL ** 4 + 6.0 * l3[:, 3] * pL ** 2 * pll
            + l2[:, 2] * (4.0 * pL * plll + 3.0 * pll ** 2) + l1[:, 1] * pllll)
    return oo


class ziP(Family):
    """Zero-inflated Poisson extended family — port of mgcv ``ziP()``
    (efam.r:3848-4147).

    The single linear predictor ``μ`` is the **log Poisson mean** ``γ``
    (so ``E(Poisson) = e^μ``); the probability of presence is
    ``p = 1 − exp(−exp(η))`` with ``η = θ₁ + (b + e^θ₂)·μ`` a fixed affine
    map (the slope ``b + e^θ₂ > b ≥ 0`` ties presence to the mean). The two
    parameters ``θ`` are estimated jointly with the smoothing parameters
    (``ziP(theta=…)`` fixes them, ``n_theta = 0``). Like :class:`betar`,
    ``dev_resids`` is the bare ``−2logLik`` and the saturated reference is
    folded back in :meth:`postproc` / :meth:`residuals_extended` via the
    :meth:`saturated_ll` Newton solver. Identity link only.
    """
    name = "zero inflated Poisson"
    canonical_link_name = "identity"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    n_theta = 2
    no_r_sq = True          # mgcv ziP sets no.r.sq=TRUE (efam.r:4142)
    _OK_LINKS = ("identity",)

    def __init__(self, theta=None, link: str = "identity", b: float = 0.0):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for ziP family; available '
                f'links are {self._OK_LINKS}')
        self._b = max(float(b), 0.0)
        if theta is not None:
            ini = np.asarray(theta, dtype=float).reshape(-1)[:2]
            self.n_theta = 0
        else:
            ini = np.array([0.0, 0.0])        # start at plain Poisson
        self._theta = ini.astype(float)
        super().__init__(link=link)

    # ----- θ accessors ---------------------------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape[0] != 2:
            raise ValueError(
                f"ziP.set_theta expects 2 params (θ₁, θ₂); got shape {v.shape}")
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        th = self._theta.copy()
        if trans:
            th[1] = self._b + np.exp(th[1])
        return th

    # ----- deviance / Dd / aic -------------------------------------------

    def _presence_lp(self, mu, theta):
        return theta[0] + (self._b + np.exp(theta[1])) * np.asarray(
            mu, dtype=float)

    def dev_resids(self, y, mu, wt, theta=None):
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        p = self._presence_lp(mu, th)
        # mgcv's R dev.resids is exactly `-2*zipll(...)$l` (efam.r:3884-3890).
        # For extreme trial η the Poisson mean e^η overflows; R folds that to
        # ±Inf silently (R's exp/`*` never warn), so mirror R's silence — the
        # Inf deviance for a rejected step is what mgcv's fitter sees too.
        with np.errstate(over="ignore", invalid="ignore"):
            return -2.0 * _zipll(np.asarray(y, dtype=float),
                                 np.asarray(mu, dtype=float), p, deriv=0)["l"]

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        return _zip_Dd(y, mu, np.asarray(theta, dtype=float), wt,
                       self._b, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None):
        th = self._theta if theta is None else np.asarray(theta, dtype=float)
        p = self._presence_lp(mu, th)
        wt = np.asarray(wt, dtype=float)
        ll = _zipll(np.asarray(y, dtype=float), np.asarray(mu, dtype=float),
                    p, deriv=0)["l"]
        return float(np.sum(-2.0 * wt * ll))

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        # ziP ls ≡ 0 (efam.r:3958-3967); deviance is −2logLik.
        n = np.asarray(y).shape[0]
        return {"ls": 0.0, "lsth1": np.zeros(2),
                "lsth2": np.zeros((2, 2)), "LSTH1": np.zeros((n, 2))}

    def ls(self, y, wt, scale):
        return np.array([0.0, 0.0, 0.0], dtype=float)

    # ----- saturated likelihood (Newton over the latent log-mean) --------

    def saturated_ll(self, y, wt):
        """mgcv ``saturated.ll`` (efam.r:4032-4068): per-datum Newton
        minimization of the ZIP deviance over the latent log-mean μ (only
        y>0 contribute). Returns the per-datum ``−2·saturated logLik``
        (0 where y==0)."""
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        theta = self._theta
        pind = y > 0
        yp = y[pind]
        wp = wt[pind]
        if yp.shape[0] == 0:
            return np.zeros(y.shape[0])
        mu = np.log(yp)
        r = self.Dd(yp, mu, theta, wp, level=0)
        l = self.dev_resids(yp, mu, wp, theta=theta)
        lmax = float(np.max(np.abs(l)))
        ucov = np.abs(r["Dmu"]) > lmax * 1e-7
        k = 0
        while True:
            step = -r["Dmu"] / r["Dmu2"]
            step[~ucov] = 0.0
            mu1 = mu + step
            l1 = self.dev_resids(yp, mu1, wp, theta=theta)
            ind = (l1 > l) & ucov
            kk = 0
            while np.sum(ind) > 0 and kk < 50:
                step[ind] = step[ind] / 2.0
                mu1 = mu + step
                l1 = self.dev_resids(yp, mu1, wp, theta=theta)
                ind = (l1 > l) & ucov
                kk += 1
            mu = mu1
            l = l1
            r = self.Dd(yp, mu, theta, wp, level=0)
            ucov = np.abs(r["Dmu"]) > lmax * 1e-7
            k += 1
            if (not np.any(ucov)) or k == 100:
                break
        out = np.zeros(y.shape[0])
        out[pind] = l
        return out

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        # mgcv ziP initialize (efam.r:3970-3978).
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError(
                "negative values not allowed for the zero inflated Poisson "
                "family")
        if not np.allclose(y, np.round(y)):
            raise ValueError(
                "Non-integer response variables are not allowed with ziP ")
        if y.min() == 0 and y.max() == 1:
            raise ValueError("Using ziP for binary data makes no sense")
        return np.log(y + (y == 0) / 5.0)

    def validmu(self, mu) -> bool:
        return bool(np.all(np.isfinite(np.asarray(mu))))

    # ----- E(y), postproc, residuals, predict ----------------------------

    def _expected_y(self, gamma):
        """E(y) = p·E(y | present): p the presence prob, E(y|present) the
        zero-truncated Poisson mean (efam.r:4110-4119). Returns
        (fv, p, mu_trunc, lambda)."""
        gamma = np.asarray(gamma, dtype=float)
        th = self._theta
        with np.errstate(over="ignore", invalid="ignore"):
            eta = th[0] + (self._b + np.exp(th[1])) * gamma
            et = np.exp(eta)
            p = 1.0 - np.exp(-et)
            lam = np.exp(gamma)
            ind = gamma < np.log(np.finfo(float).eps) / 2.0
            mu = np.where(ind, 1.0, lam / (1.0 - np.exp(-lam)))
        return p * mu, p, mu, lam

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # ziP postproc (efam.r:3982-4004): deviance folds in the saturated
        # ll; null deviance from a 1-D optimize over the constant LP.
        y = np.asarray(y, dtype=float)
        wt = np.asarray(prior_weights, dtype=float)
        lf = self.saturated_ll(y, wt)
        dev = float(np.sum(self.dev_resids(y, linear_predictors, wt) - lf))

        def fnull(gamma):
            return float(np.sum(self.dev_resids(
                y, np.full(y.shape, gamma), wt)))

        meany = float(np.mean(y))
        tol = float(np.finfo(float).eps ** 0.25)
        _, obj = _brent_fmin(fnull, meany / 5.0, meany * 3.0, tol)
        null_dev = obj - float(np.sum(lf))
        cuts = ",".join(f"{c:g}" for c in np.round(self.get_theta(True), 3))
        return {"deviance": dev, "null_deviance": null_dev,
                "family_name": f"Zero inflated Poisson({cuts})"}

    def residuals_extended(self, y, mu, wt, type: str = "deviance"):
        """ziP residuals (efam.r:4070-4088). ``mu`` is the linear predictor
        γ. ``deviance``: signed √(dev_resids − saturated_ll). ``response``:
        ``y − E(y)``."""
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        fv = self._expected_y(mu)[0]
        if type == "response":
            return y - fv
        if type != "deviance":
            raise ValueError(
                f"ziP residuals are 'deviance' or 'response'; got {type!r}")
        res = self.dev_resids(y, mu, wt) - self.saturated_ll(y, wt)
        s = np.sign(y - fv)
        return np.sqrt(np.maximum(res, 0.0)) * s

    def predict(self, se=False, X=None, beta=None, off=None, Vb=None,
                eta=None, y=None, lpi=None) -> dict:
        """ziP ``predict`` hook (efam.r:4090-4130): ``type="response"``
        returns ``E(y) = p·E(y|present)`` with optional delta-method SE."""
        th = self._theta
        if eta is None:
            gamma = X @ beta
            if off is not None:
                gamma = gamma + np.asarray(off, dtype=float)
            se_v = None
            if se:
                se_v = np.sqrt(np.maximum(
                    0.0, np.einsum("ij,jk,ik->i", X, Vb, X)))
        else:
            gamma = np.asarray(eta, dtype=float)
            se_v = None
        fv, p, mu, lam = self._expected_y(gamma)
        if se_v is None:
            return {"fit": fv}
        with np.errstate(over="ignore", invalid="ignore"):
            eta_p = th[0] + (self._b + np.exp(th[1])) * gamma
            et = np.exp(eta_p)
            dp_dg = np.exp(-et) * et * (self._b + np.exp(th[1]))
            dmu_dg = (lam + 1.0) * mu - mu ** 2
        se_fit = np.abs(dp_dg * mu + dmu_dg * p) * se_v
        return {"fit": fv, "se_fit": se_fit}

    def rd(self, rng, mu, wt, scale):
        # mgcv ziP rd (efam.r:4006-4030): presence ~ Bernoulli(p), counts ~
        # zero-truncated Poisson(λ) via the inverse-CDF.
        from scipy.stats import poisson as _pois
        gamma = np.asarray(mu, dtype=float)
        th = self._theta
        n = gamma.shape[0]
        with np.errstate(over="ignore", invalid="ignore"):
            lam = np.exp(gamma)
            finite = np.isfinite(lam)
            mlam = max(float(np.max(lam[finite])) if np.any(finite) else 0.0,
                       np.finfo(float).eps ** 0.2)
            lam = np.where(finite, lam, mlam)
            eta = th[0] + (self._b + np.exp(th[1])) * gamma
            p = 1.0 - np.exp(-np.exp(eta))
        y = np.zeros(n)
        present = p > rng.uniform(size=n)
        lami = lam[present]
        p0 = _pois.pmf(0, lami)
        nearly1 = 1.0 - np.finfo(float).eps * 10.0
        ii = p0 > nearly1
        yi = np.ones(lami.shape[0])
        m = ~ii
        if np.any(m):
            u = rng.uniform(p0[m], nearly1, size=int(np.sum(m)))
            yi[m] = _pois.ppf(u, lami[m])
        y[present] = yi
        return y

    def __repr__(self):
        return f"ziP(theta={self._theta}, b={self._b}, link={self.link.name})"


# ---------------------------------------------------------------------------
# cnorm (censored normal / Tobit) — mgcv ``cnorm()`` (efam.r:734-1163).
#
# Single log-scale θ (σ = e^θ, per-datum th = θ − log(wt)/2). The response
# is a 2-column ``cbind(y, yat)``: column 0 is the observed value, column 1
# the censoring bound. Four cases by ``yat`` vs ``y`` — uncensored
# (yat==y), interval (finite & yat≠y), left (yat==−∞), right (yat==+∞).
# Unlike betar/ziP/ocat, cnorm's ``dev_resids`` is the PROPER deviance
# (saturated reference included; uncensored → z²) and ``ls`` is a genuinely
# nonzero saturated log-lik with ZERO θ-derivatives — so no saturated_ll
# Newton, no deviance override, no residuals_extended (the default √ works).
# ---------------------------------------------------------------------------

_LOG2PI = float(np.log(2.0 * np.pi))


def _dnorm_log(x):
    """``dnorm(x, log=TRUE)`` — log of the standard normal density."""
    x = np.asarray(x, dtype=float)
    return -0.5 * x * x - 0.5 * _LOG2PI


def _logexm1(x):
    """mgcv ``logexm1`` (misc.r:18-27): log(e^x − 1), overflow-safe. For
    x ≥ log(1/eps)+1 the −1 is negligible so log(e^x−1) ≈ x."""
    x = np.array(x, dtype=float, copy=True)
    xt = np.log(1.0 / np.finfo(float).eps) + 1.0
    ii = x < xt
    with np.errstate(divide="ignore", invalid="ignore"):
        x[ii] = np.log(np.expm1(x[ii]))
    return x


def _cnorm_logexp1(x):
    """mgcv ``logexp1`` (efam.r:801-807): log(e^x + 1), overflow-safe."""
    x = np.array(x, dtype=float, copy=True)
    xt = np.log(1.0 / np.finfo(float).eps) + 1.0
    ii = x < xt
    with np.errstate(over="ignore"):
        x[ii] = np.log(np.exp(x[ii]) + 1.0)
    return x


def _cnorm_dpnorm(x0, x1, log_p=True):
    """mgcv ``dpnorm`` (misc.r:29-40): cancellation-avoiding log(Φ(x1) −
    Φ(x0)). Both-positive pairs are reflected to the lower tail first."""
    x0 = np.array(x0, dtype=float, copy=True)
    x1 = np.array(x1, dtype=float, copy=True)
    ii = (x1 > 0) & (x0 > 0)
    d = x0[ii].copy()
    x0[ii] = -x1[ii]
    x1[ii] = -d
    p0 = log_ndtr(x0)
    p1 = log_ndtr(x1)
    dp = p0 + _logexm1(p1 - p0)
    return dp if log_p else np.exp(dp)


def _cnorm_ddnorm(x0, x1, a0=0.0, a1=0.0, s0=1.0, s1=1.0):
    """mgcv ``ddnorm`` (efam.r:809-829): cancellation-avoiding evaluation
    of ``c = s1·e^{a1}·φ(x1) − s0·e^{a0}·φ(x0)``. Returns ``(log|c|,
    sign)``."""
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    shape = np.broadcast(x0, x1, a0, a1, s0, s1).shape
    a0 = np.broadcast_to(np.asarray(a0, dtype=float), shape).copy()
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), shape).copy()
    s0 = np.broadcast_to(np.asarray(s0, dtype=float), shape).astype(float).copy()
    s1 = np.broadcast_to(np.asarray(s1, dtype=float), shape).astype(float).copy()
    with np.errstate(invalid="ignore"):
        p0 = _dnorm_log(x0) + a0
        p1 = _dnorm_log(x1) + a1
    dp = p0.copy()
    # sign of c (computed on the original, pre-swap p0/p1)
    sgn = np.ones(shape)
    flip = (((s1 < 0) & (s0 > 0))
            | ((s1 > 0) & (s0 > 0) & (p1 < p0))
            | ((s1 < 0) & (s0 < 0) & (p1 > p0)))
    sgn[flip] = -1.0
    # swap so p0 ≤ p1 (keeps the logexm1/logexp1 arguments well-signed)
    swap = p0 > p1
    tmp = p1[swap].copy()
    p1[swap] = p0[swap]
    p0[swap] = tmp
    same = (s0 * s1) > 0
    dp[same] = p0[same] + _logexm1(p1[same] - p0[same])
    opp = (s0 * s1) < 0
    dp[opp] = p0[opp] + _cnorm_logexp1(p1[opp] - p0[opp])
    # s0/s1 == 0 edges (unreachable for continuous z; ported for fidelity)
    z0m = s0 == 0
    if np.any(z0m):
        sgn[z0m] = s1[z0m]
        dp[z0m] = p1[z0m]
    z1m = s1 == 0
    if np.any(z1m):
        sgn[z1m] = -s0[z1m]
        dp[z1m] = p0[z1m]
    return dp, sgn


def _cnorm_cases(y, censor):
    """Return (yat, iu, ii, il, ir): the censoring bound and the index sets
    for uncensored / interval / left / right (mgcv efam.r:836-843)."""
    y = np.asarray(y, dtype=float)
    yat = y if censor is None else np.asarray(censor, dtype=float)
    iu = np.where(yat == y)[0]
    ii = np.where(np.isfinite(yat) & (yat != y))[0]
    il = np.where(yat == -np.inf)[0]
    ir = np.where(yat == np.inf)[0]
    return yat, iu, ii, il, ir


def _cnorm_dev_resids(y, mu, wt, theta, censor):
    """mgcv cnorm ``dev.resids`` (efam.r:766-789): the proper deviance
    (−2·(logLik − l_sat)), per datum, by censoring case."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    th = float(theta) - np.log(wt) / 2.0
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    d = np.zeros(y.shape[0])
    if iu.size:
        d[iu] = (y[iu] - mu[iu]) ** 2 * np.exp(-2.0 * th[iu])
    if ii.size:
        y1 = np.maximum(yat[ii], y[ii])
        y0 = np.minimum(yat[ii], y[ii])
        ethi = np.exp(-th[ii])
        zz = (y1 - y0) * ethi / 2.0
        d[ii] = (2.0 * _cnorm_dpnorm(-zz, zz, log_p=True)
                 - 2.0 * _cnorm_dpnorm((y0 - mu[ii]) * ethi,
                                       (y1 - mu[ii]) * ethi, log_p=True))
    if il.size:
        d[il] = -2.0 * log_ndtr((y[il] - mu[il]) * np.exp(-th[il]))
    if ir.size:
        d[ir] = -2.0 * log_ndtr(-(y[ir] - mu[ir]) * np.exp(-th[ir]))
    return d


def _cnorm_aic(y, mu, wt, theta, censor):
    """mgcv cnorm ``aic`` (efam.r:1068-1089): −2·logLik (no saturated
    reference). NOTE mgcv's left-censor selector here is ``yat <= 0`` (not
    ``yat == −∞`` as in dev.resids) — replicated verbatim so hea's AIC
    matches mgcv's, quirk and all."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    th = float(theta) - np.log(wt) / 2.0
    yat = y if censor is None else np.asarray(censor, dtype=float)
    d = np.zeros(y.shape[0])
    iu = np.where(yat == y)[0]
    if iu.size:
        d[iu] = -2.0 * _dnorm_log((y[iu] - mu[iu]) * np.exp(-th[iu]))
    ii = np.where(np.isfinite(yat) & (yat != y))[0]
    if ii.size:
        y1 = np.maximum(yat[ii], y[ii])
        y0 = np.minimum(yat[ii], y[ii])
        ethi = np.exp(-th[ii])
        d[ii] = -2.0 * _cnorm_dpnorm((y0 - mu[ii]) * ethi,
                                     (y1 - mu[ii]) * ethi, log_p=True)
    il = np.where(yat <= 0)[0]
    if il.size:
        d[il] = -2.0 * log_ndtr((y[il] - mu[il]) * np.exp(-th[il]))
    ir = np.where(yat == np.inf)[0]
    if ir.size:
        d[ir] = -2.0 * log_ndtr(-(y[ir] - mu[ir]) * np.exp(-th[ir]))
    return float(np.sum(d))


def _cnorm_ls_val(y, wt, theta, censor):
    """mgcv cnorm ``ls`` (efam.r:1091-1114): the saturated log-likelihood
    VALUE (nonzero — uncensored normal entropy + interval span), with all
    θ-derivatives identically zero."""
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    th = float(theta) - np.log(wt) / 2.0
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    ls = 0.0
    if iu.size:
        ls += float(np.sum(-th[iu] - _LOG2PI / 2.0))
    if ii.size:
        y1 = np.maximum(yat[ii], y[ii])
        y0 = np.minimum(yat[ii], y[ii])
        zz = (y1 - y0) * np.exp(-th[ii]) / 2.0
        ls += float(np.sum(_cnorm_dpnorm(-zz, zz, log_p=True)))
    return ls


def _cnorm_Dd(y, mu, theta, wt, censor, level=0):
    """mgcv cnorm ``Dd`` (efam.r:791-1066): derivatives of the cnorm
    deviance w.r.t. μ and the log-scale θ, by censoring case. Verbatim port
    (1-based R → 0-based numpy index sets); cancellation handled by
    :func:`_cnorm_dpnorm` / :func:`_cnorm_ddnorm`."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(np.asarray(theta, dtype=float).reshape(-1)[0])
    th = theta - np.log(wt) / 2.0
    th3 = 3.0 * th
    eth = np.exp(-th)
    e2th = eth * eth
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)

    n = mu.shape[0]
    Dmu = np.zeros(n)
    Dmu2 = np.zeros(n)
    Dth = np.zeros(n)
    Dmuth = np.zeros(n)
    Dmu2th = np.zeros(n)
    Dmu3 = np.zeros(n)
    Dth2 = np.zeros(n)
    Dmuth2 = np.zeros(n)
    Dmu2th2 = np.zeros(n)
    Dmu4 = np.zeros(n)
    Dmu3th = np.zeros(n)

    _es = dict(divide="ignore", invalid="ignore", over="ignore")

    if iu.size:  # uncensored
        ethi = eth[iu]
        e2thi = e2th[iu]
        z = (y[iu] - mu[iu]) * ethi
        Dmui = -2.0 * z * ethi
        Dmu[iu] = Dmui
        Dmu2[iu] = 2.0 * e2thi
        if level > 0:
            Dth[iu] = -2.0 * (z ** 2 - 1.0)
            Dmuth[iu] = -2.0 * Dmui
            Dmu3[iu] = 0.0
            Dmu2th[iu] = -4.0 * e2thi
        if level > 1:
            Dmu4[iu] = 0.0
            Dmu3th[iu] = 0.0
            Dth2[iu] = 4.0 * z ** 2
            Dmuth2[iu] = 4.0 * Dmui
            Dmu2th2[iu] = 8.0 * e2thi

    if ii.size:  # interval censored
        muu = mu[ii]
        y0 = np.minimum(y[ii], yat[ii])
        y1 = np.maximum(y[ii], yat[ii])
        ethi = eth[ii]
        e2thi = e2th[ii]
        thi = th[ii]
        th3i = th3[ii]
        z0 = (y0 - muu) * ethi
        z1 = (y1 - muu) * ethi
        with np.errstate(**_es):
            ldp = _cnorm_dpnorm(z0, z1, log_p=True)
            ldd, sdd = _cnorm_ddnorm(z0, z1)
            ldzdz, szdz = _cnorm_ddnorm(z0, z1, np.log(np.abs(z0)),
                                        np.log(np.abs(z1)),
                                        np.sign(z0), np.sign(z1))
            Dmui = 2.0 * sdd * np.exp(-thi + ldd - ldp)
            Dt = 2.0 * szdz * np.exp(ldzdz - ldp)
            Dmu2i = Dmui ** 2 / 2.0 + e2thi * Dt
            Dmu[ii] = Dmui
            Dmu2[ii] = Dmu2i
            if level > 0:
                ldz2, sz2 = _cnorm_ddnorm(z0, z1, np.log(z0 ** 2),
                                          np.log(z1 ** 2))
                ldz3, sz3 = _cnorm_ddnorm(z0, z1, np.log(np.abs(z0 ** 3)),
                                          np.log(np.abs(z1 ** 3)),
                                          np.sign(z0), np.sign(z1))
                z12 = z1 ** 2
                z02 = z0 ** 2
                z13 = z12 * z1
                z03 = z02 * z0
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0 - e2thi)
                         + 2.0 * sz2 * np.exp(ldz2 - ldp - th3i))
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       + 2.0 * sz2 * np.exp(ldz2 - ldp - thi))
                Dtt = Dt ** 2 / 2.0 - Dt + 2.0 * sz3 * np.exp(ldz3 - ldp)
                Dmu2thi = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
                Dth[ii] = Dt
                Dmuth[ii] = Dmt
                Dmu3[ii] = Dmu3i
                Dmu2th[ii] = Dmu2thi
                if level > 1:
                    z14 = z13 * z1
                    z04 = z03 * z0
                    a1 = 2.0 * z13 * ethi + Dmui * z12 - 4.0 * z1 * ethi
                    a0 = 2.0 * z03 * ethi + Dmui * z02 - 4.0 * z0 * ethi
                    lda1, sa1 = _cnorm_ddnorm(z0, z1, np.log(np.abs(a0)),
                                              np.log(np.abs(a1)),
                                              np.sign(a0), np.sign(a1))
                    Dmu4[ii] = (Dmu2i * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0
                                         - e2thi)
                                + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                                + sa1 * np.exp(lda1 - ldp - th3i))
                    ldz4, sz4 = _cnorm_ddnorm(z0, z1, np.log(z0 ** 4),
                                              np.log(z1 ** 4))
                    Dmu3th[ii] = (Dmt * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0)
                                  + Dmui * (3.0 * Dmu2thi - Dmui * Dmt) / 2.0
                                  + e2thi * (2.0 * Dmui - Dmt)
                                  + (Dt - 10.0) * sz2
                                  * np.exp(ldz2 - ldp - th3i)
                                  + 2.0 * sz4 * np.exp(ldz4 - ldp - th3i))
                    Dth2[ii] = Dtt
                    Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                            + (Dt - 6.0) * sz2 * np.exp(ldz2 - ldp - thi)
                            + 2.0 * sz4 * np.exp(ldz4 - ldp - thi))
                    Dmuth2[ii] = Dmtt
                    a1b = z13 * (z12 - 3.0)
                    a0b = z03 * (z02 - 3.0)
                    lda6, sa6 = _cnorm_ddnorm(z0, z1, np.log(np.abs(a0b)),
                                              np.log(np.abs(a1b)),
                                              np.sign(a0b), np.sign(a1b))
                    Dttt = (Dtt * (Dt - 1.0) + Dt * sz3 * np.exp(ldz3 - ldp)
                            + 2.0 * sa6 * np.exp(lda6 - ldp))
                    Dmu2th2[ii] = (Dmt ** 2 + Dmui * Dmtt
                                   + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    if il.size:  # left censored (y0 = −∞)
        ethi = eth[il]
        e2thi = e2th[il]
        thi = th[il]
        th3i = th3[il]
        z1 = (y[il] - mu[il]) * ethi
        with np.errstate(**_es):
            ldp = log_ndtr(z1)
            ldn = _dnorm_log(z1)
            Dmui = 2.0 * np.exp(-thi + ldn - ldp)
            Dt = 2.0 * np.sign(z1) * np.exp(ldn + np.log(np.abs(z1)) - ldp)
            Dmu2i = Dmui ** 2 / 2.0 + e2thi * Dt
            Dmu[il] = Dmui
            Dmu2[il] = Dmu2i
            if level > 0:
                z12 = z1 ** 2
                z13 = z12 * z1
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0 - e2thi)
                         + 2.0 * np.sign(z12)
                         * np.exp(ldn + np.log(np.abs(z12)) - ldp - th3i))
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       + 2.0 * np.sign(z12)
                       * np.exp(ldn + np.log(np.abs(z12)) - ldp - thi))
                Dtt = (Dt ** 2 / 2.0 - Dt + 2.0 * np.sign(z13)
                       * np.exp(ldn + np.log(np.abs(z13)) - ldp))
                Dmu2thi = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
                Dth[il] = Dt
                Dmuth[il] = Dmt
                Dmu3[il] = Dmu3i
                Dmu2th[il] = Dmu2thi
                if level > 1:
                    z14 = z13 * z1
                    a1 = 2.0 * z13 * ethi + Dmui * z12 - 4.0 * z1 * ethi
                    Dmu4[il] = (Dmu2i * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0
                                         - e2thi)
                                + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                                + np.sign(a1)
                                * np.exp(ldn + np.log(np.abs(a1)) - ldp - th3i))
                    Dmu3th[il] = (Dmt * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0)
                                  + Dmui * (3.0 * Dmu2thi - Dmui * Dmt) / 2.0
                                  + e2thi * (2.0 * Dmui - Dmt)
                                  + (Dt - 10.0)
                                  * np.exp(ldn + np.log(z12) - ldp - th3i)
                                  + 2.0 * np.exp(ldn + np.log(z14)
                                                 - ldp - th3i))
                    Dth2[il] = Dtt
                    Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                            + (Dt - 6.0)
                            * np.exp(ldn + np.log(z12) - ldp - thi)
                            + 2.0 * np.sign(z14)
                            * np.exp(ldn + np.log(np.abs(z14)) - ldp - thi))
                    Dmuth2[il] = Dmtt
                    a1b = z13 * (z12 - 3.0)
                    Dttt = (Dtt * (Dt - 1.0) + Dt * np.sign(z13)
                            * np.exp(ldn + np.log(np.abs(z13)) - ldp)
                            + 2.0 * np.sign(a1b)
                            * np.exp(ldn + np.log(np.abs(a1b)) - ldp))
                    Dmu2th2[il] = (Dmt ** 2 + Dmui * Dmtt
                                   + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    if ir.size:  # right censored (y1 = +∞)
        ethi = eth[ir]
        e2thi = e2th[ir]
        thi = th[ir]
        th3i = th3[ir]
        z0 = (y[ir] - mu[ir]) * ethi
        with np.errstate(**_es):
            ldp = log_ndtr(-z0)
            ldn = _dnorm_log(z0)
            Dmui = -2.0 * np.exp(-thi + ldn - ldp)
            Dt = -2.0 * np.sign(z0) * np.exp(ldn + np.log(np.abs(z0)) - ldp)
            Dmu2i = Dmui ** 2 / 2.0 + e2thi * Dt
            Dmu[ir] = Dmui
            Dmu2[ir] = Dmu2i
            if level > 0:
                z02 = z0 ** 2
                z03 = z02 * z0
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0 - e2thi)
                         - 2.0 * np.sign(z02)
                         * np.exp(ldn + np.log(np.abs(z02)) - ldp - th3i))
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       - 2.0 * np.sign(z02)
                       * np.exp(ldn + np.log(np.abs(z02)) - ldp - thi))
                Dtt = (Dt ** 2 / 2.0 - Dt - 2.0 * np.sign(z03)
                       * np.exp(ldn + np.log(np.abs(z03)) - ldp))
                Dmu2thi = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
                Dth[ir] = Dt
                Dmuth[ir] = Dmt
                Dmu3[ir] = Dmu3i
                Dmu2th[ir] = Dmu2thi
                if level > 1:
                    z04 = z03 * z0
                    a1 = 2.0 * z03 * ethi + Dmui * z02 - 4.0 * z0 * ethi
                    Dmu4[ir] = (Dmu2i * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0
                                         - e2thi)
                                + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                                - np.sign(a1)
                                * np.exp(ldn + np.log(np.abs(a1)) - ldp - th3i))
                    Dmu3th[ir] = (Dmt * (3.0 * Dmu2i / 2.0 - Dmui ** 2 / 4.0)
                                  + Dmui * (3.0 * Dmu2thi - Dmui * Dmt) / 2.0
                                  + e2thi * (2.0 * Dmui - Dmt)
                                  - (Dt - 10.0)
                                  * np.exp(ldn + np.log(z02) - ldp - th3i)
                                  - 2.0 * np.exp(ldn + np.log(z04)
                                                 - ldp - th3i))
                    Dth2[ir] = Dtt
                    Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                            - (Dt - 6.0)
                            * np.exp(ldn + np.log(z02) - ldp - thi)
                            - 2.0 * np.exp(ldn + np.log(z04) - ldp - thi))
                    Dmuth2[ir] = Dmtt
                    a1b = z03 * (z02 - 3.0)
                    Dttt = (Dtt * (Dt - 1.0) - Dt * np.sign(z03)
                            * np.exp(ldn + np.log(np.abs(z03)) - ldp)
                            - 2.0 * np.sign(a1b)
                            * np.exp(ldn + np.log(np.abs(a1b)) - ldp))
                    Dmu2th2[ir] = (Dmt ** 2 + Dmui * Dmtt
                                   + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    r = {"Dmu": Dmu, "Dmu2": Dmu2, "EDmu2": Dmu2}
    if level > 0:
        r["Dth"] = Dth
        r["Dmuth"] = Dmuth
        r["Dmu3"] = Dmu3
        r["Dmu2th"] = Dmu2th
        r["EDmu2th"] = Dmu2th
    if level > 1:
        r["Dmu4"] = Dmu4
        r["Dth2"] = Dth2
        r["Dmuth2"] = Dmuth2
        r["Dmu2th2"] = Dmu2th2
        r["Dmu3th"] = Dmu3th
    return r


class cnorm(Family):
    """Censored normal (Tobit) extended family — port of mgcv ``cnorm()``
    (efam.r:734-1163).

    The single linear predictor ``μ`` is the latent Gaussian mean; the
    log-scale ``θ`` (σ = e^θ) is estimated jointly with the smoothing
    parameters (``cnorm(theta=…)`` fixes it, ``n_theta = 0``). The response
    is a 2-column ``cbind(y, yat)``: column 0 the observed value, column 1
    the censoring bound — ``yat == y`` uncensored, finite ``yat ≠ y``
    interval, ``yat == −∞`` left, ``yat == +∞`` right. A 1-column response
    is all-uncensored (plain Gaussian with σ = e^θ).

    Unlike :class:`betar` / :class:`ziP`, ``dev_resids`` is the proper
    deviance (≥ 0) and ``ls`` is a genuinely nonzero saturated log-lik with
    zero θ-derivatives, so the standard √-deviance residual and the
    ``(Dp/φ − 2·ls0)`` REML term apply directly. okLinks: identity (default),
    log, sqrt.
    """
    name = "cnorm"
    canonical_link_name = "identity"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    n_theta = 1
    _OK_LINKS = ("identity", "log", "sqrt")

    def __init__(self, theta=None, link: str = "identity"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for cnorm family; available '
                f'links are {self._OK_LINKS}')
        # mgcv θ intake (efam.r:743-753): θ>0 fixed (store log θ, n_theta=0);
        # θ≤0 an initial working value (θ<0 → store log|θ|); None → 0.
        if theta is not None:
            t = float(theta)
            if t > 0:
                ini = float(np.log(t))
                self.n_theta = 0
            else:
                ini = float(np.log(-t)) if t < 0 else t
        else:
            ini = 0.0
        self._theta = np.array([ini], dtype=float)
        # bam's bgam.fit θ-update gate (bam.r:1204-1206): estimate.theta
        # runs between PIRLS iters whenever the extended family has free
        # θ (``family$n.theta>0``; the ``scale<0`` leg never fires here —
        # the censored families carry no ``scale`` slot, so bgam.fit's
        # scale resolves to 1, bam.r:924).
        self.estimate_theta_callback = self.n_theta > 0
        self._censor = None
        self._censorfull = None
        super().__init__(link=link)

    # ----- θ accessors / censoring bound ---------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape[0] != 1:
            raise ValueError(
                f"cnorm.set_theta expects 1 param (log σ); got shape {v.shape}")
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        th = self._theta.copy()
        return np.exp(th) if trans else th

    def set_censor(self, censor) -> None:
        """Stash the censoring bound (column 1 of the ``cbind(y, yat)``
        response), aligned with the full response. ``None`` ⇒ all
        uncensored (mgcv's ``attr(y,"censor")`` being NULL)."""
        self._censor = (None if censor is None
                        else np.asarray(censor, dtype=float))
        self._censorfull = None

    def set_ind(self, ind) -> None:
        """mgcv ``subsety`` (efam.r — identical body on all censored
        families): bam's chunk loop subsets the response *and its censor
        attribute* per chunk. hea carries ``attr(y,"censor")`` on the
        family, so the windowing lands here: stash the full bound on
        first use, window to ``ind``, restore with ``ind=None``
        (mirrors gfam's ``setInd``, gfam.r:78-86)."""
        if self._censorfull is None:
            if ind is None or self._censor is None:
                return          # never windowed / uncensored: nothing to do
            self._censorfull = self._censor
        self._censor = (self._censorfull if ind is None
                        else self._censorfull[np.asarray(ind, dtype=int)])

    # ----- deviance / Dd / aic -------------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        return _cnorm_dev_resids(y, mu, wt, th, self._censor)

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        return _cnorm_Dd(y, mu, theta, wt, self._censor, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        return _cnorm_aic(y, mu, wt, th, self._censor)

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        ls = _cnorm_ls_val(y, wt, th, self._censor)
        n = np.asarray(y).shape[0]
        return {"ls": ls, "lsth1": np.array([0.0]),
                "lsth2": np.array([[0.0]]), "LSTH1": np.zeros((n, 1))}

    def ls(self, y, wt, scale):
        ls = _cnorm_ls_val(y, wt, float(self._theta[0]), self._censor)
        return np.array([ls, 0.0, 0.0], dtype=float)

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        # mgcv cnorm initialize (efam.r:1117-1124): the matrix split has
        # already happened at intake; mustart = y (identity) or
        # pmax(y, min(y>0)) — min of the LOGICAL vector y>0, i.e. a floor
        # of 1 when every y is positive and 0 otherwise.
        y = np.asarray(y, dtype=float)
        if self.link.name == "identity":
            return y.copy()
        return np.maximum(y, float(np.min(y > 0)))

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu, dtype=float)
        if self.link.name == "identity":
            return bool(np.all(np.isfinite(mu)))
        return bool(np.all(mu > 0))

    # ----- postproc ------------------------------------------------------

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # mgcv cnorm postproc (efam.r:1126-1137): null deviance from
        # find.null.dev; family relabel "cnorm(σ)".
        null_dev = find_null_dev(self, y, eta=linear_predictors,
                                 offset=offset, weights=prior_weights)
        sig = ",".join(f"{c:g}" for c in np.round(self.get_theta(True), 3))
        return {"null_deviance": null_dev, "family_name": f"cnorm({sig})"}

    def __repr__(self):
        return f"cnorm(theta={self._theta}, link={self.link.name})"


# ---------------------------------------------------------------------------
# cpois (censored Poisson) — mgcv ``cpois()`` (efam.r:344-537) and its
# ``dppois`` helper (efam.r:312-339).
#
# Same 2-column ``cbind(y, yat)`` censor encoding as cnorm — uncensored
# (yat==y), interval (finite & yat≠y), left (yat==−∞, saturated lik 1),
# right (yat==+∞) — but with NO family parameters at all (n_theta = 0;
# mgcv's getTheta/putTheta are empty functions) and φ ≡ 1. Like cnorm,
# ``dev_resids`` is the proper deviance and ``ls`` a genuinely nonzero,
# derivative-free saturated log-lik. mgcv exports no variance slot and
# its rd/qf are "NOTE - not done" stubs left out of the returned
# structure — the base-class NotImplementedError/None mirror that NULL.
# ---------------------------------------------------------------------------


def _rpow_int(x, k):
    """R's ``x ^ k`` for small positive integer k — the ``R_pow`` the
    interpreter's POWOP loop calls per element (ref/r-base/
    arithmetic.c:204-253): k==2 → ``x·x`` for ALL x; k∈{3,4} → the
    sequential multiply ONLY for −11 ≤ x ≤ 11, libm ``pow`` otherwise
    (numpy's ``**`` makes neither split — last-ulp drift vs R)."""
    x = np.asarray(x, dtype=float)
    if k == 2:
        return x * x
    seq = x * x * x if k == 3 else x * x * x * x
    fast = (x >= -11.0) & (x <= 11.0)
    if np.all(fast):
        return seq

    def _pow1(v):
        # C pow overflows to ±Inf silently; math.pow raises instead.
        try:
            return math.pow(v, float(k))
        except OverflowError:
            return math.copysign(math.inf, v) if k % 2 else math.inf

    out = np.where(fast, seq, 0.0)
    slow = np.where(~fast)[0]
    out[slow] = [_pow1(float(v)) for v in x[slow]]
    return out


def _rsum(a):
    """R's ``sum()`` over a double vector as compiled on arm64 (LDOUBLE ==
    double there, ref/r-base/arithmetic.c-style strict left-to-right
    accumulation): numpy's pairwise ``np.sum`` rounds differently in the
    last ulp; ``cumsum`` accumulates sequentially like R."""
    a = np.asarray(a, dtype=float).ravel()
    return float(np.cumsum(a)[-1]) if a.size else 0.0


def _cpois_dpois_log(x, lam):
    """R ``dpois(x, λ, log=TRUE)`` via the nmath port (rust-accelerated;
    full dpois.c edge semantics — cpois's Dd probes x−1…x−4, so negative
    x → log(0) = −Inf is a live input here)."""
    return _nmath._disp("dpois", _nmath.dpois,
                        [np.asarray(x, dtype=float),
                         np.asarray(lam, dtype=float)], (True,))


def _cpois_ppois_log(x, lam, lower_tail=True):
    """R ``ppois(x, λ, lower.tail=, log.p=TRUE)`` via the nmath port."""
    return _nmath._disp("ppois", _nmath.ppois,
                        [np.asarray(x, dtype=float),
                         np.asarray(lam, dtype=float)], (lower_tail, True))


def _dppois(y0, y1, mu, log_p=True):
    """mgcv ``dppois`` (efam.r:312-339): log(ppois(y1,μ) − ppois(y0,μ))
    without underflow to log(0). Each bound's CDF is taken in whichever
    tail is small (yᵢ < μ ⇒ lower), then the difference is assembled per
    tail-combination — same-tail pairs through ``logexm1`` cancellation
    control; the only possible opposite-tail pair (p1 upper, p0 lower)
    directly. y1 < 0 ⇒ both probabilities 0 ⇒ log(0) = −Inf."""
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    mu = np.asarray(mu, dtype=float)
    y0, y1, mu = np.broadcast_arrays(y0, y1, mu)
    p0 = np.zeros(y1.shape)
    p1 = np.zeros(y1.shape)
    p = np.zeros(y1.shape)
    i1 = y1 < mu                    # if !i1 compute log(1-p1)
    p1[i1] = _cpois_ppois_log(y1[i1], mu[i1], True)
    p1[~i1] = _cpois_ppois_log(y1[~i1], mu[~i1], False)
    i0 = y0 < mu                    # if !i0 compute log(1-p0)
    p0[i0] = _cpois_ppois_log(y0[i0], mu[i0], True)
    p0[~i0] = _cpois_ppois_log(y0[~i0], mu[~i0], False)
    free = np.ones(y1.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ii = y1 < 0                 # both prob 0
        p[ii] = -np.inf
        free[ii] = False
        ii = i1 & (y0 < 0) & free   # y0 prob is zero, y1 lower tail
        p[ii] = p1[ii]
        free[ii] = False
        ii = ~i1 & (y0 < 0) & free  # y0 prob is zero, y1 upper tail
        p[ii] = np.log(1.0 - np.exp(p1[ii]))
        free[ii] = False
        ii = i1 & i0 & free         # both lower tail
        p[ii] = p0[ii] + _logexm1(p1[ii] - p0[ii])
        free[ii] = False
        ii = ~i1 & ~i0 & free       # both upper tail
        p[ii] = p1[ii] + _logexm1(p0[ii] - p1[ii])
        free[ii] = False
        # remainder are opposite tails — no problem with cancellation
        ii = ~i1 & i0 & free        # p1 upper p0 lower (converse impossible)
        p[ii] = np.log(1.0 - np.exp(p1[ii]) - np.exp(p0[ii]))
    return p if log_p else np.exp(p)


def _cpois_dev_resids(y, mu, censor):
    """mgcv cpois ``dev.resids`` (efam.r:362-386): the proper deviance
    −2·(logLik − l_sat) per datum, by censoring case. ``wt``/``theta``
    play no role (φ ≡ 1, no family parameters). The interval case's
    saturated μ maximizes the interval probability analytically via the
    lgamma mean (efam.r:373-374)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    d = np.zeros(y.shape[0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if iu.size:
            d[iu] = 2.0 * (_cpois_dpois_log(y[iu], y[iu])
                           - _cpois_dpois_log(y[iu], mu[iu]))
        if ii.size:
            y1 = np.maximum(yat[ii], y[ii])
            y0 = np.minimum(yat[ii], y[ii])
            fy1 = np.floor(y1)
            fy0 = np.floor(y0)
            musat = np.exp((_lgammafn_arr(fy1 + 1.0)
                            - _lgammafn_arr(fy0 + 1.0)) / (fy1 - fy0))
            d[ii] = 2.0 * (_dppois(y0, y1, musat) - _dppois(y0, y1, mu[ii]))
        if il.size:   # left censored (sat lik is 1)
            d[il] = -2.0 * _cpois_ppois_log(y[il], mu[il], True)
        if ir.size:   # right censored
            d[ir] = -2.0 * _cpois_ppois_log(y[ir], mu[ir], False)
    return d


def _cpois_Dd(y, mu, censor, level=0):
    """mgcv cpois ``Dd`` (efam.r:388-443): derivatives of the cpois
    deviance w.r.t. μ only — there are no θ blocks at any level. Each
    case reduces to ratios f_k of k-shifted densities/interval-probs/
    CDF-tails against the unshifted one; the interval case's
    ``is.finite(yat·y)`` index test is mgcv's own quirk (0·∞ → NaN rows
    fall through to the pure left/right cases)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    yat = y if censor is None else np.asarray(censor, dtype=float)
    # get case indices... (efam.r:391-396)
    iu = np.where(yat == y)[0]
    with np.errstate(invalid="ignore"):
        ii = np.where(np.isfinite(yat * y) & (yat != y))[0]
    il = np.where(yat == -np.inf)[0]
    ir = np.where(yat == np.inf)[0]
    n = mu.shape[0]
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    f3 = np.zeros(n) if level > 0 else None
    f4 = np.zeros(n) if level > 1 else None

    _es = dict(divide="ignore", invalid="ignore", over="ignore")

    if iu.size:  # uncensored
        yiu = y[iu]
        miu = mu[iu]
        with np.errstate(**_es):
            lf = _cpois_dpois_log(yiu, miu)
            f1[iu] = np.exp(_cpois_dpois_log(yiu - 1.0, miu) - lf)
            f2[iu] = np.exp(_cpois_dpois_log(yiu - 2.0, miu) - lf)
            if level > 0:
                f3[iu] = np.exp(_cpois_dpois_log(yiu - 3.0, miu) - lf)
            if level > 1:
                f4[iu] = np.exp(_cpois_dpois_log(yiu - 4.0, miu) - lf)

    if ii.size:  # interval censored
        y0 = np.minimum(y[ii], yat[ii])
        y1 = np.maximum(y[ii], yat[ii])
        mii = mu[ii]
        with np.errstate(**_es):
            lg = _dppois(y0, y1, mii)
            f1[ii] = np.exp(_dppois(y0 - 1.0, y1 - 1.0, mii) - lg)
            f2[ii] = np.exp(_dppois(y0 - 2.0, y1 - 2.0, mii) - lg)
            if level > 0:
                f3[ii] = np.exp(_dppois(y0 - 3.0, y1 - 3.0, mii) - lg)
            if level > 1:
                f4[ii] = np.exp(_dppois(y0 - 4.0, y1 - 4.0, mii) - lg)

    for lt in (True, False):  # do left then right censoring...
        idx = il if lt else ir
        if idx.size:
            yil = y[idx]
            mil = mu[idx]
            with np.errstate(**_es):
                lf = _cpois_ppois_log(yil, mil, lt)
                f1[idx] = np.exp(_cpois_ppois_log(yil - 1.0, mil, lt) - lf)
                f2[idx] = np.exp(_cpois_ppois_log(yil - 2.0, mil, lt) - lf)
                if level > 0:
                    f3[idx] = np.exp(
                        _cpois_ppois_log(yil - 3.0, mil, lt) - lf)
                if level > 1:
                    f4[idx] = np.exp(
                        _cpois_ppois_log(yil - 4.0, mil, lt) - lf)

    # ^ sites through _rpow_int — R_pow, not numpy ** (see its docstring).
    r = {"Dmu": -2.0 * (f1 - 1.0), "Dmu2": -2.0 * (f2 - _rpow_int(f1, 2))}
    r["EDmu2"] = r["Dmu2"]
    if level > 0:
        r["Dmu3"] = -2.0 * (f3 - 3.0 * f1 * f2 + 2.0 * _rpow_int(f1, 3))
    if level > 1:
        r["Dmu4"] = -2.0 * (f4 - 4.0 * f3 * f1
                            + 12.0 * _rpow_int(f1, 2) * f2
                            - 3.0 * _rpow_int(f2, 2)
                            - 6.0 * _rpow_int(f1, 4))
    return r


def _cpois_aic(y, mu, censor):
    """mgcv cpois ``aic`` (efam.r:445-465): −2·logLik (no saturated
    reference). NOTE the uncensored selector here is ``is.na(yat) |
    yat==y`` (dev.resids uses plain ``yat==y``) — replicated verbatim."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    yat = y if censor is None else np.asarray(censor, dtype=float)
    d = np.zeros(y.shape[0])
    iu = np.where(np.isnan(yat) | (yat == y))[0]
    if iu.size:
        d[iu] = _cpois_dpois_log(y[iu], mu[iu])
    ii = np.where(np.isfinite(yat) & (yat != y))[0]
    if ii.size:
        y1 = np.maximum(yat[ii], y[ii])
        y0 = np.minimum(yat[ii], y[ii])
        d[ii] = _dppois(y0, y1, mu[ii])
    il = np.where(yat == -np.inf)[0]
    if il.size:
        d[il] = _cpois_ppois_log(y[il], mu[il], True)
    ir = np.where(yat == np.inf)[0]
    if ir.size:
        d[ir] = _cpois_ppois_log(y[ir], mu[ir], False)
    return -2.0 * _rsum(d)


def _cpois_ls_val(y, censor):
    """mgcv cpois ``ls`` (efam.r:467-491): the saturated log-likelihood
    VALUE — uncensored rows contribute dpois(y,y), interval rows the
    dppois at the saturated μ; left/right-censored rows are exactly 0,
    as are all θ/scale derivatives."""
    y = np.asarray(y, dtype=float)
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    d = np.zeros(y.shape[0])
    if iu.size:
        d[iu] = _cpois_dpois_log(y[iu], y[iu])
    if ii.size:
        y1 = np.maximum(yat[ii], y[ii])
        y0 = np.minimum(yat[ii], y[ii])
        fy1 = np.floor(y1)
        fy0 = np.floor(y0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            mus = np.exp((_lgammafn_arr(fy1 + 1.0)
                          - _lgammafn_arr(fy0 + 1.0)) / (fy1 - fy0))
        d[ii] = _dppois(y0, y1, mus)
    return _rsum(d)


class cpois(Family):
    """Censored Poisson extended family — port of mgcv ``cpois()``
    (efam.r:344-537) with its ``dppois`` helper (efam.r:312-339).

    The response is the same 2-column ``cbind(y, yat)`` censor encoding
    as :class:`cnorm`: ``yat == y`` uncensored, finite ``yat ≠ y``
    interval-censored (the count lies in [min(y,yat), max(y,yat)]),
    ``yat == −∞`` left-censored (count ≤ y), ``yat == +∞`` right-censored
    (count > y). A 1-column response is all-uncensored (plain Poisson
    likelihood).

    There are NO family parameters (``n_theta = 0``; mgcv's getTheta
    returns NULL) and the scale is fixed at φ = 1. ``dev_resids`` is the
    proper deviance (the interval case's saturated μ comes from the
    analytic lgamma mean) and ``ls`` the matching nonzero saturated
    log-lik with zero derivatives. mgcv exports no variance/rd/qf slots
    for cpois — the base-class NotImplementedError/None stand in for
    R's NULL. okLinks: log (default), identity, sqrt.
    """
    name = "cpois"
    canonical_link_name = "log"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    n_theta = 0
    _OK_LINKS = ("log", "identity", "sqrt")

    def __init__(self, link: str = "log"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for cpois family; available '
                f'links are {self._OK_LINKS}')
        # bam's bgam.fit θ-update gate (bam.r:1204-1206): estimate.theta
        # runs between PIRLS iters whenever the extended family has free
        # θ (``family$n.theta>0``; the ``scale<0`` leg never fires here —
        # the censored families carry no ``scale`` slot, so bgam.fit's
        # scale resolves to 1, bam.r:924).
        self.estimate_theta_callback = self.n_theta > 0
        self._censor = None
        self._censorfull = None
        super().__init__(link=link)

    # ----- θ accessors / censoring bound ---------------------------------

    def set_theta(self, values) -> None:
        # mgcv putTheta is an empty function — accept and ignore.
        pass

    def get_theta(self, trans: bool = False) -> np.ndarray:
        # mgcv getTheta has an empty body (returns NULL) — no parameters.
        return np.zeros(0)

    def set_censor(self, censor) -> None:
        """Stash the censoring bound (column 1 of the ``cbind(y, yat)``
        response), aligned with the full response. ``None`` ⇒ all
        uncensored (mgcv's ``attr(y,"censor")`` being NULL)."""
        self._censor = (None if censor is None
                        else np.asarray(censor, dtype=float))
        self._censorfull = None

    def set_ind(self, ind) -> None:
        """mgcv ``subsety`` (efam.r — identical body on all censored
        families): bam's chunk loop subsets the response *and its censor
        attribute* per chunk. hea carries ``attr(y,"censor")`` on the
        family, so the windowing lands here: stash the full bound on
        first use, window to ``ind``, restore with ``ind=None``
        (mirrors gfam's ``setInd``, gfam.r:78-86)."""
        if self._censorfull is None:
            if ind is None or self._censor is None:
                return          # never windowed / uncensored: nothing to do
            self._censorfull = self._censor
        self._censor = (self._censorfull if ind is None
                        else self._censorfull[np.asarray(ind, dtype=int)])

    # ----- deviance / Dd / aic -------------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        return _cpois_dev_resids(y, mu, self._censor)

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        return _cpois_Dd(y, mu, self._censor, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        return _cpois_aic(y, mu, self._censor)

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        ls = _cpois_ls_val(y, self._censor)
        n = np.asarray(y).shape[0]
        # mgcv: lsth1=0, LSTH1=matrix(0,n,1), lsth2=0 (efam.r:487-491).
        return {"ls": ls, "lsth1": np.array([0.0]),
                "lsth2": np.array([[0.0]]), "LSTH1": np.zeros((n, 1))}

    def ls(self, y, wt, scale):
        return np.array([_cpois_ls_val(y, self._censor), 0.0, 0.0])

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        # mgcv cpois initialize (efam.r:493-500): the matrix split happens
        # at intake; mustart = y (identity) or pmax(y, min(y>0)) — min of
        # the LOGICAL y>0 (0 if any y ≤ 0, else 1), as in cnorm.
        y = np.asarray(y, dtype=float)
        if self.link.name == "identity":
            return y.copy()
        return np.maximum(y, float(np.min(y > 0)))

    def validmu(self, mu) -> bool:
        # efam.r:359-360: identity → finite; log → μ>0; sqrt → μ≥0.
        mu = np.asarray(mu, dtype=float)
        if self.link.name == "identity":
            return bool(np.all(np.isfinite(mu)))
        if self.link.name == "log":
            return bool(np.all(mu > 0))
        return bool(np.all(mu >= 0))

    # ----- postproc ------------------------------------------------------

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # mgcv cpois postproc (efam.r:502-511): null deviance via
        # find.null.dev; the family label stays plain "cpois".
        null_dev = find_null_dev(self, y, eta=linear_predictors,
                                 offset=offset, weights=prior_weights)
        return {"null_deviance": null_dev, "family_name": "cpois"}

    def __repr__(self):
        return f"cpois(link={self.link.name})"


# ---------------------------------------------------------------------------
# clog (censored logistic) — mgcv ``clog()`` (efam.r:2192-2612, Chris Shen).
#
# Same 2-column ``cbind(y, yat)`` censor encoding as cnorm/cpois; one
# log-scale θ (σ = e^θ, per-datum s_i = e^θ/√wt_i), estimated jointly
# unless ``clog(theta>0)`` fixes it; φ ≡ 1. The local ``log1pexp``/
# ``log1mexp`` helpers are ported VERBATIM, including mgcv's own quirks:
# log1pexp's first mask reads ``x <= 37`` (not −37), so the band
# 33.3 < x ≤ 37 returns exp(x) (a plain typo in mgcv — replicated for
# bit parity); log1mexp leaves a ≤ 0 at 0. The ``aic`` slot contains
# only the SATURATED −2logLik pieces (no μ anywhere) — that is what
# mgcv reports (gam.fit4.r:794 uses the slot verbatim + 2·edf).
# ---------------------------------------------------------------------------


_M_LN2_CLOG = float(np.log(2.0))


def _clog_log1pexp(x):
    """clog's local ``log1pexp`` (efam.r:2235-2241 et al.): log(1+e^x)
    by Mächler bands, with mgcv's sequential-overwrite quirk — the first
    mask is ``x <= 37`` so 33.3 < x ≤ 37 keeps exp(x)."""
    x = np.asarray(x, dtype=float)
    result = x.copy()
    with np.errstate(over="ignore"):
        ii = x <= 37.0
        result[ii] = np.exp(x[ii])
        ii = (-37.0 < x) & (x <= 18.0)
        result[ii] = np.log1p(np.exp(x[ii]))
        ii = (18.0 < x) & (x <= 33.3)
        result[ii] = x[ii] + np.exp(-x[ii])
    return result


def _clog_log1mexp(a):
    """clog's local ``log1mexp`` (efam.r:2243-2248): log(1−e^(−a)) for
    a > 0 by the expm1/log1p split; a ≤ 0 rows stay 0 (mgcv inits the
    result to zeros and never touches them)."""
    a = np.asarray(a, dtype=float)
    result = np.zeros(a.shape)
    with np.errstate(divide="ignore"):
        ii = (0.0 < a) & (a <= _M_LN2_CLOG)
        result[ii] = np.log(-np.expm1(-a[ii]))
        ii = a > _M_LN2_CLOG
        result[ii] = np.log1p(-np.exp(-a[ii]))
    return result


def _clog_dev_resids(y, mu, wt, theta, censor):
    """mgcv clog ``dev.resids`` (efam.r:2233-2302): the proper deviance
    per datum by censoring case, s_i = e^θ/√wt_i."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(theta)
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    d = np.zeros(y.shape[0])
    if iu.size:  # uncensored
        si = np.exp(theta) / np.sqrt(wt[iu])
        mui = mu[iu]
        yi = y[iu]
        d[iu] = 2.0 * (-2.0 * _M_LN2_CLOG + ((yi - mui) / si)
                       + 2.0 * _clog_log1pexp(-(yi - mui) / si))
    if ii.size:  # interval censored
        si = np.exp(theta) / np.sqrt(wt[ii])
        mui = mu[ii]
        yl = np.minimum(y[ii], yat[ii])
        yr = np.maximum(y[ii], yat[ii])
        lm = (((yr - yl) / (2.0 * si)) + _clog_log1mexp((yr - yl) / (2.0 * si))
              - _clog_log1pexp((yr - yl) / (2.0 * si)))
        line = (_clog_log1mexp((yr - yl) / si) - _clog_log1pexp((yl - mui) / si)
                - _clog_log1pexp(-(yr - mui) / si))
        d[ii] = 2.0 * (lm - line)
    if il.size:  # left censored
        si = np.exp(theta) / np.sqrt(wt[il])
        mui = mu[il]
        yr = y[il]
        line = -_clog_log1pexp(-(yr - mui) / si)
        d[il] = 2.0 * (0.0 - line)
    if ir.size:  # right censored
        si = np.exp(theta) / np.sqrt(wt[ir])
        mui = mu[ir]
        yl = y[ir]
        line = (-(yl - mui) / si) - _clog_log1pexp(-(yl - mui) / si)
        d[ir] = 2.0 * (0.0 - line)
    return d


def _clog_Dd(y, mu, theta, wt, censor, level=0):
    """mgcv clog ``Dd`` (efam.r:2305-2459): derivatives of the clog
    deviance w.r.t. μ and the log-scale θ, by censoring case. Verbatim
    port — including the interval case's ``(4/si^2)`` Dmu3 scaling as
    written in mgcv; ``^`` sites ride :func:`_rpow_int`."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(np.asarray(theta, dtype=float).reshape(-1)[0])
    yat = y if censor is None else np.asarray(censor, dtype=float)
    iu = np.where(yat == y)[0]
    with np.errstate(invalid="ignore"):
        ii = np.where(np.isfinite(yat * y) & (yat != y))[0]
    il = np.where(yat == -np.inf)[0]
    ir = np.where(yat == np.inf)[0]

    n = mu.shape[0]
    Dmu = np.zeros(n)
    Dmu2 = np.zeros(n)
    if level > 0:
        Dth = np.zeros(n)
        Dmuth = np.zeros(n)
        Dmu3 = np.zeros(n)
        Dmu2th = np.zeros(n)
    if level > 1:
        Dmu4 = np.zeros(n)
        Dth2 = np.zeros(n)
        Dmuth2 = np.zeros(n)
        Dmu2th2 = np.zeros(n)
        Dmu3th = np.zeros(n)

    _es = dict(divide="ignore", invalid="ignore", over="ignore")

    if iu.size:  # uncensored
        si = np.exp(theta) / np.sqrt(wt[iu])
        yi = y[iu]
        mui = mu[iu]
        with np.errstate(**_es):
            alphai = 1.0 / (2.0 + np.expm1((yi - mui) / si))
            Dmui = 2.0 / si * (2.0 * alphai - 1.0)
            Dmu[iu] = Dmui
            Dmu2i = 4.0 / _rpow_int(si, 2) * alphai * (1.0 - alphai)
            Dmu2[iu] = Dmu2i
            if level > 0:
                Dmuthi = -Dmui + (yi - mui) * Dmu2i
                Dmuth[iu] = Dmuthi
                Dth[iu] = (yi - mui) * Dmui
                Dmu3i = (-1.0 / 2.0) * Dmui * Dmu2i
                Dmu3[iu] = Dmu3i
                Dmu2thi = (-2.0 * Dmu2i
                           + (1.0 / 2.0) * (mui - yi) * Dmui * Dmu2i)
                Dmu2th[iu] = Dmu2thi
            if level > 1:
                Dth2[iu] = (yi - mui) * Dmuthi
                Dmuth2[iu] = -Dmuthi + (yi - mui) * Dmu2thi
                Dmu4[iu] = (-1.0 / 2.0) * (_rpow_int(Dmu2i, 2)
                                           + Dmui * Dmu3i)
                Dmu3th[iu] = (-1.0 / 2.0) * (Dmu2i * Dmuthi
                                             + Dmui * Dmu2thi)
                Dmu2th2[iu] = (-2.0 * Dmu2thi
                               + (1.0 / 2.0) * (mui - yi)
                               * (Dmui * Dmu2thi + Dmu2i * Dmuthi))

    if ii.size:  # interval censored
        yl = np.minimum(y[ii], yat[ii])
        yr = np.maximum(y[ii], yat[ii])
        si = np.exp(theta) / np.sqrt(wt[ii])
        mui = mu[ii]
        with np.errstate(**_es):
            alphai = 1.0 / (2.0 + np.expm1(-(yr - mui) / si))
            betai = 1.0 / (2.0 + np.expm1(-(yl - mui) / si))
            Dmui = 2.0 / si * (1.0 - alphai - betai)
            Dmu[ii] = Dmui
            Dmu2i = (2.0 / _rpow_int(si, 2)
                     * (alphai - _rpow_int(alphai, 2)
                        + betai - _rpow_int(betai, 2)))
            Dmu2[ii] = Dmu2i
            if level > 0:
                Dmuthi = (-Dmui + 2.0 / _rpow_int(si, 2)
                          * ((yr - mui) * (alphai - _rpow_int(alphai, 2))
                             + (yl - mui) * (betai - _rpow_int(betai, 2))))
                Dmuth[ii] = Dmuthi
                lmth = ((yr - yl) * (1.0 / si)
                        * (1.0 / (np.expm1(-(yr - yl) / (2.0 * si))
                                  - np.expm1((yr - yl) / (2.0 * si)))))
                lth = ((1.0 / si)
                       * (-(yr - yl) * (1.0 / np.expm1((yr - yl) / si))
                          + (yl - mui)
                          * (1.0 / (np.expm1((mui - yl) / si) + 2.0))
                          - (yr - mui)
                          * (1.0 / (np.expm1((yr - mui) / si) + 2.0))))
                Dth[ii] = 2.0 * (lmth - lth)
                Dmu3i = (4.0 / _rpow_int(si, 2)
                         * (_rpow_int(alphai, 2) - _rpow_int(alphai, 3)
                            + _rpow_int(betai, 2) - _rpow_int(betai, 3))
                         - Dmu2i)
                Dmu3[ii] = Dmu3i
                Dmu2thi = (-2.0 * Dmu2i + (-1.0 / si) * (Dmuthi + Dmui)
                           + 4.0 / _rpow_int(si, 3)
                           * ((yr - mui) * (_rpow_int(alphai, 2)
                                            - _rpow_int(alphai, 3))
                              + (yl - mui) * (_rpow_int(betai, 2)
                                              - _rpow_int(betai, 3))))
                Dmu2th[ii] = Dmu2thi
            if level > 1:
                lmth2 = (-lmth - (1.0 / 2.0) * _rpow_int(yr - yl, 2)
                         * (1.0 / _rpow_int(si, 2))
                         * _rpow_int(1.0 / (np.expm1(-(yr - yl) / (2.0 * si))
                                            + -np.expm1((yr - yl)
                                                        / (2.0 * si))), 2)
                         * (2.0 + np.expm1(-(yr - yl) / (2.0 * si))
                            + np.expm1((yr - yl) / (2.0 * si))))
                lth2 = (-lth - (1.0 / _rpow_int(si, 2))
                        * (_rpow_int(yr - yl, 2)
                           * (1.0 / np.expm1((yr - yl) / si))
                           * (1.0 / -np.expm1(-(yr - yl) / si))
                           + _rpow_int(yl - mui, 2)
                           * (1.0 / (np.expm1((mui - yl) / si) + 2.0))
                           * (1.0 / (np.expm1((yl - mui) / si) + 2.0))
                           + _rpow_int(yr - mui, 2)
                           * (1.0 / (np.expm1((yr - mui) / si) + 2.0))
                           * (1.0 / (np.expm1((mui - yr) / si) + 2.0))))
                Dth2[ii] = 2.0 * (lmth2 - lth2)
                Dmuth2i = (-3.0 * Dmuthi - 2.0 * Dmui
                           + (-2.0 / _rpow_int(si, 3))
                           * (_rpow_int(yr - mui, 2) * (1.0 - 2.0 * alphai)
                              * (alphai - _rpow_int(alphai, 2))
                              + _rpow_int(yl - mui, 2) * (1.0 - 2.0 * betai)
                              * (betai - _rpow_int(betai, 2))))
                Dmuth2[ii] = Dmuth2i
                Dmu4[ii] = ((-4.0 / _rpow_int(si, 3))
                            * ((2.0 * alphai - 3.0 * _rpow_int(alphai, 2))
                               * (alphai - _rpow_int(alphai, 2))
                               + (2.0 * betai - 3.0 * _rpow_int(betai, 2))
                               * (betai - _rpow_int(betai, 2)))
                            - Dmu3i)
                Dmu3th[ii] = (-2.0 * Dmu3i + 2.0 * (1.0 / _rpow_int(si, 3))
                              * ((yr - mui)
                                 * (alphai - _rpow_int(alphai, 2))
                                 * (1.0 - 6.0 * alphai
                                    + 6.0 * _rpow_int(alphai, 2))
                                 + (yl - mui)
                                 * (betai - _rpow_int(betai, 2))
                                 * (1.0 - 6.0 * betai
                                    + 6.0 * _rpow_int(betai, 2))))
                Dmu2th2[ii] = (-2.0 * Dmu2thi
                               + (1.0 / si) * (Dmui - Dmuth2i)
                               + (-12.0 / _rpow_int(si, 3))
                               * ((yr - mui) * (_rpow_int(alphai, 2)
                                                - _rpow_int(alphai, 3))
                                  + (yl - mui) * (_rpow_int(betai, 2)
                                                  - _rpow_int(betai, 3)))
                               + (-4.0 / _rpow_int(si, 4))
                               * (_rpow_int(yr - mui, 2)
                                  * (2.0 * alphai
                                     - 3.0 * _rpow_int(alphai, 2))
                                  * (alphai - _rpow_int(alphai, 2))
                                  + _rpow_int(yl - mui, 2)
                                  * (2.0 * betai
                                     - 3.0 * _rpow_int(betai, 2))
                                  * (betai - _rpow_int(betai, 2))))

    if il.size:  # left censored
        si = np.exp(theta) / np.sqrt(wt[il])
        yr = y[il]
        mui = mu[il]
        with np.errstate(**_es):
            alphai = 1.0 / (2.0 + np.expm1(-(yr - mui) / si))
            Dmui = 2.0 / si * (1.0 - alphai)
            Dmu[il] = Dmui
            Dmu2i = 2.0 / _rpow_int(si, 2) * (alphai - _rpow_int(alphai, 2))
            Dmu2[il] = Dmu2i
            if level > 0:
                Dmuthi = -Dmui + (yr - mui) * Dmu2i
                Dmuth[il] = Dmuthi
                Dth[il] = (yr - mui) * Dmui
                Dmu3i = (1.0 / si) * (2.0 * alphai - 1.0) * Dmu2i
                Dmu3[il] = Dmu3i
                Dmu2thi = (-2.0 * Dmu2i
                           + (1.0 / si) * (yr - mui)
                           * (2.0 * alphai - 1.0) * Dmu2i)
                Dmu2th[il] = Dmu2thi
            if level > 1:
                Dth2[il] = (yr - mui) * Dmuthi
                Dmuth2[il] = -Dmuthi + (yr - mui) * Dmu2thi
                Dmu4[il] = (-_rpow_int(Dmu2i, 2)
                            + (1.0 / si) * (2.0 * alphai - 1.0) * Dmu3i)
                Dmu3th[il] = (-(yr - mui) * _rpow_int(Dmu2i, 2)
                              + (1.0 / si) * (2.0 * alphai - 1.0)
                              * (Dmu2thi - Dmu2i))
                Dmu2th2[il] = (-2.0 * Dmu2thi
                               - _rpow_int(yr - mui, 2) * _rpow_int(Dmu2i, 2)
                               + (1.0 / si) * (yr - mui)
                               * (2.0 * alphai - 1.0) * (Dmu2thi - Dmu2i))

    if ir.size:  # right censored
        si = np.exp(theta) / np.sqrt(wt[ir])
        yl = y[ir]
        mui = mu[ir]
        with np.errstate(**_es):
            betai = 1.0 / (2.0 + np.expm1(-(yl - mui) / si))
            Dmui = -(2.0 / si) * betai
            Dmu[ir] = Dmui
            Dmu2i = 2.0 / _rpow_int(si, 2) * (betai - _rpow_int(betai, 2))
            Dmu2[ir] = Dmu2i
            if level > 0:
                Dmuthi = -Dmui + (yl - mui) * Dmu2i
                Dmuth[ir] = Dmuthi
                Dth[ir] = (yl - mui) * Dmui
                Dmu3i = (-1.0 / si) * (1.0 - 2.0 * betai) * Dmu2i
                Dmu3[ir] = Dmu3i
                Dmu2thi = -(2.0 + (1.0 / si) * (yl - mui)
                            * (1.0 - 2.0 * betai)) * Dmu2i
                Dmu2th[ir] = Dmu2thi
            if level > 1:
                Dth2[ir] = (yl - mui) * Dmuthi
                Dmuth2i = -Dmuthi + (yl - mui) * Dmu2thi
                Dmuth2[ir] = Dmuth2i
                Dmu4[ir] = (-_rpow_int(Dmu2i, 2)
                            + (-1.0 / si) * (1.0 - 2.0 * betai) * Dmu3i)
                Dmu3th[ir] = ((1.0 / si) * (1.0 - 2.0 * betai)
                              * (Dmu2i - Dmu2thi)
                              - (yl - mui) * _rpow_int(Dmu2i, 2))
                Dmu2th2[ir] = ((1.0 / si) * (yl - mui) * (1.0 - 2.0 * betai)
                               * (Dmu2i - Dmu2thi)
                               + -_rpow_int(yl - mui, 2)
                               * _rpow_int(Dmu2i, 2)
                               - 2.0 * Dmu2thi)

    EDmu2t = Dmu2.copy()
    EDmu2t[Dmu2 < 0] = 0.0

    r = {"Dmu": Dmu, "Dmu2": Dmu2, "EDmu2": EDmu2t}
    if level > 0:
        r["Dth"] = Dth
        r["Dmuth"] = Dmuth
        r["Dmu3"] = Dmu3
        r["Dmu2th"] = Dmu2th
        r["EDmu2th"] = Dmu2th
    if level > 1:
        r["Dmu4"] = Dmu4
        r["Dth2"] = Dth2
        r["Dmuth2"] = Dmuth2
        r["Dmu2th2"] = Dmu2th2
        r["Dmu3th"] = Dmu3th
    return r


def _clog_aic(y, wt, theta, censor):
    """mgcv clog ``aic`` (efam.r:2462-2515): the slot contains ONLY the
    saturated −2logLik pieces (μ never enters!) — that is literally what
    mgcv reports as the model AIC contribution (gam.fit4.r:794 uses the
    slot verbatim; estimate.gam adds 2·edf). Replicated bug-for-bug."""
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(theta)
    th = theta - 0.5 * np.log(wt)
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    a = np.zeros(y.shape[0])
    if iu.size:
        si = np.exp(theta) / np.sqrt(wt[iu])
        lm = -np.log1p(si - 1.0) - 2.0 * _M_LN2_CLOG
        a[iu] = -2.0 * lm + 2.0 * th[iu]
    if ii.size:
        si = np.exp(theta) / np.sqrt(wt[ii])
        yl = np.minimum(y[ii], yat[ii])
        yr = np.maximum(y[ii], yat[ii])
        lm = (((yr - yl) / (2.0 * si))
              + _clog_log1mexp((yr - yl) / (2.0 * si))
              - _clog_log1pexp((yr - yl) / (2.0 * si)))
        a[ii] = -2.0 * lm + 2.0 * th[ii]
    if il.size:
        a[il] = 2.0 * th[il]
    if ir.size:
        a[ir] = 2.0 * th[ir]
    return _rsum(a)


def _clog_ls(y, wt, theta, censor):
    """mgcv clog ``ls`` (efam.r:2517-2564): saturated log-lik with
    NONZERO θ-derivatives (uncensored rows: ∂l_sat/∂θ = −1; interval
    rows the lmth/lmth2 machinery). Left/right rows are zero."""
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = float(theta)
    yat, iu, ii, il, ir = _cnorm_cases(y, censor)
    n = y.shape[0]
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    if iu.size:
        si = np.exp(theta) / np.sqrt(wt[iu])
        l0[iu] = -np.log1p(si - 1.0) - 2.0 * _M_LN2_CLOG
        l1[iu] = -1.0
    if ii.size:
        si = np.exp(theta) / np.sqrt(wt[ii])
        yl = np.minimum(y[ii], yat[ii])
        yr = np.maximum(y[ii], yat[ii])
        l0[ii] = (((yr - yl) / (2.0 * si))
                  + _clog_log1mexp((yr - yl) / (2.0 * si))
                  - _clog_log1pexp((yr - yl) / (2.0 * si)))
        lmth = ((yr - yl) * (1.0 / si)
                * (1.0 / (np.expm1(-(yr - yl) / (2.0 * si))
                          - np.expm1((yr - yl) / (2.0 * si)))))
        l1[ii] = lmth
        lmth2 = (-lmth - (1.0 / 2.0) * _rpow_int(yr - yl, 2)
                 * (1.0 / _rpow_int(si, 2))
                 * _rpow_int(1.0 / (np.expm1(-(yr - yl) / (2.0 * si))
                                    + -np.expm1((yr - yl) / (2.0 * si))), 2)
                 * (2.0 + np.expm1(-(yr - yl) / (2.0 * si))
                    + np.expm1((yr - yl) / (2.0 * si))))
        l2[ii] = lmth2
    return (_rsum(l0), _rsum(l1), l1, _rsum(l2))


class clog(Family):
    """Censored logistic extended family — port of mgcv ``clog()``
    (efam.r:2192-2612).

    The single linear predictor ``μ`` is the logistic location; the
    log-scale ``θ`` (σ = e^θ, per-datum s_i = σ/√wt_i) is estimated
    jointly with the smoothing parameters (``clog(theta=σ)`` with σ > 0
    fixes it; σ < 0 supplies |σ| as the starting value). The response is
    the 2-column ``cbind(y, yat)`` censor encoding shared with
    :class:`cnorm` / :class:`cpois`. ``dev_resids`` is the proper
    deviance and ``ls`` a nonzero saturated log-lik with nonzero
    θ-derivatives. mgcv exports no variance/rd/qf slots. okLinks:
    identity (default), log, sqrt.
    """
    name = "clog"
    canonical_link_name = "identity"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    n_theta = 1
    _OK_LINKS = ("log", "identity", "sqrt")

    def __init__(self, theta=None, link: str = "identity"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for clog family; available '
                f'links are {self._OK_LINKS}')
        # mgcv θ intake (efam.r:2213-2221): θ>0 fixed (store log θ,
        # n_theta=0); θ<0 an initial value (store log(−θ)); 0/None → 0.
        if theta is not None and float(theta) != 0.0:
            t = float(theta)
            if t > 0:
                ini = float(np.log(t))
                self.n_theta = 0
            else:
                ini = float(np.log(-t))
        else:
            ini = 0.0
        self._theta = np.array([ini], dtype=float)
        # bam's bgam.fit θ-update gate (bam.r:1204-1206): estimate.theta
        # runs between PIRLS iters whenever the extended family has free
        # θ (``family$n.theta>0``; the ``scale<0`` leg never fires here —
        # the censored families carry no ``scale`` slot, so bgam.fit's
        # scale resolves to 1, bam.r:924).
        self.estimate_theta_callback = self.n_theta > 0
        self._censor = None
        self._censorfull = None
        super().__init__(link=link)

    # ----- θ accessors / censoring bound ---------------------------------

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape[0] != 1:
            raise ValueError(
                f"clog.set_theta expects 1 param (log σ); got shape {v.shape}")
        self._theta = v.copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        th = self._theta.copy()
        return np.exp(th) if trans else th

    def set_censor(self, censor) -> None:
        """Stash the censoring bound (column 1 of the ``cbind(y, yat)``
        response). ``None`` ⇒ all uncensored."""
        self._censor = (None if censor is None
                        else np.asarray(censor, dtype=float))
        self._censorfull = None

    def set_ind(self, ind) -> None:
        """mgcv ``subsety`` (efam.r:2595): window the censor bound to the
        bam chunk rows; ``ind=None`` restores (see cnorm.set_ind)."""
        if self._censorfull is None:
            if ind is None or self._censor is None:
                return
            self._censorfull = self._censor
        self._censor = (self._censorfull if ind is None
                        else self._censorfull[np.asarray(ind, dtype=int)])

    # ----- deviance / Dd / aic -------------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        return _clog_dev_resids(y, mu, wt, th, self._censor)

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        return _clog_Dd(y, mu, theta, wt, self._censor, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        return _clog_aic(y, wt, th, self._censor)

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        th = float(self._theta[0] if theta is None
                   else np.asarray(theta, dtype=float).reshape(-1)[0])
        ls0, lsth1, l1_vec, lsth2 = _clog_ls(y, wt, th, self._censor)
        return {"ls": ls0, "lsth1": np.array([lsth1]),
                "lsth2": np.array([[lsth2]]),
                "LSTH1": l1_vec.reshape(-1, 1)}

    def ls(self, y, wt, scale):
        ls0, _, _, _ = _clog_ls(y, wt, float(self._theta[0]), self._censor)
        return np.array([ls0, 0.0, 0.0])

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        # mgcv clog initialize (efam.r:2566-2573): mustart = y (identity)
        # or pmax(y, min(y>0)) — the LOGICAL min, as in cnorm/cpois.
        y = np.asarray(y, dtype=float)
        if self.link.name == "identity":
            return y.copy()
        return np.maximum(y, float(np.min(y > 0)))

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu, dtype=float)
        if self.link.name == "identity":
            return bool(np.all(np.isfinite(mu)))
        return bool(np.all(mu > 0))

    # ----- postproc ------------------------------------------------------

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # mgcv clog postproc (efam.r:2575-2585): null deviance via
        # find.null.dev; family relabel "clog(σ)".
        null_dev = find_null_dev(self, y, eta=linear_predictors,
                                 offset=offset, weights=prior_weights)
        sig = f"{float(np.round(self.get_theta(True)[0], 3)):g}"
        return {"null_deviance": null_dev, "family_name": f"clog({sig})"}

    def __repr__(self):
        return f"clog(theta={self._theta}, link={self.link.name})"


# ---------------------------------------------------------------------------
# bcg (censored Box-Cox Gaussian) — mgcv ``bcg()`` (efam.r:1477-2170).
#
# Two θ parameters: the Box-Cox λ (natural scale) and log σ; the working
# model is z = bc(y, λ) ~ N(μ, s_i²) with s_i = e^θ₂/√wt_i. The censor
# encoding differs from cnorm: LEFT censoring is ``yat ≤ 0`` (bcg is for
# non-negative responses; y is censored below at 0), interval requires
# ``yat > 0``; right stays ``yat == +∞``. mgcv quirks preserved
# bit-for-bit (all live-R verified):
# * fixed θ (θ₂ > 0) stores ``c(θ₁, log(θ))`` — a LENGTH-3 .Theta whose
#   working log σ is log λ, not log σ (the requested σ is ignored);
# * θ₂ < 0 keeps the RAW (λ, θ₂) as the starting value — the
#   ``theta[2] <- log(-theta[2])`` line runs after iniTheta was taken
#   and is dead;
# * ``bc()``'s ``(y==Inf & λ<0)|(y==0 & λ>0)`` z-branch computes
#   ``z[ii] - 1/l[ii]`` and DISCARDS it (a bare expression, not an
#   assignment) — those rows keep the main-branch value;
# * dev.resids carries mgcv's ``attr(d,"sign") = sign(bc(y,λ)−μ)`` —
#   surfaced through ``residuals_extended`` (residuals.gam's reader).
# pnorm/dnorm/dpnorm ride the bit-exact nmath ports (pnorm5_vec/
# dnorm5_vec), NOT scipy's log_ndtr.
# ---------------------------------------------------------------------------


def _r_pow_scalar(x, p):
    """Scalar ``R_pow(x, p)`` (ref/r-base/arithmetic.c:204) for the
    general-exponent rows ``_rpow`` can't vectorize: the x==1/p==0/x==0
    edges precede libm pow; overflow is C-silent ±Inf."""
    if p == 2.0:
        return x * x
    if x == 1.0 or p == 0.0:
        return 1.0
    if x == 0.0:
        if p > 0.0:
            return 0.0
        if p < 0.0:
            return math.inf
        return p            # NaN
    if -11.0 <= x <= 11.0 and p == 3.0:
        return x * x * x
    if -11.0 <= x <= 11.0 and p == 4.0:
        return x * x * x * x
    if math.isfinite(x) and math.isfinite(p):
        try:
            return math.pow(x, p)
        except OverflowError:
            return math.inf if (x > 0 or p % 2 == 0) else -math.inf
    if math.isnan(x) or math.isnan(p):
        return x + p
    if not math.isfinite(x):
        if x > 0:
            return 0.0 if p < 0 else math.inf
        if math.isfinite(p) and p == math.floor(p):
            return 0.0 if p < 0 else (x if math.fmod(p, 2.0) != 0 else -x)
    if not math.isfinite(p):
        if x >= 0:
            if p > 0:
                return math.inf if x >= 1 else 0.0
            return math.inf if x < 1 else 0.0
    return math.nan


def _rpow(x, p):
    """R's elementwise ``x ^ p`` for a SCALAR p (R_pow per element):
    integer fast paths via :func:`_rpow_int`, everything else the scalar
    R_pow loop (libm pow with R's edges)."""
    p = float(p)
    if p in (2.0, 3.0, 4.0):
        return _rpow_int(x, int(p))
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape)
    of = out.ravel()
    for i, v in enumerate(x.ravel()):
        of[i] = _r_pow_scalar(float(v), p)
    return out


def _bcg_bc_z(y, la, ls_variant=False):
    """bcg's ``bc(y, λ)`` VALUE — the dev.resids/aic variant (zero-y
    masks use ``y == 0``; efam.r:1513-1534) or the ls variant
    (``y <= 0``; efam.r:2081-2103). λ is scalar in every live call.
    The ``(y==Inf & λ<0)|(y==0 & λ>0)`` branch is mgcv's dead bare
    expression — no assignment happens."""
    y = np.asarray(y, dtype=float)
    la = float(la)
    z = y.copy()
    nina = ~np.isnan(y)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logy = np.where(nina, np.log(y), y)
        if abs(la) < 1e-7:
            ylly = _rpow(y, la) * logy
            yl2 = ylly * logy
            z = np.where(nina,
                         ylly - yl2 * la / 2.0
                         + yl2 * logy * (la * la) / 6.0, z)
        else:
            z = np.where(nina, (_rpow(y, la) - 1.0) / la, z)
        # (y==Inf & l<0)|(y==0 & l>0): mgcv computes z[ii]-1/l[ii] and
        # discards it — nothing to do.
        z[nina & (y == np.inf) & (la >= 0.0)] = np.inf
        low = (y <= 0.0) if ls_variant else (y == 0.0)
        z[nina & low & (la <= 0.0)] = -np.inf
    return z


def _bcg_bc(y, la, deriv=0):
    """bcg Dd's ``bc(y, λ, deriv)`` (efam.r:1597-1644): the Box-Cox
    value z plus ∂z/∂λ (z1) and ∂²z/∂λ² (z2). The z-branch switches to
    the series at |λ| < 1e-7, the DERIVATIVE branch at |λ| < 1e-4 —
    mgcv mixes exact z with series z1 in the band between. λ scalar."""
    y = np.asarray(y, dtype=float)
    la = float(la)
    z = y.copy()
    nina = ~np.isnan(y)
    z1 = np.full(y.shape, np.nan)
    z2 = np.full(y.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logy = np.where(nina, np.log(y), y)
        ylly = logy * _rpow(y, la)
        if abs(la) < 1e-7:
            yl2 = ylly * logy
            z = np.where(nina,
                         ylly - yl2 * la / 2.0
                         + yl2 * logy * (la * la) / 6.0, z)
        else:
            z = np.where(nina, (_rpow(y, la) - 1.0) / la, z)
        z[nina & (y == np.inf) & (la >= 0.0)] = np.inf
        z[nina & (y == 0.0) & (la <= 0.0)] = -np.inf

        if deriv > 0:
            if abs(la) < 1e-4:
                c0 = ylly * logy
                c1 = c0 * logy
                z1 = np.where(nina, c0 / 2.0 - c1 * la / 6.0, z1)
                if deriv > 1:
                    c2 = c1 * logy
                    z2 = np.where(nina, c1 / 3.0 - c2 * la / 12.0, z2)
            else:
                z1 = np.where(nina, ylly / la - z / la, z1)
                if deriv > 1:
                    z2 = np.where(nina,
                                  ylly * (logy - 2.0 / la) / la
                                  + 2.0 * z / (la * la), z2)
            ii = nina & (((y == np.inf) & (la < 0.0))
                         | ((y == 0.0) & (la > 0.0)))
            z1[ii] = 1.0 / (la * la)
            if deriv > 1:
                z2[ii] = -2.0 * z1[ii] / la
            ii = nina & (y == np.inf) & (la >= 0.0)
            z1[ii] = np.inf
            if deriv > 1:
                z2[ii] = np.inf
            ii = nina & (y == 0.0) & (la <= 0.0)
            z1[ii] = np.inf
            if deriv > 1:
                z2[ii] = -np.inf
    return z, z1, z2


def _bcg_dpnorm(x0, x1, log_p=True):
    """misc.r ``dpnorm`` on the bit-exact nmath pnorm port (bcg's live
    call sites; cnorm's own copy predates and keeps scipy's log_ndtr)."""
    x0 = np.array(x0, dtype=float, copy=True)
    x1 = np.array(x1, dtype=float, copy=True)
    ii = (x1 > 0) & (x0 > 0)
    d = x0[ii].copy()
    x0[ii] = -x1[ii]
    x1[ii] = -d
    p0 = pnorm5_vec(x0, log_p=True)
    p1 = pnorm5_vec(x1, log_p=True)
    dp = p0 + _logexm1(p1 - p0)
    return dp if log_p else np.exp(dp)


def _bcg_ddnorm(x0, x1, a0=0.0, a1=0.0, s0=1.0, s1=1.0):
    """bcg's local ``ddnorm`` (efam.r:1577-1594): log|s1·e^{a1}·φ(x1) −
    s0·e^{a0}·φ(x0)| and its sign. UNLIKE cnorm's ddnorm there is no
    s==0 special-casing — zero-sign rows ride the opposite-sign
    ``logexp1`` branch (s0·s1 = 0 → not ``> 0``), exactly as in mgcv."""
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    shape = np.broadcast(x0, x1, a0, a1, s0, s1).shape
    a0 = np.broadcast_to(np.asarray(a0, dtype=float), shape)
    a1 = np.broadcast_to(np.asarray(a1, dtype=float), shape)
    s0 = np.broadcast_to(np.asarray(s0, dtype=float), shape).astype(float)
    s1 = np.broadcast_to(np.asarray(s1, dtype=float), shape).astype(float)
    with np.errstate(invalid="ignore"):
        p0 = dnorm5_vec(x0, give_log=True) + a0
        p1 = dnorm5_vec(x1, give_log=True) + a1
    dp = p0.copy()
    sgn = np.ones(shape)
    flip = (((s1 < 0) & (s0 > 0))
            | ((s1 > 0) & (s0 > 0) & (p1 < p0))
            | ((s1 < 0) & (s0 < 0) & (p1 > p0)))
    sgn[flip] = -1.0
    swap = p0 > p1
    p0 = p0.copy()
    p1 = p1.copy()
    tmp = p1[swap].copy()
    p1[swap] = p0[swap]
    p0[swap] = tmp
    with np.errstate(invalid="ignore"):
        same = (s0 * s1) > 0
        dp[same] = p0[same] + _logexm1(p1[same] - p0[same])
        dp[~same] = p0[~same] + _cnorm_logexp1(p1[~same] - p0[~same])
    return dp, sgn


def _bcg_cases(y, censor):
    """bcg's censoring index sets (efam.r:1543-1556 / 1658-1662):
    uncensored yat==y; interval finite & yat>0 & yat≠y; LEFT yat ≤ 0
    (bcg censors below at 0, not −∞); right yat==+∞. A y==0==yat row is
    in BOTH iu and il — mgcv's later block overwrites, so keep order."""
    y = np.asarray(y, dtype=float)
    yat = y if censor is None else np.asarray(censor, dtype=float)
    iu = np.where(yat == y)[0]
    ii = np.where(np.isfinite(yat) & (yat > 0) & (yat != y))[0]
    il = np.where(yat <= 0)[0]
    ir = np.where(yat == np.inf)[0]
    return yat, iu, ii, il, ir


def _bcg_dev_resids(y, mu, wt, theta, censor):
    """mgcv bcg ``dev.resids`` (efam.r:1511-1558): −2·(logLik − l_sat)
    per datum on the bc scale; returns ``(d, sign)`` with sign =
    sign(bc(y,λ) − μ) (mgcv's attr)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    th = theta[1] - np.log(wt) / 2.0
    la = float(theta[0])
    yat, iu, ii, il, ir = _bcg_cases(y, censor)
    d = np.zeros(y.shape[0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if iu.size:
            d[iu] = (_rpow_int(_bcg_bc_z(y[iu], la) - mu[iu], 2)
                     * np.exp(-2.0 * th[iu]))
        if ii.size:
            y1 = np.maximum(yat[ii], y[ii])
            y0 = np.minimum(yat[ii], y[ii])
            ethi = np.exp(-th[ii])
            zz = (_bcg_bc_z(y1, la) - _bcg_bc_z(y0, la)) * ethi / 2.0
            d[ii] = (2.0 * _bcg_dpnorm(-zz, zz, log_p=True)
                     - 2.0 * _bcg_dpnorm((_bcg_bc_z(y0, la) - mu[ii]) * ethi,
                                         (_bcg_bc_z(y1, la) - mu[ii]) * ethi,
                                         log_p=True))
        if il.size:
            d[il] = -2.0 * pnorm5_vec(
                (_bcg_bc_z(y[il], la) - mu[il]) * np.exp(-th[il]), log_p=True)
        if ir.size:
            d[ir] = -2.0 * pnorm5_vec(
                -(_bcg_bc_z(y[ir], la) - mu[ir]) * np.exp(-th[ir]),
                log_p=True)
        sgn = np.sign(_bcg_bc_z(y, la) - mu)
    return d, sgn


def _bcg_aic(y, mu, wt, theta, censor):
    """mgcv bcg ``aic`` (efam.r:2031-2075): the full −2·logLik (a clone
    of dev.resids without the saturated reference)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    th = theta[1] - np.log(wt) / 2.0
    la = float(theta[0])
    yat, iu, ii, il, ir = _bcg_cases(y, censor)
    d = np.zeros(y.shape[0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if iu.size:
            d[iu] = -2.0 * dnorm5_vec(
                (_bcg_bc_z(y[iu], la) - mu[iu]) * np.exp(-th[iu]),
                give_log=True)
        if ii.size:
            y1 = np.maximum(yat[ii], y[ii])
            y0 = np.minimum(yat[ii], y[ii])
            ethi = np.exp(-th[ii])
            d[ii] = -2.0 * _bcg_dpnorm(
                (_bcg_bc_z(y0, la) - mu[ii]) * ethi,
                (_bcg_bc_z(y1, la) - mu[ii]) * ethi, log_p=True)
        if il.size:
            d[il] = -2.0 * pnorm5_vec(
                (_bcg_bc_z(y[il], la) - mu[il]) * np.exp(-th[il]), log_p=True)
        if ir.size:
            d[ir] = -2.0 * pnorm5_vec(
                -(_bcg_bc_z(y[ir], la) - mu[ir]) * np.exp(-th[ir]),
                log_p=True)
    return _rsum(d)


def _bcg_ls(y, wt, theta, censor):
    """mgcv bcg ``ls`` (efam.r:2077-2124): the saturated log-lik VALUE
    — uncensored rows carry the Box-Cox Jacobian (λ−1)·log y − th −
    log(2π)/2, interval rows the half-width dpnorm; all θ-derivatives
    are zero. Uses the ls-variant bc (``y ≤ 0`` masks)."""
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    th = theta[1] - np.log(wt) / 2.0
    la = float(theta[0])
    yat, iu, ii, il, ir = _bcg_cases(y, censor)
    ls = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if iu.size:
            ls = ls + _rsum((la - 1.0) * np.log(y[iu]) - th[iu]
                            - _LOG2PI / 2.0)
        if ii.size:
            y1 = np.maximum(yat[ii], y[ii])
            y0 = np.minimum(yat[ii], y[ii])
            zz = (_bcg_bc_z(y1, la, ls_variant=True)
                  - _bcg_bc_z(y0, la, ls_variant=True)) * np.exp(-th[ii]) / 2.0
            ls = ls + _rsum(_bcg_dpnorm(-zz, zz, log_p=True))
    return ls


def _bcg_Dd(y, mu, theta, wt, censor, level=0):
    """mgcv bcg ``Dd`` (efam.r:1559-2029): derivatives of the bcg
    deviance w.r.t. μ and (λ, log σ), by censoring case. Verbatim port;
    θ-matrix layout is mgcv's — D*th (n,2) columns [λ, t], D*th2 (n,3)
    columns [λλ, λt, tt] (= hea's packed upper-triangle for nt=2).
    ``^`` sites ride :func:`_rpow_int`; cancellation control through
    :func:`_bcg_ddnorm` / :func:`_bcg_dpnorm`."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    th = theta[1] - np.log(wt) / 2.0
    th2 = 2.0 * th
    th3 = 3.0 * th
    la = float(theta[0])
    eth = np.exp(-th)
    e2th = eth * eth
    yat, iu, ii, il, ir = _bcg_cases(y, censor)

    with np.errstate(divide="ignore"):
        logy = np.log(y)

    n = mu.shape[0]
    Dmu = np.zeros(n)
    Dmu2 = np.zeros(n)
    if level > 0:
        Dth = np.zeros((n, 2))      # col order l, t
        Dmuth = np.zeros((n, 2))
        Dmu2th = np.zeros((n, 2))
        Dmu3 = np.zeros(n)
    if level > 1:
        Dth2 = np.zeros((n, 3))     # col order ll, lt, tt
        Dmuth2 = np.zeros((n, 3))
        Dmu2th2 = np.zeros((n, 3))
        Dmu4 = np.zeros(n)
        Dmu3th = np.zeros((n, 2))

    _es = dict(divide="ignore", invalid="ignore", over="ignore")

    if iu.size:  # uncensored
        ethi = eth[iu]
        e2thi = e2th[iu]
        with np.errstate(**_es):
            bz, bly, blly = _bcg_bc(y[iu], la, level)
            z = (bz - mu[iu]) * ethi
            Dmui = -2.0 * z * ethi
            Dmu[iu] = Dmui
            Dmu2[iu] = 2.0 * e2thi
            if level > 0:
                Dth[iu, 0] = -2.0 * (logy[iu] - ethi * z * bly)
                Dth[iu, 1] = -2.0 * (_rpow_int(z, 2) - 1.0)
                Dmuth[iu, 0] = -2.0 * e2thi * bly
                Dmuth[iu, 1] = -2.0 * Dmui
                Dmu3[iu] = 0.0
                Dmu2th[iu, 0] = 0.0
                Dmu2th[iu, 1] = -4.0 * e2thi
            if level > 1:
                Dmu4[iu] = 0.0
                Dmu3th[iu, :] = 0.0
                Dth2[iu, 0] = 2.0 * (e2thi * _rpow_int(bly, 2)
                                     + ethi * z * blly)
                Dth2[iu, 1] = -4.0 * ethi * z * bly
                Dth2[iu, 2] = 4.0 * _rpow_int(z, 2)
                Dmuth2[iu, 0] = -2.0 * e2thi * blly
                Dmuth2[iu, 1] = 4.0 * e2thi * bly
                Dmuth2[iu, 2] = 4.0 * Dmui
                Dmu2th2[iu, 0] = 0.0
                Dmu2th2[iu, 1] = 0.0
                Dmu2th2[iu, 2] = 8.0 * e2thi

    if ii.size:  # interval censored
        y0 = np.minimum(y[ii], yat[ii])
        y1 = np.maximum(y[ii], yat[ii])
        ethi = eth[ii]
        e2thi = e2th[ii]
        thi = th[ii]
        th3i = th3[ii]
        th2i = th2[ii]
        with np.errstate(**_es):
            bz0, bly0, blly0 = _bcg_bc(y0, la, level)
            z0 = (bz0 - mu[ii]) * ethi
            bz1, bly1, blly1 = _bcg_bc(y1, la, level)
            z1 = (bz1 - mu[ii]) * ethi

            ldp = _bcg_dpnorm(z0, z1, log_p=True)
            ldd, sdd = _bcg_ddnorm(z0, z1)
            ldzdz, szdz = _bcg_ddnorm(z0, z1, np.log(np.abs(z0)),
                                      np.log(np.abs(z1)),
                                      np.sign(z0), np.sign(z1))
            Dmui = 2.0 * sdd * np.exp(-thi + ldd - ldp)
            Dmu[ii] = Dmui
            Dt = 2.0 * szdz * np.exp(ldzdz - ldp)
            Dmu2i = _rpow_int(Dmui, 2) / 2.0 + e2thi * Dt
            Dmu2[ii] = Dmu2i
            if level > 0:
                ldz2, sz2 = _bcg_ddnorm(z0, z1, np.log(_rpow_int(z0, 2)),
                                        np.log(_rpow_int(z1, 2)))
                ldz3, sz3 = _bcg_ddnorm(z0, z1,
                                        np.log(np.abs(_rpow_int(z0, 3))),
                                        np.log(np.abs(_rpow_int(z1, 3))),
                                        np.sign(z0), np.sign(z1))
                ldb, sb = _bcg_ddnorm(z0, z1, np.log(np.abs(bly0)),
                                      np.log(np.abs(bly1)),
                                      np.sign(bly0), np.sign(bly1))
                ldbz, sbz = _bcg_ddnorm(z0, z1, np.log(np.abs(bly0 * z0)),
                                        np.log(np.abs(bly1 * z1)),
                                        np.sign(bly0 * z0),
                                        np.sign(bly1 * z1))
                ldbz2, sbz2 = _bcg_ddnorm(
                    z0, z1, np.log(np.abs(bly0 * _rpow_int(z0, 2))),
                    np.log(np.abs(bly1 * _rpow_int(z1, 2))),
                    np.sign(bly0), np.sign(bly1))

                z12 = _rpow_int(z1, 2)
                z02 = _rpow_int(z0, 2)
                z13 = z12 * z1
                z03 = z02 * z0
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0
                                 - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                         + 2.0 * sz2 * np.exp(ldz2 - ldp - th3i))
                Dmu3[ii] = Dmu3i
                Dl = -2.0 * sb * np.exp(ldb - ldp - thi)
                Dth[ii, 0] = Dl
                Dth[ii, 1] = Dt
                Dml = (Dl * Dmui / 2.0
                       - 2.0 * sbz * np.exp(ldbz - ldp - th2i))
                Dmuth[ii, 0] = Dml
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       + 2.0 * sz2 * np.exp(ldz2 - ldp - thi))
                Dmuth[ii, 1] = Dmt
                Dtt = (_rpow_int(Dt, 2) / 2.0 - Dt
                       + 2.0 * sz3 * np.exp(ldz3 - ldp))
                Dlt = (Dl * Dt / 2.0 - Dl
                       - 2.0 * sbz2 * np.exp(ldbz2 - ldp - thi))
                Dmu2th[ii, 0] = Dmui * Dml + e2thi * Dlt
                Dmu2th[ii, 1] = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
            if level > 1:
                z14 = z13 * z1
                z04 = z03 * z0
                th4i = 4.0 * thi
                Dmu2t = Dmu2th[ii, 1]
                Dmu2l = Dmu2th[ii, 0]
                a1 = 2.0 * z13 * ethi + Dmui * z12 - 4.0 * z1 * ethi
                a0 = 2.0 * z03 * ethi + Dmui * z02 - 4.0 * z0 * ethi
                lda1, sa1 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dmu4[ii] = (Dmu2i * (3.0 * Dmu2i / 2.0
                                     - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                            + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                            + sa1 * np.exp(lda1 - ldp - th3i))
                a1 = bly1 * z1 * (2.0 - z12)
                a0 = bly0 * z0 * (2.0 - z02)
                lda2, sa2 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dmu3th[ii, 0] = (Dml * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2l - Dmui * Dml) / 2.0
                                 - e2thi * Dml
                                 + Dl * sz2 * np.exp(ldz2 - ldp - th3i)
                                 + 2.0 * sa2 * np.exp(lda2 - ldp - th4i))
                ldz4, sz4 = _bcg_ddnorm(z0, z1, np.log(_rpow_int(z0, 4)),
                                        np.log(_rpow_int(z1, 4)))
                Dmu3th[ii, 1] = (Dmt * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2t - Dmui * Dmt) / 2.0
                                 + e2thi * (2.0 * Dmui - Dmt)
                                 + (Dt - 10.0) * sz2
                                 * np.exp(ldz2 - ldp - th3i)
                                 + 2.0 * sz4 * np.exp(ldz4 - ldp - th3i))
                a1 = _rpow_int(bly1, 2) * z1 * ethi - blly1
                a0 = _rpow_int(bly0, 2) * z0 * ethi - blly0
                lda3, sa3 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dll = (_rpow_int(Dl, 2) / 2.0
                       + 2.0 * sa3 * np.exp(lda3 - ldp - thi))
                Dth2[ii, 0] = Dll
                Dth2[ii, 1] = Dlt
                Dth2[ii, 2] = Dtt
                a1 = _rpow_int(bly1, 2) * (z12 - 1.0) * ethi - blly1 * z1
                a0 = _rpow_int(bly0, 2) * (z02 - 1.0) * ethi - blly0 * z0
                lda4, sa4 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dmll = ((Dl * Dml + Dmui * Dll) / 2.0
                        - Dl * sbz * np.exp(ldbz - ldp - th2i)
                        + 2.0 * sa4 * np.exp(lda4 - ldp - th2i))
                Dmuth2[ii, 0] = Dmll
                ldbz3, sbz3 = _bcg_ddnorm(
                    z0, z1, np.log(np.abs(bly0 * z03)),
                    np.log(np.abs(bly1 * z13)),
                    np.sign(bly0 * z0), np.sign(bly1 * z1))
                Dmlt = ((Dl * Dmt + Dmui * Dlt) / 2.0
                        + (6.0 - Dt) * sbz * np.exp(ldbz - ldp - th2i)
                        - 2.0 * sbz3 * np.exp(ldbz3 - ldp - th2i))
                Dmuth2[ii, 1] = Dmlt
                Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                        + (Dt - 6.0) * sz2 * np.exp(ldz2 - ldp - thi)
                        + 2.0 * sz4 * np.exp(ldz4 - ldp - thi))
                Dmuth2[ii, 2] = Dmtt
                a1 = z1 * (_rpow_int(bly1, 2) * (z12 - 2.0) * ethi
                           - blly1 * z1)
                a0 = z0 * (_rpow_int(bly0, 2) * (z02 - 2.0) * ethi
                           - blly0 * z0)
                lda5, sa5 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dllt = ((Dl * Dlt + Dt * Dll) / 2.0 - Dll
                        - Dl * sbz2 * np.exp(ldbz2 - ldp - thi)
                        + 2.0 * sa5 * np.exp(lda5 - ldp - thi))
                ldbz4, sbz4 = _bcg_ddnorm(
                    z0, z1, np.log(np.abs(bly0 * z04)),
                    np.log(np.abs(bly1 * z14)),
                    np.sign(bly0), np.sign(bly1))
                Dltt = ((Dl * Dtt + Dt * Dlt) / 2.0 - Dlt
                        + (6.0 - Dt) * sbz2 * np.exp(ldbz2 - ldp - thi)
                        - 2.0 * sbz4 * np.exp(ldbz4 - ldp - thi))
                a1 = z13 * (z12 - 3.0)
                a0 = z03 * (z02 - 3.0)
                lda6, sa6 = _bcg_ddnorm(z0, z1, np.log(np.abs(a0)),
                                        np.log(np.abs(a1)),
                                        np.sign(a0), np.sign(a1))
                Dttt = (Dtt * (Dt - 1.0) + Dt * sz3 * np.exp(ldz3 - ldp)
                        + 2.0 * sa6 * np.exp(lda6 - ldp))
                Dmu2th2[ii, 0] = (_rpow_int(Dml, 2) + Dmui * Dmll
                                  + e2thi * Dllt)
                Dmu2th2[ii, 1] = (Dml * Dmt + Dmui * Dmlt
                                  + e2thi * (Dltt - 2.0 * Dlt))
                Dmu2th2[ii, 2] = (_rpow_int(Dmt, 2) + Dmui * Dmtt
                                  + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    if il.size:  # left censoring (y0 = 0, z0 = -Inf, basically)
        y1 = y[il]
        ethi = eth[il]
        e2thi = e2th[il]
        thi = th[il]
        th3i = th3[il]
        th2i = th2[il]
        with np.errstate(**_es):
            bz1, bly1, blly1 = _bcg_bc(y1, la, level)
            z1 = (bz1 - mu[il]) * ethi
            ldp = pnorm5_vec(z1, log_p=True)
            ldn = dnorm5_vec(z1, give_log=True)
            Dmui = 2.0 * np.exp(-thi + ldn - ldp)
            Dmu[il] = Dmui
            Dt = 2.0 * np.sign(z1) * np.exp(ldn + np.log(np.abs(z1)) - ldp)
            Dmu2i = _rpow_int(Dmui, 2) / 2.0 + e2thi * Dt
            Dmu2[il] = Dmu2i
            if level > 0:
                z12 = _rpow_int(z1, 2)
                z13 = z12 * z1
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0
                                 - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                         + 2.0 * np.sign(z12)
                         * np.exp(ldn + np.log(np.abs(z12)) - ldp - th3i))
                Dmu3[il] = Dmu3i
                Dl = (-2.0 * np.sign(bly1)
                      * np.exp(ldn + np.log(np.abs(bly1)) - ldp - thi))
                Dth[il, 0] = Dl
                Dth[il, 1] = Dt
                a1 = bly1 * z1
                Dml = (Dl * Dmui / 2.0
                       - 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i))
                Dmuth[il, 0] = Dml
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       + 2.0 * np.sign(z12)
                       * np.exp(ldn + np.log(np.abs(z12)) - ldp - thi))
                Dmuth[il, 1] = Dmt
                Dtt = (_rpow_int(Dt, 2) / 2.0 - Dt
                       + 2.0 * np.sign(z13)
                       * np.exp(ldn + np.log(np.abs(z13)) - ldp))
                a1 = bly1 * z12
                Dlt = (Dl * Dt / 2.0 - Dl
                       - 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi))
                Dmu2th[il, 0] = Dmui * Dml + e2thi * Dlt
                Dmu2th[il, 1] = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
            if level > 1:
                z14 = z13 * z1
                th4i = 4.0 * thi
                Dmu2t = Dmu2th[il, 1]
                Dmu2l = Dmu2th[il, 0]
                a1 = 2.0 * z13 * ethi + Dmui * z12 - 4.0 * z1 * ethi
                Dmu4[il] = (Dmu2i * (3.0 * Dmu2i / 2.0
                                     - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                            + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                            + np.sign(a1)
                            * np.exp(ldn + np.log(np.abs(a1)) - ldp - th3i))
                a1 = bly1 * z1 * (2.0 - z12)
                Dmu3th[il, 0] = (Dml * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2l - Dmui * Dml) / 2.0
                                 - e2thi * Dml
                                 + Dl * np.sign(z12)
                                 * np.exp(ldn + np.log(np.abs(z12))
                                          - ldp - th3i)
                                 + 2.0 * np.sign(a1)
                                 * np.exp(ldn + np.log(np.abs(a1))
                                          - ldp - th4i))
                Dmu3th[il, 1] = (Dmt * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2t - Dmui * Dmt) / 2.0
                                 + e2thi * (2.0 * Dmui - Dmt)
                                 + (Dt - 10.0)
                                 * np.exp(ldn + np.log(z12) - ldp - th3i)
                                 + 2.0
                                 * np.exp(ldn + np.log(z14) - ldp - th3i))
                a1 = _rpow_int(bly1, 2) * z1 * ethi - blly1
                Dll = (_rpow_int(Dl, 2) / 2.0
                       + 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi))
                Dth2[il, 0] = Dll
                Dth2[il, 1] = Dlt
                Dth2[il, 2] = Dtt
                a2 = _rpow_int(bly1, 2) * (z12 - 1.0) * ethi - blly1 * z1
                a1 = bly1 * z1
                Dmll = ((Dl * Dml + Dmui * Dll) / 2.0
                        - Dl * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i)
                        + 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - th2i))
                Dmuth2[il, 0] = Dmll
                a2 = bly1 * z13
                Dmlt = ((Dl * Dmt + Dmui * Dlt) / 2.0
                        + (6.0 - Dt) * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i)
                        - 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - th2i))
                Dmuth2[il, 1] = Dmlt
                Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                        + (Dt - 6.0)
                        * np.exp(ldn + np.log(z12) - ldp - thi)
                        + 2.0 * np.sign(z14)
                        * np.exp(ldn + np.log(np.abs(z14)) - ldp - thi))
                Dmuth2[il, 2] = Dmtt
                a2 = z1 * (_rpow_int(bly1, 2) * (z12 - 2.0) * ethi
                           - blly1 * z1)
                a1 = bly1 * z12
                Dllt = ((Dl * Dlt + Dt * Dll) / 2.0 - Dll
                        - Dl * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi)
                        + 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - thi))
                a1 = bly1 * z12
                a2 = bly1 * z14
                Dltt = ((Dl * Dtt + Dt * Dlt) / 2.0 - Dlt
                        + (6.0 - Dt) * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi)
                        - 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - thi))
                a1 = z13 * (z12 - 3.0)
                Dttt = (Dtt * (Dt - 1.0)
                        + Dt * np.sign(z13)
                        * np.exp(ldn + np.log(np.abs(z13)) - ldp)
                        + 2.0 * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp))
                Dmu2th2[il, 0] = (_rpow_int(Dml, 2) + Dmui * Dmll
                                  + e2thi * Dllt)
                Dmu2th2[il, 1] = (Dml * Dmt + Dmui * Dmlt
                                  + e2thi * (Dltt - 2.0 * Dlt))
                Dmu2th2[il, 2] = (_rpow_int(Dmt, 2) + Dmui * Dmtt
                                  + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    if ir.size:  # right censoring - basically y1 = Inf
        y0 = y[ir]
        ethi = eth[ir]
        e2thi = e2th[ir]
        thi = th[ir]
        th3i = th3[ir]
        th2i = th2[ir]
        with np.errstate(**_es):
            bz0, bly0, blly0 = _bcg_bc(y0, la, level)
            z0 = (bz0 - mu[ir]) * ethi
            ldp = pnorm5_vec(-z0, log_p=True)
            ldn = dnorm5_vec(z0, give_log=True)
            Dmui = -2.0 * np.exp(-thi + ldn - ldp)
            Dmu[ir] = Dmui
            Dt = -2.0 * np.sign(z0) * np.exp(ldn + np.log(np.abs(z0)) - ldp)
            Dmu2i = _rpow_int(Dmui, 2) / 2.0 + e2thi * Dt
            Dmu2[ir] = Dmu2i
            if level > 0:
                z02 = _rpow_int(z0, 2)
                z03 = z02 * z0
                Dmu3i = (Dmui * (3.0 * Dmu2i / 2.0
                                 - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                         - 2.0 * np.sign(z02)
                         * np.exp(ldn + np.log(np.abs(z02)) - ldp - th3i))
                Dmu3[ir] = Dmu3i
                Dl = (2.0 * np.sign(bly0)
                      * np.exp(ldn + np.log(np.abs(bly0)) - ldp - thi))
                Dth[ir, 0] = Dl
                Dth[ir, 1] = Dt
                a1 = bly0 * z0
                Dml = (Dl * Dmui / 2.0
                       + 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i))
                Dmuth[ir, 0] = Dml
                Dmt = (Dmui * Dt / 2.0 - Dmui
                       - 2.0 * np.sign(z02)
                       * np.exp(ldn + np.log(np.abs(z02)) - ldp - thi))
                Dmuth[ir, 1] = Dmt
                Dtt = (_rpow_int(Dt, 2) / 2.0 - Dt
                       - 2.0 * np.sign(z03)
                       * np.exp(ldn + np.log(np.abs(z03)) - ldp))
                a1 = bly0 * z02
                Dlt = (Dl * Dt / 2.0 - Dl
                       + 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi))
                Dmu2th[ir, 0] = Dmui * Dml + e2thi * Dlt
                Dmu2th[ir, 1] = Dmui * Dmt + e2thi * (Dtt - 2.0 * Dt)
            if level > 1:
                z04 = z03 * z0
                th4i = 4.0 * thi
                Dmu2t = Dmu2th[ir, 1]
                Dmu2l = Dmu2th[ir, 0]
                a1 = 2.0 * z03 * ethi + Dmui * z02 - 4.0 * z0 * ethi
                Dmu4[ir] = (Dmu2i * (3.0 * Dmu2i / 2.0
                                     - _rpow_int(Dmui, 2) / 4.0 - e2thi)
                            + Dmui * (3.0 * Dmu3i - Dmui * Dmu2i) / 2.0
                            - np.sign(a1)
                            * np.exp(ldn + np.log(np.abs(a1)) - ldp - th3i))
                a1 = bly0 * z0 * (2.0 - z02)
                Dmu3th[ir, 0] = (Dml * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2l - Dmui * Dml) / 2.0
                                 - e2thi * Dml
                                 - Dl * np.sign(z02)
                                 * np.exp(ldn + np.log(np.abs(z02))
                                          - ldp - th3i)
                                 - 2.0 * np.sign(a1)
                                 * np.exp(ldn + np.log(np.abs(a1))
                                          - ldp - th4i))
                Dmu3th[ir, 1] = (Dmt * (3.0 * Dmu2i / 2.0
                                        - _rpow_int(Dmui, 2) / 4.0)
                                 + Dmui * (3.0 * Dmu2t - Dmui * Dmt) / 2.0
                                 + e2thi * (2.0 * Dmui - Dmt)
                                 - (Dt - 10.0)
                                 * np.exp(ldn + np.log(z02) - ldp - th3i)
                                 - 2.0
                                 * np.exp(ldn + np.log(z04) - ldp - th3i))
                a1 = _rpow_int(bly0, 2) * z0 * ethi - blly0
                Dll = (_rpow_int(Dl, 2) / 2.0
                       - 2.0 * np.sign(a1)
                       * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi))
                Dth2[ir, 0] = Dll
                Dth2[ir, 1] = Dlt
                Dth2[ir, 2] = Dtt
                a2 = _rpow_int(bly0, 2) * (z02 - 1.0) * ethi - blly0 * z0
                a1 = bly0 * z0
                Dmll = ((Dl * Dml + Dmui * Dll) / 2.0
                        + Dl * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i)
                        - 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - th2i))
                Dmuth2[ir, 0] = Dmll
                a2 = bly0 * z03
                Dmlt = ((Dl * Dmt + Dmui * Dlt) / 2.0
                        - (6.0 - Dt) * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - th2i)
                        + 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - th2i))
                Dmuth2[ir, 1] = Dmlt
                Dmtt = ((Dmt * Dt + Dmui * Dtt) / 2.0 - Dmt
                        - (Dt - 6.0)
                        * np.exp(ldn + np.log(z02) - ldp - thi)
                        - 2.0
                        * np.exp(ldn + np.log(z04) - ldp - thi))
                Dmuth2[ir, 2] = Dmtt
                a2 = z0 * (_rpow_int(bly0, 2) * (z02 - 2.0) * ethi
                           - blly0 * z0)
                a1 = bly0 * z02
                Dllt = ((Dl * Dlt + Dt * Dll) / 2.0 - Dll
                        + Dl * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi)
                        - 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - thi))
                a2 = bly0 * z04
                Dltt = ((Dl * Dtt + Dt * Dlt) / 2.0 - Dlt
                        - (6.0 - Dt) * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp - thi)
                        + 2.0 * np.sign(a2)
                        * np.exp(ldn + np.log(np.abs(a2)) - ldp - thi))
                a1 = z03 * (z02 - 3.0)
                Dttt = (Dtt * (Dt - 1.0)
                        - Dt * np.sign(z03)
                        * np.exp(ldn + np.log(np.abs(z03)) - ldp)
                        - 2.0 * np.sign(a1)
                        * np.exp(ldn + np.log(np.abs(a1)) - ldp))
                Dmu2th2[ir, 0] = (_rpow_int(Dml, 2) + Dmui * Dmll
                                  + e2thi * Dllt)
                Dmu2th2[ir, 1] = (Dml * Dmt + Dmui * Dmlt
                                  + e2thi * (Dltt - 2.0 * Dlt))
                Dmu2th2[ir, 2] = (_rpow_int(Dmt, 2) + Dmui * Dmtt
                                  + e2thi * (Dttt - 4.0 * Dtt + 4.0 * Dt))

    r = {"Dmu": Dmu, "Dmu2": Dmu2, "EDmu2": Dmu2}
    if level > 0:
        r["Dth"] = Dth
        r["Dmuth"] = Dmuth
        r["Dmu3"] = Dmu3
        r["Dmu2th"] = Dmu2th
        r["EDmu2th"] = Dmu2th
    if level > 1:
        r["Dmu4"] = Dmu4
        r["Dth2"] = Dth2
        r["Dmuth2"] = Dmuth2
        r["Dmu2th2"] = Dmu2th2
        r["Dmu3th"] = Dmu3th
    return r


class bcg(Family):
    """Censored Box-Cox Gaussian extended family — port of mgcv ``bcg()``
    (efam.r:1477-2170).

    ``bc(y, λ) ~ N(μ, σ²/wt)`` with both θ = (λ, log σ) estimated
    jointly by default. The response is non-negative with the
    2-column ``cbind(y, yat)`` censor encoding, but the LEFT-censor
    marker is ``yat ≤ 0`` (censored below at zero) and interval bounds
    must be positive; ``yat == +∞`` is right-censoring as usual.

    θ intake replicates mgcv exactly, quirks included (live-R
    verified): ``theta=c(λ,σ)`` with σ > 0 stores the LENGTH-3
    ``c(λ, log λ, log σ)`` with ``n_theta = 0`` — the working log σ is
    ``log λ``, so the requested σ is ignored; σ < 0 keeps the raw
    ``(λ, σ)`` as the starting value (σ becomes the starting LOG scale);
    σ = 0 or ``theta=None`` starts at (1, 0). ``dev_resids`` is the
    proper deviance on the bc scale; deviance-residual signs come from
    ``sign(bc(y,λ) − μ)`` via :meth:`residuals_extended` (mgcv's
    ``attr(d,"sign")``). mgcv exports no variance/rd/qf slots.
    okLinks: identity (default), log, sqrt.
    """
    name = "bcg"
    canonical_link_name = "identity"
    _newton_canonical = "none"
    scale_known = True
    is_extended = True
    n_theta = 2
    _OK_LINKS = ("log", "identity", "sqrt")

    def __init__(self, theta=None, link: str = "identity"):
        if link not in self._OK_LINKS:
            raise ValueError(
                f'link "{link}" not available for bcg family; available '
                f'links are {self._OK_LINKS}')
        if theta is not None:
            t = np.asarray(theta, dtype=float).reshape(-1)
            if t.shape[0] != 2:
                raise ValueError("theta should be length 2")
            if t[1] <= 0:
                # initial values: mgcv keeps the RAW pair (its
                # log(-theta[2]) line runs after iniTheta was taken).
                ini = t.copy()
            else:
                # fixed: mgcv's c(theta[1], log(theta)) — length 3.
                with np.errstate(invalid="ignore", divide="ignore"):
                    ini = np.concatenate([t[:1], np.log(t)])
                self.n_theta = 0
        else:
            ini = np.array([1.0, 0.0])
        self._theta = np.asarray(ini, dtype=float)
        # bam's bgam.fit θ-update gate (bam.r:1204-1206): estimate.theta
        # runs between PIRLS iters whenever the extended family has free
        # θ (``family$n.theta>0``; the ``scale<0`` leg never fires here —
        # the censored families carry no ``scale`` slot, so bgam.fit's
        # scale resolves to 1, bam.r:924).
        self.estimate_theta_callback = self.n_theta > 0
        self._censor = None
        self._censorfull = None
        super().__init__(link=link)

    # ----- θ accessors / censoring bound ---------------------------------

    def set_theta(self, values) -> None:
        # mgcv putTheta stores whatever it is handed (no validation).
        self._theta = np.asarray(values, dtype=float).reshape(-1).copy()

    def get_theta(self, trans: bool = False) -> np.ndarray:
        th = self._theta.copy()
        if trans:
            th[1] = np.exp(th[1])
        return th

    def set_censor(self, censor) -> None:
        """Stash the censoring bound (column 1 of ``cbind(y, yat)``)."""
        self._censor = (None if censor is None
                        else np.asarray(censor, dtype=float))
        self._censorfull = None

    def set_ind(self, ind) -> None:
        """mgcv ``subsety`` (efam.r:2154): window the censor bound to the
        bam chunk rows; ``ind=None`` restores (see cnorm.set_ind)."""
        if self._censorfull is None:
            if ind is None or self._censor is None:
                return
            self._censorfull = self._censor
        self._censor = (self._censorfull if ind is None
                        else self._censorfull[np.asarray(ind, dtype=int)])

    # ----- deviance / Dd / aic -------------------------------------------

    def dev_resids(self, y, mu, wt, theta=None):
        th = (self._theta if theta is None
              else np.asarray(theta, dtype=float).reshape(-1))
        d, sgn = _bcg_dev_resids(y, mu, wt, th, self._censor)
        self._dev_sign = sgn
        return d

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        return _bcg_Dd(y, mu, theta, wt, self._censor, level=level)

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        th = (self._theta if theta is None
              else np.asarray(theta, dtype=float).reshape(-1))
        return _bcg_aic(y, mu, wt, th, self._censor)

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        th = (self._theta if theta is None
              else np.asarray(theta, dtype=float).reshape(-1))
        ls = _bcg_ls(y, wt, th, self._censor)
        n = np.asarray(y).shape[0]
        # mgcv: lsth1 = c(0,0), LSTH1 = matrix(0,n,2), lsth2 = 2×2 zeros.
        return {"ls": ls, "lsth1": np.zeros(2),
                "lsth2": np.zeros((2, 2)), "LSTH1": np.zeros((n, 2))}

    def ls(self, y, wt, scale):
        return np.array([_bcg_ls(y, wt, self._theta, self._censor),
                         0.0, 0.0])

    # ----- residuals ------------------------------------------------------

    def residuals_extended(self, y, mu, wt, type: str = "deviance"):
        """Deviance residuals with mgcv's ``attr(d,"sign")`` — the sign
        lives on the bc scale (``sign(bc(y,λ)−μ)``); the raw
        ``sign(y−μ)`` default would compare y to a bc-scale μ."""
        if type != "deviance":
            raise NotImplementedError(
                f"bcg residuals: type {type!r} not implemented")
        d, sgn = _bcg_dev_resids(np.asarray(y, dtype=float),
                                 np.asarray(mu, dtype=float),
                                 np.asarray(wt, dtype=float),
                                 self._theta, self._censor)
        return np.sqrt(np.maximum(d, 0.0)) * sgn

    # ----- initialization / validity -------------------------------------

    def initialize(self, y, wt):
        # mgcv bcg initialize (efam.r:2127-2134): negative responses
        # stop; mustart = y (identity) or pmax(y, min(y>0)).
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("response must be non-negative")
        if self.link.name == "identity":
            return y.copy()
        return np.maximum(y, float(np.min(y > 0)))

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu, dtype=float)
        if self.link.name == "identity":
            return bool(np.all(np.isfinite(mu)))
        return bool(np.all(mu > 0))

    # ----- postproc ------------------------------------------------------

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        # mgcv bcg postproc (efam.r:2136-2147): null deviance via
        # find.null.dev; relabel "bcg(λ,σ)" (all getTheta(TRUE) elements
        # — three for the fixed-θ quirk case).
        null_dev = find_null_dev(self, y, eta=linear_predictors,
                                 offset=offset, weights=prior_weights)
        lab = ",".join(f"{v:g}" for v in np.round(self.get_theta(True), 3))
        return {"null_deviance": null_dev, "family_name": f"bcg({lab})"}

    def __repr__(self):
        return f"bcg(theta={self._theta}, link={self.link.name})"


# ---------------------------------------------------------------------------
# gfam (grouped families) — mgcv ``gfam()`` (gfam.r:3-604): one response
# vector drawn from several distributions, supplied as a two-column
# ``cbind(y, index)`` where column 2 indexes the family list (1-based).
# Always an ``extended.family`` with the overall scale fixed at 1;
# component scale parameters (gaussian σ², Gamma φ, tw φ, …) join the
# θ vector as log-scales and are estimated by REML alongside component
# family parameters. Regular exponential members are adapted on the fly
# (fix.family.var derivatives + the raw fix.family.ls table below).
# ---------------------------------------------------------------------------


def _gfam_exp_ls(fam, y, w, scale):
    """Raw ``fix.family.ls`` saturated log-likelihood for an exponential
    family member (gam.fit3.r:2497-2548): ``c(ls, dls/dφ, d²ls/dφ²)`` —
    derivatives w.r.t. the SCALE itself, not log scale (gfam's ls does
    its own chain rule on these, gfam.r:347-350). ``Family.ls`` can't be
    reused here: it returns log-scale derivatives, and re-dividing would
    change mgcv's rounding order.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    name = fam.name
    if name == "gaussian":
        good = w > 0
        nobs = float(np.sum(good))
        with np.errstate(divide="ignore"):
            lw = np.log(w[good])
        return np.array([
            -nobs * math.log(2.0 * math.pi * scale) / 2.0
            + _rsum(lw) / 2.0,
            -nobs / (2.0 * scale),
            nobs / (2.0 * scale * scale)])
    if name == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _nmath._disp("dpois", _nmath.dpois, [y, y], (True,))
        return np.array([_rsum(logp * w), 0.0, 0.0])
    if name == "binomial":
        # -binomial()$aic(y, n=1, mu=y, wt=w, dev=0)/2 with stats::
        # binomial's aic: m <- wt (n all 1);
        # -2·Σ ifelse(m>0, wt/m, 0)·dbinom(round(m·y), round(m), y, TRUE)
        m = w
        good = m > 0
        weight = np.where(good, w / np.where(good, m, 1.0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _dbinom_raw_disp(np.rint(m * y), np.rint(m),
                                    y, 1.0 - y, True)
        return np.array([-(-2.0 * _rsum(weight * logp)) / 2.0, 0.0, 0.0])
    if name == "Gamma":
        good = w > 0
        y = y[good]
        w = w[good]
        sw = scale / w
        isw = 1.0 / sw
        lsw = np.log(sw)
        # R-level lgamma/digamma/trigamma are nmath's (NOT scipy's) —
        # lgammafn and psigamma(·, 0/1).
        k1 = -_lgammafn_arr(isw) - lsw / sw - isw
        ls0 = _rsum(k1 - np.log(y))
        k2 = (_nmath.psigamma_vec(isw, 0) + lsw) / (sw * sw)
        d1 = _rsum(k2 / w)
        k3 = ((-_nmath.psigamma_vec(isw, 1) / sw
               + (1.0 - 2.0 * lsw - 2.0 * _nmath.psigamma_vec(isw, 0)))
              / _rpow_int(sw, 3))
        d2 = _rsum(k3 / _rpow_int(w, 2))
        return np.array([ls0, d1, d2])
    if name in ("quasi", "quasipoisson", "quasibinomial"):
        # extended quasi-likelihood form
        good = w > 0
        nobs = float(np.sum(good))
        with np.errstate(divide="ignore"):
            lw = np.log(w[good])
        return np.array([
            -nobs * math.log(scale) / 2.0 + _rsum(lw) / 2.0,
            -nobs / (2.0 * scale),
            nobs / (2.0 * scale * scale)])
    if name == "inverse.gaussian":
        good = w > 0
        nobs = float(np.sum(good))
        with np.errstate(divide="ignore"):
            lw = np.log(w[good])
        return np.array([
            -_rsum(np.log(2.0 * math.pi * scale * _rpow_int(y[good], 3)))
            / 2.0 + _rsum(lw) / 2.0,
            -nobs / (2.0 * scale),
            nobs / (2.0 * scale * scale)])
    raise ValueError("family not recognised")


def _gfam_is_ext(f: "Family") -> bool:
    """mgcv's ``inherits(fam, "extended.family")`` for gfam member
    classification. hea's ``Family.is_extended`` flag serves bam's
    Newton-branch gating and is False on ``tw`` (which the engine
    special-cases by type — gam.py ``_family_mgcv_extended``); mgcv's
    tw IS an extended family, so test both."""
    return isinstance(f, tw) or f.is_extended


def _gfam_kj(j, n):
    """gfam.r:115 ``kj(j,n) = (2n-j+1)j/2`` — packed row-major
    upper-triangle offsets: elements in the first ``j`` rows of an
    ``n×n`` symmetric upper triangle. Vector-safe on ``j``."""
    return (2 * n - j + 1) * j // 2


def _gfam_filth(n, j, nth):
    """gfam.r:116-122 ``filth``: 0-based positions, in the total packed
    θ² vector (n total θs, row-major upper triangle), of the θ²-block of
    a family whose ``nth`` θs start at 1-based position ``j``. Order
    matches the family's own packed Dth2 columns."""
    # a = A[A <= t(A)[, nth:1]] column-major = 1..nth, 1..nth-1, ..., 1
    a = np.concatenate([np.arange(1, nth - c + 1) for c in range(nth)])
    reps = np.repeat(_gfam_kj(np.arange(nth), n - j + 1),
                     np.arange(nth, 0, -1))
    return (reps + a + _gfam_kj(j - 1, n)) - 1


def _gfam_filsc(n, j, nth):
    """gfam.r:123-129 ``filsc``: 0-based positions of the (θ_k, ρ) and
    (ρ, ρ) pairs for a family with ``nth`` θs plus a trailing log-scale
    ρ, the block starting at 1-based position ``j``."""
    k = np.arange(nth + 1)
    return (_gfam_kj(k, n - j + 1) + np.arange(nth + 1, 0, -1)
            + _gfam_kj(j - 1, n)) - 1


class _GfamLink(Link):
    """Per-observation dispatching link for :class:`gfam` — the port of
    gfam's linkfun/linkinv/mu.eta/g2g/g3g/g4g/valideta slots
    (gfam.r:235-314), each looping the family list over ``fi == i``
    subsets. ``name`` is the brace-joined link string, never
    "identity", so ``Family.dDeta`` always takes its general branch —
    exactly as mgcv's dDeta does for gfam (``family$link != "identity"``
    even when every member link is identity)."""

    def __init__(self, fam: "gfam"):
        self._fam = fam
        self.name = "{" + ",".join(
            f.link.name for f in fam._fl) + "}"

    def _dispatch(self, x, method):
        x = np.asarray(x, dtype=float)
        out = x.copy()
        fi = self._fam._fi_checked(x.shape[0])
        for i, f in enumerate(self._fam._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size:
                out[ii] = getattr(f.link, method)(x[ii])
        return out

    def link(self, mu): return self._dispatch(mu, "link")
    def linkinv(self, eta): return self._dispatch(eta, "linkinv")
    def mu_eta(self, eta): return self._dispatch(eta, "mu_eta")
    def g2g(self, mu): return self._dispatch(mu, "g2g")
    def g3g(self, mu): return self._dispatch(mu, "g3g")
    def g4g(self, mu): return self._dispatch(mu, "g4g")

    def valideta(self, eta) -> bool:
        eta = np.asarray(eta, dtype=float)
        fi = self._fam._fi_checked(eta.shape[0])
        for i, f in enumerate(self._fam._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size and not f.link.valideta(eta[ii]):
                return False
        return True


# Member intake for R-style name strings (gfam.r:23 eval(parse(text=)));
# values are the hea constructors.
_GFAM_MEMBER_NAMES: dict = {}


def _gfam_member(spec) -> Family:
    """Normalize one ``fl`` entry (gfam.r:23-25): a name string, a
    constructor/callable, or a Family instance."""
    if isinstance(spec, str):
        ctor = _GFAM_MEMBER_NAMES.get(spec)
        if ctor is None:
            raise ValueError("family not recognized")
        spec = ctor
    if isinstance(spec, Family):
        return spec
    if callable(spec):
        out = spec()
        if isinstance(out, Family):
            return out
    raise ValueError("family not recognized")


class gfam(Family):
    """Grouped families — mechanical port of mgcv ``gfam(fl)``
    (gfam.r:3-604).

    The response is ``cbind(y, index)``: column 1 the observation,
    column 2 the 1-based index into ``fl`` of the family it follows.
    Members may be exponential families (adapted on the fly; scale
    fixed at 1 for poisson/binomial, otherwise a free log-scale θ) or
    extended families (tw additionally gets a free log-scale θ). The
    grouped family is itself extended with overall scale 1; general
    (multi-LP) members are not supported, as in mgcv.
    """
    canonical_link_name = "identity"    # never used; link is _GfamLink
    _newton_canonical = "none"          # gfam.r:603 canonical="none"
    scale_known = True                  # overall scale fixed at 1
    is_extended = True

    def __init__(self, fl):
        fl = [_gfam_member(f) for f in fl]
        if not fl:
            raise ValueError("family not recognized")
        n_theta = 0
        theta_parts = []
        need_rsd = False
        names = []
        for f in fl:
            if isinstance(f, GeneralFamily):
                # gfam.r:55 (fam_class check), message verbatim.
                raise NotImplementedError(
                    "general familes not implemented so far")
            if not _gfam_is_ext(f):
                # gfam.r:29-30: fix.family.ls(fix.family.var(fam)) — the
                # derivative slots are already on every hea family; the
                # fix.family.ls table membership check fires here (its
                # "family not recognised" stop is what rejects e.g. a
                # fixed-p Tweedie member, exactly as in mgcv).
                _gfam_exp_ls(f, np.ones(1), np.ones(1), 1.0)
            names.append(f.name)
            scale = self._member_scale(f)
            if _gfam_is_ext(f):
                if scale < 0:
                    n_theta += f.n_theta + 1
                    theta_parts.append(np.concatenate(
                        [np.asarray(f.get_theta(), dtype=float).reshape(-1),
                         [0.0]]))
                else:
                    n_theta += f.n_theta
                    theta_parts.append(
                        np.asarray(f.get_theta(), dtype=float).reshape(-1))
            else:
                if scale < 0:
                    n_theta += 1
                    theta_parts.append(np.zeros(1))
            if (getattr(f, "residuals", None) is not None
                    or getattr(f, "residuals_extended", None) is not None):
                need_rsd = True
        self._fl = fl
        self.name = "gfam{" + ",".join(names) + "}"
        self.n_theta = int(n_theta)
        # bam bgam.fit θ-update gate (bam.r:1204-1206): gfam's composite
        # θ (member θs + exponential-member log scales) is free whenever
        # any slot exists.
        self.estimate_theta_callback = self.n_theta > 0
        self._theta = (np.concatenate(theta_parts) if theta_parts
                       else np.zeros(0))
        if self._theta.shape[0] != self.n_theta:
            # A fixed-θ extended member (nb(theta=2), clog(theta=σ), …)
            # contributes getTheta() entries to the initial .Theta but 0
            # to n.theta (gfam.r:36-49), leaving mgcv's .Theta walk
            # misaligned with every dev.resids/Dd/ls consumer. Refuse
            # rather than replicate an inconsistent state.
            raise NotImplementedError(
                "gfam with a fixed-theta extended member: mgcv's initial "
                ".Theta length differs from n.theta (gfam.r:36-50) and "
                "the downstream walks misread it; not supported")
        self._fi: np.ndarray | None = None
        self._fifull: np.ndarray | None = None
        self.link = _GfamLink(self)
        # gfam.r:460: the residuals slot exists only when a member has
        # one (need.rsd); None keeps the engine's standard residuals.
        self.residuals = self._residuals_gfam if need_rsd else None

    @staticmethod
    def _member_scale(f: Family) -> float:
        """The mgcv ``fl[[i]]$scale`` after gfam's normalization
        (gfam.r:28-31): exponential members 1 for poisson/binomial else
        -1; extended members their own slot (tw: -1, efam.r:3263) with
        NULL → 1."""
        if _gfam_is_ext(f):
            return -1.0 if isinstance(f, tw) else 1.0
        return 1.0 if f.name in ("poisson", "binomial") else -1.0

    # ----- family-index plumbing -----------------------------------------

    def set_fi(self, fi) -> None:
        """Stash the family-index column (the ``attr(fl,"fi")`` write in
        gfam's preinitialize, gfam.r:391). Called by the gam intake once
        the two-column response is split; must precede any fitting."""
        self._fi = np.asarray(fi, dtype=float)
        self._fifull = None

    def set_ind(self, ind) -> None:
        """gfam.r:78-86 ``setInd``: subset ``fi`` by ``ind`` (prediction
        blocks, bam chunks), restore with ``ind=None``."""
        if self._fifull is None:
            if ind is None:
                return
            self._fifull = self._fi
        self._fi = (self._fifull if ind is None
                    else self._fifull[np.asarray(ind, dtype=int)])

    def get_fl(self) -> list:
        """gfam.r:60 ``getfl``."""
        return self._fl

    def _fi_checked(self, n: int) -> np.ndarray:
        if self._fi is None:
            raise ValueError(
                "gfam requires a two-column response cbind(y, index); "
                "no family index has been set")
        if self._fi.shape[0] != n:
            raise ValueError("no family index")
        return self._fi

    def _blocks(self, n: int):
        """Iterate ``(member, ii, nth, scale, i0)`` over the family list
        with mgcv's θ walk: ``i0`` is the 0-based position of the
        member's block in θ; the walk advances by ``nth + (scale<0)``."""
        fi = self._fi_checked(n)
        i0 = 0
        for i, f in enumerate(self._fl):
            ii = np.where(fi == i + 1)[0]
            nth = f.n_theta if _gfam_is_ext(f) else 0
            scale = self._member_scale(f)
            yield f, ii, nth, scale, i0
            i0 += nth + (1 if scale < 0 else 0)

    # ----- θ accessors (gfam.r:63-75) -------------------------------------

    def get_theta(self, trans: bool = False) -> np.ndarray:
        return self._theta.copy()

    def set_theta(self, values) -> None:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.shape[0] != self.n_theta:
            raise ValueError(
                f"gfam expects {self.n_theta} params; got {v.shape[0]}")
        self._theta = v.copy()
        # putTheta's component propagation (gfam.r:66-74). QUIRK kept:
        # the R loop is `for (i in ...) if (extended) { ... i0 <- i0+nth }`
        # so an exponential member's log-scale slot does NOT advance i0 —
        # a later extended member's stored θ is then set from the wrong
        # positions. Harmless for fitting (dev.resids/Dd/ls slice θ with
        # their own correct walks) but visible wherever a member reads
        # its OWN stored θ (tw postproc's "Tweedie(p=…)" label).
        i0 = 0
        for f in self._fl:
            if not _gfam_is_ext(f):
                continue
            scale = self._member_scale(f)
            nth = f.n_theta + (1 if scale < 0 else 0)
            if f.n_theta > 0:
                f.set_theta(v[i0:i0 + f.n_theta])
            i0 += nth

    # ----- deviance and its derivatives ------------------------------------

    def dev_resids(self, y, mu, wt, theta=None) -> np.ndarray:
        """gfam.r:88-109: member deviances, each divided by its
        ``exp(θ_scale)`` when the member scale is free."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        theta = (self._theta if theta is None
                 else np.asarray(theta, dtype=float).reshape(-1))
        r = mu.copy()
        for f, ii, nth, scale, i0 in self._blocks(y.shape[0]):
            if _gfam_is_ext(f):
                th = theta[i0:i0 + nth] if nth else None
                r[ii] = f.dev_resids(y[ii], mu[ii], wt[ii], th)
            else:
                r[ii] = f.dev_resids(y[ii], mu[ii], wt[ii])
            if scale < 0:
                r[ii] = r[ii] / math.exp(theta[i0 + nth])
        return r

    def Dd(self, y, mu, theta, wt, level: int = 0) -> dict:
        """gfam.r:111-232: scaled-deviance derivatives; component scale
        parameters are θ entries (log scale), the overall scale is 1."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        n_theta = theta.shape[0]
        n = mu.shape[0]
        r: dict = {"Dmu": y.copy(), "Dmu2": y.copy(), "EDmu2": y.copy()}
        if level > 0:
            r["EDmu2th"] = np.zeros((n, n_theta))
            r["Dmu2th"] = np.zeros((n, n_theta))
            r["Dth"] = np.zeros((n, n_theta))
            r["Dmuth"] = np.zeros((n, n_theta))
            r["Dmu3"] = y.copy()
        if level > 1:
            npair = n_theta * (n_theta + 1) // 2
            r["Dth2"] = np.zeros((n, npair))
            r["Dmuth2"] = np.zeros((n, npair))
            r["Dmu2th2"] = np.zeros((n, npair))
            r["Dmu3th"] = np.zeros((n, n_theta))
            r["Dmu4"] = y.copy()
        for f, ii, nth, fscale, i0 in self._blocks(n):
            ith = np.arange(i0, i0 + nth)
            th = theta[ith] if nth else None
            if fscale < 0:
                rho = theta[i0 + nth]
                isc = i0 + nth
                its = np.concatenate([ith, [isc]]) if nth else np.array([isc])
            else:
                rho = 0.0
            scale = math.exp(rho)
            if _gfam_is_ext(f):
                ri = f.Dd(y[ii], mu[ii], th, wt[ii], level=level)
                r["Dmu"][ii] = ri["Dmu"] / scale
                r["Dmu2"][ii] = ri["Dmu2"] / scale
                r["EDmu2"][ii] = ri["EDmu2"] / scale
                if level > 0:
                    if nth:
                        r["Dth"][np.ix_(ii, ith)] = ri["Dth"].reshape(
                            ii.size, nth) / scale
                        r["Dmuth"][np.ix_(ii, ith)] = ri["Dmuth"].reshape(
                            ii.size, nth) / scale
                        r["Dmu2th"][np.ix_(ii, ith)] = ri["Dmu2th"].reshape(
                            ii.size, nth) / scale
                        r["EDmu2th"][np.ix_(ii, ith)] = ri["EDmu2th"].reshape(
                            ii.size, nth) / scale
                    r["Dmu3"][ii] = ri["Dmu3"] / scale
                    if fscale < 0:
                        D = f.dev_resids(y[ii], mu[ii], wt[ii], th)
                        r["Dth"][ii, isc] = -D / scale
                        r["Dmuth"][ii, isc] = -ri["Dmu"] / scale
                        r["Dmu2th"][ii, isc] = -ri["Dmu2"] / scale
                        r["EDmu2th"][ii, isc] = -ri["EDmu2"] / scale
                if level > 1:
                    r["Dmu4"][ii] = ri["Dmu4"] / scale
                    if nth > 0:
                        ijth = _gfam_filth(n_theta, i0 + 1, nth)
                        r["Dmu3th"][np.ix_(ii, ith)] = ri["Dmu3th"].reshape(
                            ii.size, nth) / scale
                        npr = nth * (nth + 1) // 2
                        r["Dth2"][np.ix_(ii, ijth)] = ri["Dth2"].reshape(
                            ii.size, npr) / scale
                        r["Dmuth2"][np.ix_(ii, ijth)] = ri["Dmuth2"].reshape(
                            ii.size, npr) / scale
                        r["Dmu2th2"][np.ix_(ii, ijth)] = ri["Dmu2th2"].reshape(
                            ii.size, npr) / scale
                    if fscale < 0:
                        ijsc = _gfam_filsc(n_theta, i0 + 1, nth)
                        r["Dmu3th"][ii, isc] = -ri["Dmu3"] / scale
                        r["Dth2"][np.ix_(ii, ijsc)] = -r["Dth"][
                            np.ix_(ii, its)]
                        r["Dmuth2"][np.ix_(ii, ijsc)] = -r["Dmuth"][
                            np.ix_(ii, its)]
                        r["Dmu2th2"][np.ix_(ii, ijsc)] = -r["Dmu2th"][
                            np.ix_(ii, its)]
            else:  # exponential families (gfam.r:198-228)
                vi = f.variance(mu[ii])
                dv = f.dvar(mu[ii])
                ri = y[ii] - mu[ii]
                r["Dmu"][ii] = -2.0 * ri / (vi * scale)
                r["Dmu2"][ii] = 2.0 * (1.0 + ri * dv / vi) / (vi * scale)
                r["EDmu2"][ii] = 2.0 / (vi * scale)
                if level > 0:
                    d2v = f.d2var(mu[ii])
                    r["Dmu3"][ii] = (-r["Dmu2"][ii] * dv / vi
                                     + 2.0 * (ri * (d2v / vi
                                                    - _rpow_int(dv / vi, 2))
                                              - dv / vi) / (vi * scale))
                    if fscale < 0:
                        D = f.dev_resids(y[ii], mu[ii], wt[ii])
                        r["Dth"][ii, isc] = -D / scale
                        r["Dmuth"][ii, isc] = -r["Dmu"][ii]
                        r["Dmu2th"][ii, isc] = -r["Dmu2"][ii]
                        r["EDmu2th"][ii, isc] = -r["EDmu2"][ii]
                if level > 1:
                    d3v = f.d3var(mu[ii])
                    r["Dmu4"][ii] = (
                        -r["Dmu2"][ii] * d2v / vi
                        - 2.0 * r["Dmu3"][ii] * dv / vi
                        + 2.0 * (2.0 * (_rpow_int(dv / vi, 2) - d2v / vi)
                                 + ri * (d3v / vi
                                         - 3.0 * dv * d2v / _rpow_int(vi, 2)
                                         + 2.0 * _rpow_int(dv / vi, 3)))
                        / (vi * scale))
                    if fscale < 0:
                        ijsc = _gfam_filsc(n_theta, i0 + 1, nth)
                        r["Dmu3th"][ii, isc] = -r["Dmu3"][ii]
                        r["Dth2"][np.ix_(ii, ijsc)] = -r["Dth"][
                            ii, isc].reshape(-1, 1)
                        r["Dmuth2"][np.ix_(ii, ijsc)] = -r["Dmuth"][
                            ii, isc].reshape(-1, 1)
                        r["Dmu2th2"][np.ix_(ii, ijsc)] = -r["Dmu2th"][
                            ii, isc].reshape(-1, 1)
        return r

    # ----- saturated log-likelihood (gfam.r:317-356) -----------------------

    def ls_extended(self, y, wt, theta=None, scale: float = 1.0) -> dict:
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        theta = (self._theta if theta is None
                 else np.asarray(theta, dtype=float).reshape(-1))
        n_theta = self.n_theta
        ls0 = 0.0
        lsth1 = np.zeros(n_theta)
        LSTH1 = np.zeros((y.shape[0], n_theta))
        lsth2 = np.zeros((n_theta, n_theta))
        for f, ii, nth, fscale, i0 in self._blocks(y.shape[0]):
            if _gfam_is_ext(f):
                th = theta[i0:i0 + nth] if nth > 0 else 0.0
                sca = math.exp(theta[i0 + nth]) if fscale < 0 else 1.0
                li = f.ls_extended(y[ii], wt[ii], theta=th, scale=sca)
                ls0 += float(li["ls"])
                nth1 = nth + 1 if fscale < 0 else nth
                ith = np.arange(i0, i0 + nth1)
                if nth1 > 0:
                    lsth1[ith] = np.asarray(
                        li["lsth1"], dtype=float).reshape(-1)
                    LSTH1[np.ix_(ii, ith)] = np.asarray(
                        li["LSTH1"], dtype=float).reshape(ii.size, nth1)
                    lsth2[np.ix_(ith, ith)] = np.asarray(
                        li["lsth2"], dtype=float).reshape(nth1, nth1)
            else:
                if fscale < 0:
                    sca = math.exp(theta[i0])
                else:
                    sca = 1.0
                li = _gfam_exp_ls(f, y[ii], wt[ii], sca)
                ls0 += float(li[0])
                if fscale < 0:
                    # derivs w.r.t. log scale from the raw d/dφ form
                    # (gfam.r:347-350)
                    lsth1[i0] = li[1] * sca
                    lsth2[i0, i0] = li[2] * (sca * sca) + li[1] * sca
                    w01 = (wt[ii] > 0).astype(float)
                    LSTH1[ii, i0] = (sca * w01) * li[1] / float(
                        np.sum(wt[ii] > 0))
        return {"ls": ls0, "lsth1": lsth1, "LSTH1": LSTH1, "lsth2": lsth2}

    def ls(self, y, wt, scale):
        le = self.ls_extended(y, wt)
        return np.array([le["ls"], 0.0, 0.0])

    # ----- aic (gfam.r:358-378) --------------------------------------------

    def aic(self, y, mu, dev, wt, n, theta=None) -> float:
        """The ``dev`` argument is ignored and recomputed per member
        (mgcv's comment verbatim: "note dev has to be ignored and
        re-computed component wise here")."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        theta = (self._theta if theta is None
                 else np.asarray(theta, dtype=float).reshape(-1))
        n1 = np.ones(y.shape[0])
        aic = 0.0
        for f, ii, nth, fscale, i0 in self._blocks(y.shape[0]):
            if _gfam_is_ext(f):
                # gfam.r:367 `ith <- 1:nth-1+i0`: for nth==0 this is the
                # R vector c(i0, i0-1) with 1-based 0s dropped — a stale
                # 0-2 element slice passed as θ. Members with n_theta==0
                # here are parameter-free (cpois) and ignore it, but the
                # indexing is mgcv's, kept as is.
                if nth > 0:
                    th = theta[i0:i0 + nth]
                else:
                    idx = [k for k in (i0 + 1, i0) if k >= 1]
                    th = theta[np.asarray(idx, dtype=int) - 1]
                dev_i = _rsum(f.dev_resids(y[ii], mu[ii], wt[ii], th))
                aic += float(f.aic(y[ii], mu[ii], dev_i, wt[ii],
                                   n1[ii], theta=th))
            else:
                dev_i = _rsum(f.dev_resids(y[ii], mu[ii], wt[ii]))
                aic += float(f.aic(y[ii], mu[ii], dev_i, wt[ii], n1[ii]))
        return aic

    # ----- pre/post hooks ---------------------------------------------------

    def preinitialize(self, y) -> dict | None:
        """gfam.r:380-418: validate the family index, run member
        preinitializes (may modify y and θ), assemble the initial θ.
        The two-column split itself happens at the gam intake, which
        calls :meth:`set_fi` first."""
        y = np.asarray(y, dtype=float).copy()
        fi = self._fi_checked(y.shape[0])
        nf = len(self._fl)
        ui = np.unique(fi)
        ok = np.isin(ui, np.arange(1, nf + 1)).all() and np.isin(
            np.arange(1, nf + 1), ui).all()
        if not ok:
            raise ValueError("family index does not match family list")
        Theta = np.zeros(self.n_theta)
        theta_mod = False
        for f, ii, nth, fscale, i0 in self._blocks(y.shape[0]):
            if _gfam_is_ext(f) and (type(f).preinitialize
                                  is not Family.preinitialize):
                pri = f.preinitialize(y[ii]) or {}
                if pri.get("y") is not None:
                    y[ii] = np.asarray(pri["y"], dtype=float)
                if pri.get("Theta") is None:
                    Theta[i0:i0 + nth] = np.asarray(
                        f.get_theta(), dtype=float).reshape(-1)[:nth]
                else:
                    theta_mod = True
                    Theta[i0:i0 + nth] = np.asarray(
                        pri["Theta"], dtype=float).reshape(-1)
        ret: dict = {"y": y}
        if theta_mod:
            ret["Theta"] = Theta
        return ret

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """gfam.r:420-458: assemble null deviance (per-member intercept
        models), the relabelled family string, and — when any member
        postproc modifies its deviance (betar) — the total deviance."""
        y = np.asarray(y, dtype=float)
        pw = np.asarray(prior_weights, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        lp = np.asarray(linear_predictors, dtype=float)
        off = np.asarray(offset, dtype=float)
        dev_mod = False
        dev = 0.0
        nulldev = 0.0
        names = []
        for f, ii, nth, fscale, i0 in self._blocks(y.shape[0]):
            if _gfam_is_ext(f):
                pp = f.postproc(y[ii], prior_weights=pw[ii],
                                fitted=fitted[ii],
                                linear_predictors=lp[ii], offset=off[ii],
                                intercept=intercept)
                nulldev += float(pp["null_deviance"])
                names.append(pp.get("family_name", f.name))
                if pp.get("deviance") is None:
                    dev += _rsum(f.dev_resids(y[ii], fitted[ii], pw[ii]))
                else:
                    dev_mod = True
                    dev += float(pp["deviance"])
            else:
                names.append(f.name)
                dev += _rsum(f.dev_resids(y[ii], fitted[ii], pw[ii]))
                # gfam.r:450 calls the GROUPED linkinv on the subsetted
                # offset in the no-intercept case (an R quirk — the
                # grouped slot indexes with full-length fi); the member
                # linkinv is what that line can only have meant.
                wtdmu = (float(_rsum(pw[ii] * y[ii]) / _rsum(pw[ii]))
                         if intercept else f.link.linkinv(off[ii]))
                nulldev += _rsum(f.dev_resids(
                    y[ii], np.full(ii.size, wtdmu)
                    if np.isscalar(wtdmu) else wtdmu, pw[ii]))
        out = {"family_name": "gfam{" + ",".join(names) + "}",
               "null_deviance": nulldev}
        if dev_mod:
            out["deviance"] = dev
        return out

    # ----- residuals (gfam.r:460-482, defined only when need.rsd) ----------

    def _residuals_gfam(self, y, fitted, type: str = "deviance",
                        prior_weights=None):
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        wt = (np.ones(y.shape[0]) if prior_weights is None
              else np.asarray(prior_weights, dtype=float))
        if type == "working":
            # mgcv returns the fit's stored working residuals; hea
            # computes them as its standard working path does.
            eta = self.link.link(fitted)
            return (y - fitted) / self.link.mu_eta(eta)
        rsd = y.copy()
        for f, ii, nth, fscale, i0 in self._blocks(y.shape[0]):
            # residuals.gam recursion on the member sub-object
            # (gfam.r:470-478): member residuals hook if present, else
            # the standard formulas with the member family.
            ext = getattr(f, "residuals_extended", None)
            if ext is not None:
                rsd[ii] = ext(y[ii], fitted[ii], wt[ii], type)
            elif type == "deviance":
                d = np.maximum(f.dev_resids(y[ii], fitted[ii], wt[ii]), 0.0)
                rsd[ii] = np.sign(y[ii] - fitted[ii]) * np.sqrt(d)
            elif type == "response":
                rsd[ii] = y[ii] - fitted[ii]
            elif type == "pearson":
                rsd[ii] = ((y[ii] - fitted[ii])
                           * np.sqrt(wt[ii] / f.variance(fitted[ii])))
            else:
                raise ValueError(f"residual type {type!r} not available")
        return rsd

    # ----- prediction (gfam.r:484-544) --------------------------------------

    def predict(self, se=False, X=None, beta=None, off=None, Vb=None,
                eta=None, y=None, lpi=None) -> dict:
        """Response-scale prediction. ``y`` carries the family index for
        new data — a 2-column array (column 2 the index, as at fitting)
        or a bare index vector; ``None`` falls back to the stored fi
        (training-data prediction), which must match the prediction
        length."""
        if eta is None:
            n = X.shape[0]
        else:
            eta = np.asarray(eta, dtype=float)
            n = eta.shape[0]
        if y is None:
            fi = self._fi
            if fi is None or fi.shape[0] != n:
                raise ValueError("no family index")
        else:
            y = np.asarray(y, dtype=float)
            if y.ndim == 2:
                if y.shape[1] != 2:
                    raise ValueError(
                        "if response is a matrix it must have 2 columns")
                fi = y[:, 1]
            else:
                fi = y
        nf = len(self._fl)
        if not np.isin(np.unique(fi), np.arange(1, nf + 1)).all():
            raise ValueError("family index does not match list of families")
        fit = np.zeros(n)
        se_fit = np.zeros(n) if se else None
        if eta is not None:
            for i, f in enumerate(self._fl):
                ii = np.where(fi == i + 1)[0]
                if ii.size:
                    fit[ii] = f.link.linkinv(eta[ii])
            return {"fit": fit}
        off = np.zeros(n) if off is None else np.asarray(off, dtype=float)
        y_col = (y[:, 0] if (y is not None and y.ndim == 2)
                 else y)          # R's y[ii] linear-indexes column 1
        for i, f in enumerate(self._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size == 0:
                continue
            f_pred = getattr(f, "predict", None)
            if f_pred is None:
                fit[ii] = off[ii] + X[ii] @ beta
                if se:
                    se_fit[ii] = np.sqrt(np.maximum(
                        0.0, np.einsum("ij,jk,ik->i", X[ii], Vb, X[ii])))
                    se_fit[ii] = se_fit[ii] * np.abs(f.link.mu_eta(fit[ii]))
                fit[ii] = f.link.linkinv(fit[ii])
            elif se:
                pr = f_pred(se=True, X=X[ii], beta=beta, off=off[ii],
                            Vb=Vb, eta=None,
                            y=None if y_col is None else y_col[ii],
                            lpi=None)
                pf = np.asarray(pr["fit"])
                if pf.ndim > 1:
                    raise ValueError(
                        "gfam mixed response scale prediction not "
                        "possible here")
                se_fit[ii] = np.asarray(pr["se_fit"], dtype=float)
                fit[ii] = pf
            else:
                fit[ii] = off[ii] + X[ii] @ beta
                pr = f_pred(se=False, eta=fit[ii])
                pf = np.asarray(pr["fit"])
                if pf.ndim > 1:
                    raise ValueError(
                        "gfam mixed response scale prediction not "
                        "possible here")
                fit[ii] = pf
        out = {"fit": fit}
        if se:
            out["se_fit"] = se_fit
        return out

    # ----- initialize / null model (gfam.r:548-587) --------------------------

    def initialize(self, y, wt) -> np.ndarray:
        """gfam.r:548-567: member mustarts on their subsets. Members see
        their STOCK initialize (fix.family's gam patches key on the
        family name and never match "gfam{…}", so they are not applied
        inside the group — mgcv.r:1916)."""
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        mustart = y.copy()
        fi = self._fi_checked(y.shape[0])
        for i, f in enumerate(self._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size:
                mustart[ii] = f.initialize(y[ii], wt[ii])
        return mustart

    def gam_initialize(self, y, wt, n=None) -> np.ndarray:
        return self.initialize(y, wt)

    def validmu(self, mu) -> bool:
        mu = np.asarray(mu, dtype=float)
        fi = self._fi_checked(mu.shape[0])
        for i, f in enumerate(self._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size and not f.validmu(mu[ii]):
                return False
        return True

    def get_null_coef(self, X, y, wt, offset) -> tuple[np.ndarray, float]:
        """gfam.r:569-587 ``get.null.coef``: a per-member-constant null
        model — mean(y) within each family, linked member-wise — instead
        of the single weighted mean of the default. Returns
        ``(null_coef, null_scale)``; the coefficient solve is hea's
        least-squares convention for mgcv's ``qr.coef`` + NA→0. As in
        mgcv, the offset plays no part in the projection (the caller
        adds it back when forming μ_null)."""
        del offset
        y = np.asarray(y, dtype=float)
        wt = np.asarray(wt, dtype=float)
        mum = np.empty(y.shape[0])
        etam = np.empty(y.shape[0])
        fi = self._fi_checked(y.shape[0])
        for i, f in enumerate(self._fl):
            ii = np.where(fi == i + 1)[0]
            if ii.size:
                mum[ii] = float(np.mean(y[ii]))
                etam[ii] = f.link.link(mum[ii] * np.ones(ii.size))
        null_coef, *_ = np.linalg.lstsq(X, etam, rcond=None)
        null_scale = float(
            _rsum(self.dev_resids(y, mum, wt)) / X.shape[0])
        return null_coef, null_scale

    def __repr__(self):
        return f"gfam({[f.name for f in self._fl]})"


_GFAM_MEMBER_NAMES.update({
    "gaussian": Gaussian, "poisson": Poisson, "binomial": Binomial,
    "Gamma": Gamma, "inverse.gaussian": InverseGaussian,
    "quasi": Quasi, "quasipoisson": QuasiPoisson,
    "quasibinomial": QuasiBinomial,
    "nb": nb, "tw": tw, "scat": Scat, "ocat": ocat, "ziP": ziP,
    "betar": betar, "cnorm": cnorm, "cpois": cpois, "clog": clog,
    "bcg": bcg,
})


# ---------------------------------------------------------------------------
# General-family seam — mgcv gamlss.r authoring kit (§5.3 prerequisite 5).
#
# General families (gam.fit5: multiple linear predictors, likelihood
# supplied as ``ll`` instead of a deviance) are authored from per-datum
# derivative arrays of the log-likelihood w.r.t. the distribution
# parameters (μ₁..μ_K), packed in upper-triangular order. The kit:
#   * trind_generator — symmetric index lookups into the packed arrays
#   * gamlss_etamu    — chain rule μ-derivatives → η-derivatives through
#                        the per-LP link derivatives
#   * gamlss_gH       — assemble the coefficient-space gradient/Hessian/
#                        ∂H/∂ρ/tr(H⁻¹∂²H) that gam.fit5 consumes
# A custom family supplies l1..l4 + links; everything downstream is
# generic. Ported complete-array/dense paths
# only — out of scope (absent, never silent): the "remap" dropped-zero-
# column optimization (multinom-scale K), discrete (bam) X lists,
# sandwich, bootstrap deriv<0, the non-linear g.index corrections.
# Index convention: everything 0-based (R's 1-based m and dims shifted).
# ---------------------------------------------------------------------------


def trind_generator(K: int = 2, ifunc: bool = False,
                    reverse: bool | None = None) -> dict:
    """mgcv ``trind.generator`` (gamlss.r:20-112): index lookups for
    upper-triangular packed storage of symmetric derivative arrays up to
    order 4. ``i4[i,j,k,l]`` (0-based everywhere) gives the packed column
    holding the derivative w.r.t. parameters i,j,k,l in any order;
    ``i3``/``i2`` likewise.

    ``ifunc=True`` returns closed-form index *functions* instead of
    arrays (mgcv's storage saver for large K) — same 0-based in/out
    convention, mgcv's exact 1-based algebra inside. ``reverse``
    (default ``not ifunc``, mgcv's coupling) adds ``i2r``/``i3r``/
    ``i4r``: flat indices extracting the unique elements of a full
    symmetric K^d array in packing order. Values are exactly mgcv's
    minus one: mgcv's 1-based column-major ``l + (k-1)*K + …`` of the
    reversed tuple equals the 0-based C-order offset of ``[i,j,k,l]``
    (the digit sum commutes), so ``arr.ravel()[i4r]`` reads the same
    cells R does."""
    if reverse is None:
        reverse = not ifunc
    if ifunc:
        # mgcv's closed forms (1-based); wrap for 0-based i/o.
        def i2(i: int, j: int) -> int:
            i, j = sorted((i + 1, j + 1))
            return int(round((i - 1) * (2 * K + 2 - i) / 2 + j - i + 1)) - 1

        def i3(i: int, j: int, k: int) -> int:
            i, j, k = sorted((i + 1, j + 1, k + 1))
            return int(round(
                (i - 1) * (3 * K * (K + 1) + (i - 2) * (i - 3 * (K + 1))) / 6
                + (j - i) * (2 * K + 3 - i - j) / 2 + k - j + 1
            )) - 1

        def i4(i: int, j: int, k: int, ll_: int) -> int:
            i, j, k, ll_ = sorted((i + 1, j + 1, k + 1, ll_ + 1))
            i1 = i - 1
            i1i2 = i1 * (i - 2) / 2
            return int(round(
                ll_ - k + 1 + (k - j) * (2 * K + 3 - j - k) / 2
                + (j - i) * (3 * (K + 1 - i) ** 2 + 3 * (K + 1 - i)
                             + (j - i - 1) * (j + 2 * i - 3 * K - 5)) / 6
                + (i1 * (K ** 3 + 3 * K ** 2 + 2 * K)
                   + i1i2 * ((K + 1) * (2 * i - 3) - (3 * K ** 2 + 6 * K + 2)
                             - i1i2)) / 6
            )) - 1
    else:
        i4 = np.zeros((K, K, K, K), dtype=int)
        m = 0
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    for ll_ in range(k, K):
                        for perm in itertools.permutations((i, j, k, ll_)):
                            i4[perm] = m
                        m += 1
        i3 = np.zeros((K, K, K), dtype=int)
        m = 0
        for j in range(K):
            for k in range(j, K):
                for ll_ in range(k, K):
                    for perm in itertools.permutations((j, k, ll_)):
                        i3[perm] = m
                    m += 1
        i2 = np.zeros((K, K), dtype=int)
        m = 0
        for k in range(K):
            for ll_ in range(k, K):
                i2[k, ll_] = i2[ll_, k] = m
                m += 1
    i2r = i3r = i4r = None
    if reverse:
        i4r = np.array(
            [((i * K + j) * K + k) * K + ll_
             for i in range(K) for j in range(i, K)
             for k in range(j, K) for ll_ in range(k, K)], dtype=int)
        i3r = np.array(
            [(j * K + k) * K + ll_
             for j in range(K) for k in range(j, K) for ll_ in range(k, K)],
            dtype=int)
        i2r = np.array(
            [k * K + ll_ for k in range(K) for ll_ in range(k, K)], dtype=int)
    return {"i2": i2, "i3": i3, "i4": i4, "i2r": i2r, "i3r": i3r, "i4r": i4r}


def _deriv_orders(idx: tuple[int, ...]) -> np.ndarray:
    """mgcv's ``ordf`` (gamlss.r:254-278): differentiation order carried
    by each slot of a 2-4 index tuple (repeats accumulate on the first
    occurrence, later slots zero out)."""
    idx = tuple(idx)
    d = len(idx)
    ord_ = np.ones(d, dtype=int)
    if d >= 2 and idx[0] == idx[1]:
        ord_[0] += 1
        ord_[1] = 0
    if d >= 3:
        if idx[0] == idx[2]:
            ord_[0] += 1
            ord_[2] = 0
        if ord_[1] and idx[1] == idx[2]:
            ord_[1] += 1
            ord_[2] = 0
    if d == 4:
        if idx[0] == idx[3]:
            ord_[0] += 1
            ord_[3] = 0
        if ord_[1]:
            if idx[1] == idx[3]:
                ord_[1] += 1
                ord_[3] = 0
        if ord_[2] and idx[2] == idx[3]:
            ord_[2] += 1
            ord_[3] = 0
    return ord_


def gamlss_etamu(l1, l2, l3=None, l4=None, ig1=None, g2=None, g3=None,
                 g4=None, i2=None, i3=None, i4=None, deriv: int = 0) -> dict:
    """mgcv ``gamlss.etamu`` (gamlss.r:231-584), complete-array paths:
    transform packed log-likelihood derivatives w.r.t. the distribution
    parameters (μ₁..μ_K) into derivatives w.r.t. the linear predictors
    (η₁..η_K). ``ig1[:,k]`` = 1/g'(μ_k) (= dμ_k/dη_k), ``g2``-``g4`` the
    per-LP link derivatives d²g/dμ²… evaluated at μ_k. ``deriv``: 0 →
    l1,l2 only; >0 adds l3; >2 adds l4 (mgcv's convention — it is the
    ll-level deriv minus one)."""
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    K = l1.shape[1]
    d1 = l1 * ig1

    d2 = np.array(l2, dtype=float, copy=True)
    k = 0
    for i in range(K):
        for j in range(i, K):
            ord_ = _deriv_orders((i, j))
            if ord_.max() == 2:
                d2[:, k] = ((l2[:, k] - l1[:, i] * g2[:, i] * ig1[:, i])
                            * ig1[:, i] ** 2)
            else:
                d2[:, k] = l2[:, k] * ig1[:, i] * ig1[:, j]
            k += 1

    d3 = l3
    if deriv > 0:
        l3 = np.asarray(l3, dtype=float)
        d3 = np.array(l3, dtype=float, copy=True)
        k = 0
        for i in range(K):
            for j in range(i, K):
                for ll_ in range(j, K):
                    ord_ = _deriv_orders((i, j, ll_))
                    ii = np.array((i, j, ll_))
                    mo = int(ord_.max())
                    if mo == 3:
                        mind = i2[i, i]
                        d3[:, k] = ((l3[:, k]
                                     - 3.0 * l2[:, mind] * g2[:, i]
                                     * ig1[:, i]
                                     + l1[:, i] * (3.0 * g2[:, i] ** 2
                                                   * ig1[:, i] ** 2
                                                   - g3[:, i] * ig1[:, i]))
                                    * ig1[:, i] ** 3)
                    elif mo == 1:
                        d3[:, k] = (l3[:, k] * ig1[:, i] * ig1[:, j]
                                    * ig1[:, ll_])
                    else:
                        k1 = int(ii[ord_ == 1][0])
                        k2 = int(ii[ord_ == 2][0])
                        mind = i2[k2, k1]
                        d3[:, k] = ((l3[:, k] - l2[:, mind] * g2[:, k2]
                                     * ig1[:, k2])
                                    * ig1[:, k1] * ig1[:, k2] ** 2)
                    k += 1

    d4 = l4
    if deriv > 2:
        l4 = np.asarray(l4, dtype=float)
        d4 = np.array(l4, dtype=float, copy=True)
        k = 0
        for i in range(K):
            for j in range(i, K):
                for ll_ in range(j, K):
                    for m_ in range(ll_, K):
                        ord_ = _deriv_orders((i, j, ll_, m_))
                        ii = np.array((i, j, ll_, m_))
                        mo = int(ord_.max())
                        if mo == 4:
                            mi2 = i2[i, i]
                            mi3 = i3[i, i, i]
                            d4[:, k] = ((
                                l4[:, k]
                                - 6.0 * l3[:, mi3] * g2[:, i] * ig1[:, i]
                                + l2[:, mi2] * (15.0 * g2[:, i] ** 2
                                                * ig1[:, i] ** 2
                                                - 4.0 * g3[:, i]
                                                * ig1[:, i])
                                - l1[:, i] * (15.0 * g2[:, i] ** 3
                                              * ig1[:, i] ** 3
                                              - 10.0 * g2[:, i] * g3[:, i]
                                              * ig1[:, i] ** 2
                                              + g4[:, i] * ig1[:, i])
                            ) * ig1[:, i] ** 4)
                        elif mo == 1:
                            d4[:, k] = (l4[:, k] * ig1[:, i] * ig1[:, j]
                                        * ig1[:, ll_] * ig1[:, m_])
                        elif mo == 3:
                            k1 = int(ii[ord_ == 1][0])
                            k3 = int(ii[ord_ == 3][0])
                            mi2 = i2[k3, k1]
                            mi3 = i3[k3, k3, k1]
                            d4[:, k] = ((
                                l4[:, k]
                                - 3.0 * l3[:, mi3] * g2[:, k3] * ig1[:, k3]
                                + l2[:, mi2] * (3.0 * g2[:, k3] ** 2
                                                * ig1[:, k3] ** 2
                                                - g3[:, k3] * ig1[:, k3])
                            ) * ig1[:, k1] * ig1[:, k3] ** 3)
                        elif int(np.sum(ord_ == 2)) == 2:
                            two = ii[ord_ == 2]
                            k2a, k2b = int(two[0]), int(two[1])
                            mi2 = i2[k2a, k2b]
                            mi3 = i3[k2a, k2b, k2b]
                            mi3a = i3[k2a, k2a, k2b]
                            d4[:, k] = ((
                                l4[:, k]
                                - l3[:, mi3] * g2[:, k2a] * ig1[:, k2a]
                                - l3[:, mi3a] * g2[:, k2b] * ig1[:, k2b]
                                + l2[:, mi2] * g2[:, k2a] * g2[:, k2b]
                                * ig1[:, k2a] * ig1[:, k2b]
                            ) * ig1[:, k2a] ** 2 * ig1[:, k2b] ** 2)
                        else:
                            k2 = int(ii[ord_ == 2][0])
                            ones = ii[ord_ == 1]
                            k1a, k1b = int(ones[0]), int(ones[1])
                            mi3 = i3[k2, k1a, k1b]
                            d4[:, k] = ((l4[:, k] - l3[:, mi3] * g2[:, k2]
                                         * ig1[:, k2])
                                        * ig1[:, k1a] * ig1[:, k1b]
                                        * ig1[:, k2] ** 2)
                        k += 1

    return {"l1": d1, "l2": d2, "l3": d3, "l4": d4}


class DiscreteX:
    """The list-form design mgcv's general-family ``ll``/``gamlss.gH``
    consume on the discrete path — hea's analog of the bundle
    ``list(Xd=..., kd=..., ks=..., ts=..., dt=..., v=..., qc=...,
    drop=..., lpid=...)`` (bam.r:1996). ``design`` is the compressed
    :class:`hea.models.bam.DiscreteDesign` (≡ Xd/kd/ks/ts/dt/v/qc —
    hea's terms carry constraints and coefficient slices themselves);
    ``lpid[j]`` lists LP j's 0-based term indices in ascending
    coefficient order (≡ ``X$lpid``, bam.r:2550-2553). The per-LP
    coefficient index arrays (mgcv's ``attr(X, "lpi")``) stay the
    explicit ``lpi`` argument hea families already take."""
    __slots__ = ("design", "lpid")

    def __init__(self, design, lpid):
        self.design = design
        self.lpid = [list(ix) for ix in lpid]


def _discrete_kernels():
    """Deferred import of the compressed-design kernels (family.py loads
    before hea.models.bam; call-time import mirrors ``_pen_reg``)."""
    from .models.bam import Xbd, XWXd, XWyd
    return Xbd, XWXd, XWyd


class _DiscreteLPSolve:
    """The gamlss discrete-``initialize`` solve, repeated verbatim per
    family/LP in mgcv (gamlss.r:1035-1046 and its twins :1368-1378,
    :2323-2331, :2872-2882, :3201-3211): factor ``mchol(XWXd(…, lt) +
    crossprod(E_cols))`` once — pivoted Cholesky with mgcv's rank
    truncation — then pivot-backsolve right-hand sides against it.
    Kept as factor + :meth:`solve` because gumbls' mean pass 2 (:3233)
    and gevlss' ξ line-search (:2359) re-solve the SAME factor with
    fresh rhs vectors (``Xty*m`` scaled AFTER assembly there, so
    :meth:`xty` is exposed separately too)."""

    def __init__(self, design, lt, E_cols, ones_n):
        from .models.gam import _pivoted_chol
        _, _XWXd, self._XWyd = _discrete_kernels()
        self._design = design
        self._lt = lt
        self._ones_n = ones_n
        A = _XWXd(design, ones_n, lt=lt) + E_cols.T @ E_cols
        U, piv, rrank = _pivoted_chol(A)
        self._p = A.shape[0]
        self._U = U[:rrank, :rrank]
        self._piv = piv[:rrank]

    def xty(self, target):
        """``XWyd(Xd, 1, target, lt)`` — the rhs assembly (:1039)."""
        return self._XWyd(self._design, self._ones_n, target, lt=self._lt)

    def solve(self, Xty):
        """``startji[piv] <- backsolve(R, forwardsolve(t(R), Xty[piv]))``
        on the truncated factor (:1041-1045); the caller applies (or,
        where mgcv omits it, skips) the non-finite→0 guard."""
        from scipy.linalg import solve_triangular
        startji = np.zeros(self._p)
        z = solve_triangular(self._U, Xty[self._piv], lower=False,
                             trans="T")
        startji[self._piv] = solve_triangular(self._U, z, lower=False)
        return startji

    def solve_target(self, target):
        return self.solve(self.xty(target))


def gamlss_gH(X, jj, l1, l2, i2, l3=None, i3=None, l4=None, i4=None,
              d1b=None, d2b=None, deriv: int = 0, fh=None,
              D=None, sandwich: bool = False) -> dict:
    """mgcv ``gamlss.gH`` (gamlss.r:587-857): coefficient-space
    quantities from η-space derivative arrays — dense complete-array
    paths, plus the discrete branch (mgcv's ``is.list(X)``,
    gamlss.r:604-711) when ``X`` is a :class:`DiscreteX`: gradient
    blocks via ``XWyd(…, lt=lpid[i])`` (:625), Hessian LP-block pairs
    via ``XWXd(…, lt=lpid[i], rt=lpid[j])`` (:656), ``d1eta`` via
    ``Xbd(…, lt=lpid[i])`` (:683) and the ``deriv==1`` trace
    accumulation (:700-734); ``deriv>1`` stops exactly like mgcv
    (:777) — the n×p design is never materialised.

    ``jj[i]`` = LP i's column indices into X (0-based). ``deriv``:
      0 — ``lb`` (gradient) and ``lbb`` (Hessian) only;
      1 — + ``d1H`` as the vector tr(Hp⁻¹·∂H/∂ρ_l) (``fh`` must be the
          INVERSE penalized Hessian);
      2 — + ``d1H`` as the list of full ∂H/∂ρ_l matrices;
      3 — + ``trHid2H`` (``fh`` the pivoted Cholesky of the diagonally
          preconditioned Hp, ``D`` the preconditioner — gam.fit5's
          convention; or an eigendecomposition dict {values, vectors}).

    ``sandwich=True`` (gamlss.r:643-649) replaces ``l2`` so the assembled
    ``lbb`` becomes the per-observation gradient outer-product sum — the
    "filling" of the robust sandwich covariance — instead of the Hessian.
    """
    discrete = isinstance(X, DiscreteX)
    if discrete:
        _Xbd, _XWXd, _XWyd = _discrete_kernels()
        design = X.design
        lpid = X.lpid
        n = design.n
        p = design.p
    else:
        X = np.asarray(X, dtype=float)
        n, p = X.shape
    K = len(jj)
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    lb = np.zeros(p)
    if discrete:
        ones_n = np.ones(n)
        for i in range(K):
            lb[jj[i]] += _XWyd(design, ones_n, l1[:, i], lt=lpid[i])
    else:
        for i in range(K):
            lb[jj[i]] += X[:, jj[i]].T @ l1[:, i]

    if sandwich:
        # mgcv gamlss.r:643-649: reset l2 so the "Hessian" becomes the
        # sandwich filling, l2[, i2[i,j]] = l1[,i]·l1[,j]. mgcv writes the
        # pair-counter column k, which equals i2[i,j] in trind's (i, j≥i)
        # packing — indexed here explicitly.
        if deriv > 0:
            import warnings
            warnings.warn("sandwich requested with higher derivatives",
                          stacklevel=2)
        l2 = l2.copy()
        for i in range(K):
            for j in range(i, K):
                l2[:, i2[i, j]] = l1[:, i] * l1[:, j]

    # crossprod(X_i, (w·l2)·X_j) per LP block. The hot path is numpy `@`
    # (Accelerate/BLAS GEMM, ~peak FLOPS — mgcv uses the same). But an optimized
    # GEMM tiles its output rows independently AND picks its micro-kernel by
    # array alignment, so two bit-identical input columns (a rank-deficient
    # duplicate covariate) can get output rows differing by ~1e-13 — enough to
    # flip gam.fit5's end-stage QR rank-check pivot tie (gam.fit4.r:1172) →
    # a different unidentifiable column dropped, platform-dependently. That only
    # matters AT the rank check, so gam.fit5 recomputes this Hessian under
    # `deterministic_xwx()` there (and only there): `_xwx` is a fixed-order
    # reduction (rust gamlss_xwx, else einsum) — construction-identical across
    # rows for identical columns, as mgcv's reference-BLAS crossprod is.
    det = _GAMLSS_XWX_DETERMINISTIC
    lbb = np.zeros((p, p))
    for i in range(K):
        for j in range(i, K):
            if discrete:
                # gamlss.r:656-659: the (i,j) LP block straight off the
                # compressed design; XWXd's lt/rt returns exactly the
                # |jj[i]| × |jj[j]| block in coefficient order.
                A = _XWXd(design, l2[:, i2[i, j]], lt=lpid[i],
                          rt=lpid[j])
            else:
                Xi = X[:, jj[i]]
                WXj = l2[:, i2[i, j]][:, None] * X[:, jj[j]]
                A = _xwx(Xi, WXj) if det else Xi.T @ WXj
            lbb[np.ix_(jj[i], jj[j])] += A
            if j > i:
                lbb[np.ix_(jj[j], jj[i])] += A.T

    d1H = None
    trHid2H = None
    if deriv > 0:
        l3 = np.asarray(l3, dtype=float)
        d1b = np.asarray(d1b, dtype=float)
        m = d1b.shape[1]
        # Stacked per-LP derivative of η w.r.t. each ρ (gamlss.r:680-686);
        # discrete: Xbd of the FULL d1b restricted to LP i's terms (:683).
        d1eta = np.zeros((n * K, m))
        for i in range(K):
            d1eta[i * n:(i + 1) * n, :] = (
                _Xbd(design, d1b, lt=lpid[i]) if discrete
                else X[:, jj[i]] @ d1b[jj[i], :])

    if deriv == 1:
        # tr(Hp⁻¹ ∂H/∂ρ_l) accumulation; fh is the inverse penalized
        # Hessian. Discrete: gamlss.r:700-734 — form the (i,j) block of
        # ∂H/∂ρ_l as XWXd(v) and trace against the matching fh block
        # (Wood's :701 note confirms jj, not lpi, indexes fh here).
        # Dense: gamlss.r:735-773.
        fh = np.asarray(fh, dtype=float)
        d1H = np.zeros(m)
        if discrete:
            for i in range(K):
                for j in range(i, K):
                    mult = 1.0 if i == j else 2.0
                    for ll_ in range(m):
                        v = np.zeros(n)
                        for q in range(K):
                            v += (l3[:, i3[i, j, q]]
                                  * d1eta[q * n:(q + 1) * n, ll_])
                        XVX = _XWXd(design, v, lt=lpid[i], rt=lpid[j])
                        d1H[ll_] += mult * float(
                            np.sum(XVX * fh[np.ix_(jj[i], jj[j])]))
        else:
            for i in range(K):
                for j in range(i, K):
                    Hpi = fh[np.ix_(jj[i], jj[j])]
                    a = np.einsum("ij,ij->i", X[:, jj[i]] @ Hpi,
                                  X[:, jj[j]])
                    mult = 1.0 if i == j else 2.0
                    for ll_ in range(m):
                        v = np.zeros(n)
                        for q in range(K):
                            v += l3[:, i3[i, j, q]] * d1eta[q * n:(q + 1)
                                                            * n, ll_]
                        d1H[ll_] += mult * float(np.sum(a * v))

    if deriv > 1 and discrete:
        # mgcv gamlss.r:777 stops the discrete path at first-order Hessian
        # derivatives; the trace form above is all the discrete REML
        # machinery gets (sp selection runs EFS/BFGS, never full Newton).
        raise NotImplementedError(
            "er... no discrete methods for higher derivatives")

    if deriv > 1:
        # Full ∂H/∂ρ_l matrices (gamlss.r:776-796).
        d1H = []
        for ll_ in range(m):
            Hl = np.zeros((p, p))
            for i in range(K):
                for j in range(i, K):
                    v = np.zeros(n)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n:(q + 1) * n,
                                                        ll_]
                    A = X[:, jj[i]].T @ (v[:, None] * X[:, jj[j]])
                    Hl[np.ix_(jj[i], jj[j])] += A
                    if j > i:
                        Hl[np.ix_(jj[j], jj[i])] += A.T
            d1H.append(Hl)

    if deriv > 2:
        # tr(Hp⁻¹ ∂²H/∂ρ_k∂ρ_l) (gamlss.r:798-855).
        l4 = np.asarray(l4, dtype=float)
        d2b = np.asarray(d2b, dtype=float)
        Xe = np.zeros((K * n, p))
        for i in range(K):
            Xe[i * n:(i + 1) * n, jj[i]] = X[:, jj[i]]
        if isinstance(fh, dict):
            dvals = np.asarray(fh["values"], dtype=float).copy()
            dvals[dvals > 0] = 1.0 / dvals[dvals > 0]
            dvals[dvals <= 0] = 0.0
            V = np.asarray(fh["vectors"], dtype=float)
            Hinv = V @ (dvals[:, None] * V.T)
            Xe_solved = (D[:, None] * (Hinv @ (D[:, None] * Xe.T))).T
        else:
            # fh: pivoted upper-Cholesky (R chol(...,pivot=TRUE) analog)
            # with pivot vector in fh[1]; D the diagonal preconditioner.
            R_f, piv = fh
            DXt = (D[:, None] * Xe.T)[piv, :]
            tmp = solve_triangular(R_f, DXt, lower=False, trans="T")
            sol = solve_triangular(R_f, tmp, lower=False)
            ipiv = np.empty_like(piv)
            ipiv[piv] = np.arange(p)
            Xe_solved = (D[:, None] * sol[ipiv, :]).T
        d2eta = np.zeros((n * K, d2b.shape[1]))
        for i in range(K):
            d2eta[i * n:(i + 1) * n, :] = X[:, jj[i]] @ d2b[jj[i], :]
        n2 = d2b.shape[1]
        trHid2H = np.zeros(n2)
        VX = np.zeros((K * n, p))
        kk = 0
        for k_ in range(m):
            for ll_ in range(k_, m):
                VX[:] = 0.0
                for i in range(K):
                    for j in range(K):
                        v = np.zeros(n)
                        for q in range(K):
                            v += (d2eta[q * n:(q + 1) * n, kk]
                                  * l3[:, i3[i, j, q]])
                            for s in range(K):
                                v += (d1eta[q * n:(q + 1) * n, k_]
                                      * d1eta[s * n:(s + 1) * n, ll_]
                                      * l4[:, i4[i, j, q, s]])
                        VX[j * n:(j + 1) * n, jj[i]] = (v[:, None]
                                                        * X[:, jj[i]])
                trHid2H[kk] = float(np.sum(Xe_solved * VX))
                kk += 1

    return {"lb": lb, "lbb": lbb, "d1H": d1H, "trHid2H": trHid2H}


def _pen_reg(x: np.ndarray, e: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mgcv ``pen.reg`` (gamlss.r:1415-1453): penalized regression of y
    on x with square-root penalty e used as a *regularizer* — the
    penalty weight k is grown/shrunk (×10 / ÷5) until the edf lands in
    (0.85·rank(x), rank(x) − 0.1·re]. Used by general-family
    ``initialize`` when E arrives without mgcv's ``use.unscaled``
    attribute (the initial.spg path)."""
    # local import: hea.models.gam imports this module at load time,
    # so the reverse import must be deferred to call time.
    from .models.gam import _R_rank
    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)
    y = np.asarray(y, dtype=float)
    if float(np.sum(np.abs(e))) == 0.0:
        b, *_ = np.linalg.lstsq(x, y, rcond=None)
        b[~np.isfinite(b)] = 0.0
        return b
    from scipy.linalg import qr as _scipy_qr
    Q_x, R, piv = _scipy_qr(x, mode="economic", pivoting=True)
    r = R.shape[1]
    rr = _R_rank(R, tol=float(np.finfo(float).eps) ** 0.9)
    R_unpiv = np.empty_like(R)
    R_unpiv[:, piv] = R                      # R[, pivot] <- R
    R = R_unpiv
    Qy = Q_x.T @ y                           # qr.qty(...)[1:ncol(R)]

    def _edf_and_R(k):
        aug = np.vstack([R, e * k])
        Q_a, R_a = np.linalg.qr(aug, mode="reduced")
        return float(np.sum(Q_a[:r] ** 2)), R_a

    norm_R = float(np.abs(R).sum(axis=0).max())      # R norm(): "O"
    norm_e = float(np.abs(e).sum(axis=0).max())
    k = 0.01 * norm_R / norm_e
    edf, R_a = _edf_and_R(k)
    re = (min(int(np.sum(np.abs(e).sum(axis=0) != 0)), e.shape[0])
          - _R_rank(R_a, tol=float(np.finfo(float).eps) ** 0.9) + rr)
    while edf > rr - 0.1 * re:               # increase penalization
        k = k * 10.0
        edf, _ = _edf_and_R(k)
    while edf < 0.85 * rr:                   # reduce penalization
        k = k / 5.0
        edf, _ = _edf_and_R(k)
    aug = np.vstack([R, e * k])
    rhs = np.concatenate([Qy, np.zeros(e.shape[0])])
    b, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
    b[~np.isfinite(b)] = 0.0
    return b


class LogbLink(Link):
    """mgcv's ``logb`` link for gaulss's precision LP (gamlss.r:887-900):
    η = log(1/μ − b) so μ = 1/(exp(η) + b) stays below 1/b (τ = 1/σ
    bounded away from ∞ ⇒ σ > b)."""
    name = "logb"

    def __init__(self, b: float = 0.01):
        self.b = float(b)

    def link(self, mu):
        return np.log(1.0 / np.asarray(mu, dtype=float) - self.b)

    def linkinv(self, eta):
        return 1.0 / (np.exp(np.asarray(eta, dtype=float)) + self.b)

    def mu_eta(self, eta):
        ee = np.exp(np.asarray(eta, dtype=float))
        return -ee / (ee + self.b) ** 2

    def _mub(self, mu):
        return np.maximum(1.0 - np.asarray(mu, dtype=float) * self.b,
                          np.finfo(float).eps)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return (2.0 * mub - 1.0) / (mub * mu) ** 2

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return ((1.0 - mub) * mub * 6.0 - 2.0) / (mub * mu) ** 3

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = self._mub(mu)
        return ((((24.0 * mub - 36.0) * mub + 24.0) * mub - 6.0)
                / (mub * mu) ** 4)


class GeneralFamily(Family):
    """Base for mgcv "general families" (gam.fit5): several linear
    predictors, the likelihood supplied directly via :meth:`ll` instead
    of a deviance/PIRLS interface.

    **Authoring contract.** This is hea's public extension API for
    new general families (mgcv's ``general.family`` analog), frozen by
    ``test_general_family_authoring_contract`` (tests/test_gam.py).

    Attributes a subclass declares:

    - ``n_lp`` — number of linear predictors; ``gam`` takes a list of
      exactly ``n_lp`` formulas, one per LP.
    - ``links`` — list of ``n_lp`` :class:`Link` objects (set via
      ``__init__``); custom subclasses welcome. Each implements
      ``link``/``linkinv``/``mu_eta`` plus ``d2link``..``d4link`` up
      to the order ``available_derivs`` implies (the chain rule runs
      through :func:`gamlss_etamu`); clamp ``linkinv`` inside open
      supports and floor ``mu_eta`` like mgcv's links do.
    - ``available_derivs`` — 2: full outer Newton, :meth:`ll` must
      answer every ``deriv`` ≤ 4. 0: extended Fellner-Schall;
      :meth:`ll` is only ever called with ``deriv`` ≤ 1, on every
      path (free, fixed and absent sp). 1: reserved for the unported
      bfgs route — fitting refuses unless ``optimizer="efs"`` is
      passed (mgcv.r:1907).
    - ``discrete_ok`` — mgcv's ``discrete.ok``: declare ``True`` only
      when :meth:`ll` and :meth:`initialize_coef` dispatch on a
      :class:`DiscreteX` design (per-LP ``Xbd`` linear predictors and
      the ``mchol``-solve initializer, gamlss.r:936/1033 pattern);
      ``bam(list-of-formulas, discrete=True)`` refuses families
      without it.
    - conventional flags, as on :class:`gaulss`: ``scale_known =
      True``, ``n_theta = 0``; ``name`` is what summaries print.

    Engine call protocol (signatures are the contract):

    - ``ll(y, X, coef, wt, *, lpi, offset=None, deriv=0, d1b=None,
      d2b=None, fh=None, D=None)`` — ``lpi`` is a list of ``n_lp``
      0-based integer column-index arrays into the stacked ``X``;
      ``offset`` a per-LP list (entries ``None`` for offset-free
      formulas) or ``None``; ``wt`` the (n,) prior weights, forwarded
      by the engine — note mgcv's own general families (gaulss,
      twlss) leave the likelihood unweighted and consume prior
      weights only in residuals/postproc; follow your reference.
      Deriv levels: :meth:`ll`.
    - ``initialize_coef(y, X, lpi, E=None, offset=None,
      use_unscaled=False)`` — called with ``use_unscaled=True`` from
      gam.fit5 (E = the ldetS root, gam.fit4.r:974) and with the
      default ``False`` from the initial.spg seed (E = the balanced
      root, pen.reg semantics).
    - ``postproc(y, prior_weights, fitted, linear_predictors, offset,
      intercept)`` — mgcv's 6-argument form (unified 2026-06-11),
      keyword-called once on the converged fit; see :meth:`postproc`.
    - ``residuals(y, fitted, type="deviance")`` — REQUIRED for
      general families: the fit stores ``residuals(y, fitted)`` and
      ``residuals_of(type=)``/qq dispatch through it (mgcv.r:3429);
      ``fitted`` is the (n, n_lp) inverse-linked matrix. A hook MAY
      additionally declare an optional ``prior_weights`` keyword —
      the engine passes the fit's prior weights when it is declared
      (twlss's deviance residuals carry mgcv's
      ``object$prior.weights``).
    - ``rd(rng, mu, wt, scale)`` — optional; enables qq.gam's
      simulation path (``mu`` = the fitted matrix, like
      :class:`gaulss`).

    Almost always :meth:`ll` is implemented by filling the packed
    per-datum arrays l1..l4 of log-density derivatives w.r.t. the
    distribution parameters and delegating to :func:`gamlss_etamu` +
    :func:`gamlss_gH` exactly like :class:`gaulss` does
    (:func:`trind_generator` supplies the packed index tables).
    """
    is_general = True
    n_lp: int = 2
    available_derivs: int = 2
    canonical_link_name = "none"
    # mgcv ``family$discrete.ok`` (gamlss.r:1102/1409/2444/2978/3327 —
    # set on gaulss/multinom/gevlss/gammals/gumbls and consumed nowhere
    # in mgcv itself; hea's discrete general-family bam is the first
    # consumer). True declares that :meth:`ll` and
    # :meth:`initialize_coef` accept a :class:`DiscreteX` design, i.e.
    # the family carries the per-LP ``Xbd``/``mchol`` discrete branches
    # (gamlss.r:936/1033 pattern). ``bam(list-of-formulas,
    # discrete=True)`` refuses families without it.
    discrete_ok: bool = False
    # mgcv ``family$sandwich`` availability: True on the families whose
    # constructors define the slot (gaulss/gammals/gumbls/shash/gevlss/
    # twlss/multinom/ziplss); False ⇒ vcov(sandwich=True) raises exactly
    # like mgcv's ``is.null(family$sandwich)`` stop (mgcv.r:4381 — e.g.
    # cox.ph, mvn). Subclasses setting True must accept ``sandwich=`` in
    # their ``ll`` and thread it to :func:`gamlss_gH`.
    has_sandwich: bool = False

    def __init__(self, links: list[Link]):
        self.links = links
        # Family base wires a single .link; point it at LP1's for the
        # odd shared code path that asks (residual helpers etc.).
        self.link = links[0]

    def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        """Log-likelihood + coefficient-space derivatives at ``coef``.

        ``deriv``: 0 value only; 1 + lb/lbb; 2 + d1H trace vector (fh =
        Hp⁻¹); 3 + d1H matrix list; 4 + trHid2H (fh/D = gam.fit5's
        preconditioned Cholesky pieces). Returns a dict with keys
        ``l`` (+ ``lb``, ``lbb``, ``d1H``, ``trHid2H`` as available).
        """
        raise NotImplementedError

    def sandwich(self, y, X, coef, wt, *, lpi, offset=None):
        """mgcv ``family$sandwich`` (e.g. gaulss, gamlss.r:1011-1014): the
        filling of the robust sandwich covariance — ``ll(deriv=1,
        sandwich=TRUE)$lbb``, the per-observation gradient outer-product
        sum in coefficient space. mgcv defines the slot with an identical
        body on each of the 8 ``has_sandwich`` families; its slot passes
        ``offset=NULL`` to ll regardless of the offset argument (quirk
        mirrored)."""
        if not self.has_sandwich:
            # mgcv gam.sandwich: is.null(family$sandwich) → stop (mgcv.r:4381).
            raise NotImplementedError(
                "no sandwich estimate available for this model")
        return self.ll(y, X, coef, wt, lpi=lpi, offset=None,
                       deriv=1, sandwich=True)["lbb"]

    @staticmethod
    def _apply_prior_weights(wt, l1, l2, l3=None, l4=None):
        """Scale the packed per-datum derivative blocks by ``wt`` for a
        *weighted* log-likelihood (``Σ wt_i·l0_i``).

        A weighted log-likelihood scales every per-observation
        derivative row by ``wt_i``. :func:`gamlss_etamu` and
        :func:`gamlss_gH` are linear, row by row, in ``(l1, l2, l3,
        l4)`` — ``gamlss_gH`` assembles the gradient ``lb`` and Hessian
        ``lbb`` as plain sums over observations — so scaling these
        inputs by ``wt`` yields exactly the weighted-MLE
        gradient/Hessian. Absent derivative orders (``None``) pass
        through unchanged. ``wt`` is the (n,) weight vector; ``l0``
        itself is never scaled (the per-observation log-density is
        reported raw).

        Unlike mgcv's own gamlss families — which drop prior weights —
        hea's general families honour them here as a weighted
        likelihood. At unit weights the scaling is the identity, so
        unweighted fits are bit-for-bit unchanged; weighting a row by
        integer ``w`` is equivalent to duplicating that row ``w`` times.
        """
        w = wt[:, None]
        l1 = l1 * w
        l2 = l2 * w
        if l3 is not None:
            l3 = l3 * w
        if l4 is not None:
            l4 = l4 * w
        return l1, l2, l3, l4

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """Starting coefficients (mgcv ``family$initialize``).

        ``use_unscaled`` mirrors mgcv's ``attr(E, "use.unscaled")``:
        gam.fit5 passes its ldetS penalty root with the attribute set
        (E used as-is in a stacked least squares); initial.spg passes
        the balanced root WITHOUT it, and the initializer then adjusts
        the penalty weight itself (``pen.reg``)."""
        raise NotImplementedError

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """mgcv ``family$postproc`` analog: family-specific deviance /
        null-deviance overrides, evaluated on the converged fit.
        Returns a dict with optional ``deviance`` / ``null_deviance``
        keys; absent keys fall back to estimate.gam's generics
        (deviance = Σ deviance-residuals², mgcv.r:2429). ``fitted`` is
        the (n, n_lp) fitted matrix for general families."""
        return {}


class gaulss(GeneralFamily):
    """Gaussian location-scale general family — mgcv ``gaulss()``
    (gamlss.r:862-1106). LP1 models μ (links: identity/log/inverse/sqrt);
    LP2 models τ = 1/σ through the ``logb`` link (σ > b > 0).

        log f = −½(y−μ)²τ² − ½log(2π) + log τ
    """
    name = "gaulss"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2
    discrete_ok = True          # gamlss.r:1102

    _OK_MU_LINKS = ("identity", "log", "inverse", "sqrt")

    def __init__(self, link: tuple[str, str] = ("identity", "logb"),
                 b: float = 0.01):
        mu_link, tau_link = link
        if mu_link not in self._OK_MU_LINKS:
            raise ValueError(
                f'link "{mu_link}" not available for the mu parameter of '
                f"gaulss; available links are {self._OK_MU_LINKS}"
            )
        if tau_link != "logb":
            raise ValueError(
                'only the "logb" link is available for the precision '
                "parameter of gaulss"
            )
        links = [
            {"identity": IdentityLink, "log": LogLink,
             "inverse": InverseLink, "sqrt": SqrtLink}[mu_link](),
            LogbLink(b=b),
        ]
        self.b = float(b)
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            # gamlss.r:936-938: per-LP η off the compressed design —
            # Xbd of the FULL coef restricted to the LP's terms.
            _Xbd, _, _ = _discrete_kernels()
            eta = _Xbd(X.design, coef, lt=X.lpid[0])
            eta1 = _Xbd(X.design, coef, lt=X.lpid[1])
        else:
            X = np.asarray(X, dtype=float)
            eta = X[:, jj[0]] @ coef[jj[0]]
            eta1 = X[:, jj[1]] @ coef[jj[1]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                eta1 = eta1 + offset[1]
        mu = self.links[0].linkinv(eta)
        tau = self.links[1].linkinv(eta1)

        n = y.shape[0]
        wt = np.ones(n) if wt is None else np.asarray(wt, dtype=float).ravel()
        ymu = y - mu
        ymu2 = ymu * ymu
        tau2 = tau * tau
        l0 = -0.5 * ymu2 * tau2 - 0.5 * np.log(2.0 * np.pi) + np.log(tau)
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret

        l1 = np.column_stack([tau2 * ymu, 1.0 / tau - tau * ymu2])
        # second derivatives, packed (mm, ms, ss)
        l2 = np.column_stack([-tau2, 2.0 * l1[:, 0] / tau,
                              -ymu2 - 1.0 / tau2])
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(eta1)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(tau)])
        l3 = l4 = g3 = g4 = None
        if deriv > 1:
            # third derivatives, packed (mmm, mms, mss, sss)
            zeros = np.zeros(n)
            l3 = np.column_stack([zeros, -2.0 * tau, 2.0 * ymu,
                                  2.0 / tau ** 3])
            g3 = np.column_stack([self.links[0].d3link(mu),
                                  self.links[1].d3link(tau)])
        if deriv > 3:
            # fourth derivatives, packed (mmmm, mmms, mmss, msss, ssss)
            zeros = np.zeros(n)
            l4 = np.column_stack([zeros, zeros, np.full(n, -2.0), zeros,
                                  -6.0 / (tau2 * tau2)])
            g4 = np.column_stack([self.links[0].d4link(mu),
                                  self.links[1].d4link(tau)])

        tri = self.tri
        l1, l2, l3, l4 = self._apply_prior_weights(wt, l1, l2, l3, l4)
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gaulss ``initialize`` (gamlss.r:1016-1086): regress g(y) on
        LP1's columns, then the log absolute residuals on LP2's, with
        the penalty root ``E`` as a regularizer.
        ``use_unscaled`` (mgcv's ``attr(E,"use.unscaled")``, set by
        gam.fit5 on its ldetS root): stacked least squares with E
        as-is; otherwise (initial.spg's balanced root) ``pen.reg``
        adjusts the penalty weight to an edf target. A :class:`DiscreteX`
        design takes the discrete branch (gamlss.r:1033-1062): per LP
        solve ``(X'X + E'E)β = X'target`` through the pivoted Cholesky
        (``mchol``, rank-truncated) — E enters as a plain crossprod
        regularizer regardless of ``use_unscaled`` there."""
        y = np.asarray(y, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            return self._initialize_coef_discrete(y, X, jj, E, offset)
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)
        if self.links[0].name == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], yt1)
        start[jj[0]] = b1
        lres1 = np.log(np.abs(y - self.links[0].linkinv(
            X[:, jj[0]] @ b1)))
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _reg(jj[1], lres1)
        return start

    def _initialize_coef_discrete(self, y, X: DiscreteX, jj, E,
                                  offset) -> np.ndarray:
        """gaulss ``initialize``'s discrete branch (gamlss.r:1033-1062):
        per LP the :class:`_DiscreteLPSolve` factor-and-backsolve with
        non-finite entries zeroed; LP1's residuals via
        ``Xbd(…, lt=lpid[0])`` (:1048)."""
        _Xbd, _, _ = _discrete_kernels()
        design = X.design
        lpid = X.lpid
        n = y.shape[0]
        p = design.p
        if E is None:
            E = np.zeros((0, p))
        E = np.asarray(E, dtype=float)
        ones_n = np.ones(n)

        def _solve_lp(i: int, target: np.ndarray) -> np.ndarray:
            lp = _DiscreteLPSolve(design, lpid[i], E[:, jj[i]], ones_n)
            startji = lp.solve_target(target)
            startji[~np.isfinite(startji)] = 0.0
            return startji

        start = np.zeros(p)
        if self.links[0].name == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]
        start[jj[0]] = _solve_lp(0, yt1)
        eta1 = _Xbd(design, start, lt=lpid[0])
        lres1 = np.log(np.abs(y - self.links[0].linkinv(eta1)))
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _solve_lp(1, lres1)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """gaulss postproc (gamlss.r:910-918): null deviance only —
        ``Σ((y − ȳ)·τ̂)²`` (the fitted-precision-weighted null SS);
        the deviance itself falls back to estimate.gam's generic
        Σ deviance-residuals² (mgcv.r:2429)."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        return {"null_deviance": float(np.sum(
            ((y - float(np.mean(y))) * fitted[:, 1]) ** 2))}

    def rd(self, rng, mu, wt, scale):
        """gaulss rd (gamlss.r:1089): ``rnorm(n, mu[,1],
        sqrt(scale/wt)/mu[,2])`` — μ is the (n, 2) fitted matrix
        (mean, τ = 1/σ); scale ≡ 1 for gaulss fits. Drives qq.gam's
        simulation path (mgcv does NOT qqnorm-fallback for gaulss)."""
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        sd = np.sqrt(float(scale) / wt) / mu[:, 1]
        return rng.normal(mu[:, 0], sd)

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """gaulss residuals (gamlss.r:903-908): response = y − μ̂;
        deviance/pearson = (y − μ̂)·τ̂ = (y − μ̂)/σ̂. ``fitted`` is the
        (n, 2) matrix of (μ̂, τ̂)."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for gaulss residuals; got {type!r}")
        fitted = np.asarray(fitted, dtype=float)
        rsd = np.asarray(y, dtype=float) - fitted[:, 0]
        if type == "response":
            return rsd
        return rsd * fitted[:, 1]

    def __repr__(self):
        return (f"gaulss(link=({self.links[0].name!r}, 'logb'), "
                f"b={self.b:g})")


class twlss(GeneralFamily):
    """Tweedie location-scale-shape general family — mgcv ``twlss()``
    (gamlss.r:2493-2662). Three linear predictors: LP1 the mean μ
    (links: log/identity/sqrt), LP2 the transformed index θ with
    p = (a + b·e^θ)/(1 + e^θ) ∈ (a, b) (identity link), LP3
    ρ = log scale (identity link).

    ``available_derivs = 0``: mgcv supplies no third/fourth
    log-likelihood derivatives, so fitting always runs the extended
    Fellner-Schall loop (mgcv.r:1907-1908's automatic optimizer
    switch). Like mgcv, the likelihood itself ignores prior weights
    (gamlss.r:2556 — ``wt`` unread, same as gaulss); they enter the
    deviance residuals and null deviance only.
    """
    name = "twlss"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 3
    available_derivs = 0

    _OK_MU_LINKS = ("log", "identity", "sqrt")

    def __init__(self, link: tuple[str, str, str] = ("log", "identity",
                                                     "identity"),
                 a: float = 1.01, b: float = 1.99):
        mu_link, th_link, rho_link = link
        if mu_link not in self._OK_MU_LINKS:
            raise ValueError(
                f'link "{mu_link}" not available for the mu parameter '
                f"of twlss; available links are {self._OK_MU_LINKS}"
            )
        if th_link != "identity" or rho_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the theta '
                "and rho parameters of twlss"
            )
        if not (1.0 < a < b < 2.0):
            raise ValueError("1<a<b<2 (strict) required")
        self.a = float(a)
        self.b = float(b)
        links = [
            {"log": LogLink, "identity": IdentityLink,
             "sqrt": SqrtLink}[mu_link](),
            IdentityLink(), IdentityLink(),
        ]
        super().__init__(links)
        self.tri = trind_generator(3)

    def _p_of_theta(self, theta):
        """p(θ) with the ±θ-stable branches (gamlss.r:2528-2532)."""
        theta = np.asarray(theta, dtype=float)
        eth = np.exp(-np.abs(theta))
        return np.where(theta > 0,
                        (self.b + self.a * eth) / (1.0 + eth),
                        (self.b * eth + self.a) / (eth + 1.0))

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None,
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        eta = X[:, jj[0]] @ coef[jj[0]]
        theta = X[:, jj[1]] @ coef[jj[1]]
        rho = X[:, jj[2]] @ coef[jj[2]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                theta = theta + offset[1]
            if len(offset) > 2 and offset[2] is not None:
                rho = rho + offset[2]
        mu = self.links[0].linkinv(eta)

        # ldTweedie columns: l; ρ, ρρ; θ, θθ, θρ; μ, μμ, μθ, μρ —
        # reordered into the packed (μ, θ, ρ) layout (gamlss.r:2575-2580)
        ld = _ld_tweedie_work(y, mu, theta, rho, a=self.a, b=self.b)
        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        l0 = ld[:, 0]
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret
        l1 = ld[:, [6, 3, 1]]
        l2 = ld[:, [7, 8, 9, 4, 5, 2]]
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(theta),
                               self.links[2].mu_eta(rho)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(theta),
                              self.links[2].d2link(rho)])
        # no l3/l4 for this family: etamu/gH run at deriv 0 whenever
        # any derivative is requested (gamlss.r:2592-2599)
        tri = self.tri
        l1, l2, _, _ = self._apply_prior_weights(wt, l1, l2)
        de = gamlss_etamu(l1, l2, None, None, ig1, g2, None, None,
                          tri["i2"], tri["i3"], tri["i4"], 0)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=0,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """twlss ``initialize`` (gamlss.r:2609-2649): regress g(y) on
        LP1's columns, the log absolute scaled residuals
        ``log|((y−μ₁)/μ₁^1.5)|`` on LP3's (the log-scale predictor),
        and start the θ predictor at zero (p = (a+b)/2). E is a
        regularizer; mgcv's expression never references offsets here —
        ported as-is."""
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)
        if self.links[0].name == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                bvec, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                bvec[~np.isfinite(bvec)] = 0.0
                return bvec
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], yt1)
        start[jj[0]] = b1
        mu1 = self.links[0].linkinv(X[:, jj[0]] @ b1)
        lres1 = np.log(np.abs((y - mu1) / mu1 ** 1.5))
        start[jj[2]] = _reg(jj[2], lres1)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """twlss ``postproc`` (gamlss.r:2545-2554): null deviance from
        the intercept-only Tweedie MLE — mgcv calls ``tw.null.fit(y)``
        with ITS defaults a=1.001/b=1.999 even when the family was
        built with other (a, b); ported bug-for-bug — scaled by the
        FITTED per-observation e^ρ."""
        y = np.asarray(y, dtype=float)
        pw = np.asarray(prior_weights, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu0, p0, _phi0 = _tw_null_fit(y)
        y1 = y + (y == 0.0)
        th0 = (y1 ** (1.0 - p0) - mu0 ** (1.0 - p0)) / (1.0 - p0)
        ka0 = (y ** (2.0 - p0) - mu0 ** (2.0 - p0)) / (2.0 - p0)
        nd = np.sum(np.maximum(
            2.0 * (y * th0 - ka0) * pw / np.exp(fitted[:, 2]), 0.0))
        return {"null_deviance": float(nd)}

    def residuals(self, y, fitted, type: str = "deviance",
                  prior_weights=None) -> np.ndarray:
        """twlss residuals (gamlss.r:2522-2543): ``fitted`` is the
        (n, 3) matrix (μ, θ, ρ). Deviance residuals carry mgcv's
        ``object$prior.weights`` — the engine passes them through the
        optional ``prior_weights`` keyword."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for twlss residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu = fitted[:, 0]
        p = self._p_of_theta(fitted[:, 1])
        phi = np.exp(fitted[:, 2])
        if type == "pearson":
            return (y - mu) / np.sqrt(phi * mu ** p)
        if type == "response":
            return y - mu
        pw = (np.ones_like(y) if prior_weights is None
              else np.asarray(prior_weights, dtype=float))
        y1 = y + (y == 0.0)
        th = (y1 ** (1.0 - p) - mu ** (1.0 - p)) / (1.0 - p)
        ka = (y ** (2.0 - p) - mu ** (2.0 - p)) / (2.0 - p)
        return np.sign(y - mu) * np.sqrt(
            np.maximum(2.0 * (y * th - ka) * pw / phi, 0.0))

    def __repr__(self):
        return (f"twlss(link=({self.links[0].name!r}, 'identity', "
                f"'identity'), a={self.a:g}, b={self.b:g})")


class LogebLink(Link):
    """shash's ``logeb`` link for τ = log σ (gamlss.r:3356-3371):
    η = log(e^τ − b), τ = log(e^η + b) — keeps σ = e^τ > b > 0."""

    name = "logeb"

    def __init__(self, b: float = 1e-2):
        self.b = float(b)

    def link(self, mu):
        return np.log(np.exp(np.asarray(mu, dtype=float)) - self.b)

    def linkinv(self, eta):
        return np.log(np.exp(np.asarray(eta, dtype=float)) + self.b)

    def mu_eta(self, eta):
        ee = np.exp(np.asarray(eta, dtype=float))
        return ee / (ee + self.b)

    def d2link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        fr = em / (em - self.b)
        return fr * (1.0 - fr)

    def d3link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        fr = em / (em - self.b)
        oo = fr * (1.0 - fr)
        return oo - 2.0 * oo * fr

    def d4link(self, mu):
        em = np.exp(np.asarray(mu, dtype=float))
        b = self.b
        return (-b * em * (b ** 2 + 4.0 * b * em + em ** 2)
                / (em - b) ** 4)


class shash(GeneralFamily):
    """Sinh-arcsinh location-scale-shape general family — mgcv
    ``shash()`` (gamlss.r:3334-4080). Four linear predictors: LP1 the
    location μ (identity), LP2 τ = log σ through the ``logeb`` link
    (σ > b > 0), LP3 the skewness ε (identity), LP4 the log-kurtosis
    φ (identity; δ = e^φ).

        z = (y − μ)/(σδ),  l = −τ − ½log 2π + log cosh(δ·asinh z − ε)
            − ½log(1 + z²) − ½sinh²(δ·asinh z − ε) − phiPen·φ²

    The phiPen·φ² ridge is part of the LIKELIHOOD itself (mgcv's
    light regularization of the kurtosis direction). Full analytic
    derivatives to order 4 (``available_derivs = 2`` — outer Newton);
    no postproc (mgcv's is commented out, so null deviance is NaN
    like mgcv's NULL); formula offsets are rejected exactly like
    mgcv's ll (gamlss.r:3470). The ``cdf`` hook is ported for surface
    parity (mgcv consumes it only in unported NCV machinery).
    """
    name = "shash"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 4
    available_derivs = 2

    def __init__(self, link: tuple = ("identity", "logeb", "identity",
                                      "identity"),
                 b: float = 1e-2, phiPen: float = 1e-3):
        mu_link, tau_link, eps_link, phi_link = link
        if mu_link != "identity" or eps_link != "identity" \
                or phi_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the mu, eps '
                "and phi parameters of shash"
            )
        if tau_link != "logeb":
            raise ValueError(
                'only the "logeb" link is available for the scale '
                "parameter of shash"
            )
        self.b = float(b)
        self.phiPen = float(phiPen)
        super().__init__([IdentityLink(), LogebLink(b), IdentityLink(),
                          IdentityLink()])
        self.tri = trind_generator(4)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None,
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        # mgcv's shash ll rejects offsets outright (gamlss.r:3470)
        if offset is not None and any(
                o is not None and np.any(np.asarray(o) != 0.0)
                for o in offset):
            raise NotImplementedError(
                "offset not still available for this family (mgcv "
                "shash, gamlss.r:3470)")
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        etas = [X[:, jj[k]] @ coef[jj[k]] for k in range(4)]
        mu = self.links[0].linkinv(etas[0])
        tau = self.links[1].linkinv(etas[1])
        eps = self.links[2].linkinv(etas[2])
        phi = self.links[3].linkinv(etas[3])

        l0, L1, L2, L3, L4 = _shash_derivs(y, mu, tau, eps, phi,
                                           self.phiPen, deriv)
        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret
        params = (mu, tau, eps, phi)
        ig1 = np.column_stack([lnk.mu_eta(eta)
                               for lnk, eta in zip(self.links, etas)])
        g2 = np.column_stack([lnk.d2link(par)
                              for lnk, par in zip(self.links, params)])
        g3 = g4 = None
        if deriv > 1:
            g3 = np.column_stack([lnk.d3link(par)
                                  for lnk, par in zip(self.links,
                                                      params)])
        if deriv > 3:
            g4 = np.column_stack([lnk.d4link(par)
                                  for lnk, par in zip(self.links,
                                                      params)])
        tri = self.tri
        L1, L2, L3, L4 = self._apply_prior_weights(wt, L1, L2, L3, L4)
        de = gamlss_etamu(L1, L2, L3, L4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """shash ``initialize`` (gamlss.r:3973-4024): regress y on
        LP1's columns and the log absolute residuals on LP2's (the
        log-scale predictor), both E-regularized; the skewness and
        log-kurtosis predictors target the constant linkfun(0) = 0
        through plain least squares (Gaussian start)."""
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                bvec, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                bvec[~np.isfinite(bvec)] = 0.0
                return bvec
            return _pen_reg(X[:, cols], E[:, cols], target)

        b1 = _reg(jj[0], y)
        start[jj[0]] = b1
        lres1 = np.log(np.abs(y - self.links[0].linkinv(
            X[:, jj[0]] @ b1)))
        start[jj[1]] = _reg(jj[1], lres1)
        for k in (2, 3):
            target = np.zeros(X.shape[0])
            bvec, *_ = np.linalg.lstsq(X[:, jj[k]], target, rcond=None)
            bvec[~np.isfinite(bvec)] = 0.0
            start[jj[k]] = bvec
        return start

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """shash residuals (gamlss.r:3377-3411): ``fitted`` is the
        (n, 4) matrix (μ, τ, ε, φ). The raw residual subtracts the
        sinh-arcsinh mean (Bessel-K form); deviance residuals use the
        plain log-likelihood against a zero saturated reference
        (mgcv sets ls = 0 — no phiPen term here)."""
        if type not in ("deviance", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'response' for shash "
                f"residuals; got {type!r}")
        from scipy.special import kv
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu, tau, eps, phi = (fitted[:, 0], fitted[:, 1], fitted[:, 2],
                             fitted[:, 3])
        sig = np.exp(tau)
        delta = np.exp(phi)
        rsd = y - mu - sig * delta * np.exp(0.25) * (
            kv((1.0 / delta + 1.0) / 2.0, 0.25)
            + kv((1.0 / delta - 1.0) / 2.0, 0.25)) / np.sqrt(8.0 * np.pi)
        if type == "response":
            return rsd
        sgn = np.sign(rsd)
        z = (y - mu) / (sig * delta)
        dTasMe = delta * np.arcsinh(z) - eps
        ll = (-tau - 0.5 * np.log(2.0 * np.pi) + np.log(np.cosh(dTasMe))
              - 0.5 * np.log1p(z ** 2) - 0.5 * np.sinh(dTasMe) ** 2)
        return np.sqrt(np.maximum(0.0, 2.0 * (0.0 - ll))) * sgn

    def rd(self, rng, mu, wt, scale):
        """shash ``rd`` (gamlss.r:4026-4039): deviates via the quantile
        transform of uniforms — R's ``qnorm(runif(n))`` (one uniform per draw).
        Uses the bit-exact ``_qnorm5`` (AS-241), so given R-stream uniforms the
        deviates are 0-ulp to R."""
        from .R.rng import _qnorm5
        mu = np.asarray(mu, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        n = mu_e.shape[0]
        u = np.array([_qnorm5(float(v)) for v in rng.uniform(size=n)])
        return mu_e + (del_e * sig_e) * np.sinh(
            (1.0 / del_e) * np.arcsinh(u) + eps_e / del_e)

    def qf(self, p, mu, wt, scale):
        """shash quantile function (gamlss.r:4041-4053)."""
        mu = np.asarray(mu, dtype=float)
        p = np.asarray(p, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        return mu_e + (del_e * sig_e) * np.sinh(
            (1.0 / del_e) * np.arcsinh(_nmath.qnorm5_vec(p)) + eps_e / del_e)

    def cdf(self, q, mu, wt, scale, logp: bool = False):
        """shash cdf (gamlss.r:4055-4067). Ported for surface parity —
        mgcv consumes family$cdf only in (unported) NCV machinery."""
        mu = np.asarray(mu, dtype=float)
        q = np.asarray(q, dtype=float)
        mu_e = mu[:, 0]
        sig_e = np.exp(mu[:, 1])
        eps_e = mu[:, 2]
        del_e = np.exp(mu[:, 3])
        s = np.sinh((np.arcsinh((q - mu_e) / (del_e * sig_e))
                     - eps_e / del_e) * del_e)
        return _nmath.pnorm5_vec(s, log_p=True) if logp else _nmath.pnorm5_vec(s)

    def __repr__(self):
        return (f"shash(link=('identity', 'logeb', 'identity', "
                f"'identity'), b={self.b:g}, phiPen={self.phiPen:g})")


class BoundedLogLink(Link):
    """mgcv's bounded "log" link for the log-scale LP of the location-
    scale families ``gammals``/``gumbls`` (gamlss.r:2689-2718).

    (Despite the softplus *form* of its inverse, this is mgcv's bounded
    **log** link — ``name="log"`` — distinct from :class:`BoundedLogLink`, the
    genuine softplus *mean* link for `Poisson()`.)

    Inverse ``g⁻¹(η) = b + log(1 + exp(η − b))`` keeps the (already
    log-scale) parameter strictly above ``b`` — the smooth softplus
    floor mgcv substitutes for a plain ``log`` link when the user asks
    for ``link="log"`` on the scale LP. The display ``name`` is ``"log"``
    (mgcv stores the user's ``paste(link)`` string, so summaries print
    ``log``), exactly as in mgcv. ``d2link``..``d4link`` are mgcv's
    verbatim η-derivative forms; ``mu_eta`` is the logistic
    ``σ(η − b)``."""
    name = "log"

    def __init__(self, b: float = -7.0):
        self.b = float(b)

    def link(self, mu):
        # inverse of the softplus: η = b + log(exp(μ−b) − 1), with the
        # μ−b→0 floor and the μ−b→∞ linear asymptote (gamlss.r:2692-2695).
        mu = np.asarray(mu, dtype=float)
        eps = np.finfo(float).eps
        mub = mu - self.b
        eta = mub.copy()
        ii = mub < eps
        eta[ii] = np.log(eps) + self.b
        jj = mub > -np.log(eps)
        eta[jj] = mub[jj] + self.b
        kk = ~jj & ~ii
        eta[kk] = np.log(np.expm1(mub[kk])) + self.b
        return eta

    def linkinv(self, eta):
        eta = np.asarray(eta, dtype=float)
        mu = eta.copy()
        ii = eta - self.b < -np.log(np.finfo(float).eps)
        mu[ii] = self.b + np.log1p(np.exp(eta[ii] - self.b))
        return mu

    def mu_eta(self, eta):
        # dμ/dη = σ(η − b) (gamlss.r:2697-2701, stable logistic).
        return expit(np.asarray(eta, dtype=float) - self.b)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = mu - self.b
        mub = np.exp(-mub * np.sign(mub))
        return -mub / (mub - 1.0) ** 2

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = mu - self.b
        sm = -np.sign(mub)
        mub = np.exp(mub * sm)
        return sm * (mub + mub ** 2) / (mub - 1.0) ** 3

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        mub = mu - self.b
        sm = -np.sign(mub)
        mub = np.exp(mub * sm)
        return sm * (mub + 4.0 * mub ** 2 + mub ** 3) / (mub - 1.0) ** 4


class gammals(GeneralFamily):
    """Gamma location-scale general family — mgcv ``gammals()``
    (gamlss.r:2664-2980). Two linear predictors, parameterized in **log
    mean** and **log scale**: LP1 is ``log μ`` (identity link only,
    so η₁ ≡ log μ); LP2 is ``log σ`` through the bounded
    :class:`BoundedLogLink` (``link="log"``, σ > exp(b)) or identity.

        log f = (log y − μ − θ)/e^θ − log y − y·e^{−θ−μ} − log Γ(e^{−θ})

    where ``μ = η₁`` (log mean) and ``θ = η₂`` (log scale); the gamma
    has shape ``1/φ`` and scale ``mean·φ`` with ``φ = e^θ`` (so
    Var = mean²·φ). The fitted matrix is reported as ``(mean, log σ)``
    — :meth:`postproc` exponentiates the mean column, mirroring mgcv's
    in-place ``fitted.values[,1] <- exp(...)``.
    """
    name = "gammals"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2
    discrete_ok = True          # gamlss.r:2978

    def __init__(self, link: tuple[str, str] = ("identity", "log"),
                 b: float = -7.0):
        mu_link, scale_link = link
        if mu_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the mean '
                "parameter of gammals"
            )
        if scale_link not in ("identity", "log"):
            raise ValueError(
                f'link "{scale_link}" not available for the scale '
                "parameter of gammals; available links are "
                "('identity', 'log')"
            )
        links = [
            IdentityLink(),
            BoundedLogLink(b=b) if scale_link == "log" else IdentityLink(),
        ]
        self.b = float(b)
        self._scale_link_name = scale_link
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            # gamlss.r:2761-2764: per-LP η off the compressed design.
            _Xbd, _, _ = _discrete_kernels()
            eta = _Xbd(X.design, coef, lt=X.lpid[0])
            etat = _Xbd(X.design, coef, lt=X.lpid[1])
        else:
            X = np.asarray(X, dtype=float)
            eta = X[:, jj[0]] @ coef[jj[0]]
            etat = X[:, jj[1]] @ coef[jj[1]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                etat = etat + offset[1]
        mu = self.links[0].linkinv(eta)     # log mean
        th = self.links[1].linkinv(etat)    # log sigma

        eth = np.exp(-th)
        logy = np.log(y)
        ethmu = np.exp(-th - mu)
        ethmuy = ethmu * y
        etlymt = eth * (logy - mu - th)

        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        l0 = etlymt - logy - ethmuy - gammaln(eth)
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret

        digeth = digamma(eth)
        l1 = np.column_stack([
            ethmuy - eth,                              # lm
            -etlymt + ethmuy + eth * digeth - eth,     # lt
        ])
        eth2 = eth * eth
        treth = polygamma(1, eth)                       # trigamma
        l2 = np.column_stack([
            -ethmuy,                                            # lmm
            eth - ethmuy,                                       # lmt
            etlymt - ethmuy - treth * eth2 - eth * digeth + 2.0 * eth,  # ltt
        ])
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(etat)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(th)])
        l3 = l4 = g3 = g4 = None
        g3eth = None
        if deriv > 1:
            eth3 = eth2 * eth
            g3eth = polygamma(2, eth)
            l3 = np.column_stack([
                ethmuy,            # lmmm
                ethmuy,            # lmmt
                ethmuy - eth,      # lmtt
                (-etlymt + ethmuy + g3eth * eth3 + 3.0 * treth * eth2
                 + eth * digeth - 3.0 * eth),          # lttt
            ])
            g3 = np.column_stack([self.links[0].d3link(mu),
                                  self.links[1].d3link(th)])
        if deriv > 3:
            eth4 = eth3 * eth
            l4 = np.column_stack([
                -ethmuy,           # lmmmm
                -ethmuy,           # lmmmt
                -ethmuy,           # lmmtt
                eth - ethmuy,      # lmttt
                (etlymt - ethmuy - polygamma(3, eth) * eth4
                 - 6.0 * g3eth * eth3 - 7.0 * treth * eth2
                 - eth * digeth + 4.0 * eth),          # ltttt
            ])
            g4 = np.column_stack([self.links[0].d4link(mu),
                                  self.links[1].d4link(th)])

        tri = self.tri
        l1, l2, l3, l4 = self._apply_prior_weights(wt, l1, l2, l3, l4)
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gammals ``initialize`` (gamlss.r:2855-2920): regress
        ``log(y + max(y)·eps^0.75)`` on LP1's columns, then the
        link-transformed log absolute residuals on LP2's, with ``E`` as
        regularizer (``use_unscaled`` ⇒ stacked LS, else ``pen.reg``).
        A :class:`DiscreteX` design takes the discrete branch
        (:2870-2897): the per-LP ``mchol`` solve, E as a plain
        crossprod regularizer regardless of ``use_unscaled``."""
        y = np.asarray(y, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            return self._initialize_coef_discrete(y, X, jj, E, offset)
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        yt1 = np.log(y + float(np.max(y)) * np.finfo(float).eps ** 0.75)
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]
        b1 = _reg(jj[0], yt1)
        start[jj[0]] = b1
        lres1 = self.links[1].link(np.log(np.abs(
            y - self.links[0].linkinv(X[:, jj[0]] @ b1))))
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _reg(jj[1], lres1)
        return start

    def _initialize_coef_discrete(self, y, X: DiscreteX, jj, E,
                                  offset) -> np.ndarray:
        """gammals ``initialize``'s discrete branch (gamlss.r:2870-2897):
        the per-LP ``mchol`` solves; LP1's fitted mean via
        ``Xbd(…, lt=lpid[0])`` (:2885). mgcv guards LP1's solution
        against non-finite entries (:2883) but NOT LP2's (:2896-2897)
        — asymmetry mirrored."""
        _Xbd, _, _ = _discrete_kernels()
        design = X.design
        lpid = X.lpid
        p = design.p
        if E is None:
            E = np.zeros((0, p))
        E = np.asarray(E, dtype=float)
        ones_n = np.ones(y.shape[0])

        start = np.zeros(p)
        yt1 = np.log(y + float(np.max(y)) * np.finfo(float).eps ** 0.75)
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]
        startji = _DiscreteLPSolve(design, lpid[0], E[:, jj[0]],
                                   ones_n).solve_target(yt1)
        startji[~np.isfinite(startji)] = 0.0
        start[jj[0]] = startji
        eta1 = _Xbd(design, start, lt=lpid[0])
        lres1 = self.links[1].link(np.log(np.abs(
            y - self.links[0].linkinv(eta1))))
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _DiscreteLPSolve(design, lpid[1], E[:, jj[1]],
                                        ones_n).solve_target(lres1)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """gammals postproc (gamlss.r:2737-2742): exponentiate the mean
        column of the fitted matrix (LP1 carries log μ) and compute the
        null deviance ``2·Σ((y−ȳ)/ȳ − log(y/ȳ))·e^{−θ̂}``."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        my = float(np.mean(y))
        nd = 2.0 * float(np.sum(
            ((y - my) / my - np.log(y / my)) * np.exp(-fitted[:, 1])))
        new_fitted = fitted.copy()
        new_fitted[:, 0] = np.exp(fitted[:, 0])
        return {"null_deviance": nd, "fitted": new_fitted}

    def rd(self, rng, mu, wt, scale):
        """gammals rd (gamlss.r:2922-2926): ``rgamma(n, 1/φ, mean·φ)``
        with ``φ = e^{θ̂}``. ``mu`` is the (n, 2) fitted matrix
        (mean, log σ); the mean column is already exponentiated by
        :meth:`postproc`."""
        mu = np.asarray(mu, dtype=float)
        phi = np.exp(mu[:, 1])
        return rng.gamma(1.0 / phi, mu[:, 0] * phi)

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """gammals residuals (gamlss.r:2721-2735). ``fitted`` is the
        (n, 2) matrix (mean, log σ) — col 0 already exponentiated by
        :meth:`postproc`."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for gammals residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu = fitted[:, 0]
        rho = fitted[:, 1]
        if type == "response":
            return y - mu
        if type == "pearson":
            return (y - mu) / (np.exp(rho * 0.5) * mu)
        rsd = 2.0 * ((y - mu) / mu - np.log(y / mu)) * np.exp(-rho)
        return np.sqrt(np.maximum(0.0, rsd)) * np.sign(y - mu)

    def predict(self, *, se: bool = False, eta=None, y=None, X=None,
                beta=None, off=None, Vb=None, lpi=None) -> dict:
        """gammals ``family$predict`` (gamlss.r:2928-2969): response-scale
        fit ``(mean, σ) = (e^{η₁}, g₂⁻¹(η₂))`` with delta-method SEs.
        Either ``eta`` (the (n, 2) linear-predictor matrix) or
        ``{X, beta, off, Vb, lpi}`` is supplied; returns ``{"fit": (n, 2)
        [, "se_fit": (n, 2)]}``."""
        if eta is None:
            X = np.asarray(X, dtype=float)
            beta = np.asarray(beta, dtype=float)
            nobs = X.shape[0]
            eta = np.zeros((nobs, 2))
            ve = np.zeros((nobs, 2))
            for i in range(2):
                cols = np.asarray(lpi[i], dtype=int)
                Xi = X[:, cols]
                eta[:, i] = Xi @ beta[cols]
                if off is not None and off[i] is not None:
                    eta[:, i] = eta[:, i] + off[i]
                if se:
                    Vii = Vb[np.ix_(cols, cols)]
                    ve[:, i] = np.maximum(
                        0.0, np.einsum("ij,jk,ik->i", Xi, Vii, Xi))
        else:
            eta = np.asarray(eta, dtype=float)
            se = False
        gamma = np.column_stack([np.exp(eta[:, 0]),
                                 self.links[1].linkinv(eta[:, 1])])
        if se:
            vp = np.column_stack([
                np.abs(gamma[:, 0]) * np.sqrt(ve[:, 0]),
                np.abs(self.links[1].mu_eta(eta[:, 1])) * np.sqrt(ve[:, 1]),
            ])
            return {"fit": gamma, "se_fit": vp}
        return {"fit": gamma}

    def __repr__(self):
        return (f"gammals(link=('identity', {self._scale_link_name!r}), "
                f"b={self.b:g})")


_EULER = 0.5772156649015328606065121   # Euler-Mascheroni constant (gumbls)


class gumbls(GeneralFamily):
    """Gumbel location-scale general family — mgcv ``gumbls()``
    (gamlss.r:2985-3329). Two linear predictors: LP1 the Gumbel
    **location** μ (identity link only, η₁ ≡ μ); LP2 ``log β`` (the
    Gumbel scale) through the bounded :class:`BoundedLogLink`
    (``link="log"``) or identity.

        log f = −β − z − e^{−z},   z = (y − μ)·e^{−β}

    where ``β = η₂`` is log-scale. The fitted matrix is reported as
    ``(mean, log β)`` with ``mean = μ + e^{β}·γ`` (γ = Euler's constant)
    — :meth:`postproc` adds the correction in place, mirroring mgcv's
    ``fitted.values[,1] <- ... + exp(...)·.euler``. Null deviance is NA
    (mgcv leaves it undefined for gumbls).
    """
    name = "gumbls"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2
    discrete_ok = True          # gamlss.r:3327

    def __init__(self, link: tuple[str, str] = ("identity", "log"),
                 b: float = -7.0):
        mu_link, scale_link = link
        if mu_link != "identity":
            raise ValueError(
                'only the "identity" link is available for the location '
                "parameter of gumbls"
            )
        if scale_link not in ("identity", "log"):
            raise ValueError(
                f'link "{scale_link}" not available for the scale '
                "parameter of gumbls; available links are "
                "('identity', 'log')"
            )
        links = [
            IdentityLink(),
            BoundedLogLink(b=b) if scale_link == "log" else IdentityLink(),
        ]
        self.b = float(b)
        self._scale_link_name = scale_link
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            # gamlss.r:3092-3096: per-LP η off the compressed design.
            _Xbd, _, _ = _discrete_kernels()
            eta = _Xbd(X.design, coef, lt=X.lpid[0])
            etab = _Xbd(X.design, coef, lt=X.lpid[1])
        else:
            X = np.asarray(X, dtype=float)
            eta = X[:, jj[0]] @ coef[jj[0]]
            etab = X[:, jj[1]] @ coef[jj[1]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                etab = etab + offset[1]
        mu = self.links[0].linkinv(eta)      # Gumbel location
        beta = self.links[1].linkinv(etab)   # log scale

        eb = np.exp(-beta)
        z = (y - mu) * eb
        ez = np.exp(-z)
        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        l0 = -beta - z - ez
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret

        lz = ez - 1.0
        zm = -eb
        zb = -z
        l1 = np.column_stack([lz * zm, lz * zb - 1.0])
        lzz = -ez
        zmb = eb
        zbb = z
        l2 = np.column_stack([
            lzz * zm ** 2,
            lzz * zm * zb + lz * zmb,
            lzz * zb ** 2 + lz * zbb,
        ])
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(etab)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(beta)])
        l3 = l4 = g3 = g4 = None
        if deriv > 1:
            lzzz = ez
            zbbb = -z
            zmbb = -eb
            l3 = np.column_stack([
                lzzz * zm ** 3,
                lzzz * zm ** 2 * zb + 2.0 * lzz * zm * zmb,
                (lzzz * zb ** 2 * zm + 2.0 * lzz * zb * zmb
                 + lzz * zbb * zm + lz * zmbb),
                lzzz * zb ** 3 + 3.0 * lzz * zb * zbb + lz * zbbb,
            ])
            g3 = np.column_stack([self.links[0].d3link(mu),
                                  self.links[1].d3link(beta)])
        if deriv > 3:
            lzzzz = -ez
            zbbbb = z
            zmbbb = eb
            l4 = np.column_stack([
                lzzzz * zm ** 4,
                lzzzz * zm ** 3 * zb + 3.0 * lzzz * zm ** 2 * zmb,
                (lzzzz * zm ** 2 * zb ** 2 + 4.0 * lzzz * zm * zb * zmb
                 + lzzz * zm ** 2 * zbb + 2.0 * lzz * zmb ** 2
                 + 2.0 * lzz * zm * zmbb),
                (lzzzz * zb ** 3 * zm + 3.0 * lzzz * zb ** 2 * zmb
                 + 3.0 * lzzz * zm * zb * zbb + 3.0 * lzz * zmb * zbb
                 + 3.0 * lzz * zb * zmbb + lzz * zm * zbbb + lz * zmbbb),
                (lzzzz * zb ** 4 + 6.0 * lzzz * zb ** 2 * zbb
                 + 3.0 * lzz * zbb ** 2 + 4.0 * lzz * zb * zbbb
                 + lz * zbbbb),
            ])
            g4 = np.column_stack([self.links[0].d4link(mu),
                                  self.links[1].d4link(beta)])

        tri = self.tri
        l1, l2, l3, l4 = self._apply_prior_weights(wt, l1, l2, l3, l4)
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gumbls ``initialize`` (gamlss.r:3184-3264): two passes —
        regress y on LP1, then ``g₂(½log((y−μ̂)²) − ¼)`` on LP2, then
        re-regress ``y − 0.57721·e^{η₂}`` on LP1. A :class:`DiscreteX`
        design takes the discrete branch (:3199-3235): the per-LP
        ``mchol`` solves, pass 2 re-solving LP1's factor."""
        y = np.asarray(y, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            return self._initialize_coef_discrete(y, X, jj, E, offset)
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        yt1 = y.copy()
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]
        start[jj[0]] = _reg(jj[0], yt1)
        lres1 = self.links[1].link(
            np.log((y - self.links[0].linkinv(X[:, jj[0]] @ start[jj[0]]))
                   ** 2) / 2.0 - 0.25)
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _reg(jj[1], lres1)
        eta2 = X[:, jj[1]] @ start[jj[1]]
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            eta2 = eta2 + offset[1]
        yt1 = yt1 - 0.57721 * np.exp(self.links[1].linkinv(eta2))
        start[jj[0]] = _reg(jj[0], yt1)
        return start

    def _initialize_coef_discrete(self, y, X: DiscreteX, jj, E,
                                  offset) -> np.ndarray:
        """gumbls ``initialize``'s discrete branch (gamlss.r:3199-3235):
        the per-LP ``mchol`` solves with the mean pass 2 (:3228-3235)
        re-solving LP1's factor on ``yt1 − 0.57721·e^{η₂}``. mgcv
        guards LP1's solutions against non-finite entries (:3212,
        :3234) but NOT LP2's (:3226-3227) — asymmetry mirrored. (mgcv
        recycles LP2's ``startji`` vector into pass 2, relying on R's
        assignment-extension; in the full-rank regime every LP1
        position is overwritten, which is the fresh-vector semantics
        used here.)"""
        _Xbd, _, _ = _discrete_kernels()
        design = X.design
        lpid = X.lpid
        p = design.p
        if E is None:
            E = np.zeros((0, p))
        E = np.asarray(E, dtype=float)
        ones_n = np.ones(y.shape[0])

        start = np.zeros(p)
        yt1 = y.copy()
        if offset is not None and offset[0] is not None:
            yt1 = yt1 - offset[0]
        lp1 = _DiscreteLPSolve(design, lpid[0], E[:, jj[0]], ones_n)
        startji = lp1.solve_target(yt1)
        startji[~np.isfinite(startji)] = 0.0
        start[jj[0]] = startji
        eta1 = _Xbd(design, start, lt=lpid[0])
        lres1 = self.links[1].link(
            np.log((y - self.links[0].linkinv(eta1)) ** 2) / 2.0 - 0.25)
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            lres1 = lres1 - offset[1]
        start[jj[1]] = _DiscreteLPSolve(design, lpid[1], E[:, jj[1]],
                                        ones_n).solve_target(lres1)
        # pass 2 at the mean parameter (:3228-3235)
        eta2 = _Xbd(design, start, lt=lpid[1])
        if offset is not None and len(offset) > 1 and offset[1] is not None:
            eta2 = eta2 + offset[1]
        yt1 = yt1 - 0.57721 * np.exp(self.links[1].linkinv(eta2))
        startji = lp1.solve_target(yt1)
        startji[~np.isfinite(startji)] = 0.0
        start[jj[0]] = startji
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """gumbls postproc (gamlss.r:3063-3073): convert the location
        column to the **mean** ``μ + e^{β}·γ`` in place; null deviance
        is left undefined (NA)."""
        fitted = np.asarray(fitted, dtype=float)
        new_fitted = fitted.copy()
        new_fitted[:, 0] = fitted[:, 0] + np.exp(fitted[:, 1]) * _EULER
        return {"fitted": new_fitted}

    def rd(self, rng, mu, wt, scale):
        """gumbls rd (gamlss.r:3268-3275): inverse-CDF Gumbel draws
        ``mean − β·(γ + log(−log U))`` with ``β = e^{fitted[:,1]}``;
        ``mu[:,0]`` is the mean (post-:meth:`postproc`)."""
        mu = np.asarray(mu, dtype=float)
        u = rng.uniform(size=mu.shape[0])
        beta = np.exp(mu[:, 1])
        return mu[:, 0] - beta * (_EULER + np.log(-np.log(u)))

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """gumbls residuals (gamlss.r:3043-3061). ``fitted`` is the
        (n, 2) matrix (mean, log β); the location is recovered as
        ``mean − e^{β}·γ``."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for gumbls residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mean = fitted[:, 0]
        beta = np.exp(fitted[:, 1])
        mu = mean - beta * _EULER
        if type == "response":
            return y - mean
        if type == "pearson":
            return (y - mean) / (np.pi * beta / np.sqrt(6.0))
        z = (y - mu) / beta
        rsd = 2.0 * (z + np.exp(-z) - 1.0)
        return np.sqrt(np.maximum(0.0, rsd)) * np.sign(y - mu)

    def predict(self, *, se: bool = False, eta=None, y=None, X=None,
                beta=None, off=None, Vb=None, lpi=None) -> dict:
        """gumbls ``family$predict`` (gamlss.r:3277-3318): returns the
        (Gumbel **location**, log β) — mgcv does NOT add the Euler mean
        correction here, so the response ``fit`` differs from the mean
        column of ``fitted_values`` (a deliberate mgcv asymmetry). SEs
        are delta-method; mgcv's gumbls uses ``ve[2]`` (a scalar-index
        typo) for the scale SE — hea uses the correct per-row
        ``ve[:,2]``."""
        if eta is None:
            X = np.asarray(X, dtype=float)
            beta = np.asarray(beta, dtype=float)
            nobs = X.shape[0]
            eta = np.zeros((nobs, 2))
            ve = np.zeros((nobs, 2))
            for i in range(2):
                cols = np.asarray(lpi[i], dtype=int)
                Xi = X[:, cols]
                eta[:, i] = Xi @ beta[cols]
                if off is not None and off[i] is not None:
                    eta[:, i] = eta[:, i] + off[i]
                if se:
                    Vii = Vb[np.ix_(cols, cols)]
                    ve[:, i] = np.maximum(
                        0.0, np.einsum("ij,jk,ik->i", Xi, Vii, Xi))
        else:
            eta = np.asarray(eta, dtype=float)
            se = False
        gamma = np.column_stack([eta[:, 0],
                                 self.links[1].linkinv(eta[:, 1])])
        if se:
            vp = np.column_stack([
                np.sqrt(ve[:, 0]),
                np.abs(self.links[1].mu_eta(eta[:, 1])) * np.sqrt(ve[:, 1]),
            ])
            return {"fit": gamma, "se_fit": vp}
        return {"fit": gamma}

    def __repr__(self):
        return (f"gumbls(link=('identity', {self._scale_link_name!r}), "
                f"b={self.b:g})")


class ShiftedLogitLink(Link):
    """mgcv's shifted-logit link for gevlss's shape LP (gamlss.r:
    1970-1978): ``g⁻¹(η) = 1.5·logistic(η) − 1`` confines ξ to
    (−1, 0.5) (Smith 1985 — the −1 bound is needed for MLE
    consistency). Display ``name`` is ``"logit"`` (mgcv's link string).
    The d2link..d4link are the plain logit-link η-derivatives of
    ``m = (ξ+1)/1.5`` scaled by ``1.5^{−k}``."""
    name = "logit"

    @staticmethod
    def _clamp_eta(eta):
        thr = -np.log(np.finfo(float).eps)
        return np.clip(np.asarray(eta, dtype=float), -thr, thr)

    def link(self, mu):
        return logit((np.asarray(mu, dtype=float) + 1.0) / 1.5)

    def linkinv(self, eta):
        e = self._clamp_eta(eta)
        return 1.5 * (np.exp(e) / (1.0 + np.exp(e))) - 1.0

    def mu_eta(self, eta):
        e = self._clamp_eta(eta)
        return np.exp(e) / (1.0 + np.exp(e)) ** 2 * 1.5

    def d2link(self, mu):
        m = (np.asarray(mu, dtype=float) + 1.0) / 1.5
        return (1.0 / (1.0 - m) ** 2 - 1.0 / m ** 2) / 1.5 ** 2

    def d3link(self, mu):
        m = (np.asarray(mu, dtype=float) + 1.0) / 1.5
        return (2.0 / (1.0 - m) ** 3 + 2.0 / m ** 3) / 1.5 ** 3

    def d4link(self, mu):
        m = (np.asarray(mu, dtype=float) + 1.0) / 1.5
        return (6.0 / (1.0 - m) ** 4 - 6.0 / m ** 4) / 1.5 ** 4


def _gevlss_derivs(y, mu, rho, xi, deriv):
    """gevlss log-density + packed parameter-space derivatives — mgcv's
    gevlss ``ll`` body (gamlss.r:2060-2278): the auto-generated /
    auto-simplified Maxima code transcribed VERBATIM (R ``^``→``**``,
    ``exp1^{kρ}``→``e^{kρ}``, dotted aux names underscored; the aa/bb/…
    auxiliaries kept to minimize transcription error). ``xi`` is
    pre-clamped away from 0 by the caller. Returns ``(l0, l1, l2, l3,
    l4)``; higher orders are ``None`` below the requested ``deriv``.
    """
    n = y.shape[0]
    ymu = y - mu
    aa0 = xi * ymu * np.exp(-rho)
    log_aa1 = np.log1p(aa0)
    aa1 = aa0 + 1.0
    aa2 = 1.0 / xi
    l0 = -aa2 * (1.0 + xi) * log_aa1 - aa1 ** (-aa2) - rho
    if deriv == 0:
        return l0, None, None, None, None

    er = np.exp(rho)
    l1 = np.zeros((n, 3))
    bb1 = 1.0 / er
    bb2 = bb1 * xi * ymu + 1.0
    l1[:, 0] = bb1 * (xi + 1.0) / bb2 - bb1 * bb2 ** ((-1.0 / xi) - 1.0)
    cc2 = ymu
    cc0 = bb1 * xi * cc2
    log_cc3 = np.log1p(cc0)
    cc3 = cc0 + 1.0
    l1[:, 1] = (-bb1 * cc2 * cc3 ** ((-1.0 / xi) - 1.0)
                + bb1 * (xi + 1.0) * cc2 / cc3 - 1.0)
    dd3 = xi + 1.0
    dd6 = 1.0 / cc3
    dd7 = log_cc3
    dd8 = 1.0 / xi ** 2
    l1[:, 2] = (-(dd8 * dd7 - bb1 * aa2 * cc2 * dd6) / cc3 ** aa2
                + dd8 * dd3 * dd7 - aa2 * dd7 - bb1 * aa2 * dd3 * cc2 * dd6)

    l2 = np.zeros((n, 6))
    ee1 = 1.0 / er ** 2
    ee3 = -1.0 / xi
    l2[:, 0] = (ee1 * (ee3 - 1.0) * xi * aa1 ** (ee3 - 2.0)
                + ee1 * xi * (xi + 1.0) / aa1 ** 2)
    ff7 = ee3 - 1.0
    l2[:, 1] = (bb1 * cc3 ** ff7 + ee1 * ff7 * xi * cc2 * cc3 ** (ee3 - 2.0)
                - bb1 * dd3 / cc3 + ee1 * xi * dd3 * cc2 / cc3 ** 2)
    gg7 = -aa2
    l2[:, 2] = (-bb1 * cc3 ** (gg7 - 1.0)
                * (log_cc3 / xi ** 2 - bb1 * aa2 * cc2 * dd6)
                + ee1 * cc2 * cc3 ** (gg7 - 2.0) + bb1 * dd6
                - ee1 * (xi + 1.0) * cc2 / cc3 ** 2)
    hh4 = cc2 ** 2
    l2[:, 3] = (bb1 * cc2 * cc3 ** ff7 + ee1 * ff7 * xi * hh4 * cc3 ** (ee3 - 2.0)
                - bb1 * dd3 * cc2 / cc3 + ee1 * xi * dd3 * hh4 / cc3 ** 2)
    l2[:, 4] = (-bb1 * cc2 * cc3 ** (gg7 - 1.0)
                * (log_cc3 / xi ** 2 - bb1 * aa2 * cc2 * dd6)
                + ee1 * hh4 * cc3 ** (gg7 - 2.0) + bb1 * cc2 * dd6
                - ee1 * (xi + 1.0) * hh4 / cc3 ** 2)
    jj08 = 1.0 / cc3 ** 2
    jj12 = 1.0 / xi ** 3
    jj13 = 1.0 / cc3 ** aa2
    l2[:, 5] = (-jj13 * (dd8 * dd7 - bb1 * aa2 * cc2 * dd6) ** 2
                - jj13 * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6
                          - 2.0 * jj12 * dd7)
                - 2.0 * jj12 * dd3 * dd7 + 2.0 * dd8 * dd7
                + 2.0 * bb1 * dd8 * dd3 * cc2 * dd6 - 2.0 * bb1 * aa2 * cc2 * dd6
                + ee1 * aa2 * dd3 * hh4 * jj08)

    l3 = l4 = None
    if deriv > 1:
        l3 = np.zeros((n, 10))
        kk1 = 1.0 / er ** 3
        kk2 = xi ** 2
        l3[:, 0] = (2.0 * kk1 * kk2 * (xi + 1.0) / aa1 ** 3
                    - kk1 * (ee3 - 2.0) * (ee3 - 1.0) * kk2 * aa1 ** (ee3 - 3.0))
        ll5 = xi * cc2 / er + 1.0
        ll8 = ee3 - 2.0
        l3[:, 1] = (-2.0 * ee1 * ff7 * xi * ll5 ** ll8
                    - kk1 * ll8 * ff7 * kk2 * cc2 * ll5 ** (ee3 - 3.0)
                    - 2.0 * ee1 * xi * dd3 / ll5 ** 2
                    + 2.0 * kk1 * kk2 * dd3 * cc2 / ll5 ** 3)
        mm10 = cc3 ** (gg7 - 3.0)
        mm11 = gg7 - 2.0
        mm12 = cc3 ** mm11
        l3[:, 2] = (ee1 * (gg7 - 1.0) * xi * mm12
                    * (log_cc3 / xi ** 2 - bb1 * aa2 * cc2 / cc3)
                    - ee1 * mm12 - kk1 * mm11 * xi * cc2 * mm10
                    + kk1 * cc2 * mm10 + ee1 * dd3 * jj08 + ee1 * xi * jj08
                    - 2.0 * kk1 * xi * dd3 * cc2 / cc3 ** 3)
        l3[:, 3] = (-bb1 * cc3 ** ff7 - 3.0 * ee1 * ff7 * xi * cc2 * cc3 ** ll8
                    - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0)
                    + bb1 * dd3 / cc3 - 3.0 * ee1 * xi * dd3 * cc2 / cc3 ** 2
                    + 2.0 * kk1 * kk2 * dd3 * hh4 / cc3 ** 3)
        oo10 = gg7 - 1.0
        oo13 = log_cc3 / xi ** 2
        l3[:, 4] = (bb1 * cc3 ** oo10 * (bb1 * oo10 * cc2 * dd6 + oo13)
                    + ee1 * oo10 * xi * cc2 * mm12
                    * (bb1 * mm11 * cc2 * dd6 + oo13)
                    + ee1 * aa2 * cc2 * mm12 + ee1 * oo10 * cc2 * mm12
                    - bb1 * dd6 + 2.0 * ee1 * dd3 * cc2 * jj08
                    + ee1 * xi * cc2 * jj08
                    - 2.0 * xi * dd3 * cc2 ** 2 / (er ** 3 * cc3 ** 3))
        pp07 = (-1.0 / xi) - 1.0
        pp08 = cc3 ** pp07
        l3[:, 5] = (-bb1 * pp08 * (bb1 * pp07 * cc2 * dd6 + dd8 * dd7) ** 2
                    - bb1 * pp08 * (-ee1 * pp07 * hh4 * jj08
                                    + 2.0 * bb1 * dd8 * cc2 * dd6
                                    - 2.0 * dd7 / xi ** 3)
                    - 2.0 * ee1 * cc2 * jj08
                    + 2.0 * (xi + 1.0) * hh4 / (er ** 3 * cc3 ** 3))
        qq05 = cc2 ** 3
        l3[:, 6] = (-bb1 * cc2 * cc3 ** ff7 - 3.0 * ee1 * ff7 * xi * hh4 * cc3 ** ll8
                    - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0)
                    + bb1 * dd3 * cc2 / cc3 - 3.0 * ee1 * xi * dd3 * hh4 / cc3 ** 2
                    + 2.0 * kk1 * kk2 * dd3 * qq05 / cc3 ** 3)
        rr17 = log_cc3 / xi ** 2 - bb1 * aa2 * cc2 * dd6
        l3[:, 7] = (bb1 * cc2 * cc3 ** oo10 * rr17
                    + ee1 * oo10 * xi * hh4 * mm12 * rr17 - 2.0 * ee1 * hh4 * mm12
                    - kk1 * mm11 * xi * qq05 * mm10 + kk1 * qq05 * mm10
                    - bb1 * cc2 * dd6 + 2.0 * ee1 * dd3 * hh4 * jj08
                    + ee1 * xi * hh4 * jj08 - 2.0 * kk1 * xi * dd3 * qq05 / cc3 ** 3)
        l3[:, 8] = (-bb1 * cc2 * pp08 * (bb1 * pp07 * cc2 * dd6 + dd8 * dd7) ** 2
                    - bb1 * cc2 * pp08 * (-ee1 * pp07 * hh4 * jj08
                                          + 2.0 * bb1 * dd8 * cc2 * dd6
                                          - 2.0 * dd7 / xi ** 3)
                    - 2.0 * ee1 * hh4 * jj08
                    + 2.0 * (xi + 1.0) * cc2 ** 3 / (er ** 3 * cc3 ** 3))
        tt08 = 1.0 / cc3 ** 3
        tt16 = 1.0 / xi ** 4
        tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6
        l3[:, 9] = (-jj13 * tt18 ** 3
                    - 3.0 * jj13 * (ee1 * aa2 * hh4 * jj08
                                    + 2.0 * bb1 * dd8 * cc2 * dd6
                                    - 2.0 * jj12 * dd7) * tt18
                    - jj13 * (-2.0 * kk1 * aa2 * qq05 * tt08
                              - 3.0 * ee1 * dd8 * hh4 * jj08
                              - 6.0 * bb1 * jj12 * cc2 * dd6 + 6.0 * tt16 * dd7)
                    + 6.0 * tt16 * dd3 * dd7 - 6.0 * jj12 * dd7
                    - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6
                    + 6.0 * bb1 * dd8 * cc2 * dd6
                    - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08
                    + 3.0 * ee1 * aa2 * hh4 * jj08
                    - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08)

    if deriv > 3:
        l4 = np.zeros((n, 15))
        uu1 = 1.0 / er ** 4
        uu2 = xi ** 3
        l4[:, 0] = (uu1 * (ee3 - 3.0) * (ee3 - 2.0) * (ee3 - 1.0) * uu2
                    * aa1 ** (ee3 - 4.0)
                    + 6.0 * uu1 * uu2 * (xi + 1.0) / aa1 ** 4)
        vv09 = ee3 - 3.0
        l4[:, 1] = (3.0 * kk1 * ll8 * ff7 * kk2 * ll5 ** vv09
                    + uu1 * vv09 * ll8 * ff7 * uu2 * cc2 * ll5 ** (ee3 - 4.0)
                    - 6.0 * kk1 * kk2 * dd3 / ll5 ** 3
                    + 6.0 * uu1 * uu2 * dd3 * cc2 / ll5 ** 4)
        ww11 = gg7 - 3.0
        ww12 = cc3 ** (gg7 - 4.0)
        ww15 = cc3 ** ww11
        l4[:, 2] = (-kk1 * mm11 * oo10 * kk2 * ww15
                    * (log_cc3 / kk2 - bb1 * aa2 * cc2 / cc3)
                    + 2.0 * kk1 * mm11 * xi * ww15 - kk1 * ww15
                    + uu1 * ww11 * mm11 * kk2 * cc2 * ww12
                    - uu1 * oo10 * xi * cc2 * ww12 - uu1 * ww11 * xi * cc2 * ww12
                    + 2.0 * kk1 * kk2 * tt08 + 4.0 * kk1 * xi * dd3 * tt08
                    - 6.0 * uu1 * kk2 * dd3 * cc2 / cc3 ** 4)
        l4[:, 3] = (4.0 * ee1 * ff7 * xi * ll5 ** ll8
                    + 5.0 * kk1 * ll8 * ff7 * kk2 * cc2 * ll5 ** vv09
                    + uu1 * vv09 * ll8 * ff7 * uu2 * hh4 * ll5 ** (ee3 - 4.0)
                    + 4.0 * ee1 * xi * dd3 / ll5 ** 2
                    - 10.0 * kk1 * kk2 * dd3 * cc2 / ll5 ** 3
                    + 6.0 * uu1 * uu2 * dd3 * hh4 / ll5 ** 4)
        yy18 = log_cc3 / kk2
        l4[:, 4] = (-2.0 * ee1 * oo10 * xi * mm12 * (bb1 * mm11 * cc2 * dd6 + yy18)
                    - kk1 * mm11 * oo10 * kk2 * cc2 * ww15
                    * (bb1 * ww11 * cc2 * dd6 + yy18)
                    - 2.0 * ee1 * aa2 * mm12 - 2.0 * ee1 * oo10 * mm12
                    - 2.0 * kk1 * mm11 * oo10 * xi * cc2 * ww15
                    - kk1 * oo10 * cc2 * ww15 - kk1 * mm11 * cc2 * ww15
                    - 2.0 * ee1 * dd3 * jj08 - 2.0 * ee1 * xi * jj08
                    + 2.0 * kk1 * kk2 * cc2 * tt08
                    + 8.0 * kk1 * xi * dd3 * cc2 * tt08
                    - 6.0 * kk2 * dd3 * cc2 ** 2 / (er ** 4 * cc3 ** 4))
        l4[:, 5] = (ee1 * oo10 * xi * mm12 * tt18 ** 2 - 2.0 * ee1 * mm12 * tt18
                    - 2.0 * kk1 * mm11 * xi * cc2 * ww15 * tt18
                    + 2.0 * kk1 * cc2 * ww15 * tt18
                    + ee1 * oo10 * xi * mm12 * (ee1 * aa2 * hh4 * jj08
                                                + 2.0 * bb1 * dd8 * cc2 * dd6
                                                - 2.0 * dd7 / xi ** 3)
                    + 4.0 * kk1 * cc2 * ww15 + 2.0 * uu1 * ww11 * xi * hh4 * ww12
                    - 4.0 * uu1 * hh4 * ww12 + 2.0 * ee1 * jj08
                    - 4.0 * kk1 * dd3 * cc2 * tt08 - 4.0 * kk1 * xi * cc2 * tt08
                    + 6.0 * uu1 * xi * dd3 * hh4 / cc3 ** 4)
        l4[:, 6] = (bb1 * cc3 ** ff7 + 7.0 * ee1 * ff7 * xi * cc2 * cc3 ** ll8
                    + 6.0 * kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** vv09
                    + uu1 * vv09 * ll8 * ff7 * uu2 * qq05 * cc3 ** (ee3 - 4.0)
                    - bb1 * dd3 / cc3 + 7.0 * ee1 * xi * dd3 * cc2 / cc3 ** 2
                    - 12.0 * kk1 * kk2 * dd3 * hh4 / cc3 ** 3
                    + 6.0 * uu1 * uu2 * dd3 * qq05 / cc3 ** 4)
        l4[:, 7] = (-bb1 * cc3 ** oo10 * (bb1 * oo10 * cc2 * dd6 + yy18)
                    - 3.0 * ee1 * oo10 * xi * cc2 * mm12
                    * (bb1 * mm11 * cc2 * dd6 + yy18)
                    - kk1 * mm11 * oo10 * kk2 * hh4 * ww15
                    * (bb1 * ww11 * cc2 * dd6 + yy18)
                    - 3.0 * ee1 * aa2 * cc2 * mm12 - 3.0 * ee1 * oo10 * cc2 * mm12
                    - 2.0 * kk1 * mm11 * oo10 * xi * hh4 * ww15
                    - kk1 * oo10 * hh4 * ww15 - kk1 * mm11 * hh4 * ww15
                    + bb1 * dd6 - 4.0 * ee1 * dd3 * cc2 * jj08
                    - 3.0 * ee1 * xi * cc2 * jj08 + 2.0 * kk1 * kk2 * hh4 * tt08
                    + 10.0 * kk1 * xi * dd3 * hh4 * tt08
                    - 6.0 * kk2 * dd3 * cc2 ** 3 / (er ** 4 * cc3 ** 4))
        ad17 = 2.0 * bb1 * dd8 * cc2 * dd6
        ad19 = -2.0 * dd7 / xi ** 3
        ad20 = cc3 ** oo10
        ad21 = dd8 * dd7
        ad22 = ad21 + bb1 * mm11 * cc2 * dd6
        l4[:, 8] = (bb1 * ad20 * (bb1 * oo10 * cc2 * dd6 + ad21) ** 2
                    + ee1 * oo10 * xi * cc2 * mm12 * ad22 ** 2
                    + 2.0 * ee1 * aa2 * cc2 * mm12 * ad22
                    + 2.0 * ee1 * oo10 * cc2 * mm12 * ad22
                    + bb1 * ad20 * (-ee1 * oo10 * hh4 * jj08 + ad17 + ad19)
                    + ee1 * oo10 * xi * cc2 * mm12
                    * (-ee1 * mm11 * hh4 * jj08 + ad17 + ad19)
                    + 4.0 * ee1 * cc2 * jj08 - 6.0 * kk1 * dd3 * hh4 * tt08
                    - 4.0 * kk1 * xi * hh4 * tt08
                    + 6.0 * xi * dd3 * cc2 ** 3 / (er ** 4 * cc3 ** 4))
        ae16 = dd8 * dd7 + bb1 * pp07 * cc2 * dd6
        l4[:, 9] = (-bb1 * pp08 * ae16 ** 3
                    - 3.0 * bb1 * pp08 * (-ee1 * pp07 * hh4 * jj08
                                          + 2.0 * bb1 * dd8 * cc2 * dd6
                                          - 2.0 * jj12 * dd7) * ae16
                    - bb1 * pp08 * (2.0 * kk1 * pp07 * qq05 * tt08
                                    - 3.0 * ee1 * dd8 * hh4 * jj08
                                    - 6.0 * bb1 * jj12 * cc2 * dd6
                                    + 6.0 * dd7 / xi ** 4)
                    + 6.0 * kk1 * hh4 * tt08
                    - 6.0 * (xi + 1.0) * qq05 / (er ** 4 * cc3 ** 4))
        af05 = cc2 ** 4
        l4[:, 10] = (bb1 * cc2 * cc3 ** ff7 + 7.0 * ee1 * ff7 * xi * hh4 * cc3 ** ll8
                     + 6.0 * kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** vv09
                     + uu1 * vv09 * ll8 * ff7 * uu2 * af05 * cc3 ** (ee3 - 4.0)
                     - bb1 * dd3 * cc2 / cc3 + 7.0 * ee1 * xi * dd3 * hh4 / cc3 ** 2
                     - 12.0 * kk1 * kk2 * dd3 * qq05 / cc3 ** 3
                     + 6.0 * uu1 * uu2 * dd3 * af05 / cc3 ** 4)
        ag23 = log_cc3 / kk2 - bb1 * aa2 * cc2 * dd6
        l4[:, 11] = (-bb1 * cc2 * cc3 ** oo10 * ag23
                     - 3.0 * ee1 * oo10 * xi * hh4 * mm12 * ag23
                     - kk1 * mm11 * oo10 * kk2 * qq05 * ww15 * ag23
                     + 4.0 * ee1 * hh4 * mm12 + 5.0 * kk1 * mm11 * xi * qq05 * ww15
                     - 4.0 * kk1 * qq05 * ww15
                     + uu1 * ww11 * mm11 * kk2 * af05 * ww12
                     - uu1 * oo10 * xi * af05 * ww12 - uu1 * ww11 * xi * af05 * ww12
                     + bb1 * cc2 * dd6 - 4.0 * ee1 * dd3 * hh4 * jj08
                     - 3.0 * ee1 * xi * hh4 * jj08 + 2.0 * kk1 * kk2 * qq05 * tt08
                     + 10.0 * kk1 * xi * dd3 * qq05 * tt08
                     - 6.0 * uu1 * kk2 * dd3 * af05 / cc3 ** 4)
        ah24 = (-2.0 * dd7 / xi ** 3 + 2.0 * bb1 * dd8 * cc2 * dd6
                + ee1 * aa2 * hh4 * jj08)
        ah27 = tt18 ** 2
        l4[:, 12] = (bb1 * cc2 * ad20 * ah27 + ee1 * oo10 * xi * hh4 * mm12 * ah27
                     - 4.0 * ee1 * hh4 * mm12 * tt18
                     - 2.0 * kk1 * mm11 * xi * qq05 * ww15 * tt18
                     + 2.0 * kk1 * qq05 * ww15 * tt18 + bb1 * cc2 * ad20 * ah24
                     + ee1 * oo10 * xi * hh4 * mm12 * ah24 + 6.0 * kk1 * qq05 * ww15
                     + 2.0 * uu1 * ww11 * xi * af05 * ww12 - 4.0 * uu1 * af05 * ww12
                     + 4.0 * ee1 * hh4 * jj08 - 6.0 * kk1 * dd3 * qq05 * tt08
                     - 4.0 * kk1 * xi * qq05 * tt08
                     + 6.0 * uu1 * xi * dd3 * af05 / cc3 ** 4)
        l4[:, 13] = (-bb1 * cc2 * pp08 * ae16 ** 3
                     - 3.0 * bb1 * cc2 * pp08 * (-ee1 * pp07 * hh4 * jj08
                                                 + 2.0 * bb1 * dd8 * cc2 * dd6
                                                 - 2.0 * jj12 * dd7) * ae16
                     - bb1 * cc2 * pp08 * (2.0 * kk1 * pp07 * qq05 * tt08
                                           - 3.0 * ee1 * dd8 * hh4 * jj08
                                           - 6.0 * bb1 * jj12 * cc2 * dd6
                                           + 6.0 * dd7 / xi ** 4)
                     + 6.0 * kk1 * qq05 * tt08
                     - 6.0 * (xi + 1.0) * cc2 ** 4 / (er ** 4 * cc3 ** 4))
        aj08 = 1.0 / cc3 ** 4
        aj20 = 1.0 / xi ** 5
        aj23 = (-2.0 * jj12 * dd7 + 2.0 * bb1 * dd8 * cc2 * dd6
                + ee1 * aa2 * hh4 * jj08)
        l4[:, 14] = (-jj13 * tt18 ** 4 - 6.0 * jj13 * aj23 * tt18 ** 2
                     - 3.0 * jj13 * aj23 ** 2
                     - 4.0 * jj13 * (-2.0 * kk1 * aa2 * qq05 * tt08
                                     - 3.0 * ee1 * dd8 * hh4 * jj08
                                     - 6.0 * bb1 * jj12 * cc2 * dd6
                                     + 6.0 * tt16 * dd7) * tt18
                     - jj13 * (6.0 * uu1 * aa2 * af05 * aj08
                               + 8.0 * kk1 * dd8 * qq05 * tt08
                               + 12.0 * ee1 * jj12 * hh4 * jj08
                               + 24.0 * bb1 * tt16 * cc2 * dd6 - 24.0 * aj20 * dd7)
                     - 24.0 * aj20 * dd3 * dd7 + 24.0 * tt16 * dd7
                     + 24.0 * bb1 * tt16 * dd3 * cc2 * dd6
                     - 24.0 * bb1 * jj12 * cc2 * dd6
                     + 12.0 * ee1 * jj12 * dd3 * hh4 * jj08
                     - 12.0 * ee1 * dd8 * hh4 * jj08
                     + 8.0 * kk1 * dd8 * dd3 * qq05 * tt08
                     - 8.0 * kk1 * aa2 * qq05 * tt08
                     + 6.0 * uu1 * aa2 * dd3 * af05 * aj08)

    return l0, l1, l2, l3, l4


class gevlss(GeneralFamily):
    """Generalized extreme value location-scale-shape general family —
    mgcv ``gevlss()`` (gamlss.r:1945-2446). Three linear predictors:
    LP1 the location μ (identity/log), LP2 ``ρ = log σ`` (identity),
    LP3 the shape ξ through the :class:`ShiftedLogitLink` (ξ∈(−1,.5))
    or identity. The GEV has **parameter-dependent support** — the
    log-likelihood is −∞ wherever ``1 + ξ(y−μ)/σ ≤ 0`` — so this family
    is hea's fixture for gam.fit5's robustness protocols (non-finite-ll
    step rejection, step-halving → steepest-ascent, saddle
    perturbation). No ``predict`` hook (response = per-LP linkinv);
    null deviance is NA.
    """
    name = "gevlss"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 3
    available_derivs = 2
    discrete_ok = True          # gamlss.r:2444

    def __init__(self, link: tuple[str, str, str]
                 = ("identity", "identity", "logit")):
        if len(link) != 3:
            raise ValueError("gevlss requires 3 links")
        ok = [("log", "identity"), ("identity",), ("identity", "logit")]
        names = ("location", "log-scale", "shape")
        for i in range(3):
            if link[i] not in ok[i]:
                raise ValueError(
                    f'link "{link[i]}" not available for the {names[i]} '
                    f"parameter of gevlss; available links are {ok[i]}")
        loc = {"identity": IdentityLink, "log": LogLink}[link[0]]()
        scale = IdentityLink()
        shape = (ShiftedLogitLink() if link[2] == "logit"
                 else IdentityLink())
        self._link_names = tuple(link)
        self.tri = trind_generator(3)
        super().__init__([loc, scale, shape])

    @staticmethod
    def _clamp_xi(xi):
        eps = 1e-7
        xi = np.asarray(xi, dtype=float).copy()
        xi[(xi >= 0) & (xi < eps)] = eps
        xi[(xi < 0) & (xi > -eps)] = -eps
        return xi

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            # gamlss.r:2026-2030: per-LP η off the compressed design.
            _Xbd, _, _ = _discrete_kernels()
            eta = _Xbd(X.design, coef, lt=X.lpid[0])
            etar = _Xbd(X.design, coef, lt=X.lpid[1])
            etax = _Xbd(X.design, coef, lt=X.lpid[2])
        else:
            X = np.asarray(X, dtype=float)
            eta = X[:, jj[0]] @ coef[jj[0]]
            etar = X[:, jj[1]] @ coef[jj[1]]
            etax = X[:, jj[2]] @ coef[jj[2]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                etar = etar + offset[1]
            if len(offset) > 2 and offset[2] is not None:
                etax = etax + offset[2]
        mu = self.links[0].linkinv(eta)
        rho = self.links[1].linkinv(etar)
        xi = self._clamp_xi(self.links[2].linkinv(etax))

        # The GEV support is parameter-dependent: 1 + ξ(y−μ)/σ ≤ 0 makes
        # the log-density −∞ (NaN from log1p/power). That is the EXPECTED
        # signal gam.fit5 step-rejects on — mgcv warns "NaNs produced"
        # there too — so the invalid-value warnings are silenced; the
        # non-finite values still propagate to the fitter.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            l0, l1, l2, l3, l4 = _gevlss_derivs(y, mu, rho, xi, deriv)
        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        ret: dict = {"l": float(np.sum(wt * l0)), "l0": l0}
        if deriv == 0:
            return ret

        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(etar),
                               self.links[2].mu_eta(etax)])
        g2 = np.column_stack([self.links[0].d2link(mu),
                              self.links[1].d2link(rho),
                              self.links[2].d2link(xi)])
        g3 = g4 = None
        if deriv > 1:
            g3 = np.column_stack([self.links[0].d3link(mu),
                                  self.links[1].d3link(rho),
                                  self.links[2].d3link(xi)])
        if deriv > 3:
            g4 = np.column_stack([self.links[0].d4link(mu),
                                  self.links[1].d4link(rho),
                                  self.links[2].d4link(xi)])

        tri = self.tri
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            l1, l2, l3, l4 = self._apply_prior_weights(wt, l1, l2, l3, l4)
            de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                              tri["i2"], tri["i3"], tri["i4"], deriv - 1)
            gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                           l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                           i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                           fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gevlss ``initialize`` (gamlss.r:2301-2425): regress g₁(y) on
        LP1, log|residuals| on LP2, then seed ξ near 0 (``g₃(1e-3)``)
        and run mgcv's crude ll line-search over a scaling ``m`` of the
        ξ-start to escape the non-finite regime. A :class:`DiscreteX`
        design takes the discrete branch (:2318-2377): the per-LP
        ``mchol`` solves, the line-search re-solving LP3's factor on
        ``Xty·m`` and evaluating ``ll`` through the compressed design
        (unlike the dense branch's plain-LS ξ seed, the discrete LP3
        solve is E-regularized — mgcv's own asymmetry, :2350)."""
        y = np.asarray(y, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            return self._initialize_coef_discrete(y, X, jj, E, offset)
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        if self._link_names[0] == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)
        start[jj[0]] = _reg(jj[0], yt1)
        lres1 = np.log(np.abs(y - self.links[0].linkinv(X[:, jj[0]]
                                                        @ start[jj[0]])))
        start[jj[1]] = _reg(jj[1], lres1)

        # LP3: plain LS of the constant g₃(1e-3) target, then line-search.
        x1 = X[:, jj[2]]
        yt3 = np.full(x1.shape[0], self.links[2].link(np.array(1e-3)))

        def fob(m=1.0):
            bji, *_ = np.linalg.lstsq(x1, yt3 * m, rcond=None)
            bji[~np.isfinite(bji)] = 0.0
            st = start.copy()
            st[jj[2]] = bji
            return self.ll(y, X, st, lpi=lpi, offset=offset)["l"], st

        f0l, f0s = fob()
        dm = 0.2
        mm = 1.0
        up = False
        f1l = f0l
        while -4.2 < mm < 4.2:
            f1l, f1s = fob(mm + dm)
            if np.isfinite(f1l) and f1l > f0l:
                up = True
                f0l, f0s = f1l, f1s
                mm = mm + dm
            elif up:
                break
            elif dm > 0:
                dm = -dm
            else:
                break
        if not np.isfinite(f1l):
            f1l, f1s = fob(mm - dm)
            if np.isfinite(f1l):
                f0l, f0s = f1l, f1s
        return f0s

    def _initialize_coef_discrete(self, y, X: DiscreteX, jj, E,
                                  offset) -> np.ndarray:
        """gevlss ``initialize``'s discrete branch (gamlss.r:2318-2377):
        ``mchol`` solves for LP1 (g₁(y) target) and LP2
        (log|residuals| via ``Xbd``), then the ξ line-search re-solves
        LP3's factor on the once-assembled ``Xty`` scaled by ``m``
        (:2358-2363), scoring ``ll`` through the compressed design."""
        _Xbd, _, _ = _discrete_kernels()
        design = X.design
        lpid = X.lpid
        p = design.p
        if E is None:
            E = np.zeros((0, p))
        E = np.asarray(E, dtype=float)
        ones_n = np.ones(y.shape[0])

        def _guarded(lp, target):
            startji = lp.solve_target(target)
            startji[~np.isfinite(startji)] = 0.0
            return startji

        start = np.zeros(p)
        if self._link_names[0] == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.links[0].link(np.abs(y) + float(np.max(y)) * 1e-7)
        start[jj[0]] = _guarded(
            _DiscreteLPSolve(design, lpid[0], E[:, jj[0]], ones_n), yt1)
        lres1 = _Xbd(design, start, lt=lpid[0])
        lres1 = np.log(np.abs(y - self.links[0].linkinv(lres1)))
        start[jj[1]] = _guarded(
            _DiscreteLPSolve(design, lpid[1], E[:, jj[1]], ones_n), lres1)

        # LP3 (:2348-2363): constant g₃(1e-3) target, factor kept for
        # the line-search's re-solves on Xty·m.
        yt3 = np.full(y.shape[0], self.links[2].link(np.array(1e-3)))
        lp3 = _DiscreteLPSolve(design, lpid[2], E[:, jj[2]], ones_n)
        Xty3 = lp3.xty(yt3)

        def fob(m=1.0):
            bji = lp3.solve(Xty3 * m)
            bji[~np.isfinite(bji)] = 0.0
            st = start.copy()
            st[jj[2]] = bji
            return self.ll(y, X, st, lpi=jj, offset=offset)["l"], st

        f0l, f0s = fob()
        dm = 0.2
        mm = 1.0
        up = False
        f1l = f0l
        while -4.2 < mm < 4.2:
            f1l, f1s = fob(mm + dm)
            if np.isfinite(f1l) and f1l > f0l:
                up = True
                f0l, f0s = f1l, f1s
                mm = mm + dm
            elif up:
                break
            elif dm > 0:
                dm = -dm
            else:
                break
        if not np.isfinite(f1l):
            f1l, f1s = fob(mm - dm)
            if np.isfinite(f1l):
                f0l, f0s = f1l, f1s
        return f0s

    def rd(self, rng, mu, wt, scale):
        """gevlss rd (gamlss.r:2427-2434): GEV inverse-CDF draws
        ``μ + ((−log U)^{−ξ} − 1)·σ/ξ`` with ``σ = e^{fitted[:,1]}``,
        ``ξ`` clamped away from 0."""
        mu = np.asarray(mu, dtype=float)
        z = rng.uniform(size=mu.shape[0])
        loc = mu[:, 0]
        sigma = np.exp(mu[:, 1])
        xi = mu[:, 2].copy()
        xi[np.abs(xi) < 1e-8] = 1e-8
        return loc + ((-np.log(z)) ** (-xi) - 1.0) * sigma / xi

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """gevlss residuals (gamlss.r:1981-2000). ``fitted`` is the
        (n, 3) matrix (μ, ρ=log σ, ξ); the GEV mean is
        ``fv = μ + e^ρ·(Γ(1−ξ)−1)/ξ``."""
        if type not in ("deviance", "pearson", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'pearson', 'response' "
                f"for gevlss residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu = fitted[:, 0]
        rho = fitted[:, 1]
        xi = fitted[:, 2].copy()
        fv = mu + np.exp(rho) * (_gamma_fn(1.0 - xi) - 1.0) / xi
        eps = 1e-7
        xi[(xi >= 0) & (xi < eps)] = eps
        xi[(xi < 0) & (xi > -eps)] = -eps
        if type == "response":
            return y - fv
        if type == "pearson":
            sd = np.exp(rho) / xi * np.sqrt(np.maximum(
                0.0, _gamma_fn(1.0 - 2.0 * xi) - _gamma_fn(1.0 - xi) ** 2))
            return (y - fv) / sd
        ymr = (y - mu) * np.exp(-rho) * xi
        rsd = ((xi + 1.0) / xi * np.log(1.0 + ymr) + (1.0 + ymr) ** (-1.0 / xi)
               + (1.0 + xi) * np.log(1.0 + xi) - (1.0 + xi))
        return np.sqrt(np.maximum(0.0, rsd)) * np.sign(y - fv)

    def __repr__(self):
        return f"gevlss(link={self._link_names!r})"


_COX_SORT_CACHE: dict = {}


def _cox_sort(X, d, time, n, p):
    """Descending-time sort structure for ``_coxlpl`` — ``(order, r, nt, X, d)``
    with ``r`` the 0-based unique-time group per (sorted) row, ``X``/``d`` the
    sorted, contiguous design + event indicator. Depends only on time/X/d, which
    are fixed across a fit's many ll calls, so it is memoized (keyed on a content
    fingerprint, bounded). Mirrors the in-function sort the C caller does once."""
    key = (n, p, float(time[0]), float(time[-1]), float(time.sum()),
           float(X[0, 0]), float(X[-1, -1]), int(np.asarray(d).sum()))
    hit = _COX_SORT_CACHE.get(key)
    if hit is not None:
        return hit
    order = np.argsort(-time, kind="stable")
    ts = -time[order]                                     # ascending
    tr = np.unique(ts)
    nt = tr.size
    r = np.ascontiguousarray(np.searchsorted(tr, ts).astype(np.int64))
    Xc = np.ascontiguousarray(X[order])
    dc = np.ascontiguousarray(np.asarray(d, np.int64)[order])
    val = (order, r, nt, Xc, dc)
    if len(_COX_SORT_CACHE) > 16:
        _COX_SORT_CACHE.clear()
    _COX_SORT_CACHE[key] = val
    return val


def _coxlpl(eta, X, d, time, deriv, d1b=None, d2b=None, D=None,
            eigen=None):
    """mgcv's ``coxlpl`` C kernel (src/coxph.c:141-394): the Cox log
    PARTIAL likelihood (Peto/Breslow tie handling) and its coefficient-
    and smoothing-parameter derivatives, in pure numpy.

    ``deriv`` is the C-level code (= the engine's ll deriv minus one):
      0 → ``l``, ``lb`` (= g), ``lbb`` (= H);
      1,2 → + ``d1H`` the per-ρ ∂H/∂ρ matrices (original basis);
      3 → + ``trHid2H`` = tr(Hp⁻¹ ∂²H/∂ρ∂ρ'), computed in the
          eigenbasis of the (raw) penalized Hessian — ``eigen`` is
          ``{"values","vectors"}`` and ``D`` the preconditioner, exactly
          mgcv's ``X<-X%*%(ev$vectors*D); d1b<-t(V)%*%(d1b/D)`` step.

    Rows need NOT be pre-sorted: risk sets are formed by descending
    ``time`` internally, and every return is coefficient-space (so it is
    invariant to the row storage order)."""
    eta = np.asarray(eta, float)
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    d = np.asarray(d, int)
    n, p = X.shape
    M = 0 if d1b is None else d1b.shape[1]
    # risk sets are cumulative in DESCENDING time; sort rows internally so the
    # engine may pass them in any order (l/lb/lbb/d1H/trHid2H are all
    # coefficient-space, hence invariant to the row permutation). The sort
    # structure is cached (fixed across a fit); only eta changes per call.
    # d1b/d2b are coefficient-space and stay put.
    order, r, nt, X, d = _cox_sort(X, d, time, n, p)
    eta = np.ascontiguousarray(eta[order])
    if _rs_cox_l is not None and deriv < 0:    # the many line-search l evals
        return {"l": float(_rs_cox_l(eta, d, r, int(nt)))}
    if _rs_cox_lpl0 is not None and 0 <= deriv <= 3:
        # the per-iteration l/lb/lbb (+d1H/+trHid2H) evaluations: the C
        # single-pass risk-set sweep, no (n,p,p[,M]) gXX/d1A_p temporaries
        # (coxph.c:266-368). X/d/r are pre-sorted contiguous from the cache;
        # d1b/d2b/eigen stay coefficient-/eigen-space.
        lpl, g, H = _rs_cox_lpl0(eta, X, d, r, int(nt))
        if deriv == 0:
            return {"l": float(lpl), "lb": g, "lbb": H}
        gamma = np.exp(eta)
        if deriv <= 2:                    # + ∂H/∂ρ matrices (original basis)
            d1gamma = np.ascontiguousarray((X @ d1b) * gamma[:, None])
            lpl, g, H, d1H = _rs_cox_lpl_d1(eta, X, d, r, int(nt), d1gamma)
            return {"l": float(lpl), "lb": g, "lbb": H, "d1H": d1H}
        # deriv == 3: + trHid2H in the eigenbasis (reuse original-basis l/g/H);
        # the eigenbasis transform is cheap BLAS, the sweep is rust (cox_d2h).
        val = np.asarray(eigen["values"], float)
        vec = np.asarray(eigen["vectors"], float)
        dvec = np.where(val > 0, 1.0 / np.where(val > 0, val, 1.0), 0.0)
        Xp = X @ (D[:, None] * vec)
        d1bp = vec.T @ (d1b / D[:, None])
        d2bp = vec.T @ (d2b / D[:, None])
        d1eta = Xp @ d1bp
        d1gamma = d1eta * gamma[:, None]
        nhh = M * (M + 1) // 2
        pairs = [(a, b) for a in range(M) for b in range(a, M)]
        d2eta = Xp @ d2bp
        d2gamma = np.empty((n, nhh))
        for off, (a, b) in enumerate(pairs):
            d2gamma[:, off] = gamma * (d2eta[:, off] + d1eta[:, a] * d1eta[:, b])
        d2H = _rs_cox_d2h(np.ascontiguousarray(Xp), d, r, int(nt), eta,
                          np.ascontiguousarray(d1gamma),
                          np.ascontiguousarray(d2gamma))
        trHid2H = np.sum(np.asarray(d2H) * dvec[:, None], 0)
        return {"l": float(lpl), "lb": g, "lbb": H, "trHid2H": trHid2H}
    gamma = np.exp(eta)
    last = np.searchsorted(r, np.arange(nt), side="right") - 1
    gamma_p = np.cumsum(gamma)[last]
    ev = np.asarray(d, int) == 1
    dr = np.zeros(nt)
    np.add.at(dr, r[ev], 1.0)
    eta_sum = np.zeros(nt)
    np.add.at(eta_sum, r[ev], eta[ev])

    lpl = float(np.sum(eta_sum - dr * np.log(gamma_p)))
    if deriv < 0:                  # C coxlpl: deriv<0 returns lp only, no b_p/A_p/g/H
        return {"l": lpl}
    b_p = np.cumsum(gamma[:, None] * X, 0)[last]
    gXX = gamma[:, None, None] * X[:, :, None] * X[:, None, :]
    A_p = np.cumsum(gXX, 0)[last]
    g_ev = X[ev].sum(0) if ev.any() else np.zeros(p)
    g = g_ev - np.sum((dr / gamma_p)[:, None] * b_p, 0)
    H = -np.sum(dr[:, None, None] * (
        A_p / gamma_p[:, None, None]
        - b_p[:, :, None] * b_p[:, None, :]
        / (gamma_p ** 2)[:, None, None]), 0)
    out = {"l": lpl, "lb": g, "lbb": H}
    if deriv < 1:
        return out

    if deriv <= 2:                          # ∂H/∂ρ matrices, original basis
        d1eta = X @ d1b
        d1gamma = d1eta * gamma[:, None]
        d1b_p = np.cumsum(d1gamma[:, None, :] * X[:, :, None], 0)[last]
        d1gamma_p = np.cumsum(d1gamma, 0)[last]
        d1A_p = np.cumsum(
            d1gamma[:, None, None, :] * gXX[..., None]
            / gamma[:, None, None, None], 0)[last]
        xx0 = dr / gamma_p
        xx1 = xx0 / gamma_p
        d1H = np.zeros((p, p, M))
        for m in range(M):
            xx = d1gamma_p[:, m] * xx0 / gamma_p
            xx2 = xx1 * 2 * d1gamma_p[:, m] / gamma_p
            term = (xx1[:, None, None]
                    * (d1b_p[:, :, None, m] * b_p[:, None, :]
                       + b_p[:, :, None] * d1b_p[:, None, :, m])
                    - xx2[:, None, None] * b_p[:, :, None] * b_p[:, None, :]
                    + xx[:, None, None] * A_p
                    - xx0[:, None, None] * d1A_p[:, :, :, m])
            d1H[:, :, m] = term.sum(0)
        out["d1H"] = d1H
        return out

    # deriv == 3: trHid2H in the eigenbasis (recompute risk sums there)
    val = np.asarray(eigen["values"], float)
    vec = np.asarray(eigen["vectors"], float)
    dvec = np.where(val > 0, 1.0 / np.where(val > 0, val, 1.0), 0.0)
    Xp = X @ (D[:, None] * vec)
    d1bp = vec.T @ (d1b / D[:, None])
    d2bp = vec.T @ (d2b / D[:, None])
    gXXp = gamma[:, None, None] * Xp[:, :, None] * Xp[:, None, :]
    b_p = np.cumsum(gamma[:, None] * Xp, 0)[last]
    A_p = np.cumsum(gXXp, 0)[last]
    Adiag = np.diagonal(A_p, axis1=1, axis2=2)
    d1eta = Xp @ d1bp
    d1gamma = d1eta * gamma[:, None]
    d1gamma_p = np.cumsum(d1gamma, 0)[last]
    d1b_p = np.cumsum(d1gamma[:, None, :] * Xp[:, :, None], 0)[last]
    d1A_p = np.cumsum(d1gamma[:, None, None, :] * gXXp[..., None]
                      / gamma[:, None, None, None], 0)[last]
    d1Adiag = np.diagonal(d1A_p, axis1=1, axis2=2)        # (nt,M,p)
    nhh = M * (M + 1) // 2
    pairs = [(a, b) for a in range(M) for b in range(a, M)]
    d2eta = Xp @ d2bp
    d2gamma = np.empty((n, nhh))
    for off, (a, b) in enumerate(pairs):
        d2gamma[:, off] = gamma * (d2eta[:, off] + d1eta[:, a] * d1eta[:, b])
    d2gamma_p = np.cumsum(d2gamma, 0)[last]
    d2b_p = np.cumsum(d2gamma[:, None, :] * Xp[:, :, None], 0)[last]
    d2ldA_p = np.cumsum(d2gamma[:, None, :] * (Xp ** 2)[:, :, None], 0)[last]
    xx = dr / gamma_p
    xx0 = xx / gamma_p
    xx1 = xx0 / gamma_p
    xx2 = xx1 / gamma_p
    d2H = np.zeros((p, nhh))
    for off, (m, k) in enumerate(pairs):
        xx3 = -2 * xx1 * d1gamma_p[:, m]
        contrib = (
            xx3[:, None] * (Adiag * d1gamma_p[:, k][:, None]
                            + 2 * d1b_p[:, :, k] * b_p)
            + xx0[:, None] * (d1Adiag[:, m, :] * d1gamma_p[:, k][:, None]
                              + Adiag * d2gamma_p[:, off][:, None]
                              + d2b_p[:, :, off] * b_p
                              + 2 * d1b_p[:, :, k] * d1b_p[:, :, m]
                              + b_p * d2b_p[:, :, off])
            + xx0[:, None] * d1gamma_p[:, m][:, None] * d1Adiag[:, k, :]
            - xx[:, None] * d2ldA_p[:, :, off]
            + 6 * xx2[:, None] * d1gamma_p[:, m][:, None] * b_p * b_p
            * d1gamma_p[:, k][:, None]
            - 2 * xx1[:, None] * (2 * d1b_p[:, :, m] * b_p
                                  * d1gamma_p[:, k][:, None]
                                  + b_p * b_p * d2gamma_p[:, off][:, None]))
        d2H[:, off] = contrib.sum(0)
    out["trHid2H"] = np.sum(d2H * dvec[:, None], 0)
    return out


def _coxpp(eta, X, d, time):
    """mgcv's ``coxpp`` C kernel (src/coxph.c:61-137): baseline cumulative
    hazard ``h``, its variance ``q``, the Kaplan-Meier hazard ``km`` and
    the ``a`` vectors used for survival-curve standard errors. Reduces to
    reverse-cumulative sums of the same risk-set quantities ``_coxlpl``
    forms. Sorts internally; ``r`` is the 0-based unique-time index per
    (original-order) row."""
    eta = np.asarray(eta, float)
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    d = np.asarray(d, int)
    n, p = X.shape
    order = np.argsort(-time, kind="stable")     # descending time
    eta = eta[order]
    X = X[order]
    time = time[order]
    d = d[order]
    gamma = np.exp(eta)
    tr_neg = np.unique(-time)
    nt = tr_neg.size
    tr = -tr_neg
    r_sorted = np.searchsorted(tr_neg, -time)
    last = np.searchsorted(r_sorted, np.arange(nt), side="right") - 1
    r = np.empty(n, int)
    r[order] = r_sorted                          # back to original row order
    gamma_p = np.cumsum(gamma)[last]
    gamma_np = (last + 1).astype(float)
    b_p = (np.cumsum(gamma[:, None] * X, 0)[last] if p
           else np.zeros((nt, 0)))
    dr = np.zeros(nt)
    np.add.at(dr, r_sorted[d == 1], 1.0)         # sorted-order events

    def _rev(a):
        return np.cumsum(a[::-1], 0)[::-1]

    h = _rev(dr / gamma_p)
    km = _rev(dr / gamma_np)
    q = _rev(dr / gamma_p ** 2)
    a = (_rev(b_p * (dr / gamma_p ** 2)[:, None]) if p
         else np.zeros((nt, 0)))
    return {"tr": tr, "h": h, "q": q, "km": km, "nt": nt, "r": r, "a": a}


def _coxpred(Xnew, tnew, beta, off, Vb, a, h, q, tr):
    """mgcv's ``coxpred`` C kernel (src/coxph.c:20-59): predicted survivor
    function and its s.e. for new ``(Xnew, tnew)`` given the fitted
    baseline-hazard pieces (``a``, ``h``, ``q``, unique times ``tr``).
    New data must arrive in DESCENDING time order (the interval pointer
    advances monotonically); the caller sorts and unsorts."""
    Xnew = np.asarray(Xnew, float)
    n = Xnew.shape[0]
    nt = tr.size
    s = np.zeros(n)
    se = np.zeros(n)
    ir = 0
    for i in range(n):
        while ir < nt and tnew[i] < tr[ir]:
            ir += 1
        if ir == nt:                # earlier than every fit time
            s[i] = 1.0
            se[i] = 0.0
            continue
        hi = h[ir]
        eta = float(Xnew[i] @ beta)
        v = a[ir] - Xnew[i] * hi
        exp_eta = np.exp(eta + off[i])
        s[i] = np.exp(-hi * exp_eta)
        vVv = float(v @ Vb @ v)
        se[i] = exp_eta * s[i] * np.sqrt(max(q[ir] + vVv, 0.0))
    return s, se


class cox_ph(GeneralFamily):
    """Cox proportional-hazards general family — mgcv ``cox.ph()``
    (coxph.r). A SINGLE linear predictor (``n_lp == 1``), the intercept
    dropped (the baseline hazard absorbs it), fit by partial likelihood
    over risk sets. The response is the event time; the prior
    ``weights`` are the censoring indicator (1 = event, 0 = censored).

    Unlike the gamlss families the likelihood is supplied directly by
    :func:`_coxlpl` (its own gradient/Hessian and smoothing-parameter
    derivatives), not through :func:`gamlss_gH`. ``ll`` reconstructs the
    raw penalized Hessian from gam.fit5's preconditioned Cholesky pieces
    (``fh``/``D``) at deriv 4 to match mgcv's own eigen-decomposition of
    ``Hp``.
    """
    name = "Cox PH"
    scale_known = True
    n_theta = 0
    n_lp = 1
    available_derivs = 2
    drop_intercept = True

    def __init__(self, link="identity"):
        if link != "identity":
            raise ValueError(
                f"{link!r} link not available for cox.ph family; the only "
                "available link is 'identity'")
        super().__init__([IdentityLink()])
        self._fit_ctx = None
        self._cox_data = None

    def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, float)
        X = np.asarray(X, float)
        coef = np.asarray(coef, float)
        d = np.rint(np.asarray(wt, float)).astype(int)
        eta = X @ coef
        if offset is not None and offset[0] is not None:
            eta = eta + offset[0]
        if deriv == 0:
            return {"l": _coxlpl(eta, X, d, y, -1)["l"]}
        cderiv = deriv - 1                  # C-level deriv code
        if cderiv < 3:
            res = _coxlpl(eta, X, d, y, cderiv, d1b=d1b)
            out = {"l": res["l"], "lb": res["lb"], "lbb": res["lbb"]}
            if deriv == 2:                  # d1H as a TRACE vector (fh=Hp⁻¹)
                fh = np.asarray(fh, float)
                out["d1H"] = np.array(
                    [float(np.sum(fh * res["d1H"][:, :, m]))
                     for m in range(res["d1H"].shape[2])])
            elif deriv == 3:                # d1H as a matrix list
                out["d1H"] = [res["d1H"][:, :, m]
                              for m in range(res["d1H"].shape[2])]
            return out
        # deriv == 4 → trHid2H: rebuild eigen(raw Hp) from fh=(L,piv)+D
        eigen = self._eigen_from_fh(fh, D, X.shape[1])
        res = _coxlpl(eta, X, d, y, 3, d1b=d1b, d2b=d2b, D=np.asarray(D, float),
                      eigen=eigen)
        return {"l": res["l"], "lb": res["lb"], "lbb": res["lbb"],
                "trHid2H": res["trHid2H"]}

    @staticmethod
    def _eigen_from_fh(fh, D, p):
        """Raw penalized Hessian Hp = D⁻¹·unpivot(LᵀL)·D⁻¹ from gam.fit5's
        preconditioned pivoted Cholesky (gamlss_gH's ``fh``/``D``
        convention), then its symmetric eigen-decomposition — what mgcv's
        cox ll gets by ``eigen(Hp)`` directly."""
        if isinstance(fh, dict):
            return fh
        R_f, piv = fh
        D = np.asarray(D, float)
        M = R_f.T @ R_f                     # (LᵀL) in pivoted order
        ipiv = np.empty_like(piv)
        ipiv[piv] = np.arange(p)
        M = M[np.ix_(ipiv, ipiv)]           # natural order
        Hr = M / D[:, None] / D[None, :]
        val, vec = np.linalg.eigh(Hr)
        return {"values": val, "vectors": vec}

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        # mgcv: start <- rep(0, ncol(x))  (coxph.r:76)
        return np.zeros(np.asarray(X, float).shape[1])

    def set_fit_context(self, *, X, coef, offset):
        """Stash the converged design (original basis, original row order)
        + coefficients so :meth:`postproc`/:meth:`predict` can form the
        baseline hazard's ``a`` vectors, which need ``X`` (absent from the
        6-arg postproc signature)."""
        self._fit_ctx = {"X": np.asarray(X, float),
                         "coef": np.asarray(coef, float),
                         "offset": offset}

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """cox.ph postproc (coxph.r:48-72): baseline hazard estimation,
        the per-observation survivor function into ``fitted``, and the
        null deviance (no-covariate baseline survival)."""
        eps = float(np.finfo(float).eps)
        y = np.asarray(y, float)
        eta = np.asarray(linear_predictors, float)[:, 0]
        d = np.rint(np.asarray(prior_weights, float)).astype(int)
        ctx = self._fit_ctx
        Xc = ctx["X"] if ctx is not None else np.zeros((y.size, 0))
        hz = _coxpp(eta, Xc, d, y)
        self._cox_data = hz
        # baseline survival of the NULL (no-covariate) model
        hz0 = _coxpp(np.zeros_like(eta), np.zeros((y.size, 0)), d, y)
        s0 = np.exp(-hz0["h"][hz0["r"]])
        s0 = np.minimum(s0, 1.0 - 2.0 * eps)
        s_model = np.exp(-hz["h"][hz["r"]] * np.exp(eta))   # survivor
        w = np.asarray(prior_weights, float)
        null_dev = 2.0 * float(np.sum(np.abs(
            w + np.log(s0) + w * np.log(-np.log(s0)))))
        return {"fitted": s_model, "null_deviance": null_dev}

    def residuals(self, y, fitted, type: str = "deviance",
                  prior_weights=None):
        """cox.ph residuals (coxph.r:165-196): martingale and deviance
        (score/schoenfeld need the fit covariance and live on the model
        object). ``fitted`` is the (n,) survivor function; ``prior_weights``
        is the event indicator, passed by the engine."""
        if type not in ("deviance", "martingale"):
            raise NotImplementedError(
                f"cox.ph residuals support 'deviance'/'martingale'; got "
                f"{type!r} (score/schoenfeld need Vp and are model-level)")
        w = np.asarray(prior_weights, float)
        log_s = np.log(np.asarray(fitted, float))
        res = w + log_s                     # martingale residuals
        if type == "martingale":
            return res
        log_s = np.minimum(log_s, -1e-50)
        return np.sign(res) * np.sqrt(np.maximum(
            -2.0 * (res + w * np.log(-log_s)), 0.0))

    def predict(self, *, se, y, X, beta, off, Vb, lpi, eta=None):
        """cox.ph survivor-function prediction (coxph.r:199-245). ``y``
        carries the NEW event times (predict.gam supplies them for
        cox.ph). Returns survival probabilities (and their s.e.)."""
        if self._cox_data is None:
            raise RuntimeError("cox.ph predict needs a fitted model")
        if y is None:
            raise ValueError(
                "cox.ph response-scale prediction needs the event time "
                "column in newdata")
        t = np.asarray(y, float)
        X = np.asarray(X, float)
        n = X.shape[0]
        if isinstance(off, (list, tuple)):       # per-LP list (n_lp == 1)
            off = off[0]
        off = np.zeros(n) if off is None else np.asarray(off, float)
        order = np.argsort(-t, kind="stable")     # descending
        s = np.zeros(n)
        sef = np.zeros(n)
        cd = self._cox_data
        s_o, se_o = _coxpred(X[order], t[order], np.asarray(beta, float),
                             off[order], np.asarray(Vb, float),
                             cd["a"], cd["h"], cd["q"], cd["tr"])
        s[order] = s_o
        sef[order] = se_o
        return {"fit": s, "se_fit": sef if se else None}

    def __repr__(self):
        return "cox_ph(link='identity')"


# --- zero-inflated Poisson (ziplss, gamlss.r:1455-1939) --------------------
# Robustified scalar helpers behind the ziplss likelihood: l1ee/lee1 evaluate
# log(1-e^{-e^x}) / log(e^{e^x}-1) accurately into the tails; ldg/lde give the
# y>0 log-likelihood derivatives w.r.t. gamma = log(Poisson mean) and the
# presence linear predictor eta.

def _l1ee(x):
    """``log(1 - exp(-exp(x)))`` (mgcv l1ee, gamlss.r:1483-1492)."""
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    xmax = np.finfo(float).max
    # divide: log(1-e^{-e^x})→-inf for very negative x, overwritten below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ex = np.exp(x)
        l = np.log(1.0 - np.exp(-ex))
        ind = x < np.log(eps) / 3.0
        exi = ex[ind]
        l[ind] = np.log(exi - exi ** 2 / 2.0 + exi ** 3 / 6.0)
        ind = x < -np.log(xmax)
        l[ind] = x[ind]
    return l


def _lee1(x):
    """``log(exp(exp(x)) - 1)`` (mgcv lee1, gamlss.r:1494-1505)."""
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    xmax = np.finfo(float).max
    # divide: log(e^{e^x}-1)→-inf for very negative x, overwritten below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ex = np.exp(x)
        l = np.log(np.exp(ex) - 1.0)
        ind = x < np.log(eps) / 3.0
        exi = ex[ind]
        l[ind] = np.log(exi + exi ** 2 / 2.0 + exi ** 3 / 6.0)
        ind = x < -np.log(xmax)
        l[ind] = x[ind]
        ind = x > np.log(np.log(xmax))
        l[ind] = ex[ind]
    return l


def _ldg(g, deriv=4):
    """Derivatives of the y>0 ziplss log-lik w.r.t. gamma = log(Poisson
    mean), robustified in both tails (mgcv ldg, gamlss.r:1507-1550).
    Returns ``(l1, l2, l3, l4)`` with l3/l4 = ``None`` below the requested
    order."""
    g = np.asarray(g, dtype=float)
    eps = np.finfo(float).eps
    xmax = np.finfo(float).max
    lo = np.log(eps) / 3.0

    def alpha(gg):
        a = gg.copy()
        m = gg > lo
        eg = np.exp(gg)
        a[m] = eg[m] / (1.0 - np.exp(-eg[m]))
        nm = ~m
        a[nm] = 1.0 + eg[nm] / 2.0 + eg[nm] ** 2 / 12.0
        return a

    with np.errstate(over="ignore", invalid="ignore"):
        ind = g < lo
        ghi = np.log(np.log(xmax)) + 1.0
        ii = g > ghi
        a = alpha(g)
        eg = np.exp(g)
        l2 = a * (a - eg - 1.0)
        egi = eg[ind]
        b = egi * (1.0 + egi / 6.0) / 2.0
        l2[ind] = a[ind] * (b - egi)
        l2[ii] = -np.exp(g[ii])
        l3 = l4 = None
        if deriv > 1:
            l3 = a * (a * (-2.0 * a + 3.0 * (eg + 1.0)) - 3.0 * eg
                      - eg ** 2 - 1.0)
            l3[ind] = a[ind] * (-b - 2.0 * b ** 2 + 3.0 * b * egi - egi ** 2)
            l3[ii] = -np.exp(g[ii])
        if deriv > 2:
            l4 = a * (6.0 * a ** 3 - 12.0 * (eg + 1.0) * a ** 2
                      + 4.0 * eg * a + 7.0 * (eg + 1.0) ** 2 * a
                      - (4.0 + 3.0 * eg) * eg - (eg + 1.0) ** 3)
            l4[ind] = a[ind] * (
                6.0 * b * (3.0 + 3.0 * b + b ** 2)
                - 12.0 * egi * (1.0 + 2.0 * b + b ** 2)
                - 12.0 * b * (2.0 - b) + 4.0 * egi * (1.0 + b)
                + 7.0 * (egi ** 2 + 2.0 * egi + b * egi ** 2
                         + 2.0 * b * egi + b)
                - (4.0 + 3.0 * egi) * egi
                - egi * (3.0 + 3.0 * egi + egi ** 2))
            l4[ii] = -np.exp(g[ii])
        l1 = -a
        # final overflow guard: above log(xmax)/5 every order saturates to
        # -exp of that threshold (mgcv gamlss.r:1544-1548).
        ghi2 = np.log(xmax) / 5.0
        ii2 = g > ghi2
        if np.any(ii2):
            clamp = -np.exp(ghi2)
            l1[ii2] = clamp
            l2[ii2] = clamp
            if l3 is not None:
                l3[ii2] = clamp
            if l4 is not None:
                l4[ii2] = clamp
    return l1, l2, l3, l4


def _lde(eta, deriv=4):
    """Derivatives of the y>0 ziplss presence log-lik ``log(1-e^{-e^eta})``
    w.r.t. eta, robustified (mgcv lde, gamlss.r:1552-1583)."""
    eta = np.asarray(eta, dtype=float)
    eps = np.finfo(float).eps
    xmax = np.finfo(float).max
    lo = np.log(eps) / 3.0
    lxmax = np.log(xmax)
    with np.errstate(over="ignore", invalid="ignore"):
        ind = eta < lo
        ii = eta > lxmax
        et = np.exp(eta)
        l1 = et.copy()
        eti = et[ind]
        nm = ~ind
        l1[nm] = et[nm] / (np.exp(et[nm]) - 1.0)
        b = -eti * (1.0 + eti / 6.0) / 2.0
        l1[ind] = 1.0 + b
        l1[ii] = 0.0
        l2 = l1 * ((1.0 - et) - l1)
        l2[ind] = -b * (1.0 + eti + b) - eti
        l2[ii] = 0.0
        l3 = l4 = None
        if deriv > 1:
            ii2 = eta > lxmax / 2.0
            l3 = l1 * ((1.0 - et) ** 2 - et - 3.0 * (1.0 - et) * l1
                       + 2.0 * l1 ** 2)
            l3[ind] = l1[ind] * (-3.0 * eti + eti ** 2
                                 - 3.0 * (-eti + b - eti * b)
                                 + 2.0 * b * (2.0 + b))
            l3[ii2] = 0.0
        if deriv > 2:
            ii3 = eta > lxmax / 3.0
            l4 = l1 * ((3.0 * et - 4.0) * et + 4.0 * et * l1
                       + (1.0 - et) ** 3 - 7.0 * (1.0 - et) ** 2 * l1
                       + 12.0 * (1.0 - et) * l1 ** 2 - 6.0 * l1 ** 3)
            l4[ii3] = 0.0
            l4[ind] = l1[ind] * (4.0 * l1[ind] * eti - eti ** 3 - b
                                 - 7.0 * b * eti ** 2 - eti ** 2 - 5.0 * eti
                                 - 10.0 * b * eti - 12.0 * eti * b ** 2
                                 - 6.0 * b ** 2 - 6.0 * b ** 3)
    return l1, l2, l3, l4


def _zipll(y, g, eta, deriv=0):
    """Zero-inflated Poisson log-likelihood and its derivatives w.r.t.
    gamma = log(Poisson mean) (=``g``) and the presence LP ``eta``, where
    1-p = exp(-exp(eta)), lambda = exp(g) (mgcv zipll, gamlss.r:1586-1640).
    Packed columns follow mgcv: l1 (g, e); l2 (gg, ge, ee); l3 (ggg, gge,
    gee, eee); l4 (gggg, ggge, ggee, geee, eeee). The expected Hessian
    ``El2`` (cols gg, ge, ee) is consumed by the single-formula ``ziP``
    family's ``EDmu2`` (Fisher weight); the ``ziplss`` ``ll`` hook ignores
    it (it uses the observed ``l2``)."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    eta = np.asarray(eta, dtype=float)
    n = y.shape[0]
    zind = y == 0
    nz = ~zind
    yp = y[nz]
    with np.errstate(over="ignore", invalid="ignore"):
        et = np.exp(eta)
        l = et.copy()
        l[zind] = -et[zind]
        l[nz] = (_l1ee(eta[nz]) + yp * g[nz] - _lee1(g[nz])
                 - gammaln(yp + 1.0))
    ret = {"l": l}
    if not deriv:
        return ret
    le = _lde(eta, deriv)
    lg = _ldg(g, deriv)
    l1 = np.zeros((n, 2))
    l1[nz, 0] = yp + lg[0][nz]       # l_gamma, y>0
    l1[zind, 1] = l[zind]            # l_eta, y==0 (all e-derivs = -exp(eta))
    l1[nz, 1] = le[0][nz]            # l_eta, y>0
    l2 = np.zeros((n, 3))            # order gg, ge, ee
    l2[nz, 0] = lg[1][nz]
    l2[nz, 2] = le[1][nz]
    l2[zind, 2] = l[zind]
    # Expected Hessian (mgcv El2, gamlss.r:1620-1621): E[l_gg] = p·lg2,
    # E[l_ee] = −(1−p)·e^eta + p·le2 (cols gg, ge≡0, ee). p = 1 − e^{−e^eta}.
    with np.errstate(over="ignore", invalid="ignore"):
        p = 1.0 - np.exp(-et)
        El2 = np.zeros((n, 3))
        El2[:, 0] = p * lg[1]
        El2[:, 2] = -(1.0 - p) * et + p * le[1]
    ret["l1"] = l1
    ret["l2"] = l2
    ret["El2"] = El2
    if deriv > 1:                    # order ggg, gge, gee, eee
        l3 = np.zeros((n, 4))
        l3[nz, 0] = lg[2][nz]
        l3[nz, 3] = le[2][nz]
        l3[zind, 3] = l[zind]
        ret["l3"] = l3
    if deriv > 3:                    # order gggg, ggge, ggee, geee, eeee
        l4 = np.zeros((n, 5))
        l4[nz, 0] = lg[3][nz]
        l4[nz, 4] = le[3][nz]
        l4[zind, 4] = l[zind]
        ret["l4"] = l4
    return ret


# lambda maximizing the zero-truncated Poisson likelihood for y=2..17
# (mgcv ziplss residuals/postproc, gamlss.r:1670-1672); above y=17 the
# maximizer is essentially y, and y<2 contributes 0.
_ZIPLSS_GLO = np.array([
    1.593624, 2.821439, 3.920690, 4.965114, 5.984901, 6.993576,
    7.997309, 8.998888, 9.999546, 10.999816, 11.999926, 12.999971,
    13.999988, 14.999995, 15.999998, 16.999999])


def _ziplss_ls(y):
    """ziplss saturated (per-datum max) log-likelihood (gamlss.r:1665-1678
    / 1772-1785): the maximizing lambda is tabulated for 2≤y≤17, ≈y above,
    and y<2 contributes 0 (the zero-truncated Poisson is degenerate there).
    Evaluated at presence eta=1e10 (always present)."""
    y = np.asarray(y, dtype=float)
    l = y.copy()
    l[y < 2] = 0.0
    g = y.copy()
    ind = (y > 1) & (y < 18)
    g[ind] = _ZIPLSS_GLO[y[ind].astype(int) - 2]
    ind2 = y > 1
    l[ind2] = _zipll(y[ind2], np.log(g[ind2]),
                     np.full(int(ind2.sum()), 1e10))["l"]
    return l


class ziplss(GeneralFamily):
    """Zero-inflated Poisson location-scale general family — mgcv
    ``ziplss()`` (gamlss.r:1643-1939). Two identity-linked linear
    predictors: LP1 is gamma = log(Poisson mean) given presence
    (lambda = e^{η₁}); LP2 sets the probability of presence through
    1 − p = exp(−exp(η₂)). The response is non-negative integer counts.

        log f = −e^{η₂}                                       (y = 0)
        log f = log(1−e^{−e^{η₂}}) + y·η₁ − log(e^{e^{η₁}}−1) − log y!  (y > 0)

    ``available_derivs = 2``: full Newton. Like mgcv's other general
    families the likelihood ignores prior weights (gamlss.r:1814 — ``wt``
    unread); they enter neither the deviance nor the null deviance here.
    The fitted matrix stays (gamma, presence-eta) — ziplss does not rewrite
    it in :meth:`postproc`.
    """
    name = "ziplss"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2

    def __init__(self, link: tuple[str, str] = ("identity", "identity")):
        la, lb = link
        if la != "identity" or lb != "identity":
            raise ValueError(
                'only the "identity" link is available for both parameters '
                "of ziplss")
        self.tri = trind_generator(2)
        super().__init__([IdentityLink(), IdentityLink()])

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        eta = X[:, jj[0]] @ coef[jj[0]]
        eta1 = X[:, jj[1]] @ coef[jj[1]]
        if offset is not None:
            if offset[0] is not None:
                eta = eta + offset[0]
            if len(offset) > 1 and offset[1] is not None:
                eta1 = eta1 + offset[1]
        lam = self.links[0].linkinv(eta)     # gamma = log Poisson mean
        p = self.links[1].linkinv(eta1)      # presence LP
        zl = _zipll(y, lam, p, deriv)
        wt = np.ones(y.shape[0]) if wt is None else np.asarray(
            wt, dtype=float).ravel()
        ret: dict = {"l": float(np.sum(wt * zl["l"])), "l0": zl["l"]}
        if deriv == 0:
            return ret
        ig1 = np.column_stack([self.links[0].mu_eta(eta),
                               self.links[1].mu_eta(eta1)])
        g2 = np.column_stack([self.links[0].d2link(lam),
                              self.links[1].d2link(p)])
        g3 = g4 = None
        if deriv > 1:
            g3 = np.column_stack([self.links[0].d3link(lam),
                                  self.links[1].d3link(p)])
        if deriv > 3:
            g4 = np.column_stack([self.links[0].d4link(lam),
                                  self.links[1].d4link(p)])
        tri = self.tri
        l1, l2, l3, l4 = self._apply_prior_weights(
            wt, zl["l1"], zl["l2"], zl.get("l3"), zl.get("l4"))
        de = gamlss_etamu(l1, l2, l3, l4,
                          ig1, g2, g3, g4, tri["i2"], tri["i3"],
                          tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """ziplss ``initialize`` (gamlss.r:1882-1929): regress the binarized
        response on LP2's columns (the presence model), down-weight the
        zeros whose fitted presence < 0.5, then regress
        ``log(|y| + 0.2·(y==0))`` on LP1's columns under those weights.
        Validates integer, non-binary counts. mgcv's ziplss initialize
        ignores ``offset`` — so does this. ``E``/``use_unscaled`` as in
        :meth:`gaulss.initialize_coef`."""
        y = np.asarray(y, dtype=float)
        if not np.allclose(y, np.round(y)):
            raise ValueError(
                "non-integer response values are not allowed with ziplss")
        if y.min() == 0 and y.max() == 1:
            raise ValueError("using ziplss for binary data makes no sense")
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        n = X.shape[0]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(xx, cols, target):
            if use_unscaled:
                xa = np.vstack([xx, E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(xx, E[:, cols], target)

        # LP2: binarized response on the presence design.
        yt = (y != 0).astype(float)
        b2 = _reg(X[:, jj[1]], jj[1], yt)
        start[jj[1]] = b2
        pres = X[:, jj[1]] @ b2
        w = np.ones(n)
        w[(y == 0) & (pres < 0.5)] = 0.1
        # LP1: log|y| (presence-conditional Poisson log-mean) under w; the
        # data rows are w-scaled, the penalty root E1 is not (mgcv stacks
        # rbind(w·X1, E1)).
        yt2 = self.links[0].link(np.log(np.abs(y) + (y == 0) * 0.2)) * w
        x1 = w[:, None] * X[:, jj[0]]
        start[jj[0]] = _reg(x1, jj[0], yt2)
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """ziplss postproc (gamlss.r:1769-1807): null deviance from a
        two-parameter null model — the zero/non-zero part ``fp`` maximized
        over p and the y>0 Poisson part ``flam`` maximized over lambda by
        R's 1-D ``optimize`` (Brent), reproduced via :func:`_brent_fmin`.
        No fitted rewrite."""
        y = np.asarray(y, dtype=float)
        nz = int(np.sum(y == 0))
        npos = int(np.sum(y > 0))
        eps_h = np.sqrt(np.finfo(float).eps)

        def fp(pp):
            l1p = np.log(1.0 - pp) if pp > eps_h else -pp - pp * pp / 2.0
            return l1p * nz + np.log(pp) * npos

        ypos = y[y > 0]

        def flam(lam):
            return float(np.sum(ypos * np.log(lam) - np.log(np.exp(lam) - 1.0)
                                - gammaln(ypos + 1.0)))

        tol = np.finfo(float).eps ** 0.25     # R optimize() default
        _, neg1 = _brent_fmin(lambda v: -fp(v), 1e-60, 1.0 - 1e-10, tol)
        my = float(np.mean(ypos))
        _, neg2 = _brent_fmin(lambda v: -flam(v), my / 2.0, my * 2.0, tol)
        lnull = -neg1 - neg2
        nd = 2.0 * (float(np.sum(_ziplss_ls(y))) - lnull)
        return {"null_deviance": nd}

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """ziplss residuals (gamlss.r:1664-1696): response = y − E(y) with
        E(y) = p·λ/(1−e^{−λ}) (→ p as λ→0); deviance =
        sign(y−E(y))·√(2(ls−ll̂)). ``fitted`` is the (n, 2) matrix
        (gamma, presence-eta). Only ``deviance``/``response`` (mgcv's two
        types)."""
        if type not in ("deviance", "response"):
            raise ValueError(
                "type must be one of 'deviance', 'response' for ziplss "
                f"residuals; got {type!r}")
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            p = 1.0 - np.exp(-np.exp(fitted[:, 1]))
            lam = np.exp(fitted[:, 0])
            ind = lam > np.sqrt(np.finfo(float).eps)
            ey = p.copy()
            ey[ind] = p[ind] * lam[ind] / (1.0 - np.exp(-lam[ind]))
        rsd = y - ey
        if type == "response":
            return rsd
        d = np.maximum(0.0, 2.0 * (_ziplss_ls(y)
                       - _zipll(y, fitted[:, 0], fitted[:, 1])["l"]))
        return np.sqrt(d) * np.sign(rsd)

    def rd(self, rng, mu, wt, scale):
        """ziplss rd (gamlss.r:1747-1767): draw presence ~ Bernoulli(p),
        then a zero-truncated Poisson(λ) for the present rows by inverse
        CDF. ``mu`` is the (n, 2) fitted matrix (gamma, presence-eta)."""
        from scipy.stats import poisson as _poisson
        mu = np.asarray(mu, dtype=float)
        gamma = mu[:, 0]
        eta = mu[:, 1]
        n = gamma.shape[0]
        lam = np.exp(gamma)
        p = 1.0 - np.exp(-np.exp(eta))
        y = np.zeros(n)
        ind = p > rng.uniform(0.0, 1.0, n)
        m = int(np.sum(ind))
        if m:
            lo = np.exp(-lam[ind])             # dpois(0, lambda)
            u = rng.uniform(lo, 1.0, m)
            one_eps = 1.0 - np.finfo(float).eps ** 0.75
            u[u > one_eps] = one_eps
            y[ind] = _poisson.ppf(u, lam[ind])
        return y

    def predict(self, *, se: bool = False, eta=None, y=None, X=None,
                beta=None, off=None, Vb=None, lpi=None) -> dict:
        """ziplss ``family$predict`` (gamlss.r:1698-1744): response-scale
        fit E(y) = p·μ with μ the zero-truncated Poisson mean (→ 1 as
        λ→0). Returns a single ``fit`` column (mgcv emits one column for
        ziplss, not n_lp). With ``se`` a delta-method SE; mgcv reuses
        gamma's variance for the eta term ``v.e`` (gamlss.r:1718 — a
        copy-paste of the line above), reproduced bug-for-bug so the SE
        matches ``predict.gam``."""
        if eta is None:
            X = np.asarray(X, dtype=float)
            beta = np.asarray(beta, dtype=float)
            c1 = np.asarray(lpi[0], dtype=int)
            c2 = np.asarray(lpi[1], dtype=int)
            X1 = X[:, c1]
            X2 = X[:, c2]
            gamma = X1 @ beta[c1]
            eta_p = X2 @ beta[c2]
            if off is not None:
                if off[0] is not None:
                    gamma = gamma + off[0]
                if len(off) > 1 and off[1] is not None:
                    eta_p = eta_p + off[1]
            if se:
                v_g = np.maximum(0.0, np.einsum(
                    "ij,jk,ik->i", X1, Vb[np.ix_(c1, c1)], X1))
                v_e = v_g     # mgcv copy-paste (gamlss.r:1718 reuses X1/lpi1)
                v_eg = np.maximum(0.0, np.einsum(
                    "ij,jk,ik->i", X1, Vb[np.ix_(c1, c2)], X2))
        else:
            eta = np.asarray(eta, dtype=float)
            se = False
            gamma = eta[:, 0]
            eta_p = eta[:, 1]
        with np.errstate(over="ignore", invalid="ignore"):
            et = np.exp(eta_p)
            p = 1.0 - np.exp(-et)
            lam = np.exp(gamma)
            mu = np.where(gamma < np.log(np.finfo(float).eps) / 2.0, 1.0,
                          lam / (1.0 - np.exp(-lam)))
            ey = p * mu
        if not se:
            return {"fit": ey}
        with np.errstate(over="ignore", invalid="ignore"):
            df_de = np.where(eta_p < np.log(np.finfo(float).max) / 2.0,
                             np.exp(-et) * et, 0.0) * mu
            df_dg = ((lam + 1.0) * mu - mu ** 2) * p
            se_fit = np.sqrt(df_dg ** 2 * v_g + df_de ** 2 * v_e
                             + 2.0 * df_de * df_dg * v_eg)
        return {"fit": ey, "se_fit": se_fit}

    def __repr__(self):
        return "ziplss(link=('identity', 'identity'))"


# --- multinomial logistic (multinom, gamlss.r:1107-1411) -------------------

def _multinom_derivs(y, eta, tri, deriv):
    """Dense l0..l4 log-likelihood derivatives for the multinomial logistic
    model (mgcv multinom ll, gamlss.r:1246-1330). ``eta`` is the (n, K)
    matrix of the K real linear predictors (category 0 is the reference,
    η₀ ≡ 0); ``y`` integer classes 0..K. The packed l1..l4 arrays follow
    mgcv exactly (NO remap — every column dense) and feed gamlss_gH with no
    etamu transform (the links are identity, so ∂/∂η ≡ ∂/∂μ)."""
    y = np.round(np.asarray(y, dtype=float)).astype(int)
    eta = np.asarray(eta, dtype=float)
    n, K = eta.shape
    ee = np.exp(eta)                       # (n, K) = exp of the real LPs
    beta = 1.0 + ee.sum(axis=1)            # normalizer 1 + Σ exp(η_j)
    alpha = np.log(beta)
    # mgcv pads eta with a dummy first column of 1's (eta[,1] <- 1): for a
    # y=0 datum l0 gathers that 1, NOT 0 — a constant +1 per reference-class
    # datum baked into mgcv's reported log-lik (it cancels from grad/Hess).
    # Reproduced bug-for-bug so the REML value matches mgcv.
    eta_full = np.column_stack([np.ones(n), eta])
    l0 = eta_full[np.arange(n), y] - alpha
    out = {"l0": l0, "l": float(np.sum(l0))}
    if not deriv:
        return out
    i2 = tri["i2"]
    i3 = tri["i3"]
    b2 = beta * beta
    alpha1 = ee / beta[:, None]            # ee_i / beta
    # second derivatives (packed i<=j) — built while l1 still holds alpha1.
    l2 = np.zeros((n, K * (K + 1) // 2))
    for i in range(K):
        for j in range(i, K):
            col = i2[i, j]
            if i == j:
                l2[:, col] = -alpha1[:, i] + ee[:, i] ** 2 / b2
            else:
                l2[:, col] = ee[:, i] * ee[:, j] / b2
    # first derivatives: (y == category i+1) − ee_i/beta.
    l1 = np.zeros((n, K))
    for i in range(K):
        l1[:, i] = (y == i + 1).astype(float) - alpha1[:, i]
    out["l1"] = l1
    out["l2"] = l2
    if deriv > 1:                          # third derivatives (packed i<=j<=k)
        b3 = b2 * beta
        l3 = np.zeros((n, int(i3.max()) + 1))
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    col = i3[i, j, k]
                    eijk = ee[:, i] * ee[:, j] * ee[:, k]
                    if i == j == k:
                        l3[:, col] = (l2[:, i2[i, i]] + 2.0 * ee[:, i] ** 2 / b2
                                      - 2.0 * ee[:, i] ** 3 / b3)
                    elif i != j and j != k and i != k:
                        l3[:, col] = -2.0 * eijk / b3
                    else:                  # two equal, one different
                        kk = k if i == j else j
                        l3[:, col] = l2[:, i2[i, kk]] - 2.0 * eijk / b3
        out["l3"] = l3
    if deriv > 3:                          # fourth derivs (packed i<=j<=k<=l)
        i4 = tri["i4"]
        b3 = b2 * beta
        b4 = b3 * beta
        l3 = out["l3"]
        l4 = np.zeros((n, int(i4.max()) + 1))
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    for m in range(k, K):
                        col = i4[i, j, k, m]
                        eijkl = ee[:, i] * ee[:, j] * ee[:, k] * ee[:, m]
                        uni = np.unique([i, j, k, m])      # sorted (= 1st-seen)
                        nun = uni.size
                        if nun == 1:
                            l4[:, col] = (l3[:, i3[i, i, i]]
                                          + 4.0 * ee[:, i] ** 2 / b2
                                          - 10.0 * ee[:, i] ** 3 / b3
                                          + 6.0 * ee[:, i] ** 4 / b4)
                        elif nun == 4:
                            l4[:, col] = 6.0 * eijkl / b4
                        elif nun == 3:
                            l4[:, col] = (l3[:, i3[uni[0], uni[1], uni[2]]]
                                          + 6.0 * eijkl / b4)
                        else:              # nun == 2: split 2+2 or 3+1
                            cnt0 = int(np.sum(np.array([i, j, k, m]) == uni[0]))
                            if cnt0 == 2:
                                l4[:, col] = (l3[:, i3[uni[0], uni[1], uni[1]]]
                                              - 2.0 * ee[:, uni[0]] ** 2
                                              * ee[:, uni[1]] / b3
                                              + 6.0 * eijkl / b4)
                            else:          # 3 of one, 1 of the other
                                u = uni if cnt0 == 3 else uni[::-1]
                                l4[:, col] = (l3[:, i3[u[0], u[0], u[1]]]
                                              - 4.0 * ee[:, u[0]] ** 2
                                              * ee[:, u[1]] / b3
                                              + 6.0 * eijkl / b4)
        out["l4"] = l4
    return out


class multinom(GeneralFamily):
    """Multinomial logistic regression general family — mgcv ``multinom(K)``
    (gamlss.r:1107-1411). The only **variable-K** family: K linear
    predictors for K+1 categories coded ``0..K``, category 0 the reference
    (η₀ ≡ 0), all identity-linked.

        P(y = j) = e^{η_j} / (1 + Σ_{m=1}^K e^{η_m}),  j = 1..K;
        P(y = 0) = 1 / (1 + Σ_{m=1}^K e^{η_m}).

    ``gam`` takes K formulas (the first carries the integer-class response).
    ``available_derivs = 2``: full Newton. The likelihood derivatives are
    built as DENSE packed tensors exactly as mgcv does (no ``remap``
    sparsity — dormant in shipped mgcv) and passed straight to
    :func:`gamlss_gH` (identity links ⇒ no :func:`gamlss_etamu` step)."""
    name = "multinom"
    has_sandwich = True
    scale_known = True
    n_theta = 0
    available_derivs = 2
    discrete_ok = True          # gamlss.r:1409

    def __init__(self, K: int = 1):
        if K < 1:
            raise ValueError("number of categories must be at least 2 "
                             "(multinom K must be >= 1)")
        self.n_lp = int(K)
        self.tri = trind_generator(int(K))
        super().__init__([IdentityLink() for _ in range(int(K))])

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None,
           sandwich: bool = False) -> dict:
        y = np.asarray(y, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        K = len(jj)
        n = y.shape[0]
        discrete = isinstance(X, DiscreteX)
        if discrete:
            _Xbd, _, _ = _discrete_kernels()
        else:
            X = np.asarray(X, dtype=float)
        eta = np.zeros((n, K))
        for i in range(K):
            # gamlss.r:1256: per-LP η off the compressed design.
            eta[:, i] = (_Xbd(X.design, coef, lt=X.lpid[i]) if discrete
                         else X[:, jj[i]] @ coef[jj[i]])
            if (offset is not None and i < len(offset)
                    and offset[i] is not None):
                eta[:, i] = eta[:, i] + offset[i]
        d = _multinom_derivs(y, eta, self.tri, deriv)
        wt = np.ones(n) if wt is None else np.asarray(wt, dtype=float).ravel()
        ret: dict = {"l": float(np.sum(wt * d["l0"])), "l0": d["l0"]}
        if deriv == 0:
            return ret
        l1, l2, l3, l4 = self._apply_prior_weights(
            wt, d["l1"], d["l2"], d.get("l3"), d.get("l4"))
        gh = gamlss_gH(X, jj, l1, l2, self.tri["i2"],
                       l3=l3, i3=self.tri["i3"], l4=l4,
                       i4=self.tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D, sandwich=sandwich)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """multinom ``initialize`` (gamlss.r:1356-1399): for each LP k
        regress the binarized signal ``6·(y==k) − 3`` on that LP's columns
        with the penalty root ``E`` as a regularizer. Validates the integer
        class coding 0..K. mgcv's multinom initialize ignores ``offset`` —
        so does this. A :class:`DiscreteX` design takes the discrete
        branch (:1364-1382): the per-LP ``mchol`` solve."""
        y = np.round(np.asarray(y, dtype=float)).astype(int)
        K = len(lpi)
        if y.min() < 0 or y.max() > K:
            raise ValueError(
                f"multinom response must be integer classes in 0..{K} "
                f"(K={K} linear predictors); got range "
                f"{int(y.min())}..{int(y.max())}")
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        if isinstance(X, DiscreteX):
            return self._initialize_coef_discrete(y, X, jj, E, offset)
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        for k in range(K):
            yt = 6.0 * (y == k + 1).astype(float) - 3.0
            start[jj[k]] = _reg(jj[k], yt)
        return start

    def _initialize_coef_discrete(self, y, X: DiscreteX, jj, E,
                                  offset) -> np.ndarray:
        """multinom ``initialize``'s discrete branch (gamlss.r:
        1364-1382): per category k the ``mchol`` solve on the
        binarized ``6·(y==k) − 3`` target."""
        design = X.design
        lpid = X.lpid
        p = design.p
        if E is None:
            E = np.zeros((0, p))
        E = np.asarray(E, dtype=float)
        ones_n = np.ones(y.shape[0])
        start = np.zeros(p)
        for k in range(len(jj)):
            yt = 6.0 * (y == k + 1).astype(float) - 3.0
            startji = _DiscreteLPSolve(design, lpid[k], E[:, jj[k]],
                                       ones_n).solve_target(yt)
            startji[~np.isfinite(startji)] = 0.0
            start[jj[k]] = startji
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """multinom postproc (gamlss.r:1219-1231): null deviance from the
        class-frequency null model — solve ``(diag(n/n_j) − 1)·γ = 1`` for
        the K non-reference categories, normalize to probabilities, then
        ``−2·Σ_i log P̂(y_i)``."""
        y = np.round(np.asarray(y, dtype=float)).astype(int)
        K = self.n_lp
        nj = np.bincount(y, minlength=K + 1).astype(float)
        ntot = float(nj.sum())
        A = np.diag(ntot / nj[1:]) - np.ones((K, K))
        gamma = np.concatenate([[1.0], np.linalg.solve(A, np.ones(K))])
        gamma = np.log(gamma / gamma.sum())
        return {"null_deviance": -2.0 * float(np.sum(gamma[y]))}

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """multinom residuals (gamlss.r:1133-1146): deviance only, signed +
        when the most-probable category equals the observed class, − else,
        magnitude ``√(−2 log P̂(y))``. ``fitted`` is the (n, K) η matrix."""
        if type != "deviance":
            raise ValueError(
                "only 'deviance' residuals are available for multinom; "
                f"got {type!r}")
        y = np.round(np.asarray(y, dtype=float)).astype(int)
        p = self.predict(eta=np.asarray(fitted, dtype=float))["fit"]
        n = y.shape[0]
        pc = np.argmax(p, axis=1)
        sgn = np.where(pc == y, 1.0, -1.0)
        py = p[np.arange(n), y]
        return sgn * np.sqrt(-2.0 * np.log(
            np.maximum(np.finfo(float).eps, py)))

    def rd(self, rng, mu, wt, scale):
        """multinom rd (gamlss.r:1348-1354): sample a category by inverse
        CDF of the softmax probabilities (category 0 the reference, η₀ ≡ 0).
        ``mu`` is the (n, K) η matrix."""
        mu = np.asarray(mu, dtype=float)
        n = mu.shape[0]
        p = np.exp(np.column_stack([np.zeros(n), mu]))
        p = p / p.sum(axis=1, keepdims=True)
        cp = np.cumsum(p, axis=1)
        u = rng.uniform(0.0, 1.0, n)
        return (cp > u[:, None]).argmax(axis=1).astype(float)

    def predict(self, *, se: bool = False, eta=None, y=None, X=None,
                beta=None, off=None, Vb=None, lpi=None) -> dict:
        """multinom ``family$predict`` (gamlss.r:1148-1217): the (n, K+1)
        matrix of category probabilities (more columns than ``n_lp``), with
        delta-method SEs over the full η covariance when ``se``."""
        if eta is None:
            X = np.asarray(X, dtype=float)
            beta = np.asarray(beta, dtype=float)
            K = len(lpi)
            n = X.shape[0]
            eta = np.zeros((n, K))
            ve = np.zeros((n, K))
            ce = np.zeros((n, K * (K - 1) // 2))
            ii = 0
            for i in range(K):
                ci = np.asarray(lpi[i], dtype=int)
                Xi = X[:, ci]
                eta[:, i] = Xi @ beta[ci]
                if off is not None and i < len(off) and off[i] is not None:
                    eta[:, i] = eta[:, i] + off[i]
                if se:
                    ve[:, i] = np.maximum(0.0, np.einsum(
                        "ij,jk,ik->i", Xi, Vb[np.ix_(ci, ci)], Xi))
                    for k in range(i + 1, K):
                        ck = np.asarray(lpi[k], dtype=int)
                        ce[:, ii] = np.maximum(0.0, np.einsum(
                            "ij,jk,ik->i", Xi, Vb[np.ix_(ci, ck)], X[:, ck]))
                        ii += 1
        else:
            eta = np.asarray(eta, dtype=float)
            se = False
        n, K = eta.shape
        gamma = np.column_stack([np.ones(n), np.exp(eta)])
        bb = gamma.sum(axis=1)
        gamma = gamma / bb[:, None]
        if not se:
            return {"fit": gamma}
        vp = np.zeros((n, K + 1))
        for jcat in range(K + 1):
            if jcat == 0:
                dp = -gamma[:, 1:] / bb[:, None]
            else:
                dp = -gamma[:, jcat:jcat + 1] * gamma[:, 1:]
                dp[:, jcat - 1] = gamma[:, jcat] * (1.0 - gamma[:, jcat])
            vj = (dp ** 2 * ve).sum(axis=1)
            ii = 0
            for i in range(K):
                for k in range(i + 1, K):
                    vj = vj + 2.0 * dp[:, i] * dp[:, k] * ce[:, ii]
                    ii += 1
            vp[:, jcat] = np.sqrt(np.maximum(0.0, vj))
        return {"fit": gamma, "se_fit": vp}

    def __repr__(self):
        return f"multinom(K={self.n_lp})"


def _mvn_ll(y, X, coef, lpi, m, *, deriv=0, d1b=None, fh=None) -> dict:
    """Multivariate-normal log-likelihood + derivatives — a numpy port of
    mgcv's C kernel ``mvn_ll`` (src/mvn.c).

    Model: each row ``y_i`` (m-dimensional) is ``N(μ_i, Σ)`` with a shared
    precision ``Σ⁻¹ = RᵀR``; ``R`` is the m×m upper-triangular Choleski
    factor whose ``d(d+1)/2`` parameters ``θ`` are the trailing coefs
    (diagonal stored as ``log R_ii``, off-diagonals stored directly).
    ``coef = [β (the m mean-LP blocks), θ]``; ``lpi[k]`` indexes LP k's
    (contiguous) mean columns. The trailing ``X`` columns for ``θ`` are
    structural zeros (mgcv's dummy columns).

        ll = −½ Σ_i ‖R(y_i−μ_i)‖² + n·log|R|.

    ``deriv`` follows mgcv's mvn codes (NOT the gamlss_gH shift): 0 → l;
    1 → +grad ``lb`` / Hessian ``lbb``; 2 → +``d1H`` = the per-ρ traces
    ``tr(fh·∂H/∂ρ)`` given ``fh = H_p⁻¹`` and ``d1b = ∂coef/∂ρ``; 3 →
    +``d1H`` as the list of ``∂H/∂ρ`` matrices. ``XX`` is recomputed from
    the supplied (already-reparameterized) mean columns each call —
    self-consistent with mgcv's ``Sl.repara``-d ``attr(X,"XX")``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float)
    n = y.shape[0]
    ntheta = m * (m + 1) // 2
    nb = coef.size
    ncoef = nb - ntheta
    theta = coef[ncoef:nb]
    # build R (upper-tri), the θ→(row,col) maps, and dR_ii/dθ = R_ii (1 off-diag)
    R = np.zeros((m, m))
    rri = np.zeros(ntheta, int)
    rci = np.zeros(ntheta, int)
    dth = np.zeros(ntheta)
    ldetR = 0.0
    k = 0
    for i in range(m):
        dth[k] = np.exp(theta[k])
        R[i, i] = dth[k]
        ldetR += theta[k]
        rri[k] = rci[k] = i
        k += 1
        for j in range(i + 1, m):
            R[i, j] = theta[k]
            dth[k] = 1.0
            rri[k] = i
            rci[k] = j
            k += 1
    jj = [np.asarray(ix, dtype=int) for ix in lpi]
    mu = np.zeros((n, m))
    for l in range(m):
        mu[:, l] = X[:, jj[l]] @ coef[jj[l]]
    e = y - mu
    Re = e @ R.T                          # Re[obs,c] = Σ_d R[c,d] e[obs,d]
    ll = -0.5 * float(np.sum(Re * Re)) + ldetR * n
    if deriv == 0:
        return {"l": ll}
    P = R.T @ R                           # precision
    Xm = X[:, :ncoef]
    XX = Xm.T @ Xm
    din = np.zeros(ncoef, int)
    for l in range(m):
        din[jj[l]] = l
    # ---- gradient ----
    lb = np.zeros(nb)
    Me = e @ P
    for l in range(m):
        lb[jj[l]] = Xm[:, jj[l]].T @ Me[:, l]
    Ree_diag = np.einsum("ij,ij->j", Re, e)
    for kk in range(ntheta):
        i, j = rri[kk], rci[kk]
        lb[ncoef + kk] = (n - dth[kk] * Ree_diag[i] if i == j
                          else -float(np.dot(Re[:, i], e[:, j])))
    # ---- Hessian ----
    lbb = np.zeros((nb, nb))
    lbb[:ncoef, :ncoef] = -XX * P[np.ix_(din, din)]
    Xe = Xm.T @ e
    XRe = Xm.T @ Re
    for j in range(ntheta):
        ri, rj = rri[j], rci[j]
        col = dth[j] * ((din == rj) * XRe[:, ri]
                        + (ri <= din) * R[ri, din] * Xe[:, rj])
        lbb[:ncoef, ncoef + j] = col
        lbb[ncoef + j, :ncoef] = col
    ee = e.T @ e
    Ree = Re.T @ e
    for kk in range(ntheta):
        ri, rj = rri[kk], rci[kk]
        for ll_ in range(kk + 1):
            ril, rjl = rri[ll_], rci[ll_]
            xx = 0.0
            if kk == ll_ and ri == rj:
                xx -= dth[kk] * Ree[ri, ri]
            if ril == ri:
                yy = ee[rjl, rj] * dth[kk]
                if ril == rjl:
                    yy *= dth[ll_]
                xx -= yy
            lbb[ncoef + kk, ncoef + ll_] = xx
            lbb[ncoef + ll_, ncoef + kk] = xx
    out = {"l": ll, "lb": lb, "lbb": lbb}
    if deriv == 1:
        return out
    # ---- ∂H/∂ρ (mvn.c lines 168-262), given d1b = ∂coef/∂ρ ----
    nsp = d1b.shape[1]
    yX = Xe.T                             # (m, ncoef)
    yRX = XRe.T
    yty = ee
    d1H_list = []
    for r in range(nsp):
        db = d1b[:, r]
        dtheta = db[ncoef:]
        dbq = db[:ncoef]
        dH = np.zeros((nb, nb))
        # mean-coef block: dH[i,j] = -XX[i,j]·C[din[i],din[j]]
        C = np.zeros((m, m))
        for q in range(ntheta):
            ri_q, rj_q, w = rri[q], rci[q], dth[q] * dtheta[q]
            C[rj_q, :] += R[ri_q, :] * w        # rj_q==l term
            C[:, rj_q] += R[ri_q, :] * w        # rj_q==k term
        dH[:ncoef, :ncoef] = -XX * C[np.ix_(din, din)]
        # mixed block (mvn.c:191-228) — vectorized over the ncoef mean-coef
        # index i. The inner q-loop is the matmuls XX @ (R[ri,din]·db) and
        # XX @ (db·[din==rj]); the kk2-loop is ntheta-small. Bit-identical to
        # the C triple loop up to fp summation order (~5e-16 on random inputs,
        # ~10× faster at d=4). `din==·` masks select mgcv's `l == r*` branches.
        for j in range(ntheta):
            ri, rj, zz = rri[j], rci[j], dth[j]
            Rri_din = R[ri, din]
            termA = XX @ (Rri_din * dbq)
            termB = XX @ (dbq * (din == rj))
            partQ = -zz * ((din == rj) * termA + Rri_din * termB)
            partK = np.zeros(ncoef)
            for kk2 in range(ntheta):
                rik, rjk = rri[kk2], rci[kk2]
                if ri == rik:
                    c = dth[j] * dth[kk2]
                    z2 = ((din == rj) * (yX[rjk, :] * c)
                          + (din == rjk) * (yX[rj, :] * c))
                    if kk2 == j and rik == rjk:
                        z2 = z2 + (((din == rj) | (din == rjk))
                                   * (dth[kk2] * yRX[rj, :]))
                    partK = partK + z2 * dtheta[kk2]
                if kk2 == j and rik == rjk:
                    partK = partK + (dtheta[kk2] * dth[kk2]) * (R[rj, din]
                                                               * yX[rj, :])
            col = partQ + partK
            dH[:ncoef, ncoef + j] = col
            dH[ncoef + j, :ncoef] = col
        # theta block (mvn.c:230-262) — the ncoef sum is vectorized; the
        # trailing theta×theta accumulation stays a small ntheta loop.
        for j in range(ntheta):
            rij, rjj = rri[j], rci[j]
            for kk2 in range(j, ntheta):
                rik, rjk = rri[kk2], rci[kk2]
                contrib = np.zeros(ncoef)
                if rij == rik:
                    m_jj = (din == rjj)
                    m_jk = (din == rjk)
                    z2 = (m_jj * (yX[rjk, :] * dth[j] * dth[kk2])
                          + m_jk * (yX[rjj, :] * dth[j] * dth[kk2]))
                    if kk2 == j and rik == rjk:
                        z2 = z2 + m_jj * (dth[kk2] * yRX[rjj, :])
                    contrib = contrib + z2 * dbq
                if kk2 == j and rij == rjj:
                    contrib = contrib + dbq * dth[kk2] * (R[rjj, din]
                                                          * yX[rjj, :])
                xx = float(contrib.sum())
                for i in range(ntheta):
                    ri, rj = rri[i], rci[i]
                    z2 = 0.0
                    if j == kk2 and ri == rij and rjk == rik:
                        z2 += dth[j] * dth[i] * yty[rjj, rj]
                    if i == kk2 and rik == rij and rj == ri:
                        z2 += dth[j] * dth[i] * yty[rjj, rjk]
                    if i == j and rik == rij and rj == ri:
                        z2 += dth[kk2] * dth[i] * yty[rj, rjk]
                    if i == j and j == kk2 and ri == rj:
                        z2 += dth[kk2] * Ree[ri, ri]
                    xx += -z2 * dtheta[i]
                dH[ncoef + kk2, ncoef + j] = xx
                dH[ncoef + j, ncoef + kk2] = xx
        d1H_list.append(dH)
    if deriv == 2:
        out["d1H"] = np.array([float(np.sum(fh * dH)) for dH in d1H_list])
    else:
        out["d1H"] = d1H_list
    return out


class mvn(GeneralFamily):
    """Multivariate normal additive model — mgcv ``mvn(d)`` (mvam.r).

    A d-dimensional Gaussian response with a per-row mean from d linear
    predictors (all identity-linked) and a SHARED covariance, parameterized
    by the upper-triangular Choleski factor ``R`` of the precision matrix
    (``Σ⁻¹ = RᵀR``). ``gam`` takes d formulas, each carrying its own
    dimension's response (``gam(list(y0~s(x), y1~s(z)), family=mvn(2))``);
    the responses stack into the (n, d) matrix response.

    ``available_derivs = 1`` — the ll supplies the gradient/Hessian of the
    log-lik and its first ρ-derivative (``∂H/∂ρ``) but NOT ``∂²H/∂ρ²``, so
    the smoothing parameters are estimated by the **BFGS** outer optimizer
    (mgcv.r:1907), not Newton. The d(d+1)/2 covariance params ride the
    coefficient vector as unpenalized "dummy" columns appended to the
    design (``n_extra_coef``, mgcv's ``preinitialize``)."""
    name = "Multivariate normal"
    scale_known = True
    n_theta = 0
    available_derivs = 1
    matrix_response = True

    def __init__(self, d: int = 2):
        if d < 2:
            raise ValueError("mvn requires 2 or more dimensional data")
        self.d = int(d)
        self.n_lp = int(d)
        self.n_extra_coef = int(d) * (int(d) + 1) // 2
        self._R = None
        self._ibeta = None
        super().__init__([IdentityLink() for _ in range(int(d))])

    def preinitialize_general(self, *, y, X, lpi, slots) -> None:
        """mgcv mvn ``preinitialize`` coefficient seeding (mvam.r:115-125):
        per response dimension k, ``magic(y[,k], X[,lpi[[k]]], rep(-1,·),
        S_k, off_k)`` — a GCV fit of LP k's columns with LP k's own
        penalties, all sp estimated — seeds the mean coefficients
        (``um$b``) and the k-th DIAGONAL θ at ``−½·log(um$scale)`` (the
        initial log root precision); off-diagonal θ start at 0. Stored as
        ``family$ibeta``; mgcv's ``initialize`` expression (mvam.r:152-155,
        run in gam.fit5 AND initial.spg) returns it as-is —
        :meth:`initialize_coef` mirrors that. The engine calls this hook
        once with the INITIAL-REPARA'D design and the ORIGINAL-gauge
        penalties: mgcv reparas ``G$X`` (mgcv.r:1902) before
        preinitialize runs (mgcv.r:1985), so the seed fit's gauge mix is
        mgcv's own, and the stored ibeta is in the irp gauge gam.fit5
        consumes. NOTE on parity: the magic port itself matches mgcv's
        ``magic`` to 8+ digits on identical inputs; the MIXED-gauge seed
        value additionally depends on the Sl.setup eigenvector sign/
        rotation convention (R ``eigen``/dsyevr vs numpy ``eigh``/dsyevd),
        which no consistent-gauge quantity ever sees — so the seed can
        differ from mgcv's while every fitted quantity still pins."""
        # local import: hea.models.gam imports this module at load time,
        # so the reverse import must be deferred to call time.
        from .models.gam import _magic_gcv
        from types import SimpleNamespace
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        m = self.d
        ncoef = p - self.n_extra_coef
        ibeta = np.zeros(p)
        kdiag = 0
        for kdim in range(m):
            cols = jj[kdim]
            pos = {int(c): i for i, c in enumerate(cols)}
            # penalties belonging to LP k: mgcv ``sin <- G$off %in%
            # lpi[[k]]`` with local offsets ``match(G$off[sin], lpi[[k]])``
            # (mvam.r:117-120).
            sub = []
            for sl_ in slots:
                if int(sl_.col_start) in pos:
                    a = pos[int(sl_.col_start)]
                    sub.append(SimpleNamespace(
                        col_start=a,
                        col_end=a + (sl_.col_end - sl_.col_start),
                        S=np.asarray(sl_.S, dtype=float)))
            um = _magic_gcv(y[:, kdim], X[:, cols], sub)
            ibeta[cols] = um["b"]
            ibeta[ncoef + kdiag] = -0.5 * np.log(um["scale"])
            kdiag += m - kdim          # next diagonal θ slot (mvam.r:124)
        self._ibeta = ibeta

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        if offset is not None and any(
                o is not None and np.any(np.asarray(o) != 0) for o in offset):
            raise NotImplementedError("mvn does not handle offsets")
        if deriv >= 4:
            raise NotImplementedError(
                "mvn supplies ll only to deriv 3 (available_derivs=1); "
                "the smoothing parameters are estimated by bfgs, which "
                "never asks for trHid2H.")
        return _mvn_ll(y, X, coef, lpi, self.d, deriv=deriv, d1b=d1b, fh=fh)

    def _R_from_coef(self, coef) -> np.ndarray:
        """Rebuild the precision Choleski factor R from the trailing θ
        coefs (mvam.r postproc: diag = exp(θ), off-diag = θ)."""
        coef = np.asarray(coef, dtype=float)
        m = self.d
        theta = coef[coef.size - self.n_extra_coef:]
        R = np.zeros((m, m))
        k = 0
        for i in range(m):
            for j in range(i, m):
                R[i, j] = np.exp(theta[k]) if i == j else theta[k]
                k += 1
        return R

    def set_fit_context(self, *, X=None, coef=None, offset=None) -> None:
        """Stash the converged R factor so postproc/residuals can use it
        (the 6-arg postproc signature lacks the coefficient vector)."""
        self._R = self._R_from_coef(coef)

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """mvn ``initialize`` (mvam.r:152-155): ``start <- family$ibeta``
        — the magic-GCV seed stored by :meth:`preinitialize_general`,
        returned as-is on every call (gam.fit5 and initial.spg alike;
        mgcv does not re-parameterize it). The no-preinit fallback (a
        direct family use outside the gam engine) penalized-regresses
        each LP with ``E`` and seeds the diagonal θ from the residual
        scale."""
        if self._ibeta is not None:
            return self._ibeta.copy()
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        p = X.shape[1]
        m = self.d
        if E is None:
            E = np.zeros((0, p))
        start = np.zeros(p)

        def _reg(cols, target):
            if use_unscaled:
                xa = np.vstack([X[:, cols], E[:, cols]])
                b, *_ = np.linalg.lstsq(
                    xa, np.concatenate([target, np.zeros(E.shape[0])]),
                    rcond=None)
                b[~np.isfinite(b)] = 0.0
                return b
            return _pen_reg(X[:, cols], E[:, cols], target)

        ntheta = self.n_extra_coef
        ncoef = p - ntheta
        kdiag = 0
        for kdim in range(m):
            bk = _reg(jj[kdim], y[:, kdim])
            start[jj[kdim]] = bk
            resid = y[:, kdim] - X[:, jj[kdim]] @ bk
            df = max(len(jj[kdim]), 1)
            scale = max(float(resid @ resid) / max(len(resid) - df, 1),
                        1e-8)
            start[ncoef + kdiag] = -0.5 * np.log(scale)
            kdiag += m - kdim          # advance to next diagonal θ slot
        return start

    def postproc(self, y, prior_weights, fitted, linear_predictors,
                 offset, intercept) -> dict:
        """mvn postproc (mvam.r:134-150): deviance ``Σ‖R(y−μ̂)‖²`` and null
        deviance ``Σ‖R(y−ȳ)‖²`` using the fitted precision factor R."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        R = self._R if self._R is not None else self._R_from_coef(
            np.zeros(self.d))
        rsd = (y - fitted) @ R.T
        dev = float(np.sum(rsd * rsd))
        rsd0 = (y - np.mean(y, axis=0)) @ R.T
        return {"deviance": dev, "null_deviance": float(np.sum(rsd0 * rsd0))}

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """mvn residuals (mvam.r:162-167): ``response`` = ``y − μ̂``;
        ``deviance`` = ``(y − μ̂)·Rᵀ`` (the whitened residual)."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        res = y - fitted
        if type == "response":
            return res
        if type != "deviance":
            raise ValueError(
                "mvn residuals are 'response' or 'deviance'; got "
                f"{type!r}")
        R = self._R if self._R is not None else self._R_from_coef(
            np.zeros(self.d))
        return res @ R.T

    def __repr__(self):
        return f"mvn(d={self.d})"


def _coerce_response(y_series: pl.Series, family: "Family") -> np.ndarray:
    """Cast the response column to a numeric float array, with R's
    factor-response convention for :class:`Binomial`.

    R's ``glm(y ~ x, family=binomial)`` accepts a 2-level factor on the
    LHS: level 1 → 0 (failure), level 2 → 1 (success). Boolean is the
    same shape (FALSE → 0, TRUE → 1). For other families and numeric y
    we just float-cast.

    Unused factor levels are dropped before the 2-level check — matches
    R's ``glm()``, which calls ``model.frame(..., drop.unused.levels=
    TRUE)`` so a 3-level Enum filtered down to 2 actually-present
    levels still fits cleanly. The filter preserves the declared order
    of the surviving levels, so ``levels[0]`` (the "failure" reference)
    matches what R would pick after ``droplevels()``.
    """
    dt = y_series.dtype
    if isinstance(family, (Binomial, QuasiBinomial)):
        if dt == pl.Boolean:
            return y_series.to_numpy().astype(float)
        if dt == pl.String or isinstance(dt, (pl.Categorical, pl.Enum)):
            if isinstance(dt, pl.Enum):
                declared = list(dt.categories)
            else:
                # No declared order — fall back to alphabetical, which is
                # R's ``factor()`` default when ``levels=`` is unspecified.
                declared = sorted(y_series.drop_nulls().unique().to_list())
            present = set(y_series.drop_nulls().unique().to_list())
            levels = [lvl for lvl in declared if lvl in present]
            if len(levels) != 2:
                raise ValueError(
                    f"Binomial response factor must have 2 levels present "
                    f"in the data; got {len(levels)}: {levels}"
                )
            return (y_series.to_numpy() != levels[0]).astype(float)
    return y_series.to_numpy().astype(float).flatten()


# Convenience exports — mirror R's lowercase/CapCase convention so user code
# reads almost identically: ``gam(..., family=Gamma(link='log'))``.
gaussian = Gaussian
poisson = Poisson
binomial = Binomial
inverse_gaussian = InverseGaussian
quasi = Quasi
quasipoisson = QuasiPoisson
quasibinomial = QuasiBinomial
scat = Scat   # mgcv-style lowercase alias
__all__ = [
    "Family", "Link",
    "Gaussian", "gaussian",
    "Gamma",
    "Poisson", "poisson",
    "Binomial", "binomial",
    "InverseGaussian", "inverse_gaussian",
    "Quasi", "quasi",
    "QuasiPoisson", "quasipoisson",
    "QuasiBinomial", "quasibinomial",
    "Tweedie", "tw",
    "Scat", "scat",
    "nb", "betar", "ocat", "ziP", "cnorm",
    "GeneralFamily", "gaulss", "twlss", "shash", "gammals", "gumbls",
    "gevlss", "cox_ph", "ziplss", "multinom", "mvn",
    "LogebLink", "SoftplusLink", "BoundedLogLink", "ShiftedLogitLink",
    "trind_generator", "gamlss_etamu", "gamlss_gH", "DiscreteX",
    "IdentityLink", "LogLink", "InverseLink",
    "SqrtLink", "LogitLink", "ProbitLink", "CauchitLink", "CloglogLink",
    "InverseSquareLink", "PowerLink", "power",
]
