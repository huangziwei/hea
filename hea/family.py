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

import itertools

import numpy as np
import polars as pl
from scipy.linalg import solve_triangular
from scipy.special import digamma, expit, gamma as _gamma_fn, gammaln, logit, polygamma

from .R import nmath as _nmath
from .R.nmath import _dpois_raw, _dbinom_raw
from .R._dispatch import rs_fn as _rs_fn

# The GLM/GLMM aic hooks evaluate the saddlepoint log-density primitives
# (_dpois_raw / _dbinom_raw) on n-vectors every objective eval. Route them to the
# Rust kernels when present (bit-identical to the pure-Python ones — verified by
# the T1 parity gate — so the cumsum reduction stays bit-for-bit); the scalar
# Python path was a measured hot spot (≈16% of a cbpp glmer fit via _bd0/
# _stirlerr). See plans/rust-port-implementation.md.
_rs_dbinom_raw = _rs_fn("dbinom_raw")
_rs_dpois_raw = _rs_fn("dpois_raw")


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
    def link(self, mu): return np.log(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # mgcv clamps to .Machine$double.eps to avoid 0 — replicate so divisions
        # by μ in PIRLS / V'(μ) etc. don't blow up at extreme negative η.
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    def mu_eta(self, eta):
        return np.maximum(np.exp(np.asarray(eta, dtype=float)),
                          np.finfo(float).eps)
    def d2link(self, mu): return -1.0 / np.asarray(mu, dtype=float)**2
    def d3link(self, mu): return 2.0 / np.asarray(mu, dtype=float)**3
    def d4link(self, mu): return -6.0 / np.asarray(mu, dtype=float)**4
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
    def mu_eta(self, eta): return -1.0 / np.asarray(eta, dtype=float)**2
    def d2link(self, mu): return 2.0 / np.asarray(mu, dtype=float)**3
    def d3link(self, mu): return -6.0 / np.asarray(mu, dtype=float)**4
    def d4link(self, mu): return 24.0 / np.asarray(mu, dtype=float)**5
    # inverse link: g'=-1/μ², g''=2/μ³, g'''=-6/μ⁴, g''''=24/μ⁵ →
    # g2g = g''/g'² = (2/μ³)·μ⁴ = 2μ;  g3g = g'''/g'³ = (-6/μ⁴)·(-μ⁶) = 6μ²;
    # g4g = g''''/g'⁴ = (24/μ⁵)·μ⁸ = 24μ³.
    # mgcv gam.fit3.r:2234-2236.
    def g2g(self, mu): return 2.0 * np.asarray(mu, dtype=float)
    def g3g(self, mu): return 6.0 * np.asarray(mu, dtype=float)**2
    def g4g(self, mu): return 24.0 * np.asarray(mu, dtype=float)**3
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
    # g4g = g⁗/g′⁴ = -15·μ^-1.5.
    def g2g(self, mu): return -np.asarray(mu, dtype=float) ** -0.5
    def g3g(self, mu): return 3.0 / np.asarray(mu, dtype=float)
    def g4g(self, mu): return -15.0 * np.asarray(mu, dtype=float) ** -1.5
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
        # R clamps to (eps, 1-eps) inside C_logit_linkinv. expit is symmetric
        # around 0 and stable; the clamp is what keeps PIRLS from sliding to
        # μ=0 or 1 where V(μ) = μ(1-μ) collapses.
        eps = np.finfo(float).eps
        return np.clip(expit(np.asarray(eta, dtype=float)), eps, 1.0 - eps)
    def mu_eta(self, eta):
        # μ_η = e^η / (1+e^η)² = μ(1-μ); compute as e^{-|η|}/(1+e^{-|η|})²
        # to avoid overflow at large |η|. Lower-clamp to eps (mgcv).
        eps = np.finfo(float).eps
        a = np.exp(-np.abs(np.asarray(eta, dtype=float)))
        return np.maximum(a / (1.0 + a) ** 2, eps)
    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 1.0 / (1.0 - mu) ** 2 - 1.0 / mu ** 2
    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 / (1.0 - mu) ** 3 + 2.0 / mu ** 3
    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 6.0 / (1.0 - mu) ** 4 - 6.0 / mu ** 4


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

    def dDeta(self, y, mu, wt, theta, level: int = 0) -> dict:
        """Convert ``Dd`` (μ-space derivatives) to η-space via the link
        chain rule. Mirrors mgcv ``dDeta`` (R/efam.r). For identity link
        it copies ``Dmu → Deta``, ``Dmu2 → Deta2``, ...; for non-identity
        it applies ``Deta = Dmu · μ_η`` etc with the ``g2g``/``g3g``/
        ``g4g`` link curvature terms.

        Returns a dict with at minimum ``Deta``, ``Deta2``, ``EDeta2``
        (level 0). ``Deta.Deta2 = Dmu/(Dmu2·μ_η - Dmu·g2g)`` is the
        Newton-step working-response numerator that bam's PIRLS reads.
        """
        r = self.Dd(y, mu, theta, wt, level=level)
        link = self.link
        if link.name == "identity":
            d = {
                "Deta": r["Dmu"],
                "Deta2": r["Dmu2"],
                "EDeta2": r["EDmu2"],
                "Deta.Deta2": r["Dmu"] / r["Dmu2"],
                "Deta.EDeta2": r["Dmu"] / r["EDmu2"],
            }
            if level > 0:
                d.update({
                    "Dth": r["Dth"],
                    "Detath": r["Dmuth"],
                    "Deta3": r["Dmu3"],
                    "Deta2th": r["Dmu2th"],
                    "EDeta2th": r["EDmu2th"],
                    "EDeta3": r.get("EDmu3"),
                })
            if level > 1:
                d.update({
                    "Deta4": r["Dmu4"],
                    "Dth2": r["Dth2"],
                    "Detath2": r["Dmuth2"],
                    "Deta2th2": r["Dmu2th2"],
                    "Deta3th": r["Dmu3th"],
                })
            return d
        # Non-identity link path. mgcv ``dDeta`` expects ``link.g2g(μ)``,
        # ``g3g``, ``g4g`` to be implemented on the link object.
        ig1 = link.mu_eta(link.link(np.asarray(mu, dtype=float)))
        ig12 = ig1 * ig1
        g2g = link.g2g(mu)
        d = {
            "Deta": r["Dmu"] * ig1,
            "Deta2": r["Dmu2"] * ig12 - r["Dmu"] * g2g * ig1,
            "EDeta2": r["EDmu2"] * ig12,
        }
        d["Deta.Deta2"] = r["Dmu"] / (r["Dmu2"] * ig1 - r["Dmu"] * g2g)
        d["Deta.EDeta2"] = r["Dmu"] / (r["EDmu2"] * ig1)
        if level > 0:
            ig13 = ig12 * ig1
            d["Dth"] = r["Dth"]
            d["Detath"] = r["Dmuth"] * ig1
            g3g = link.g3g(mu)
            d["Deta3"] = (r["Dmu3"] * ig13
                          - 3.0 * r["Dmu2"] * g2g * ig12
                          + r["Dmu"] * (3.0 * g2g * g2g - g3g) * ig1)
            EDmu3 = r.get("EDmu3")
            if EDmu3 is not None:
                d["EDeta3"] = EDmu3 * ig13 - 3.0 * r["EDmu2"] * g2g * ig12
            d["Deta2th"] = r["Dmu2th"] * ig12 - r["Dmuth"] * g2g * ig1
            EDmu2th = r.get("EDmu2th")
            if EDmu2th is not None:
                d["EDeta2th"] = EDmu2th * ig12
        if level > 1:
            g4g = link.g4g(mu)
            ig14 = ig12 * ig12
            d["Deta4"] = (ig14 * r["Dmu4"]
                          - 6.0 * r["Dmu3"] * ig13 * g2g
                          + r["Dmu2"] * (15.0 * g2g * g2g - 4.0 * g3g) * ig12
                          - r["Dmu"]
                          * (15.0 * g2g ** 3 - 10.0 * g2g * g3g + g4g)
                          * ig1)
            d["Dth2"] = r["Dth2"]
            d["Detath2"] = r["Dmuth2"] * ig1
            d["Deta2th2"] = r["Dmu2th2"] * ig12 - r["Dmuth2"] * g2g * ig1
            d["Deta3th"] = (r["Dmu3th"] * ig13
                            - 3.0 * r["Dmu2th"] * g2g * ig12
                            + r["Dmuth"] * (3.0 * g2g * g2g - g3g) * ig1)
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
        # k1 = -lgamma(1/sw) - log(sw)/sw - 1/sw
        k1 = -gammaln(1.0 / sw) - np.log(sw) / sw - 1.0 / sw
        ls0 = float(np.sum(k1 - np.log(y)))
        # k2 = (digamma(1/sw) + log(sw)) / sw²       (mgcv's d/dφ)
        k2 = (digamma(1.0 / sw) + np.log(sw)) / (sw * sw)
        d1_phi = float(np.sum(k2 / w))
        # k3 = (-trigamma(1/sw)/sw + 1 - 2 log(sw) - 2 digamma(1/sw)) / sw³
        k3 = (-polygamma(1, 1.0 / sw) / sw
              + 1.0 - 2.0 * np.log(sw) - 2.0 * digamma(1.0 / sw)) / (sw ** 3)
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

    Returns ``(log_a, j_bar, j_var, j_psi_bar, j2_psi_bar, j2_psi2_bar,
    j2_trig_bar)`` — the log of the series sum plus six moments of ``j``
    under ``p_j = W_j/Σ W_k``: E[j], Var[j], E[j·ψ(-j·α)], E[j²·ψ(-j·α)],
    E[(j·ψ(-j·α))²], and E[j²·ψ′(-j·α)]. The first two feed the
    φ-derivatives of log a; E[j·ψ] the p-derivative (Tweedie.dls_dp);
    the last three the p-second-derivatives (Tweedie._d2ls_dp — tw's
    analytic ``lsth2``, family-review B4).
    """
    om1 = 1.0 - p                  # negative
    tm = 2.0 - p                   # positive
    alpha = tm / om1               # negative
    one_minus_alpha = 1.0 - alpha  # > 1; equals 1/(p-1)

    # log W_j = j·log_z - lgamma(j+1) - lgamma(-j·α).
    # Pull constants out of the j loop.
    log_z = (-alpha * np.log(y_i) + alpha * np.log(p - 1.0)
             - one_minus_alpha * np.log(phi_i) - np.log(tm))

    # Continuous-extension dominant index (Dunn-Smyth §3): with ψ(x) ≈ log x,
    # d_j log W_j = log_z - ψ(j+1) + α·ψ(-jα) ≈ 0 ⇒
    #     j*  ≈ exp((log_z + α·log(-α)) / (1-α))
    j_star = np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha)
    j_star = max(j_star, 1.0)
    j_int = max(1, int(round(j_star)))

    def _lw(j):
        return j * log_z - gammaln(j + 1.0) - gammaln(-j * alpha)

    # Walk outward from j_int both ways. Record (j, log W_j) for each kept
    # term; track the running max so log-sum-exp is numerically stable. The
    # `min_steps` guard keeps a few neighbours even when the immediate
    # neighbour is already below the eps gate (rare; happens at small j*).
    log_max = _lw(j_int)
    j_list = [float(j_int)]
    lw_list = [log_max]

    # Right tail.
    j = j_int + 1
    near = 5
    while j < _LD_J_MAX:
        v = _lw(j)
        if v - log_max < -_LD_EPS and (j - j_int) > near:
            break
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        j += 1

    # Left tail.
    j = j_int - 1
    while j >= 1:
        v = _lw(j)
        if v - log_max < -_LD_EPS and (j_int - j) > near:
            break
        j_list.append(float(j))
        lw_list.append(v)
        if v > log_max:
            log_max = v
        j -= 1

    j_arr = np.array(j_list, dtype=float)
    lw_arr = np.array(lw_list, dtype=float)
    weights = np.exp(lw_arr - log_max)
    sum_w = float(np.sum(weights))
    log_a = log_max + float(np.log(sum_w))

    p_w = weights / sum_w
    j_bar = float(np.sum(p_w * j_arr))
    j_var = float(np.sum(p_w * (j_arr - j_bar) ** 2))
    # ψ(-j·α) is well-defined for α<0, j≥1 (so -j·α > 0). We compute it on
    # the same j-grid so that the moment matches the series we just summed.
    psi_arr = digamma(-j_arr * alpha)
    j_psi_bar = float(np.sum(p_w * j_arr * psi_arr))
    j2_psi_bar = float(np.sum(p_w * j_arr * j_arr * psi_arr))
    j2_psi2_bar = float(np.sum(p_w * (j_arr * psi_arr) ** 2))
    j2_trig_bar = float(np.sum(
        p_w * j_arr * j_arr * polygamma(1, -j_arr * alpha)
    ))
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


def _tweedie_log_a_vec(y, phi, p, _chunk_bytes: int = 256 * 1024 * 1024):
    """Vectorised over y (and per-obs phi). Returns seven arrays of shape
    ``y.shape``: ``log_a``, ``j_bar``, ``j_var``, ``j_psi_bar``,
    ``j2_psi_bar``, ``j2_psi2_bar``, ``j2_trig_bar`` (the same moment
    set as :func:`_tweedie_log_a_one`). Entries with y==0 are 0 (the
    y=0 row uses the closed-form point mass, not the series). Per-obs
    phi handles weights via ``φ_i = φ/wt_i``.

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
    j2_psi_bar = np.zeros_like(y)
    j2_psi2_bar = np.zeros_like(y)
    j2_trig_bar = np.zeros_like(y)
    flat_y = y.ravel()
    flat_phi = phi_arr.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar,
                j2_psi_bar, j2_psi2_bar, j2_trig_bar)
    ya = flat_y[active]
    pha = flat_phi[active]

    om1 = 1.0 - p
    tm = 2.0 - p
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha

    log_z = (-alpha * np.log(ya) + alpha * np.log(p - 1.0)
             - one_minus_alpha * np.log(pha) - np.log(tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())

    # Series decay rate scales with |alpha|; p close to 2 (slow decay)
    # needs a wider window before the eps gate fires. Empirically
    # ``1/|alpha| + 1`` × j_int_max suffices for ``p`` up to 1.99.
    margin_mult = max(2.0, 1.0 / abs(alpha) + 1.0)
    safe_margin = max(50, int(np.ceil(margin_mult * j_int_max)) + 20)
    J = min(j_int_max + safe_margin, _LD_J_MAX)

    j_grid = np.arange(1, J + 1, dtype=float)
    j_grid_int = j_grid.astype(int)
    lgamma_jp1 = gammaln(j_grid + 1.0)
    lgamma_neg_ja = gammaln(-j_grid * alpha)
    psi_arr = digamma(-j_grid * alpha)
    trig_arr = polygamma(1, -j_grid * alpha)

    # Chunk on the n_active axis to bound the (chunk, J) working set.
    # Each row carries 5 J-wide arrays in flight (lw / 2 masks / w /
    # transient), 8 bytes each → 40 J bytes per row.
    n_active = ya.size
    chunk = max(1, _chunk_bytes // (40 * J))

    out_la = np.empty(n_active)
    out_jb = np.empty(n_active)
    out_jv = np.empty(n_active)
    out_jpb = np.empty(n_active)
    out_j2pb = np.empty(n_active)
    out_j2p2b = np.empty(n_active)
    out_j2tb = np.empty(n_active)
    near = 5
    for s in range(0, n_active, chunk):
        e = min(s + chunk, n_active)
        lz_c = log_z[s:e]
        ji_c = j_int[s:e]
        lw = (j_grid[None, :] * lz_c[:, None]
              - lgamma_jp1[None, :] - lgamma_neg_ja[None, :])
        log_max = np.max(lw, axis=1)
        within_near = np.abs(j_grid_int[None, :] - ji_c[:, None]) <= near
        above_eps = lw >= (log_max[:, None] - _LD_EPS)
        keep = within_near | above_eps
        w = np.where(keep, np.exp(lw - log_max[:, None]), 0.0)
        sum_w = np.sum(w, axis=1)
        out_la[s:e] = log_max + np.log(sum_w)
        p_w = w / sum_w[:, None]
        jb_c = np.sum(p_w * j_grid[None, :], axis=1)
        out_jb[s:e] = jb_c
        out_jv[s:e] = np.sum(
            p_w * (j_grid[None, :] - jb_c[:, None]) ** 2, axis=1,
        )
        out_jpb[s:e] = np.sum(
            p_w * j_grid[None, :] * psi_arr[None, :], axis=1,
        )
        jpsi = j_grid[None, :] * psi_arr[None, :]
        out_j2pb[s:e] = np.sum(p_w * j_grid[None, :] * jpsi, axis=1)
        out_j2p2b[s:e] = np.sum(p_w * jpsi * jpsi, axis=1)
        out_j2tb[s:e] = np.sum(
            p_w * j_grid[None, :] ** 2 * trig_arr[None, :], axis=1,
        )

    flat_la = log_a.ravel()
    flat_jb = j_bar.ravel()
    flat_jv = j_var.ravel()
    flat_jpb = j_psi_bar.ravel()
    flat_la[active] = out_la
    flat_jb[active] = out_jb
    flat_jv[active] = out_jv
    flat_jpb[active] = out_jpb
    j2_psi_bar.ravel()[active] = out_j2pb
    j2_psi2_bar.ravel()[active] = out_j2p2b
    j2_trig_bar.ravel()[active] = out_j2tb
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


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
    j2_psi_bar = np.zeros_like(y)
    j2_psi2_bar = np.zeros_like(y)
    j2_trig_bar = np.zeros_like(y)
    flat_y = y.ravel()
    active = flat_y > 0.0
    if not np.any(active):
        return (log_a, j_bar, j_var, j_psi_bar,
                j2_psi_bar, j2_psi2_bar, j2_trig_bar)
    ya = flat_y[active]
    pha = phi_arr.ravel()[active]
    pa = p_arr.ravel()[active]

    om1 = 1.0 - pa
    tm = 2.0 - pa
    alpha = tm / om1
    one_minus_alpha = 1.0 - alpha

    log_z = (-alpha * np.log(ya) + alpha * np.log(pa - 1.0)
             - one_minus_alpha * np.log(pha) - np.log(tm))
    j_star = np.maximum(
        np.exp((log_z + alpha * np.log(-alpha)) / one_minus_alpha), 1.0,
    )
    j_int = np.maximum(1, np.round(j_star).astype(int))
    j_int_max = int(j_int.max())

    # widest window needed across rows: decay slows as |alpha| shrinks
    # (p → 2), so size the shared grid from the slowest-decaying row.
    margin_mult = max(2.0, 1.0 / float(np.min(np.abs(alpha))) + 1.0)
    safe_margin = max(50, int(np.ceil(margin_mult * j_int_max)) + 20)
    J = min(j_int_max + safe_margin, _LD_J_MAX)

    j_grid = np.arange(1, J + 1, dtype=float)
    j_grid_int = j_grid.astype(int)
    lgamma_jp1 = gammaln(j_grid + 1.0)

    # per-row α ⇒ the -jα tables are (chunk, J); budget ~9 J-wide
    # doubles per row in flight → 72 J bytes per row.
    n_active = ya.size
    chunk = max(1, _chunk_bytes // (72 * J))

    out_la = np.empty(n_active)
    out_jb = np.empty(n_active)
    out_jv = np.empty(n_active)
    out_jpb = np.empty(n_active)
    out_j2pb = np.empty(n_active)
    out_j2p2b = np.empty(n_active)
    out_j2tb = np.empty(n_active)
    near = 5
    for s in range(0, n_active, chunk):
        e = min(s + chunk, n_active)
        lz_c = log_z[s:e]
        ji_c = j_int[s:e]
        nja = -j_grid[None, :] * alpha[s:e, None]      # (c, J), > 0
        lw = (j_grid[None, :] * lz_c[:, None]
              - lgamma_jp1[None, :] - gammaln(nja))
        log_max = np.max(lw, axis=1)
        within_near = np.abs(j_grid_int[None, :] - ji_c[:, None]) <= near
        above_eps = lw >= (log_max[:, None] - _LD_EPS)
        keep = within_near | above_eps
        w = np.where(keep, np.exp(lw - log_max[:, None]), 0.0)
        sum_w = np.sum(w, axis=1)
        out_la[s:e] = log_max + np.log(sum_w)
        p_w = w / sum_w[:, None]
        jb_c = np.sum(p_w * j_grid[None, :], axis=1)
        out_jb[s:e] = jb_c
        out_jv[s:e] = np.sum(
            p_w * (j_grid[None, :] - jb_c[:, None]) ** 2, axis=1,
        )
        psi_c = digamma(nja)
        out_jpb[s:e] = np.sum(p_w * j_grid[None, :] * psi_c, axis=1)
        jpsi = j_grid[None, :] * psi_c
        out_j2pb[s:e] = np.sum(p_w * j_grid[None, :] * jpsi, axis=1)
        out_j2p2b[s:e] = np.sum(p_w * jpsi * jpsi, axis=1)
        out_j2tb[s:e] = np.sum(
            p_w * j_grid[None, :] ** 2 * polygamma(1, nja), axis=1,
        )

    log_a.ravel()[active] = out_la
    j_bar.ravel()[active] = out_jb
    j_var.ravel()[active] = out_jv
    j_psi_bar.ravel()[active] = out_jpb
    j2_psi_bar.ravel()[active] = out_j2pb
    j2_psi2_bar.ravel()[active] = out_j2p2b
    j2_trig_bar.ravel()[active] = out_j2tb
    return (log_a, j_bar, j_var, j_psi_bar,
            j2_psi_bar, j2_psi2_bar, j2_trig_bar)


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
    dpth1 = eth * (b - a) / (1.0 + eth) ** 2
    dpth2 = np.where(
        pos,
        ((a - b) * eth + (b - a) * eth * eth) / (eth + 1.0) ** 3,
        ((a - b) * eth * eth + (b - a) * eth) / (eth + 1.0) ** 3,
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
        la, jb, jv, jpb, j2pb, j2p2b, j2tb = _tweedie_log_a_vec_pv(
            y_i, phii, p_i)
        al = twop / onep                      # α < 0
        alp = 1.0 / onep ** 2                 # dα/dp
        alpp = 2.0 / onep ** 3                # d²α/dp²
        lphi = np.log(phii)
        ly = np.log(y_i)
        lp1 = np.log(p_i - 1.0)
        Lp = (-alp * ly + alp * lp1 + al / (p_i - 1.0) + alp * lphi
              + 1.0 / twop)
        Lpp = (-alpp * ly + alpp * lp1 + 2.0 * alp / (p_i - 1.0)
               - al / (p_i - 1.0) ** 2 + alpp * lphi + 1.0 / twop ** 2)
        one_m_al = 1.0 - al
        cov_j_jpsi = j2pb - jb * jpb
        dla_dp = jb * Lp + alp * jpb
        d2la_dp2 = (Lp ** 2 * jv + 2.0 * Lp * alp * cov_j_jpsi
                    + alp ** 2 * (j2p2b - jpb ** 2)
                    + jb * Lpp + alpp * jpb - alp ** 2 * j2tb)
        d2la_dpdrho = (-one_m_al * (Lp * jv + alp * cov_j_jpsi)
                       + alp * jb)
        d1 = dpth1[ind]
        d2_ = dpth2[ind]
        ld[ind, 0] += la
        ld[ind, 1] += -one_m_al * jb
        ld[ind, 2] += one_m_al ** 2 * jv
        ld[ind, 3] += d1 * dla_dp
        ld[ind, 4] += d1 ** 2 * d2la_dp2 + d2_ * dla_dp
        ld[ind, 5] += d1 * d2la_dpdrho
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
        return ld.sum(axis=0)

    lds = _ld_sums(th)
    for _ in range(50):
        g = lds[[6, 3, 1]].copy()
        if np.sum(np.abs(g) > 1e-9 * abs(lds[0])) == 0:
            break
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
        while True:
            th1 = th - step
            lds1 = _ld_sums(th1)
            if lds1[0] < lds[0]:
                step = step / 2.0
            else:
                th = th1
                lds = lds1
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

    def _log_density(self, y, mu, phi):
        """Per-obs log f(y_i; μ_i, φ, p), shape (n,) — one unmodified φ for
        every row (mgcv's ``ldTweedie(y, mu, p, phi=scale)``; prior weights
        multiply the summed log-density at the call site, they never divide
        the dispersion — same convention as ``ls``)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        phi_i = np.full_like(y, float(phi))
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
            la_, jb_, jv_ = _tweedie_log_a_vec(y_g[~zero], phi_i[~zero], p)[:3]
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
            _, jb_, _, jpb_, *_rest2 = _tweedie_log_a_vec(
                y_g[~zero], phi_i[~zero], p
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
        log_phi = np.log(phi_i)

        # --- density part (μ = y) ---------------------------------------
        y_tm = y_safe ** tm
        th_y = y_tm / om1                  # θ_y·y = y^(2-p)/(1-p)
        k_y = y_tm / tm
        x_dens = (th_y * (1.0 / om1 - L) + k_y * (L - 1.0 / tm)) / phi_i
        d2p_dens = (th_y * (L * L - 2.0 * L / om1 + 2.0 / (om1 * om1))
                    - k_y * (L * L - 2.0 * L / tm + 2.0 / (tm * tm))) / phi_i
        cross_dens = -x_dens               # already in log φ form

        # --- series part -------------------------------------------------
        ap = 1.0 / (om1 * om1)             # α′
        app = 2.0 / (om1 * om1 * om1)      # α″
        inv_pm1 = 1.0 / (p - 1.0)          # 1 − α
        d2p_ser = np.zeros_like(y_g)
        cross_ser = np.zeros_like(y_g)
        if np.any(~zero):
            (_, jb, jv, jpb, j2pb, j2p2b, j2tb) = _tweedie_log_a_vec(
                y_g[~zero], phi_i[~zero], p
            )
            C = log_phi[~zero] + np.log(p - 1.0) - L[~zero] - tm
            E_jK = jb * C + jpb
            G_mean = ap * E_jK + jb / tm
            E_j2 = jv + jb * jb
            coef = ap * C + 1.0 / tm
            E_G2 = (coef * coef * E_j2 + 2.0 * coef * ap * j2pb
                    + ap * ap * j2p2b)
            var_G = E_G2 - G_mean * G_mean
            d2p_ser[~zero] = (app * E_jK
                              + ap * (inv_pm1 + 1.0) * jb
                              - ap * ap * j2tb
                              + jb / (tm * tm)
                              + var_G)
            cross_ser[~zero] = (ap * jb
                                - inv_pm1 * (coef * jv
                                             + ap * (j2pb - jpb * jb)))

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
    outer Newton (the family-generic Dd chain supplies the θ gradient;
    the Hessian θ rows are central differences of that gradient). The
    fitted ``p̂`` is stored on ``family.p``; the converged θ̂ on
    ``family.theta``.
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
        # p(θ) = (a + b·e^θ)/(1 + e^θ); use sigmoid form for stability.
        s = float(expit(theta))
        return self.a * (1.0 - s) + self.b * s

    def dp_dtheta(self) -> float:
        """``dp/dθ = (b - a)·σ(θ)·(1 - σ(θ))`` where σ is the logistic.
        Used by the outer Newton chain rule when joint-estimating θ_tw.
        """
        s = float(expit(self.theta))
        return (self.b - self.a) * s * (1.0 - s)

    def d2p_dtheta2(self) -> float:
        """``d²p/dθ² = (b-a)·σ·(1-σ)·(1 - 2σ)``."""
        s = float(expit(self.theta))
        return (self.b - self.a) * s * (1.0 - s) * (1.0 - 2.0 * s)

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

        Note: hea's outer-Newton θ rows are still central differences
        of the analytical gradient (gam.py `_reml_hessian`) — they
        don't read lsth2 yet; mgcv's `estimate.theta` Newton and any
        future analytic θ-row port do.
        """
        saved = None
        if theta is not None:
            th = np.asarray(theta, dtype=float).reshape(-1)
            if not np.allclose(th, self.get_theta()):
                saved = self.get_theta().copy()
                self.set_theta(th)
        try:
            ls3 = np.asarray(self.ls(y, wt, scale), dtype=float)
            dp1 = float(self.dp_dtheta())
            dp2 = float(self.d2p_dtheta2())
            dls_dp = float(self.dls_dp(y, wt, scale))
            dls_dth = dls_dp * dp1
            d2ls_dp2, d2ls_dpdlphi = self._d2ls_dp(y, wt, scale)
            lsth2 = np.empty((2, 2))
            lsth2[0, 0] = d2ls_dp2 * dp1 * dp1 + dls_dp * dp2
            lsth2[0, 1] = lsth2[1, 0] = d2ls_dpdlphi * dp1
            lsth2[1, 1] = float(ls3[2])
            return {
                "ls": float(ls3[0]),
                "lsth1": np.array([dls_dth, float(ls3[1])]),
                "lsth2": lsth2,
                "LSTH1": None,
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
        if th > 0:
            p = (b + a * np.exp(-th)) / (1 + np.exp(-th))
            dpth1 = np.exp(-th) * (b - a) / (1 + np.exp(-th)) ** 2
            dpth2 = (((a - b) * np.exp(-th) + (b - a) * np.exp(-2 * th))
                     / (np.exp(-th) + 1) ** 3)
        else:
            p = (b * np.exp(th) + a) / (np.exp(th) + 1)
            dpth1 = np.exp(th) * (b - a) / (np.exp(th) + 1) ** 2
            dpth2 = (((a - b) * np.exp(2 * th) + (b - a) * np.exp(th))
                     / (np.exp(th) + 1) ** 3)
        mu1p = mu ** (1 - p)
        mup = mu ** p
        r = {}
        ymupi = y / mup
        r["Dmu"] = 2 * wt * (mu1p - ymupi)
        r["Dmu2"] = 2 * wt * (mu ** (-1 - p) * p * y + (1 - p) / mup)
        r["EDmu2"] = (2 * wt) / mup
        if level > 0:
            i1p = 1 / (1 - p)
            y1 = y + (y == 0)
            logmu = np.log(mu)
            mu2p = mu * mu1p
            r["Dth"] = 2 * wt * (
                (y ** (2 - p) * np.log(y1) - mu2p * logmu) / (2 - p)
                + (y * mu1p * logmu - y ** (2 - p) * np.log(y1)) / (1 - p)
                - (y ** (2 - p) - mu2p) / (2 - p) ** 2
                + (y ** (2 - p) - y * mu1p) * i1p ** 2
            ) * dpth1
            r["Dmuth"] = 2 * wt * logmu * (ymupi - mu1p) * dpth1
            mup1 = mu ** (-p - 1)
            r["Dmu3"] = -2 * wt * mup1 * p * (y / mu * (p + 1) + 1 - p)
            r["Dmu2th"] = 2 * wt * (
                mup1 * y * (1 - p * logmu) - (logmu * (1 - p) + 1) / mup
            ) * dpth1
            r["EDmu3"] = -2 * wt * p * mup1
            r["EDmu2th"] = -2 * wt * logmu / mup * dpth1
        if level > 1:
            mup2 = mup1 / mu
            r["Dmu4"] = 2 * wt * mup2 * p * (p + 1) * (y * (p + 2) / mu + 1 - p)
            y2plogy = y ** (2 - p) * np.log(y1)
            y2plog2y = y2plogy * np.log(y1)
            r["Dth2"] = 2 * wt * (
                (mu2p * logmu ** 2 - y2plog2y) / (2 - p)
                + (y2plog2y - y * mu1p * logmu ** 2) / (1 - p)
                + 2 * (y2plogy - mu2p * logmu) / (2 - p) ** 2
                + 2 * (y * mu1p * logmu - y2plogy) / (1 - p) ** 2
                + 2 * (mu2p - y ** (2 - p)) / (2 - p) ** 3
                + 2 * (y ** (2 - p) - y * mu ** (1 - p)) / (1 - p) ** 3
            ) * dpth1 ** 2 + r["Dth"] * dpth2 / dpth1
            r["Dmuth2"] = (2 * wt * ((mu1p * logmu ** 2
                                      - logmu ** 2 * ymupi) * dpth1 ** 2)
                           + r["Dmuth"] * dpth2 / dpth1)
            r["Dmu2th2"] = (2 * wt * ((mup1 * logmu * y * (logmu * p - 2)
                            + logmu / mup * (logmu * (1 - p) + 2)) * dpth1 ** 2)
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
            r["Dmu3"] = 4.0 * wt * (yth / muth ** 3 - y / mu ** 3)
            r["Dmu2th"] = 2.0 * wt * Th * (2.0 * yth / muth - 1.0) / muth ** 2
            r["EDmu2th"] = 2.0 * wt / muth ** 2
        if level > 1:
            r["Dmu4"] = 2.0 * wt * (6.0 * y / mu ** 4
                                    - 6.0 * yth / muth ** 4)
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
            r["Dmu3th"] = 4.0 * wt * Th * (1.0 - 3.0 * yth / muth) / muth ** 3
        return r

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv nb()$aic (efam.r:239-246); `dev` is unused (Θ-form direct).
        th = self._theta if theta is None else np.asarray(theta,
                                                          dtype=float)
        Th = float(np.exp(np.asarray(th).reshape(-1)[0]))
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        term = ((y + Th) * np.log(mu + Th) - y * np.log(mu)
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
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
        ylogy = np.where(y > 0, y * np.log(np.maximum(y, 1e-300)), 0.0)
        term = ((y + Th) * np.log(y + Th) - ylogy
                + gammaln(y + 1.0) - Th * np.log(Th) + gammaln(Th)
                - gammaln(Th + y))
        ls0 = -float(np.sum(term * w))
        yth = y + Th
        lyth = np.log(yth)
        psi0_yth = digamma(yth)
        psi0_th = digamma(Th)
        term1 = Th * (lyth - psi0_yth + psi0_th - th0)
        LSTH = (-term1 * w)[:, None]
        lsth = float(np.sum(LSTH))
        psi1_yth = polygamma(1, yth)
        psi1_th = polygamma(1, Th)
        term2 = Th * (lyth - Th * psi1_yth - psi0_yth + Th / yth
                      + Th * psi1_th + psi0_th - th0 - 1.0)
        lsth2 = -float(np.sum(term2 * w))
        return {
            "ls": ls0,
            "lsth1": np.array([lsth]),
            "lsth2": np.array([[lsth2]]),
            "LSTH1": LSTH,
        }

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


def trind_generator(K: int = 2) -> dict:
    """mgcv ``trind.generator`` (gamlss.r:20-112): index arrays for
    upper-triangular packed storage of symmetric derivative arrays up to
    order 4. ``i4[i,j,k,l]`` (0-based everywhere) gives the packed column
    holding the derivative w.r.t. parameters i,j,k,l in any order;
    ``i3``/``i2`` likewise."""
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
    return {"i2": i2, "i3": i3, "i4": i4}


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


def gamlss_gH(X, jj, l1, l2, i2, l3=None, i3=None, l4=None, i4=None,
              d1b=None, d2b=None, deriv: int = 0, fh=None,
              D=None) -> dict:
    """mgcv ``gamlss.gH`` (gamlss.r:587-857), dense complete-array paths:
    coefficient-space quantities from η-space derivative arrays.

    ``jj[i]`` = LP i's column indices into X (0-based). ``deriv``:
      0 — ``lb`` (gradient) and ``lbb`` (Hessian) only;
      1 — + ``d1H`` as the vector tr(Hp⁻¹·∂H/∂ρ_l) (``fh`` must be the
          INVERSE penalized Hessian);
      2 — + ``d1H`` as the list of full ∂H/∂ρ_l matrices;
      3 — + ``trHid2H`` (``fh`` the pivoted Cholesky of the diagonally
          preconditioned Hp, ``D`` the preconditioner — gam.fit5's
          convention; or an eigendecomposition dict {values, vectors}).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    K = len(jj)
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    lb = np.zeros(p)
    for i in range(K):
        lb[jj[i]] += X[:, jj[i]].T @ l1[:, i]

    lbb = np.zeros((p, p))
    for i in range(K):
        for j in range(i, K):
            A = X[:, jj[i]].T @ (l2[:, i2[i, j]][:, None] * X[:, jj[j]])
            lbb[np.ix_(jj[i], jj[j])] += A
            if j > i:
                lbb[np.ix_(jj[j], jj[i])] += A.T

    d1H = None
    trHid2H = None
    if deriv > 0:
        l3 = np.asarray(l3, dtype=float)
        d1b = np.asarray(d1b, dtype=float)
        m = d1b.shape[1]
        # Stacked per-LP derivative of η w.r.t. each ρ (gamlss.r:680-686).
        d1eta = np.zeros((n * K, m))
        for i in range(K):
            d1eta[i * n:(i + 1) * n, :] = X[:, jj[i]] @ d1b[jj[i], :]

    if deriv == 1:
        # tr(Hp⁻¹ ∂H/∂ρ_l) accumulation (gamlss.r:735-773, dense branch);
        # fh is the inverse penalized Hessian.
        fh = np.asarray(fh, dtype=float)
        d1H = np.zeros(m)
        for i in range(K):
            for j in range(i, K):
                Hpi = fh[np.ix_(jj[i], jj[j])]
                a = np.einsum("ij,ij->i", X[:, jj[i]] @ Hpi, X[:, jj[j]])
                mult = 1.0 if i == j else 2.0
                for ll_ in range(m):
                    v = np.zeros(n)
                    for q in range(K):
                        v += l3[:, i3[i, j, q]] * d1eta[q * n:(q + 1) * n,
                                                        ll_]
                    d1H[ll_] += mult * float(np.sum(a * v))

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
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2

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
           d1b=None, d2b=None, fh=None, D=None) -> dict:
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
        mu = self.links[0].linkinv(eta)
        tau = self.links[1].linkinv(eta1)

        n = y.shape[0]
        ymu = y - mu
        ymu2 = ymu * ymu
        tau2 = tau * tau
        l0 = -0.5 * ymu2 * tau2 - 0.5 * np.log(2.0 * np.pi) + np.log(tau)
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gaulss ``initialize`` (gamlss.r:1016-1086, dense branch):
        regress g(y) on LP1's columns, then the log absolute residuals
        on LP2's, with the penalty root ``E`` as a regularizer.
        ``use_unscaled`` (mgcv's ``attr(E,"use.unscaled")``, set by
        gam.fit5 on its ldetS root): stacked least squares with E
        as-is; otherwise (initial.spg's balanced root) ``pen.reg``
        adjusts the penalty weight to an edf target."""
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
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None) -> dict:
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
        l0 = ld[:, 0]
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
        de = gamlss_etamu(l1, l2, None, None, ig1, g2, None, None,
                          tri["i2"], tri["i3"], tri["i4"], 0)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=0,
                       fh=fh, D=D)
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
           deriv: int = 0, d1b=None, d2b=None, fh=None, D=None) -> dict:
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
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
        de = gamlss_etamu(L1, L2, L3, L4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
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


class SoftplusLink(Link):
    """mgcv's bounded "log" link for the log-scale LP of the location-
    scale families ``gammals``/``gumbls`` (gamlss.r:2689-2718).

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
    :class:`SoftplusLink` (``link="log"``, σ > exp(b)) or identity.

        log f = (log y − μ − θ)/e^θ − log y − y·e^{−θ−μ} − log Γ(e^{−θ})

    where ``μ = η₁`` (log mean) and ``θ = η₂`` (log scale); the gamma
    has shape ``1/φ`` and scale ``mean·φ`` with ``φ = e^θ`` (so
    Var = mean²·φ). The fitted matrix is reported as ``(mean, log σ)``
    — :meth:`postproc` exponentiates the mean column, mirroring mgcv's
    in-place ``fitted.values[,1] <- exp(...)``.
    """
    name = "gammals"
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2

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
            SoftplusLink(b=b) if scale_link == "log" else IdentityLink(),
        ]
        self.b = float(b)
        self._scale_link_name = scale_link
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
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
        n = y.shape[0]

        l0 = etlymt - logy - ethmuy - gammaln(eth)
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gammals ``initialize`` (gamlss.r:2855-2920, dense branch):
        regress ``log(y + max(y)·eps^0.75)`` on LP1's columns, then the
        link-transformed log absolute residuals on LP2's, with ``E`` as
        regularizer (``use_unscaled`` ⇒ stacked LS, else ``pen.reg``)."""
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
    Gumbel scale) through the bounded :class:`SoftplusLink`
    (``link="log"``) or identity.

        log f = −β − z − e^{−z},   z = (y − μ)·e^{−β}

    where ``β = η₂`` is log-scale. The fitted matrix is reported as
    ``(mean, log β)`` with ``mean = μ + e^{β}·γ`` (γ = Euler's constant)
    — :meth:`postproc` adds the correction in place, mirroring mgcv's
    ``fitted.values[,1] <- ... + exp(...)·.euler``. Null deviance is NA
    (mgcv leaves it undefined for gumbls).
    """
    name = "gumbls"
    scale_known = True
    n_theta = 0
    n_lp = 2
    available_derivs = 2

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
            SoftplusLink(b=b) if scale_link == "log" else IdentityLink(),
        ]
        self.b = float(b)
        self._scale_link_name = scale_link
        self.tri = trind_generator(2)
        super().__init__(links)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
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
        l0 = -beta - z - ez
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                       i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                       fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gumbls ``initialize`` (gamlss.r:3236-3264, dense branch): two
        passes — regress y on LP1, then ``g₂(½log((y−μ̂)²) − ¼)`` on LP2,
        then re-regress ``y − 0.57721·e^{η₂}`` on LP1."""
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
    scale_known = True
    n_theta = 0
    n_lp = 3
    available_derivs = 2

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
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
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
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
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
            de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                              tri["i2"], tri["i3"], tri["i4"], deriv - 1)
            gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                           l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                           i4=tri["i4"], d1b=d1b, d2b=d2b, deriv=deriv - 1,
                           fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """gevlss ``initialize`` (gamlss.r:2378-2423, dense branch):
        regress g₁(y) on LP1, log|residuals| on LP2, then seed ξ near 0
        (``g₃(1e-3)``) and run mgcv's crude ll line-search over a
        scaling ``m`` of the ξ-start to escape the non-finite regime."""
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
    # risk sets are cumulative in DESCENDING time; sort rows internally so
    # the engine may pass them in any order (l/lb/lbb/d1H/trHid2H are all
    # coefficient-space, hence invariant to the row permutation). d1b/d2b
    # are coefficient-space and stay put.
    order = np.argsort(-time, kind="stable")
    eta = eta[order]
    X = X[order]
    time = time[order]
    d = d[order]
    gamma = np.exp(eta)
    tr = np.unique(-time)
    nt = tr.size
    r = np.searchsorted(tr, -time)                        # 0-based group
    last = np.searchsorted(r, np.arange(nt), side="right") - 1
    gamma_p = np.cumsum(gamma)[last]
    b_p = np.cumsum(gamma[:, None] * X, 0)[last]
    gXX = gamma[:, None, None] * X[:, :, None] * X[:, None, :]
    A_p = np.cumsum(gXX, 0)[last]
    ev = np.asarray(d, int) == 1
    dr = np.zeros(nt)
    np.add.at(dr, r[ev], 1.0)
    eta_sum = np.zeros(nt)
    np.add.at(eta_sum, r[ev], eta[ev])
    g_ev = X[ev].sum(0) if ev.any() else np.zeros(p)

    lpl = float(np.sum(eta_sum - dr * np.log(gamma_p)))
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
    "nb",
    "GeneralFamily", "gaulss", "twlss", "shash", "gammals", "gumbls",
    "gevlss", "cox_ph", "LogebLink", "SoftplusLink", "ShiftedLogitLink",
    "trind_generator", "gamlss_etamu", "gamlss_gH",
    "IdentityLink", "LogLink", "InverseLink",
    "SqrtLink", "LogitLink", "ProbitLink", "CauchitLink", "CloglogLink",
    "InverseSquareLink", "PowerLink", "power",
]
