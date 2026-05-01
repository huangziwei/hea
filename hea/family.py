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

import numpy as np
from scipy.special import digamma, expit, gammaln, ndtr, ndtri, polygamma
from scipy.stats import binom as _binom_dist
from scipy.stats import gamma as _gamma_dist
from scipy.stats import poisson as _poisson_dist


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
    def valideta(self, eta):
        eta = np.asarray(eta)
        return bool(np.all(np.isfinite(eta)) and np.all(eta > 0))


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
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


class ProbitLink(Link):
    """``g(μ) = Φ⁻¹(μ)`` — probit binomial link."""
    name = "probit"
    def link(self, mu): return ndtri(np.asarray(mu, dtype=float))
    def linkinv(self, eta):
        # R: clamp η to ±qnorm(eps); pnorm of clamped η.
        eta = np.asarray(eta, dtype=float)
        thresh = -ndtri(np.finfo(float).eps)
        return ndtr(np.clip(eta, -thresh, thresh))
    def mu_eta(self, eta):
        # dnorm(η), lower-clamped.
        eps = np.finfo(float).eps
        return np.maximum(_dnorm(np.asarray(eta, dtype=float)), eps)
    def d2link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return eta / d ** 2
    def d3link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
        d = np.maximum(_dnorm(eta), np.finfo(float).eps)
        return (1.0 + 2.0 * eta * eta) / d ** 3
    def d4link(self, mu):
        eta = ndtri(np.asarray(mu, dtype=float))
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

    def __init__(self, link=None):
        self.link = _resolve_link(link, self.canonical_link_name)

    @property
    def is_canonical(self) -> bool:
        return self.link.name == self.canonical_link_name

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

    def postproc(self, y, mu, wt) -> dict:
        """One-shot post-fit hook for display strings. mgcv
        ``family$postproc`` rewrites ``family$family`` to e.g.
        ``"Scaled t(5,0.3)"`` reflecting fitted θ. Default: empty dict.
        """
        return {}

    def initialize(self, y, wt) -> np.ndarray:
        """Starting μ̂ for PIRLS. Return a length-n positive (or family-valid)
        vector. Default: y; subclasses override when y can be at the boundary.
        """
        return np.asarray(y, dtype=float).copy()

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

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        return wt * (y - mu) ** 2

    def aic(self, y, mu, dev, wt, n, theta=None):
        n_eff = float(np.sum(wt))
        sigma2 = dev / n_eff
        # mgcv's gaussian()$aic: n·(log(2πσ²)+1) + 2 — note the +2 is the
        # "+1 family df" placeholder; downstream adds 2·edf for the model.
        return n_eff * (np.log(2.0 * np.pi * sigma2) + 1.0) + 2.0

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
        mu = np.asarray(mu, dtype=float); return mu * mu
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float); return 2.0 * mu
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
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
            logp = _gamma_dist.logpdf(y, a=1.0 / disp, scale=mu * disp)
        return -2.0 * float(np.sum(logp * wt)) + 2.0

    def ls(self, y, wt, scale):
        # Direct port of mgcv:::fix.family.ls's Gamma branch (raw d/dφ form),
        # then a log-scale chain rule to match the hea convention:
        #   d/dlogφ  = φ · d/dφ
        #   d²/dlogφ² = φ · d/dφ + φ² · d²/dφ²
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        good = wt > 0
        y = y[good]; w = wt[good]
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
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
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
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _poisson_dist.logpmf(y, mu)
        return -2.0 * float(np.sum(logp * wt))

    def ls(self, y, wt, scale):
        # Saturated log-lik at μ=y; scale-known so d/dlogφ = d²/dlogφ² = 0.
        # mgcv: sum(dpois(y, y, log=TRUE) · w).
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _poisson_dist.logpmf(y, y)
        ls0 = float(np.sum(logp * wt))
        return np.array([ls0, 0.0, 0.0], dtype=float)


class Binomial(Family):
    """``y·m ~ Binomial(m, μ)``; ``y`` is the success proportion in [0,1],
    ``wt`` is the binomial size ``m`` (= 1 for Bernoulli).

    The cbind(success, failure) input form that R supports is *not* handled
    here — the caller must pre-convert it to (proportion, size) before
    constructing the family.
    """
    name = "binomial"
    canonical_link_name = "logit"
    scale_known = True

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float); return mu * (1.0 - mu)
    def dvar(self, mu):
        return 1.0 - 2.0 * np.asarray(mu, dtype=float)
    def d2var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), -2.0)
    def d3var(self, mu):
        return np.zeros_like(np.asarray(mu, dtype=float))

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv (C_binomial_dev_resids): 2 wt [ y_log_y(y, μ) + y_log_y(1-y, 1-μ) ]
        # where y_log_y(y, μ) = y log(y/μ) for y>0, else 0.
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)

        def yly(a, b):
            # 0·log(0/0) := 0; mask both arguments inside the log so numpy
            # doesn't evaluate log(0) on the dead branch and emit warnings.
            pos = a > 0
            safe_a = np.where(pos, a, 1.0)
            safe_b = np.where(pos, b, 1.0)
            return np.where(pos, a * np.log(safe_a / safe_b), 0.0)

        return 2.0 * wt * (yly(y, mu) + yly(1.0 - y, 1.0 - mu))

    def initialize(self, y, wt):
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError("y values must be 0 <= y <= 1 for the 'binomial' family")
        # mgcv/R: mustart = (wt·y + 0.5) / (wt + 1) keeps μ in (0,1) so the
        # logit link starts finite even when y is exactly 0 or 1.
        return (wt * y + 0.5) / (wt + 1.0)

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0) and np.all(mu < 1))

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv: m = wt; -2 Σ (wt/m) · dbinom(round(m·y), round(m), μ, log=TRUE).
        # With m = wt this collapses to -2 Σ dbinom(round(wt·y), round(wt), μ, log=TRUE).
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        m = wt
        nt = np.rint(m).astype(int)
        s = np.rint(np.where(m > 0, m * y, 0.0)).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = _binom_dist.logpmf(s, nt, mu)
        weight = np.where(m > 0, wt / np.where(m > 0, m, 1.0), 0.0)
        return -2.0 * float(np.sum(weight * logp))

    def ls(self, y, wt, scale):
        # mgcv: ls = -binomial$aic(y, n, y, w, 0) / 2; scale-known.
        ls0 = -0.5 * self.aic(y, y, 0.0, wt, None)
        return np.array([ls0, 0.0, 0.0], dtype=float)


class InverseGaussian(Family):
    """``y ~ IG(μ, φ)``; mean μ, variance φ·μ³; scale φ unknown."""
    name = "inverse.gaussian"
    canonical_link_name = "1/mu^2"
    scale_known = False

    def variance(self, mu):
        mu = np.asarray(mu, dtype=float); return mu ** 3
    def dvar(self, mu):
        mu = np.asarray(mu, dtype=float); return 3.0 * mu * mu
    def d2var(self, mu):
        return 6.0 * np.asarray(mu, dtype=float)
    def d3var(self, mu):
        return np.full_like(np.asarray(mu, dtype=float), 6.0)

    def dev_resids(self, y, mu, wt, theta=None):
        # mgcv: wt · (y - μ)² / (y · μ²).
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
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
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        sw = float(wt.sum())
        return (sw * (1.0 + np.log(dev / sw * 2.0 * np.pi))
                + 3.0 * float(np.sum(np.log(y) * wt)) + 2.0)

    def ls(self, y, wt, scale):
        # mgcv (raw φ form):
        #   ls0 = -½ · Σ log(2π φ y³) + ½ · Σ log w[w>0]
        #   d/dφ ls = -nobs/(2φ),  d²/dφ² ls = +nobs/(2φ²)
        # Chain rule to log-scale: d/dlogφ = -nobs/2, d²/dlogφ² = 0
        # (same algebraic cancellation as Gaussian — the y³ term has no φ).
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        good = wt > 0
        nobs = int(np.sum(good))
        ls0 = (-0.5 * float(np.sum(np.log(2.0 * np.pi * scale * y[good] ** 3)))
               + 0.5 * float(np.sum(np.log(wt[good]))))
        return np.array([ls0, -0.5 * nobs, 0.0], dtype=float)


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
        # R's quasi(variance='mu(1-mu)') uses (y+0.5)/2 — different from
        # Binomial's (wt·y+0.5)/(wt+1). Other variance choices match their
        # parametric counterpart's start.
        if self.variance_name == "mu(1-mu)":
            y = np.asarray(y, dtype=float)
            if np.any(y < 0) or np.any(y > 1):
                raise ValueError(
                    "y values must be 0 <= y <= 1 for quasi(variance='mu(1-mu)')"
                )
            return (y + 0.5) / 2.0
        return self._shadow.initialize(y, wt)

    def validmu(self, mu):
        return self._shadow.validmu(mu)

    def aic(self, y, mu, dev, wt, n, theta=None):
        return float("nan")

    def ls(self, y, wt, scale):
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

    Returns ``(log_a, j_bar, j_var, j_psi_bar)`` — the log of the series sum
    plus three moments of ``j`` under ``p_j = W_j/Σ W_k``: E[j], Var[j],
    and E[j·ψ(-j·α)]. The first two feed the φ-derivatives of log a; the
    third (with the digamma weight) is needed for the p-derivative — see
    Tweedie.dls_dp.
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
    return log_a, j_bar, j_var, j_psi_bar


def _tweedie_log_a_vec(y, phi, p):
    """Vectorised over y (and per-obs phi). Returns four arrays of shape
    ``y.shape``: ``log_a``, ``j_bar``, ``j_var``, ``j_psi_bar``. Entries
    with y==0 are 0 (the y=0 row uses the closed-form point mass, not the
    series). Per-obs phi handles weights via ``φ_i = φ/wt_i``.
    """
    y = np.asarray(y, dtype=float)
    phi_arr = np.broadcast_to(np.asarray(phi, dtype=float), y.shape).astype(float, copy=True)
    log_a = np.zeros_like(y)
    j_bar = np.zeros_like(y)
    j_var = np.zeros_like(y)
    j_psi_bar = np.zeros_like(y)
    flat_y = y.ravel()
    flat_phi = phi_arr.ravel()
    out_la = log_a.ravel()
    out_jb = j_bar.ravel()
    out_jv = j_var.ravel()
    out_jpb = j_psi_bar.ravel()
    for i in range(flat_y.size):
        if flat_y[i] > 0.0:
            la, jb, jv, jpb = _tweedie_log_a_one(
                float(flat_y[i]), float(flat_phi[i]), p
            )
            out_la[i] = la
            out_jb[i] = jb
            out_jv[i] = jv
            out_jpb[i] = jpb
    return log_a, j_bar, j_var, j_psi_bar


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
        # mgcv's Tweedie(): mustart = y + 0.1 — keeps log(μ) finite for y=0
        # rows under the canonical log link. Same shape as Poisson.
        return y + 0.1

    def validmu(self, mu):
        mu = np.asarray(mu)
        return bool(np.all(np.isfinite(mu)) and np.all(mu > 0))

    def _log_density(self, y, mu, phi, wt):
        """Per-obs log f(y_i; μ_i, φ/wt_i, p), shape (n,). Weight-aware via
        the per-obs scale convention φ_i = φ/w_i (matches mgcv)."""
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        good = wt > 0
        phi_i = np.where(good, float(phi) / np.where(good, wt, 1.0), 1.0)
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
            la, _, _, _ = _tweedie_log_a_vec(y[~zero], phi_i[~zero], p)
            out[~zero] = -np.log(y[~zero]) + la + cumulant[~zero] / phi_i[~zero]
        return out

    def aic(self, y, mu, dev, wt, n, theta=None):
        # mgcv's ``Tweedie()$aic``: -2·Σ wt·log f at the fitted (μ, φ̂) plus
        # +2 for the φ "extra df". φ̂ is the Pearson moment scale (matches
        # mgcv:::fix.family.aic which expects the post-fit scale).
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        wt = np.asarray(wt, dtype=float)
        n_eff = float(wt.sum())
        V = mu ** self.p
        phi = float(np.sum(wt * (y - mu) ** 2 / np.maximum(V, 1e-300))
                    / max(n_eff, 1.0))
        if not (np.isfinite(phi) and phi > 0):
            phi = max(float(dev) / max(n_eff, 1.0), 1e-12)
        log_f = self._log_density(y, mu, phi, wt)
        return -2.0 * float(np.sum(log_f * wt)) + 2.0

    def ls(self, y, wt, scale):
        """Saturated log-lik Σ w_i·log f(y_i; y_i, φ/w_i, p) and its 1st/2nd
        derivatives wrt log φ (hea log-scale convention).

        Per-obs scale ``φ_i = φ/w_i`` ⇒ d log φ_i / d log φ = 1, so the chain
        rule is trivial. For y_i = 0 with μ_i = y_i = 0 the cumulant is 0 and
        log f = 0; the entry contributes nothing to ls or its derivatives.
        For y_i > 0:

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
        phi_i = float(scale) / w_g
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
            la_, jb_, jv_, _ = _tweedie_log_a_vec(y_g[~zero], phi_i[~zero], p)
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
        phi_i = float(scale) / w_g
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
            _, jb_, _, jpb_ = _tweedie_log_a_vec(
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

    Joint estimation in ``hea.gam`` is via Brent's method on θ over the
    interior of ``(a, b)``: each Brent iterate fits the full GAM at a fixed
    candidate ``p``; the score returned is the converged REML/ML criterion
    at that ``p``. Cheaper than analytical joint outer-Newton but typically
    converges in 10-20 inner fits. The fitted ``p̂`` is stored on
    ``family.p``; the converged θ̂ on ``family.theta``.
    """
    name = "Tweedie"
    n_theta = 1

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
        nu = float(np.exp(self._theta[0]) + self._min_df)
        sig = float(np.exp(self._theta[1]))
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
        nu = float(np.exp(th[0]) + self._min_df)
        sig = float(np.exp(th[1]))
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
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
        nu = float(np.exp(th[0]) + self._min_df)
        sig = float(np.exp(th[1]))
        y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
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
        nu = float(np.exp(th[0]) + self._min_df)
        sig = float(np.exp(th[1]))
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
        y = np.asarray(y, dtype=float); wt = np.asarray(wt, dtype=float)
        nu = float(np.exp(self._theta[0]) + self._min_df)
        sig = float(np.exp(self._theta[1]))
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
        min_df = self._min_df
        th = np.asarray(theta, dtype=float)
        nu = float(np.exp(th[0]) + min_df)
        sig = float(np.exp(th[1]))
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
                             * float(np.exp(th[0])))
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

    def postproc(self, y, mu, wt) -> dict:
        # mgcv builds "Scaled t(ν, σ)" with values rounded to 3 decimals;
        # if ν > 999 it's reported as Inf.  (efam.r:3742-3749)
        nu, sig = self.get_theta(trans=True)
        nu_disp = float(np.round(nu, 3))
        sig_disp = float(np.round(sig, 3))
        if nu_disp > 999.0:
            nu_disp_str = "Inf"
        else:
            nu_disp_str = f"{nu_disp:g}"
        return {"family_name": f"Scaled t({nu_disp_str},{sig_disp:g})"}

    def rd(self, mu, wt, scale, rng: np.random.Generator | None = None):
        nu, sig = self.get_theta(trans=True)
        n = np.asarray(mu, dtype=float).shape[0]
        gen = rng if rng is not None else np.random.default_rng()
        return gen.standard_t(nu, size=n) * sig + np.asarray(mu, dtype=float)

    def __repr__(self):
        nu, sig = self.get_theta(trans=True)
        return (f"Scat(theta=({nu:.4g}, {sig:.4g}), "
                f"link={self.link.name}, min_df={self._min_df:g})")


# Convenience exports — mirror R's lowercase/CapCase convention so user code
# reads almost identically: ``gam(..., family=Gamma(link='log'))``.
gaussian = Gaussian
poisson = Poisson
binomial = Binomial
inverse_gaussian = InverseGaussian
quasi = Quasi
scat = Scat   # mgcv-style lowercase alias
__all__ = [
    "Family", "Link",
    "Gaussian", "gaussian",
    "Gamma",
    "Poisson", "poisson",
    "Binomial", "binomial",
    "InverseGaussian", "inverse_gaussian",
    "Quasi", "quasi",
    "Tweedie", "tw",
    "Scat", "scat",
    "IdentityLink", "LogLink", "InverseLink",
    "SqrtLink", "LogitLink", "ProbitLink", "CauchitLink", "CloglogLink",
    "InverseSquareLink",
]
