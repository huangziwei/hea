"""Generalized additive model — a port of mgcv's ``gam()``.

Built on hea.formula's ``parse → expand → materialize / materialize_smooths``
pipeline: the parametric side comes from ``materialize`` (R-canonical
column names); each smooth call (``s``/``te``/``ti``/``t2``) is passed to
``materialize_smooths`` which mirrors mgcv's ``smoothCon(..., absorb.cons=
TRUE, scale.penalty=TRUE)``. Identifiability across nested smooths
(``s(x1) + te(x1, x2)``) is handled by a port of mgcv's ``gam.side`` /
``fixDependence``; rank-deficient designs are dropped to identifiability
like mgcv's ``pls_fit1`` (coef 0 / SE 0 reporting).

Fitting follows gam.fit3/gdi: an inner PIRLS loop (Fisher for canonical
links, full Newton otherwise) whose every solve is the QR of the
augmented matrix ``[√|W|·X; E]`` — ``E`` being ``gam.reparam``'s stable
penalty square root — with gdiPK's SVD correction for negative Newton
weights; an outer analytical-Newton optimization of REML/ML (default)
or GCV/UBRE over ``ρ = log λ`` (plus log φ and family θ for tw()),
seeded at ``initial.spg`` and using ``get_stableS`` for log|Sλ|₊ and
its derivatives. Smooths sharing ``id=`` are linked through mgcv's
L-matrix working parameterization.

Post-fit mirrors mgcv's reporting: Vp/Ve/Vc (sp-uncertainty corrected,
optionally ``edge_correct``-ed), edf/edf1/edf2, the testStat/reTest
smooth p-values (Wood 2013), Fletcher or exp(φ̂) scale by family class,
``predict``/``summary``/``check``/``vcomp``/plotting.

References
----------
Wood (2011), "Fast stable REML and ML estimation of semiparametric GLMs",
JRSS B 73(1), §3-4.
Wood (2013), "On p-values for smooth components of an extended
generalized additive model", Biometrika 100(1).
Wood (2017), *Generalized Additive Models* (2nd ed.), ch. 6.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.transforms import blended_transform_factory
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular
from scipy.linalg.lapack import dgeqrf, dormqr

from ..R import distributions as _dist
from ..R import nmath as _nmath
from ..family import (
    Binomial,
    DiscreteX as _DiscreteX,
    Family,
    Gaussian,
    GeneralFamily,
    Quasi,
    QuasiBinomial,
    _coerce_response,
    bcg as _bcg_family,
    clog as _clog_family,
    cnorm as _cnorm_family,
    cpois as _cpois_family,
    deterministic_xwx as _deterministic_xwx,
    gfam as _gfam_family,
    negbin as _negbin_family,
    tw as _tw_family,
)
from ..formula import (
    BasisSpec,
    Call,
    SmoothBlock,
    _eval_atom,
    _eval_lhs_expr,
    materialize_smooths,
    normalize_data,
    parse,
    prepare_design,
)
from .lm import _label_top_n, _lowess, _qq_plot
from ..utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    format_signif_jointly,
    significance_code,
)
from .._dispatch import rs_fn

# mgcv pls_fit1 (gdi.c) — the per-PIRLS-iteration penalized least-squares solve.
# The rust kernel (rust/src/linalg/pls.rs) does it as one call with a row-blocked
# parallel TSQR; the pure-Python ``_pls_qr`` below is the bit-exact oracle and
# the HEA_NO_RS fallback. ``None`` when the extension is absent.
_pls_fit1_rs = rs_fn("pls_fit1")
_PLS_EMPTY = np.empty(0, dtype=np.float64)
# Below this row count the numpy dgeqrf path is used (see ``_pls_qr``): it is the
# accuracy oracle for the small-n ill-conditioned fixtures and the rust win is
# negligible at small n. Matches the rust TSQR block threshold (n_blocks).
_PLS_RS_MIN_N = 1024

__all__ = ["gam", "gam_control"]


def gam_control(
    *,
    epsilon: float = 1e-7,
    maxit: int = 200,
    irls_reg: float = 0.0,
    rank_tol: float | None = None,
    newton: dict | None = None,
    nlm: dict | None = None,
    optim: dict | None = None,
    scale_est: str = "fletcher",
    edge_correct: bool | float = False,
    efs_lspmax: float = 15.0,
    efs_tol: float = 0.1,
    efs_maxit: int = 200,
    idLinksBases: bool = True,
    scalePenalty: bool = True,
    nthreads: int = 1,
    keepData: bool = False,
    trace: bool = False,
) -> dict:
    """mgcv's ``gam.control`` (mgcv.r:2476-2533) for ``gam(control=)``.

    Names follow the dots→underscores convention (``scale.est`` →
    ``scale_est``; the ``newton`` sublist keys are ``conv_tol``,
    ``maxNstep``, ``maxSstep``, ``maxHalf``). A plain dict
    with these keys works too — ``gam()`` revalidates through here.

    Wired knobs: ``epsilon``/``maxit`` (inner PIRLS — newton caps the
    effective ε at conv_tol/100, gam.fit3.r:1308), the ``newton`` step
    controls, ``scale_est`` ("fletcher"/"pearson"/"deviance",
    gam.fit3.r:596-606), ``edge_correct`` (was a gam() argument;
    mgcv keeps it in control), ``efs_lspmax``/``efs_tol`` (general-
    family EFS). ``efs_maxit`` caps the EFS outer loop (mgcv's
    ``efsud`` hard-codes ``for (iter in 1:200)``, gam.fit4.r:1493);
    it defaults to **200 to preserve mgcv parity** and is a hea-only
    knob for native fits of hard multi-LP families (≥2 flat shape
    directions) that genuinely need more iterations to satisfy
    ``efs_tol`` — raising it makes hea diverge from mgcv, which still
    stops at 200, so leave it at 200 for cross-engine work.
    ``rank_tol`` is accepted but the fit path overrides it
    to eps·100 exactly like mgcv (gam.fit3.r:133 — the knob only feeds
    the unported magic path); ``nthreads``/``keepData``/``trace`` are
    accepted no-ops. Unported knobs raise (never silent): non-zero
    ``irls_reg`` (performance-iteration only), ``idLinksBases=False``,
    ``scalePenalty=False``; mgcv's ``mgcv.tol``/``mgcv.half`` govern the
    performance-iteration path and are not accepted here.
    """
    if scale_est not in ("fletcher", "pearson", "deviance"):
        raise ValueError(
            "scale_est must be one of 'fletcher', 'pearson', 'deviance'; "
            f"got {scale_est!r}"
        )
    if not isinstance(edge_correct, bool) and (
        not isinstance(edge_correct, (int, float)) or edge_correct < 0
    ):
        raise ValueError("edge_correct must be logical or a positive number")
    if not (np.isscalar(epsilon) and epsilon > 0):
        raise ValueError("value of epsilon must be > 0")
    if not (np.isscalar(maxit) and maxit > 0):
        raise ValueError("maximum number of iterations must be > 0")
    if not (np.isscalar(irls_reg) and irls_reg >= 0):
        raise ValueError(
            "IRLS regularizing parameter must be a non-negative number."
        )
    if irls_reg != 0.0:
        raise NotImplementedError(
            "irls_reg is consumed only by mgcv's performance-iteration/"
            "magic path (mgcv.r:2622), which is not ported; the outer-"
            "Newton path ignores it in mgcv too."
        )
    if rank_tol is None:
        rank_tol = float(np.finfo(float).eps ** 0.5)
    elif rank_tol < 0 or rank_tol > 1:
        import warnings
        warnings.warn("silly value supplied for rank_tol: reset to "
                      "square root of machine precision.", stacklevel=2)
        rank_tol = float(np.finfo(float).eps ** 0.5)
    if idLinksBases is not True:
        raise NotImplementedError(
            "idLinksBases=False (id-linked smooths with own-data bases) "
            "is not ported; hea always uses mgcv's default pooled bases."
        )
    if scalePenalty is not True:
        raise NotImplementedError(
            "scalePenalty=FALSE is not ported; penalties are always "
            "rescaled like mgcv's default."
        )
    if efs_tol <= 0:
        efs_tol = 0.1                       # mgcv's silent reset
    if not (np.isscalar(efs_maxit) and efs_maxit > 0):
        raise ValueError("efs_maxit (EFS iteration cap) must be > 0")
    nt = dict(newton or {})
    newton_full = {
        "conv_tol": float(nt.pop("conv_tol", 1e-6)),
        "maxNstep": float(nt.pop("maxNstep", 5.0)),
        "maxSstep": float(nt.pop("maxSstep", 2.0)),
        "maxHalf": int(nt.pop("maxHalf", 30)),
    }
    if nt:
        raise ValueError(
            f"unknown newton control entries: {sorted(nt)} (accepted: "
            "conv_tol, maxNstep, maxSstep, maxHalf)"
        )
    # nlm defaults (gam.control, mgcv.r:2500-2517): ndigit from epsilon
    # (capped at the IEEE 15), gradtol = 10*epsilon, stepmax 2 (nlm
    # aborts after hitting stepmax 5 consecutive times, so not too
    # small), steptol 1e-4, iterlim 200, no analytic-derivative checks.
    nl = dict(nlm or {})
    ndigit = nl.pop("ndigit", None)
    if ndigit is None or ndigit < 2:
        ndigit = max(2, int(np.ceil(-np.log10(epsilon))))
    ndigit = int(round(ndigit))
    ndigit_max = int(np.floor(-np.log10(np.finfo(float).eps)))
    if ndigit > ndigit_max:
        ndigit = ndigit_max
    stepmax = abs(float(nl.pop("stepmax", 2.0)))
    if stepmax == 0.0:
        stepmax = 2.0
    nlm_full = {
        "ndigit": ndigit,
        "gradtol": abs(float(nl.pop("gradtol", epsilon * 10))),
        "stepmax": stepmax,
        "steptol": abs(float(nl.pop("steptol", 1e-4))),
        "iterlim": abs(int(nl.pop("iterlim", 200))),
        "check_analyticals": bool(nl.pop("check_analyticals", False)),
    }
    if nl:
        raise ValueError(
            f"unknown nlm control entries: {sorted(nl)} (accepted: "
            "ndigit, gradtol, stepmax, steptol, iterlim, "
            "check_analyticals)"
        )
    # optim defaults (gam.control, mgcv.r:2526-2528)
    om = dict(optim or {})
    optim_full = {"factr": abs(float(om.pop("factr", 1e7)))}
    if om:
        raise ValueError(
            f"unknown optim control entries: {sorted(om)} (accepted: "
            "factr)"
        )
    return {
        "epsilon": float(epsilon),
        "maxit": int(maxit),
        "irls_reg": float(irls_reg),
        "rank_tol": float(rank_tol),
        "newton": newton_full,
        "nlm": nlm_full,
        "optim": optim_full,
        "scale_est": scale_est,
        "edge_correct": edge_correct,
        "efs_lspmax": float(efs_lspmax),
        "efs_tol": float(efs_tol),
        "efs_maxit": int(efs_maxit),
        "idLinksBases": True,
        "scalePenalty": True,
        "nthreads": int(nthreads),
        "keepData": bool(keepData),
        "trace": bool(trace),
    }


# Defaults for readers that may run without an instance control dict
# (bam inherits several gam methods without running gam.__init__).
_GAM_CONTROL_DEFAULTS = gam_control()


# ---------------------------------------------------------------------------
# psum.chisq — distribution of a linear combination of chi-squared variables.
#
# Direct port of mgcv's ``psum.chisq`` (Wood, Feb 2020):
# - Davies, R. B. (1980) "The Distribution of a Linear Combination of chi^2
#   Random Variables", AS 155, JRSS-C 29, 323-333.
# - Liu, H., Tang, Y., Zhang, H. H. (2009) "A new chi-square approximation
#   to the distribution of non-negative definite quadratic forms in
#   non-central normal variables", CSDA 53, 853-856 — fallback when Davies
#   fails.
#
# mgcv exposes this as the R-callable ``psum.chisq``; it is the engine
# behind ``reTest`` (Wood 2013) and the fractional-rank correction inside
# ``testStat``.
# ---------------------------------------------------------------------------


def _ln1(x: float, first: bool) -> float:
    return math.log1p(x) if first else (math.log1p(x) - x)


def _errbd(u: float, sigsq: float, n: np.ndarray, lb: np.ndarray,
           nc: np.ndarray) -> tuple[float, float]:
    """mgcv davies::errbd — bound on tail probability, returns (errbd, cx)."""
    cx = u * sigsq
    sum1 = u * cx
    u2 = u * 2.0
    r = lb.size
    for j in range(r - 1, -1, -1):
        nj = n[j]
        lj = lb[j]
        ncj = nc[j]
        x = u2 * lj
        y = 1.0 - x
        cx += lj * (ncj / y + nj) / y
        xy = x / y
        sum1 += ncj * xy * xy + nj * (x * xy + _ln1(-x, False))
    return math.exp(-0.5 * sum1), cx


def _ctff(accx: float, upn: float, mean: float, lmin: float, lmax: float,
          sigsq: float, n: np.ndarray, lb: np.ndarray,
          nc: np.ndarray) -> tuple[float, float]:
    """mgcv davies::ctff — find ctff so Pr(qf>ctff)<accx (upn>0) or
    Pr(qf<ctff)<accx (upn<0). Returns (cutoff, upn_out)."""
    u2 = upn
    u1 = 0.0
    c1 = mean
    rb = 2.0 * lmax if u2 > 0 else 2.0 * lmin
    while True:
        eb, c2 = _errbd(u2 / (1.0 + u2 * rb), sigsq, n, lb, nc)
        if eb <= accx:
            break
        u1 = u2
        c1 = c2
        u2 *= 2.0
    while True:
        denom = c2 - mean
        if denom == 0.0:
            break
        if (c1 - mean) / denom >= 0.9:
            break
        u = (u1 + u2) * 0.5
        eb, cst = _errbd(u / (1.0 + u * rb), sigsq, n, lb, nc)
        if eb > accx:
            u1 = u
            c1 = cst
        else:
            u2 = u
            c2 = cst
    return c2, u2


def _truncation(u: float, tausq: float, sigsq: float, n: np.ndarray,
                lb: np.ndarray, nc: np.ndarray) -> float:
    """mgcv davies::truncation — bound integration error from cutoff."""
    pi = math.pi
    sum1 = 0.0
    prod2 = 0.0
    prod3 = 0.0
    s = 0
    sum2 = (sigsq + tausq) * u * u
    prod1 = 2.0 * sum2
    u = u * 2.0
    r = lb.size
    for j in range(r):
        lj = lb[j]
        ncj = nc[j]
        nj = n[j]
        x = u * lj
        x = x * x
        sum1 += ncj * x / (1.0 + x)
        if x > 1.0:
            prod2 += nj * math.log(x)
            prod3 += nj * _ln1(x, True)
            s += nj
        else:
            prod1 += nj * _ln1(x, True)
    sum1 *= 0.5
    prod2 += prod1
    prod3 += prod1
    x = math.exp(-sum1 - 0.25 * prod2) / pi
    y = math.exp(-sum1 - 0.25 * prod3) / pi
    err1 = 1.0 if s == 0 else 2.0 * x / s
    err2 = 2.5 * y if prod3 > 1.0 else 1.0
    if err2 < err1:
        err1 = err2
    x = 0.5 * sum2
    err2 = 1.0 if x <= y else y / x
    return err1 if err1 < err2 else err2


def _findu(utx: float, accx: float, sigsq: float, n: np.ndarray,
           lb: np.ndarray, nc: np.ndarray) -> float:
    """mgcv davies::findu — locate u such that truncation(u) ~ accx."""
    a = (2.0, 1.4, 1.2, 1.1)
    ut = utx
    u = ut * 0.25
    if _truncation(u, 0.0, sigsq, n, lb, nc) > accx:
        while _truncation(ut, 0.0, sigsq, n, lb, nc) > accx:
            ut *= 4.0
    else:
        ut = u
        u = u / 4.0
        while _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
            u = u / 4.0
    for ai in a:
        u = ut / ai
        if _truncation(u, 0.0, sigsq, n, lb, nc) <= accx:
            ut = u
    return ut


def _integrate(nterm: int, interv: float, tausq: float, main: bool,
               c: float, sigsq: float, n: np.ndarray, lb: np.ndarray,
               nc: np.ndarray, intl: float, ersm: float) -> tuple[float, float]:
    """mgcv davies::integrate — running update of integral and error sums."""
    pi = math.pi
    inpi = interv / pi
    r = lb.size
    for k in range(nterm, -1, -1):
        u = (k + 0.5) * interv
        sum1 = -2.0 * u * c
        sum2 = abs(sum1)
        sum3 = -0.5 * sigsq * u * u
        for j in range(r - 1, -1, -1):
            nj = n[j]
            x = 2.0 * lb[j] * u
            y = x * x
            sum3 -= 0.25 * nj * _ln1(y, True)
            y = nc[j] * x / (1.0 + y)
            z = nj * math.atan(x) + y
            sum1 += z
            sum2 += abs(z)
            sum3 += -0.5 * x * y
        x = inpi * math.exp(sum3) / u
        if not main:
            x *= (1.0 - math.exp(-0.5 * tausq * u * u))
        sum1 = math.sin(0.5 * sum1) * x
        sum2 = 0.5 * sum2 * x
        intl += sum1
        ersm += sum2
    return intl, ersm


def _cfe(x: float, th: np.ndarray, ln28: float, n: np.ndarray, lb: np.ndarray,
         nc: np.ndarray) -> tuple[float, bool]:
    """mgcv davies::cfe — coef of tausq in error from convergence factor.

    Returns (coef, fail)."""
    pi = math.pi
    axl = abs(x)
    sxl = -1 if x < 0 else 1
    sum1 = 0.0
    r = lb.size
    for j in range(r - 1, -1, -1):
        t = int(th[j])
        if lb[t] * sxl > 0.0:
            lj = abs(lb[t])
            axl1 = axl - lj * (n[t] + nc[t])
            axl2 = lj / ln28
            if axl1 > axl2:
                axl = axl1
            else:
                if axl > axl2:
                    axl = axl2
                sum1 = (axl - axl1) / lj
                for k in range(j - 1, -1, -1):
                    sum1 += n[int(th[k])] + nc[int(th[k])]
                break
    if sum1 > 100.0:
        return 1.0, True
    return (2.0 ** (sum1 * 0.25)) / (pi * axl * axl), False


def _davies(lb: np.ndarray, nc: np.ndarray, n: np.ndarray, sigma: float,
            c: float, lim: int, acc: float) -> tuple[float, int]:
    """Direct port of mgcv davies(...). Computes Pr(Q < c) where
    Q = sum_j lb[j] X_j + sigma X_0, X_j ~ chi^2_n[j](nc[j]), X_0 ~ N(0,1).

    Returns (cdf, ifault). ifault: 0=ok; 1=accuracy not met;
    2=round-off concern; 3=invalid params; 4=can't locate params.
    """
    ln28 = math.log(2.0) / 8.0
    pi = math.pi
    intl = 0.0
    ersm = 0.0
    acc1 = float(acc)

    r = lb.size
    th = np.argsort(-np.abs(lb)).astype(int)  # indices by descending |lb|

    sd = sigma * sigma
    sigsq = sd
    lmax = 0.0
    lmin = 0.0
    mean = 0.0
    for j in range(r):
        nj = n[j]
        lj = lb[j]
        ncj = nc[j]
        if nj < 0 or ncj < 0:
            return 0.0, 3
        sd += lj * lj * (2.0 * nj + 4.0 * ncj)
        mean += lj * (nj + ncj)
        if lmax < lj:
            lmax = lj
        elif lmin > lj:
            lmin = lj
    if sd == 0.0:
        return (1.0 if c > 0.0 else 0.0), 0
    if lmin == 0.0 and lmax == 0.0 and sigma == 0.0:
        return 0.0, 3

    sd = math.sqrt(sd)
    almx = -lmin if lmax < -lmin else lmax

    utx = 16.0 / sd
    up = 4.5 / sd
    un = -up

    utx = _findu(utx, 0.5 * acc1, sigsq, n, lb, nc)

    if c != 0.0 and almx > 0.07 * sd:
        cf, fail = _cfe(c, th, ln28, n, lb, nc)
        tausq = 0.25 * acc1 / cf
        if not fail:
            if _truncation(utx, tausq, sigsq, n, lb, nc) < 0.2 * acc1:
                sigsq = sigsq + tausq
                utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)

    acc1 = 0.5 * acc1

    ok = True
    while ok:
        d1, up = _ctff(acc1, up, mean, lmin, lmax, sigsq, n, lb, nc)
        d1 -= c
        if d1 < 0.0:
            return 1.0, 0
        d2_val, un = _ctff(acc1, un, mean, lmin, lmax, sigsq, n, lb, nc)
        d2 = c - d2_val
        if d2 < 0.0:
            return 0.0, 0
        intv = 2.0 * pi / (d1 if d1 > d2 else d2)
        x = utx / intv
        nt = int(math.floor(x))
        if x - nt > 0.5:
            nt += 1
        x = 3.0 / math.sqrt(acc1)
        ntm = int(math.floor(x))
        if x - ntm > 0.5:
            ntm += 1
        if nt > ntm * 1.5:
            intv1 = utx / ntm
            x = 2.0 * pi / intv1
            if x <= abs(c):
                break
            cf1, fail1 = _cfe(c - x, th, ln28, n, lb, nc)
            cf2, fail2 = _cfe(c + x, th, ln28, n, lb, nc)
            tausq = 0.33 * acc1 / (1.1 * (cf1 + cf2))
            if fail1 or fail2:
                break
            acc1 *= 0.67
            if ntm > lim:
                return 0.0, 1
            intl, ersm = _integrate(ntm, intv1, tausq, False, c, sigsq, n, lb,
                                    nc, intl, ersm)
            lim -= ntm
            sigsq = sigsq + tausq
            utx = _findu(utx, 0.25 * acc1, sigsq, n, lb, nc)
            acc1 = 0.75 * acc1
        else:
            ok = False

    if nt > lim:
        return 0.0, 1
    intl, ersm = _integrate(nt, intv, 0.0, True, c, sigsq, n, lb, nc, intl,
                            ersm)
    cdf = 0.5 - intl

    ifault = 0
    x = ersm + acc / 10.0
    j = 1
    for _ in range(4):
        if j * x == j * ersm:
            ifault = 2
        j *= 2
    return cdf, ifault


def _liu2(x: float, lb: np.ndarray, h: np.ndarray) -> float:
    """mgcv:::liu2 (Liu-Tang-Zhang 2009) survival probability fallback.

    mgcv anchor: ``liu2`` (mgcv.r:3500) — Liu et al. (2009) moment-matched
    χ² approximation to the mixture tail.
    """
    lh = lb * h
    muQ = float(lh.sum())
    lh = lh * lb
    c2 = float(lh.sum())
    lh = lh * lb
    c3 = float(lh.sum())
    if x <= 0.0 or c2 <= 0.0:
        return 1.0
    s1 = c3 / (c2 ** 1.5)
    s2 = float((lh * lb).sum()) / (c2 ** 2)
    sigQ = math.sqrt(2.0 * c2)
    t = (x - muQ) / sigQ
    if s1 * s1 > s2:
        a = 1.0 / (s1 - math.sqrt(s1 * s1 - s2))
        delta = s1 * a ** 3 - a * a
        l_df = a * a - 2.0 * delta
    else:
        a = 1.0 / s1
        delta = 0.0
        if c3 == 0.0:
            return 1.0
        l_df = c2 ** 3 / (c3 ** 2)
    muX = l_df + delta
    sigX = math.sqrt(2.0) * a
    arg = t * sigX + muX
    if delta == 0.0:
        return float(_dist.pchisq(arg, df=l_df, lower_tail=False))
    from scipy.stats import ncx2
    return float(ncx2.sf(arg, df=l_df, nc=delta))


def psum_chisq(q: float, lb: np.ndarray, df: np.ndarray | None = None,
               nc: np.ndarray | None = None, sigma: float = 0.0,
               lower_tail: bool = False, tol: float = 2e-5,
               nlim: int = 100_000) -> float:
    """Survival (or CDF) of Q = sum_j lb[j] * chi^2_{df[j]}(nc[j]) + sigma*N(0,1).

    Mirrors mgcv's ``psum.chisq(q, lb, df, nc, sigz=sigma, lower.tail=...)``.
    Returns Pr(Q > q) by default (lower_tail=False).

    Falls back on Liu-Tang-Zhang (2009) when Davies' algorithm fails to
    converge — same fallback strategy as mgcv.

    mgcv anchor: ``psum.chisq`` (mgcv.r:3466) — the Davies (1980) /
    Liu et al. (2009) mixture-of-χ² tail used for smooth-term p-values.
    """
    lb = np.ascontiguousarray(lb, dtype=float).ravel()
    r = lb.size
    if df is None:
        df = np.ones(r, dtype=int)
    else:
        df = np.asarray(df).ravel()
        if df.size == 1 and r > 1:
            df = np.repeat(df, r)
        df = np.rint(df).astype(int)
    if nc is None:
        nc = np.zeros(r, dtype=float)
    else:
        nc = np.asarray(nc, dtype=float).ravel()
        if nc.size == 1 and r > 1:
            nc = np.repeat(nc, r)
    if df.size != r or nc.size != r:
        raise ValueError("lengths of lb, df, nc must match")
    if (df < 1).any():
        raise ValueError("df must be positive integers")
    if (lb == 0).all():
        raise ValueError("at least one element of lb must be non-zero")
    sigma = max(0.0, float(sigma))

    cdf, ifault = _davies(lb, nc, df, sigma, float(q), int(nlim), float(tol))
    if ifault not in (0, 2):
        # Davies failed — fall back to Liu approximation when central.
        if (nc == 0).all():
            sf = _liu2(float(q), lb, df.astype(float))
            return sf if not lower_tail else 1.0 - sf
        return float("nan")
    sf = 1.0 - cdf
    sf = min(1.0, max(0.0, sf))
    return sf if not lower_tail else 1.0 - sf


def _r_cat_num(x: float) -> str:
    """R's ``cat(x)`` default scalar formatting: ``getOption("digits")`` = 7
    significant figures with trailing zeros dropped. Used for print.gam's
    ``total =`` and ``{method} score:`` values (which mgcv emits via ``cat``)."""
    return f"{float(x):.7g}"


def _sigfig_decimals(v: float, digits: int) -> int:
    """Decimal places in the ``digits``-significant-figure rendering of ``|v|``
    (R ``scientific()``'s ``nsig - kpower - 1``, floored at 0). ``%.*g`` already
    rounds to ``digits`` sig figs and drops trailing zeros, so its fractional
    width is exactly that count."""
    if v == 0:
        return 0
    s = f"{abs(v):.{digits}g}"
    if "e" in s or "E" in s:                       # below 1e-4 / very large
        mant, _, exp = s.partition("e")
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return max(dec - int(exp), 0)
    return len(s.split(".")[1]) if "." in s else 0


def _format_edf_vector(vals: list[float]) -> list[str]:
    """mgcv print.gam's ``format(round(edf, 4), digits = 3, scientific = FALSE)``
    (mgcv.r:2457). R's vector ``format`` picks one decimal count — the most any
    element needs to show 3 significant figures — applies it to all, and
    right-aligns to a common width. So large elements can show >3 sig figs
    (``123.456`` → ``123.5`` when a sibling forces one decimal)."""
    rounded = [round(float(v), 4) for v in vals]
    rgt = max((_sigfig_decimals(v, 3) for v in rounded), default=0)
    strs = [f"{v:.{rgt}f}" for v in rounded]
    width = max((len(s) for s in strs), default=0)
    return [s.rjust(width) for s in strs]



def _pls_qr(X: np.ndarray, lwork: dict, w: np.ndarray, z: np.ndarray,
            E_aug: np.ndarray, Xtwz: np.ndarray | None = None):
    """Penalized least-squares solve via QR of the augmented matrix —
    mgcv's ``pls_fit1`` (gdi.c): never forms X'WX, so the working
    condition number is κ([√W·X; E]) rather than its square.

        min ‖√W(z − Xβ)‖² + β'Sλβ,   E'E = Sλ

    Returns ``(beta, R_upper, log_det, ok)`` with ``R_upper`` a p×p
    triangular factor satisfying ``R'R = X'WX + Sλ`` (a drop-in
    Cholesky-factor replacement, ``lower=False``) and
    ``log_det = log|X'WX + Sλ|``.

    ``Xtwz``: pls_fit1's ``use.wy`` mode (gam.fit4.r:378-390) — the
    rhs formed directly from the finite vector ``wz = W·z`` as
    ``X'wz``, for extended-family rows where ``w ≈ 0`` makes ``z``
    itself non-finite. Algebraically ``Q₁'(√w·z) = R⁻ᵀ·X'Wz``, so it
    substitutes one triangular solve for the orthogonal projection;
    everything downstream (factor, determinant, SVD correction) is
    unchanged.

    Negative Newton weights are handled by gdiPK's SVD correction
    (gdi.c:1816-1901): with ``Q₁`` the data-rows orthogonal factor of
    ``[√|W|·X; E]`` and ``Ĩ Q₁ = U D V'`` its negative-w rows,

        X'WX + Sλ = R'V(I − 2D²)V'R,

    solved through ``(I − 2D²)⁻¹`` and refactored triangular via a
    small QR of ``(I − 2D²)^{1/2}V'R``. Any ``1 − 2d² ≤ 0`` means the
    penalized Hessian is indefinite — return ``ok=False``, mirroring
    pls_fit1's ``n<0`` signal (gam.fit3.r:341 retries the step with
    Fisher weights; gam.fit4.r:392 retries with positive weights).
    """
    pcol = X.shape[1]
    if _pls_fit1_rs is not None and X.shape[0] >= _PLS_RS_MIN_N:
        # Rust does the whole solve (TSQR + neg-weight correction) in one
        # call — see rust/src/linalg/pls.rs. Returns the same (β, Cholesky
        # factor, log-det) as the numpy path below, to the BLAS floor.
        # Gated to large n: rust's row-blocked TSQR factor is marginally
        # less accurate than LAPACK dgeqrf on *deliberately* ill-conditioned
        # designs (κ≈1e10), and those fixtures are all small-n; the per-call
        # rust win there is negligible anyway (the glue it removes is a small
        # share of a fast fit). Large-n fits — where the QR cost and the
        # per-call glue actually dominate — take the rust path.
        if Xtwz is not None:
            ok, beta, R_out, ld = _pls_fit1_rs(
                X, w, E_aug, _PLS_EMPTY, Xtwz, True)
        else:
            ok, beta, R_out, ld = _pls_fit1_rs(
                X, w, E_aug, z, _PLS_EMPTY, False)
        if not ok:
            return None, None, float("nan"), False
        return beta, R_out, ld, True
    neg = w < 0.0
    any_neg = bool(np.any(neg))
    sqw = np.sqrt(np.abs(w))
    aug = np.vstack([X * sqw[:, None], E_aug])
    # mgcv pls_fit1: factor [√|W|X; E] ONCE via LAPACK geqrf WITHOUT
    # forming Q — the economic Q is dorgqr's O(np²) cost, the dominant
    # per-PIRLS-iter term. Both the ordinary and the negative-Newton-weight
    # paths run off this single factorization. lwork is shape-stable
    # (driven by the fixed aug shape) so it is queried once and cached.
    if lwork.get("g") is None:
        _, _, wq, _ = dgeqrf(aug, lwork=-1)
        lwork["g"] = int(wq[0])
    qr_f, tau, _, info = dgeqrf(aug, lwork=lwork["g"],
                                overwrite_a=True)
    R = np.triu(qr_f[:pcol])
    diag_R = np.diag(R)
    if info != 0 or (not np.all(np.isfinite(R))) or np.any(diag_R == 0.0):
        return None, None, float("nan"), False

    if not any_neg:
        # Apply Q' to the rhs via Householder reflectors (ormqr), no Q
        # formed — exactly what pls_fit1 runs. R is byte-identical to the
        # formed-Q path; only Q'·rhs differs ~1 ulp (ormqr vs formed-Q·
        # dgemv), under the fit's BLAS floor.
        if Xtwz is not None:
            # z may be non-finite (w≈0) — use R only, project X'Wz instead.
            c = solve_triangular(R, Xtwz, lower=False, trans="T")
        else:
            b_aug = np.concatenate([sqw * z, np.zeros(E_aug.shape[0])])
            if lwork.get("o") is None:
                _, wq, _ = dormqr("L", "T", qr_f, tau, b_aug[:, None],
                                  lwork=-1)
                lwork["o"] = int(wq[0])
            cqv, _, _ = dormqr("L", "T", qr_f, tau, b_aug[:, None],
                               lwork=lwork["o"])
            c = cqv[:pcol, 0]
        beta = solve_triangular(R, c, lower=False)
        log_det = 2.0 * float(np.sum(np.log(np.abs(diag_R))))
        if not np.all(np.isfinite(beta)):
            return None, None, float("nan"), False
        # Normalize to a positive diagonal — downstream consumers
        # (the Cholesky-derivative chain in _compute_Vc2) assume the
        # factor *is* the unique Cholesky factor of R'R, not just
        # any triangular root.
        sgn = np.where(diag_R < 0, -1.0, 1.0)
        return beta, R * sgn[:, None], log_det, True
    # --- negative Newton weights: eigen determinant correction ------
    # X'WX + Sλ = R'(I − 2·IQ'IQ)R = R'V(I − 2D²)V'R, IQ = Q₁[neg]. Since
    # [√|W|X; E] = QR ⇒ Q₁ = √|W|X·R⁻¹, so IQ'IQ = R⁻ᵀ·(X[neg]'diag|w_neg|·
    # X[neg])·R⁻¹: form the p×p weighted gram and eigendecompose it
    # (V diag(d²) V') directly — never the (n+e)×p economic Q (dorgqr) nor
    # the n_neg×p IQ/SVD. The gauss/log & ig/log pathology takes this path
    # every PIRLS step with n_neg ≫ p; eigh of the p×p gram is ~2.5× the
    # SVD's speed, and U / the SVD's full_matrices padding are never used.
    # The correction V(I−2D²)⁻¹V' is invariant to eigenvector order/sign,
    # so this matches the formed-Q SVD path to ~1e-15 (the rhs uses mgcv's
    # own R⁻ᵀX'Wz use_wy-stable form, gdi.c:3132).
    A = sqw[neg][:, None] * X[neg]              # √|w_neg| · X[neg]
    Y = solve_triangular(R, A.T @ A, lower=False, trans="T")  # R⁻ᵀ·G
    Z = solve_triangular(R, Y.T, lower=False, trans="T").T    # R⁻ᵀGR⁻¹
    evals, V = np.linalg.eigh(0.5 * (Z + Z.T))
    d2 = 1.0 - 2.0 * evals                  # eigenvalues of I − 2D²
    if np.any(d2 <= 0.0):
        return None, None, float("nan"), False
    Vt = V.T
    # rhs: Q₁'·(signed √|W|z) = R⁻ᵀ·X'Wz, X'Wz = X'(w⊙z) (w signed).
    XWz = Xtwz if Xtwz is not None else X.T @ (w * z)
    c = Vt @ solve_triangular(R, XWz, lower=False, trans="T")
    beta = solve_triangular(
        R, Vt.T @ (c / d2), lower=False,
    )
    if not np.all(np.isfinite(beta)):
        return None, None, float("nan"), False
    # Triangular refactor: A = M'M, M = (I−2D²)^{1/2} V' R → QR(M),
    # normalized to the unique (positive-diagonal) Cholesky factor.
    M = np.sqrt(d2)[:, None] * (Vt @ R)
    R_corr = np.linalg.qr(M, mode="r")
    dR = np.diag(R_corr)
    if (not np.all(np.isfinite(R_corr))) or np.any(dR == 0.0):
        return None, None, float("nan"), False
    sgn = np.where(dR < 0, -1.0, 1.0)
    R_corr = R_corr * sgn[:, None]
    log_det = (2.0 * float(np.sum(np.log(np.abs(diag_R))))
               + float(np.sum(np.log(d2))))
    return beta, R_corr, log_det, True


def _s_lambda(slots, p: int, rho: np.ndarray) -> np.ndarray:
    """Assemble the p×p total penalty Sλ = Σ exp(ρᵢ)·S_i (mgcv's total
    penalty at log-sp ρ). Free-function form; ``gam._build_S_lambda`` is a

    mgcv anchor: the total penalty Σ λ_i S_i is assembled inside
    ``gam.reparam`` (gam.fit3.r:9) and gam.fit3's setup; hea factors
    that sum into this free helper.
    thin bind over the model's slots."""
    S = np.zeros((p, p))
    for rho_i, slot in zip(rho, slots):
        lam = float(np.exp(rho_i))
        a, b = slot.col_start, slot.col_end
        S[a:b, a:b] += lam * slot.S
    return S


def _reparam_eval(UrS, cache: dict, rho: np.ndarray):
    """gam.reparam at ρ (log|Sλ|+ det/det1/det2 on the fixed range space),
    memoized on ``cache`` for the repeated same-ρ hits within one outer
    evaluation. None ⇔ no reparam basis (UrS is None)."""
    if UrS is None:
        return None
    rho = np.asarray(rho, dtype=float)
    key = rho.tobytes()
    if cache.get("key") == key:
        return cache["val"]
    out = _gam_reparam(UrS, rho, deriv=2)
    cache["key"] = key
    cache["val"] = out
    return out


def _penalty_root_of(slots, p: int, UrS, reparam_Y, keep_cols,
                     cache: dict, rho: np.ndarray) -> np.ndarray:
    """Square root E (e×p) of Sλ, ``E'E = Sλ`` (mgcv's ``Sr``). Primary
    source is gam.reparam's leakage-free root mapped back to the original
    basis; falls back to an eigen root of the assembled Sλ."""
    if not slots:
        return np.zeros((0, p))
    rp = _reparam_eval(UrS, cache, rho)
    if rp is not None and reparam_Y is not None:
        E_full = rp.get("E_orig")
        if E_full is None:
            E_full = rp["E"] @ rp["Qs"].T @ reparam_Y.T
            if keep_cols is not None:
                E_full = E_full[:, keep_cols]
            rp["E_orig"] = E_full
        return E_full
    Sλ = _s_lambda(slots, p, rho)
    Sλ = 0.5 * (Sλ + Sλ.T)
    wv, V = np.linalg.eigh(Sλ)
    w_max = float(wv.max()) if wv.size else 0.0
    if w_max <= 0:
        return np.zeros((0, p))
    keep = wv > w_max * float(np.finfo(float).eps)
    return (V[:, keep] * np.sqrt(wv[keep])).T


def _gam_fit3(x, y, rho, *, slots, UrS, reparam_Y, keep_cols,
              reparam_cache, weights, start, etastart, mustart, offset,
              family, control, null_coef, binom_n, warm_eta,
              scale_fixed_value, scale_known, pls_lwork):
    """mgcv ``gam.fit3`` (gam.fit3.r:67) — penalized IRLS at log-sp ρ for
    ordinary exponential families. Free monolithic fitter; the extended
    (gam.fit4) and general (gam.fit5) dispatch lives in the caller until
    those fold in. ``null_coef`` is mgcv's get.null.coef baseline; the
    converged predictor is returned in ``eta`` for the caller's warm
    start (gam.fit3.r:1366)."""
    link = family.link
    X = x
    off = offset
    n, p = x.shape
    wt = weights
    Sλ = _s_lambda(slots, p, rho)
    Sλ = 0.5 * (Sλ + Sλ.T)
    E_aug = _penalty_root_of(slots, p, UrS, reparam_Y, keep_cols,
                             reparam_cache, rho)
    # ``eta`` here is the *offset-stripped* β-only predictor X·β; the
    # full linear predictor is ``eta + off``. Mirrors glm._irls. We
    # solve weighted LS on (z - off) ~ X to recover β each step.

    # Start μ̂ from the family's mustart (= y for Gamma/IG). The
    # *baseline* for step-halving and divergence is mgcv's ``null.coef``
    # pattern: project a constant valid η onto colspan(X) so that the
    # triple (β_null, η_null, μ_null) lives inside the family's valid
    # region for every canonical link. The plain β=0 ⇒ η=0 baseline
    # fails for canonical IG (1/μ² requires η>0 finite) — halving an
    # invalid η_new toward η_old=0 never escapes — and using the
    # saturated η as baseline gives old_pdev=0, so any positive iter-1
    # pdev would look like divergence.
    mu = (family.gam_initialize(y, wt, n=binom_n)
          if binom_n is not None
          else family.gam_initialize(y, wt))
    # User starting values (gam(start=/etastart=/mustart=)) —
    # gam.fit3.r:259-272 precedence: etastart > start > (user
    # mustart, kept past initialize). The null baseline below stays
    # user-independent like get.null.coef's mean(y) (mgcv.r:1863).
    if warm_eta is not None:
        # Warm start from the previous score-eval's converged predictor
        # (mgcv gam.fit3.r:1366) — takes precedence over the user seed,
        # which only seeds the first fit. The null baseline below stays
        # ρ-independent (from null_coef, mgcv get.null.coef).
        eta_warm = warm_eta
        eta = eta_warm - off            # β-only η
        mu = link.linkinv(eta_warm)
    elif mustart is not None:
        mu = np.asarray(mustart, dtype=float)
        if etastart is not None:
            eta_user = np.asarray(etastart, dtype=float)
            eta = eta_user - off        # R's η includes the offset
            mu = link.linkinv(eta_user)
        elif start is not None:
            eta = X @ start  # β-only η
            mu = link.linkinv(eta + off)
        else:
            eta = link.link(mu) - off    # β-only η
    elif etastart is not None:
        eta_user = np.asarray(etastart, dtype=float)
        eta = eta_user - off            # R's η includes the offset
        mu = link.linkinv(eta_user)
    elif start is not None:
        eta = X @ start     # β-only η
        mu = link.linkinv(eta + off)
    else:
        eta = link.link(mu) - off       # β-only η
    beta = np.zeros(p)

    # Null baseline coefficients (mgcv get.null.coef, passed in as
    # null_coef); η_null/μ_null follow (gam.fit3.r:283-285).
    eta_null = X @ null_coef
    mu_null = link.linkinv(eta_null + off)
    # gam.fit3.r:286-292: shrink invalid starting values toward the
    # null η (20 tries, then R's exact refusal). Only reachable with
    # user-supplied values — family mustarts are valid by design.
    if (mustart is not None
            or etastart is not None
            or start is not None):
        ii = 0
        while not (family.validmu(mu)
                   and link.valideta(eta + off)
                   and bool(np.all(np.isfinite(eta)))):
            ii += 1
            if ii > 20:
                raise ValueError("Can't find valid starting values: "
                                 "please specify some")
            eta = 0.9 * eta + 0.1 * eta_null
            mu = link.linkinv(eta + off)
    beta_old = null_coef.copy()
    eta_old = eta_null.copy()
    dev = float(np.sum(family.dev_resids(y, mu, wt)))
    # mgcv: old.pdev = sum(dev.resids at null) + null.coef' St null.coef.
    old_pdev = (float(np.sum(family.dev_resids(y, mu_null, wt)))
                + float(null_coef @ Sλ @ null_coef))

    # mgcv startup loop: if family.initialize returns a boundary value
    # (rare; e.g., Bernoulli at y=0/1 with linkinv-clamped initialize),
    # nudge η toward the null baseline until valid. Typically a no-op.
    ii = 0
    while not (link.valideta(eta + off) and family.validmu(mu)):
        ii += 1
        if ii > 20:
            raise FloatingPointError(
                "PIRLS init: cannot find valid starting μ̂"
            )
        eta = 0.9 * eta + 0.1 * eta_old
        mu = link.linkinv(eta + off)

    # gam.fit3.r:118: Fisher scoring iff the link is canonical; for
    # non-canonical links the inner loop takes full-Newton (α-weighted)
    # steps, falling back to Fisher for any step where the Newton
    # weights make the penalized Hessian indefinite (gam.fit3.r:341).
    # Both weightings solve the same penalized score equation, so β̂ is
    # invariant — but the iteration path (and step-halving behaviour)
    # follows mgcv's.
    fisher = family.is_canonical
    # control$epsilon with newton()'s conv.tol/100 cap
    # (gam.fit3.r:1308); defaults: min(1e-7, 1e-8) = 1e-8.
    ctrl = control or _GAM_CONTROL_DEFAULTS
    eps = min(ctrl["epsilon"], ctrl["newton"]["conv_tol"] / 100.0)
    max_it = ctrl["maxit"]            # gam.control(maxit = 200)
    # mgcv's pdev test scales by |scale|+|pdev|: gam.fit3's `scale`
    # argument is -1 (unknown φ), +1 (poisson/binomial), or the
    # gam(scale=)-fixed value — |scale| = 1 except under a user-
    # fixed scale.
    scale_abs = (scale_fixed_value
                 if scale_known else 1.0)
    # gam.fit3.r:227: Gaussian-identity needs exactly one penalized LS
    # solve; mgcv breaks before the pdev bookkeeping ("strictly.additive").
    strictly_additive = (
        isinstance(family, Gaussian) and link.name == "identity"
    )

    conv = False
    boundary = False
    warn_msgs: list[str] = []
    for it in range(1, max_it + 1):
        eta_full = eta + off
        mu_eta_v = link.mu_eta(eta_full)
        V = family.variance(mu)
        if np.any(np.isnan(V)):
            raise FloatingPointError("NAs in V(mu) in PIRLS")
        if np.any(V == 0):
            raise FloatingPointError("0s in V(mu) in PIRLS")
        if np.any(np.isnan(mu_eta_v)):
            raise FloatingPointError("NAs in d(mu)/d(eta) in PIRLS")
        # gam.fit3.r:308 drops zero-weight rows and rows with μ'(η)=0
        # from the working model (`good <- (weights > 0) & (mu.eta.val
        # != 0)`). Vectorized equivalent: w = 0 *and* z = 0 (so w·z is
        # 0, not 0·inf) — identical X'WX and X'Wz.
        good = (wt > 0.0) & (mu_eta_v != 0.0)
        if not np.any(good):
            warn_msgs.append(
                f"PIRLS: no informative observations at iteration {it}"
            )
            beta = beta_old
            eta = eta_old
            mu = link.linkinv(eta + off)
            break
        safe_mu_eta = np.where(good, mu_eta_v, 1.0)
        if fisher:
            z = np.where(good, eta + (y - mu) / safe_mu_eta, 0.0)
            w = np.where(good, wt * mu_eta_v ** 2 / V, 0.0)
        else:
            alpha_it = 1.0 + (y - mu) * (
                family.dvar(mu) / V + link.d2link(mu) * mu_eta_v
            )
            alpha_it = np.where(
                alpha_it == 0.0, np.finfo(float).eps, alpha_it
            )
            z = np.where(
                good, eta + (y - mu) / (safe_mu_eta * alpha_it), 0.0
            )
            w = np.where(good, wt * alpha_it * mu_eta_v ** 2 / V, 0.0)

        start, _R_it, _ld_it, ok = _pls_qr(X, pls_lwork, w, z, E_aug)
        if (not ok) and not fisher:
            # Newton weights made X'WX+Sλ indefinite — pls_fit1
            # signals this (oo$n<0) and gam.fit3 redoes the step with
            # Fisher weights (gam.fit3.r:341-352).
            z = np.where(good, eta + (y - mu) / safe_mu_eta, 0.0)
            w = np.where(good, wt * mu_eta_v ** 2 / V, 0.0)
            start, _R_it, _ld_it, ok = _pls_qr(X, pls_lwork, w, z, E_aug)
        if not ok:
            # Singularity beyond what the augmented QR survives —
            # legacy normal-equations ridge as the last resort.
            A = (X.T * w) @ X + Sλ
            A = 0.5 * (A + A.T)
            ridge = 1e-8 * np.trace(A) / p
            A_chol_r, lower_r = cho_factor(
                A + ridge * np.eye(p), lower=True, overwrite_a=False,
            )
            start = cho_solve((A_chol_r, lower_r), X.T @ (w * z))
        if np.any(~np.isfinite(start)):
            # mgcv warns and bails out of the main loop with conv=FALSE
            # (gam.fit3.r:358-363). Keep the last consistent iterate so
            # the outer Newton sees a finite (rejectable) score.
            warn_msgs.append(
                f"PIRLS: non-finite coefficients at iteration {it}"
            )
            break
        eta_new = X @ start         # β-only η
        mu_new = link.linkinv(eta_new + off)
        dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
        pen_new = float(start @ Sλ @ start)

        # "inner loop 1" (gam.fit3.r:372-396): non-finite deviance →
        # halve toward the previous iterate until finite.
        if not np.isfinite(dev_new):
            warn_msgs.append("PIRLS: step size truncated due to divergence")
            ii = 0
            while not np.isfinite(dev_new):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError(
                        "inner loop 1; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
                dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
            boundary = True
            pen_new = float(start @ Sλ @ start)

        # "inner loop 2" (gam.fit3.r:397-413): η/μ left the valid
        # region → halve toward the previous iterate.
        if not (link.valideta(eta_new + off) and family.validmu(mu_new)):
            warn_msgs.append("PIRLS: step size truncated: out of bounds")
            ii = 0
            while not (link.valideta(eta_new + off)
                       and family.validmu(mu_new)):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError(
                        "inner loop 2; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
            boundary = True
            pen_new = float(start @ Sλ @ start)
            dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))

        pdev_new = dev_new + pen_new

        # "inner loop 3" (gam.fit3.r:421-443): penalized deviance rose
        # above the divergence threshold → halve until it comes back.
        div_thresh = 10.0 * (0.1 + abs(old_pdev)) * (np.finfo(float).eps ** 0.5)
        if pdev_new - old_pdev > div_thresh:
            if it == 1:
                # Immediate divergence: shrink toward the null baseline
                # (gam.fit3.r:427-429). A no-op unless a start= arrives
                # (Tier 3): beta_old/eta_old are initialized at null.
                beta_old = null_coef.copy()
                eta_old = eta_null.copy()
            ii = 0
            while pdev_new - old_pdev > div_thresh:
                ii += 1
                if ii > 100:
                    raise FloatingPointError(
                        "inner loop 3; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
                # mgcv computes the deviance straight away (halving
                # toward the valid previous iterate); guard NaN — R
                # would error on the NaN comparison, numpy would
                # silently exit the loop and accept the step.
                if not (link.valideta(eta_new + off)
                        and family.validmu(mu_new)):
                    continue
                dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
                pen_new = float(start @ Sλ @ start)
                pdev_cand = dev_new + pen_new
                pdev_new = pdev_cand if np.isfinite(pdev_cand) else np.inf

        beta = start
        eta = eta_new
        mu = mu_new
        dev = dev_new
        pen = pen_new

        if strictly_additive:
            conv = True
            break

        # Convergence (gam.fit3.r:447-462): small pdev change, then
        # confirm with the penalized-deviance gradient at the new β —
        # the implicit differentiation in the score derivatives needs
        # the inner problem solved to gradient-level accuracy, and pdev
        # can stall one halving short of that.
        if abs(pdev_new - old_pdev) < eps * (scale_abs + abs(pdev_new)):
            grad = 2.0 * (X.T @ (w * (eta_new - z))) + 2.0 * (Sλ @ start)
            if float(np.max(np.abs(grad))) > eps * (abs(pdev_new) + scale_abs):
                old_pdev = pdev_new
                beta_old = start.copy()
                eta_old = eta_new.copy()
            else:
                conv = True
                break
        else:
            old_pdev = pdev_new
            beta_old = start.copy()
            eta_old = eta_new.copy()

    if not conv:
        warn_msgs.append("PIRLS algorithm did not converge")
    if boundary:
        warn_msgs.append("PIRLS algorithm stopped at boundary value")

    # Final consistent state (recompute w, z, alpha at converged β̂ for
    # downstream derivative routines — they expect these exact values).
    # PIRLS inner loop above used Fisher W (matches mgcv gam.fit3.r:270).
    # For the analytical score (REML / GCV) and its ρ-derivatives we use
    # the Newton-form "exact" W = α · μ_η² / V (Wood 2011). At the
    # PIRLS-converged β̂ both Fisher and Newton solve the same penalized-
    # score equation (so β̂ is invariant), but the log|X'WX + Sλ| term
    # and the chain-rule ingredients (dw/dη, d²w/dη²) depend on which
    # W enters. mgcv's score computation uses Newton W; we evaluate α
    # at the Fisher-converged β̂ here so downstream code sees Newton W.
    eta_full = eta + off
    mu_eta_v = link.mu_eta(eta_full)
    V = family.variance(mu)
    d2g = link.d2link(mu)
    # Same `good` masking as the loop (gam.fit3 recomputes the mask for
    # the derivative call): zero-weight rows and rows with μ'(η)=0 get
    # w=0, z=0.
    good = (wt > 0.0) & (mu_eta_v != 0.0)
    safe_mu_eta = np.where(good, mu_eta_v, 1.0)
    alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
    alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)
    # offset-stripped working response; w = wf·α with the Fisher part
    # wf = wt·μ'²/V carrying the prior weights (gam.fit3.r:512-515).
    z = np.where(good, eta + (y - mu) / (safe_mu_eta * alpha), 0.0)
    w = np.where(good, wt * alpha * mu_eta_v ** 2 / V, 0.0)
    # mgcv keeps the *signed* Newton weights in the score machinery —
    # gam.fit3.r:505-515 passes w = wf·α (negatives included) to gdi1,
    # which handles them via gdiPK's SVD determinant correction
    # (gdi.c:1816-1901). ``_pls_qr`` is that machinery: the factor is
    # built from the augmented QR of [√|W|·X; E] (κ, not κ²) with the
    # (I−2D²) correction for negative rows, and
    # log|X'WX+S| = 2Σlog|R_ii| + log|I−2D²| exactly. The *indefinite*
    # corner (some 1−2d² ≤ 0, where gdi clamps to a pseudo-determinant
    # basis) is still not carried through the derivative chain — that
    # rare corner falls back to Fisher weights (a deliberate residual).
    is_fisher_fallback = False
    _b_fin, R_fin, log_det_A, ok = _pls_qr(X, pls_lwork, w, z, E_aug)
    if (not ok) and np.any(w < 0):
        alpha = np.ones(n)
        z = np.where(good, eta + (y - mu) / safe_mu_eta, 0.0)
        w = np.where(good, wt * mu_eta_v ** 2 / V, 0.0)
        is_fisher_fallback = True
        _b_fin, R_fin, log_det_A, ok = _pls_qr(X, pls_lwork, w, z, E_aug)
    if ok:
        A_chol, lower = R_fin, False
    else:
        A = (X.T * w) @ X + Sλ
        A = 0.5 * (A + A.T)
        ridge = 1e-8 * np.trace(A) / p
        A_chol, lower = cho_factor(
            A + ridge * np.eye(p), lower=True, overwrite_a=False,
        )
        log_det_A = 2.0 * float(np.log(np.abs(np.diag(A_chol))).sum())

    # ``eta`` here is offset-stripped; downstream consumers
    # (linear_predictors, predict, residuals_of) expect the full
    # linear predictor — return ``eta + off``.
    eta_full = eta + off
    return _FitState(
        beta=beta, dev=dev, pen=pen,
        A_chol=A_chol, A_chol_lower=lower,
        S_full=Sλ, log_det_A=log_det_A,
        eta=eta_full, mu=mu, w=w, z=z, alpha=alpha,
        is_fisher_fallback=is_fisher_fallback,
        converged=conv, boundary=boundary, warn=warn_msgs,
        E_aug=E_aug,
    )


def _S_pinv(S_full, penalty_rank):
    """Pseudo-inverse of Sλ on its fixed range space.

    Eigendecompose Sλ and take the top ``penalty_rank`` eigenpairs,
    same convention as ``_log_det_S_pos`` so derivatives stay
    consistent with the determinant. Used by ``_reml_grad`` to
    compute ``∂log|S|+/∂ρ_k = λ_k tr(S^+ S_k)``.

    mgcv anchor: the penalty log-determinant and its ρ-derivatives are
    returned by ``gam.reparam`` (gam.fit3.r:9) as ``det``/``det1``/``det2``;
    hea recomputes them from the eigen-reduced Sλ here.
    """
    r = penalty_rank
    if r <= 0:
        return np.zeros_like(S_full)
    Sλ = 0.5 * (S_full + S_full.T)
    w, V = np.linalg.eigh(Sλ)
    order = np.argsort(w)[::-1]
    w_top = np.clip(w[order[:r]], 1e-300, None)
    V_top = V[:, order[:r]]
    return (V_top / w_top) @ V_top.T

def _make_K(A_chol, A_chol_lower, X):
    """K (n × p) such that ``K K' = X · A⁻¹ · X'`` — the n × p factor
    of the unweighted hat matrix. Mirrors mgcv's ``K = Q1`` from
    ``gdiPK`` (gdi.c:1691): the n-rows of the orthogonal factor of
    the augmented QR ``[√W X; rt(Sλ)] = Q R``, satisfying
    ``R'R = X'WX + Sλ`` and ``K K' = √W X (X'WX+Sλ)⁻¹ X' √W``.

    We don't run a separate augmented QR — we already have the
    Cholesky of ``A = X'WX + Sλ`` from PIRLS (lower form ``A = L L'``,
    upper form ``A = U' U``), and ``A⁻¹ = R⁻¹ R⁻ᵀ`` with ``R`` upper
    triangular such that ``R'R = A``. Then ``K = X · R⁻¹`` satisfies
    ``K K' = X · R⁻¹ · R⁻ᵀ · X' = X · A⁻¹ · X'``. Using ``K`` instead
    of materializing the n × n hat matrix is what mgcv does to scale
    to large n: every n-side trace/diagonal in the Hessian path
    reduces to operations on ``K`` (n × p) and ``K' D K`` (p × p)
    without ever forming the n × n product.

    Note hea's convention is *unweighted* ``P = X·A⁻¹·X'`` whereas
    mgcv's K K' is the *weighted* hat ``√W·X·A⁻¹·X'·√W``; the ``√W``
    factors are tracked explicitly at each call site rather than
    absorbed into K. The two conventions agree for canonical-link or
    Gaussian-identity (W = I).
    """
    if A_chol_lower:
        # A = L L'  →  K = X · L⁻ᵀ  →  K' = L⁻¹ · X'.
        K_T = solve_triangular(A_chol, X.T, lower=True)
    else:
        # A = U' U  →  K = X · U⁻¹  →  K' = U⁻ᵀ · X'.
        K_T = solve_triangular(A_chol, X.T, lower=False, trans="T")
    return K_T.T

def _dbeta_drho(fit, rho, slots, p):
    """Implicit-function-theorem derivative ∂β̂/∂ρ_k at PIRLS-converged β̂.

    The penalized score equation `s(β̂) = ∂ℓ/∂β |_β̂ - Sλ(ρ) β̂ = 0`
    differentiated in ρ_k gives, with H = -∂²ℓ_p/∂β∂β' = X'WX + Sλ
    (Newton-form W) at converged β̂:

        ∂β̂/∂ρ_k = -λ_k · H⁻¹ · S_k · β̂

    This holds for any family/link as long as PIRLS uses Newton weights
    (so X'WX = -∂²ℓ/∂β∂β' at β̂); for canonical links Newton ≡ Fisher
    and the formula reduces to the Gaussian case used implicitly in
    ``_reml_hessian``'s ``AinvSbeta``. Returns a (p, n_sp) array.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((p, 0))
    sp = np.exp(rho)
    out = np.empty((p, n_sp))
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        Sk_beta_full = np.zeros(p)
        Sk_beta_full[a:b] = slot.S @ fit.beta[a:b]
        Ainv_Skb = cho_solve((fit.A_chol, fit.A_chol_lower), Sk_beta_full)
        out[:, k] = -sp[k] * Ainv_Skb
    return out

def _fit_link_derivs(fit, family):
    """``(μ_eta, V, V', V'', V''', g'', g''', g'''')`` at the converged
    (μ, η), computed ONCE and cached on the fit. The Newton weight-
    derivative chain (``_dw_deta``/``_d2w_deta2``) otherwise recomputes
    these family/link derivatives independently each call though they
    depend only on the shared converged predictor — mgcv evaluates them
    once per ``gam.fit3`` (gdi1). Bit-identical to the separate calls
    (``d234link``'s g'',g''' equal ``d23link``'s for every link)."""
    c = fit._lderivs
    if c is None:
        link = family.link
        mu = fit.mu
        mu_eta = link.mu_eta(fit.eta)
        g2, g3, g4 = link.d234link(mu)
        c = fit._lderivs = (
            mu_eta, family.variance(mu), family.dvar(mu),
            family.d2var(mu), family.d3var(mu), g2, g3, g4,
        )
    return c


def _ml_logdet_adj(fit, Mp):
    """Adjustment to convert log|H+S| (REML) → log|H_pp+S_pp| (ML).

    Direct port of mgcv ``MLpenalty1`` (gdi.c:1532-1680): for ML the
    Laplace approximation marginalises only over the *range* of Sλ
    (dropping Mp null-space columns of the QR factor R before the
    log-det). For REML it uses the full Hessian.

    Identity used here (block determinant on (range, null) basis):

        log|A_pp| = log|A| + log|U_nᵀ A⁻¹ U_n|

    where U_n is an orthonormal basis for null(Sλ), Mp = dim null(Sλ).

    Returns (logdet_adj, M_inv, B) with B = A⁻¹U_n (q×Mp) and
    M = U_nᵀ B (Mp×Mp). The latter two feed the gradient correction
    in ``_dlog_det_H_drho_ml``. ``logdet_adj = log|M|`` is added to
    ``fit.log_det_A`` to obtain log|H_pp + S_pp|.
    """
    Mp = int(Mp)                 # null-space dim (a count); callers may pass float
    if Mp == 0:
        return 0.0, None, None
    # Null basis from eigendecomp of Sλ. Bottom Mp eigenvalues are
    # exactly 0 by construction (structural null space), so taking the
    # bottom-Mp eigenvectors picks out a stable U_n regardless of ρ.
    Sλ_sym = 0.5 * (fit.S_full + fit.S_full.T)
    w, V = np.linalg.eigh(Sλ_sym)
    U_n = V[:, :Mp]
    B = cho_solve((fit.A_chol, fit.A_chol_lower), U_n)
    M = U_n.T @ B
    sign, logdet_M = np.linalg.slogdet(M)
    if sign <= 0 or not np.isfinite(logdet_M):
        return 0.0, None, None
    M_inv = np.linalg.inv(M)
    return float(logdet_M), M_inv, B

def _profile_log_phi_fixed_sp(fit, log_phi0, Mp, gamma, wt, y, family, reml_ind):
    """Minimize the (RE)ML criterion over log φ at fixed ρ — the 1-D
    analogue of mgcv's newton() run when every sp is user-fixed and
    the scale is free (the lsp vector is then just [log φ],
    gam.fit3.r:121-123).

    β̂ is φ-independent, so only the φ-terms of ``_reml`` move:

        2V(φ)        = (Dp/φ − 2·ls0(φ))/γ + const(ρ)
                       − remlInd·Mp·log(2πφ)
        d2V/dlogφ    = (−Dp/φ − 2·ls1)/γ − remlInd·Mp
        d²2V/dlogφ²  = (Dp/φ − 2·ls2)/γ

    with (ls0, ls1, ls2) = family.ls in hea's log-φ convention —
    ≡ gam.fit3.r:629-631's dlr.dlphi/d2lr.d2lphi rows after the
    φ ↔ log φ chain rule. Guarded Newton: steps clamped to mgcv's
    maxNstep=5, halved while the criterion rises, gradient-step
    fallback if the φ-curvature goes non-convex (Tweedie series
    tails). Gaussian/quasi never get here — their gradient zero is
    the closed-form profile φ̂ = Dp/denom (the seed).
    """
    Dp = float(fit.dev + fit.pen)
    Mp = float(Mp)

    def score_g_h(lp: float):
        phi = float(np.exp(lp))
        ls0, ls1, ls2 = (float(v) for v in family.ls(y, wt, phi)[:3])
        v2 = ((Dp / phi - 2.0 * ls0) / gamma
              - reml_ind * Mp * float(np.log(2.0 * np.pi * phi)))
        g = (-Dp / phi - 2.0 * ls1) / gamma - reml_ind * Mp
        h = (Dp / phi - 2.0 * ls2) / gamma
        return v2, g, h

    lp = float(log_phi0)
    v2, g, h = score_g_h(lp)
    for _ in range(100):
        if abs(g) <= 1e-9 * (1.0 + abs(v2)):
            break
        step = (-g / h) if h > 0.0 else (-np.sign(g))
        step = float(np.clip(step, -5.0, 5.0))   # newton maxNstep
        lp_new = lp + step
        v2_new, g_new, h_new = score_g_h(lp_new)
        ii = 0
        while (not np.isfinite(v2_new)) or v2_new > v2:
            ii += 1
            if ii > 30:
                break
            step *= 0.5
            lp_new = lp + step
            v2_new, g_new, h_new = score_g_h(lp_new)
        if (not np.isfinite(v2_new)) or v2_new > v2:
            break                                # no improving step
        if abs(v2 - v2_new) <= 1e-12 * (1.0 + abs(v2)):
            lp, v2, g, h = lp_new, v2_new, g_new, h_new
            break
        lp, v2, g, h = lp_new, v2_new, g_new, h_new
    return lp

def _dDp_drho(fit, rho, slots):
    """∂Dp/∂ρ_k at PIRLS-converged β̂. Length-n_sp.

    Dp = -2·ℓ(β̂) + β̂'Sλ β̂ (deviance + penalty). Differentiating in ρ_k
    and applying β̂(ρ) chain rule:

        ∂Dp/∂ρ_k = (∂(-2ℓ)/∂β |_β̂) · ∂β̂/∂ρ_k
                 + 2·β̂' Sλ · ∂β̂/∂ρ_k
                 + λ_k · β̂' S_k β̂

    At convergence the penalized score is zero: -∂ℓ/∂β |_β̂ + Sλ β̂ = 0,
    i.e. ∂ℓ/∂β |_β̂ = Sλ β̂. Substituting cancels the first two terms:

        ∂Dp/∂ρ_k = λ_k · β̂' S_k β̂

    Same closed form as the Gaussian special case (`g_k` in `_reml_grad`).
    Holds for any family with PIRLS-converged β̂.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    sp = np.exp(rho)
    out = np.empty(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        beta_k = fit.beta[a:b]
        out[k] = sp[k] * float(beta_k @ slot.S @ beta_k)
    return out

# ----------------------- extra family-θ derivatives -------------------
#
# When ``family.n_theta > 0`` the outer Newton estimates extra family
# parameters jointly with (ρ, log φ). For ``tw()`` this is the single θ
# mapping to Tweedie's power p. The pieces below give the *analytical*
# gradient of the REML score wrt each extra family parameter; the outer
# Newton uses central-FD on the whole gradient for the new Hessian
# rows/cols (a small lift over the existing analytical (ρ, log φ)
# block).
#
# Notation: θ_f stands for one extra family parameter. The chain rule
# to the *physical* family quantity (e.g. p for tw) is handled inside
# the family methods (``family.dp_dtheta`` etc.); the gam-side
# derivatives below are written wrt the parameter that the family
# methods directly update on ``set_theta``.

def _db_drho(rho, beta, A_chol, A_chol_lower, slots, p):
    """Analytical ∂β/∂ρ_k = -exp(ρ_k)·A⁻¹ S_k β, returned as (p, n_sp).

    Differentiate A(ρ) β = X'y wrt ρ_k: ∂A/∂ρ_k = exp(ρ_k) S_k since
    A = X'X + Σ_k exp(ρ_k) S_k. The k-th slot's S is k×k embedded at
    its block's column range, so the RHS is non-zero only there.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    n_sp = len(slots)
    db = np.zeros((p, n_sp))
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        sp_k = float(np.exp(rho[k]))
        v = np.zeros(p)
        v[a:b] = -sp_k * (slot.S @ beta[a:b])
        db[:, k] = cho_solve((A_chol, A_chol_lower), v)
    return db


def _dlog_det_S_drho(rho, S_pinv=None, S_full=None, *, slots, p, penalty_rank):
    """∂log|Sλ|+/∂ρ_k = λ_k · tr(S⁺ S_k). Length-n_sp.

    S⁺ is the rank-stable pseudo-inverse from `_S_pinv` (top
    ``penalty_rank`` eigenpairs of Sλ). For exact-rank-stable
    scenarios this matches the existing term in `_reml_grad`.

    mgcv anchor: the penalty log-determinant and its ρ-derivatives are
    returned by ``gam.reparam`` (gam.fit3.r:9) as ``det``/``det1``/``det2``;
    hea recomputes them from the eigen-reduced Sλ here.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    if S_pinv is None:
        if S_full is None:
            S_full = _s_lambda(slots, p, rho)
        S_pinv = _S_pinv(S_full, penalty_rank)
    sp = np.exp(rho)
    out = np.empty(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        tr_SpinvSk = float(np.einsum(
            "ij,ji->", S_pinv[a:b, a:b], slot.S
        ))
        out[k] = sp[k] * tr_SpinvSk
    return out

def _d2log_det_S_drho_drho(rho, S_pinv=None, S_full=None, *, slots, p, penalty_rank):
    """∂²log|Sλ|+/∂ρ_i∂ρ_j Hessian. Shape ``(n_sp, n_sp)``.

    Identity:
        ∂²log|S|+/∂ρ_i∂ρ_j = -λ_i·λ_j·tr(S⁺·S_i·S⁺·S_j)
                            + δ_ij·λ_i·tr(S⁺·S_i)
    Formula matches the in-line version inside ``_reml_hessian``
    (line 1616-1632), exposed here so the POI optimizer in
    ``hea.bam._bgam_fit_loop`` can reuse it without re-running the
    full Hessian path. ``SpinvS_block[k]`` here has shape
    ``(p, slot_k_size)`` exactly mirroring the structure used in
    ``_reml_hessian`` so the einsum string ``"ab,ba->"`` indexes
    the right block sub-slices.

    mgcv anchor: the penalty log-determinant and its ρ-derivatives are
    returned by ``gam.reparam`` (gam.fit3.r:9) as ``det``/``det1``/``det2``;
    hea recomputes them from the eigen-reduced Sλ here.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((0, 0))
    if S_pinv is None:
        if S_full is None:
            S_full = _s_lambda(slots, p, rho)
        S_pinv = _S_pinv(S_full, penalty_rank)
    sp = np.exp(rho)
    # SpinvS_block[k] = S⁺ · slot_k.S applied on the right of S⁺ —
    # shape (p, slot_k_size). Same layout as ``_reml_hessian``.
    SpinvS_block: list[np.ndarray] = []
    tr_SpinvS = np.zeros(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        SpinvS_block.append(S_pinv[:, a:b] @ slot.S)
        tr_SpinvS[k] = float(np.einsum(
            "ij,ji->", S_pinv[a:b, a:b], slot.S
        ))
    H = np.zeros((n_sp, n_sp))
    for i in range(n_sp):
        a_i, b_i = slots[i].col_start, slots[i].col_end
        for j in range(i, n_sp):
            a_j, b_j = slots[j].col_start, slots[j].col_end
            tr_SpSiSpSj = float(np.einsum(
                "ab,ba->",
                SpinvS_block[i][a_j:b_j, :],
                SpinvS_block[j][a_i:b_i, :],
            ))
            H[i, j] = H[j, i] = -sp[i] * sp[j] * tr_SpSiSpSj
        H[i, i] += sp[i] * tr_SpinvS[i]
    return H

def _pearson_and_deriv(rho, fit, deriv=True, *, family, wt, y, slots, X, p):
    """Raw Pearson statistic ``P = Σ wᵢ(yᵢ−μᵢ)²/V(μᵢ)`` and (when
    ``deriv``) its ρ-gradient ``∂P/∂ρ`` (length n_sp), at PIRLS-
    converged β̂.

    mgcv's ``oo$P``/``oo$P1`` from ``pearson2`` (gdi.c:1207-1255). For
    the GCV/GACV path (scoreType has ``REML=0``) this is the *raw*
    statistic; the P-REML/P-ML scale is ``P/(n−Mp)`` (gdi.c:2696-2703,
    unpenalized ``i=0``) — see `_phi_pearson`. Re-derived directly from
    ``Pᵢ = w·r²/V`` (``r = y−μ``):

        dPᵢ/dη   = −(w·r/V)·(2 + r·V'(μ)/V)·μ'(η)
        ∂P/∂ρ_k  = Σᵢ (dPᵢ/dη)·(X·∂β̂/∂ρ)ᵢₖ

    β̂'s ρ-dependence uses the Newton IFT (`_dbeta_drho`), as in the
    deviance derivative `_gcv_grad_pieces` builds.
    """
    n_sp = len(slots)
    if fit.is_working_gaussian:
        # bam reduced fit: on the gaussian working problem (V ≡ 1, prior
        # w absorbed into R) the Pearson statistic IS the working RSS —
        # gam.fit3's gaussian pearson ≡ dev, and bam threads the same
        # rss.extra into both (bam.r:1270-1271), so P = fit.dev exactly.
        # Its ρ-derivative is the deviance derivative: at the normal
        # equations R'(f−Rβ̂) = Sλβ̂, so ∂P/∂ρ_k = −2·(Sλβ̂)'·∂β̂/∂ρ_k —
        # all p-space (the length-n response quantities do not exist on
        # the reduced problem).
        P = float(fit.dev)
        if not deriv or n_sp == 0:
            return P, (np.zeros(n_sp) if deriv else None)
        db_drho = _dbeta_drho(fit, rho, slots, p)          # (p, n_sp)
        Slb = fit.S_full @ fit.beta                        # (p,)
        return P, -2.0 * (Slb @ db_drho)                   # (n_sp,)
    mu = fit.mu
    V = family.variance(mu)
    r = y - mu
    P = float(np.sum(wt * r * r / V))
    if not deriv or n_sp == 0:
        return P, (np.zeros(n_sp) if deriv else None)
    Vp = family.dvar(mu)
    me = family.link.mu_eta(fit.eta)
    dP_deta = -(wt * r / V) * (2.0 + r * Vp / V) * me     # (n,)
    dP_deta = np.where(np.isfinite(dP_deta), dP_deta, 0.0)
    db_drho = _dbeta_drho(fit, rho, slots, p)                  # (p, n_sp)
    deta_drho = X @ db_drho                    # (n, n_sp)
    return P, dP_deta @ deta_drho                         # (n_sp,)


def _fisher_view(fit, *, family, family_mgcv_extended, y, wt, X, pls_lwork, n):
    """Return a Fisher-W view of a PIRLS-converged fit.

    mgcv's GCV/UBRE score and reported m$edf use the Fisher weight
    ``W_F = μ_η²/V`` (gam.fit3.r:644), while the REML log|H+S| term
    uses the Newton "exact" weight ``W_N = α·μ_η²/V`` (gdi2.c). At
    PIRLS-converged β̂ both Fisher and Newton solve the same penalized
    score equation so β̂ is invariant; only the W that multiplies X
    in ``X'WX + Sλ`` differs. This helper rebuilds the Fisher
    factorization on top of the same β̂.

    For canonical-link or Fisher-fallback fits Newton ≡ Fisher and we
    return ``fit`` unchanged. ``is_fisher_fallback=True`` is set on
    the returned view so ``_dw_deta`` / ``_d2w_deta2`` skip the α'/α
    terms (consistent with W_F not carrying an α factor).
    """
    eta = fit.eta
    mu = fit.mu
    # Canonical-link short circuit: α≡1 by canonical identity ⇒ W_F = W_N.
    if fit.is_fisher_fallback:
        return fit
    if family_mgcv_extended:
        # gam.fit4.r:564: wf = pmax(0, ½·EDeta2) — the expected
        # deviance curvature. For exponential-family deviances this
        # equals wt·μ'²/V; for scat it does not.
        dd = family.dDeta(y, mu, wt,
                          family.get_theta(), level=0)
        W_F = np.maximum(0.0, 0.5 * np.asarray(dd["EDeta2"],
                                               dtype=float))
        W_F = np.where(np.isfinite(W_F) & (wt > 0.0), W_F, 0.0)
    else:
        mu_eta = family.link.mu_eta(eta)
        V = family.variance(mu)
        # wf = wt·μ'²/V (gam.fit3.r:512/644) — prior weights included;
        # zero-weight rows stay excluded (μ'=0 rows zero out by
        # algebra).
        W_F = np.where(wt > 0.0, wt * mu_eta ** 2 / V, 0.0)
    if np.allclose(W_F, fit.w):
        return fit
    if fit.E_aug is not None:
        _b, A_F_chol, log_det_A_F, ok = _pls_qr(X, pls_lwork, 
            W_F, fit.z, fit.E_aug,
        )
        lower = False
    else:
        ok = False
    if not ok:
        sqW_F = np.sqrt(W_F)
        Xw = X * sqW_F[:, None]
        A_F = Xw.T @ Xw + fit.S_full
        A_F = 0.5 * (A_F + A_F.T)
        A_F_chol, lower = cho_factor(A_F, lower=False)
        log_det_A_F = 2.0 * float(
            np.sum(np.log(np.abs(np.diag(A_F_chol))))
        )
    return _FitState(
        beta=fit.beta, dev=fit.dev, pen=fit.pen,
        A_chol=A_F_chol, A_chol_lower=lower,
        S_full=fit.S_full, log_det_A=log_det_A_F,
        eta=eta, mu=mu, w=W_F, z=fit.z, alpha=np.ones(n),
        is_fisher_fallback=True,
        converged=fit.converged, boundary=fit.boundary, warn=fit.warn,
        E_aug=fit.E_aug,
    )

def _Dd(fit, level, *, family, y, wt):
    """Cached raw ``family.Dd`` (μ-space deviance table) at the converged
    fit. Shared by :meth:`_dDeta` (passed in as ``dd=``) and the raw-``Dd``
    consumer :meth:`_db_dtheta_fam` so the per-obs ``Dd`` is computed ONCE
    per (fit, level) — a cached higher level is a superset → serves
    lower-level reads. Pure function of the converged fit ⇒ 0-ulp. (``Dd``
    takes ``(y, mu, θ, wt)`` — note the order differs from ``dDeta``.)
    Keyed on the EXACT level (a higher level is NOT a safe substitute — e.g.
    ziP's ``Dd`` omits ``EDmu2th`` and has level-dependent keys), which still

    mgcv anchor: the family's ``Dd``/``dDeta`` deviance-derivative
    method (efam.r), consumed by ``gam.fit4``'s gdi block (gam.fit4.r).
    removes the dominant same-level repeats."""
    cache = fit._ddraw
    if cache is None:
        cache = fit._ddraw = {}
    elif cache.get(level) is not None:
        return cache[level]
    res = family.Dd(y, fit.mu, family.get_theta(),
                         wt, level=level)
    cache[level] = res
    return res


def _dDeta(fit, level, *, family, y, wt):
    """Cached ``family.dDeta`` at the converged fit. The REML grad/Hessian,
    ``db.drho``/``d2b.drho`` and ``_dw_deta``/``_d2w_deta2`` each call
    ``dDeta`` with the SAME ``(y, fit.mu, wt, θ)`` — recomputing the per-obs
    ``Dd`` table (the extended-family #1 cost). Profiling a tw fit found 62%
    of the ``Dd`` calls were redundant repeats; memoising on the FitState by
    level removes them (a cached higher level is a superset → serves
    lower-level reads). Computes via the shared raw-``Dd`` cache so a single
    ``Dd`` feeds both the η-transform and raw-``Dd`` consumers. Pure
    function of the converged fit ⇒ 0-ulp; the gam.fit4 analog of the
    ``_dwdeta``/``_lderivs`` caches. Keyed on the EXACT level (see

    mgcv anchor: the family's ``Dd``/``dDeta`` deviance-derivative
    method (efam.r), consumed by ``gam.fit4``'s gdi block (gam.fit4.r).
    :meth:`_Dd` — a higher level is not a safe substitute)."""
    cache = fit._ddeta
    if cache is None:
        cache = fit._ddeta = {}
    elif cache.get(level) is not None:
        return cache[level]
    res = family.dDeta(y, fit.mu, wt,
                            family.get_theta(), level,
                            dd=_Dd(fit, level, family=family, y=y, wt=wt))
    cache[level] = res
    return res


def _dw_deta(fit, *, y, family_mgcv_extended, family, wt):
    """∂w_i/∂η_i at PIRLS-converged β̂. Length-n.

    PIRLS Newton weights are w(μ) = α(μ)·μ_eta(μ)²/V(μ) with
    α(μ) = 1 + (y-μ)·B(μ), B(μ) = V'/V + g''·μ_eta. Differentiating:

        ∂(log w)/∂μ = α'/α − 2·g''·μ_eta − V'/V
        α'(μ)       = −B(μ) + (y-μ)·B'(μ)
        B'(μ)       = V''/V − (V'/V)² + g'''·μ_eta − (g'')²·μ_eta²

    and dw/dη = (dw/dμ)·μ_eta = w·μ_eta·∂(log w)/∂μ.

    For canonical links the Newton form gives α≡1 (B≡0 by canonical
    identity g'V=1), so α'/α=0 and only the (-2·g''·μ_eta − V'/V)
    terms survive — that's the Fisher derivative. For
    ``fit.is_fisher_fallback`` we explicitly drop the α'/α term to
    stay consistent with the α=1 override the PIRLS path applied.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    # ∂w/∂η depends only on the converged fit; the REML grad/Hessian +
    # db.drho/d2b.drho call this several times per fit (and the extended
    # branch's dDeta(level=1) is a 9-output Dd — the tw #1 cost). Cache on
    # the fit, like _fit_link_derivs (W3.3d part 2, here for gam.fit4).
    if fit._dwdeta is not None:
        return fit._dwdeta
    if fit.is_working_gaussian:
        # bam reduced fit: the working family is gaussian-identity
        # (bam.r:932), so dW/dη ≡ 0 exactly. Length-p (= len(beta), the
        # compressed row count) so consumers broadcasting against the
        # p×p reduced design line up — same rationale as bam's
        # `_dw_deta` method override.
        res = np.zeros(len(fit.beta))
        fit._dwdeta = res
        return res
    mu = fit.mu
    w = fit.w
    alpha = fit.alpha

    # Extended families: w = ½·∂²D/∂η² so ∂w/∂η = ½·Deta3 directly
    # from the family's Dd tables (gam.fit4/gdi2 convention). Rows
    # dropped from the working model (stored w == 0) contribute 0.
    if family_mgcv_extended:
        dd = _dDeta(fit, 1, family=family, y=y, wt=wt)
        d3 = 0.5 * np.asarray(dd["Deta3"], dtype=float)
        res = np.where((w != 0.0) & np.isfinite(d3), d3, 0.0)
        fit._dwdeta = res
        return res

    mu_eta, V, Vp, Vpp, _Vppp, g2, g3, _g4 = _fit_link_derivs(fit, family)

    # α'/α term — set to zero for the Fisher fallback path.
    if fit.is_fisher_fallback:
        alpha_prime_over_alpha = np.zeros_like(mu)
    else:
        B = Vp / V + g2 * mu_eta
        Bp = Vpp / V - (Vp / V) ** 2 + g3 * mu_eta - g2 ** 2 * mu_eta ** 2
        alpha_prime = -B + (y - mu) * Bp
        alpha_prime_over_alpha = alpha_prime / alpha

    dlogw_dmu = alpha_prime_over_alpha - 2.0 * g2 * mu_eta - Vp / V
    res = w * mu_eta * dlogw_dmu
    fit._dwdeta = res
    return res

def _d2beta_drho_drho(fit, rho, db_drho=None, dw_deta=None, *, slots, p, X, y, family_mgcv_extended, family, wt):
    """∂²β̂/∂ρ_l∂ρ_k at PIRLS-converged β̂. Returns a (p, n_sp, n_sp) array.

    Differentiating dβ_k = -λ_k·H⁻¹·S_k·β̂ in ρ_l and using the IFT
    identity ∂H⁻¹/∂ρ_l = -H⁻¹·(∂H/∂ρ_l)·H⁻¹:

        ∂²β̂/∂ρ_l∂ρ_k = δ_lk · dβ_k
                      − H⁻¹ · (∂H/∂ρ_l) · dβ_k
                      − λ_k · H⁻¹ · S_k · dβ_l

    with ∂H/∂ρ_l = X'·diag(h'·v_l)·X + λ_l·S_l (v_l := X·dβ_l).
    Symmetric in (l, k) by construction of the formula:
        ∂²β̂/∂ρ_l∂ρ_k = δ_lk·dβ_k
                      − H⁻¹·X'·(h' · v_l · v_k)
                      − λ_l · H⁻¹·S_l·dβ_k
                      − λ_k · H⁻¹·S_k·dβ_l
    — the two S terms swap when (l, k) swap; the X'·(h'·v_l·v_k) term
    is invariant under the swap. Symmetry is exploited in the loop.

    For Gaussian-identity, h' ≡ 0 so the W-derivative term drops and
    the result reduces to the standard penalty-only IFT formula.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((p, 0, 0))
    if db_drho is None:
        db_drho = _dbeta_drho(fit, rho, slots, p)
    sp = np.exp(rho)
    v = X @ db_drho                     # (n, n_sp): v_l = X·dβ_l

    # h'(η) — only present for PIRLS fits (fit.w not None). Gaussian fast
    # path doesn't reach this method.
    if dw_deta is None:
        dw_deta = _dw_deta(fit, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)

    # Per-slot S_k·dβ_k[a:b] in the embedded p-vector, stored once.
    Skdb_full = np.zeros((n_sp, p, n_sp))
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        for ll in range(n_sp):
            Skdb_full[k, a:b, ll] = slot.S @ db_drho[a:b, ll]

    out = np.empty((p, n_sp, n_sp))
    for k in range(n_sp):
        for m in range(k, n_sp):
            # H⁻¹·X'·(h' · v_l · v_k)  — the W-deriv contribution.
            rhs_W = X.T @ (dw_deta * v[:, m] * v[:, k])
            # H⁻¹·S_l·dβ_k (full p-vector, only nonzero at slot l's range)
            # and H⁻¹·S_k·dβ_l, embedded already in Skdb_full.
            rhs = (
                rhs_W
                + sp[m] * Skdb_full[m, :, k]
                + sp[k] * Skdb_full[k, :, m]
            )
            # The implicit-function-theorem formula above:
            #   ∂²β̂/∂ρ_l∂ρ_k = δ_lk·dβ_k − H⁻¹·rhs_combined
            d2 = -cho_solve(
                (fit.A_chol, fit.A_chol_lower), rhs
            )
            if m == k:
                d2 = d2 + db_drho[:, k]
            out[:, m, k] = d2
            if m != k:
                out[:, k, m] = d2
    return out

def _d2w_deta2(fit, *, y, family_mgcv_extended, family, wt):
    """∂²w_i/∂η_i² at PIRLS-converged β̂. Length-n.

    Differentiating h(η) := w(η) twice (with y, ρ fixed; only η varies):

        d log h / dη   = μ_eta · D                where D = α'/α − 2 g'' μ_eta − V'/V
        d²h/dη²        = h · μ_eta² · (D² + D' − D · g'' · μ_eta)

    with D' = ∂D/∂μ:

        D' = α''/α − (α'/α)² − 2 g''' μ_eta + 2 (g'')² μ_eta² − V''/V + (V'/V)²
        α''(μ) = −2 B' + (y−μ) · B''
        B''(μ) = V'''/V − 3 V'·V''/V² + 2 V'³/V³
                 + g'''' μ_eta − 3 g'' g''' μ_eta² + 2 (g'')³ μ_eta³

    For the Fisher fallback path (PIRLS forced α=1 because Newton-w<0),
    α'/α and α''/α are both dropped — same convention as ``_dw_deta``.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    if fit._d2wdeta2 is not None:
        return fit._d2wdeta2
    if fit.is_working_gaussian:
        # bam reduced fit — gaussian working family: ∂²w/∂η² ≡ 0,
        # length-p (see `_dw_deta`).
        res = np.zeros(len(fit.beta))
        fit._d2wdeta2 = res
        return res
    mu = fit.mu
    w = fit.w
    alpha = fit.alpha

    # Extended families: ∂²w/∂η² = ½·Deta4 from the Dd tables.
    if family_mgcv_extended:
        dd = _dDeta(fit, 2, family=family, y=y, wt=wt)
        d4 = 0.5 * np.asarray(dd["Deta4"], dtype=float)
        res = np.where((w != 0.0) & np.isfinite(d4), d4, 0.0)
        fit._d2wdeta2 = res
        return res

    mu_eta, V, Vp, Vpp, Vppp, g2, g3, g4 = _fit_link_derivs(fit, family)

    Vp_V = Vp / V
    Vpp_V = Vpp / V

    # B(μ) = V'/V + g''·μ_eta and its first derivative — already used in
    # `_dw_deta` for α'.
    Bp = Vpp_V - Vp_V ** 2 + g3 * mu_eta - g2 ** 2 * mu_eta ** 2
    # Second derivative B''(μ) = ∂B'/∂μ.
    Bpp = (
        Vppp / V - 3.0 * Vp * Vpp / (V * V) + 2.0 * Vp ** 3 / V ** 3
        + g4 * mu_eta - 3.0 * g2 * g3 * mu_eta ** 2
        + 2.0 * g2 ** 3 * mu_eta ** 3
    )

    if fit.is_fisher_fallback:
        alpha_prime_over_alpha = np.zeros_like(mu)
        alpha_pp_over_alpha = np.zeros_like(mu)
    else:
        B = Vp_V + g2 * mu_eta
        alpha_prime = -B + (y - mu) * Bp
        alpha_prime_over_alpha = alpha_prime / alpha
        alpha_pp = -2.0 * Bp + (y - mu) * Bpp
        alpha_pp_over_alpha = alpha_pp / alpha

    D = alpha_prime_over_alpha - 2.0 * g2 * mu_eta - Vp_V
    Dp = (
        alpha_pp_over_alpha - alpha_prime_over_alpha ** 2
        - 2.0 * g3 * mu_eta + 2.0 * g2 ** 2 * mu_eta ** 2
        - Vpp_V + Vp_V ** 2
    )
    res = w * mu_eta ** 2 * (D ** 2 + Dp - D * g2 * mu_eta)
    fit._d2wdeta2 = res
    return res


def _log_det_S_pos(rho, *, penalty_rank, slots, p):
    """log|Sλ|_+ — log-determinant of Sλ on its fixed range space.

    The range space is fixed (dimension p − Mp, set at init from the
    *structural* penalty), and we take the top ``penalty_rank``
    eigenvalues by magnitude. This is what makes the REML criterion
    push back against λ_j → 0: those directions still count, and their
    vanishing eigenvalues drive ``log(λ_small) → −∞``. A pure
    ``eigenvalue > tol`` filter would silently drop them and remove
    the penalty — exactly the failure mode for tensor / by-factor
    smooths with multiple λ's.

    mgcv anchor: the penalty log-determinant and its ρ-derivatives are
    returned by ``gam.reparam`` (gam.fit3.r:9) as ``det``/``det1``/``det2``;
    hea recomputes them from the eigen-reduced Sλ here.
    """
    r = penalty_rank
    if r <= 0:
        return 0.0
    Sλ = _s_lambda(slots, p, rho)
    Sλ = 0.5 * (Sλ + Sλ.T)
    w = np.linalg.eigvalsh(Sλ)
    # Take the top-r eigenvalues (descending). Clip to a tiny positive
    # floor so we don't take log of an FP-noise negative; exact-zero
    # null-space directions are excluded by the rank cap.
    w_sorted = np.sort(w)[::-1]
    top = w_sorted[:r]
    top = np.clip(top, 1e-300, None)
    return float(np.sum(np.log(top)))

def _dlog_det_H_drho(fit, rho, db_drho=None, *, X, slots, p, y, family_mgcv_extended, family, wt):
    """∂log|H|/∂ρ_k where H = X'WX + Sλ at converged β̂. Length-n_sp.

    Determinant identity: ∂log|H|/∂ρ_k = tr(H⁻¹ ∂H/∂ρ_k).

        ∂H/∂ρ_k = X' diag(∂w/∂ρ_k) X + λ_k S_k

    Trace decomposition with d_i := (X H⁻¹ X')_{ii} (length-n):

        tr(H⁻¹ X' diag(∂w/∂ρ_k) X) = Σ_i d_i · (∂w_i/∂ρ_k)
        ∂w_i/∂ρ_k = (∂w/∂η)_i · (X · ∂β̂/∂ρ_k)_i

    For Gaussian-identity, ∂w/∂η ≡ 0, and the first term vanishes —
    recovering the existing `λ_k · tr(H⁻¹ S_k)` form in `_reml_grad`.

    mgcv anchor: computed inside the single ``gdi1`` C call (gdi.c),
    dispatched from ``gam.fit3`` at gam.fit3.r:550 (deriv=2 adds the
    second-derivative block). hea splits that monolith into per-quantity
    Python helpers.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    sp = np.exp(rho)

    # diag(X H⁻¹ X') in O(n·p²): solve H · M = X' for each obs row,
    # then row-wise einsum. We compute H⁻¹ X' as a (p, n) matrix once.
    Hinv_Xt = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)
    d = np.einsum("ij,ji->i", X, Hinv_Xt)   # diag(X H⁻¹ X'), shape (n,)

    # For Gaussian-identity (PIRLS not used) fit.w is None — the
    # caller never reaches this path. PIRLS-converged fits always
    # have w populated.
    dw_deta = _dw_deta(fit, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)

    if db_drho is None:
        db_drho = _dbeta_drho(fit, rho, slots, p)

    # ∂η/∂ρ has shape (n, n_sp); ∂w/∂ρ = dw_deta[:, None] · ∂η/∂ρ.
    deta_drho = X @ db_drho                  # (n, n_sp)
    dw_drho = dw_deta[:, None] * deta_drho   # (n, n_sp)

    # H⁻¹ (p×p) is ρ-fixed across the slot loop — compute once, not once
    # per slot (the old in-loop cho_solve(eye) was O(n_sp) redundant
    # full-inverse solves). Bit-identical: same factor, same RHS.
    A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
    out = np.empty(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        # tr(H⁻¹ S_k): same block trick as `_reml_grad`.
        tr_Hinv_Sk = float(np.einsum("ij,ji->", A_inv[a:b, a:b], slot.S))
        out[k] = float(np.sum(d * dw_drho[:, k])) + sp[k] * tr_Hinv_Sk
    return out

def _fisher_edf(fit, *, X, XtX, p, family, family_mgcv_extended, y, wt, pls_lwork, n):
    """τ = tr(A_F⁻¹ X'W_F X), the Fisher-weight effective degrees of
    freedom — mgcv's ``oo$trA`` for the GCV/UBRE/GACV criteria
    (gam.fit3.r:644). Shared by `_gcv`, `_gcv_grad_pieces`, `_gacv`."""
    fit_F = _fisher_view(fit, family=family, family_mgcv_extended=family_mgcv_extended, y=y, wt=wt, X=X, pls_lwork=pls_lwork, n=n)
    if fit_F.w is None or np.allclose(fit_F.w, 1.0):
        XtWX = XtX
    else:
        Xw = X * np.sqrt(fit_F.w)[:, None]
        XtWX = Xw.T @ Xw
    A_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(p))
    return float(np.trace(A_inv @ XtWX))

# ---- magic: mgcv's performance-iteration GCV/UBRE optimizer ----------
# For a Gaussian-identity additive model under GCV.Cp (the default
# ``gam()`` call) mgcv does NOT run the outer Newton over gam.fit3 — it
# dispatches to ``am.fit`` → the C routine ``magic`` (mgcv.r:1932-2002,
# 2580-2640). magic QR-decomposes the n×q design ONCE, then optimizes the
# GCV/UBRE score over log-sp entirely on the q×q *reduced* system (each
# trial augments ``[R; St^½]`` and SVDs it — magic.c fit_magic), instead
# of re-fitting the full (n+e)×q system per score-eval as the Newton path
# does. That single reuse is the whole speedup (constant weights ⇒ the QR
# is valid for every sp). Ported mechanically from src/magic.c; the GCV
# score is bit-identical to ``_gcv`` (validated, dev/magic_fit_validate.py).

def _phi_pearson(fit, *, Mp, n, family, wt, y, slots, X, p):
    """The Pearson-Laplace ("REMLish") scale φ = P/(n−Mp) for the
    P-REML/P-ML criteria (gam.fit3.r:641, with pearson.extra=0 and
    n.true=nobs for standard families). P is the *unpenalized* Pearson
    statistic; Mp is the penalty null-space dimension."""
    P, _ = _pearson_and_deriv(None, fit, deriv=False, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    denom = float(n - Mp)
    return P / denom if denom > 0 else P


def _fit3_scale_est(fit, *, family, y, wt, n, X, control, fisher_view_fn):
    """gam.fit3's ``scale.est`` for unknown-scale standard families
    (gam.fit3.r:594-607): the weighted Pearson statistic over (n − τ)
    with mgcv's default Fletcher (2012) correction, or the raw Pearson /
    deviance estimators per ``gam.control(scale.est=)``. τ = tr(A⁻¹X'WX)
    at the Fisher weights (gdi's ``oo$trA``), via the triangular factor.
    Read by newton's score.scale and by efsudr's scale-slot refresh
    (mgcv ``lsp[length(lsp)] <- log(fit$scale)``, gam.fit4.r:845)."""
    if fit.is_working_gaussian:
        # bam reduced fit: the working-gaussian Pearson ≡ dev (the full
        # working RSS with rss.extra absorbed — mgcv's
        # (pearson+pearson.extra) at gam.fit3.r:598), and the Fletcher
        # s̄-correction is identically zero (gaussian V' ≡ 0), so all
        # three scale.est kinds coincide at dev/(n − τ).
        fit_F = fisher_view_fn(fit)
        w_F = fit_F.w
        if w_F is None or np.allclose(w_F, 1.0):
            Xw = X
        else:
            Xw = X * np.sqrt(np.maximum(w_F, 0.0))[:, None]
        if fit_F.A_chol_lower:
            Kw = solve_triangular(fit_F.A_chol, Xw.T, lower=True)
        else:
            Kw = solve_triangular(fit_F.A_chol, Xw.T, lower=False,
                                  trans="T")
        tau = float(np.sum(Kw * Kw))
        return float(fit.dev) / max(n - tau, 1.0)
    mu_arr = fit.mu
    V_arr = family.variance(mu_arr)
    # Weighted Pearson — gam.fit3.r:597 sum(weights*(y-mu)^2/V).
    pearson = float(np.sum(wt * (y - mu_arr) ** 2 / V_arr))
    fit_F = fisher_view_fn(fit)
    w_F = fit_F.w
    if w_F is None or np.allclose(w_F, 1.0):
        Xw = X
    else:
        Xw = X * np.sqrt(np.maximum(w_F, 0.0))[:, None]
    # τ = tr(A⁻¹X'WX) = ‖√W·X·C⁻¹‖_F² with A = C'C — the factor route,
    # not the κ²-squaring explicit product.
    if fit_F.A_chol_lower:
        Kw = solve_triangular(fit_F.A_chol, Xw.T, lower=True)
    else:
        Kw = solve_triangular(fit_F.A_chol, Xw.T, lower=False, trans="T")
    tau = float(np.sum(Kw * Kw))
    df_resid = max(n - tau, 1.0)
    scale_est = pearson / df_resid
    se_kind = (control or _GAM_CONTROL_DEFAULTS)["scale_est"]
    if se_kind == "deviance":
        # gam.fit3.r:606 — deviance estimator replaces the Pearson one
        # entirely.
        scale_est = float(fit.dev) / df_resid
    elif se_kind == "fletcher":
        s_bar = max(-0.9, float(np.mean(
            family.dvar(mu_arr) * (y - mu_arr) / V_arr
        )))
        if np.isfinite(s_bar):
            scale_est = scale_est / (1.0 + s_bar)
    # "pearson": keep the uncorrected estimate.
    return scale_est

def _gcv_grad_pieces(rho, fit, *, X, XtX, slots, family, n, p, y, family_mgcv_extended, wt, pls_lwork):
    """Shared first-derivative ingredients for the GCV/UBRE *and* GACV
    gradients — ``(dev, trA, ∂D/∂ρ, ∂τ/∂ρ)`` at PIRLS-converged β̂.

    See `_gcv_grad` for the maths. Factored out so `_gacv_grad`
    (gam.fit3.r:769) reuses the exact same deviance/trace derivatives
    rather than re-deriving them.
    """
    fit_F = _fisher_view(fit, family=family, family_mgcv_extended=family_mgcv_extended, y=y, wt=wt, X=X, pls_lwork=pls_lwork, n=n)
    n_sp = len(slots)
    if n_sp == 0:
        return float(fit.dev), 0.0, np.zeros(0), np.zeros(0)
    sp = np.exp(rho)
    n, p = n, p

    # Fisher X'W_F X (= XtX when W_F ≡ 1, e.g. Gaussian-identity).
    w_F = fit_F.w if fit_F.w is not None else np.ones(n)
    if np.allclose(w_F, 1.0):
        XtWX_F = XtX
    else:
        Xw = X * np.sqrt(w_F)[:, None]
        XtWX_F = Xw.T @ Xw

    A_F_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(p))
    F_F = A_F_inv @ XtWX_F
    edf_total = float(np.trace(F_F))

    # ∂D/∂ρ_k via chain through β̂ (Newton IFT — uses Newton fit.A_chol).
    db_drho = _dbeta_drho(fit, rho, slots, p)              # (p, n_sp)
    Sλ_beta = fit.S_full @ fit.beta                    # (p,)
    dD_drho = -2.0 * (Sλ_beta @ db_drho)               # (n_sp,)

    # ∂τ/∂ρ_k pieces. The n × n hat matrix ``P_F = X·A_F⁻¹·X'`` is
    # NOT formed — at n=54k that's 23 GB. n-side quantities are
    # built from ``K_F`` (n × p) with ``K_F K_F' = P_F`` (mgcv
    # gdi.c:952). For Gaussian-identity / Gamma+log (``dW_F/dη ≡ 0``)
    # the ``w_piece`` consumer of those n-side quantities is
    # identically zero, so we skip the K_F build there entirely.
    # Penalty piece: −λ_k · tr(A_F⁻¹·S_k·F_F).  No n-side quantities.
    pen_piece = np.empty(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        AinvSk = A_F_inv[:, a:b] @ slot.S
        pen_piece[k] = -sp[k] * float(
            np.einsum("ij,ji->", AinvSk, F_F[a:b, :])
        )

    # W_F-deriv piece: (d − s)' hv_F,k. dW_F/dη = 0 for Gaussian-identity
    # and for Gamma+log (W_F ≡ 1). When zero we skip building K_F entirely.
    # bam's reduced working fits are gaussian-identity by construction
    # (bam.r:932) whatever the response family — same skip.
    if ((family.name == "gaussian" and family.link.name == "identity")
            or fit.is_working_gaussian):
        w_piece = np.zeros(n_sp)
    else:
        dw_deta = _dw_deta(fit_F, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                 # (n,) — Fisher form
        v = X @ db_drho                                # (n, n_sp)
        hv = dw_deta[:, None] * v                      # (n, n_sp)
        K_F = _make_K(fit_F.A_chol, fit_F.A_chol_lower, X)   # (n, p)
        d_diag = np.einsum("ij,ij->i", K_F, K_F)               # (n,) diag(P_F)
        M_w = (K_F * w_F[:, None]).T @ K_F                     # (p, p) K' diag(w) K
        KM_w = K_F @ M_w                                        # (n, p)
        s = np.einsum("ij,ij->i", KM_w, K_F)                   # (n,) diag(K·M_w·K')
        w_piece = (d_diag - s) @ hv                    # (n_sp,)

    return float(fit.dev), edf_total, dD_drho, w_piece + pen_piece


def _reml(rho, log_phi=0.0, fit=None, *, Mp, wt, y, binom_n, gamma, family, family_mgcv_extended, use_ml_proj, pearson_scale_criterion, reml_ind, penalty_rank, slots, p, UrS, reparam_cache):
    """Laplace-approximate (RE)ML in 2·V units, family/link-agnostic.

    Direct port of mgcv's gam.fit3.r:616 with `remlInd ∈ {1, 0}`:

        2·V = Dp/φ − 2·ls0 + log|H_*| − log|Sλ|_+
              − remlInd·Mp·(log(2π·φ) − log γ)

    where the Hessian log-determinant differs by method:

        REML: log|H_*| = log|X'WX + Sλ|             (full)
        ML  : log|H_*| = log|U_rᵀ(X'WX + Sλ)U_r|    (range only)

    with U_r an orthonormal basis for range(Sλ). For ML the Laplace
    approximation marginalises only over the penalised subspace, so
    the Mp null-space directions are dropped — see mgcv's
    ``MLpenalty1`` in gdi.c:1532-1680. We compute the range-space
    log-det as ``fit.log_det_A + log|U_nᵀ A⁻¹ U_n|`` (block
    determinant identity, with U_n the null basis).

    ``remlInd = 1`` for ``method="REML"`` (mgcv's default; profiles
    out the unpenalized fixed-effect null-space prior of dimension
    Mp). ``remlInd = 0`` for ``method="ML"`` (treats those β as
    deterministic — score is comparable across different fixed-
    effect structures, suitable for likelihood-ratio tests).

    Dp = fit.dev + β̂'Sλβ̂ at PIRLS-converged β̂ and
    ls0 = family.ls(y, wt, φ)[0]. ``fit.log_det_A`` is the un-φ-scaled
    log|X'WX + Sλ|; the φ-coefficients of the prior-normalisation term
    and the Hessian/penalty Jacobi cancel everywhere except the
    −Mp·log(2π·φ) prior-rank term — see the Laplace derivation in
    Wood 2017 §6.6.

    Reduction-to-Gaussian (REML): profile out φ̂ = Dp/(n−Mp) and
    substitute. With Gaussian ls0 = −n·log(2πφ)/2 (wt=1),

        2·V_R(φ̂) = (n−Mp)·(1 + log(2π·Dp/(n−Mp)))
                   + log|A| − log|S|_+

    which equals ``_reml(rho)`` exactly under method="REML". For
    method="ML" the analogous profile-out is φ̂ = Dp/n.
    """
    Dp = fit.dev + fit.pen
    if not np.isfinite(Dp):
        return 1e15
    # Dp ≤ 0 is degenerate ONLY for scale-unknown families, where the
    # criterion profiles φ̂ = Dp/(n−Mp) and needs log(Dp). For scale-
    # known families Dp enters only as Dp/φ (φ fixed): betar's
    # "−2logLik as deviance" is legitimately negative — keep it.
    if Dp <= 0 and not family.scale_known:
        return 1e15
    Mp = float(Mp)
    phi = float(np.exp(log_phi))
    if not (np.isfinite(phi) and phi > 0):
        return 1e15
    # ``family.ls`` returns (ls0, d_ls/d_log_φ, d²_ls/d_log_φ²) at the
    # prior weights — only ls0 enters the criterion; the derivatives
    # feed the (ρ, log φ) gradient/Hessian rows. Extended families
    # carry θ in their saturated log-lik (gam.fit4.r:730 calls the
    # 4-arg ls(y, w, θ, scale)) — dispatch through ls_extended.
    if family_mgcv_extended:
        ls0 = float(family.ls_extended(
            y, wt, theta=family.get_theta(), scale=phi,
        )["ls"])
    elif binom_n is not None:
        # cbind responses: fix.family.ls's binomial ls is
        # -aic(y, n, y, w, 0)/2 with n = trials (gam.fit3.r:614 passes
        # n separately from the prior weights).
        ls0 = float(family.ls(y, wt, phi,
                                   n=binom_n)[0])
    else:
        ls0 = float(family.ls(y, wt, phi)[0])
    rp = _reparam_eval(UrS, reparam_cache, rho)
    log_det_S = (rp["det"] if rp is not None
                 else _log_det_S_pos(rho, penalty_rank=penalty_rank, slots=slots, p=p))
    log_det_H = fit.log_det_A
    if use_ml_proj:
        adj, _, _ = _ml_logdet_adj(fit, Mp)
        log_det_H = log_det_H + adj
    # mgcv (gam.fit3.r:622): ``gamma`` divides the data-fit piece
    # (Dp/φ − 2·ls0) and adds a +Mp·log(γ) constant to compensate the
    # −Mp·log(2πφ) prior-rank term so the criterion stays consistent
    # with the partially-profiled likelihood interpretation. For
    # method="ML", remlInd=0 drops both Mp pieces — β is treated as
    # deterministic, so there is no fixed-effect prior to integrate out.
    # The Pearson-Laplace criteria (P-REML/P-ML) have no γ at all
    # (gam.fit3.r:652) — γ≡1 there; ``_reml`` is then called with
    # φ = Pearson/(n−Mp) plugged in, reproducing mgcv's P-REML value.
    gamma = 1.0 if pearson_scale_criterion else gamma
    return (
        (Dp / phi - 2.0 * ls0) / gamma
        + log_det_H
        - log_det_S
        - reml_ind * (Mp * float(np.log(2.0 * np.pi * phi))
                      - Mp * float(np.log(gamma)))
    )

def _gcv(rho, fit=None, *, gamma, n, scale_fixed_value, scale_known, X, XtX, p, family, family_mgcv_extended, y, wt, pls_lwork):
    """GCV (scale-unknown) or UBRE/Mallows-Cp (scale-known). Wood 2017 §4.4.

        scale_unknown:  V_g = n · D / (n − τ)²
        scale_known:    V_u = D/n + 2·τ/n − 1     (φ ≡ 1)

    with D = Σ family.dev_resid(y, μ̂, wt) the deviance and
    τ = tr((X'W_F X + Sλ)⁻¹ X'W_F X) the Fisher-W effective degrees of
    freedom at PIRLS-converged β̂. mgcv's GCV/UBRE plugs in Fisher
    W_F = μ_η²/V here, not the Newton W_N = α·μ_η²/V used in the REML
    log|H+S| term (verified empirically against trees+Gamma+log:
    τ_F = 4.4222538 = mgcv m$edf, V_g(τ_F) = 0.008082356 = mgcv GCV).
    For canonical links Fisher ≡ Newton; for Gaussian-identity W = I
    and this collapses to D=rss, τ=tr(A⁻¹ X'X), bit-identical to the
    pre-Stage-2 closed form.

    mgcv anchor: the GCV/UBRE score and its ρ-derivatives are assembled
    in ``gam.fit3``'s score tail (gam.fit3.r:590-640) from the gdi1 trA
    and deviance outputs.
    """
    edf_total = _fisher_edf(fit, X=X, XtX=XtX, p=p, family=family, family_mgcv_extended=family_mgcv_extended, y=y, wt=wt, pls_lwork=pls_lwork, n=n)
    # mgcv (gam.fit3.r): ``gamma`` inflates the apparent edf cost in
    # the criterion: V_g = n·D / (n − γ·τ)²; V_u = D/n + 2·γ·τ/n − 1.
    if scale_known:
        # mgcv (gam.fit3.r:753): UBRE = dev/n − s + 2γ·τ·s/n at the
        # known scale s (1 on the poisson/binomial defaults — the
        # ``·s`` keeps those bit-identical; gam(scale=) sets s).
        s_phi = scale_fixed_value
        return (fit.dev / n + 2.0 * gamma * edf_total * s_phi / n
                - s_phi)
    denom = n - gamma * edf_total
    if denom <= 0:
        return 1e15
    return n * fit.dev / (denom * denom)

def _gacv(rho, fit=None, *, gamma, n, X, XtX, p, family, family_mgcv_extended, y, wt, pls_lwork, slots):
    """GACV (generalized approximate cross-validation, scale-unknown).
    mgcv gam.fit3.r:751:

        GACV = dev/n + P·2γ·trA / (δ·n),   δ = n − γ·trA

    with P the raw Pearson statistic and trA the Fisher edf. The
    scale-*known* ``GACV.Cp`` degenerates to UBRE and is routed through
    `_gcv` instead (mgcv.r:1956)."""
    trA = _fisher_edf(fit, X=X, XtX=XtX, p=p, family=family, family_mgcv_extended=family_mgcv_extended, y=y, wt=wt, pls_lwork=pls_lwork, n=n)
    P, _ = _pearson_and_deriv(None, fit, deriv=False, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    delta = n - gamma * trA
    if delta <= 0:
        return 1e15
    return fit.dev / n + P * 2.0 * gamma * trA / (delta * n)


def _pearson_hess(fit, rho, *, db_drho=None, d2b=None, X, slots, wt, y, family, p, family_mgcv_extended):
    """Pearson statistic ρ-Hessian ``P2 = ∂²P/∂ρ_m∂ρ_k`` (n_sp × n_sp) at
    PIRLS-converged β̂ — mechanical port of mgcv ``pearson2`` (gdi.c:1207-
    1273), the ``deriv2`` branch. With ``Pe1 = ∂P_i/∂η`` (the
    `_pearson_and_deriv` per-obs term) and ``Pe2 = ∂²P_i/∂η²``:

        P2[m,k] = Σ_i [ Pe1ᵢ·(X·∂²β̂/∂ρ_m∂ρ_k)ᵢ + Pe2ᵢ·(X∂β̂/∂ρ_m)ᵢ·(X∂β̂/∂ρ_k)ᵢ ]

    mgcv normalises ``V1 = V'/V``, ``V2 = V''/V`` and uses ``g1 = g'``,
    ``g2/g1 = g''/g'²`` (= ``link.g2g``) before pearson2 (gam.fit3.r:534-
    535); pearson2's Pe2 (gdi.c:1232-1233) becomes, in those units with
    ``me = μ_η = 1/g'``:

        Pe2 = −Pe1·g2g + me²·(2w/V + 2·xx·V1) − me·Pe1·V1 − me²·xx·r·(V2 − V1²)

    with ``xx = r·w/V``, ``r = y − μ̂``. β̂'s ρ-derivatives use the Newton
    IFT (`_dbeta_drho` / `_d2beta_drho_drho`), as `_pearson_and_deriv` does.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((0, 0))
    if fit.is_working_gaussian:
        # bam reduced fit: P ≡ D on the gaussian working problem (see
        # `_pearson_and_deriv`), so P2 ≡ the deviance ρ-Hessian. With
        # Pe1 = −2·(working residual), Pe2 = 2 and R'(f−Rβ̂) = Sλβ̂:
        #   P2[m,k] = −2·(Sλβ̂)'·∂²β̂/∂ρ_m∂ρ_k + 2·(R·∂β̂_m)'(R·∂β̂_k)
        # — all p-space (X here IS the reduced R).
        if db_drho is None:
            db_drho = _dbeta_drho(fit, rho, slots, p)
        if d2b is None:
            d2b = _d2beta_drho_drho(fit, rho, db_drho=db_drho, slots=slots, p=p, X=X, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)
        v = X @ db_drho                                    # (p, n_sp)
        Slb = fit.S_full @ fit.beta                        # (p,)
        P2 = np.empty((n_sp, n_sp))
        for m in range(n_sp):
            for k in range(m, n_sp):
                val = float(-2.0 * (Slb @ d2b[:, m, k])
                            + 2.0 * (v[:, m] @ v[:, k]))
                P2[m, k] = P2[k, m] = val
        return P2
    mu = fit.mu
    V = family.variance(mu)
    Vp = family.dvar(mu)
    Vpp = family.d2var(mu)
    me = family.link.mu_eta(fit.eta)
    g2g = family.link.g2g(mu)
    r = y - mu
    V1n = Vp / V
    V2n = Vpp / V
    xx = r * wt / V
    Pe1 = -xx * (2.0 + r * V1n) * me
    Pe2 = (-Pe1 * g2g
           + me * me * (2.0 * wt / V + 2.0 * xx * V1n)
           - me * Pe1 * V1n
           - me * me * xx * r * (V2n - V1n * V1n))
    Pe1 = np.where(np.isfinite(Pe1), Pe1, 0.0)
    Pe2 = np.where(np.isfinite(Pe2), Pe2, 0.0)
    if db_drho is None:
        db_drho = _dbeta_drho(fit, rho, slots, p)
    if d2b is None:
        d2b = _d2beta_drho_drho(fit, rho, db_drho=db_drho, slots=slots, p=p, X=X, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)
    v = X @ db_drho                                        # (n, n_sp)
    P2 = np.empty((n_sp, n_sp))
    for m in range(n_sp):
        for k in range(m, n_sp):
            Xd2b = X @ d2b[:, m, k]
            val = float(np.sum(Pe1 * Xd2b + Pe2 * v[:, m] * v[:, k]))
            P2[m, k] = P2[k, m] = val
    return P2

def _gacv_grad(rho, fit=None, *, gamma, slots, n, X, XtX, family, p, y, family_mgcv_extended, wt, pls_lwork):
    """Analytical gradient of `_gacv` (mgcv gam.fit3.r:769):

        GACV1_k = D1_k/n + 2P/δ²·trA1_k + 2γ·trA·P1_k/(δ·n)

    reusing `_gcv_grad_pieces` for D1=∂dev/∂ρ and trA1=∂τ/∂ρ, plus the
    Pearson statistic P and its derivative P1."""
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    dev, trA, dD_drho, dtau_drho = _gcv_grad_pieces(rho, fit, X=X, XtX=XtX, slots=slots, family=family, n=n, p=p, y=y, family_mgcv_extended=family_mgcv_extended, wt=wt, pls_lwork=pls_lwork)
    P, P1 = _pearson_and_deriv(rho, fit, deriv=True, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    delta = n - gamma * trA
    if delta <= 0:
        return np.zeros(n_sp)
    return (
        dD_drho / n
        + 2.0 * P / (delta * delta) * dtau_drho
        + 2.0 * gamma * trA * P1 / (delta * n)
    )

def _gcv_grad(rho, fit=None, *, gamma, slots, n, scale_fixed_value, scale_known, X, XtX, family, p, y, family_mgcv_extended, wt, pls_lwork):
    """Analytical gradient of `_gcv`. Length n_sp. Wood 2008 §4.

        scale_unknown:  ∂V_g/∂ρ_k = n·∂D/∂ρ_k / (n−τ)²
                                   + 2·n·D·∂τ/∂ρ_k / (n−τ)³
        scale_known:    ∂V_u/∂ρ_k = ∂D/∂ρ_k / n + 2·∂τ/∂ρ_k / n

    Pieces (PIRLS-converged β̂):

      ∂D/∂ρ_k = −2·(Sλ β̂)' · ∂β̂/∂ρ_k       (Newton IFT for ∂β̂/∂ρ_k)

      τ = tr(A_F⁻¹ X'W_F X) with A_F = X'W_F X + Sλ, W_F = μ_η²/V
          (Fisher; mgcv gam.fit3.r:644).
      ∂τ/∂ρ_k = (d − s)' · hv_F,k − λ_k · tr(A_F⁻¹ S_k F_F)

    with d = diag(X A_F⁻¹ X'), s = (X A_F⁻¹ X')² · W_F (row-sum),
    F_F = A_F⁻¹ X'W_F X, hv_F,k = ∂W_F/∂ρ_k = dW_F/dη · (X·∂β̂/∂ρ_k).

    β̂'s ρ-dependence comes from the Newton IFT (since the penalized
    score's β-Jacobian at β̂ is the Newton H = X'W_N X + Sλ, regardless
    of which W enters the score function being optimized), so
    `_dbeta_drho(fit, rho)` keeps the original Newton ``fit.A_chol``.
    For Gaussian-identity hv ≡ 0 ⇒ standard `−λ_k·tr(A⁻¹ S_k F)` form.
    For Gamma+log dW_F/dη ≡ 0 ⇒ same simpler form.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    dev, edf_total, dD_drho, dtau_drho = _gcv_grad_pieces(rho, fit, X=X, XtX=XtX, slots=slots, family=family, n=n, p=p, y=y, family_mgcv_extended=family_mgcv_extended, wt=wt, pls_lwork=pls_lwork)

    # ``gamma`` inflates τ in the criterion: V_g = n·D/(n−γ·τ)²,
    # V_u = D/n + 2γτ/n − 1. Chain-rule the τ-derivative pieces by γ.
    if scale_known:
        s_phi = scale_fixed_value
        return dD_drho / n + 2.0 * gamma * dtau_drho * s_phi / n
    denom = n - gamma * edf_total
    if denom <= 0:
        return np.zeros(n_sp)
    return (
        n * dD_drho / (denom * denom)
        + 2.0 * n * gamma * dev * dtau_drho / (denom**3)
    )

def _gcv_hessian(rho, fit=None, *, return_pieces=False, X, XtX, gamma, slots, n, p, scale_fixed_value, scale_known, y, family_mgcv_extended, family, wt, pls_lwork):
    """Analytical Hessian of `_gcv`. Shape (n_sp, n_sp). Wood 2008 §4.

    With ``return_pieces=True`` the shared ρ-second-derivative blocks
    (deviance ``D2``, edf ``trA2``, and the first derivatives ``D1``/
    ``trA1``/``trA`` plus ``db_drho``/``d2b``) are returned as a dict
    *before* the GCV/UBRE composition — these feed the analytic GACV2
    (`_gacv_hessian`) and P-REML2/P-ML2 (`_preml_hessian`) assemblies,
    which share the same deviance/edf Hessians (mgcv gdi1 ``oo$D2``/
    ``oo$trA2``, gam.fit3.r:773-775). The default path is unchanged.

    scale_unknown:
        V_g = n D / (n−τ)²
        ∂²V_g/∂ρ_l∂ρ_k = n·∂²D/(n−τ)²
                        + 2n·(∂D⊗∂τ + ∂τ⊗∂D)/(n−τ)³
                        + 2n·D·∂²τ/(n−τ)³
                        + 6n·D·(∂τ⊗∂τ)/(n−τ)⁴
    scale_known:
        V_u = D/n + 2τ/n − 1
        ∂²V_u/∂ρ_l∂ρ_k = ∂²D/n + 2·∂²τ/n

    Pieces (PIRLS-converged β̂):

      ∂²D/∂ρ_l∂ρ_k = 2 λ_l λ_k β̂' S_l A_N⁻¹ S_k β̂
                    − 2 (∂β̂/∂ρ_l)' Sλ (∂β̂/∂ρ_k)
                    − 2 (Sλβ̂)' ∂²β̂/(∂ρ_l ∂ρ_k)

        All β̂-derivatives use Newton A_N = X'W_N X + Sλ (the IFT
        Hessian); ``_d2beta_drho_drho`` internally calls ``_dw_deta``
        on the Newton fit — kept that way.

      ∂²τ/∂ρ_l∂ρ_k uses Fisher A_F, F_F = A_F⁻¹ X'W_F X, and Fisher
      W-derivatives dW_F/dη, d²W_F/dη² (mgcv gam.fit3.r:644). The
      d²w_lk = d²W_F/dη² · v_l v_k + dW_F/dη · X·∂²β̂/(∂ρ_l ∂ρ_k)
      term mixes Fisher (dW_F/dη) with Newton (∂²β̂/∂ρ²) — both are
      correct for their respective roles.

    Gaussian-identity: hv ≡ 0 and d²w ≡ 0, so Q_k ≡ 0 and the W-deriv
    terms collapse to ``2 λ_l λ_k tr[A⁻¹ S_l A⁻¹ S_k F] − δ_lk·λ_k·
    tr[A⁻¹ S_k F]``. For Gamma+log Fisher W_F ≡ 1 ⇒ same closed form
    with A_F = X'X + Sλ.
    """
    fit_F = _fisher_view(fit, family=family, family_mgcv_extended=family_mgcv_extended, y=y, wt=wt, X=X, pls_lwork=pls_lwork, n=n)
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((0, 0))

    sp = np.exp(rho)
    n, p = n, p

    # Fisher X'W_F X for τ.
    w_F = fit_F.w if fit_F.w is not None else np.ones(n)
    if np.allclose(w_F, 1.0):
        XtWX_F = XtX
    else:
        Xw = X * np.sqrt(w_F)[:, None]
        XtWX_F = Xw.T @ Xw

    # Fisher precomputations for τ. The n × n hat matrix
    # ``P_F = X·A_F⁻¹·X'`` is NOT formed (~23 GB at n=54k). n-side
    # quantities are built from ``K_F`` (n × p) with
    # ``K_F K_F' = P_F`` (mgcv gdi.c:952). The (p × n) ``M_F`` and
    # its derived ``MhX_k`` builds in the d2tau loop are also
    # gated behind ``needs_w`` — for Gaussian-identity / Gamma+log
    # they multiply by zero ``hv`` and the W-deriv contributions
    # collapse to a closed-form sparse-block expression that
    # touches only ``AinvS_block`` and ``F_F`` (both p-sized).
    A_F_inv = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), np.eye(p))
    F_F = A_F_inv @ XtWX_F                                      # (p, p)
    edf_total = float(np.trace(F_F))

    # First-derivative ingredients. ∂β̂/∂ρ uses Newton A_N (fit.A_chol).
    db_drho = _dbeta_drho(fit, rho, slots, p)                  # (p, n_sp)
    Sλβ = fit.S_full @ fit.beta                            # (p,)
    dD_drho = -2.0 * (Sλβ @ db_drho)                       # (n_sp,)

    # W-derivative arrays. Two distinct chains:
    #   Fisher (W_F): for τ-related ingredients (hv_F, d²W_F/dη²).
    #   Newton (W_N): for ∂²β̂/∂ρ² IFT inside `_d2beta_drho_drho`.
    # For canonical or Fisher-fallback fits these coincide.
    dw_deta_F = _dw_deta(fit_F, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                       # (n,) Fisher
    d2w_deta2_F = _d2w_deta2(fit_F, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                   # (n,) Fisher
    dw_deta_N = _dw_deta(fit, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                         # (n,) Newton
    v = X @ db_drho                                        # (n, n_sp)
    hv = dw_deta_F[:, None] * v                            # (n, n_sp)

    # ``d_diag = diag(P_F)`` and ``s = (P_F⊙P_F)·w_F`` are needed by
    # ``w_piece`` (gradient) and ``T6_minus_T3B`` (Hessian). For
    # families with dW_F/dη ≡ 0 and d²W_F/dη² ≡ 0 (Gaussian-identity,
    # Gamma+log) both consumers multiply by ``hv`` or ``d2w_lk`` and
    # vanish identically, so the K_F build is skipped — keeping the
    # n × n hat matrix off the heap. Mirrors mgcv gdi.c:952.
    needs_w = bool(np.any(hv)) or bool(np.any(d2w_deta2_F))
    if needs_w:
        # M_F (p × n) feeds the per-slot ``MhX_k = M_F @ (hv·X)`` in
        # the d2tau loop. K_F (n × p) supports diag(P_F) and the
        # ``s = (P_F⊙P_F)·w_F`` reduction. Both are O(p²·n) to build
        # — only paid in this branch.
        M_F = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), X.T)
        K_F = _make_K(fit_F.A_chol, fit_F.A_chol_lower, X)   # (n, p)
        d_diag = np.einsum("ij,ij->i", K_F, K_F)               # (n,) diag(K_F K_F') = diag(P_F)
        M_w = (K_F * w_F[:, None]).T @ K_F                     # (p, p) K_F' diag(w_F) K_F
        KM_w = K_F @ M_w                                        # (n, p)
        s = np.einsum("ij,ij->i", KM_w, K_F)                   # (n,) diag(K_F·M_w·K_F') = (P_F⊙P_F)·w_F
        d_minus_s = d_diag - s
    else:
        M_F = None
        d_minus_s = None

    # Per-slot block precomputations.
    AinvS_block: list[np.ndarray] = []
    Sbeta_full = np.zeros((n_sp, p))
    AinvSbeta = np.empty((n_sp, p))
    tr_AinvSk_F = np.zeros(n_sp)
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        AinvS_block.append(A_F_inv[:, a:b] @ slot.S)
        beta_k = fit.beta[a:b]
        Sb = slot.S @ beta_k
        Sbeta_full[k, a:b] = Sb
        # Note: the bSAS_b piece of ∂²D uses Newton A (the IFT Hessian),
        # since it expresses (∂β̂/∂ρ_l)' Sλ (∂β̂/∂ρ_k) and ∂β̂/∂ρ uses A_N⁻¹.
        AinvSbeta[k] = cho_solve(
            (fit.A_chol, fit.A_chol_lower), Sbeta_full[k]
        )
        tr_AinvSk_F[k] = float(np.einsum(
            "ij,ji->", AinvS_block[k], F_F[a:b, :]
        ))

    pen_piece = -sp * tr_AinvSk_F                          # (n_sp,)
    if d_minus_s is not None:
        w_piece = d_minus_s @ hv                           # (n_sp,)
    else:
        w_piece = np.zeros(n_sp)
    dtau_drho = w_piece + pen_piece

    # ---- ∂²D/∂ρ_l∂ρ_k — uses Newton A throughout β̂-derivatives. -----
    # bSAS_b[l, k] = β̂' S_l A_N⁻¹ S_k β̂ (already symmetric).
    bSAS_b = Sbeta_full @ AinvSbeta.T                      # (n_sp, n_sp)
    Sλ_db = fit.S_full @ db_drho                            # (p, n_sp)
    db_Sλ_db = db_drho.T @ Sλ_db                            # (n_sp, n_sp)
    d2b = _d2beta_drho_drho(
        fit, rho, db_drho=db_drho, dw_deta=dw_deta_N
    , slots=slots, p=p, X=X, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                                                      # (p, n_sp, n_sp)
    Sλβ_d2b = np.einsum("p,pij->ij", Sλβ, d2b)              # (n_sp, n_sp)

    sp_outer = np.outer(sp, sp)
    d2D = (
        2.0 * sp_outer * bSAS_b
        - 2.0 * db_Sλ_db
        - 2.0 * Sλβ_d2b
    )
    d2D = 0.5 * (d2D + d2D.T)

    # ---- ∂²τ/∂ρ_l∂ρ_k — Fisher A_F, F_F, dW_F. ----------------------
    # Two paths:
    #
    # * needs_w=True (general families): build the dense Y_k = A_F⁻¹·P_F,k
    #   and U_k = M_F·diag(hv_k)·X (each p × p, n_sp of them) and pair
    #   them through F_F. This is the literal Wood 2008 §4 form.
    #
    # * needs_w=False (Gaussian-identity / Gamma+log): hv = 0 and
    #   d²W_F/dη² = 0, so U_k ≡ 0 and Y_k = sp[k]·AinvS_block[k]
    #   embedded sparsely in cols (a_k:b_k). The d²τ formula collapses
    #   (per the docstring) to
    #     d²τ_lk = 2·sp[l]·sp[k]·tr(A_F⁻¹·S_l·A_F⁻¹·S_k·F_F)
    #            − δ_lk·sp[k]·tr(A_F⁻¹·S_k·F_F)
    #   We compute the doubly-traced piece block-locally on
    #   AinvS_block (p × k_size) without ever materializing Y_k as a
    #   dense p × p matrix or running an O(p³) p × p matmul. Mirrors
    #   mgcv ``get_trA2`` (gdi.c:1132-1158), which uses the
    #   pre-reduced PtSP / PtSPKtK p × p arrays through ``diagABt``.
    d2tau = np.zeros((n_sp, n_sp))
    if needs_w:
        Y_full = np.empty((n_sp, p, p))
        U_full = np.empty((n_sp, p, p))
        for k in range(n_sp):
            a, b = slots[k].col_start, slots[k].col_end
            MhX_k = M_F @ (hv[:, k:k+1] * X)
            U_full[k] = MhX_k
            Y_k = MhX_k.copy()
            Y_k[:, a:b] += sp[k] * AinvS_block[k]
            Y_full[k] = Y_k

        for ll in range(n_sp):
            for k in range(ll, n_sp):
                YlYk = Y_full[ll] @ Y_full[k]
                T_a = float(np.einsum("ij,ji->", YlYk, F_F))
                if ll == k:
                    T_b = T_a
                else:
                    YkYl = Y_full[k] @ Y_full[ll]
                    T_b = float(np.einsum("ij,ji->", YkYl, F_F))
                T1_T2 = T_a + T_b

                T4 = float(np.einsum("ij,ji->", Y_full[k], U_full[ll]))
                T5 = float(np.einsum("ij,ji->", Y_full[ll], U_full[k]))

                # d²W_F_lk = d²W_F/dη² · v_l v_k + dW_F/dη · X·∂²β̂/(∂ρ_l ∂ρ_k).
                Xd2b_lk = X @ d2b[:, ll, k]
                d2w_lk = (
                    d2w_deta2_F * v[:, ll] * v[:, k]
                    + dw_deta_F * Xd2b_lk
                )
                T6_minus_T3B = float(d_minus_s @ d2w_lk)
                delta_S = -sp[k] * tr_AinvSk_F[k] if ll == k else 0.0

                val = T1_T2 - T4 - T5 + T6_minus_T3B + delta_S
                d2tau[ll, k] = val
                if ll != k:
                    d2tau[k, ll] = val
    else:
        # Block-aware Gaussian-identity / Gamma+log path.
        #   tr(Y_l·Y_k·F_F) = sp[l]·sp[k]·tr(A_F⁻¹·S_l_full·A_F⁻¹·S_k_full·F_F).
        # With Y_x sparse in cols (a_x:b_x) holding sp[x]·AinvS_block[x],
        #   (Y_l·Y_k)[r, c] = sp[l]·sp[k]·(AinvS_block[l] @
        #                     AinvS_block[k][a_l:b_l, :])[r, c-a_k]
        # for c ∈ (a_k:b_k), zero elsewhere; trace against F_F reduces to
        #   sp[l]·sp[k]·tr(M_lk · F_F[a_k:b_k, :])
        # with M_lk = AinvS_block[l] @ AinvS_block[k][a_l:b_l, :], shape
        # (p, k_k). Cost per pair: O(p·k_l·k_k + p·k_k), summed over
        # the n_sp(n_sp+1)/2 pairs.
        for ll in range(n_sp):
            a_ll, b_ll = slots[ll].col_start, slots[ll].col_end
            for k in range(ll, n_sp):
                a_k, b_k = slots[k].col_start, slots[k].col_end
                M_lk = AinvS_block[ll] @ AinvS_block[k][a_ll:b_ll, :]   # (p, k_k)
                T_a = sp[ll] * sp[k] * float(
                    np.einsum("rc,cr->", M_lk, F_F[a_k:b_k, :])
                )
                if ll == k:
                    T_b = T_a
                else:
                    M_kl = AinvS_block[k] @ AinvS_block[ll][a_k:b_k, :]  # (p, k_ll)
                    T_b = sp[k] * sp[ll] * float(
                        np.einsum("rc,cr->", M_kl, F_F[a_ll:b_ll, :])
                    )
                delta_S = -sp[k] * tr_AinvSk_F[k] if ll == k else 0.0
                val = T_a + T_b + delta_S
                d2tau[ll, k] = val
                if ll != k:
                    d2tau[k, ll] = val

    d2tau = 0.5 * (d2tau + d2tau.T)

    if return_pieces:
        return {"D1": dD_drho, "trA1": dtau_drho, "trA": edf_total,
                "D2": d2D, "trA2": d2tau, "db_drho": db_drho, "d2b": d2b}

    # ---- Compose criterion Hessian --------------------------------
    # ``gamma`` inflates the τ-coefficient in V_u and V_g; chain-rule
    # picks up γ at every τ-derivative encounter.
    if scale_known:
        s_phi = scale_fixed_value
        return d2D / n + 2.0 * gamma * d2tau * s_phi / n

    denom = n - gamma * edf_total
    if denom <= 0:
        return np.full((n_sp, n_sp), 1e15)

    Dn = float(fit.dev)
    dD_dτ = np.outer(dD_drho, dtau_drho)
    dτ_dτ = np.outer(dtau_drho, dtau_drho)
    H = (
        n * d2D / (denom * denom)
        + 2.0 * n * gamma * (dD_dτ + dD_dτ.T) / (denom**3)
        + 2.0 * n * gamma * Dn * d2tau / (denom**3)
        + 6.0 * n * (gamma ** 2) * Dn * dτ_dτ / (denom**4)
    )
    return H


def _gacv_hessian(rho, fit=None, *, gamma, slots, n, X, XtX, p, scale_fixed_value, scale_known, y, family_mgcv_extended, family, wt, pls_lwork):
    """Analytic GACV Hessian ``GACV2`` (n_sp × n_sp) — mechanical port of
    mgcv gam.fit3.r:786-790:

        GACV2 = D2/n + outer(trA1,trA1)·4P/δ³ + 2P·trA2/δ²
              + 2·outer(trA1,P1)/δ² + 2·outer(P1,trA1)·(1/(δn)+trA/(nδ²))
              + 2·trA·P2/(δn);   GACV2 = (GACV2+GACV2ᵀ)/2

    with δ = n − γ·trA. D2 (deviance Hessian) and trA2 (edf Hessian) come
    from the shared `_gcv_hessian` pieces; P/P1/P2 are the raw Pearson
    statistic and its ρ-derivatives. Replaces the former FD Hessian."""
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((0, 0))
    pc = _gcv_hessian(rho, fit, return_pieces=True, X=X, XtX=XtX, gamma=gamma, slots=slots, n=n, p=p, scale_fixed_value=scale_fixed_value, scale_known=scale_known, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt, pls_lwork=pls_lwork)
    trA1, trA, trA2 = pc["trA1"], pc["trA"], pc["trA2"]
    D2 = pc["D2"]
    P, P1 = _pearson_and_deriv(rho, fit, deriv=True, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    P2 = _pearson_hess(fit, rho, db_drho=pc["db_drho"], d2b=pc["d2b"], X=X, slots=slots, wt=wt, y=y, family=family, p=p, family_mgcv_extended=family_mgcv_extended)
    delta = n - gamma * trA
    if delta <= 0:
        return np.full((n_sp, n_sp), 1e15)
    d2 = delta * delta
    d3 = delta * d2
    G = (D2 / n
         + np.outer(trA1, trA1) * 4.0 * P / d3
         + 2.0 * P * trA2 / d2
         + 2.0 * np.outer(trA1, P1) / d2
         + 2.0 * np.outer(P1, trA1) * (1.0 / (delta * n)
                                       + trA / (n * d2))
         + 2.0 * trA * P2 / (delta * n))
    return 0.5 * (G + G.T)


def _as_n_nt(arr, nt):
    """Normalise a per-θ family derivative (``Detath``/``Deta2th``/…) to
    shape (n, nt). hea families return (n,) for n_theta==1 and either
    (n, nt) or (nt, n) otherwise."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a[:, None]
    if a.shape[1] == nt:
        return a
    return a.T

def _theta2_arr(arr, nt, n):
    """Normalise a θ-θ second-derivative family array (``Dth2``,
    ``Detath2``, ``Deta2th2``) to shape (n, nt, nt). For n_theta==1 it is
    the single (n,) array; otherwise mgcv packs the upper-triangular
    (i≤k) θ-pairs in column order — unpack to the symmetric (n,nt,nt)."""
    a = np.asarray(arr, dtype=float)
    if nt == 1:
        return a.reshape(n, 1, 1)
    out = np.zeros((n, nt, nt))
    if a.ndim == 3:                       # already (n, nt, nt)
        return a
    # packed (n, npairs) in (i≤k) order: (0,0),(0,1),..,(0,nt-1),(1,1),..
    if a.shape[0] != n:
        a = a.T
    col = 0
    for i in range(nt):
        for k in range(i, nt):
            out[:, i, k] = out[:, k, i] = a[:, col]
            col += 1
    return out

def _dbeta_dp_tw(fit, *, X, wt, y, family):
    """``dβ̂/dp`` for Tweedie at PIRLS-converged β̂ (Fisher score IFT).

    Score equation: X' · u_F(β; p) = S_λ · β̂  with
    ``u_F_i = μ_η_i · (y_i - μ_i)/V_i`` (Fisher form). Differentiating
    in p (PIRLS-converged β̂) and using H = X'·W·X + S_λ for the LHS:

        H · dβ̂/dp = X' · ∂u_F/∂p|_{β̂}
        ∂u_F/∂p|_{β̂} = -μ_η · (y - μ) · log(μ) / V

    because V = μ^p ⇒ ∂(1/V)/∂p = -log(μ)/V.

    Returns a length-p vector. Used by ``_dlog_det_H_dp_tw`` for the
    β-coupled chain in ∂log|H+S|/∂p; consumers should derive the chain
    to θ_tw via ``family.dp_dtheta()``.

    mgcv anchor: the family-parameter (θ for nb/scat, p for tw)
    derivatives of W and log|H+S| come from ``gam.fit4``'s gdi block
    (gam.fit4.r) via the family's ``Dd`` θ-derivatives (Dth/Deta2th).
    """
    mu = fit.mu
    eta = fit.eta
    mu_eta = family.link.mu_eta(eta)
    V = family.variance(mu)
    log_mu = np.log(mu)
    # Weighted Fisher score u_F = wt·μ_η·(y−μ)/V — the prior weights
    # ride along in ∂u_F/∂p.
    duf_dp = -wt * mu_eta * (y - mu) * log_mu / V
    rhs = X.T @ duf_dp
    return cho_solve((fit.A_chol, fit.A_chol_lower), rhs)

def _db_dtheta_fam(fit, *, X, family, y, wt):
    """∂β̂/∂θ_fam — the family-θ columns of mgcv's ``db.drho``
    (gdi computes them through the same Dd chain). Implicit
    differentiation of the penalized score at fixed ρ:

        0 = X'(Dmu·μ'(η))/2 + Sλβ̂   ⇒
        ∂β̂/∂θ_j = −A⁻¹ · X'(Dmuth_j·μ'(η))/2

    with ``A = X'WX + Sλ`` the converged penalized Newton Hessian
    from PIRLS (``fit.A_chol``) and ``Dmuth = ∂²D/∂μ∂θ`` from
    ``family.Dd(level=1)`` (shared via the per-fit ``_Dd`` cache)."""
    dd = _Dd(fit, 1, family=family, y=y, wt=wt)
    mu_eta = family.link.mu_eta(fit.eta)
    dmuth = np.asarray(dd["Dmuth"], dtype=float)
    if dmuth.ndim == 1:
        dmuth = dmuth[None, :]                 # (1, n) for n_theta=1
    elif dmuth.shape[0] != family.n_theta:
        dmuth = dmuth.T                        # (n, nt) → (nt, n)
    rhs = X.T @ (dmuth * mu_eta).T / 2.0    # (p, n_theta)
    return -cho_solve((fit.A_chol, fit.A_chol_lower), rhs)

def _compute_Vc2(rho, fit, Vr, sigma_squared, *, L_pen, slots, p):
    """Cholesky-derivative correction Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T,
    where M_k = ∂L^{-T}/∂ρ_k and A = L L^T is hea's lower-Cholesky of
    ``X'X + Sλ``.

    Differentiating L L^T = A gives ``L^{-1} dA L^{-T}`` whose lower
    triangle (with halved diag) is ``L^{-1} dL`` — the standard
    formula ``dL = L · Φ(L^{-1} dA L^{-T})`` with ``Φ`` zeroing the
    strict upper and halving the diagonal. Then differentiating
    ``L L^{-1} = I``:

        d(L^{-1}) = -L^{-1} dL L^{-1}
        d(L^{-T}) = -L^{-T} (dL)^T L^{-T}     (transpose)

    So M_k = -L^{-T} (dL_k)^T L^{-T}. The ρ-uncertainty in the
    Bayesian draw β̃ = β̂ + σ L^{-T} z propagates as σ Σ_k ε_k M_k z
    with ε ~ N(0, Vr), z ~ N(0, I_p), giving covariance contribution
    σ² Σ_{i,j} Vr[i,j] M_i M_j^T.

    Mirrors mgcv's gam.fit3.post.proc — closes the residual ~0.1 AIC
    gap on bs='re' models that's left after Vc1 alone.

    The Cholesky seed must be the FISHER penalized Hessian on the
    single-formula path: post.proc's ``R`` comes from
    ``qr(sqrt(object$weights)*X)`` and gam.fit3/4 both return
    Fisher-type working weights there (gam.fit4.r:798 "note that
    these are Fisher type weights"; wf = pmax(0, ½·EDeta2) for
    extended families). Callers pass the ``_fisher_view`` fit (or
    fit5's own lbb-root duck) — using the PIRLS Newton factor here
    was the B9 root cause: the entire extended-family Σedf2 ~1e-3
    band (tw measured Vc2 trace 0.24536 vs mgcv 0.23983) traced to
    this one factor choice.
    """
    n_sp = len(slots)
    if n_sp == 0 or sigma_squared <= 0 or not np.isfinite(sigma_squared):
        return np.zeros((p, p))
    # scipy's cho_factor leaves the unused triangle untouched (random
    # memory), so explicitly mask before using as a triangular
    # operand. The factor convention varies by producer (PIRLS'
    # augmented QR stores upper, legacy cho_factor lower) — normalize
    # to the lower form L with A = L L'.
    if fit.A_chol_lower:
        L = np.tril(fit.A_chol)
    else:
        L = np.triu(fit.A_chol).T

    M = np.empty((n_sp, p, p))
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        sp_k = float(np.exp(rho[k]))
        # dA_k = sp_k · S_k embedded at the slot's column range.
        dA = np.zeros((p, p))
        dA[a:b, a:b] = sp_k * slot.S
        # X = L^{-1} dA L^{-T} — two triangular solves.
        Y = solve_triangular(L, dA, lower=True)
        X = solve_triangular(L, Y.T, lower=True).T
        # Φ(X): strict_lower(X) + 0.5·diag(X). Symmetric in floating
        # point because X is symmetric (since dA is symmetric), so we
        # build it from the lower triangle directly.
        Phi = np.tril(X, -1)
        np.fill_diagonal(Phi, 0.5 * np.diag(X))
        dL = L @ Phi
        # M_k = -L^{-T} (dL)^T L^{-T}. Compute as two triangular
        # solves: G = (dL)^T L^{-T} = (L^{-1} dL)^T, then solve
        # L^T M_k = -G.
        G = solve_triangular(L, dL, lower=True).T
        M[k] = solve_triangular(L.T, -G, lower=False)

    # Working-space contraction: M_k is linear in dA_k, and the dA per
    # working θ_j is the L-weighted sum of per-penalty dA's — mgcv
    # Vb.corr's ``dH[[j]] <- Σ_i L[i,j]·dH1[[i]]`` reweighting
    # (gam.fit3.r:915-925). Vr is the working-space covariance.
    if L_pen is not None:
        M = np.einsum("kj,kab->jab", L_pen, M)

    # Vc2[a,b] = Σ_{i,j} Vr[i,j] M_i[a,c] M_j[b,c] — contract over
    # the trailing axis of both M operands.
    Vc2 = np.einsum("ij,iac,jbc->ab", Vr, M, M)
    return sigma_squared * Vc2


def _dW_dtheta_total(fit, dd1=None, *, X, family, y, wt, family_mgcv_extended):
    """Total ∂W/∂θ for an extended family at the converged fit,
    shape (n, n_theta): the direct fixed-β̂ piece ½·Deta2th plus the
    β̂(θ)-coupled piece (∂w/∂η)·(X·∂β̂/∂θ). The Dd-table version of
    the old tw-only ``_dW_dp_tw_total`` (identical values for tw by

    mgcv anchor: the family-parameter (θ for nb/scat, p for tw)
    derivatives of W and log|H+S| come from ``gam.fit4``'s gdi block
    (gam.fit4.r) via the family's ``Dd`` θ-derivatives (Dth/Deta2th).
    the exponential-family identity)."""
    if dd1 is None:
        dd1 = _dDeta(fit, 1, family=family, y=y, wt=wt)
    Deta2th = np.asarray(dd1["Deta2th"], dtype=float)
    if Deta2th.ndim == 1:
        Deta2th = Deta2th[:, None]
    dW_direct = 0.5 * Deta2th                           # (n, nt)
    db_dth = _db_dtheta_fam(fit, X=X, family=family, y=y, wt=wt)                   # (p, nt)
    deta_dth = X @ db_dth                    # (n, nt)
    dw_deta = _dw_deta(fit, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)                        # (n,)
    total = dW_direct + dw_deta[:, None] * deta_dth
    # Rows dropped from the working model contribute nothing.
    keep = (fit.w != 0.0)[:, None] & np.isfinite(total)
    return np.where(keep, total, 0.0)

def _d2beta_theta(fit, rho, *, db_drho, db_dtheta, dd2, X, slots, family, p):
    """∂²β̂/∂θ_a∂ρ_b and ∂²β̂/∂θ_a∂θ_c at PIRLS-converged β̂ — the
    family-θ rows of mgcv's ``b2`` (gdi.c ``ift2``:1412-1457).

    Mechanical port of ift2's second-derivative loop for the θ-involving
    pairs (the sp-sp pairs are ``_d2beta_drho_drho``, already confirmed to
    equal ift2's sp-sp case). For a parameter pair (i, k) ift2 forms

        Db = Xᵀ(−η₁ᵢ·η₁ₖ·Deta3) − t2 − t3 − t4 ;   ∂²β = A⁻¹·(½·Db)

    (the ½ because mgcv's ``PPt = (X'WX+S)⁻¹`` is twice the inverse
    Hessian) with, in joint [θ, ρ] order:

      t2 = Xᵀ(Deta2th_k·η₁ᵢ)   if k is θ  else  2·sp_k·S_k·(∂β/∂param_i)
      t3 = Xᵀ(Deta2th_i·η₁ₖ)   if i is θ  else  2·sp_i·S_i·(∂β/∂param_k)
      t4 = Xᵀ·Detath2_{ik}     if both θ  else (i==k) 2·sp_i·S_i·β  else 0

    η₁ are the first ∂η/∂param (= X·∂β/∂param). Returns
    (∂²β over (θ, ρ); shape (p, nt, n_sp)) and
    (∂²β over (θ, θ'); shape (p, nt, nt)).
    """
    nt = family.n_theta
    n_sp = len(slots)
    n = X.shape[0]
    sp = np.exp(rho)
    chol = (fit.A_chol, fit.A_chol_lower)
    v = X @ db_drho                               # (n, n_sp) η₁_ρ
    eta1_th = X @ db_dtheta                        # (n, nt)   η₁_θ
    Deta3 = np.asarray(dd2["Deta3"], dtype=float)
    Deta2th = _as_n_nt(dd2["Deta2th"], nt)    # (n, nt)
    Detath2 = _theta2_arr(dd2["Detath2"], nt, n)  # (n, nt, nt)

    def Sk_dot(vec_full: np.ndarray, k: int) -> np.ndarray:
        """sp_k·S_k·vec embedded back into the full p-vector."""
        a, b = slots[k].col_start, slots[k].col_end
        out = np.zeros(p)
        out[a:b] = sp[k] * (slots[k].S @ vec_full[a:b])
        return out

    d2b_thr = np.empty((p, nt, n_sp))
    for a in range(nt):
        for b in range(n_sp):
            # i = θ_a (θ), k = ρ_b (sp).
            Db = X.T @ (-eta1_th[:, a] * v[:, b] * Deta3)   # first term
            Db -= 2.0 * Sk_dot(db_dtheta[:, a], b)          # t2 (k sp)
            Db -= X.T @ (Deta2th[:, a] * v[:, b])           # t3 (i θ)
            # t4: i θ, k sp, i≠k → none.
            d2b_thr[:, a, b] = cho_solve(chol, 0.5 * Db)

    d2b_thth = np.empty((p, nt, nt))
    for a in range(nt):
        for c in range(a, nt):
            # i = θ_a, k = θ_c (both θ).
            Db = X.T @ (-eta1_th[:, a] * eta1_th[:, c] * Deta3)
            Db -= X.T @ (Deta2th[:, c] * eta1_th[:, a])     # t2 (k θ)
            Db -= X.T @ (Deta2th[:, a] * eta1_th[:, c])     # t3 (i θ)
            Db -= X.T @ Detath2[:, a, c]                    # t4 (both θ)
            sol = cho_solve(chol, 0.5 * Db)
            d2b_thth[:, a, c] = sol
            if c != a:
                d2b_thth[:, c, a] = sol
    return d2b_thr, d2b_thth

def _dW_dp_tw_total(fit, db_dp=None, *, X, wt, y, family, family_mgcv_extended):
    """Total ∂W/∂p for Tweedie at the converged fit: the direct
    fixed-β̂ piece plus the β̂(p)-coupled piece (see
    ``_dlog_det_H_dp_tw``). Shared by the unprojected log|H+S| θ

    mgcv anchor: the family-parameter (θ for nb/scat, p for tw)
    derivatives of W and log|H+S| come from ``gam.fit4``'s gdi block
    (gam.fit4.r) via the family's ``Dd`` θ-derivatives (Dth/Deta2th).
    derivative and the ML projected-Hessian correction."""
    mu = fit.mu
    eta = fit.eta
    mu_eta = family.link.mu_eta(eta)
    V = family.variance(mu)
    log_mu = np.log(mu)
    # W = wt·α·μ_η²/V — prior weights scale every ∂W/∂p piece.
    muV = wt * mu_eta ** 2 / V

    # Direct ∂W/∂p|_{β̂}.
    if fit.is_fisher_fallback:
        dW_dp_direct = -log_mu * muV
    else:
        alpha = fit.alpha
        dW_dp_direct = muV * ((y - mu) / mu - alpha * log_mu)

    # Indirect via β̂(p).
    if db_dp is None:
        db_dp = _dbeta_dp_tw(fit, X=X, wt=wt, y=y, family=family)
    deta_dp = X @ db_dp
    dw_deta = _dw_deta(fit, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)
    return dW_dp_direct + dw_deta * deta_dp


def _dlog_det_H_dtheta(fit, dd1=None, *, X, family, y, wt, family_mgcv_extended):
    """∂log|H+S|/∂θ for an extended family, shape (n_theta,):
    tr(A⁻¹·X'·diag(∂W/∂θ_j)·X) = Σᵢ dᵢ·(∂W/∂θ_j)ᵢ with
    dᵢ = (X·A⁻¹·X')ᵢᵢ — the family-generic version of the old

    mgcv anchor: the family-parameter (θ for nb/scat, p for tw)
    derivatives of W and log|H+S| come from ``gam.fit4``'s gdi block
    (gam.fit4.r) via the family's ``Dd`` θ-derivatives (Dth/Deta2th).
    ``_dlog_det_H_dp_tw``."""
    dW_dth = _dW_dtheta_total(fit, dd1=dd1, X=X, family=family, y=y, wt=wt, family_mgcv_extended=family_mgcv_extended)        # (n, nt)
    Hinv_Xt = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)
    d = np.einsum("ij,ji->i", X, Hinv_Xt)
    return dW_dth.T @ d

def _dlog_det_H_dp_tw(fit, db_dp=None, *, X, wt, y, family, family_mgcv_extended):
    """``∂log|H+S|/∂p`` for Tweedie. Two pieces:

        ∂H/∂p|_{β̂} (direct):
            ∂W_i/∂p|_{β̂} = (μ_η_i²/V_i) · [(y_i - μ_i)/μ_i - α_i·log(μ_i)]
        indirect via β̂(p):
            ∂W_i/∂p|_{β̂(p)} = (∂W/∂η)_i · (X · dβ̂/dp)_i

    Then tr(H⁻¹ · X'·diag(total)·X) = Σ d_i · total_i with
    ``d_i = (X · A⁻¹ · X')_{ii}`` (same diag-trick the ρ derivative uses).

    For Fisher-fallback fits (α≡1, no Newton corrections) the direct
    piece simplifies to ``-log(μ)·μ_η²/V`` since ∂α/∂p = (y-μ)/μ is
    zeroed out alongside the rest of the Newton α-machinery.

    mgcv anchor: the family-parameter (θ for nb/scat, p for tw)
    derivatives of W and log|H+S| come from ``gam.fit4``'s gdi block
    (gam.fit4.r) via the family's ``Dd`` θ-derivatives (Dth/Deta2th).
    """
    dW_dp_total = _dW_dp_tw_total(fit, db_dp=db_dp, X=X, wt=wt, y=y, family=family, family_mgcv_extended=family_mgcv_extended)

    # tr(A⁻¹ · X'·diag(dW_dp_total)·X) = Σ d_i · dW_dp_total_i
    Hinv_Xt = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)
    d = np.einsum("ij,ji->i", X, Hinv_Xt)
    return float(np.sum(d * dW_dp_total))


def _reml_grad(rho, log_phi=0.0, fit=None, include_log_phi=False, include_family_theta=False, *, Mp, X, gamma, slots, wt, y, family, pearson_scale_criterion, reml_ind, use_ml_proj, p, family_mgcv_extended, penalty_rank, UrS, reparam_cache, dw_deta):
    """Analytical gradient of `_reml` (2·V_R units).

    Length depends on flags:
      * n_sp                                  (defaults)
      * n_sp + 1                              if ``include_log_phi``
      * n_sp + n_lp + family.n_theta          if ``include_family_theta``,
        with the family entries appended last; ``include_log_phi`` is
        then required to be True for unknown-scale Tweedie/tw().

    Wood 2011 §4 + mgcv gam.fit3.r:622, 630:

        ∂(2·V_R)/∂ρ_k    = (∂Dp/∂ρ_k)/φ + ∂log|H|/∂ρ_k − ∂log|S|+/∂ρ_k
        ∂(2·V_R)/∂log φ  = −Dp/φ − 2·ls'_hea − Mp

    For each extra family parameter θ_f (only ``tw`` exercises this
    today; θ_f ↦ Tweedie p via the sigmoid reparametrisation):

        ∂(2·V_R)/∂θ_f =   (∂Dp/∂p · dp/dθ_f) / (φ·γ)
                        − 2·(∂ls0/∂p · dp/dθ_f) / γ
                        + ∂log|H+S|/∂p · dp/dθ_f

    The Dp piece uses the envelope theorem at PIRLS-converged β̂; the
    log|H+S| piece is fully analytical via :meth:`_dlog_det_H_dp_tw`
    (direct ∂W/∂p|_β̂ + indirect via :meth:`_dbeta_dp_tw`); the ls0
    piece comes from :meth:`Tweedie.dls_dp` using the Dunn-Smyth
    ``j_psi_bar`` moment.

    ls'_hea is the d/d(log φ) chain-rule output from `family.ls(y, wt, φ)[1]`
    (hea convention, see family.py:338 docstring).
    """
    n_sp = len(slots)
    phi = float(np.exp(log_phi))
    if not (np.isfinite(phi) and phi > 0):
        size = n_sp + (1 if include_log_phi else 0)
        return np.full(size, 1e15)

    # γ≡1 for the Pearson-Laplace criteria (see `_reml`); `_preml_grad`
    # calls this with include_log_phi=False and adds the φ(ρ) chain term.
    gamma = 1.0 if pearson_scale_criterion else gamma
    if n_sp == 0:
        grad_rho = np.zeros(0)
    else:
        dDp = _dDp_drho(fit, rho, slots)
        dlog_H = _dlog_det_H_drho(fit, rho, X=X, slots=slots, p=p, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)
        rp = _reparam_eval(UrS, reparam_cache, rho)
        dlog_S = (rp["det1"].copy() if rp is not None
                  else _dlog_det_S_drho(rho, S_full=fit.S_full, slots=slots, p=p, penalty_rank=penalty_rank))
        # method="ML" uses the range-only Hessian log-det. With the
        # block-determinant identity log|H_pp+S_pp| = log|H+S| + log|M|
        # (M = U_nᵀ A⁻¹ U_n), the gradient picks up
        #     ∂log|M|/∂ρ_k = −tr(M⁻¹ · B′(∂A/∂ρ_k)B)
        # ∂A/∂ρ_k = X′·diag(∂w/∂ρ_k)·X + λ_k·S_k (the W-dep term is
        # nonzero for non-canonical families like binomial), so the
        # correction has two pieces. Mirrors mgcv's ``MLpenalty1`` →
        # ``get_ddetXWXpS`` in gdi.c, which fills trA1 with the
        # ML-version derivatives via the same projected-Hessian logic.
        if use_ml_proj:
            _, M_inv, B = _ml_logdet_adj(fit, Mp)
            if M_inv is not None:
                sp = np.exp(rho)
                Y = X @ B                  # (n, Mp)
                Y_Minv = Y @ M_inv                    # (n, Mp)
                q = np.einsum("ij,ij->i", Y, Y_Minv)  # (n,) y_i' M⁻¹ y_i
                db_drho = _dbeta_drho(fit, rho, slots, p)
                deta_drho = X @ db_drho     # (n, n_sp)
                dw_drho = dw_deta[:, None] * deta_drho # (n, n_sp)
                for k, slot in enumerate(slots):
                    a, b = slot.col_start, slot.col_end
                    Bk = B[a:b, :]
                    Pk = Bk.T @ slot.S @ Bk
                    # −tr(M⁻¹ Y′ diag(dw/dρ_k) Y) − λ_k tr(M⁻¹ P_k)
                    dlog_H[k] += (
                        -float(np.sum(dw_drho[:, k] * q))
                        - sp[k] * float(np.einsum("ij,ji->", M_inv, Pk))
                    )
        # ∂Dp/∂ρ comes from the data-fit term, so γ divides it; the
        # log|H| / log|S|+ Jacobi pieces are γ-independent.
        grad_rho = dDp / (phi * gamma) + dlog_H - dlog_S

    if not include_log_phi and not include_family_theta:
        return grad_rho

    out = grad_rho
    Dp = fit.dev + fit.pen
    if include_log_phi:
        Mp = float(Mp)
        ls = np.asarray(family.ls(y, wt, phi),
                        dtype=float)
        ls1 = float(ls[1])    # d ls / d(log φ), already chain-ruled
        # Data-fit pieces (-Dp/φ - 2·ls1) divide by γ; the -Mp piece
        # comes from -Mp·log(2πφ) (γ-independent) and is REML-only —
        # under method="ML" remlInd=0 drops it (gam.fit3.r:628).
        d_logphi = (-Dp / phi - 2.0 * ls1) / gamma - reml_ind * Mp
        out = np.concatenate([out, [d_logphi]])

    if include_family_theta and family.n_theta > 0:
        # Family-generic θ block (gam.fit4.r:744 in 2·V_R units):
        #   ∂(2V_R)/∂θ_j = (Σᵢ Dthᵢⱼ)/(φγ) − 2·lsth1_j/γ
        #                  + ∂log|H+S|/∂θ_j
        # The Dp piece is the envelope theorem at PIRLS-converged β̂
        # (∂(D+β'Sβ)/∂β = 0 kills the β-coupled chain); lsth1 comes
        # from the family's extended ls; the log|H+S| piece is the
        # Dd-based dW/dθ trace (direct ½·Deta2th + indirect via
        # ∂β̂/∂θ). For tw this equals the old dD_dp/dls_dp/dp_dtheta
        # chain exactly (tw.Dd's D*th already carry dp/dθ).
        nt = family.n_theta
        theta = family.get_theta()
        dd1 = _dDeta(fit, 1, family=family, y=y, wt=wt)
        Dth = np.asarray(dd1["Dth"], dtype=float)
        if Dth.ndim == 1:
            Dth = Dth[:, None]
        dDp_dth = Dth.sum(axis=0)                       # (nt,)
        ls_ext = family.ls_extended(y, wt, theta=theta,
                                    scale=phi)
        lsth1 = np.asarray(ls_ext["lsth1"], dtype=float).reshape(-1)[:nt]
        dlogH_dth = _dlog_det_H_dtheta(fit, dd1=dd1, X=X, family=family, y=y, wt=wt, family_mgcv_extended=family_mgcv_extended)
        if use_ml_proj:
            # Projected-Hessian correction ∂log|M|/∂θ_j
            # (M = U_n'A⁻¹U_n) — the penalty carries no θ, so only
            # the ∂W/∂θ piece contributes:
            #   ∂log|M|/∂θ_j = −Σ_i (∂W/∂θ_j)_i·q_i,
            #   q_i = (XB)_i'M⁻¹(XB)_i.
            _, M_inv, B = _ml_logdet_adj(fit, Mp)
            if M_inv is not None:
                Y = X @ B
                q = np.einsum("ij,ij->i", Y, Y @ M_inv)
                dW_dth = _dW_dtheta_total(fit, dd1=dd1, X=X, family=family, y=y, wt=wt, family_mgcv_extended=family_mgcv_extended)  # (n, nt)
                dlogH_dth = dlogH_dth - dW_dth.T @ q
        d_theta = (dDp_dth / (phi * gamma)
                   - 2.0 * lsth1 / gamma
                   + dlogH_dth)
        out = np.concatenate([out, d_theta])

    return out

def _reml_hessian(rho, log_phi=0.0, fit=None, include_log_phi=False, include_family_theta=False, *, X, gamma, slots, wt, y, family, p, pearson_scale_criterion, use_ml_proj, penalty_rank, family_mgcv_extended, Mp, UrS, reparam_cache, dw_deta, d2w_deta2):
    """Analytical Hessian of `_reml` (2·V_R units).

    Returns ((n_sp+1) × (n_sp+1)) when ``include_log_phi=True``, else
    (n_sp × n_sp). With ``include_family_theta``, the Hessian is
    further augmented by ``family.n_theta`` rows/columns computed
    **analytically** — the family-θ rows of mgcv's REML2
    (``gam.fit4.r:748``: ``REML2 = ((D2+bSb2)/(2φ) − ls2)/γ + ldet2/2``
    over the joint (θ,ρ) space, with the log φ row from
    ``gam.fit4.r:756-762``). ``D2`` (the deviance Hessian, ``gdi.c``
    ``gdi2``:2145-2166), ``bSb2``/``P2`` (the penalty Hessian, ``get_bSb``
    gdi.c:159-188) and ``ldet2`` (the ``log|H|`` Hessian, ``get_ddetXWXpS``
    gdi.c:911-940) are assembled from the per-obs ``family.dDeta(level=2)``
    tables and the IFT β-derivatives (:meth:`_d2beta_theta`). hea works in
    2·V_R units, so ``hea_H = (D2+bSb2)/(φγ) − 2·ls2/γ + ldet2``. Pinned to
    mgcv's analytic REML2 to ~1e-13.

    Wood 2011 §4 for non-Gaussian, with Newton-form W:

      ∂²(2·V_R)/∂ρ_l∂ρ_k = (1/φ)·∂²Dp/∂ρ_l∂ρ_k
                          + ∂²log|H|/∂ρ_l∂ρ_k
                          − ∂²log|S|+/∂ρ_l∂ρ_k

    Pieces:

      ∂²Dp/∂ρ_l∂ρ_k    = δ_lk·g_k − 2·λ_l·λ_k·β̂' S_l A⁻¹ S_k β̂   (Gaussian form)

      ∂²log|S|+/∂ρ_l∂ρ_k = δ_lk·λ_k·tr(S⁺ S_k)
                          − λ_l·λ_k·tr(S⁺ S_l S⁺ S_k)         (Gaussian form)

      ∂²log|H|/∂ρ_l∂ρ_k = −tr(H⁻¹·∂H/∂ρ_l·H⁻¹·∂H/∂ρ_k)
                          + tr(H⁻¹·∂²H/∂ρ_l∂ρ_k)

    with ∂H/∂ρ_l = X' diag(h'·v_l) X + λ_l S_l (v_l := X·dβ_l) and

      ∂²H/∂ρ_l∂ρ_k = X' diag(h''·v_l·v_k + h'·X·d²β_lk) X
                     + δ_lk·λ_l·S_l

    Cross-derivatives wrt log φ:

      ∂²(2·V_R)/∂ρ_k∂log φ = −g_k / φ
      ∂²(2·V_R)/∂log φ²    = Dp/φ − 2·ls'_hea_2

    where ``ls'_hea_2 = family.ls(y, wt, φ)[2]`` (chain-ruled to log φ).

    Under ``method="ML"`` the log|H| log-det becomes the projected form
    log|U_rᵀ(H+S)U_r|; the additional ∂²log|M_proj| Hessian correction
    (with M_proj = U_nᵀA⁻¹U_n) is added in-loop. Mirrors the gradient
    correction in ``_reml_grad`` and mgcv's ``MLpenalty1`` →
    ``get_ddetXWXpS`` in gdi.c, which fills ``det2`` on the post-drop K,P.

    For Gaussian-identity (h' ≡ h'' ≡ 0) only the SS Wood block and the
    Gaussian Dp/log|S|+ pieces survive, so the result equals 2·`_reml_hessian`
    in the unprofiled REML formulation (the existing `_reml_hessian`
    operates on the φ-profiled Gaussian path and returns V_R-scale).
    """
    n_sp = len(slots)
    phi = float(np.exp(log_phi))
    size = n_sp + (1 if include_log_phi else 0)
    if not (np.isfinite(phi) and phi > 0):
        return np.full((size, size), 1e15)
    gamma = 1.0 if pearson_scale_criterion else gamma
    if n_sp == 0:
        H = np.zeros((size, size))
        if include_log_phi:
            Dp0 = fit.dev + fit.pen
            ls = np.asarray(family.ls(y,
                                           wt, phi))
            H[0, 0] = (Dp0 / phi - 2.0 * float(ls[2])) / gamma
        return H

    sp = np.exp(rho)
    # ∂²log|S|+ via gam.reparam when available (stable under extreme
    # λ ratios); the S⁺-based fallback otherwise.
    rp = _reparam_eval(UrS, reparam_cache, rho)
    S_pinv = None if rp is not None else _S_pinv(fit.S_full, penalty_rank)

    # Common precomputations. The n × n hat matrix ``P = X H⁻¹ X'``
    # and its elementwise square ``Rsq = P*P`` are NOT formed — at
    # large n that's tens of GB. We mirror mgcv (gdi.c:952
    # ``get_trA2``) and operate on ``K`` (n × p) with ``K K' = P``;
    # the bilinear form ``hv_i' Rsq hv_j`` (the only Rsq consumer)
    # equals ``Σ_{p,q} G_i[p,q]·G_j[p,q]`` with ``G_k = K' diag(hv_k) K``.
    # ``M = H⁻¹ X'`` is also gated — it's only needed to feed
    # ``diag_MtSM`` (the WS / SW trace pieces), and those vanish for
    # families with ``dw/dη ≡ 0``. See the ``needs_w`` branch below.

    db_drho = _dbeta_drho(fit, rho, slots, p)                   # (p, n_sp)
    # dw_deta / d2w_deta2 supplied by the shim via the polymorphic
    # self._dw_deta / self._d2w_deta2 (bam overrides to length-p zeros on its
    # reduced Gaussian working model; gam computes the real length-n derivs).
    d2b = _d2beta_drho_drho(fit, rho, db_drho=db_drho,
                                 dw_deta=dw_deta, slots=slots, p=p, X=X, y=y, family_mgcv_extended=family_mgcv_extended, family=family, wt=wt)          # (p, n_sp, n_sp)
    v = X @ db_drho                                        # (n, n_sp)
    hv = dw_deta[:, None] * v                              # h'·v_l, shape (n, n_sp)

    # Build K (n × p) and M (p × n) only when an n-side trace
    # actually needs them. For families with ``dw/dη ≡ 0``
    # (Gaussian-identity, Gamma+log, any canonical link satisfying
    # B(μ) = V'/V + g''·μ_η = 0 and α'/α = 0) all of ``hv``,
    # ``d2w_deta2`` are zero, the K-based traces collapse, and the
    # ``diag_MtSM[k]`` consumers (tr_WS / tr_SW) are hv-weighted and
    # vanish — so neither K nor M is needed. Both are O(p²·n) to
    # build, which dominates at large n.
    needs_w = bool(np.any(hv)) or bool(np.any(d2w_deta2))
    if needs_w:
        M = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)   # (p, n) = H⁻¹ X'
        K = _make_K(fit.A_chol, fit.A_chol_lower, X)       # (n, p)
        d_diag = np.einsum("ij,ij->i", K, K)                  # (n,) diag(KK') = diag(P)
        # G_k = K' diag(hv_k) K, p × p, n_sp of them. Symmetric.
        G_arr = np.empty((n_sp, p, p))
        for k in range(n_sp):
            Khv = K * hv[:, k:k+1]                            # (n, p)
            Gk = K.T @ Khv                                    # (p, p)
            G_arr[k] = 0.5 * (Gk + Gk.T)                      # enforce symmetry
    else:
        K = None    # θ-rows below rebuild K/d_diag when a family with
        M = None    # dw/dη ≡ 0 still carries free θ (uncensored bcg).
        d_diag = None
        G_arr = None

    # Per-slot blocks reused for ∂²Dp / log|S|+ / log|H| Gaussian-style traces.
    AinvS_block: list[np.ndarray] = []
    SpinvS_block: list[np.ndarray] = []
    Sbeta_full = np.zeros((n_sp, p))
    AinvSbeta = np.empty((n_sp, p))
    diag_MtSM: list[np.ndarray] | None = (
        [] if needs_w else None
    )  # diag(M' S_k_full M) = (n,) per k; only needed for tr_WS / tr_SW.
    g = np.zeros(n_sp)
    tr_AinvS = np.zeros(n_sp)
    tr_SpinvS = np.zeros(n_sp)
    A_inv = cho_solve((fit.A_chol, fit.A_chol_lower), np.eye(p))
    for k, slot in enumerate(slots):
        a, b = slot.col_start, slot.col_end
        beta_k = fit.beta[a:b]
        Sb = slot.S @ beta_k
        Sbeta_full[k, a:b] = Sb
        AinvSbeta[k] = cho_solve(
            (fit.A_chol, fit.A_chol_lower), Sbeta_full[k]
        )
        g[k] = sp[k] * float(beta_k @ Sb)
        AinvS_block.append(A_inv[:, a:b] @ slot.S)
        tr_AinvS[k] = float(np.einsum("ij,ji->", A_inv[a:b, a:b], slot.S))
        if S_pinv is not None:
            SpinvS_block.append(S_pinv[:, a:b] @ slot.S)
            tr_SpinvS[k] = float(np.einsum(
                "ij,ji->", S_pinv[a:b, a:b], slot.S
            ))
        if diag_MtSM is not None:
            # diag(M' S_k_full M)_i = M[a:b, i]' · S_k · M[a:b, i]
            SkM = slot.S @ M[a:b, :]                       # (m_k, n)
            diag_MtSM.append(np.einsum("ji,ji->i", M[a:b, :], SkM))

    # ML range-projection correction. Under method="ML" the Hessian
    # log-det is log|U_rᵀ(H+S)U_r|, which by the block-determinant
    # identity equals log|H+S| + log|M_proj| with M_proj = U_nᵀ A⁻¹ U_n
    # (B_proj = A⁻¹ U_n; see ``_ml_logdet_adj``). The Hessian therefore
    # gains the ∂²log|M_proj|/∂ρ_l∂ρ_k term:
    #
    #     ∂²log|M_proj|/∂ρ_l∂ρ_k = −tr(M_proj⁻¹·∂M_proj_l·M_proj⁻¹·∂M_proj_k)
    #                             + tr(M_proj⁻¹·∂²M_proj_lk)
    #
    # with ∂M_proj_k = −B_projᵀ·(∂A/∂ρ_k)·B_proj and
    #   ∂²M_proj_lk = B_projᵀ·(∂A/∂ρ_l)·A⁻¹·(∂A/∂ρ_k)·B_proj
    #              + B_projᵀ·(∂A/∂ρ_k)·A⁻¹·(∂A/∂ρ_l)·B_proj
    #              − B_projᵀ·(∂²A/∂ρ_l∂ρ_k)·B_proj
    # ∂A/∂ρ_k = X'·diag(h'·v_k)·X + λ_k·S_k_full and ∂²A/∂ρ_l∂ρ_k as in
    # the comment above. Mirrors mgcv's ``MLpenalty1`` → ``get_ddetXWXpS``
    # in gdi.c, which fills ``det2`` from the projected K, P.
    ml_active = False
    if use_ml_proj:
        _, M_proj_inv, B_proj = _ml_logdet_adj(fit, Mp)
        if M_proj_inv is not None:
            ml_active = True
            Mp_dim = M_proj_inv.shape[0]
            Y_proj = X @ B_proj                                 # (n, Mp)
            Y_proj_Minv = Y_proj @ M_proj_inv                   # (n, Mp)
            q_vec = np.einsum("ij,ij->i", Y_proj, Y_proj_Minv)  # (n,)
            Yk_arr = np.zeros((n_sp, p, Mp_dim))
            Zk_arr = np.zeros((n_sp, p, Mp_dim))
            Pk_M_arr = np.zeros(n_sp)
            Minv_dMk_arr = np.zeros((n_sp, Mp_dim, Mp_dim))
            for kk, slot_kk in enumerate(slots):
                a_k, b_k = slot_kk.col_start, slot_kk.col_end
                Yk_ = X.T @ (hv[:, kk:kk + 1] * Y_proj)
                Yk_[a_k:b_k, :] += sp[kk] * (slot_kk.S @ B_proj[a_k:b_k, :])
                Yk_arr[kk] = Yk_
                Zk_arr[kk] = cho_solve(
                    (fit.A_chol, fit.A_chol_lower), Yk_
                )
                Minv_dMk_arr[kk] = M_proj_inv @ (-B_proj.T @ Yk_)
                Bk_proj = B_proj[a_k:b_k, :]
                Pk_M_arr[kk] = float(np.einsum(
                    "ij,ji->", M_proj_inv, Bk_proj.T @ slot_kk.S @ Bk_proj
                ))

    # Hessian assembly — symmetric loop.
    H2 = np.zeros((n_sp, n_sp))
    for i in range(n_sp):
        a_i, b_i = slots[i].col_start, slots[i].col_end
        for j in range(i, n_sp):
            a_j, b_j = slots[j].col_start, slots[j].col_end

            # ∂²Dp/∂ρ_i∂ρ_j: same family-agnostic form as Gaussian.
            bSiAinvSj_b = float(Sbeta_full[i] @ AinvSbeta[j])
            d2Dp = -2.0 * sp[i] * sp[j] * bSiAinvSj_b

            # tr(H⁻¹·∂H/∂ρ_i·H⁻¹·∂H/∂ρ_j) — four pieces.
            # WW: (h'·v_i)' · Rsq · (h'·v_j) where Rsq = P⊙P, P = KK'.
            # Identity: hv_i' (KK' ⊙ KK') hv_j = Σ_{p,q} G_i[p,q]·G_j[p,q]
            # with G_k = K' diag(hv_k) K (Wood 2008 §4 + mgcv gdi.c:952).
            # WS / SW: tr(H⁻¹·A_i·H⁻¹·S_j) = (h'·v_i)' · diag_MtSM[j].
            # All three are zero when ``hv ≡ 0`` (Gaussian-identity etc.).
            if G_arr is not None:
                tr_WW = float(np.sum(G_arr[i] * G_arr[j]))
                tr_WS = float(hv[:, i] @ diag_MtSM[j])
                tr_SW = float(hv[:, j] @ diag_MtSM[i])
            else:
                tr_WW = 0.0
                tr_WS = 0.0
                tr_SW = 0.0
            # SS: tr(H⁻¹·S_i·H⁻¹·S_j) — Gaussian block trick.
            tr_SS = float(np.einsum(
                "ab,ba->",
                AinvS_block[i][a_j:b_j, :],
                AinvS_block[j][a_i:b_i, :],
            ))
            tr_HinvHpHinvHp = (
                tr_WW
                + sp[j] * tr_WS
                + sp[i] * tr_SW
                + sp[i] * sp[j] * tr_SS
            )

            # tr(H⁻¹·∂²H/∂ρ_i∂ρ_j).
            #   X'·diag(h''·v_i·v_j)·X contribution: Σ d_i·h''·v_i·v_j.
            #   X'·diag(h'·X·d²β_ij)·X        contribution: Σ d_i·h'·(X·d²β_ij).
            # Both are weighted by ``d_diag = diag(K K')``; if K wasn't
            # built (W-derivs identically zero) both summands vanish.
            Xd2b = X @ d2b[:, i, j]                       # (n,)
            if d_diag is not None:
                tr_d2H = (
                    float(np.sum(d_diag * d2w_deta2 * v[:, i] * v[:, j]))
                    + float(np.sum(d_diag * dw_deta * Xd2b))
                )
            else:
                tr_d2H = 0.0
            # δ_lk·λ_l·tr(H⁻¹·S_l) is the off-square diagonal term.
            d2logH_ij = -tr_HinvHpHinvHp + tr_d2H

            if ml_active:
                # ∂²log|M_proj|/∂ρ_i∂ρ_j = T1 + T2_mixed − T2_d2A:
                #   T1       = −tr(M_proj⁻¹·∂M_proj_i·M_proj⁻¹·∂M_proj_j)
                #   T2_mixed = 2·tr(M_proj⁻¹·Y_iᵀ·Z_j)
                #   T2_d2A   = tr(M_proj⁻¹·B_projᵀ·(∂²A_ij)·B_proj)
                # T2_mixed uses Y_iᵀ Z_j = Y_iᵀ A⁻¹ Y_j (symmetric in i,j up
                # to transpose of an inside Mp×Mp block; M_proj⁻¹ symmetric
                # → both orders give the same trace, so the factor of 2
                # absorbs the symmetric pair).
                T1_ml = -float(np.einsum(
                    "ab,ba->", Minv_dMk_arr[i], Minv_dMk_arr[j]
                ))
                T2_mixed = 2.0 * float(np.einsum(
                    "ab,ba->", M_proj_inv, Yk_arr[i].T @ Zk_arr[j]
                ))
                D_ij_diag = (
                    d2w_deta2 * v[:, i] * v[:, j]
                    + dw_deta * Xd2b
                )
                T2_d2A = float(np.sum(D_ij_diag * q_vec))
                if i == j:
                    # δ_ij·λ_i·tr(M_proj⁻¹·B_projᵀ·S_i_full·B_proj) from
                    # ∂²A's penalty piece.
                    T2_d2A += sp[i] * Pk_M_arr[i]
                d2logH_ij += T1_ml + T2_mixed - T2_d2A

            # ∂²log|S|+/∂ρ_i∂ρ_j: gam.reparam's det2 carries the full
            # matrix (off-diagonal −λλ·tr(S⁻¹S_iS⁻¹S_j), diagonal
            # +det1); the S⁺ fallback assembles the same pieces.
            if rp is not None:
                cross_2VR = (d2Dp / (phi * gamma) + d2logH_ij
                             - rp["det2"][i, j])
                if i == j:
                    H2[i, i] = (
                        cross_2VR
                        + g[i] / (phi * gamma)
                        + sp[i] * tr_AinvS[i]
                    )
                else:
                    H2[i, j] = H2[j, i] = cross_2VR
            else:
                tr_SpSiSpSj = float(np.einsum(
                    "ab,ba->",
                    SpinvS_block[i][a_j:b_j, :],
                    SpinvS_block[j][a_i:b_i, :],
                ))
                d2logS_ij = -sp[i] * sp[j] * tr_SpSiSpSj

                cross_2VR = d2Dp / (phi * gamma) + d2logH_ij - d2logS_ij
                if i == j:
                    # Diagonal also picks up the δ_lk·g_k from ∂²Dp,
                    # δ_lk·λ_l·tr(H⁻¹·S_l) from ∂²H, and
                    # δ_lk·λ_k·tr(S⁺ S_k) from ∂²log|S|+. Only the
                    # ∂²Dp piece is γ-scaled.
                    H2[i, i] = (
                        cross_2VR
                        + g[i] / (phi * gamma)
                        + sp[i] * tr_AinvS[i]
                        - sp[i] * tr_SpinvS[i]
                    )
                else:
                    H2[i, j] = H2[j, i] = cross_2VR

    if not include_log_phi and not include_family_theta:
        return H2

    if include_log_phi:
        # Augment with log φ row/col. Cross / log φ² come from the
        # data-fit term (Dp/φ − 2·ls0), so they scale by 1/γ.
        H_aug = np.zeros((n_sp + 1, n_sp + 1))
        H_aug[:n_sp, :n_sp] = H2
        for k in range(n_sp):
            cross = -g[k] / (phi * gamma)
            H_aug[k, n_sp] = cross
            H_aug[n_sp, k] = cross
        Dp = fit.dev + fit.pen
        ls = np.asarray(family.ls(y, wt, phi))
        H_aug[n_sp, n_sp] = (Dp / phi - 2.0 * float(ls[2])) / gamma
    else:
        # Scale-known extended family (scat): θ-vector is (ρ, θ_fam)
        # with no log φ slot — matches mgcv's gam.fit4 layout where
        # the scale row only exists when scale < 0.
        H_aug = H2

    if not include_family_theta or family.n_theta == 0:
        return H_aug

    # Family-θ rows/cols — analytic port of mgcv's gam.fit4.r:748 REML2
    # θ-rows: REML2 = ((D2+bSb2)/(2φ) − ls2)/γ + ldet2/2 over the joint
    # (θ,ρ) space, with the log φ row/col bolted on (gam.fit4.r:756-762).
    # D2 (gdi.c:2145-2166), bSb2/P2 (get_bSb gdi.c:159-188) and ldet2
    # (get_ddetXWXpS gdi.c:911-940) are computed from the per-obs Dd
    # tables + the IFT β-derivatives. hea works in 2·V_R units, so
    # hea_H = (D2+bSb2)/(φγ) − 2·ls2/γ + ldet2. Replaces the former
    # central-difference of _reml_grad (a non-mechanical FD shortcut).
    n_extra = family.n_theta
    base_size = n_sp + (1 if include_log_phi else 0)
    new_size = base_size + n_extra
    H_full = np.zeros((new_size, new_size))
    H_full[:base_size, :base_size] = H_aug
    nt = n_extra

    # Per-obs deviance derivatives (η-space; mgcv Det*/Dth* names) and the
    # θ first/second β-derivatives via the IFT.
    dd2 = _dDeta(fit, 2, family=family, y=y, wt=wt)
    Deta = np.asarray(dd2["Deta"], dtype=float)
    Deta2 = np.asarray(dd2["Deta2"], dtype=float)
    Deta3 = np.asarray(dd2["Deta3"], dtype=float)
    Deta4 = np.asarray(dd2["Deta4"], dtype=float)
    Detath = _as_n_nt(dd2["Detath"], nt)          # (n, nt)
    Deta3th = _as_n_nt(dd2["Deta3th"], nt)        # (n, nt)
    Dth = _as_n_nt(dd2["Dth"], nt)                # (n, nt)
    Dth2 = _theta2_arr(dd2["Dth2"], nt, X.shape[0])       # (n,nt,nt)
    Deta2th2 = _theta2_arr(dd2["Deta2th2"], nt, X.shape[0])

    db_dtheta = _db_dtheta_fam(fit, X=X, family=family, y=y, wt=wt)               # (p, nt)
    eta1_th = X @ db_dtheta                             # (n, nt) η₁_θ
    dW_dth = _dW_dtheta_total(fit, X=X, family=family, y=y, wt=wt, family_mgcv_extended=family_mgcv_extended)                # (n, nt) ∂w/∂θ
    d2b_thr, d2b_thth = _d2beta_theta(
        fit, rho, db_drho=db_drho, db_dtheta=db_dtheta, dd2=dd2, X=X, slots=slots, family=family, p=p)
    eta2_thr = np.einsum("ij,jab->iab", X, d2b_thr)    # (n, nt, n_sp)
    eta2_thth = np.einsum("ij,jac->iac", X, d2b_thth)  # (n, nt, nt)

    # Penalty pieces (get_bSb): Sβ_total and the embedded S·v.
    S_beta = (sp[:, None] * Sbeta_full).sum(axis=0)    # Σ sp_k S_k β

    def _S_total_dot(vec_full: np.ndarray) -> np.ndarray:
        out = np.zeros(p)
        for kk, slot_kk in enumerate(slots):
            aa, bb = slot_kk.col_start, slot_kk.col_end
            out[aa:bb] += sp[kk] * (slot_kk.S @ vec_full[aa:bb])
        return out

    # ldet2 traces reuse the ρρ K/M machinery; build any pieces the ρρ
    # pass skipped (needs_w False ⇒ ∂w/∂ρ≡0 so G_arr≡0, but the θ rows
    # still need diag(KK'), M and diag(M'S_kM)).
    if K is None:
        K = _make_K(fit.A_chol, fit.A_chol_lower, X)
        d_diag = np.einsum("ij,ij->i", K, K)
    if M is None:
        M = cho_solve((fit.A_chol, fit.A_chol_lower), X.T)
    if G_arr is None:
        G_arr = np.zeros((n_sp, p, p))
    if diag_MtSM is None:
        diag_MtSM = []
        for kk, slot_kk in enumerate(slots):
            aa, bb = slot_kk.col_start, slot_kk.col_end
            SkM = slot_kk.S @ M[aa:bb, :]
            diag_MtSM.append(np.einsum("ji,ji->i", M[aa:bb, :], SkM))
    # Gθ[a] = K' diag(∂w/∂θ_a) K — the θ analogue of G_arr.
    Gth = np.empty((nt, p, p))
    for a in range(nt):
        G_a = K.T @ (K * dW_dth[:, a:a + 1])
        Gth[a] = 0.5 * (G_a + G_a.T)

    # ML range-projection θ-corrections (mirrors the ρρ ml block, with
    # ∂A/∂θ_a = X'diag(∂w/∂θ_a)X — θ carries no penalty so the S term
    # drops). Only built under method="ML".
    if ml_active:
        Mp_dim = M_proj_inv.shape[0]
        Yth_arr = np.zeros((nt, p, Mp_dim))
        Zth_arr = np.zeros((nt, p, Mp_dim))
        Minv_dMth_arr = np.zeros((nt, Mp_dim, Mp_dim))
        for a in range(nt):
            Yth = X.T @ (dW_dth[:, a:a + 1] * Y_proj)
            Yth_arr[a] = Yth
            Zth_arr[a] = cho_solve((fit.A_chol, fit.A_chol_lower), Yth)
            Minv_dMth_arr[a] = M_proj_inv @ (-B_proj.T @ Yth)

    # Saturated-likelihood 2nd derivatives (θ,θ) and (θ,log φ): ls2.
    ls_ext = family.ls_extended(y, wt,
                                theta=family.get_theta(), scale=phi)
    lsth2 = np.asarray(ls_ext["lsth2"], dtype=float)   # (nt+1, nt+1)

    for a in range(nt):
        col_idx = base_size + a
        # θ_a × ρ_i rows.
        for i in range(n_sp):
            d2_dev = float(np.sum(
                Deta2 * eta1_th[:, a] * v[:, i]
                + Deta * eta2_thr[:, a, i]
                + Detath[:, a] * v[:, i]))
            d2_pen = (
                2.0 * float(d2b_thr[:, a, i] @ S_beta)
                + 2.0 * float(db_drho[:, i] @ _S_total_dot(db_dtheta[:, a]))
                + 2.0 * float(db_dtheta[:, a] @ (sp[i] * Sbeta_full[i])))
            d2w = 0.5 * (Deta4 * eta1_th[:, a] * v[:, i]
                         + Deta3 * eta2_thr[:, a, i]
                         + Deta3th[:, a] * v[:, i])
            d2logH = (
                -(float(np.sum(Gth[a] * G_arr[i]))
                  + sp[i] * float(dW_dth[:, a] @ diag_MtSM[i]))
                + float(np.sum(d_diag * d2w)))
            if ml_active:
                T1 = -float(np.einsum(
                    "ab,ba->", Minv_dMth_arr[a], Minv_dMk_arr[i]))
                T2 = 2.0 * float(np.einsum(
                    "ab,ba->", M_proj_inv, Yth_arr[a].T @ Zk_arr[i]))
                d2logH += T1 + T2 - float(np.sum(d2w * q_vec))
            val = (d2_dev + d2_pen) / (phi * gamma) + d2logH
            H_full[i, col_idx] = H_full[col_idx, i] = val
        # θ_a × log φ row (gam.fit4.r:756-757, 2·V_R units).
        if include_log_phi:
            val = (-float(np.sum(Dth[:, a])) / (phi * gamma)
                   - 2.0 * lsth2[a, nt] / gamma)
            H_full[n_sp, col_idx] = H_full[col_idx, n_sp] = val
        # θ_a × θ_c block.
        for c in range(a, nt):
            d2_dev = float(np.sum(
                Deta2 * eta1_th[:, a] * eta1_th[:, c]
                + Deta * eta2_thth[:, a, c]
                + Dth2[:, a, c]
                + Detath[:, a] * eta1_th[:, c]
                + Detath[:, c] * eta1_th[:, a]))
            # ∂²(β'Sβ)/∂θ_a∂θ_c = 2·(∂²β)'Sβ + 2·(∂β/∂θ_c)'S(∂β/∂θ_a)
            # (get_bSb gdi.c:167-172). NO diagonal `+bSb1` term for θ:
            # bSb1[θ]=0 during get_bSb's Hessian loop (gdi.c:156; the
            # `2·b1'Sb` augmentation at gdi.c:194 runs *after*), and S
            # carries no θ-dependence so there is no penalty-curvature term.
            d2_pen = (
                2.0 * float(d2b_thth[:, a, c] @ S_beta)
                + 2.0 * float(db_dtheta[:, c]
                              @ _S_total_dot(db_dtheta[:, a])))
            d2w = 0.5 * (Deta4 * eta1_th[:, a] * eta1_th[:, c]
                         + Deta3 * eta2_thth[:, a, c]
                         + Deta3th[:, a] * eta1_th[:, c]
                         + Deta3th[:, c] * eta1_th[:, a]
                         + Deta2th2[:, a, c])
            d2logH = (-float(np.sum(Gth[a] * Gth[c]))
                      + float(np.sum(d_diag * d2w)))
            if ml_active:
                T1 = -float(np.einsum(
                    "ab,ba->", Minv_dMth_arr[a], Minv_dMth_arr[c]))
                T2 = 2.0 * float(np.einsum(
                    "ab,ba->", M_proj_inv, Yth_arr[a].T @ Zth_arr[c]))
                d2logH += T1 + T2 - float(np.sum(d2w * q_vec))
            val = ((d2_dev + d2_pen) / (phi * gamma)
                   - 2.0 * lsth2[a, c] / gamma + d2logH)
            ci = base_size + c
            H_full[col_idx, ci] = H_full[ci, col_idx] = val
    return H_full


def _preml_grad(rho, fit, *, slots, Mp, n, y, wt, family, reml_ind, X, p, gamma, pearson_scale_criterion, use_ml_proj, family_mgcv_extended, penalty_rank, UrS, reparam_cache, dw_deta):
    """Gradient of the P-REML/P-ML criterion in hea's 2·V units
    (length n_sp), mgcv gam.fit3.r:656 (×2).

    φ = P/(n−Mp) is the Pearson-Laplace scale, a function of ρ, so the
    criterion's ρ-gradient is the plain-(RE)ML ρ-block (reused from
    `_reml_grad`, which runs at γ≡1 under `_pearson_scale`) minus the
    φ(ρ) chain term:

        2V1_k = [Dp1_k/φ + ∂log|H|_k − ∂log|S|₊_k]        (= `_reml_grad`)
                − φ1_k·(Dp/φ² + Mp/φ·remlInd + 2·ls'(φ))

    with φ1_k = P1_k/(n−Mp), ls'(φ) = family.ls[1]/φ (hea's ls is the
    d/d(logφ) value, so dividing by φ recovers mgcv's dls/dφ = ls[2]).
    The ∂log|H| block carries the ML range-projection for P-ML via the
    `_use_ml_proj` predicate inside `_reml_grad`.
    """
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros(0)
    phi = _phi_pearson(fit, Mp=Mp, n=n, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    log_phi = float(np.log(max(phi, 1e-300)))
    base = _reml_grad(rho, log_phi, fit=fit, include_log_phi=False, Mp=Mp, X=X, gamma=gamma, slots=slots, wt=wt, y=y, family=family, pearson_scale_criterion=pearson_scale_criterion, reml_ind=reml_ind, use_ml_proj=use_ml_proj, p=p, family_mgcv_extended=family_mgcv_extended, penalty_rank=penalty_rank, UrS=UrS, reparam_cache=reparam_cache, dw_deta=dw_deta)
    Dp = float(fit.dev + fit.pen)
    Mp = float(Mp)
    denom = float(n - Mp)
    _, P1 = _pearson_and_deriv(rho, fit, deriv=True, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    phi1 = (P1 / denom) if denom > 0 else P1
    ls1 = float(family.ls(y, wt, phi)[1])
    ls_dphi = ls1 / phi                       # mgcv's ls[2] = dls/dφ
    corr = phi1 * (Dp / (phi * phi) + Mp / phi * reml_ind + 2.0 * ls_dphi)
    return base - corr

def _preml_hessian(rho, fit=None, *, slots, Mp, n, family, wt, y, X, p, gamma, pearson_scale_criterion, use_ml_proj, penalty_rank, family_mgcv_extended, UrS, reparam_cache, reml_ind, dw_deta, d2w_deta2):
    """Analytic P-REML / P-ML Hessian ``REML2`` (n_sp × n_sp, mgcv's
    V-scale), equal to mgcv gam.fit3.r:658-664.

    The P-REML/P-ML score is the Laplace (RE)ML score with the
    Pearson-Laplace scale φ_P(ρ) = P(ρ)/(n−Mp) plugged in for the scale
    (mgcv.r coerces only the *known*-scale case to REML; for unknown
    scale this Pearson plug-in is what distinguishes P-REML from REML).
    So V_P(ρ) = V_REML(ρ, log φ_P(ρ)) and the ρ-Hessian is the chain
    rule over hea's analytic `_reml_hessian` (which supplies the
    (ρ,ρ), (ρ,log φ) and (log φ,log φ) blocks and the log φ gradient):

        ∂²V_P/∂ρ_i∂ρ_j = Hρρ[i,j] + Hρφ[i]·u_j + Hρφ[j]·u_i
                        + Hφφ·u_i·u_j + g_φ·u_ij

    with u = log φ_P, u_i = ∂log P/∂ρ_i = P1_i/P,
    u_ij = ∂²log P/∂ρ_i∂ρ_j = P2_ij/P − P1_i·P1_j/P². hea's
    `_reml_hessian`/`_reml_grad` are in 2·V units (and run at γ≡1 for
    the Pearson criteria), so the assembled 2·V Hessian is halved to
    mgcv's V-scale. Replaces the former FD Hessian on `_preml_grad`."""
    n_sp = len(slots)
    if n_sp == 0:
        return np.zeros((0, 0))
    phi = _phi_pearson(fit, Mp=Mp, n=n, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    log_phi = float(np.log(max(phi, 1e-300)))
    # 2·V REML grad/Hessian augmented with the log φ row/col.
    H_full = _reml_hessian(rho, log_phi, fit=fit,
                                include_log_phi=True, X=X, gamma=gamma, slots=slots, wt=wt, y=y, family=family, p=p, pearson_scale_criterion=pearson_scale_criterion, use_ml_proj=use_ml_proj, penalty_rank=penalty_rank, family_mgcv_extended=family_mgcv_extended, Mp=Mp, UrS=UrS, reparam_cache=reparam_cache, dw_deta=dw_deta, d2w_deta2=d2w_deta2)
    g_full = _reml_grad(rho, log_phi, fit=fit, include_log_phi=True, Mp=Mp, X=X, gamma=gamma, slots=slots, wt=wt, y=y, family=family, pearson_scale_criterion=pearson_scale_criterion, reml_ind=reml_ind, use_ml_proj=use_ml_proj, p=p, family_mgcv_extended=family_mgcv_extended, penalty_rank=penalty_rank, UrS=UrS, reparam_cache=reparam_cache, dw_deta=dw_deta)
    Hrr = H_full[:n_sp, :n_sp]
    Hrf = H_full[:n_sp, n_sp]
    Hff = float(H_full[n_sp, n_sp])
    gf = float(g_full[n_sp])
    # Pearson scale derivatives: u = log φ_P = log P − log(n−Mp).
    P, P1 = _pearson_and_deriv(rho, fit, deriv=True, family=family, wt=wt, y=y, slots=slots, X=X, p=p)
    P2 = _pearson_hess(fit, rho, X=X, slots=slots, wt=wt, y=y, family=family, p=p, family_mgcv_extended=family_mgcv_extended)
    u1 = P1 / P                                       # ∂log φ_P/∂ρ
    u2 = P2 / P - np.outer(P1, P1) / (P * P)          # ∂²log φ_P/∂ρ²
    REML2_2V = (Hrr
                + np.outer(Hrf, u1) + np.outer(u1, Hrf)
                + Hff * np.outer(u1, u1)
                + gf * u2)
    return 0.5 * (REML2_2V + REML2_2V.T) / 2.0


def _gam_fit3_score(rho, log_phi, fit, deriv, scoreType, *,
                    T_work, include_log_phi, include_family_theta,
                    dw_deta, d2w_deta2,
                    Mp, wt, y, binom_n, gamma, family, family_mgcv_extended,
                    use_ml_proj, pearson_scale_criterion, reml_ind,
                    penalty_rank, slots, p, UrS, reparam_cache,
                    X, XtX, n, scale_fixed_value, scale_known, pls_lwork):
    """mgcv ``gam.fit3`` score tail (gam.fit3.r:538-864) — assemble the
    smoothness-selection criterion and its ρ-derivatives from a converged
    PIRLS fit, dispatched on ``scoreType`` exactly as mgcv's ``scoreType``
    branch. Returns ``{"score", "grad", "hess"}`` with grad/hess already in
    WORKING coords (``T'g`` / ``T'HT``, ``T`` = blockdiag(L, I_extra)).

    ``deriv`` mirrors mgcv's argument: 0 → score only, 1 → +grad, 2 →
    +grad+hess (the caller passes 0 on rejected step-halving trials and 2
    on accepted points, so grad/hess aren't wasted). The general
    (gam.fit5/REML5) path is the fitter's own monolithic REML/REML1/REML2
    return and is handled by the caller, not here.

    ``dw_deta`` / ``d2w_deta2`` are the working-weight derivatives supplied
    by the caller's shim (polymorphic — bam overrides them; see the SB1
    fold); only the REML/P-REML derivative branches consume them."""
    def _tw(g):
        return g if T_work is None else T_work.T @ g

    def _twh(H):
        return H if T_work is None else T_work.T @ H @ T_work

    grad = hess = None
    if scoreType == "REML":
        score = 0.5 * _reml(
            rho, log_phi, fit, Mp=Mp, wt=wt, y=y, binom_n=binom_n,
            gamma=gamma, family=family,
            family_mgcv_extended=family_mgcv_extended, use_ml_proj=use_ml_proj,
            pearson_scale_criterion=pearson_scale_criterion, reml_ind=reml_ind,
            penalty_rank=penalty_rank, slots=slots, p=p, UrS=UrS,
            reparam_cache=reparam_cache)
        if deriv >= 1:
            grad = _tw(0.5 * _reml_grad(
                rho, log_phi, fit, include_log_phi, include_family_theta,
                Mp=Mp, X=X, gamma=gamma, slots=slots, wt=wt, y=y, family=family,
                pearson_scale_criterion=pearson_scale_criterion,
                reml_ind=reml_ind, use_ml_proj=use_ml_proj, p=p,
                family_mgcv_extended=family_mgcv_extended,
                penalty_rank=penalty_rank, UrS=UrS, reparam_cache=reparam_cache,
                dw_deta=dw_deta))
        if deriv >= 2:
            hess = _twh(0.5 * _reml_hessian(
                rho, log_phi, fit, include_log_phi, include_family_theta,
                X=X, gamma=gamma, slots=slots, wt=wt, y=y, family=family, p=p,
                pearson_scale_criterion=pearson_scale_criterion,
                use_ml_proj=use_ml_proj, penalty_rank=penalty_rank,
                family_mgcv_extended=family_mgcv_extended, Mp=Mp, UrS=UrS,
                reparam_cache=reparam_cache, dw_deta=dw_deta,
                d2w_deta2=d2w_deta2))
    elif scoreType == "GCV":
        score = _gcv(
            rho, fit, gamma=gamma, n=n, scale_fixed_value=scale_fixed_value,
            scale_known=scale_known, X=X, XtX=XtX, p=p, family=family,
            family_mgcv_extended=family_mgcv_extended, y=y, wt=wt,
            pls_lwork=pls_lwork)
        if deriv >= 1:
            grad = _tw(_gcv_grad(
                rho, fit, gamma=gamma, slots=slots, n=n,
                scale_fixed_value=scale_fixed_value, scale_known=scale_known,
                X=X, XtX=XtX, family=family, p=p, y=y,
                family_mgcv_extended=family_mgcv_extended, wt=wt,
                pls_lwork=pls_lwork))
        if deriv >= 2:
            hess = _twh(_gcv_hessian(
                rho, fit, X=X, XtX=XtX, gamma=gamma, slots=slots, n=n, p=p,
                scale_fixed_value=scale_fixed_value, scale_known=scale_known,
                y=y, family_mgcv_extended=family_mgcv_extended, family=family,
                wt=wt, pls_lwork=pls_lwork))
    elif scoreType == "GACV":
        score = _gacv(
            rho, fit, gamma=gamma, n=n, X=X, XtX=XtX, p=p, family=family,
            family_mgcv_extended=family_mgcv_extended, y=y, wt=wt,
            pls_lwork=pls_lwork, slots=slots)
        if deriv >= 1:
            grad = _tw(_gacv_grad(
                rho, fit, gamma=gamma, slots=slots, n=n, X=X, XtX=XtX,
                family=family, p=p, y=y,
                family_mgcv_extended=family_mgcv_extended, wt=wt,
                pls_lwork=pls_lwork))
        if deriv >= 2:
            hess = _twh(_gacv_hessian(
                rho, fit, gamma=gamma, slots=slots, n=n, X=X, XtX=XtX, p=p,
                scale_fixed_value=scale_fixed_value, scale_known=scale_known,
                y=y, family_mgcv_extended=family_mgcv_extended, family=family,
                wt=wt, pls_lwork=pls_lwork))
    elif scoreType == "PREML":
        phi = _phi_pearson(fit, Mp=Mp, n=n, family=family, wt=wt, y=y,
                           slots=slots, X=X, p=p)
        score = 0.5 * _reml(
            rho, float(np.log(max(phi, 1e-300))), fit, Mp=Mp, wt=wt, y=y,
            binom_n=binom_n, gamma=gamma, family=family,
            family_mgcv_extended=family_mgcv_extended, use_ml_proj=use_ml_proj,
            pearson_scale_criterion=pearson_scale_criterion, reml_ind=reml_ind,
            penalty_rank=penalty_rank, slots=slots, p=p, UrS=UrS,
            reparam_cache=reparam_cache)
        if deriv >= 1:
            grad = _tw(0.5 * _preml_grad(
                rho, fit, slots=slots, Mp=Mp, n=n, y=y, wt=wt, family=family,
                reml_ind=reml_ind, X=X, p=p, gamma=gamma,
                pearson_scale_criterion=pearson_scale_criterion,
                use_ml_proj=use_ml_proj,
                family_mgcv_extended=family_mgcv_extended,
                penalty_rank=penalty_rank, UrS=UrS, reparam_cache=reparam_cache,
                dw_deta=dw_deta))
        if deriv >= 2:
            hess = _twh(_preml_hessian(
                rho, fit, slots=slots, Mp=Mp, n=n, family=family, wt=wt, y=y,
                X=X, p=p, gamma=gamma,
                pearson_scale_criterion=pearson_scale_criterion,
                use_ml_proj=use_ml_proj, penalty_rank=penalty_rank,
                family_mgcv_extended=family_mgcv_extended, UrS=UrS,
                reparam_cache=reparam_cache, reml_ind=reml_ind, dw_deta=dw_deta,
                d2w_deta2=d2w_deta2))
    else:
        raise ValueError(
            f"_gam_fit3_score: unknown scoreType {scoreType!r}")
    return {"score": float(score), "grad": grad, "hess": hess}


def _gam_fit4(x, y, rho, *, slots, UrS, reparam_Y, keep_cols,
              reparam_cache, weights, start, etastart, mustart, offset,
              family, control, null_coef, warm_eta, pls_lwork,
              efs_scale=None):
    """Penalized IRLS for mgcv-extended families — gam.fit4's inner
    loop (gam.fit4.r:340-548), line-by-line.

    Weights and pseudodata come from the family's deviance-derivative
    tables (``dDeta``):

        w  = ½·Deta2                    (signed Newton weights)
        z  = η_β − Deta/Deta2           (ratio form, can blow up at w≈0)
        wz = w·η_β − ½·Deta             (finite even at w = 0)

    Rows with non-finite z switch the solve to pls_fit1's ``use.wy``
    mode (rhs from X'wz). An indefinite penalized Hessian (the
    ``oo$n<0`` signal) retries the step with the negative-Deta2 rows
    zeroed — gam.fit4.r:392-416's "positive weights" retry, NOT
    gam.fit3's Fisher retry. Convergence is
    ``|Δpdev|/(0.1+|pdev|) < ε`` confirmed by the penalized-deviance
    gradient ``X'Deta + 2Sλβ`` (gam.fit4.r:523-537).

    ``efs_scale`` (non-None ⇔ mgcv ``scoreType=="EFS"``) is the trial
    scale efsudr threads through ``lsp``: each accepted iterate then
    re-estimates θ — and, for scale-unknown families (mgcv
    ``family$scale < 0``, i.e. free-θ tw), jointly log φ — by
    ``estimate.theta`` at the current μ (gam.fit4.r:507-515), recomputes
    pdev under the new θ so the next step control is consistent
    (gam.fit4.r:543-546), and enters the ε-gradient test as mgcv's
    ``scale`` (gam.fit4.r:527). The final local scale is returned as
    ``scale_est`` (mgcv ``scale.est=scale``, gam.fit4.r:807).
    """
    link = family.link
    X = x
    off = offset
    n, p = x.shape
    wt = weights
    theta = family.get_theta()
    # mgcv's local ``scale`` in EFS mode (updated by estimate.theta for
    # families with family$scale < 0); mgcv family$scale is -1 for tw,
    # NULL for nb/scat — hea: extended & not scale_known ⇔ tw.
    scale_cur = efs_scale
    efs_family_scale = (-1.0 if (efs_scale is not None
                                 and not family.scale_known) else None)
    Sλ = _s_lambda(slots, p, rho)
    Sλ = 0.5 * (Sλ + Sλ.T)
    E_aug = _penalty_root_of(slots, p, UrS, reparam_Y, keep_cols,
                             reparam_cache, rho)

    mu = family.gam_initialize(y, wt)
    # User starting values — same glm precedence as the standard
    # branch (gam.fit4 mirrors gam.fit3's block); the null baseline
    # below stays user-independent.
    if warm_eta is not None:
        # Warm start from the previous score-eval's converged predictor.
        # mgcv carries etastart<-b$linear.predictors across the outer
        # Newton (gam.fit3.r:1366-1367) and passes it straight into
        # gam.fit4 (gam.fit3.r:111-113), so each penalized IRLS starts near
        # its solution. Takes precedence over the user seed (which only
        # seeds the first fit); the null baseline below stays ρ-independent
        # (from null_coef). Mirrors _fit_given_rho's gam.fit3 warm start.
        eta_warm = warm_eta
        eta = eta_warm - off
        mu = link.linkinv(eta_warm)
    else:
        if mustart is not None:
            mu = np.asarray(mustart, dtype=float)
        if etastart is not None:
            eta_user = np.asarray(etastart, dtype=float)
            eta = eta_user - off
            mu = link.linkinv(eta_user)
        elif start is not None:
            eta = X @ start
            mu = link.linkinv(eta + off)
        else:
            eta = link.link(mu) - off       # β-only η

    # Null baseline coefficients (mgcv get.null.coef, passed in as
    # null_coef); η_null/μ_null follow (gam.fit4.r:283-285). Same
    # constant-η projection as gam.fit3 — a model is fit3 XOR fit4.
    eta_null = X @ null_coef
    mu_null = link.linkinv(eta_null + off)
    if (mustart is not None
            or etastart is not None
            or start is not None):
        ii = 0
        while not (family.validmu(mu)
                   and link.valideta(eta + off)
                   and bool(np.all(np.isfinite(eta)))):
            ii += 1
            if ii > 20:
                raise ValueError("Can't find valid starting values: "
                                 "please specify some")
            eta = 0.9 * eta + 0.1 * eta_null
            mu = link.linkinv(eta + off)
    beta = null_coef.copy()
    beta_old = null_coef.copy()
    eta_old = eta_null.copy()
    old_pdev = (float(np.sum(family.dev_resids(y, mu_null, wt)))
                + float(null_coef @ Sλ @ null_coef))

    ctrl = control or _GAM_CONTROL_DEFAULTS
    # control$epsilon under newton()'s conv.tol/100 cap (gam.fit3.r:1308)
    eps = min(ctrl["epsilon"], ctrl["newton"]["conv_tol"] / 100.0)
    max_it = ctrl["maxit"]
    # mgcv's grad test uses control$epsilon*(abs(pdev)+scale)
    # (gam.fit4.r:527) — EFS threads the live trial scale; the newton
    # path keeps hea's established 1.0 convention.
    scale_abs = 1.0 if scale_cur is None else float(scale_cur)

    def _work(mu_c, eta_c):
        # weights/pseudodata at the current iterate (gam.fit4.r:367-371).
        dd_c = family.dDeta(y, mu_c, wt, theta, level=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_c = 0.5 * np.asarray(dd_c["Deta2"], dtype=float)
            wz_c = w_c * eta_c - 0.5 * np.asarray(dd_c["Deta"],
                                                  dtype=float)
            z_c = eta_c - np.asarray(dd_c["Deta.Deta2"], dtype=float)
        good_c = np.isfinite(z_c) & np.isfinite(w_c)
        return dd_c, w_c, wz_c, z_c, good_c

    dd, w, wz, z, good = _work(mu, eta)
    conv = False
    boundary = False
    posdef = True
    warn_msgs: list[str] = []
    dev = float(np.sum(family.dev_resids(y, mu, wt)))

    def _solve(w_c, wz_c, z_c, good_c):
        # Masked pls_fit1 call: zero-weight rows drop out of X'WX and
        # X'Wz identically to mgcv's x[good,] subsetting. use.wy
        # whenever any z is non-finite (gam.fit4.r:377-381).
        w_m = np.where(good_c, w_c, 0.0)
        if np.all(good_c):
            return _pls_qr(X, pls_lwork, w_m, z_c, E_aug)
        good_w = np.isfinite(w_c) & np.isfinite(wz_c)
        w_m = np.where(good_w, w_c, 0.0)
        wz_m = np.where(good_w, wz_c, 0.0)
        return _pls_qr(X, pls_lwork, w_m, np.zeros(n), E_aug, Xtwz=X.T @ wz_m)

    for it in range(1, max_it + 1):
        if not np.any(good):
            warn_msgs.append(
                f"PIRLS(extended): no good data at iteration {it}"
            )
            break
        start, _R_it, _ld_it, ok = _solve(w, wz, z, good)
        posdef = ok
        if not ok:
            # Indefinite penalized Hessian → positive-weights retry
            # (gam.fit4.r:392-416).
            pos = np.isfinite(np.asarray(dd["Deta2"], dtype=float))
            pos &= np.where(pos, np.asarray(dd["Deta2"]) > 0.0, False)
            w = np.where(pos, w, 0.0)
            wz = w * eta - 0.5 * np.asarray(dd["Deta"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = eta - np.asarray(dd["Deta.Deta2"], dtype=float)
            good = np.isfinite(z) & np.isfinite(w)
            start, _R_it, _ld_it, ok = _solve(w, wz, z, good)
        if not ok:
            # Last resort: legacy ridge on the normal equations.
            w_m = np.where(good & (w > 0), w, 0.0)
            A = (X.T * w_m) @ X + Sλ
            A = 0.5 * (A + A.T)
            ridge = 1e-8 * np.trace(A) / p
            A_chol_r, lower_r = cho_factor(
                A + ridge * np.eye(p), lower=True, overwrite_a=False,
            )
            wz_m = np.where(good, wz, 0.0)
            start = cho_solve((A_chol_r, lower_r), X.T @ wz_m)
        if np.any(~np.isfinite(start)):
            warn_msgs.append(
                f"PIRLS(extended): non-finite coefficients at "
                f"iteration {it}"
            )
            break
        eta_new = X @ start
        mu_new = link.linkinv(eta_new + off)
        dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
        pen_new = float(start @ Sλ @ start)

        # inner loop 1 (gam.fit4.r:433-458): non-finite deviance.
        if not np.isfinite(dev_new):
            boundary = True
            ii = 0
            while not np.isfinite(dev_new):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError(
                        "inner loop 1; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
                dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
            pen_new = float(start @ Sλ @ start)

        # inner loop 2 (gam.fit4.r:461-477): invalid η/μ.
        if not (link.valideta(eta_new + off) and family.validmu(mu_new)):
            boundary = True
            ii = 0
            while not (link.valideta(eta_new + off)
                       and family.validmu(mu_new)):
                ii += 1
                if ii > max_it:
                    raise FloatingPointError(
                        "inner loop 2; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
            dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
            pen_new = float(start @ Sλ @ start)

        pdev_new = dev_new + pen_new

        # inner loop 3 (gam.fit4.r:486-505): pdev divergence.
        div_thresh = (10.0 * (0.1 + abs(old_pdev))
                      * (np.finfo(float).eps ** 0.5))
        if pdev_new - old_pdev > div_thresh:
            if it == 1:
                beta_old = null_coef.copy()
                eta_old = eta_null.copy()
            ii = 0
            while pdev_new - old_pdev > div_thresh:
                ii += 1
                if ii > 100:
                    raise FloatingPointError(
                        "inner loop 3; can't correct step size"
                    )
                start = 0.5 * (start + beta_old)
                eta_new = 0.5 * (eta_new + eta_old)
                mu_new = link.linkinv(eta_new + off)
                if not (link.valideta(eta_new + off)
                        and family.validmu(mu_new)):
                    continue
                dev_new = float(np.sum(family.dev_resids(y, mu_new, wt)))
                pen_new = float(start @ Sλ @ start)
                pdev_cand = dev_new + pen_new
                pdev_new = pdev_cand if np.isfinite(pdev_cand) else np.inf

        beta = start
        eta = eta_new
        mu = mu_new
        dev = dev_new

        if efs_scale is not None and family.n_theta > 0:
            # EFS θ-estimation at the accepted μ (gam.fit4.r:507-515):
            # estimate.theta at scale1 = family$scale (tw: -1 → joint
            # (θ, log φ) Newton) or the current trial scale; for
            # family$scale < 0 the trailing slot updates the local scale.
            from .bam import _estimate_theta
            scale1 = (efs_family_scale if efs_family_scale is not None
                      else float(scale_cur))
            theta = _estimate_theta(family, y, mu, scale=scale1, wt=wt,
                                    tol=1e-7)
            if efs_family_scale is not None and efs_family_scale < 0:
                scale_cur = float(np.exp(theta[family.n_theta]))
                scale_abs = scale_cur
                theta = theta[:family.n_theta]
            family.set_theta(theta)

        # Fresh weights/pseudodata at the accepted iterate — needed
        # both for the gradient confirmation and the next step
        # (gam.fit4.r:517-521).
        dd, w, wz, z, good = _work(mu, eta)

        # Convergence (gam.fit4.r:523-537): pdev change relative to
        # (0.1 + |pdev|), then gradient confirmation. Note w·η − wz
        # = ½·Deta on good rows, so the grad is X'Deta + 2Sλβ.
        if posdef and (abs(pdev_new - old_pdev) / (0.1 + abs(pdev_new))
                       < eps):
            w_m = np.where(good, w, 0.0)
            wz_m = np.where(good, wz, 0.0)
            grad = (2.0 * (X.T @ (w_m * eta - wz_m))
                    + 2.0 * (Sλ @ beta))
            if float(np.max(np.abs(grad))) > eps * (abs(pdev_new)
                                                    + scale_abs):
                old_pdev = pdev_new
                beta_old = beta.copy()
                eta_old = eta.copy()
            else:
                conv = True
                break
        else:
            old_pdev = pdev_new
            beta_old = beta.copy()
            eta_old = eta.copy()
        if efs_scale is not None and family.n_theta > 0:
            # gam.fit4.r:543-546: recompute pdev under the new θ so the
            # next iteration's step control has a consistent baseline.
            dev = float(np.sum(family.dev_resids(y, mu, wt)))
            old_pdev = dev + pen_new

    if not conv:
        warn_msgs.append("PIRLS algorithm did not converge")
    if boundary:
        warn_msgs.append("PIRLS algorithm stopped at boundary value")

    # Final consistent state at converged β̂ (gam.fit4.r:561-572):
    # signed w = ½Deta2, z with non-finite→0, wz finite everywhere.
    is_fisher_fallback = False
    good_f = np.isfinite(wz) & np.isfinite(w)
    w_f = np.where(good_f, w, 0.0)
    wz_f = np.where(good_f, wz, 0.0)
    z_f = np.where(np.isfinite(z), z, 0.0)
    use_wy = not np.all(np.isfinite(z) | ~good_f)
    if use_wy:
        _b_fin, R_fin, log_det_A, ok = _pls_qr(X, pls_lwork, 
            w_f, np.zeros(n), E_aug, Xtwz=X.T @ wz_f,
        )
    else:
        _b_fin, R_fin, log_det_A, ok = _pls_qr(X, pls_lwork, 
            w_f, np.where(good_f, z_f, 0.0), E_aug,
        )
    if not ok:
        # Indefinite at convergence — keep only the positive-weight
        # rows for the stored factor (same deliberate residual as the
        # standard path's Fisher fallback).
        pos = w_f > 0.0
        w_f = np.where(pos, w_f, 0.0)
        wz_f = np.where(pos, wz_f, 0.0)
        is_fisher_fallback = True
        _b_fin, R_fin, log_det_A, ok = _pls_qr(X, pls_lwork, 
            w_f, np.zeros(n), E_aug, Xtwz=X.T @ wz_f,
        )
    if ok:
        A_chol, lower = R_fin, False
    else:
        A = (X.T * w_f) @ X + Sλ
        A = 0.5 * (A + A.T)
        ridge = 1e-8 * np.trace(A) / p
        A_chol, lower = cho_factor(
            A + ridge * np.eye(p), lower=True, overwrite_a=False,
        )
        log_det_A = 2.0 * float(np.log(np.abs(np.diag(A_chol))).sum())

    # Full linear predictor for the return (caller sets warm start).
    eta_full = eta + off
    return _FitState(
        beta=beta, dev=dev, pen=float(beta @ Sλ @ beta),
        A_chol=A_chol, A_chol_lower=lower,
        S_full=Sλ, log_det_A=log_det_A,
        eta=eta_full, mu=mu, w=w_f, z=z_f, alpha=None,
        is_fisher_fallback=is_fisher_fallback,
        converged=conv, boundary=boundary, warn=warn_msgs,
        E_aug=E_aug,
        scale_est=(float(scale_cur) if scale_cur is not None else None),
    )


def newton(theta0, *, include_log_phi, criterion="REML",
           include_family_theta=False, max_iter=200, conv_tol=1e-6,
           max_step=5.0, max_sd_step=2.0, max_half=30, qerror_thresh=0.8,
           family, work_dim, X, y, wt, n, scale_fixed_value, control,
           g5, edge_correct, fit_fn, score_fn, fisher_view_fn,
           rho_full_fn, T_working_fn):
    """Unified analytical Newton on V_R(ρ, log φ[, θ_fam]) or V_g/V_u(ρ)
    — mgcv's gam.outer, extended with a family-θ slot for ``tw()``.

    Direct port of mgcv's ``newton`` (gam.fit3.r:1290-1719). Each
    outer iteration:

    1. Eigendecompose H, flag ``pdef`` (no negative or floor-clamped
       eigenvalues) and ``indef`` (any meaningfully negative one,
       threshold ``|λ_max|·√eps``). Set ``d ← |λ|`` then floor at
       ``max(d)·eps^0.7`` (Gill-Murray-Wright, gam.fit3.r:1447-1453).
    2. Newton direction ``Nstep = −V·diag(1/d)·V'·grad`` (using
       clamped d), capped to ``max_step``.
    3. Accept Nstep at α=1 only if ``score_change < 0`` AND ``pdef``
       AND quadratic-error gate ``qerror < qerror_thresh`` with
       ``qerror = |pred − actual| / (max(|pred|,|actual|) + score_scale·conv_tol)``.
       Otherwise step-halve: at the 4th halving (and ``it<10``)
       switch to the steepest-descent direction at the same length;
       after ``max_half/2`` halvings drop the qerror requirement
       (gam.fit3.r:1518-1572).
    4. If ``!pdef`` AND SD not yet tried, run a separate SD line
       search (start at ``2·max_sd_step``, halve up to 40 times,
       keep best descent that satisfies qerror) and replace the
       accepted step with SD-best if it scored lower
       (gam.fit3.r:1580-1641). This is what stops Newton from
       sliding into UBRE/GCV saturation tails on flat smooths when
       the seed Hessian is fully indefinite.

    Convergence (gam.fit3.r:1646-1658) requires ``!indef``,
    ``max(|grad|) ≤ score_scale·conv_tol·5``, AND
    ``|Δscore| ≤ score_scale·conv_tol``, with
    ``score_scale = |scale.est| + |score|`` (GCV/UBRE) or
    ``|log(scale.est)| + |score|`` (REML).

    ``theta`` layout: ρ first, then a single log φ column when
    ``include_log_phi`` is set (unknown-scale REML), then
    ``family.n_theta`` extra columns when ``include_family_theta`` is
    set. For known-scale REML (Poisson, Binomial) log φ is fixed at 0;
    for GCV.Cp log φ and family θ are always off the outer vector.

    Each ``_eval`` calls ``family.set_theta(t[base:])`` so the family's
    internal state (e.g. ``family.p`` for tw) tracks the current outer
    iterate before PIRLS is run.

    ``criterion`` selects the objective:
    - ``"REML"``: minimizes V_R via ``_reml`` (returns 2·V_R, hence
      the 0.5 scaling), ``_reml_grad``, ``_reml_hessian``.
    - ``"GCV"``: minimizes V_g (scale-unknown) or V_u (scale-known)
      via ``_gcv``, ``_gcv_grad``, ``_gcv_hessian``. ``include_log_phi``
      and ``include_family_theta`` must both be False (GCV does not
      put log φ or family θ in the outer vector — φ̂ is the Pearson
      estimate post-fit, not optimized; family θ has no GCV path).
    """
    edge_theta1 = None
    if criterion not in ("REML", "GCV", "REML5", "GACV", "PREML"):
        raise ValueError(
            "criterion must be 'REML', 'GCV', 'REML5', 'GACV' or "
            f"'PREML', got {criterion!r}")
    # GCV/GACV (performance-style) and PREML (Pearson-Laplace, φ profiled
    # analytically at each ρ) are all ρ-only outer problems — no log φ or
    # family θ in θ.
    if criterion in ("GCV", "GACV", "PREML") and include_log_phi:
        raise ValueError(
            f"{criterion} path does not include log φ in outer θ.")
    if criterion in ("GCV", "GACV", "PREML") and include_family_theta:
        raise ValueError(
            f"{criterion} path does not include family θ in outer θ.")
    if criterion == "REML5" and (include_log_phi or include_family_theta):
        # general families: scale.est ≡ 1, family θ never in outer θ
        raise ValueError("REML5 (gam.fit5) outer θ is ρ-only.")

    n_work = work_dim
    n_theta_fam = family.n_theta if include_family_theta else 0
    theta = np.asarray(theta0, dtype=float).copy()
    # θ's sp-part is the *working* log-sp; criterion derivatives come
    # back per-penalty (FULL space) and chain to working space through
    # T = blockdiag(L, I_extra): g_θ = T'g, H_θ = T'HT. T is None ⇔
    # identity (no id linkage) — zero-cost on the common path.
    T_work = T_working_fn((1 if include_log_phi else 0) + n_theta_fam)

    def _to_working(g):
        return g if T_work is None else T_work.T @ g

    def _to_working_hess(H):
        return H if T_work is None else T_work.T @ H @ T_work

    def _split(t):
        # → (FULL per-penalty ρ, log φ) from a working-layout θ.
        # ρ always occupies the first n_work slots; log φ (when
        # present) sits right after; family θ trails — so plain
        # ``t[:n_work]`` is correct for every layout, including the
        # scale-known (ρ, θ_fam) one.
        rho_t = rho_full_fn(t[:n_work])
        # Known-scale layouts: log φ is the FIXED log(scale) — 0 on
        # the poisson/binomial defaults, log(gam(scale=)) otherwise.
        lp_t = (float(t[n_work]) if include_log_phi
                else float(np.log(scale_fixed_value)))
        return rho_t, lp_t

    def _apply_family_theta(t):
        if n_theta_fam > 0:
            base = n_work + (1 if include_log_phi else 0)
            family.set_theta(t[base:base + n_theta_fam])

    # Non-REML5 criteria route score/grad/hess through the single free
    # `_gam_fit3_score` (mgcv's gam.fit3 score tail, via `_score_all`);
    # REML5 (gam.fit5) keeps its own monolithic REML/REML1/REML2 return.
    def _mk_score_closures(sType):
        def _eval(t, deriv=2):  # deriv: REML5-only hint, ignored here
            rho_t, lp_t = _split(t)
            _apply_family_theta(t)
            try:
                fit_t = fit_fn(rho_t)
            except Exception:
                return float("inf"), None
            return score_fn(
                rho_t, lp_t, fit_t, 0, sType,
                include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
                T_work=T_work)["score"], fit_t

        def _grad(rho, log_phi, fit):
            return score_fn(
                rho, log_phi, fit, 1, sType,
                include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
                T_work=T_work)["grad"]

        def _hess(rho, log_phi, fit):
            return score_fn(
                rho, log_phi, fit, 2, sType,
                include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
                T_work=T_work)["hess"]
        return _eval, _grad, _hess

    if criterion == "REML":
        _eval, _grad, _hess = _mk_score_closures("REML")
    elif criterion == "REML5":
        # general families: the criterion and its ρ-derivatives come
        # straight from gam.fit5 (REML/REML1/REML2). mgcv's newton
        # carries coefficient start values forward across EVERY
        # evaluation (gam.fit3.r newton, "carries start values
        # forward...", incl. rejected trials) — the carry lives in
        # the pre-rp basis, exactly what _gam_fit5 returns/expects.
        # mgcv evaluates the first Newton trial at deriv = pdef·2
        # (gam.fit3.r:1486): when the Hessian is +def it gets the score
        # AND grad/hess from ONE fit and reuses them on accept (no second
        # fit per iterate); only step-halving / SD trials run deriv=0,
        # with one deriv=2 refit on the accepted halving (gam.fit3.r:1531,
        # 1544). The newton below passes that `deriv` hint through `_eval`;
        # `_grad`/`_hess` reuse the fit's REML1/REML2 when present and only
        # refit when the accepted point came from a deriv=0 trial. Inner
        # epsilon: newton caps control$epsilon at conv.tol/100
        # (gam.fit3.r:1308) → 1e-8 at gam defaults.
        g5 = g5

        def _fit5_at(rho_t, d):
            fit_t = _gam_fit5(
                g5["X"], g5["y"], rho_t, g5["sl"],
                family=family, lpi=g5["lpi"],
                weights=g5["weights"], offset=g5["offsets"],
                Mp=g5["Mp"], deriv=d, start=g5["start"],
                gamma=g5["gamma"], epsilon=1e-8,
            )
            g5["start"] = fit_t["coefficients"]
            return fit_t

        def _eval(t, deriv=2):
            rho_t, _ = _split(t)
            try:
                fit_t = _fit5_at(rho_t, deriv)
            except Exception:
                return float("inf"), None
            return float(fit_t["REML"]), fit_t

        def _grad(rho, log_phi, fit):
            # Reuse the deriv=2 data if the accepted fit already carries it
            # (first-trial-pdef accept); else this point came from a deriv=0
            # trial (halving/SD) and needs the one deriv=2 refit. NB a
            # deriv=0 fit5 still has the "REML2" KEY (value None), so test
            # the value, not key presence.
            if fit.get("REML2") is None:
                fit.update(_fit5_at(rho, 2))
            # newton's b IS the last accepted deriv-2 fit — keep it
            # so the caller never refits at the optimum (a refit
            # re-enters Newton from the converged coefs and can
            # only exit through the step-failure paths).
            g5["fit"] = fit
            return _to_working(np.asarray(fit["REML1"], dtype=float))

        def _hess(rho, log_phi, fit):
            return _to_working_hess(np.asarray(fit["REML2"],
                                               dtype=float))
    elif criterion == "GACV":
        # GACV (gam.fit3.r:751): a GCV sibling for scale-unknown standard
        # families. ρ-only; analytic gradient + Hessian.
        _eval, _grad, _hess = _mk_score_closures("GACV")
    elif criterion == "PREML":
        # P-REML / P-ML (gam.fit3.r:640-665): the Laplace (RE)ML criterion
        # with the Pearson-Laplace scale φ = P/(n−Mp) plugged in (γ≡1),
        # ρ-only. `_gam_fit3_score` profiles φ_P internally for the value
        # (via `_phi_pearson`) and uses `_preml_grad`/`_preml_hessian`.
        _eval, _grad, _hess = _mk_score_closures("PREML")
    else:  # GCV
        _eval, _grad, _hess = _mk_score_closures("GCV")

    # P-REML/P-ML are (RE)ML-type for the convergence score scale
    # (|log scale.est| + |score|); GCV/GACV/UBRE use |scale.est| + |score|.
    is_reml = criterion in ("REML", "REML5", "PREML")

    def _score_scale(fit_, val):
        # mgcv's score.scale: |scale.est| + |score| (GCV/UBRE) or
        # |log(scale.est)| + |score| (REML). scale.est is gam.fit3's
        # estimator — Pearson with the Fletcher (2012) correction
        # (gam.fit3.r:596-603, mgcv's default scale.est); for
        # known-scale families it is 1.
        if family.scale_known:
            scale_est = 1.0
        else:
            scale_est = _fit3_scale_est(
                fit_, family=family, y=y, wt=wt, n=n, X=X,
                control=control, fisher_view_fn=fisher_view_fn)
        if is_reml:
            # log(scale.est); guard against scale_est ≤ 0
            scale_est_safe = max(scale_est, 1e-300)
            return abs(np.log(scale_est_safe)) + abs(val)
        return abs(scale_est) + abs(val)

    f_prev, fit = _eval(theta)
    if fit is None:
        outer_info = {
            "conv": "initial fit failed", "iter": 0,
            "grad": np.zeros_like(theta), "hess": np.zeros((theta.size, theta.size)),
            "score": float(f_prev), "score_scale": float("nan"),
        }
        outer_fit = None
        return {"theta": theta, "outer_info": outer_info, "outer_fit": outer_fit, "edge_theta1": edge_theta1}

    # Initial grad/hess at θ₀ and starting active set
    # (gam.fit3.r:1383-1385). Dimensions whose gradient is already
    # below ``score_scale·conv_tol`` start out inactive (excluded
    # from the Newton step). If everything is below threshold, mark
    # all active so the iter has something to move.
    rho0, log_phi0 = _split(theta)
    grad = _grad(rho0, log_phi0, fit)
    H = _hess(rho0, log_phi0, fit)
    H = 0.5 * (H + H.T)
    score_scale = _score_scale(fit, f_prev)
    uconv_ind = np.abs(grad) > score_scale * conv_tol
    if not np.any(uconv_ind):
        uconv_ind = np.ones_like(uconv_ind, dtype=bool)

    conv_text = "iteration limit reached"
    last_grad = grad
    last_hess = H
    it_done = 0
    for it in range(max_iter):
        score_scale = _score_scale(fit, f_prev)

        # Active-set masking (gam.fit3.r:1430-1436). Exclude
        # apparently-converged dims from the Newton step. mgcv also
        # computes a tighter ``uconv.ind1`` mask there but never
        # uses it; we follow suit. Safety net: if everything is
        # marked inactive, force the largest-|grad| dim active so
        # the iter still has something to move.
        if not np.any(uconv_ind):
            j = int(np.argmax(np.abs(grad))) if grad.size > 0 else 0
            uconv_ind = np.zeros_like(uconv_ind, dtype=bool)
            if grad.size > 0:
                uconv_ind[j] = True
        if H.size > 0:
            H1 = H[np.ix_(uconv_ind, uconv_ind)]
            grad1 = grad[uconv_ind]
        else:
            H1 = H
            grad1 = grad

        # Eigen analysis on the active subblock with mgcv's
        # pdef/indef flags (gam.fit3.r:1438-1455). ``indef``
        # triggers the SD-fallback; ``pdef`` False blocks
        # immediate-step acceptance.
        if H1.size > 0:
            w_eig, V_eig = np.linalg.eigh(H1)
            sqrt_eps = float(np.finfo(float).eps ** 0.5)
            # mgcv: sum(-d > abs(d[1])*sqrt(eps)) with R's eigen
            # sorting descending — d[1] is the largest *algebraic*
            # eigenvalue, not the largest magnitude.
            d_top = float(w_eig[-1])
            indef = bool(np.any(-w_eig > abs(d_top) * sqrt_eps))
            # 1-D special case: a tiny single eigenvalue can register
            # as indefinite at the |λ_max|·√eps threshold; require it
            # be meaningfully negative on the score-scale instead.
            if indef and w_eig.size == 1:
                indef = bool(w_eig[0] < -score_scale * sqrt_eps)
            d = np.abs(w_eig)
            pdef = bool(np.all(w_eig > 0))
            low_d = d.max() * (np.finfo(float).eps ** 0.7) if d.size else 0.0
            clamp_mask = d < low_d
            if np.any(clamp_mask):
                pdef = False
                d = np.where(clamp_mask, low_d, d)
            d_inv = np.where(d > 0, 1.0 / d, 0.0)
            Nstep_active = -V_eig @ (d_inv * (V_eig.T @ grad1))
            Nstep = np.zeros_like(grad)
            Nstep[uconv_ind] = Nstep_active
        else:
            Nstep = np.zeros_like(grad)
            pdef = True
            indef = False

        # Cap Newton step length
        ms = float(np.abs(Nstep).max()) if Nstep.size else 0.0
        if ms > max_step:
            Nstep = Nstep * (max_step / ms)

        # Steepest descent direction (length-1 in max-norm).
        gmax = float(np.abs(grad).max()) if grad.size else 0.0
        Sstep = (-grad / gmax) if gmax > 0 else np.zeros_like(grad)

        def _qerror(step, score_change):
            if step.size == 0:
                return 0.0
            pred = float(grad @ step + 0.5 * step @ (H @ step))
            denom = max(abs(pred), abs(score_change)) + score_scale * conv_tol
            return abs(pred - score_change) / denom if denom > 0 else 0.0

        # ----- step acceptance (gam.fit3.r:1492-1573) -----
        accepted_step = None
        accepted_f = float("inf")
        accepted_fit = None
        sd_unused = True

        # mgcv evaluates the first trial at deriv = pdef·2 (gam.fit3.r:1486):
        # +def ⇒ get grad/hess in this fit so an immediate accept needs no
        # second fit (REML5); indefinite ⇒ deriv=0, defer to the SD trial.
        f_try, fit_try = _eval(theta + Nstep, deriv=2 if pdef else 0)
        score_change = f_try - f_prev
        qerror = _qerror(Nstep, score_change)
        if (
            np.isfinite(f_try) and score_change < 0
            and pdef and qerror < qerror_thresh
        ):
            accepted_step, accepted_f, accepted_fit = Nstep.copy(), f_try, fit_try
        else:
            step = Nstep.copy()
            for ii in range(max_half):
                if ii == 3 and it < 9:
                    # Newton failing — switch to SD direction at the
                    # current step length (gam.fit3.r:1521; mgcv's
                    # i<10 with 1-based iterations = it<9 here).
                    s_length = min(float(np.linalg.norm(step)), max_sd_step)
                    sd_norm = float(np.linalg.norm(Sstep))
                    if sd_norm > 0:
                        step = Sstep * (s_length / sd_norm)
                        sd_unused = False
                else:
                    step = step / 2
                f_try, fit_try = _eval(theta + step, deriv=0)
                score_change = f_try - f_prev
                if ii > min(4, max_half // 2):
                    qerror = qerror_thresh / 2  # drop qerror requirement
                else:
                    qerror = _qerror(step, score_change)
                if (
                    np.isfinite(f_try)
                    and score_change < 0
                    and qerror < qerror_thresh
                ):
                    accepted_step = step.copy()
                    accepted_f, accepted_fit = f_try, fit_try
                    break

        # ----- indefinite SD fallback (gam.fit3.r:1580-1641) -----
        # If the Hessian wasn't PD and we haven't already used the SD
        # direction in step-halving, run an independent SD line
        # search and pick whichever direction scored lower. This is
        # what keeps Newton out of UBRE/GCV saturation tails when
        # the seed lies near a local maximum (all-negative eig).
        if (not pdef) and sd_unused and Sstep.size > 0:
            sd_best_step = None
            sd_best_f = float("inf")
            sd_best_fit = None
            # mgcv starts at 2·Sstep so the first halving gives
            # Sstep itself (max-norm 1) — gam.fit3.r:1581.
            sd_step = Sstep * 2
            for kk in range(40):
                sd_step = sd_step / 2
                f_sd, fit_sd = _eval(theta + sd_step, deriv=0)
                score_change_sd = f_sd - f_prev
                qerror_sd = _qerror(sd_step, score_change_sd)
                accept_sd = (
                    np.isfinite(f_sd)
                    and (
                        sd_best_step is None
                        or (f_sd <= sd_best_f and qerror_sd < qerror_thresh)
                    )
                )
                if accept_sd:
                    sd_best_step = sd_step.copy()
                    sd_best_f, sd_best_fit = f_sd, fit_sd
                # Stop once we've found descent and a shorter step
                # makes things worse.
                if (
                    sd_best_step is not None and sd_best_f < f_prev
                    and np.isfinite(f_sd) and f_sd > sd_best_f
                ):
                    break
            if sd_best_step is not None and sd_best_f < accepted_f:
                accepted_step = sd_best_step
                accepted_f = sd_best_f
                accepted_fit = sd_best_fit

        if accepted_step is None:
            # No improving step. mgcv checks the gradient convergence test
            # BEFORE declaring step failure (gam.fit3.r:1646-1657): if the
            # current point already satisfies max|grad| ≤ score_scale·conv_tol·5
            # this is convergence AT the optimum (the Newton step just can't
            # improve on it), reported "full convergence" — NOT a step failure.
            # Only a still-large residual gradient is a genuine step failure
            # (the ill-posed corner where mgcv warns "check results carefully").
            _ss = _score_scale(fit, f_prev)
            gmax = float(np.abs(grad).max()) if grad.size > 0 else 0.0
            conv_text = ("full convergence"
                         if gmax <= _ss * conv_tol * 5.0 else "step failed")
            it_done = it + 1
            break
        theta = theta + accepted_step
        df = abs(accepted_f - f_prev)
        f_prev = accepted_f
        fit = accepted_fit
        it_done = it + 1

        # Recompute grad/hess at the new θ (gam.fit3.r:1505-1508).
        # The convergence test and active-set update use these
        # post-step values, mirroring mgcv's gam.fit3 deriv=2 refit.
        # The family θ state must be re-pinned to the ACCEPTED point:
        # _eval mutates it at every trial candidate, and the accepted
        # candidate is not necessarily the last one evaluated
        # (step-halving / SD line searches probe past it).
        _apply_family_theta(theta)
        rho_n, log_phi_n = _split(theta)
        grad = _grad(rho_n, log_phi_n, fit)
        H = _hess(rho_n, log_phi_n, fit)
        H = 0.5 * (H + H.T)
        last_grad, last_hess = grad, H

        # mgcv's outer convergence test (gam.fit3.r:1646-1658):
        # require non-indefinite Hessian, max(|grad|) ≤ 5·score_scale·conv_tol,
        # AND |Δscore| ≤ score_scale·conv_tol.
        score_scale = _score_scale(fit, f_prev)
        converged = not indef
        # Refresh active set from new grad/hess (gam.fit3.r:1650-1651).
        diag_H = np.diag(H) if H.size > 0 else np.array([])
        uconv_ind = (
            (np.abs(grad) > score_scale * conv_tol * 0.1)
            | (np.abs(diag_H) > score_scale * conv_tol * 0.1)
        )
        if grad.size > 0 and float(np.abs(grad).max()) > score_scale * conv_tol * 5.0:
            converged = False
        if df > score_scale * conv_tol:
            if converged:
                # Otherwise can't progress (gam.fit3.r:1654).
                uconv_ind = np.ones_like(uconv_ind, dtype=bool)
            converged = False
        if converged:
            conv_text = "full convergence"
            break

    # mgcv's newton warns the user when the outer optimizer terminates on a
    # step failure or the iteration cap rather than full convergence
    # (gam.fit3.r:1660-1666) — surface the same "check results carefully" so an
    # unreliable fit is never silent (the indefinite-Hessian / Fisher-fallback
    # corner reaches this exactly when mgcv itself does).
    if conv_text == "step failed":
        import warnings as _warnings
        _warnings.warn(
            "Fitting terminated with step failure - check results carefully",
            stacklevel=2)
    elif conv_text == "iteration limit reached":
        import warnings as _warnings
        _warnings.warn(
            "Iteration limit reached without full convergence - check "
            "carefully", stacklevel=2)

    outer_info = {
        "conv": conv_text,
        "iter": it_done,
        "grad": last_grad,
        "hess": last_hess,
        "score": float(f_prev),
        "score_scale": float(_score_scale(fit, f_prev)),
    }

    # gam.control(edge.correct=) — mgcv newton(), gam.fit3.r:1670-1700:
    # smoothing parameters at "working infinity" (Hessian-flat
    # directions) get walked back toward the seed in unit log steps
    # until the RE/ML criterion has risen by α (default 0.02) per
    # parameter; derivative quantities recomputed there feed a better
    # Vc (post.proc's k=2 pass). Only the corrected θ is stored here;
    # the constructor recomputes the Vc pieces from it.
    edge_theta1 = None
    if edge_correct and is_reml and theta.size > 0:
        # The k=2 Vc pass runs whether or not anything is flat (mgcv
        # refits at lsp1 = lsp when `flat` is empty — the corrected
        # Vc still differs through the weaker 1e-7 Vr prior).
        grad2 = np.diag(last_hess) if last_hess.size else np.zeros(0)
        flat = np.where(np.abs(grad2) < np.abs(last_grad) * 100.0)[0]
        alpha_ec = (0.02 if edge_correct is True
                    else float(abs(edge_correct)))
        theta1 = theta.copy()
        if flat.size:
            step_dir = (np.asarray(theta0, dtype=float) > theta) * 2.0 - 1.0
            f1 = f_prev
            for i in flat:
                target = f1 + alpha_ec
                # mgcv walks unbounded; cap defensively (a unit log-sp
                # step per iteration covers any practical surface
                # within a few dozen steps).
                for _ in range(100):
                    if f1 >= target:
                        break
                    theta1[i] = theta1[i] + step_dir[i]
                    f1_new, _fit1 = _eval(theta1, deriv=0)
                    if not np.isfinite(f1_new):
                        theta1[i] = theta1[i] - step_dir[i]
                        break
                    f1 = f1_new
            # Restore family θ to the converged values (the walk's
            # _eval calls mutate family state for tw).
            _apply_family_theta(theta)
        edge_theta1 = theta1
    # Cache the accepted-step fit at the converged θ for the caller to
    # reuse (mgcv's `object <- b$object`, no refit). `fit` is the last
    # accepted deriv-0 `_fit_given_rho` at `_rho_full(theta[:n_work])` ==
    # the ρ̂ the caller will build from; the edge-correct walk above leaves
    # it untouched. REML5 caches its own deriv-2 fit on `_g5["fit"]`.
    outer_fit = fit
    return {"theta": theta, "outer_info": outer_info, "outer_fit": outer_fit, "edge_theta1": edge_theta1}


def bfgs(theta0, *, conv_tol=1e-6, max_Nstep=3.0, max_step=200,
         g5, family, L, rho_full_fn):
    """mgcv ``bfgs`` (gam.fit3.r:1722-2141) for general families.

    BFGS over the REML5 score using only ``gam.fit5`` at deriv ≤ 1
    (score ``REML`` + gradient ``REML1`` + the ``dVkk`` curvature
    matrix). This is the outer optimizer mgcv coerces to for
    ``available.derivs == 1`` families (mgcv.r:1907) — ``mvn`` and any
    custom family that supplies ll only to the dH/trace order
    (gamlss_gH deriv ≤ 2), where Newton's REML2 (needing the ll's
    ``trHid2H``, gamlss_gH deriv 3) is unavailable.

    Step lengths meet the Wolfe conditions via Nocedal & Wright
    (2006) Algorithms 3.5 (main loop) and 3.6 (``zoom`` bisection).
    The initial inverse Hessian is seeded by finite-differencing the
    gradient, then adjusted on the first accepted step (mgcv's p143
    variant). ``score.scale = 1 + |score|`` (REML path; general
    families have ``scale.est ≡ 1``). The working-infinite-sp
    roll-back (gam.fit3.r:2065-2100) is ported for parity.

    ``theta0`` is the *working* log-sp seed; the returned vector is
    the converged working log-sp. ``outer_info`` is populated
    with ``conv``/``iter``/``grad``/``hess``/``score.hist`` like
    mgcv's ``object$outer.info``; the accepted deriv-1 fit is cached
    on ``g5["fit"]`` so the caller never refits.
    """

    fam = family
    L = L                      # working→full sp (None ⇔ I)
    n_sp = int(np.asarray(theta0).size)
    eps_mach = float(np.finfo(float).eps)

    # mgcv caps the inner PIRLS tol at conv.tol/100 (gam.fit3.r:1797).
    inner_eps = min(1e-7, conv_tol / 100.0)

    def _to_working(g):
        return g if L is None else L.T @ g

    def _fit5(lsp_work, d):
        rho = rho_full_fn(np.asarray(lsp_work, dtype=float))
        fit = _gam_fit5(
            g5["X"], g5["y"], rho, g5["sl"], family=fam,
            lpi=g5["lpi"], weights=g5["weights"],
            offset=g5["offsets"], Mp=g5["Mp"], deriv=d,
            start=g5["start"], gamma=g5["gamma"], epsilon=inner_eps,
        )
        g5["start"] = fit["coefficients"]
        return fit

    def _dvkk_diag(fit):
        dV = np.asarray(fit["dVkk"], dtype=float)
        if L is None:
            return np.diag(dV).copy()
        return np.diag(L.T @ dV @ L)

    def _grad(fit):
        return _to_working(np.asarray(fit["REML1"], dtype=float))

    # all sp slots carry curvature info for general families (mvn has
    # no family$n.theta and gam.fit5's dVkk is n_sp×n_sp) — spind ≡
    # every working sp (mgcv.r-bfgs: nind = ncol(L), spind = 1:nind).
    spind = np.ones(n_sp, dtype=bool)

    ilsp = np.asarray(theta0, dtype=float).copy()
    initial_lsp = ilsp.copy()

    # ---- initial fit + gradient -----------------------------------
    b = _fit5(ilsp, 1)
    score = float(b["REML"])
    grad = _grad(b)
    i_dvkk = _dvkk_diag(b)
    start0 = g5["start"].copy()
    i_score = score
    i_grad = grad.copy()
    score_scale = 1.0 + abs(i_score)

    # ---- FD inverse-Hessian seed (gam.fit3.r:1852-1873) -----------
    Bmat = np.eye(n_sp)
    feps = 1e-4
    for k in range(n_sp):
        jlsp = ilsp.copy()
        jlsp[k] += feps
        g5["start"] = start0.copy()
        bk = _fit5(jlsp, 1)
        grad1 = _grad(bk)
        Bmat[k, :] = (grad1 - grad) / feps
    Bmat = (Bmat + Bmat.T) / 2.0
    evals, evecs = np.linalg.eigh(Bmat)
    evals = np.abs(evals)
    thresh = float(np.max(evals)) * 1e-4 if evals.size else 0.0
    evals[evals < thresh] = thresh
    # B ← V diag(1/λ) V'  (the approximate INVERSE Hessian)
    Bmat = evecs @ ((evecs / evals).T)
    g5["start"] = start0.copy()

    c1, c2 = 1e-4, 0.9          # Wolfe constants
    score_hist = [i_score]
    uconv = np.ones(n_sp, dtype=bool)
    rolled_back = False
    # the "initial" record for the current line search
    cur = {"alpha": 0.0, "score": i_score, "grad": i_grad,
           "dVkk": i_dvkk, "start": start0.copy()}
    step = np.zeros(n_sp)
    trial = None
    ct = "iteration limit reached"
    iters = 0

    def zoom(lo, hi):
        # N&W Alg 3.6: bisection for a Wolfe-satisfying step.
        for _ in range(40):
            al = (lo["alpha"] + hi["alpha"]) / 2.0
            tr = {"alpha": al}
            lspz = ilsp + step * al
            g5["start"] = cur["start"].copy()
            bz = _fit5(lspz, 0)
            tr["score"] = float(bz["REML"])
            tr["start"] = g5["start"].copy()
            if (tr["score"] > cur["score"] + al * c1 * cur["dscore"]
                    or tr["score"] >= lo["score"]):
                hi = tr
            else:
                g5["start"] = cur["start"].copy()
                bz = _fit5(lspz, 1)
                tr["grad"] = _grad(bz)
                tr["dVkk"] = _dvkk_diag(bz)
                tr["start"] = g5["start"].copy()
                tr["dscore"] = float(np.sum(step * tr["grad"]))
                if abs(tr["dscore"]) <= -c2 * cur["dscore"]:
                    return tr
                if tr["dscore"] * (hi["alpha"] - lo["alpha"]) >= 0:
                    hi = lo
                lo = tr
        return None

    for it in range(1, max_step + 1):
        iters = it
        # trial step from the approximate inverse Hessian
        step = np.zeros(n_sp)
        step[uconv] = -(Bmat[np.ix_(uconv, uconv)] @ i_grad[uconv])
        if float(np.sum(step * i_grad)) >= 0:    # not descending
            step = -np.diag(Bmat) * i_grad
            step[~uconv] = 0.0
        ms = float(np.max(np.abs(step))) if step.size else 0.0
        if ms > max_Nstep:
            alpha = max_Nstep / ms
            alpha_max = alpha * 1.05
        else:
            alpha = 1.0
            alpha_max = min(2.0, max_Nstep / ms) if ms > 0 else 2.0
        cur["dscore"] = float(np.sum(step * i_grad))
        prev = dict(cur)
        trial = {"alpha": alpha}
        deriv = 1
        while True:                       # N&W Alg 3.5
            lsp = ilsp + trial["alpha"] * step
            g5["start"] = prev["start"].copy()
            b = _fit5(lsp, deriv)
            trial["score"] = float(b["REML"])
            if deriv > 0:
                trial["grad"] = _grad(b)
                trial["dVkk"] = _dvkk_diag(b)
                trial["dscore"] = float(np.sum(trial["grad"] * step))
                deriv = 0
            else:
                trial["grad"] = None
                trial["dscore"] = None
            trial["start"] = g5["start"].copy()
            # Wolfe 1: sufficient decrease
            if (trial["score"] > cur["score"]
                    + c1 * trial["alpha"] * cur["dscore"]
                    or (deriv == 0 and trial["score"] >= prev["score"])):
                trial = zoom(prev, trial)
                break
            if trial["dscore"] is None:   # need gradient at trial
                g5["start"] = trial["start"].copy()
                b = _fit5(lsp, 1)
                trial["grad"] = _grad(b)
                trial["dscore"] = float(np.sum(trial["grad"] * step))
                trial["dVkk"] = _dvkk_diag(b)
                trial["start"] = g5["start"].copy()
            if abs(trial["dscore"]) <= -c2 * cur["dscore"]:
                break                     # Wolfe 2 met
            if trial["dscore"] >= 0:      # increase at trial end
                trial = zoom(trial, prev)
                break
            prev = dict(trial)
            if trial["alpha"] == alpha_max:
                break
            trial["alpha"] = min(prev["alpha"] * 1.3, alpha_max)

        if trial is None:                 # step failed
            lsp = ilsp.copy()
            if rolled_back:
                break
            uconv = np.abs(i_grad) > score_scale * conv_tol * 0.1
            uconv[spind] = uconv[spind] | (
                np.abs(i_dvkk)[spind] > score_scale * conv_tol * 0.1)
            if np.sum(~uconv) == 0:
                break
            trial = dict(cur)
            converged = True
        else:                             # BFGS inverse-Hessian update
            yg = trial["grad"] - i_grad
            step_full = step * trial["alpha"]
            rho_bfgs = float(np.sum(yg * step_full))
            if rho_bfgs > 0:
                if it == 1:
                    Bmat = Bmat * trial["alpha"]
                rinv = 1.0 / rho_bfgs
                Bmat = Bmat - rinv * np.outer(step_full, yg @ Bmat)
                Bmat = (Bmat - rinv * np.outer(Bmat @ yg, step_full)
                        + rinv * np.outer(step_full, step_full))
            score_hist.append(trial["score"])
            ilsp = ilsp + step_full
            lsp = ilsp.copy()
            converged = True
            score_scale = 1.0 + abs(trial["score"])
            uconv = np.abs(trial["grad"]) > score_scale * conv_tol
            if np.sum(uconv):
                converged = False
            uconv = np.abs(trial["grad"]) > score_scale * conv_tol * 0.1
            uconv[spind] = uconv[spind] | (
                np.abs(trial["dVkk"])[spind]
                > score_scale * conv_tol * 0.1)
            if abs(i_score - trial["score"]) > score_scale * conv_tol:
                if not np.sum(uconv):
                    uconv = np.ones(n_sp, dtype=bool)
                converged = False

        # roll back any "working infinite" sps (gam.fit3.r:2065-2100)
        if converged:
            if np.sum(~uconv) == 0 or rolled_back:
                break
            rolled_back = True
            counter = 0
            uconv0 = uconv.copy()
            while np.sum(~uconv0) > 0 and counter < 5:
                lsp[~uconv0] = (lsp[~uconv0] * 0.8
                                + initial_lsp[~uconv0] * 0.2)
                g5["start"] = trial["start"].copy()
                b = _fit5(lsp, 1)
                trial["score"] = float(b["REML"])
                trial["grad"] = _grad(b)
                trial["dscore"] = float(np.sum(trial["grad"] * step))
                trial["dVkk"] = _dvkk_diag(b)
                trial["start"] = g5["start"].copy()
                counter += 1
                uconv0 = np.abs(trial["grad"]) > score_scale * conv_tol * 20
                uconv0[spind] = uconv0[spind] | (
                    np.abs(trial["dVkk"])[spind]
                    > score_scale * conv_tol * 20)
                uconv0 = uconv0 | uconv
            uconv = np.ones(n_sp, dtype=bool)
            ilsp = lsp.copy()

        cur = dict(trial)
        cur["alpha"] = 0.0
        i_score = trial["score"]
        i_grad = np.asarray(trial["grad"], dtype=float).copy()
        i_dvkk = np.asarray(trial["dVkk"], dtype=float).copy()

    if trial is None:
        ct = "step failed"
        lsp = ilsp.copy()
    elif iters == max_step:
        ct = "iteration limit reached"
    else:
        ct = "full convergence"

    # ---- final fit (gam.fit3.r:2116) ------------------------------
    g5["start"] = (cur.get("start", start0)).copy()
    bfin = _fit5(lsp, 1)
    g5["fit"] = bfin
    gfin = _grad(bfin)
    # approximate Hessian (invert the inverse-Hessian B)
    evals, evecs = np.linalg.eigh((Bmat + Bmat.T) / 2.0)
    keep = evals > float(np.max(evals)) * eps_mach ** 0.9
    inv = np.where(keep, 1.0 / np.where(keep, evals, 1.0), 0.0)
    hess = evecs @ (inv[:, None] * evecs.T)
    outer_info = {
        "conv": ct, "iter": iters, "grad": gfin, "hess": hess,
        "score.hist": np.asarray(score_hist, dtype=float),
    }
    return {"theta": ilsp, "outer_info": outer_info}


def gam_outer(theta0, *, optimizer, criterion, control,
              include_log_phi=False, include_family_theta=False,
              newton_fn=None, bfgs_fn=None, magic_fn=None):
    """mgcv ``gam.outer`` (mgcv.r:1634): dispatch the smoothing-parameter
    optimization to the chosen outer optimizer and return the converged
    working log-sp θ̂.

    The seed θ₀ and the post-optimization fit unpacking are estimate.gam's
    job (hea's constructor); the extended-Fellner-Schall (efs) path is
    handled there too, as in mgcv. The optimizer machinery is threaded as
    callables — `newton`/`bfgs` are the free module fns (via the class's
    thin `_outer_*` shims, which source their many data/poly args from
    ``self``); ``magic`` is the Gaussian-additive fast path — so the
    standard, extended (bam) and general-family constructors all reuse this
    one dispatch. Control knobs come from ``control["newton"]`` (mgcv's
    ``G$control``): conv.tol, maxNstep, maxSstep, maxHalf."""
    _nt = control["newton"]
    if optimizer == "magic":
        return magic_fn(theta0)
    if optimizer == "bfgs":
        return bfgs_fn(theta0, conv_tol=_nt["conv_tol"],
                       max_Nstep=_nt["maxNstep"])
    return newton_fn(
        theta0, criterion=criterion, include_log_phi=include_log_phi,
        include_family_theta=include_family_theta,
        conv_tol=_nt["conv_tol"], max_step=_nt["maxNstep"],
        max_sd_step=_nt["maxSstep"], max_half=_nt["maxHalf"],
    )


def gam_outer_nlm_optim(theta0, *, optimizer2, criterion, fscale, control,
                        include_log_phi, include_family_theta, n_work,
                        family, scale_fixed_value, fit_fn, score_fn,
                        rho_full_fn, T_working_fn):
    """mgcv ``gam.outer``'s nlm/optim branch (mgcv.r:1692-1717) with the
    three gam.fit3 objective wrappers it drives — ``gam2objective``
    (deriv-0 score), ``gam2derivative`` (deriv-1 gradient) and
    ``gam4objective`` (score + gradient attribute for nlm), all from
    gam.fit3.r:2145-2211.

    The optimizer sees mgcv's lsp layout ``[θ_fam | ρ_work | log φ?]``
    (estimate.gam prepends family θ, mgcv.r:2040-2057, and appends the
    log-scale slot) so that every dot product and line-search decision
    inside :func:`hea.R.optimize.nlm` / :func:`hea.R.optimize.optim`
    accumulates in exactly R's coordinate order; hea's working layout
    ``[ρ_work | log φ? | θ_fam]`` is recovered by an exact permutation
    at the score boundary. ``L``/``lsp0`` (id-linked / partially fixed
    sp) are supported exactly as in mgcv: the objectives map working →
    full lsp via ``rho_full_fn`` and chain gradients back through
    ``T_work`` (mgcv's ``t(L)%*%ret``).

    nlm is called as mgcv does (mgcv.r:1697-1703): ``typsize = lsp``
    (the *initial* iterate — negative entries are made positive by
    UNCMIN's optchk), ``fscale = null.scale``, and the gam.control nlm
    knobs; optim as mgcv.r:1706-1708: L-BFGS-B with ``fnscale =
    null.scale``, ``factr`` from control and ``lmm = min(5,
    length(lsp))``. When ``lsp`` is empty mgcv coerces the method to
    ``"no.sps"`` (mgcv.r:1646-1648): no optimizer runs and the single
    final ``gam2objective`` fit is returned (``outer_info`` None).

    Like mgcv, the returned object is the FINAL ``gam2objective`` call's
    deriv-0 fit (mgcv.r:1711-1716) — so no ``db.drho``-derived pieces
    (Vc/edf2/sp-uncertainty CIs) exist on this path, exactly as for a
    deriv-0 mgcv fit. Returns ``{"theta", "fit", "outer_info",
    "score"}`` with ``theta`` in hea's working layout.
    """
    from ..R.optimize import nlm as _r_nlm
    from ..R.optimize import optim as _r_optim

    theta0 = np.asarray(theta0, dtype=float)
    n_theta_fam = family.n_theta if include_family_theta else 0
    n_extra = (1 if include_log_phi else 0) + n_theta_fam
    T_work = T_working_fn(n_extra)

    def _to_lsp(t):
        # hea working [ρ | log φ? | θ_fam] → mgcv lsp [θ_fam | ρ | log φ?]
        base = n_work + (1 if include_log_phi else 0)
        return np.concatenate([t[base:base + n_theta_fam], t[:base]])

    def _to_working(lsp):
        return np.concatenate([lsp[n_theta_fam:],
                               lsp[:n_theta_fam]])

    def _eval(lsp, deriv):
        t = _to_working(np.asarray(lsp, dtype=float))
        rho_t = rho_full_fn(t[:n_work])
        lp_t = (float(t[n_work]) if include_log_phi
                else float(np.log(scale_fixed_value)))
        if n_theta_fam > 0:
            base = n_work + (1 if include_log_phi else 0)
            family.set_theta(t[base:base + n_theta_fam])
        fit_t = fit_fn(rho_t)
        out = score_fn(rho_t, lp_t, fit_t, deriv, criterion,
                       include_log_phi=include_log_phi,
                       include_family_theta=include_family_theta,
                       T_work=T_work)
        return fit_t, out

    fit_cell = [None]

    def gam2objective(lsp):
        fit_t, out = _eval(lsp, 0)
        fit_cell[0] = fit_t
        return float(out["score"])

    def gam2derivative(lsp):
        _, out = _eval(lsp, 1)
        return _to_lsp(np.asarray(out["grad"], dtype=float))

    def gam4objective(lsp):
        fit_t, out = _eval(lsp, 1)
        fit_cell[0] = fit_t
        return (float(out["score"]),
                _to_lsp(np.asarray(out["grad"], dtype=float)))

    ctrl = control or _GAM_CONTROL_DEFAULTS
    lsp = _to_lsp(theta0)
    if lsp.size == 0:
        b = None                       # mgcv's "no.sps"
    elif optimizer2 == "nlm":
        cn = ctrl["nlm"]
        b = _r_nlm(gam4objective, lsp, typsize=lsp.copy(),
                   fscale=fscale, stepmax=cn["stepmax"],
                   ndigit=cn["ndigit"], gradtol=cn["gradtol"],
                   steptol=cn["steptol"], iterlim=cn["iterlim"],
                   check_analyticals=cn["check_analyticals"])
        lsp = np.asarray(b["estimate"], dtype=float)
    else:
        b = _r_optim(lsp, gam2objective, gam2derivative,
                     method="L-BFGS-B",
                     control={"fnscale": fscale,
                              "factr": ctrl["optim"]["factr"],
                              "lmm": min(5, lsp.size)})
        lsp = np.asarray(b["par"], dtype=float)
    # final model fit, with warnings (mgcv.r:1711)
    score = gam2objective(lsp)
    return {"theta": _to_working(lsp), "fit": fit_cell[0],
            "outer_info": b, "score": score}


def efsudr(rho0, *, log_phi0, family, family_mgcv_extended, fit_fn,
           reml_fn, scale_est_fn, fisher_view_fn, UrS, reparam_Y,
           reparam_cache, p, n, control, scale_fixed_value):
    """mgcv ``efsudr`` (gam.fit4.r:822-938): the extended Fellner-Schall
    outer loop for regular AND extended families, PIRLS by
    gam.fit3/gam.fit4 at ``scoreType="EFS"`` (→ REML value at deriv 0,
    plus ``ldetS1 = ∂log|Sλ|₊/∂ρ`` from gam.reparam).

    mgcv threads one ``lsp = [θ_fam | ρ_sp | log φ?]`` vector through the
    fits; hea's fitters read θ from the family state and φ never enters
    PIRLS (regular) / enters as ``efs_scale`` (extended EFS), so this
    port carries the (θ, log φ) slots as explicit state snapshots and
    re-seeds ``family.set_theta`` before every fit — reproducing lsp
    threading exactly, including the re-seed from the pre-trial state on
    step contraction and on a rejected step extension, and mgcv's
    scale-slot quirk on extension (``lsp2[len] <- log(fit$scale)`` reads
    the lsp1-fit, not fit2, gam.fit4.r:900).

    Per-fit REML (mgcv units = hea ``_reml``/2, evaluated by
    ``reml_fn``): regular families at the lsp trial scale
    (gam.fit3.r:121-124); extended families at the fit's own final scale
    (``fit.scale_est`` — gam.fit4's local ``scale``, estimate.theta-
    updated for scale-unknown tw, gam.fit4.r:735). ``gamma`` is 1
    throughout — gam.outer's efs call drops it (mgcv.r:1665) and efsudr
    hard-codes ``gamma=1`` (so a user gamma≠1 is ignored on this path,
    exactly as in mgcv; hea's post-fit ``REML_criterion`` re-evaluates
    with the user gamma and diverges from mgcv's ``gcv.ubre`` only in
    that corner).

    The EFS update per penalty (gam.fit4.r:870-878):

        trVS_j = tr(UrSⱼ'Y'(V/φ)Y·UrSⱼ),  bSbⱼ = ‖β'Y·UrSⱼ‖²,
        a = max(0, ldetS1·e^{−ρ} − trVS),  r = a/max(0,bSb)·φ,
        ρ' = min(ρ + log(r)·mult, efs.lspmax)

    with Y the total-penalty range basis (mgcv ``U1[,1:(p−Mp)]`` =
    ``reparam_Y``), V/φ = (X'W_F·X+Sλ)⁻¹ — mgcv's ``rV`` is built from
    the EXPECTED (Fisher) weights ``wf``, re-factoring when PIRLS ran
    full Newton (gdi.c:2262 "get rV and K using E(W)"; hea's
    ``fisher_view_fn``) — and φ the edf-corrected ``fit$scale·n/(n−edf)``
    for scale-unknown extended families. Step control: ×2 extension when max|Δρ|<.05 improved,
    halving while worse and mult>1; stop when the EFS step is small and
    REML flat over 3 steps (``efs.tol``), or the deviance stalls
    (``100·eps·|dev|`` — mgcv's ``control$eps`` partial-matches
    ``epsilon``), or after ``efs_maxit`` (mgcv hard-codes 200).

    mgcv's efsudr takes NO ``L``/``lsp0`` (gam.outer:1665 never passes
    them) — id-linked or partially-fixed smoothing parameters are
    unsupported upstream; ``estimate_gam`` raises before calling this.
    """
    ctrl = control or _GAM_CONTROL_DEFAULTS
    epsilon = ctrl["epsilon"]
    efs_lspmax = ctrl["efs_lspmax"]
    efs_tol = ctrl["efs_tol"]
    efs_maxit = ctrl["efs_maxit"]
    if UrS is None or reparam_Y is None:
        raise RuntimeError(
            "optimizer='efs' needs the gam.reparam range-space basis "
            "(UrS); this model fell back to the assembled-eigen path.")
    estimate_scale = log_phi0 is not None
    nsp = len(UrS)
    rho = np.asarray(rho0, dtype=float) + 2.5   # lsp[spind] + 2.5
    log_phi = log_phi0
    mult = 1.0
    n_theta = int(family.n_theta)

    def _fit_and_score(rho_c, log_phi_c, theta_c):
        # One gam.fit3/4 call at scoreType="EFS": seed θ from the lsp
        # snapshot, thread the trial scale, evaluate REML (mgcv units)
        # at the scale the R code would (trial for fit3, the fit's own
        # local scale for fit4). Returns (fit, REML, log φ_REML,
        # scale.est) — the last is mgcv's ``fit$scale`` (R $ partial-
        # matches scale.est): gam.fit3's Fletcher/Pearson/deviance
        # estimator, gam.fit4's local scale.
        if theta_c is not None:
            family.set_theta(theta_c)
        if family_mgcv_extended:
            trial = (float(np.exp(log_phi_c)) if estimate_scale
                     else float(scale_fixed_value))
            fit_c = fit_fn(rho_c, efs_scale=trial)
            se_c = float(fit_c.scale_est)
            lp_reml = float(np.log(se_c))
        else:
            fit_c = fit_fn(rho_c)
            se_c = (float(scale_est_fn(fit_c)) if estimate_scale
                    else None)
            lp_reml = (float(log_phi_c) if estimate_scale
                       else float(np.log(scale_fixed_value)))
        reml_c = 0.5 * float(reml_fn(rho_c, lp_reml, fit_c))
        return fit_c, reml_c, lp_reml, se_c

    def _get_theta():
        return family.get_theta().copy() if n_theta > 0 else None

    def _refresh_log_phi(se_c):
        # lsp[length(lsp)] <- log(fit$scale) (gam.fit4.r:845/887).
        return float(np.log(se_c)) if estimate_scale else None

    theta_state = _get_theta()
    fit, fit_reml, fit_lp_reml, fit_se = _fit_and_score(rho, log_phi,
                                                        theta_state)
    theta_state = _get_theta()                # lsp[thind] <- getTheta()
    log_phi = _refresh_log_phi(fit_se)

    score_hist = np.zeros(efs_maxit)
    bSb = np.zeros(nsp)
    trVS = np.zeros(nsp)
    Y = reparam_Y
    old_dev = None
    it = 0
    for it in range(1, efs_maxit + 1):
        beta = np.asarray(fit.beta, dtype=float)
        Yb = Y.T @ beta                        # coefs in penalty range space
        fit_F = fisher_view_fn(fit)            # rV is Fisher-weight (gdi2)
        for i in range(nsp):
            M_i = Y @ UrS[i]                   # p × k_i, S_i = M_i·M_i'
            Z_i = cho_solve((fit_F.A_chol, fit_F.A_chol_lower), M_i)
            trVS[i] = float(np.sum(M_i * Z_i))  # tr(M_i'(V/φ)M_i)
            xx = Yb @ UrS[i]
            bSb[i] = float(np.sum(xx * xx))     # β'S_iβ
        # φ for the update: mgcv's ``phi <- fit$scale`` — the ACCEPTED
        # fit's scale.est, edf-corrected for scale-unknown extended
        # families (gam.fit4.r:866-871) — or the fixed scale.
        if estimate_scale:
            phi = float(fit_se)
            if family_mgcv_extended:
                edf = float(p) - float(np.sum(trVS * np.exp(rho)))
                phi = phi * n / (n - edf)
        else:
            phi = float(scale_fixed_value)
        det1 = np.asarray(
            _reparam_eval(UrS, reparam_cache, rho)["det1"], dtype=float)
        a = np.maximum(0.0, det1 * np.exp(-rho) - trVS)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / np.maximum(0.0, bSb) * phi
        r[(a == 0.0) & (bSb == 0.0)] = 1.0
        r[~np.isfinite(r)] = 1e6
        with np.errstate(divide="ignore"):
            log_r = np.log(r)
        rho1 = np.minimum(rho + log_r * mult, efs_lspmax)
        max_step = float(np.max(np.abs(rho1 - rho)))
        old_reml = fit_reml
        fit, fit_reml, fit_lp_reml, fit_se = _fit_and_score(
            rho1, log_phi, theta_state)
        theta1 = _get_theta()
        log_phi1 = _refresh_log_phi(fit_se)

        if fit_reml <= old_reml:                # improvement
            if max_step < 0.05:                 # near optimum: try ×2 step
                rho2 = np.minimum(rho + log_r * mult * 2.0, efs_lspmax)
                fit2, fit2_reml, fit2_lp_reml, fit2_se = _fit_and_score(
                    rho2, log_phi, theta_state)
                theta2 = _get_theta()
                log_phi2 = log_phi1  # mgcv quirk: log(fit$scale), not fit2's
                if fit2_reml < fit_reml:        # accept extension
                    fit, fit_reml, fit_lp_reml, fit_se = (
                        fit2, fit2_reml, fit2_lp_reml, fit2_se)
                    rho, theta_state, log_phi = rho2, theta2, log_phi2
                    mult = mult * 2.0
                else:                           # keep the ×1 step; the next
                    # fit re-seeds θ from lsp1's slots (θ of the ×1 fit),
                    # discarding fit2's family-state θ — as mgcv's lsp does.
                    rho, theta_state, log_phi = rho1, theta1, log_phi1
            else:
                rho, theta_state, log_phi = rho1, theta1, log_phi1
        else:                                   # no improvement: contract,
            # never below mult=1 (the update needn't improve REML)
            while fit_reml > old_reml and mult > 1.0:
                mult = mult / 2.0
                rho1 = np.minimum(rho + log_r * mult, efs_lspmax)
                # lsp1 <- lsp: θ/scale re-seed from the pre-trial state
                fit, fit_reml, fit_lp_reml, fit_se = _fit_and_score(
                    rho1, log_phi, theta_state)
                theta1 = _get_theta()
                log_phi1 = _refresh_log_phi(fit_se)
            rho, theta_state, log_phi = rho1, theta1, log_phi1
            if mult < 1.0:
                mult = 1.0
        score_hist[it - 1] = fit_reml
        # break if the EFS step is small and REML flat over 3 steps...
        if (it > 3 and max_step < 0.05
                and float(np.max(np.abs(np.diff(
                    score_hist[it - 4:it])))) < efs_tol):
            break
        # ...or if the deviance has stopped changing
        if it == 1:
            old_dev = float(fit.dev)
        else:
            if abs(old_dev - float(fit.dev)) < 100.0 * epsilon * abs(
                    float(fit.dev)):
                break
            old_dev = float(fit.dev)

    # Leave the family at the ACCEPTED fit's θ (the lsp state): a
    # rejected extension/contraction trial leaves its own θ in the
    # family env — in mgcv that leak reaches only the getTheta()
    # display string, while the returned object (rV/K/REML → edf/Vp)
    # is entirely the accepted fit's; hea recomputes those post-fit
    # from (fit, family state), so the state must be the accepted θ.
    if n_theta > 0:
        family.set_theta(theta_state)
    outer_info = {
        "conv": ("iteration limit reached" if it == efs_maxit
                 else "full convergence"),
        "iter": it,
        "score_hist": score_hist[:it].copy(),
    }
    return {"fit": fit, "rho": rho,
            "log_phi_reml": (fit_lp_reml if estimate_scale else None),
            "outer_info": outer_info}


# ---------------------------------------------------------------------------
# magic — GCV/UBRE fast path for the Gaussian-additive model. Ports mgcv's C
# engine (src/magic.c) + the R wrapper (mgcv.r:4678): QR the √w·X design once,
# then Newton over log-sp on the reduced (q+rank_S)×q system. Free fns 1:1 with
# magic.c's fit_magic (c:62) / magic_gH (c:193) / magic driver (c:286). The
# dispatch predicate `_use_magic` stays a `gam` method (it is estimate.gam-level,
# not part of magic itself — mgcv.r:2001 chooses am.fit → magic there).
# ---------------------------------------------------------------------------


def _magic_setup(*, wt, y, offset, struct_R, keep_cols, X):
    """magic's ``getRpqr`` (magic.c:451): QR the n×q √w-weighted design
    ONCE → R (q×q), y0 = Q₁'(√w·y), yy = ‖√w·y‖². Reused for every
    reduced score-eval — this is the work the Newton path repeats n times."""
    sqw = np.sqrt(wt)
    yv = y - offset
    b = sqw * yv
    if struct_R is not None and keep_cols is None:
        # Reuse the structural-drop QR (same √w·X under gaussian-identity,
        # no drop): R is bit-identical to qr(√wt·X).R, and the getRpqr
        # projection y0 = Q₁'b = R⁻ᵀ(Xw'b) needs no second QR (Xw = √w·X,
        # so Xw'b = X'(wt·yv)). Matches the explicit-Q route to ~1e-15.
        R = struct_R
        y0 = solve_triangular(
            R, X.T @ (wt * yv), lower=False, trans="T")
    else:
        Q, R = np.linalg.qr(sqw[:, None] * X)
        y0 = Q.T @ b
    yy = float(b @ b)
    return R, y0, yy


def _magic_penalty_roots(*, p, slots) -> list[np.ndarray]:
    """Per-penalty full p×cS_i roots rS_i (rS_i rS_iᵀ = S_i), one per
    ``slots`` entry (== the ρ order). cS_i = rank(S_i)."""
    roots = []
    for slot in slots:
        a, b = slot.col_start, slot.col_end
        ev, V = np.linalg.eigh(0.5 * (slot.S + slot.S.T))
        pos = ev > ev.max() * 1e-12 if ev.max() > 0 else np.zeros_like(ev, bool)
        rb = V[:, pos] * np.sqrt(ev[pos])           # k × cS_i
        full = np.zeros((p, rb.shape[1]))
        full[a:b, :] = rb
        roots.append(full)
    return roots


def _magic_fit_reduced(rho, R, y0, yy, *, wt, gamma, slots, p,
                       norm_const=0.0, n_score=None, scale=None, gcv=True):
    """Port of magic.c ``fit_magic``: GCV/UBRE score from the SVD of the
    reduced ``[R; St^½]`` ((q+rank_S)×q). St = Σ exp(ρ_k) S_k.
    Returns score, scale, rss, trA, rank, β, plus the SVD pieces
    (U1, V, d, y1, delta) the derivative routine reuses.

    ``norm_const`` is magic's additive RSS constant (``extra.rss`` — the
    part of ‖y‖² orthogonal to the reduced design, bam.r:1675/1146);
    ``n_score`` overrides the score-n (magic.c:148 ``n = *n_score``);
    ``gcv=False`` selects the UBRE score at the supplied ``scale``
    (magic.c:151)."""
    q = R.shape[1]
    if n_score is None:
        n_score = float(np.sum(wt != 0.0))
    else:
        n_score = float(n_score)
    rank_tol = np.sqrt(np.finfo(float).eps)
    St = _s_lambda(slots, p, rho)
    St = 0.5 * (St + St.T)
    ev, Vev = np.linalg.eigh(St)
    pos = ev > ev.max() * 1e-14 if ev.max() > 0 else np.zeros_like(ev, bool)
    root = (Vev[:, pos] * np.sqrt(ev[pos])).T       # rank_S × q, rootᵀroot = St
    R_aug = np.vstack([R, root])
    U, d, Vt = np.linalg.svd(R_aug, full_matrices=False)
    rank = q
    thresh = d[0] * rank_tol
    while d[rank - 1] < thresh:
        rank -= 1
    Vr = Vt[:rank].T                                # q × rank
    U1 = U[:q, :rank]                               # q × rank (top q rows)
    y1 = U1.T @ y0
    yAy = float(y1 @ y1)
    b_proj = U1 @ y1
    yAAy = float(b_proj @ b_proj)
    norm = yy - 2 * yAy + yAAy
    if norm < 0.0:
        norm = 0.0
    trA = float(np.sum(U1 * U1))
    delta = n_score - gamma * trA
    if gcv:
        # magic.c:150 — scale estimated alongside the GCV score.
        score = n_score * (norm + norm_const) / (delta * delta)
        scale_out = (norm + norm_const) / (n_score - trA)
    else:
        # magic.c:151 — UBRE/approximate AIC at the known scale.
        score = ((norm + norm_const) / n_score
                 - 2.0 * scale / n_score * delta + scale)
        scale_out = scale
    beta = Vr @ (y1 / d[:rank])
    return dict(score=score, scale=scale_out, norm=norm, trA=trA, rank=rank,
                beta=beta, U1=U1, V=Vr, d=d[:rank], y1=y1, delta=delta,
                n_score=n_score, gamma=gamma, gcv=gcv, norm_const=norm_const,
                ubre_scale=scale)


def _magic_gH(mg, rho, roots):
    """Port of magic.c ``magic_gH``: gradient (exact) and Hessian of the
    GCV score wrt ρ, from the reduced SVD in ``mg`` — O(m·rank²), no
    re-fit. The Hessian is magic's frozen-basis *search-direction* approx
    (it differentiates the explicit sp-dependence, not the SVD basis); the
    driver converges on the exact gradient with an SD fallback."""
    U1, V, d, y1 = mg["U1"], mg["V"], mg["d"], mg["y1"]
    gamma, n = mg["gamma"], mg["n_score"]
    norm, delta = mg["norm"], mg["delta"]
    m = len(roots)
    Dinv = 1.0 / d
    U1U1 = U1.T @ U1
    esp = np.exp(np.asarray(rho, float))
    M = [None] * m
    K = [None] * m
    My = [None] * m
    yK = [None] * m
    Ky = [None] * m
    for i in range(m):
        VSi = (V.T @ roots[i]) * Dinv[:, None]      # D⁻¹V'rS_i (rank×cS_i)
        Mi = VSi @ VSi.T                            # D⁻¹V'S_iVD⁻¹
        Ki = Mi @ U1U1
        M[i], K[i] = Mi, Ki
        My[i] = Mi @ y1
        yK[i] = Ki.T @ y1
        Ky[i] = Ki @ y1
    ddelta = np.zeros(m)
    dnorm = np.zeros(m)
    d2delta = np.zeros((m, m))
    d2norm = np.zeros((m, m))
    for i in range(m):
        ddelta[i] = gamma * np.trace(K[i]) * esp[i]
        for j in range(i + 1):
            v = float(np.sum(M[j] * K[i]))
            d2delta[i, j] = d2delta[j, i] = -gamma * 2 * esp[i] * esp[j] * v
        d2delta[i, i] += ddelta[i]
        dnorm[i] = 2 * esp[i] * float(y1 @ (My[i] - Ky[i]))
        for j in range(i + 1):
            v = float(np.sum(My[i] * Ky[j] + My[j] * Ky[i]
                             - 2 * My[i] * My[j] + yK[i] * My[j]))
            d2norm[i, j] = d2norm[j, i] = v * 2 * esp[i] * esp[j]
        d2norm[i, i] += dnorm[i]
    grad = np.zeros(m)
    hess = np.zeros((m, m))
    if mg.get("gcv", True):
        # GCV (control[0]==1) — magic.c:263-273; the RSS is inflated by
        # norm_const (``norm += *norm_const``) before the score algebra.
        norm = norm + mg.get("norm_const", 0.0)
        xx = n / (delta * delta)
        xx1 = xx * 2 * norm / delta
        x1 = -2 * xx / delta
        x2 = 3 * xx1 / delta
        for i in range(m):
            grad[i] = xx * dnorm[i] - xx1 * ddelta[i]
            for j in range(i + 1):
                hess[i, j] = hess[j, i] = (
                    x1 * (ddelta[j] * dnorm[i] + ddelta[i] * dnorm[j])
                    + xx * d2norm[i, j] + x2 * ddelta[i] * ddelta[j]
                    - xx1 * d2delta[i, j])
    else:
        # UBRE — magic.c:275-279 (norm_const drops out of derivatives).
        scale = mg["ubre_scale"]
        for i in range(m):
            grad[i] = (dnorm[i] - 2.0 * scale * ddelta[i]) / n
            for j in range(i + 1):
                hess[i, j] = hess[j, i] = (
                    d2norm[i, j] - 2.0 * scale * d2delta[i, j]) / n
    return grad, hess


def _magic_optimize(rho0, *, tol=1e-7, max_half=15, wt, y, offset, struct_R,
                    keep_cols, X, gamma, slots, p, norm_const=0.0,
                    n_score=None, scale=None, gcv=True, L=None, lsp0=None):
    """Port of magic.c ``magic`` (the driver, magic.c:286-707): Newton
    over log-sp backed by steepest descent with step halving, plus the
    infinite-sp check, on the reduced GCV/UBRE system. Returns ρ̂, the
    once-computed QR (``magic_R``/``magic_y0``) the final fit reuses, and
    the outer_info. (Flat-deriv reset deferred — bam's call is always
    autoinit, magic.c:510, where the reset block is skipped.)

    ``L``/``lsp0`` mirror the C driver's working→penalty map: score
    evaluations run at ``ρ_full = L·ρ + lsp0`` and the gradient/Hessian
    transform as ``L'g`` / ``L'HL`` (magic.c:606-621); ``None`` ≡
    identity. ``norm_const``/``n_score``/``scale``/``gcv`` thread the
    score flavor through to ``_magic_fit_reduced``."""
    R, y0, yy = _magic_setup(wt=wt, y=y, offset=offset, struct_R=struct_R,
                             keep_cols=keep_cols, X=X)
    roots = _magic_penalty_roots(p=p, slots=slots)
    mp = len(rho0)
    sp0 = np.asarray(rho0, float).copy()
    Lm = None if L is None else np.asarray(L, dtype=float)
    l0 = None
    if Lm is not None:
        l0 = (np.zeros(Lm.shape[0]) if lsp0 is None
              else np.asarray(lsp0, dtype=float))

    def sp_full(sp):
        return sp if Lm is None else Lm @ sp + l0

    def feval(sp):
        return _magic_fit_reduced(sp_full(sp), R, y0, yy, wt=wt, gamma=gamma,
                                  slots=slots, p=p, norm_const=norm_const,
                                  n_score=n_score, scale=scale, gcv=gcv)

    mg = feval(sp0)
    min_score = mg["score"]
    n_step = np.zeros(mp)
    sd_step = np.zeros(mp)
    use_sd = True
    d_score = np.inf
    grad = np.zeros(mp)
    converged = False
    it = 0
    fit_calls = 1
    step_fail = False
    while not converged:
        it += 1
        if it > 400:
            converged = True
            break
        last_try = 0
        if it > 1:
            step = sd_step if use_sd else n_step
            try_i = 0
            ok = True
            while ok:
                try_i += 1
                if try_i == 4 and not use_sd:
                    use_sd = True
                    step = sd_step
                nsp = sp0 + step
                mg = feval(nsp)
                fit_calls += 1
                if mg["score"] < min_score:
                    d_score = min_score - mg["score"]
                    min_score = mg["score"]
                    sp0 = nsp.copy()
                    ok = False
                else:
                    step = step / 2
                if try_i == max_half - 1 and ok:
                    step = step * 0.0
                if try_i == max_half:
                    ok = False
            last_try = try_i
        if it > 3:
            converged = True
            if d_score > tol * (1 + min_score):
                converged = False
            gnorm = np.sqrt(float(grad @ grad))
            if gnorm > tol ** (1 / 3) * (1 + abs(min_score)):
                converged = False
            if last_try == max_half:
                # magic.c:599 — a fully-halved (failed) step means the
                # score can't be improved: force convergence.
                converged = True
                step_fail = True
        # mgcv's magic builds the gradient/Hessian from the fit it
        # already has in hand — magic.c:611 calls magic_gH on the U1/V/d
        # left by the last fit_magic, with NO re-fit at sp0. `mg` already
        # holds the fit at the current sp0 in every branch (the it==1
        # seed; the accepted step, which sets sp0=nsp; or the step-failure
        # path, whose last try evaluates sp0+0), and the reduced SVD is
        # deterministic, so reusing it is bit-identical to re-evaluating
        # — and saves one (q+e)×q SVD per outer iteration.
        grad, hess = _magic_gH(mg, sp_full(sp0), roots)
        if Lm is not None:
            # magic.c:617-620 — grad/Hess w.r.t. the working parameters.
            grad = Lm.T @ grad
            hess = Lm.T @ hess @ Lm
        ev, U = np.linalg.eigh(0.5 * (hess + hess.T))
        use_sd = bool(np.any(ev <= 0.0))
        if not use_sd:
            gtil = U.T @ grad
            n_step = -(U @ (gtil / ev))
            mx = np.max(np.abs(n_step))
            if mx > 5.0:
                n_step *= 5.0 / mx
        gmx = np.max(np.abs(grad))
        sd_step = -grad / gmx if gmx > 0 else grad * 0.0
    # infinite-sp check (magic.c:638-656): push each sp in its descent
    # direction by ±2 in log-sp (≤5 steps) while the score keeps improving.
    for k in range(mp):
        steps_left = 5
        sign = 1.0 if grad[k] < 0.0 else -1.0
        while steps_left:
            sp0[k] += sign * 2.0
            steps_left -= 1
            sc = feval(sp0)["score"]
            fit_calls += 1
            if sc < min_score:
                min_score = sc
            else:
                sp0[k] -= sign * 2.0
                steps_left = 0
    # magic.c:664 — one FINAL fit_magic at the converged sp; ITS score and
    # scale are what magic returns (``*gamma = score``; scale in-place).
    mg = feval(sp0)
    fit_calls += 1
    outer_info = {
        "iter": it,
        "conv": "full convergence" if converged else "iteration limit reached",
        "grad": grad,
        "score": mg["score"],
        "scale": mg["scale"],
        "step_fail": step_fail,
        "optimizer": "magic",
    }
    return {"sp": sp0, "magic_R": R, "magic_y0": y0, "outer_info": outer_info}


def _initial_sp(Xw: np.ndarray, slots) -> np.ndarray:
    """mgcv ``initial.sp(X, S, off)`` (mgcv.r:4626-4673) on an (already
    weighted, if applicable) design: per penalty

        def.sp[k] = mean(diag(X'X)[ind]) / mean(diag(S_k)[ind])

    with ``ind`` filtering S_k to its penalised rows/cols (``thresh =
    eps^0.8·max|S_k|`` on row-mean, col-mean and diagonal simultaneously),
    then the global ×10 rebalance pushing ``mean(ldxx/(ldxx+ldss))``
    across 0.4 (mgcv.r:4666-4670). Returns def.sp (not logged).
    ``initial.spg`` calls this on ``√w·X`` (mgcv.r:4605); plain ``magic``
    calls it on the raw X (mgcv.r:4712)."""
    ldxx = np.einsum("ij,ij->j", Xw, Xw)  # diag(X'X)
    ldss = np.zeros_like(ldxx)
    pen = np.zeros(ldxx.size, dtype=bool)
    n_sp = len(slots)
    def_sp = np.ones(n_sp)
    for k, slot in enumerate(slots):
        S_k = slot.S
        absS = np.abs(S_k)
        maS = float(absS.max()) if absS.size else 0.0
        if maS <= 0.0:
            continue  # mgcv would stop(); a free penalty seeds at ρ=0
        thresh = float(np.finfo(float).eps ** 0.8) * maS
        rsS = absS.mean(axis=1)
        csS = absS.mean(axis=0)
        dS = np.abs(np.diag(S_k))
        ind = (rsS > thresh) & (csS > thresh) & (dS > thresh)
        if not np.any(ind):
            continue
        ss = np.diag(S_k)[ind]
        sl = slice(slot.col_start, slot.col_end)
        xx = ldxx[sl][ind]
        pen[sl] |= ind
        sizeXX = float(np.mean(xx))
        sizeS = float(np.mean(ss))
        if sizeS <= 0.0 or sizeXX <= 0.0:
            continue
        def_sp[k] = sizeXX / sizeS
        ldss[sl] += def_sp[k] * np.diag(S_k)

    bind = (ldss > 0) & pen & (ldxx > 0)
    if np.any(bind):
        lx = ldxx[bind]
        ls = ldss[bind].copy()
        while float(np.mean(lx / (lx + ls))) > 0.4:
            def_sp *= 10.0
            ls *= 10.0
        while float(np.mean(lx / (lx + ls))) < 0.4:
            def_sp /= 10.0
            ls /= 10.0
    return def_sp


def _magic_gcv(y: np.ndarray, X: np.ndarray, slots, *,
               gamma: float = 1.0) -> dict:
    """mgcv ``magic(y, X, sp=rep(-1,m), S, off)`` (mgcv.r:4678) essentials
    as consumed by mvn's preinitialize (mvam.r:119): every sp estimated by
    GCV from the ``initial.sp`` seed (mgcv.r:4712 ``def.sp``), unit
    weights, no offset/L/H/C, magic's own control defaults (tol=1e-6,
    step.half=25 — NOT gam.control's 1e-7/15). Returns ``{"b", "scale",
    "sp"}`` — mgcv's ``um$b`` / ``um$scale = norm/(n−trA)``."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    wt = np.ones(n)
    if len(slots) == 0:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rsd = y - X @ b
        scale = float(rsd @ rsd) / max(n - p, 1)
        return {"b": b, "scale": scale, "sp": np.zeros(0)}
    def_sp = _initial_sp(X, slots)
    rho0 = np.log(np.maximum(def_sp, 1e-300))
    res = _magic_optimize(rho0, tol=1e-6, max_half=25, wt=wt, y=y,
                          offset=np.zeros(n), struct_R=None,
                          keep_cols=None, X=X, gamma=gamma,
                          slots=slots, p=p)
    mg = _magic_fit_reduced(res["sp"], res["magic_R"], res["magic_y0"],
                            float(y @ y), wt=wt, gamma=gamma,
                            slots=slots, p=p)
    return {"b": mg["beta"], "scale": float(mg["scale"]),
            "sp": np.exp(np.asarray(res["sp"], dtype=float))}


def _magic_fit_state(rho, *, magic_R, magic_y0, fit_given_rho_fn, family, X,
                     offset, y, wt, slots, p, UrS, reparam_Y, keep_cols,
                     reparam_cache):
    """Build the Gaussian-identity ``_FitState`` at ρ̂ from magic's cached
    QR, WITHOUT the full (n+e)×q re-fit. The reduced augmented system
    ``[R_mag; E]`` (R_mag = QR factor of √w·X, cached by `_magic_optimize`)
    has ``R_fin'R_fin = R_mag'R_mag + Sλ = X'WX + Sλ`` and RHS reduces to
    ``[y0; 0]`` (y0 = Q_mag'(√w·z), z = y−off) — so β, A_chol, log|A| are
    the same as `_pls_qr`'s full solve at (q+e)×q cost. Mirrors the
    Gaussian branch of `_fit_given_rho` field-for-field. Falls back to the
    full fit if anything is degenerate (rank deficiency / non-finite)."""
    R_mag = magic_R
    y0 = magic_y0
    if R_mag is None:
        return fit_given_rho_fn(rho)
    link = family.link
    off = offset
    q = R_mag.shape[1]
    Sλ = _s_lambda(slots, p, rho)
    Sλ = 0.5 * (Sλ + Sλ.T)
    E_aug = _penalty_root_of(slots, p, UrS, reparam_Y, keep_cols,
                             reparam_cache, rho)
    # reduced augmented QR (mirrors _pls_qr's no-negative-weight branch)
    aug = np.vstack([R_mag, E_aug])
    Q, R_fin = np.linalg.qr(aug)
    diag_R = np.diag(R_fin)
    if (not np.all(np.isfinite(R_fin))) or np.any(diag_R == 0.0):
        return fit_given_rho_fn(rho)
    c = Q[:q].T @ y0
    beta = solve_triangular(R_fin, c, lower=False)
    if not np.all(np.isfinite(beta)):
        return fit_given_rho_fn(rho)
    log_det_A = 2.0 * float(np.sum(np.log(np.abs(diag_R))))
    sgn = np.where(diag_R < 0, -1.0, 1.0)
    A_chol = R_fin * sgn[:, None]
    # Gaussian-identity fit fields — identical to _fit_given_rho's tail.
    eta = X @ beta                       # offset-stripped
    eta_full = eta + off
    mu = link.linkinv(eta_full)
    dev = float(np.sum(family.dev_resids(y, mu, wt)))
    pen = float(beta @ Sλ @ beta)
    mu_eta_v = link.mu_eta(eta_full)
    V = family.variance(mu)
    d2g = link.d2link(mu)
    good = (wt > 0.0) & (mu_eta_v != 0.0)
    safe_mu_eta = np.where(good, mu_eta_v, 1.0)
    alpha = 1.0 + (y - mu) * (family.dvar(mu) / V + d2g * mu_eta_v)
    alpha = np.where(alpha == 0.0, np.finfo(float).eps, alpha)
    z = np.where(good, eta + (y - mu) / (safe_mu_eta * alpha), 0.0)
    w = np.where(good, wt * alpha * mu_eta_v ** 2 / V, 0.0)
    return _FitState(
        beta=beta, dev=dev, pen=pen,
        A_chol=A_chol, A_chol_lower=False,
        S_full=Sλ, log_det_A=log_det_A,
        eta=eta_full, mu=mu, w=w, z=z, alpha=alpha,
        is_fisher_fallback=False,
        converged=True, boundary=False, warn=[],
        E_aug=E_aug,
    )


class _G:
    """mgcv's ``G`` setup bundle (``gam.setup`` output, mgcv.r:1096) — the
    subset ``estimate_gam`` reads for the ordinary-family path. mgcv threads
    the full G (design/penalties/…) and calls ``gam.fit3`` BY NAME; hea threads
    this data plus the fitter callables (the same C1 poly seam newton/bfgs
    already use). Extensible to the general-family / bam paths later."""

    def __init__(self, *, n, n_sp, n_work, Mp, L, lsp0, wt, y_arr, family,
                 scale_known_fit, pearson_scale_criterion, control):
        self.n = n
        self.n_sp = n_sp
        self.n_work = n_work
        self.Mp = Mp
        self.L = L
        self.lsp0 = lsp0
        self.wt = wt
        self.y_arr = y_arr
        self.family = family
        self.scale_known_fit = scale_known_fit
        self.pearson_scale_criterion = pearson_scale_criterion
        self.control = control


class _GGeneral:
    """The general-family (``gam.fit5``) slice of ``G`` — present when the
    family is a general.family (Cox/gamlss/multinom/…). mgcv's ``estimate.gam``
    branches internally on ``inherits(G$family,"general.family")`` (mgcv.r:1893/
    1984/2021/2041); hea mirrors that with ``estimate_gam(..., general=...)``.
    Carries the general-path data (reparameterized design ``X``, penalty ``sl``,
    ``lpi``, ``g5`` warm-fit cache) + the shared bits the general branch reads."""

    def __init__(self, *, n_work, Mp, wt, family, control, gamma, X, y, sl,
                 md_L, lpi, offsets, p, slots, g5, optimizer,
                 seed_start=None):
        self.n_work = n_work
        self.Mp = Mp
        self.wt = wt
        self.family = family
        self.control = control
        self.gamma = gamma
        self.X = X
        self.y = y
        self.sl = sl
        self.md_L = md_L
        self.lpi = lpi
        self.offsets = offsets
        self.p = p
        self.slots = slots
        self.g5 = g5
        self.optimizer = optimizer
        # user start= in the basis initial.spg's family calls expect:
        # irp'd beside the irp'd dense X (mgcv.r:1903 → :1998 ordering),
        # MODEL-basis on the discrete rail.
        self.seed_start = seed_start


def estimate_gam(G, sp, method, *, rho_full, outer_newton, fit_given_rho=None,
                 initial_sp_rho=None, use_magic=None, phi_pearson=None,
                 profile_log_phi_fixed_sp=None, magic_fit_state=None,
                 magic_optimize=None, get_outer_fit=None, general=None,
                 outer_bfgs=None, get_outer_info=None,
                 optimizer=("outer", "newton"), outer_efsudr=None,
                 outer_nlm_optim=None):
    """mgcv ``estimate.gam`` (mgcv.r:1872) — smoothness selection + final fit,
    for BOTH ordinary and general families (mgcv branches internally on the
    family class; hea takes the general slice via ``general=``). Ordinary
    returns ``{fit, rho_hat, sp, log_phi_hat, used_magic, tw_info}``; general
    returns ``{fit, rho_hat, sp, outer_info, REML_criterion, converged}``. The
    caller (``gam.__init__`` / ``gam._init_general``) assigns these onto ``self``.

    mgcv threads the setup bundle ``G`` (data) and calls ``gam.fit3`` by name
    (it dispatches to fit4/fit5 internally). hea threads ``G`` plus the fitter
    callables — ``fit_given_rho`` (= gam.fit3/gam.fit4 dispatch), ``outer_newton``
    (= newton), ``magic_optimize`` (= magic driver), etc. This is the same C1
    seam newton/bfgs use one level down; ``fit_given_rho``/``initial_sp_rho`` are
    bam-polymorphic, but ``estimate_gam`` is only ever called on a plain ``gam``
    (bam has its own ``__init__``), so the callables always resolve to gam's.
    ``get_outer_fit`` reads the fit ``outer_newton`` cached as a side effect
    (mgcv reuses newton's last accepted gam.fit3, mgcv.r:1684 / gam.fit3.r:1718).
    """
    if general is not None:
        # ---- general.family branch (gam.fit5 pipeline, mgcv.r:1984+) --------
        gg = general
        n_work = gg.n_work
        family = gg.family
        avail_derivs = int(getattr(family, "available_derivs", 2) or 0)
        if isinstance(gg.X, _DiscreteX) and avail_derivs > 1:
            # discrete rail: gamlss.gH tops out at the deriv-1 trace
            # (gamlss.r:777), so the EFFECTIVE ll order is 1 whatever
            # the family declares — mgcv.r:1907 then coerces the outer
            # optimizer to bfgs (and the fixed-sp refit below to
            # deriv 0) exactly as for a declared available.derivs==1.
            avail_derivs = 1
        efs_forced = gg.optimizer[0] == "efs"
        # mgcv coerces available.derivs==1 → bfgs unless efs requested
        # (mgcv.r:1907): the ll supplies ≤ dH/trace order, so Newton's REML2
        # (needing trHid2H) is unavailable. optimizer=("outer","bfgs") forces
        # bfgs on any general family.
        bfgs_forced = len(gg.optimizer) > 1 and gg.optimizer[1] == "bfgs"
        use_efs = avail_derivs == 0 or efs_forced
        use_bfgs = (avail_derivs == 1 or bfgs_forced) and not efs_forced
        outer_info = None
        if sp is not None:
            sp_arr = np.asarray(sp, dtype=float).flatten()
            if sp_arr.shape != (n_work,):
                raise ValueError(
                    f"sp must have length {n_work} (working smoothing "
                    f"parameters), got {sp_arr.shape}")
            if np.any(sp_arr <= 0) or not np.all(np.isfinite(sp_arr)):
                raise ValueError("sp entries must be positive and finite")
            theta_hat = np.log(sp_arr)
            outer_info = {"conv": "fixed sp", "iter": 0}
        elif n_work == 0:
            theta_hat = np.zeros(0)
            outer_info = {"conv": "no smoothing parameters", "iter": 0}
        elif use_efs:
            # available_derivs == 0 → extended Fellner-Schall (mgcv.r:1907-1908).
            if gg.md_L is not None:
                raise NotImplementedError(
                    "efs (available_derivs=0) with id-linked smoothing "
                    "parameters is not supported.")
            sl_setup = _sl_setup(gg.slots, gg.p)
            theta0 = _initial_sp_general(
                gg.X, gg.y, family, gg.slots, gg.lpi, weights=gg.wt,
                offsets=gg.offsets, L=None, start=gg.seed_start)
            fit_efs, theta_hat, it_efs = _efsud(
                gg.X, gg.y, theta0, gg.sl, sl_setup, family=family,
                lpi=gg.lpi, weights=gg.wt, offset=gg.offsets,
                Mp=gg.Mp, start=gg.g5["start"], control=gg.control)
            gg.g5["fit"] = fit_efs
            gg.g5["start"] = fit_efs["coefficients"]
            outer_info = {
                "conv": ("iteration limit reached"
                         if it_efs == gg.control["efs_maxit"]
                         else "full convergence"),
                "iter": it_efs,
            }
        elif use_bfgs:
            # available_derivs == 1 → BFGS over the REML5 score (mgcv.r:1722).
            theta0 = _initial_sp_general(
                gg.X, gg.y, family, gg.slots, gg.lpi, weights=gg.wt,
                offsets=gg.offsets, L=gg.md_L, start=gg.seed_start)
            theta_hat = gam_outer(
                theta0, optimizer="bfgs", criterion="REML5",
                control=gg.control, bfgs_fn=outer_bfgs)
            # gam_outer's shim set self._outer_info as a side effect.
            outer_info = get_outer_info()
        else:
            theta0 = _initial_sp_general(
                gg.X, gg.y, family, gg.slots, gg.lpi, weights=gg.wt,
                offsets=gg.offsets, L=gg.md_L, start=gg.seed_start)
            theta_hat = gam_outer(
                theta0, optimizer="newton", criterion="REML5",
                control=gg.control, newton_fn=outer_newton)
            outer_info = get_outer_info()

        rho_hat = rho_full(theta_hat)
        sp_out = np.exp(theta_hat)
        # The converged deriv-2 fit: newton's/efs's last accepted iterate
        # (cached in g5 — mgcv's b; estimate.gam never refits). Fixed-sp/no-sp
        # paths fit directly, deriv 0 for derivs-0 families (mgcv.r:1479+).
        fit = gg.g5.get("fit")
        if fit is None:
            fit = _gam_fit5(
                gg.X, gg.y, rho_hat, gg.sl, family=family, lpi=gg.lpi,
                weights=gg.wt, offset=gg.offsets, Mp=gg.Mp,
                deriv=2 if avail_derivs >= 2 else 0,
                start=gg.g5["start"], gamma=gg.gamma, epsilon=1e-8)
        if fit["warn"]:
            import warnings as _warnings
            for w_msg in fit["warn"]:
                _warnings.warn(w_msg, stacklevel=2)
        # hea stores 2·V_R (single-formula convention: mgcv's printed REML is
        # REML_criterion/2).
        # mgcv never sets $converged on a general-family gam (live 1.9-4
        # receipt: NULL; the user-facing signal is outer.info$conv), so
        # the flag is hea's contract: the inner fit's flag, rescued by a
        # fully-converged outer trajectory — the last accepted refit may
        # legitimately end on gam.fit4.r:1206's step-fail endgame when
        # warm-started at a neighbouring trial's optimum (e.g. a user
        # start= at the converged coefficients).
        conv_flag = bool(fit["converged"])
        if (not conv_flag and outer_info is not None
                and outer_info.get("conv") == "full convergence"):
            conv_flag = True
        return {"fit": fit, "rho_hat": rho_hat, "sp": sp_out,
                "outer_info": outer_info,
                "REML_criterion": 2.0 * fit["REML"],
                "converged": conv_flag}

    # ---- ordinary-family branch ---------------------------------------------
    n, n_sp, n_work = G.n, G.n_sp, G.n_work
    family = G.family
    # gam.outer's nbGetTheta stop (mgcv.r:1649-1650): a θ-vector negbin's
    # range search only ever lived in the deprecated performance
    # iteration, so every live path errors here.
    if (family.name.startswith("Negative Binomial")
            and np.asarray(family.get_theta()).size > 1):
        raise ValueError(
            "Please provide a single value for theta or use nb to "
            "estimate it")
    log_phi_hat = None
    tw_info = None
    used_magic = False
    used_nlm_optim = False
    if n_sp == 0:
        # No smooths — degenerate to unpenalized least squares. Still go
        # through the fit so all mgcv post-fit attributes are populated.
        sp_out = np.zeros(0)
        rho_hat = np.zeros(0)
        fit = fit_given_rho(rho_hat)
    elif sp is not None:
        sp_arr = np.asarray(sp, dtype=float)
        # mgcv semantics: ``sp`` supplies the *working* smoothing parameters —
        # one per column of L (== one per penalty when no id linkage).
        if sp_arr.shape != (n_work,):
            raise ValueError(
                f"sp must have length {n_work} (one per estimated "
                f"smoothing parameter; id-linked penalties share one), "
                f"got {sp_arr.shape}"
            )
        if np.any(sp_arr < 0):
            raise ValueError("sp entries must be non-negative")
        # guard log(0) — a hard zero sp means "no penalty".
        rho_hat = rho_full(np.log(np.maximum(sp_arr, 1e-10)))
        sp_out = sp_arr
        fit = fit_given_rho(rho_hat)
        # Unknown-scale (RE)ML at fixed sp: set log φ̂ to the criterion
        # minimizer over log φ at this ρ (gam.fit3.r:121-123 appends log scale;
        # φ-row grad/Hess gam.fit3.r:629-631). β̂/PIRLS are φ-independent.
        if (not G.scale_known_fit) and G.pearson_scale_criterion:
            # P-REML/P-ML: φ̂ = Pearson-Laplace plug-in P/(n−Mp) (gam.fit3.r:641).
            log_phi_hat = float(np.log(max(phi_pearson(fit), 1e-300)))
        elif (not G.scale_known_fit) and method in ("REML", "ML"):
            Dp = float(fit.dev + fit.pen)
            denom = (max(float(n - G.Mp), 1.0) if method == "REML"
                     else max(float(n), 1.0))
            log_phi = float(np.log(max(Dp / denom, 1e-300)))
            if not isinstance(family, (Gaussian, Quasi)):
                log_phi = profile_log_phi_fixed_sp(fit, log_phi)
            log_phi_hat = log_phi
    else:
        # Unified outer optimization (mgcv gam.outer). ``include_log_phi`` is
        # True for unknown-scale families (θ ⊇ (ρ, log φ)); ``include_family_theta``
        # is True for tw() (appends the reparametrised Tweedie power).
        include_log_phi = ((not G.scale_known_fit)
                           and method in ("REML", "ML"))
        include_family_theta = (
            family.n_theta > 0 and method in ("REML", "ML")
        )
        if family.n_theta > 0 and method == "GCV.Cp":
            raise ValueError(
                f"family={family!r} (n_theta={family.n_theta}) requires "
                "method='REML' or 'ML'; got method='GCV.Cp'"
            )
        # Seed at mgcv's ``initial.spg`` balance (estimate.gam mgcv.r:1998:
        # ``lsp <- lsp2`` for REML/ML/GCV alike); id linkage / fixed entries
        # map the full-space seed to working space by least squares
        # (mgcv's ``coef(lm(lsp ~ L - 1 + offset(lsp0)))``, mgcv.r:4617-4618).
        rho0_full = initial_sp_rho()
        if G.lsp0 is not None:
            rho0_full = rho0_full - G.lsp0
        if G.L is None:
            cur_rho = rho0_full
        else:
            cur_rho, *_ = np.linalg.lstsq(G.L, rho0_full, rcond=None)
        # mgcv's null.scale = Σ dev_resids(y, ȳ)/n from get.null.coef
        # (mgcv.r:1854-1870): the log φ seed (mgcv.r:2027-2029) AND the
        # fscale gam.outer hands nlm/optim (mgcv.r:2062, 1698, 1708).
        # mum = mean(y) unweighted (mgcv.r:1863) but the dev.resids
        # carry the prior weights (mgcv.r:1868).
        mu_null0 = np.full(n, float(np.mean(G.y_arr)))
        null_scale = float(np.sum(family.dev_resids(
            G.y_arr, mu_null0, G.wt
        ))) / n
        if include_log_phi:
            cur_logphi = float(np.log(max(null_scale / 10.0, 1e-12)))
        else:
            cur_logphi = 0.0  # GCV does not put log φ in θ

        if optimizer[0] == "efs":
            # mgcv gam.outer:1658-1668: optimizer "efs" (non-general) →
            # efsudr. mgcv's efsudr takes NO L/lsp0 (gam.outer never
            # passes them), so id-linked or partially-fixed smoothing
            # parameters are unsupported on this path.
            if G.L is not None or (G.lsp0 is not None
                                   and np.any(np.asarray(G.lsp0) != 0.0)):
                raise NotImplementedError(
                    "optimizer='efs' with id-linked or partially-fixed "
                    "smoothing parameters is not supported — mgcv's "
                    "efsudr has no L/lsp0 arguments (gam.outer, "
                    "mgcv.r:1665).")
            res_efs = outer_efsudr(
                cur_rho, cur_logphi if include_log_phi else None)
            fit = res_efs["fit"]
            theta_sp = np.asarray(res_efs["rho"], dtype=float)
            # log φ at which the final fit's REML was evaluated: the
            # extended fit's own scale (mgcv object$scale <- scale.est,
            # mgcv.r:1722) / the regular trial slot; None ⇔ scale known.
            log_phi_hat = res_efs["log_phi_reml"]
            sp_out = np.exp(theta_sp)
            rho_hat = rho_full(theta_sp)
            if include_family_theta and isinstance(family, _tw_family):
                tw_info = {
                    "theta_hat": float(family.get_theta()[0]),
                    "p_hat": float(family.p),
                    "log_phi_hat": log_phi_hat,
                }
            return {"fit": fit, "rho_hat": rho_hat, "sp": sp_out,
                    "log_phi_hat": log_phi_hat, "used_magic": False,
                    "used_nlm_optim": False, "tw_info": tw_info}

        theta0_parts = [cur_rho]
        if include_log_phi:
            theta0_parts.append(np.array([cur_logphi]))
        if include_family_theta:
            theta0_parts.append(np.asarray(family.get_theta(), dtype=float))
        theta0 = np.concatenate(theta0_parts)

        # Map the resolved method onto the outer-Newton criterion (mgcv's
        # scoreType, mgcv.r:1945-1959).
        if method in ("REML", "ML"):
            _criterion = "REML"
        elif method in ("P-REML", "P-ML"):
            _criterion = "PREML"
        elif method == "GACV.Cp" and not G.scale_known_fit:
            _criterion = "GACV"
        else:
            _criterion = "GCV"
        if use_magic(_criterion, include_log_phi, include_family_theta):
            # Gaussian-identity additive + GCV.Cp → mgcv's `magic` fast path
            # (am.fit, mgcv.r:2001). theta == per-penalty ρ here (L is None).
            # mgcv never reaches gam.outer on this path (outer.looping is
            # FALSE, mgcv.r:1933), so optimizer[1] is ignored like mgcv.
            used_magic = True
            theta_hat = gam_outer(
                theta0, optimizer="magic", criterion=_criterion,
                control=G.control, magic_fn=magic_optimize)
        elif optimizer[1] in ("nlm", "optim"):
            # gam.outer's "methods calling gam.fit3" branch
            # (mgcv.r:1692-1717): nlm on gam4objective / optim L-BFGS-B
            # on gam2objective+gam2derivative, then one final deriv-0
            # gam2objective fit.
            used_nlm_optim = True
            theta_hat = outer_nlm_optim(
                theta0, optimizer2=optimizer[1], criterion=_criterion,
                include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
                fscale=null_scale)
        else:
            theta_hat = gam_outer(
                theta0, optimizer="newton", criterion=_criterion,
                control=G.control, include_log_phi=include_log_phi,
                include_family_theta=include_family_theta,
                newton_fn=outer_newton)

        theta_sp = theta_hat[:n_work]
        base = n_work
        if include_log_phi:
            log_phi_hat = float(theta_hat[base])
            base += 1
        else:
            log_phi_hat = None
        if include_family_theta:
            family.set_theta(theta_hat[base:base + family.n_theta])
            if isinstance(family, _tw_family):
                tw_info = {
                    "theta_hat": float(theta_hat[base]),
                    "p_hat": float(family.p),
                    "log_phi_hat": log_phi_hat,
                }
        # ``m$sp`` is the *working* sp vector; ``m$full.sp`` is derived below
        # for every path via ``rho_full``.
        sp_out = np.exp(theta_sp)
        rho_hat = rho_full(theta_sp)
        # Build the final fit WITHOUT re-solving PIRLS at ρ̂: mgcv's gam.outer
        # reuses newton's accepted-step fit (object <- b$object, mgcv.r:1684).
        # magic builds from its cached QR; else use newton's cached fit; fall
        # back to a solve only if the optimizer produced none (init failure).
        if used_magic:
            fit = magic_fit_state(rho_hat)
        else:
            outer_fit = get_outer_fit()
            if outer_fit is not None:
                fit = outer_fit
            else:
                fit = fit_given_rho(rho_hat)
        # P-REML/P-ML carry no log φ in θ — the scale is the analytic
        # Pearson-Laplace plug-in at ρ̂ (φ = P/(n−Mp)).
        if G.pearson_scale_criterion:
            log_phi_hat = float(np.log(max(phi_pearson(fit), 1e-300)))
    return {"fit": fit, "rho_hat": rho_hat, "sp": sp_out,
            "log_phi_hat": log_phi_hat, "used_magic": used_magic,
            "used_nlm_optim": used_nlm_optim, "tw_info": tw_info}


def _cbind_response_intake(formula, data, family):
    """Two-column ``cbind(a, b) ~ ...`` response intake — shared by gam
    and bam. mgcv converts the matrix response inside each family's
    ``initialize`` (binomial: proportion + n-weights, gam.fit3.r:219;
    censored families: vector + ``attr(y,"censor")``, efam.r:1117 and
    twins; gfam: ``preinitialize`` index stash, gfam.r:384-392). hea
    rewrites up front instead, stashing both columns as frame columns so
    ``prepare_design``'s NA-omit keeps them row-aligned; the second
    column is re-read post-design by :func:`_cbind_family_stash`.

    Routing (parse the LHS and let the AST decide — a substring test
    would miss the ``[a, b]`` bracket alias, which lowers to a cbind
    Call):

    * cnorm/cpois/clog/bcg — censored ``cbind(y, yat)`` (col 0 the
      observed value, col 1 the censoring bound) → ``kind="censor"``.
    * gfam — ``cbind(y, index)`` (col 1 the 1-based family index,
      gfam.r:5-8) → ``kind="gfam"``; ``fi_expr`` carries the index
      expression for predict-on-newdata (mgcv re-reads the response from
      newdata, mgcv.r:2819).
    * (Quasi)Binomial — R's two-column counts response →
      ``kind="binom"`` (proportion/trials rewrite).
    * anything else — raise (mgcv dies obscurely here: "logical
      subscript too long").

    Returns ``(formula, data, kind, fi_expr)``; ``kind=None`` when the
    response is not a two-column cbind (formula/data returned unchanged
    — a non-cbind LHS is never touched).
    """
    _cbind = isinstance(formula, str)
    if _cbind:
        lhs = parse(formula).lhs
        _cbind = isinstance(lhs, Call) and lhs.fn == "cbind"
    if not _cbind:
        return formula, data, None, None
    if len(lhs.args) != 2 or lhs.kwargs:
        raise ValueError(
            "cbind() response must have exactly two columns: "
            "cbind(successes, failures)"
        )
    if isinstance(family,
                  (_cnorm_family, _cpois_family, _clog_family,
                   _bcg_family)):
        data = normalize_data(data)
        cols = set(data.columns)
        yobs, yat = (
            data.select(_eval_lhs_expr(a, cols).alias("_v"))["_v"]
            .to_numpy().astype(float)
            for a in lhs.args
        )
        # The observed column becomes the response; the censoring bound
        # is re-read after prepare_design and handed to the family.
        data = data.with_columns(
            pl.Series("_hea_cnorm_y", yobs),
            pl.Series("_hea_cnorm_yat", yat),
        )
        formula = "_hea_cnorm_y ~" + formula.split("~", 1)[1]
        return formula, data, "censor", None
    if isinstance(family, _gfam_family):
        data = normalize_data(data)
        cols = set(data.columns)
        yobs, yfi = (
            data.select(_eval_lhs_expr(a, cols).alias("_v"))["_v"]
            .to_numpy().astype(float)
            for a in lhs.args
        )
        data = data.with_columns(
            pl.Series("_hea_gfam_y", yobs),
            pl.Series("_hea_gfam_fi", yfi),
        )
        formula = "_hea_gfam_y ~" + formula.split("~", 1)[1]
        return formula, data, "gfam", lhs.args[1]
    if not isinstance(family, (Binomial, QuasiBinomial)):
        raise ValueError(
            "cbind(successes, failures) ~ ... requires "
            "family=Binomial() or QuasiBinomial(); got "
            f"{family.name!r}"
        )
    data = normalize_data(data)
    cols = set(data.columns)
    succ, fail = (
        data.select(_eval_lhs_expr(a, cols).alias("_v"))["_v"]
        .to_numpy().astype(float)
        for a in lhs.args
    )
    if np.any(succ < 0) or np.any(fail < 0):
        raise ValueError("negative counts in cbind() response")
    if (np.any(np.abs(succ - np.rint(succ)) > 0.001)
            or np.any(np.abs(fail - np.rint(fail)) > 0.001)):
        import warnings
        warnings.warn("non-integer counts in a binomial glm!",
                      stacklevel=3)
    tot = succ + fail
    pos = tot > 0
    prop = np.where(pos, succ / np.where(pos, tot, 1.0), 0.0)
    # NaN counts → NaN proportion so the standard NA-omit drops the row
    # (R's model.frame does the same before initialize).
    prop = np.where(np.isnan(tot), np.nan, prop)
    data = data.with_columns(
        pl.Series("_hea_cbind_p", prop),
        pl.Series("_hea_cbind_n", tot),
    )
    formula = "_hea_cbind_p ~" + formula.split("~", 1)[1]
    return formula, data, "binom", None


def _cbind_family_stash(kind, d, family):
    """Post-``prepare_design`` half of the cbind intake: re-read the
    (NA-aligned) stash columns from the design frame and hand them to
    the family — the censoring bound rides ``family.set_censor`` (the
    ``attr(y,"censor")`` of efam.r), the gfam index rides
    ``family.set_fi`` (the ``attr(fl,"fi")`` stash of gfam.r:384-392).
    Returns the binomial trials vector for ``kind="binom"`` (caller
    folds it into the prior weights: weights ← weights·n, R binomial
    initialize), else None."""
    if kind == "censor":
        family.set_censor(
            d.data["_hea_cnorm_yat"].to_numpy().astype(float))
    elif kind == "gfam":
        family.set_fi(
            d.data["_hea_gfam_fi"].to_numpy().astype(float))
    elif kind == "binom":
        return d.data["_hea_cbind_n"].to_numpy().astype(float)
    return None


class gam:
    """Generalized additive model — mgcv's ``gam()``.

    Parameters
    ----------
    formula : str
        mgcv-style formula, e.g. ``"y ~ x1 + s(x2) + s(x3, bs='cr') +
        te(u, v) + offset(log(e))"``.
    data : polars.DataFrame
        Data table; rows with NA in any referenced column are dropped
        before fitting.
    method : str, default "GCV.Cp"
        Smoothing-parameter selection criterion — one of ``"REML"``,
        ``"ML"``, ``"GCV.Cp"``, ``"GACV.Cp"``, ``"P-REML"``, ``"P-ML"``
        (mgcv's default is ``"GCV.Cp"`` too; prefer ``"REML"`` for most
        work). ``"ML"`` is Laplace marginal likelihood — like REML but
        does not profile out the unpenalized fixed effects; useful for
        ``anova(m1, m2)``-style likelihood-ratio comparisons across
        different fixed-effect structures, where REML scores aren't
        comparable. ``"GACV.Cp"`` is the generalized-ACV sibling of GCV
        (a Pearson-weighted denominator). ``"P-REML"``/``"P-ML"`` are
        the Pearson-Laplace variants — φ estimated from the Pearson
        statistic rather than the deviance — and coincide with
        ``"REML"``/``"ML"`` when the scale is known. mgcv's ``"NCV"``/
        ``"QNCV"`` (neighbourhood cross-validation) are not yet ported
        and raise ``NotImplementedError``.
    optimizer : str or (str, str), default ("outer", "newton")
        mgcv's ``gam(optimizer=)``. ``"efs"`` selects the extended
        Fellner-Schall loop — ``efsudr`` (gam.fit4.r:822) on the
        single-formula path, ``efsud`` for general (formula-list)
        families — coercing ``method`` to REML like mgcv (mgcv.r:1914);
        it is also the automatic choice when
        ``family.available_derivs == 0``. Like mgcv's, efs ignores
        ``gamma`` and computes no smoothing-parameter-uncertainty
        pieces (``Vc``/``edf2``/vcomp CIs). The second element picks
        the outer method: ``"newton"`` (mgcv's default analytic
        Newton), or the derivative-driven quasi-Newton alternatives
        ``"nlm"`` (Dennis-Schnabel UNCMIN on the score + exact
        gradient, mgcv.r:1697) and ``"optim"`` (L-BFGS-B,
        mgcv.r:1706) — both run R's own optimizer algorithms
        (:mod:`hea.R.uncmin` / :mod:`hea.R.lbfgsb`) and, like mgcv,
        finish on a derivative-free fit, so they also report no
        ``Vc``/``edf2``/vcomp CIs. ``"bfgs"`` is available for
        general families only (standard-family bfgs: roadmap C9).
    sp : None or array-like, optional
        Supplied smoothing parameters. Non-negative entries are fixed
        at that value; **negative entries are estimated** (mgcv's
        convention), so ``sp=[2, -1]`` fixes the first and optimizes
        the second. Length is the number of *working* smoothing
        parameters — one per penalty slot, except that smooths sharing
        an ``id=`` contribute a single shared parameter (mgcv's
        ``m$sp``). Per-smooth values can be given in the formula
        instead — ``s(x, sp=2)``, ``te(x, z, sp=c(1, -1))`` — and
        override the gam-level vector for that term (mgcv.r:1417-1440).
    family : hea.family.Family, optional
        Response family (Gaussian, Gamma, Poisson, Binomial,
        InverseGaussian, Tweedie, tw, …). Default Gaussian-identity.
    offset : array-like, optional
        Added to the linear predictor (combined with any formula
        ``offset(...)`` terms).
    weights : array-like, optional
        mgcv's prior weights ``w_i``: per-observation multipliers on the
        log-likelihood contribution (frequency/precision weights). For
        ``Binomial()`` this is the trials vector ``m_i`` with ``y`` the
        success *proportion* (R's proportion + ``weights=`` idiom), or
        use the ``cbind(succ, fail) ~ ...`` response form directly —
        like R, it multiplies any ``weights=`` by the per-row trials.
        Zero-weight rows are excluded from fitting (mgcv's ``good`` mask)
        but still get fitted values. Default: ones.
    gamma : float, default 1.0
        mgcv's smoothing-strength multiplier (Wood §4.6 suggests 1.4 for
        extra over-fitting protection).
    select : bool, default False
        Mirror of mgcv's ``select=TRUE``. When ``True``, an extra penalty
        is added to each smooth term over its null-space directions, so
        the smoothing-parameter selection can shrink any term entirely
        to zero — i.e., perform model selection alongside smoothness
        estimation. Each smooth gains one additional smoothing parameter.
    knots : dict, optional
        Per-covariate knot overrides (mgcv's ``knots=list(...)``).
    control : dict, optional
        mgcv's ``gam.control()`` — build with :func:`gam_control` or
        pass a plain dict with the same keys (``epsilon``, ``maxit``,
        ``newton={'conv_tol': ..., 'maxNstep': ..., 'maxSstep': ...,
        'maxHalf': ...}``, ``scale_est`` ("fletcher"/"pearson"/
        "deviance"), ``edge_correct``, ``efs_lspmax``, ``efs_tol``, …).
        ``edge_correct`` improves ``Vc`` near smoothing parameters at
        "working infinity" (REML/ML only; a number sets the target
        criterion increase per flat parameter, ``True`` means 0.02).
    scale : float, default 0.0
        mgcv's ``gam(scale=)``: 0 resolves by family (binomial/poisson
        known at 1, everything else estimated); ``> 0`` treats φ as
        KNOWN at that value (REML/ML drop the log φ slot, GCV.Cp
        switches to UBRE at that φ, summary uses z/Chi-sq statistics);
        ``< 0`` forces φ estimation (GCV even for poisson/binomial
        under GCV.Cp, with t/F statistics). Under (RE)ML, binomial/
        poisson are always scale=1 — a user value is silently
        overridden, as in mgcv (mgcv.r:1947).
    start, etastart, mustart : array-like, optional
        glm-style starting values for the inner PIRLS (length-p
        coefficients / length-n linear predictor / length-n means),
        precedence etastart > start > mustart (gam.fit3.r:259-272).
        Invalid values shrink toward the null model (20 tries, then
        R's "Can't find valid starting values" error). hea seeds every
        inner PIRLS restart with them; mgcv seeds the first inner fit
        and then carries coefficients forward between criterion
        evaluations — the converged fit is identical. For formula-list
        (general-family) fits only ``start`` applies.

    Attributes (always set)
    -----------------------
    n, p : int
        Sample size, total # of model coefficients (parametric + smooth).
    p_param : int
        Number of parametric coefficients.
    bhat, se_bhat : polars.DataFrame
        Coefficient estimates / Bayesian SEs (one row each, keyed by
        R-canonical coefficient names ``(Intercept)``, ``MachineB``,
        ``s(x).1``, ``s(x).2``, …).
    t_values, p_values : polars.DataFrame
        Per-coefficient Wald t-stat and p-value — only meaningful for
        *parametric* rows; smooth-basis rows are reported but users
        should interpret via the smooth-level table (``smooth_table``).
    linear_predictors : np.ndarray
        Length-n linear predictor ``η = Xβ̂``.
    fitted_values : np.ndarray
        Length-n fitted mean ``μ̂ = g⁻¹(η)``. For Gaussian-identity, μ = η.
    fitted : np.ndarray
        Alias for ``fitted_values`` (was ``η``; equivalent for Gaussian).
    residuals : np.ndarray
        Length-n response residuals ``y − μ̂``. Use ``residuals_of(type=…)``
        to request deviance/Pearson/working/response variants.
    sigma, sigma_squared : float
        Residual SD and variance (``scale`` in mgcv).
    sp : np.ndarray
        Optimized (or fixed) *working* smoothing parameters (mgcv's
        ``m$sp``): one per estimated parameter — fixed entries are
        folded out, id-linked penalties share one.
    full_sp : np.ndarray
        Per-penalty expansion ``exp(L·log(sp) + lsp0)`` (mgcv's
        ``m$full.sp``); equals ``sp`` when nothing is linked or fixed.
    edf : np.ndarray
        Per-coefficient effective degrees of freedom, diagonal of the
        influence matrix in coefficient space
        ``F = (XᵀX + Sλ)⁻¹ XᵀX``. Parametric entries are 1.
    edf_by_smooth : dict[str, float]
        Summed edf per smooth label (``"s(x)"``, ``"te(u,v)"``, …).
    edf_total : float
        ``sum(edf)`` — total model degrees of freedom (β + 1 for σ
        is *not* added; use ``npar`` for the MLE parameter count).
    Vp : np.ndarray
        Bayesian posterior covariance ``σ² (XᵀX + Sλ)⁻¹``. Matches
        mgcv's ``$Vp``.
    Ve : np.ndarray
        Frequentist covariance ``σ² (XᵀX + Sλ)⁻¹ XᵀX (XᵀX + Sλ)⁻¹``.
        Matches mgcv's ``$Ve``.
    r_squared, r_squared_adjusted : float
        As mgcv: 1 − rss/tss and the df-adjusted variant.
    deviance : float
        ``rss`` for Gaussian.
    loglike : float
        Unpenalized Gaussian log-likelihood at the fitted β̂.
    AIC, BIC : float
        ``-2·loglike + 2·npar`` (and ``log(n)·npar`` for BIC), where
        ``npar = edf_total + 1`` for the residual variance — matches R's
        ``AIC(gam_fit)``.
    npar : float
        ``edf_total + 1``. Not an integer because edf isn't.
    formula : str
    data : polars.DataFrame

    Attributes (method="REML" only)
    -------------------------------
    REML_criterion : float
        Optimized Laplace-approximate REML criterion, ``-2·V_R(ρ̂)``.

    Attributes (method="ML" only)
    -----------------------------
    ML_criterion : float
        Optimized Laplace-approximate ML criterion, ``-2·V_ML(ρ̂)``.
        Differs from ``REML_criterion`` by a ``Mp·log(2π·φ)`` constant
        — comparable across different fixed-effect structures.

    Attributes (method="GCV.Cp" only)
    ---------------------------------
    GCV_score : float
        Optimized GCV score, ``n · rss / (n − edf_total)²``.
    """

    # Estimated model rank (mgcv's ``oo$rank.est``), set by ``__init__``
    # via ``_estimate_rank``. Class-level ``None`` is the fallback for
    # subclasses that build their own state without running gam's
    # constructor (bam) — readers treat ``None`` as "assume full rank".
    rank: int | None = None
    # Working→per-penalty log-sp map (mgcv's ``L``; see ``_rho_full``).
    # ``None`` ⇔ identity (no id linkage) — also the bam fallback.
    _L: np.ndarray | None = None
    # Fixed-sp offset (mgcv's ``lsp0``): ρ_full = L·θ + lsp0. ``None`` ⇔
    # zeros — set only when sp= / s(..., sp=) fixes a strict subset of
    # the smoothing parameters (mgcv.r:1513-1538's fold).
    _lsp0: np.ndarray | None = None
    # gam.control dict — class default covers bam and any pre-__init__
    # reader; instances overwrite with the validated user control.
    _control: dict | None = None
    # Resolved gam(scale=) state (mgcv estimate.gam, mgcv.r:1936-1971):
    # > 0 ⇒ the scale is KNOWN/fixed at that value, < 0 ⇒ estimated.
    # ``None`` (class default, bam fallback) ⇒ resolve from the family
    # (binomial/poisson known at 1, everything else estimated) — the
    # pre-scale= behavior, byte-identical.
    _scale_resolved: float | None = None
    # glm-style PIRLS starting values (gam(start=/etastart=/mustart=));
    # class defaults cover bam and pre-__init__ readers.
    _pirls_start: np.ndarray | None = None
    _pirls_etastart: np.ndarray | None = None
    _pirls_mustart: np.ndarray | None = None
    # PIRLS warm start: the previous score-eval's converged linear predictor,
    # carried across outer-Newton steps so each penalized IRLS starts near its
    # solution (mgcv gam.fit3.r:1366-1368 sets etastart<-b$linear.predictors).
    # ρ-trajectory points are close, so this cuts ~12 PIRLS iters/eval to ~2-3;
    # result-preserving (the PIRLS solution at each ρ is unique).
    _pirls_warm_eta: np.ndarray | None = None
    # The outer optimizer's last accepted-step fit at the converged ρ̂ —
    # mgcv's gam.outer reuses newton's `object=b` as the final fit rather than
    # re-solving (mgcv.r:1684, gam.fit3.r:1718). Cached by `_outer_newton` so
    # the constructor never refits at the optimum. None ⇔ optimizer never ran
    # (magic path) or its initial fit failed.
    _outer_fit: "_FitState | None" = None
    # The n×q QR factor R of √w₀·X from the design-time structural rank drop,
    # reused by the magic path's getRpqr so √w·X is factored ONCE (magic.c:393).
    _struct_R: "np.ndarray | None" = None

    @property
    def _scale_known_fit(self) -> bool:
        """Fit-level "is φ known?" — replaces ``family.scale_known`` in
        every dispatch ``gam(scale=)`` can override."""
        if self._scale_resolved is None:
            return bool(self.family.scale_known)
        return self._scale_resolved > 0

    @property
    def _scale_fixed_value(self) -> float:
        """The known φ (only meaningful when ``_scale_known_fit``);
        1.0 on the family-default paths."""
        if self._scale_resolved is None or self._scale_resolved <= 0:
            return 1.0
        return float(self._scale_resolved)

    # ---- Phase-4 method= predicates (keyed on the resolved ``self.method``;
    # mgcv's scoreType after the scale-known reductions). Properties, not
    # init-time state, so ``bam`` (which sets ``self.method`` to REML/ML/
    # GCV.Cp after mapping fREML→REML) inherits them correctly.
    @property
    def _is_laplace(self) -> bool:
        """Outer Laplace-criterion path (REML/ML/P-REML/P-ML), as opposed to
        the GCV/UBRE/GACV performance-style path."""
        return self.method in ("REML", "ML", "P-REML", "P-ML")

    @property
    def _reml_ind(self) -> float:
        """mgcv's ``remlInd`` (gam.fit3.r:545): 1 for REML/P-REML (profile out
        the Mp fixed-effect prior), 0 for ML/P-ML."""
        return 1.0 if self.method in ("REML", "P-REML") else 0.0

    @property
    def _use_ml_proj(self) -> bool:
        """ML-style range-only log|H| projection (ML and P-ML)."""
        return self.method in ("ML", "P-ML")

    @property
    def _pearson_scale_criterion(self) -> bool:
        """Pearson-Laplace scale criteria (P-REML/P-ML): φ = Pearson/(n−Mp)
        is the analytic plug-in (ρ-only outer problem, γ≡1), not the
        profiled/​outer-variable scale of plain REML/ML. (Distinct from the
        instance attribute ``_pearson_scale``, which is the Pearson scale
        *value* reported by every fit.)"""
        return self.method in ("P-REML", "P-ML")

    # Class-level fallbacks so inherited methods stay usable from ``bam``
    # (same pattern as ``_L``): edge.correct off, no edge-corrected θ, no
    # family-θ slots in the augmented Hessian, no reparam basis (bam's
    # criterion machinery is its own; it keeps the legacy log|S|+ path).
    _edge_correct: bool | float = False
    _edge_theta1: np.ndarray | None = None
    _n_theta_aug: int = 0
    _UrS: list[np.ndarray] | None = None
    _reparam_cache: dict = {}
    # Rank-deficiency drop (mgcv pls_fit1's column dropping): None ⇔ full
    # rank. ``_keep_cols`` is the original-p boolean mask; ``_block_keep``
    # the per-block local masks (for predict-time bases, which rebuild
    # the *full* columns).
    _keep_cols: np.ndarray | None = None
    _block_keep: list[np.ndarray] | None = None
    # Range basis Y of the balanced total penalty (totalPenaltySpace) —
    # maps gam.reparam's range-space quantities back to the full
    # coefficient space (penalty root E_full = rp$E·Qs'·Y').
    _reparam_Y: np.ndarray | None = None

    @property
    def _work_dim(self) -> int:
        """Number of *working* (estimated) smoothing parameters —
        ``ncol(L)``; equals ``len(self._slots)`` when no id linkage."""
        return len(self._slots) if self._L is None else self._L.shape[1]

    # Binomial trials vector from a cbind(succ, fail) response — class
    # default so bam (which shares the criterion/aic machinery but not
    # this __init__) reads None until it grows its own cbind intake.
    _binom_n: np.ndarray | None = None

    @property
    def _family_mgcv_extended(self) -> bool:
        """mgcv's ``extended.family`` set — families whose θ enters the
        outer optimization (``tw`` in any θ mode, Scat). Distinct from
        hea's ``Family.is_extended``, which gates bam's bgam.fitd Newton
        branch (Scat only today). Gates mgcv's extended-family special
        cases: sig2 = exp(φ̂) reporting, Dd-based initial.spg weights,
        and the null-deviance offset-correction exclusion."""
        return isinstance(self.family, _tw_family) or self.family.is_extended

    def __init__(
        self,
        formula: str,
        data,
        *,
        method: str = "GCV.Cp",
        optimizer: str | tuple | list = ("outer", "newton"),
        sp: np.ndarray | None = None,
        family: Family | None = None,
        offset: np.ndarray | list | None = None,
        weights: np.ndarray | list | None = None,
        gamma: float = 1.0,
        select: bool = False,
        knots: dict | None = None,
        xt: dict | None = None,
        control: dict | None = None,
        scale: float = 0.0,
        start: np.ndarray | list | None = None,
        etastart: np.ndarray | list | None = None,
        mustart: np.ndarray | list | None = None,
    ):
        # ``data`` may be a polars DataFrame OR a mapping of name → 1-D /
        # 2-D ndarray. 2-D entries become matrix columns
        # (``Array(Float64, m)``) for mgcv's summation-convention smooths
        # (Wood §7.4.1). ``prepare_design`` calls ``normalize_data``
        # internally; we keep the parameter untyped for flexibility.
        if isinstance(family, type) and issubclass(family, Family):
            # R: gam(family=quasipoisson) passes the constructor itself;
            # mgcv calls it (`if (is.function(family)) family <- family()`,
            # mgcv.r:2324).
            family = family()
        # mgcv gam(optimizer=) intake: a 1- or 2-vector; first element
        # "outer"|"efs" (estimate.gam, mgcv.r:1913), second the outer
        # method defaulting to "newton" (gam.outer, mgcv.r:1643-1644).
        # hea validates both elements up front — mgcv's second-element
        # check sits in gam.outer and is skipped only by paths hea does
        # not have (the magic additive-GCV route). newton, efs, nlm and
        # optim are ported; standard-family bfgs raises honestly.
        opt = ((optimizer,) if isinstance(optimizer, str)
               else tuple(str(o) for o in optimizer))
        if not 1 <= len(opt) <= 2:
            raise ValueError("optimizer must have one or two elements")
        if opt[0] not in ("outer", "efs"):
            raise ValueError("unknown optimizer")
        opt = (opt[0], opt[1] if len(opt) == 2 else "newton")
        if opt[1] not in ("newton", "bfgs", "nlm", "optim"):
            raise ValueError("unknown outer optimization method.")
        if opt[0] == "outer" and opt[1] == "bfgs":
            # bfgs is ported for GENERAL families (item 7 — the gam.fit5
            # outer loop, mgcv's bfgs gam.fit3.r:1722); the standard
            # gam.fit3 bfgs remains unported (roadmap C9).
            _gen = getattr(family, "is_general", False)
            if not _gen:
                raise NotImplementedError(
                    "optimizer=('outer', 'bfgs') is ported for general "
                    "families only; the standard-family gam.fit3 bfgs "
                    "(gam.fit3.r:1722) is not ported (roadmap C9). Use "
                    "'newton', 'nlm', 'optim' or 'efs'.")
        self.optimizer = opt
        if isinstance(formula, (list, tuple)):
            # Multiple linear predictors → general-family fitting via
            # gam.fit5 (estimate.gam's general branch, mgcv.r:1894-1903).
            if family is None or not getattr(family, "is_general", False):
                raise NotImplementedError(
                    "gam with a formula list (multiple linear "
                    "predictors) requires a general family — e.g. "
                    "family=gaulss()."
                )
            if etastart is not None or mustart is not None:
                raise NotImplementedError(
                    "etastart/mustart apply to the PIRLS fitters only; "
                    "general-family (formula list) fits take start=."
                )
            self._init_general(
                [str(f) for f in formula], data, method=method, sp=sp,
                family=family, offset=offset, weights=weights,
                gamma=gamma, select=select, knots=knots,
                control=control, start=start, optimizer=opt,
            )
            return
        if getattr(family, "is_general", False):
            if family.n_lp == 1:
                # single-formula general entry (cox.ph is nlp=1 with no
                # formula list, coxph.r:349-357): route the bare formula
                # through the general path as a 1-element list.
                if etastart is not None or mustart is not None:
                    raise NotImplementedError(
                        "etastart/mustart apply to the PIRLS fitters "
                        "only; general-family fits take start=.")
                self._init_general(
                    [str(formula)], data, method=method, sp=sp,
                    family=family, offset=offset, weights=weights,
                    gamma=gamma, select=select, knots=knots,
                    control=control, start=start, optimizer=opt,
                )
                return
            raise ValueError(
                f"family {family!r} has {family.n_lp} linear predictors"
                " — pass a list of formulas, one per linear predictor."
            )
        if opt[0] == "efs":
            # Single-formula efs → efsudr (gam.fit4.r:822); mgcv coerces
            # the method to REML first (mgcv.r:1914).
            method = "REML"
        if method in ("NCV", "QNCV"):
            raise NotImplementedError(
                "method='NCV'/'QNCV' (neighbourhood cross validation, "
                "mgcv gam.fit3.r:667 + ncv.c) is not yet ported. Use 'REML', "
                "'P-REML', 'ML', 'P-ML', 'GCV.Cp' or 'GACV.Cp'."
            )
        if method not in ("REML", "ML", "GCV.Cp", "GACV.Cp", "P-REML", "P-ML"):
            raise ValueError(
                "method must be one of 'REML', 'ML', 'GCV.Cp', 'GACV.Cp', "
                f"'P-REML', 'P-ML', got {method!r}"
            )
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")
        if knots is not None and not isinstance(knots, dict):
            raise TypeError(
                "knots must be a dict mapping covariate name -> knot sequence "
                "(mgcv's knots=list(...)), or None"
            )
        # mgcv's gam.control umbrella — validated/defaulted through
        # ``gam_control`` whether the caller passed its output or a raw
        # dict. ``edge_correct`` lives here like mgcv's edge.correct
        # (gam.fit3.r:1670-1716; REML/ML only).
        self._control = gam_control(**(control or {}))
        self._edge_correct = self._control["edge_correct"]
        self._edge_theta1: np.ndarray | None = None

        self.formula = formula
        self.method = method
        # mgcv's per-covariate knot override; threaded into materialize_smooths
        # and consumed by the cr/cc/ps/cp/bs builders. None ⇒ data-adaptive
        # defaults (byte-identical to pre-knots behavior).
        self.knots = knots
        # mgcv's per-smooth ``xt`` extras, keyed by covariate name (the
        # object-arg channel, like ``knots=``). Currently consumed by the mrf
        # builder (penalty / nb / polys). None ⇒ no extras.
        self.xt = xt
        self._select = bool(select)
        # mgcv's smoothing-strength multiplier. ``gamma > 1`` produces
        # smoother fits by inflating the apparent edf cost in the GCV/UBRE
        # criterion, or by dividing the data-fit term in REML. Wood §4.6
        # recommends ``gamma=1.4`` as a reasonable default for over-fitting
        # protection. Stored on self and threaded into the criterion
        # functions (_reml, _gcv, ...) and their gradients/hessians.
        self._gamma = float(gamma)
        # mgcv accepts the family constructor as well as the constructed
        # object — ``if (is.function(family)) family <- family()``
        # (mgcv.r:2324) — so ``family=gaulss`` works like
        # ``family=gaulss()``. Family *instances* are never re-called.
        if not isinstance(family, Family) and callable(family):
            family = family()
        self.family = Gaussian() if family is None else family
        # mgcv coerces extended families onto (RE)ML for any criterion other
        # than REML/ML/NCV — gam.fit4 has no GCV/UBRE/GACV/Pearson-Laplace
        # path (mgcv.r:1892; silent there, so silent here). So GCV.Cp,
        # GACV.Cp, P-REML and P-ML all collapse to REML for tw/scat/nb/...
        if self._family_mgcv_extended and method not in ("REML", "ML"):
            method = "REML"
            self.method = method

        # gam(scale=) resolution (estimate.gam, mgcv.r:1936-1971):
        #   scale = 0  → family default (binomial/poisson known at 1,
        #                everything else estimated) — the historical path;
        #   scale > 0  → φ KNOWN at that value: REML/ML drop the log φ
        #                slot, GCV.Cp switches to UBRE at φ=scale;
        #   scale < 0  → force φ estimation (GCV even for poisson/
        #                binomial under GCV.Cp).
        # Under (RE)ML, binomial/poisson are ALWAYS scale=1 — a user
        # scale= is silently overridden (mgcv.r:1947, same silence).
        if not (np.isscalar(scale) and np.isfinite(scale)):
            raise ValueError(f"scale must be a finite number, got {scale!r}")
        scale = float(scale)
        if self._family_mgcv_extended:
            # mgcv.r:1948-1949: extended families resolve scale from the
            # family when scale≤0 (φ=1 for the tw/nb/scat deviance); a user
            # scale>0 fixes φ at that value (verified: nb scale=2 → φ=2).
            if scale > 0.0:
                self._scale_resolved = scale
            else:
                self._scale_resolved = 1.0 if self.family.scale_known else -1.0
        elif method in ("REML", "ML", "P-REML", "P-ML"):
            if self.family.scale_known:
                self._scale_resolved = 1.0          # mgcv.r:1947
            else:
                self._scale_resolved = scale if scale > 0 else -1.0
            # Known scale collapses the Pearson-Laplace criteria onto their
            # plain Laplace siblings (mgcv.r:1968-1970): there is no φ to
            # profile, so P-REML ≡ REML and P-ML ≡ ML.
            if self._scale_resolved > 0:
                if method == "P-REML":
                    method = "REML"
                elif method == "P-ML":
                    method = "ML"
                self.method = method
        else:  # GCV.Cp / GACV.Cp
            if scale == 0.0:
                self._scale_resolved = (1.0 if self.family.scale_known
                                        else -1.0)
            else:
                self._scale_resolved = scale  # >0 → UBRE at φ; <0 → GCV/GACV
        if isinstance(self.family, _negbin_family):
            # Fixed-θ negbin: "scale <- 1; ## no choice" — estimate.gam
            # overrides whatever scale= said AFTER the branches above
            # (mgcv.r:1963-1966, re-set with G$sig2 at 1975-1979), and
            # GCV.Cp/GACV.Cp become UBRE — delivered by the scale-known
            # criterion dispatch below. (Verified live: scale=-1 and
            # scale=5 fits are identical to the default.)
            self._scale_resolved = 1.0
        # mgcv's object$scale.estimated.
        self.scale_estimated = not self._scale_known_fit
        # GCV.Cp dispatches by family.scale_known: scale-unknown (Gaussian,
        # Gamma, IG) → GCV `n·D/(n−τ)²`; scale-known (Poisson, Binomial) →
        # UBRE `D/n + 2·τ/n − 1`. mgcv's `gam.outer` does the same dispatch
        # under method="GCV.Cp".

        # cbind(a, b) ~ ... two-column response intake (binomial counts /
        # censored bound / gfam index) — shared with bam, see
        # :func:`_cbind_response_intake`. ``self.formula`` keeps the
        # original text; the trials vector for the binomial rewrite is
        # picked up below once ``self._wt`` exists.
        formula, data, _cbind_kind, _gfam_expr = _cbind_response_intake(
            formula, data, self.family)
        if _gfam_expr is not None:
            # Keep the index expression: predict evaluates it on newdata
            # to recover the family index (mgcv reads the response from
            # newdata, mgcv.r:2819/3174).
            self._gfam_fi_expr = _gfam_expr
        d = prepare_design(formula, data)
        self._expanded = d.expanded
        # Materialise smooth-arg expressions once into ``self.data`` so the
        # synth columns (``s(I(b.depth^.5))`` ⇒ ``"I(b.depth^0.5)"``) are
        # visible to every downstream consumer — plot_smooth's rug, partial
        # residuals, summary, residuals_of, etc. — without each having to
        # re-evaluate the expression. ``materialize_smooths`` will idempotently
        # see the columns already present and skip the work.
        from ..formula import _apply_smooth_arg_exprs, _smooth_arg_expr_map
        _expr_map = _smooth_arg_expr_map(self._expanded)
        self.data = _apply_smooth_arg_exprs(d.data, _expr_map) if _expr_map else d.data
        X_param_df = d.X
        # R's binomial initialize accepts a 2-level factor / boolean
        # response (level 1 = failure); same coercion as glm's intake.
        y = _coerce_response(d.y, self.family)
        # Censored bound → family.set_censor / gfam index → family.set_fi
        # (before the family's preinitialize runs); binomial trials held
        # for the weights fold below.
        _binom_trials = _cbind_family_stash(_cbind_kind, d, self.family)
        X_param = X_param_df.to_numpy().astype(float)
        if X_param.shape[1] == 0:
            # 0-column polars frame → to_numpy() collapses to (0, 0); keep
            # the row count (pure-smooth no-intercept models, `y ~ s(x)-1`).
            X_param = np.zeros((y.shape[0], 0))
        n, p_param = X_param.shape

        # Sum any ``offset(...)`` atoms from the formula plus the kwarg
        # offset. mgcv's gam adds these to η just like glm does:
        # η = X·β + offset for both fitting and prediction.
        off = (np.zeros(n) if offset is None
               else np.asarray(offset, dtype=float).flatten())
        if off.shape != (n,):
            raise ValueError(f"offset must have length {n}, got {off.shape}")
        for off_node in d.expanded.offsets:
            blk = _eval_atom(off_node, d.data)
            off = off + blk.values.flatten().astype(float)
        self._offset = off

        # mgcv's prior weights (gam(weights=)). One canonical array,
        # ``self._wt``, set before any fitting so the PIRLS loop, the
        # REML/ML criterion (family.ls), initial.spg, and every post-fit
        # consumer (Pearson scale, residuals, AIC, null deviance) read the
        # same values. glm semantics: zero allowed (row excluded from the
        # working model via the `good` mask, still predicted), negative
        # rejected.
        if weights is None:
            self._wt = np.ones(n)
        else:
            wt_prior = np.asarray(weights, dtype=float).flatten()
            if wt_prior.shape != (n,):
                raise ValueError(
                    f"weights must have length {n}, got {wt_prior.shape}"
                )
            if not np.all(np.isfinite(wt_prior)):
                raise ValueError("missing or non-finite values in weights")
            if np.any(wt_prior < 0):
                raise ValueError("negative weights not allowed")
            self._wt = wt_prior
        if _binom_trials is not None:
            # weights ← weights·n (R binomial initialize). Trials re-read
            # from the design frame, which prepare_design may have
            # NA-filtered — keeps rows aligned. A zero-trials row gets
            # weight 0: excluded from the fit via the `good` mask but
            # still predicted, like R.
            self._binom_n = _binom_trials
            self._wt = self._wt * self._binom_n
        self.prior_weights = self._wt

        # mgcv's extended-family preinitialize hook (mgcv.r:1983-1995):
        # one-shot data-dependent θ start (scat: ν, σ from sd(y)) and/or
        # a response transform, run once before any fitting machinery.
        pre = self.family.preinitialize(y)
        if pre:
            if pre.get("Theta") is not None:
                self.family.set_theta(np.asarray(pre["Theta"], dtype=float))
            if pre.get("y") is not None:
                y = np.asarray(pre["y"], dtype=float).reshape(-1)

        sb_lists = (
            materialize_smooths(d.expanded, d.data, knots=knots, xt=self.xt)
            if d.expanded.smooths else []
        )
        blocks: list[SmoothBlock] = [b for group in sb_lists for b in group]
        # Per-block ``id`` (mgcv's sp-linkage key), parallel to ``blocks``.
        # Every block born from the same smooth call inherits that call's
        # id (a by=factor smooth's level blocks all share it — that's the
        # ``s(x, by=fac, id=1)`` single-λ idiom). Both block-list
        # transforms below are length/order-preserving, so this stays
        # aligned.
        from ..formula import _smooth_id_value, _smooth_sp_value
        block_ids: list[str | None] = []
        # Per-block ``sp=`` from the smooth spec (mgcv: each by=factor
        # level block inherits the spec's sp), parallel to ``blocks``
        # under the same length/order-preserving guarantee.
        block_sps: list[tuple[float, ...] | None] = []
        for call_node, group_blocks in zip(d.expanded.smooths, sb_lists):
            block_ids.extend([_smooth_id_value(call_node)] * len(group_blocks))
            block_sps.extend([_smooth_sp_value(call_node)] * len(group_blocks))
        # mgcv: select=TRUE adds a null-space penalty per smooth inside
        # smoothCon — i.e., before gam.side. Mirror that order so the
        # subsequent column drops (gam.side) restrict Sf to the kept-cols
        # subspace exactly the way mgcv does.
        if self._select:
            blocks = _add_null_space_penalties(blocks)
        # mgcv's gam.side: when one smooth's variable set is a strict subset
        # of another's (e.g. `s(x1) + te(x1, x2)`), the wider smooth's basis
        # contains a copy of the narrower's main effect, which makes the
        # combined design rank-deficient and the REML/GCV optimum drift away
        # from mgcv's. Apply orthogonality constraints (column-rotate the
        # wider smooth so its columns are orthogonal in the data space to
        # the narrower's). This typically drops one column per overlap, so
        # `te(x1, x2)` next to `s(x1) + s(x2)` shrinks 24 → 22 cols, matching
        # mgcv's `model.matrix` exactly.
        blocks = _apply_gam_side(blocks)

        # Build full design X = [X_param | X_block_1 | X_block_2 | …] and the
        # parallel list of penalty "slots" (one per (block, S_j) pair). Each
        # slot carries its column range in the full design so we can embed the
        # k×k penalty in the p×p full-design template without allocating a
        # zero-padded copy per evaluation.
        Xs = [X_param]
        slots: list[_PenaltySlot] = []
        block_col_ranges: list[tuple[int, int]] = []
        col_cursor = p_param
        for b in blocks:
            Xb = np.asarray(b.X, dtype=float)
            Xs.append(Xb)
            k = Xb.shape[1]
            a, bcol = col_cursor, col_cursor + k
            block_col_ranges.append((a, bcol))
            for j, S_j in enumerate(b.S):
                slots.append(_PenaltySlot(block=b, col_start=a, col_end=bcol,
                                          S=np.asarray(S_j, dtype=float),
                                          S_scale=_block_s_scale(b, j)))
            col_cursor = bcol
        X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else X_param
        p = X.shape[1]

        # ------------- L matrix: working → per-penalty log-sp ---------------
        # mgcv's gam.setup (mgcv.r:1280-1320): ``ρ_full = L·θ`` maps the
        # *working* (estimated) log smoothing parameters θ to the log-sp
        # multiplying each S_k. A block whose ``id`` was seen before reuses
        # the first such block's working columns (its j-th penalty shares
        # the j-th column); everything else extends L block-diagonally
        # with an identity. ``self._L is None`` ⇔ no linkage — the mapping
        # is the identity and every code path below stays byte-identical
        # to the pre-L behavior. (mgcv's lsp0 offset enters just below,
        # when sp= / s(..., sp=) fixes a subset of the parameters.)
        slot_work_col: list[int] = []
        n_work = 0
        id_first_cols: dict[str, tuple[int, int]] = {}
        slot_cursor = 0
        # Per-block working-column range + whether this block *defines*
        # its id group (mgcv's idx[[id]]$sp.done: only the defining
        # term's sp= is consumed, mgcv.r:1430-1438).
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
            slot_cursor += nS
        if n_work == len(slots):
            self._L = None                       # identity — no id linkage
        else:
            L = np.zeros((len(slots), n_work))
            L[np.arange(len(slots)), slot_work_col] = 1.0
            self._L = L
        self._n_work = n_work

        # ------------- sp=: gam-level + per-smooth merge, fixed fold -------
        # mgcv gam.setup (mgcv.r:1400-1459): the working sp vector starts
        # from gam(sp=) — or all -1 ("estimate") — then any s(..., sp=) /
        # te(..., sp=) values overwrite their term's working entries
        # (id groups: defining term only). Entries >= 0 are then folded
        # out of the optimization (mgcv.r:1513-1538):
        #   lsp0 = L[, fixed] @ log(sp_fixed);  L <- L[, free]
        # and the remaining negative entries stay estimated. All-fixed
        # input keeps the historical fixed-sp path; all-negative input is
        # mgcv's "estimate everything" (≡ sp=None).
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
            for (wstart, nS, defining), bsp in zip(block_work_info,
                                                   block_sps):
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
                    # (mgcv.r:1519-1527), ported bug-for-bug: the i-th
                    # zero reads the i-th *penalty's* X-block and S
                    # (G$off[i]/G$S[[i]] with i the literal loop
                    # counter), not the zero entry's own penalty.
                    eps = np.finfo(float).eps
                    for i, dst in enumerate(np.flatnonzero(zero)):
                        sl = slots[i]
                        Xblk = X[:, sl.col_start:sl.col_end]
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
                sp = None       # nothing fixed — free optimization

        # Column names: parametric (R-canonical) + "s(x).1", "s(x).2", … per
        # block. Matches mgcv's `coef(gam_fit)` labels. For multi-block
        # smooths (by = factor), the block label already includes the level
        # suffix (see formula._apply_by_and_absorb).
        column_names = list(X_param_df.columns)
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            for i in range(1, bcol - a + 1):
                column_names.append(f"{b.label}.{i}")
        assert len(column_names) == p

        # ------------- identifiability: rank detection + column drop -------
        # mgcv's pls_fit1 detects rank deficiency by pivoted QR of the
        # augmented penalized problem and *drops* the unidentifiable
        # columns, zero-filling their coefficients on output
        # (gdi.c:1740-1775; gam reports rank < p, coef 0, SE 0, t NaN).
        # hea detects once, structurally, at the family's initial Fisher
        # weights — exact collinearity is weight-independent, and mgcv's
        # own comments note per-iteration drop sets aren't guaranteed
        # stable anyway (gam.fit3.r:449-450). The fit then runs on the
        # reduced design; reporting re-inflates (see attribute assembly).
        self._p_orig = p
        self._column_names_orig = list(column_names)
        # mgcv evaluates the criterion on the *original* problem — G$Mp
        # and gam.reparam's UrS are built at setup, before pls_fit1 ever
        # drops a column (the drop constrains β̂ to the identifiable
        # subspace; the REML prior-rank term and log|Sλ|+ keep the full
        # dimensions). Capture both before any reduction.
        Mp = p_param
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            k = bcol - a
            if not b.S:
                Mp += k
                continue
            S_sum = np.sum([np.asarray(s, dtype=float) for s in b.S], axis=0)
            rank_b = _sym_rank(S_sum)
            Mp += k - rank_b
        self._Mp = Mp
        self._penalty_rank = p - Mp
        slots_full = list(slots)
        # gam(start=/etastart=/mustart=) — glm-style PIRLS starting
        # values (gam.fit3.r:259-292). hea seeds every inner PIRLS with
        # them (mgcv seeds the first inner fit, then carries previous
        # coefficients forward between criterion evaluations,
        # gam.fit3.r:1366-1368 — hea restarts each evaluation instead;
        # the converged fit is identical, each inner problem is convex).
        # Stored before the rank-drop so the initial.spg weights see
        # them; ``start`` is masked to the kept columns just after.
        if start is not None:
            st = np.asarray(start, dtype=float).flatten()
            if st.shape != (p,):
                # mgcv's message shape (gam.fit3.r:264).
                raise ValueError(
                    f"Length of start should equal {p} and correspond "
                    "to initial coefs."
                )
            self._pirls_start = st
        if etastart is not None:
            es = np.asarray(etastart, dtype=float).flatten()
            if es.shape != (n,):
                raise ValueError(f"etastart must have length {n}")
            self._pirls_etastart = es
        if mustart is not None:
            ms = np.asarray(mustart, dtype=float).flatten()
            if ms.shape != (n,):
                raise ValueError(f"mustart must have length {n}")
            self._pirls_mustart = ms
        w0 = self._init_fisher_w(y, X=X)
        Xw0 = X * np.sqrt(np.maximum(w0, 0.0))[:, None]
        rank0, drop0, R0_struct = _pls_rank_drop(Xw0, slots, p)
        # Cache the n×q QR factor of √w₀·X for the magic path to reuse as its
        # getRpqr R (no second n×q QR). Valid only with NO column drop — after a
        # drop the magic design is the reduced X, factored fresh. Under
        # gaussian-identity (the magic prerequisite) w₀ ≡ wt exactly, so this R
        # equals magic_setup's qr(√wt·X).R bit-for-bit. (mgcv's magic does ONE
        # n×q QR, magic.c:393; hea was doing two — structural drop + getRpqr.)
        self._struct_R = R0_struct
        if drop0.size:
            import warnings as _w
            dropped_names = [column_names[j] for j in drop0]
            _w.warn(
                f"model is rank deficient: rank {rank0} < {p} coefficients; "
                f"dropping {dropped_names} (mgcv pls_fit1 semantics — "
                "dropped coefficients are reported as 0 with SE 0)",
                stacklevel=2,
            )
            keep_mask = np.ones(p, dtype=bool)
            keep_mask[drop0] = False
            self._keep_cols = keep_mask
            X = X[:, keep_mask]
            self._block_keep = []
            new_ranges: list[tuple[int, int]] = []
            for b, (a, bcol) in zip(blocks, block_col_ranges):
                bmask = keep_mask[a:bcol]
                self._block_keep.append(bmask)
                if not bmask.all():
                    b.X = np.asarray(b.X, dtype=float)[:, bmask]
                    b.S = [np.asarray(S_j, dtype=float)[np.ix_(bmask, bmask)]
                           for S_j in b.S]
                na = int(np.sum(keep_mask[:a]))
                new_ranges.append((na, na + int(bmask.sum())))
            block_col_ranges = new_ranges
            slots = []
            for b, (a, bcol) in zip(blocks, block_col_ranges):
                for j, S_j in enumerate(b.S):
                    slots.append(_PenaltySlot(
                        block=b, col_start=a, col_end=bcol,
                        S=np.asarray(S_j, dtype=float),
                        S_scale=_block_s_scale(b, j),
                    ))
            p_param = int(np.sum(keep_mask[:p_param]))
            column_names = [nm for nm, k in zip(column_names, keep_mask) if k]
            p = X.shape[1]
            if self._pirls_start is not None:
                # User start corresponds to the original columns; the
                # fit runs reduced (mgcv drops inside pls_fit1 instead).
                self._pirls_start = self._pirls_start[keep_mask]

        # ------------- sufficient statistics -------------------------------
        XtX = X.T @ X
        Xty = X.T @ y
        yty = float(y @ y)
        y_mean = float(y.mean())
        has_intercept = "(Intercept)" in X_param_df.columns
        tss = float(np.sum((y - y_mean) ** 2)) if has_intercept else yty

        self.X = X_param_df              # parametric design (user-facing)
        self._X_full = X                 # penalized full design
        self.y = d.y
        self._y_arr = y
        self.n = n
        self.p = p
        self.p_param = p_param
        self._blocks = blocks
        self._slots = slots
        self._block_col_ranges = block_col_ranges
        self.column_names = column_names
        self._XtX = XtX
        self._Xty = Xty
        self._yty = yty
        self._has_intercept = has_intercept
        self._tss = tss
        self.parametric_columns = list(X_param_df.columns)
        # R model.matrix's ``assign`` for the parametric block (0 =
        # intercept, i = expanded.terms[i-1]): the exact term→column map
        # the pTerms joint Wald tests group by.
        self._param_assign = list(d.param_assign or [])

        # Mp / _penalty_rank were computed before the identifiability drop
        # (mgcv's G$Mp is a setup-time quantity — see the drop block).
        # gam.reparam basis (totalPenaltySpace + mini.roots): log|Sλ|+ and
        # its ρ-derivatives are evaluated through get_stableS in this fixed
        # range basis — immune to λ-ratio "machine zero leakage" between
        # penalty components (Wood 2011; gam.fit3.r:9-62, gdi.c:550). Like
        # mgcv, the basis is built from the *pre-drop* penalties (UrS is
        # constructed before pls_fit1 drops anything).
        self._setup_reparam(slots_full, self._p_orig)

        # ------------- smoothing-param optimization ------------------------
        n_sp = len(slots)
        # Set by the optimizer branch below when log φ enters the outer
        # vector (PIRLS path, unknown-scale family). None means φ is
        # profiled (Gaussian-identity strict-additive) or fixed at 1
        # (scale-known families) — i.e., off the outer-vec.
        self._log_phi_hat: float | None = None
        # Set by `_outer_newton` when the optimizer runs. None for the
        # no-smooth and fixed-`sp` paths — `gam.check()` skips the
        # convergence block in those cases.
        self._outer_info: dict | None = None
        # True when the Gaussian-additive GCV `magic` fast path runs (mgcv's
        # am.fit). Its post-proc mirrors magic.post.proc (mgcv.r:4475): edf1
        # only, edf2 NULL → edf, no Vc — mgcv computes no sp-uncertainty
        # correction for the GCV/magic path.
        self._used_magic = False
        # mgcv get.null.coef (mgcv.r:1854) is ρ-independent and computed ONCE,
        # passed into every gam.fit3/gam.fit4 call; hea recomputed the lstsq
        # per `_fit_given_rho` score-eval. Cache the (null_coef, eta_null,
        # mu_null) baseline here, computed lazily on the first fit.
        self._null_baseline_cache: tuple | None = None
        # Cached LAPACK workspace sizes for the _pls_qr no-Q path (geqrf +
        # ormqr); shape-stable (driven by p, fixed per fit) → queried once.
        self._pls_lwork: dict = {"g": None, "o": None}
        # Set by the joint outer Newton in the tw() path (estimated-p
        # Tweedie); None otherwise. Holds θ̂, p̂, log φ̂.
        self._tw_info: dict | None = None
        # Free-θ extended families (tw, scat, nb) estimate θ jointly with
        # the smoothing parameters; an ALL-fixed ``sp`` skips the outer
        # optimizer and would silently freeze θ at the init value. Refuse
        # early so the user picks the fixed-θ constructor form instead
        # (Tweedie(p=...), scat(theta=(ν,σ)), ...). Mixed sp is fine —
        # by this point ``sp is not None`` ⇔ every entry fixed (the
        # merge/fold above rebound partially-fixed input to sp=None with
        # the fixed part in lsp0), and the outer Newton still runs θ.
        if self.family.n_theta > 0 and sp is not None:
            raise ValueError(
                f"{type(self.family).__name__} estimates its family "
                "parameters jointly with the smoothing parameters; passing "
                "a fixed `sp` is incompatible. Fix the family parameters "
                "in the constructor instead (e.g. Tweedie(p=...), "
                "scat(theta=(nu, sigma)))."
            )
        # mgcv's estimate.gam (mgcv.r:1872): smoothness selection + final
        # fit. Bundle the setup outputs into G and thread the fitter callables
        # (the C1 poly seam newton/bfgs already use), then assign the returned
        # state onto self and continue to the shared post-fit assembly below.
        G = _G(n=n, n_sp=n_sp, n_work=n_work, Mp=self._Mp, L=self._L,
               lsp0=self._lsp0, wt=self._wt, y_arr=self._y_arr,
               family=self.family, scale_known_fit=self._scale_known_fit,
               pearson_scale_criterion=self._pearson_scale_criterion,
               control=self._control)
        _res = estimate_gam(
            G, sp, method,
            fit_given_rho=self._fit_given_rho, rho_full=self._rho_full,
            initial_sp_rho=self._initial_sp_rho, use_magic=self._use_magic,
            phi_pearson=self._phi_pearson,
            profile_log_phi_fixed_sp=self._profile_log_phi_fixed_sp,
            magic_fit_state=self._magic_fit_state,
            outer_newton=self._outer_newton,
            magic_optimize=self._magic_optimize,
            get_outer_fit=lambda: self._outer_fit,
            optimizer=self.optimizer, outer_efsudr=self._outer_efsudr,
            outer_nlm_optim=self._outer_nlm_optim)
        fit = _res["fit"]
        rho_hat = _res["rho_hat"]
        self.sp = _res["sp"]
        self._log_phi_hat = _res["log_phi_hat"]
        self._used_magic = _res["used_magic"]
        self._tw_info = _res["tw_info"]
        # True iff the efsudr loop actually ran (fixed-sp / no-smooth efs
        # requests degenerate to plain fits before the optimizer). efs
        # fits are deriv-0: mgcv's gam.fit3.post.proc then has db.drho
        # NULL and computes NO sp-uncertainty pieces (edf2/Vc,
        # gam.fit3.r:978) and gam.vcomp has no outer.info$hess — the
        # post-fit blocks below mirror that. The nlm/optim outer path
        # ends on a deriv-0 gam2objective fit too (mgcv.r:1711), so it
        # shares every deriv-0 post-fit consequence.
        self._used_efs = (self.optimizer[0] == "efs" and sp is None
                          and n_sp > 0)
        self._used_nlm_optim = _res.get("used_nlm_optim", False)
        self._outer_deriv0 = self._used_efs or self._used_nlm_optim

        # Surface inner-loop warnings once, for the final fit only — mgcv
        # accumulates them in gam.fit3's warn list and intermediate newton()
        # evaluations run with printWarn=FALSE (gam.fit3.r:796-807).
        if fit.warn:
            import warnings as _w
            for _msg in fit.warn:
                _w.warn(_msg, stacklevel=2)

        # Unpack fit results. ``fit.A_chol`` is the Newton-W factorization
        # used by REML's log|H+S| term and the IFT for ∂β̂/∂ρ. mgcv's
        # post-fit reporting (m$edf, m$Vp, m$Ve) instead plugs in the
        # Fisher weight W_F = μ_η²/V (gam.fit3.r:644). Build a Fisher view
        # for those; for canonical links Newton ≡ Fisher and the view
        # reuses fit's chol — cheap.
        beta = fit.beta

        self._rho_hat = rho_hat
        # mgcv's ``m$full.sp`` (mgcv.r:2399-2401): the per-penalty sp
        # expansion exp(L·log(sp) + lsp0). Equals ``sp`` itself when no
        # id linkage and nothing fixed.
        self.full_sp = np.exp(np.asarray(rho_hat, dtype=float))

        fit_F = self._fisher_view(fit)
        A_chol = fit_F.A_chol
        A_chol_lower = fit_F.A_chol_lower
        # Fisher working weights — needed by reTest (Wood 2013) so summary()
        # can rebuild X'WX without re-running PIRLS. None ↔ unit weights.
        self._fisher_w = (
            np.asarray(fit_F.w, dtype=float).copy() if fit_F.w is not None else None
        )
        # Posterior β covariance Vp = σ²·A_F⁻¹. We get A_F⁻¹ once via
        # cho_solve(I) rather than via diag-tricks, since we need the full
        # matrix for Ve, per-coef SEs, and predict().
        A_inv = cho_solve((A_chol, A_chol_lower), np.eye(p))
        # F = A⁻¹X'WX (mgcv's edf matrix) and X'WX itself, computed
        # through the triangular factor instead of the explicit product:
        # with A = C'C and K_w = √W·X·C⁻¹ (= the data-rows orthogonal
        # factor when C came from the augmented QR),
        #     X'WX = C'(K_w'K_w)C,   F = C⁻¹(K_w'K_w)C.
        # The explicit √W·X' @ √W·X squares the condition number and made
        # edf garbage (negative totals) on κ(X) ≈ 1e10 designs that the
        # QR fit path handles exactly.
        if fit_F.w is None or np.allclose(fit_F.w, 1.0):
            Xw_F = X
        else:
            Xw_F = X * np.sqrt(np.maximum(fit_F.w, 0.0))[:, None]
        # (cho_factor leaves junk in the unused triangle; the explicit
        # matmuls below need it masked.)
        C_F = np.triu(A_chol) if not A_chol_lower else np.triu(A_chol.T)
        # K_w = Xw·C⁻¹  ⇔  C' K_w' = Xw'
        Kw_F = solve_triangular(C_F, Xw_F.T, lower=False, trans="T").T
        KtK_F = Kw_F.T @ Kw_F
        A_inv_XtWX = solve_triangular(C_F, KtK_F @ C_F, lower=False)
        # Per-coefficient edf = diag(F) where F = A⁻¹ X'WX. F is not
        # symmetric, so individual diag entries can be negative — mgcv
        # reports them verbatim (matches m$edf), and the per-smooth sum
        # remains non-negative and interpretable.
        edf = np.diag(A_inv_XtWX).copy()
        edf_total = float(edf.sum())
        # Prior weights (set at __init__ intake — gam(weights=) or ones).
        # residuals_of and the Pearson scale below share the exact array
        # PIRLS fit with.
        wt = self._wt
        # df.residual used in mgcv = n - edf_total. For unknown-scale
        # families, mgcv reports `m$sig2 = m$scale = scale.est`, regardless
        # of method — and its default estimator is **Fletcher (2012)**
        # (`gam.control(scale.est="fletcher")`, mgcv.r:2476): the Pearson
        # estimate divided by (1 + s̄) with
        #
        #     s̄ = max(-0.9, mean(V'(μ̂)·(y − μ̂)/V(μ̂)))      (gam.fit3.r:596-603)
        #
        # (unweighted mean; the -0.9 floor caps the correction at 10×
        # Pearson). s̄ = 0 — Fletcher ≡ Pearson — whenever V'·(y−μ)/V is a
        # score component of the fit (Gaussian: V'≡0; Gamma+log with
        # intercept: Σ(y−μ)/μ = 0 at convergence), so the correction only
        # moves Tweedie / IG / quasi-style fits.
        # This differs from `m$reml.scale = exp(log φ̂)` — the optimizer's
        # converged scale that enters the score formula. For REML on
        # Gaussian-identity the two coincide at the optimum (FOC enforces
        # Dp/(n−Mp) = dev/(n−edf)); for ML they differ since φ̂_ML = Dp/n.
        # ``_log_phi_hat`` is preserved separately for score evaluation.
        df_resid = float(n - edf_total)
        scale_est_kind = (self._control or _GAM_CONTROL_DEFAULTS)["scale_est"]
        if df_resid > 0 and not self._scale_known_fit:
            V = self.family.variance(fit.mu)
            pearson_scale = float(np.sum(wt * (y - fit.mu) ** 2 / V)) / df_resid
            s_bar = max(-0.9, float(np.mean(
                self.family.dvar(fit.mu) * (y - fit.mu) / V
            )))
            fletcher_scale = (
                pearson_scale / (1.0 + s_bar) if np.isfinite(s_bar)
                else pearson_scale
            )
            deviance_scale = float(fit.dev) / df_resid
        else:
            pearson_scale = (self._scale_fixed_value
                             if self._scale_known_fit else float("nan"))
            fletcher_scale = pearson_scale
            deviance_scale = pearson_scale
        self._pearson_scale = pearson_scale
        self._fletcher_scale = fletcher_scale
        if self._scale_known_fit:
            # mgcv: G$sig2 <- scale (mgcv.r:1942) — the known value is
            # reported, 1.0 on the family-default paths.
            scale = self._scale_fixed_value
        elif (self._family_mgcv_extended and self._log_phi_hat is not None):
            # mgcv-extended families (tw): mgcv reports the optimizer's
            # converged φ̂ — gam.fit3's efam scale.est is the scale passed
            # in as exp(ρ_φ), not the Fletcher estimator (verified: mgcv
            # tw sig2 ≡ exp(φ̂_REML) to 8 digits). Fletcher applies only
            # to *standard* unknown-scale families.
            scale = float(np.exp(self._log_phi_hat))
        elif scale_est_kind == "pearson":
            # gam.control(scale.est="pearson"): no Fletcher correction
            # (gam.fit3.r:596-598).
            scale = pearson_scale
        elif scale_est_kind == "deviance":
            # gam.fit3.r:606: (dev + dev.extra)/(n.true − trA).
            scale = deviance_scale
        else:
            scale = fletcher_scale
        sigma_squared = scale                 # alias kept for back-compat
        sigma = float(np.sqrt(sigma_squared)) if np.isfinite(sigma_squared) and sigma_squared >= 0 else float("nan")

        Vp = sigma_squared * A_inv
        Ve = sigma_squared * A_inv_XtWX @ A_inv

        # ------------- coefficient basis change (G_P) -----------------------
        # When a smooth's predict basis differs from its fit basis (today
        # only ``t2`` with null_dim ≥ 1), β was fit in a basis that doesn't
        # match what ``predict_mat`` returns. ``estimate.gam`` (mgcv,
        # smooth.r:264-267) handles this with a single ``coefficients <-
        # G$P %*% coefficients`` (and ``Vp <- G$P Vp G$P^T``) post-fit.
        # ``G_P`` is identity except: each remapped block's columns rotate
        # by ``M`` and contribute ``X̄ · β_block`` into the intercept row,
        # encoding ``X_fit = 1·X̄ + X_predict @ M`` exactly. With this in
        # place ``X_fit @ β_partial = X_predict @ (M β_partial) + (X̄ ·
        # β_partial)·1`` — so the in-sample η is unchanged and out-of-sample
        # ``predict_mat(new) @ G_P @ β_partial`` equals what the fit basis
        # would have produced.
        intercept_idx: Optional[int] = (
            column_names.index("(Intercept)") if has_intercept else None
        )
        if any(b.spec is not None and b.spec.coef_remap is not None for b in blocks):
            G_P = np.eye(p)
            for b, (a_col, b_col) in zip(blocks, block_col_ranges):
                if b.spec is None or b.spec.coef_remap is None:
                    continue
                M_b, X_bar_b = b.spec.coef_remap
                G_P[a_col:b_col, a_col:b_col] = M_b
                if intercept_idx is not None:
                    G_P[intercept_idx, a_col:b_col] = X_bar_b
            beta = G_P @ beta
            Vp = G_P @ Vp @ G_P.T
            Ve = G_P @ Ve @ G_P.T

        # ------------- attribute assembly ----------------------------------
        from ..R import NamedVector
        se = np.sqrt(np.diag(Vp))
        self._beta = beta
        self._se = se
        # User-facing coefficient reporting. When columns were dropped for
        # identifiability, mgcv zero-fills: coef 0, SE 0, t NaN, on the
        # *original* parameter vector — internals stay reduced.
        if self._keep_cols is not None:
            beta_rep = np.zeros(self._keep_cols.size)
            beta_rep[self._keep_cols] = np.asarray(beta).reshape(-1)
            se_rep = np.zeros(self._keep_cols.size)
            se_rep[self._keep_cols] = se
            names_rep = self._column_names_orig
        else:
            beta_rep = np.asarray(beta).reshape(-1)
            se_rep = se
            names_rep = column_names
        self._beta_report = beta_rep
        self._se_report = se_rep
        self.bhat = _row_frame(beta_rep, names_rep)
        self.coef = NamedVector(list(names_rep), beta_rep)
        self.coefficients = self.coef
        self.se_bhat = _row_frame(se_rep, names_rep)
        # Wald stats — useful for the parametric-row summary table; smooth
        # rows use the chi-squared-style test built on F per smooth, not per
        # basis column.
        t_stats = np.divide(beta_rep, se_rep,
                            out=np.full_like(beta_rep, np.nan),
                            where=se_rep > 0)
        self.t_values = _row_frame(t_stats, names_rep)
        # Use Student-t on df.residual (parametric Wald in mgcv summary).
        if df_resid > 0 and np.isfinite(df_resid):
            pv = 2 * _dist.pt(np.abs(t_stats), df_resid, lower_tail=False)
        else:
            pv = np.full_like(t_stats, np.nan)
        self.p_values = _row_frame(pv, names_rep)

        eta = fit.eta
        mu = fit.mu
        self.linear_predictors = eta
        self.fitted_values = mu
        self.fitted = mu                      # alias; for Gaussian μ = η
        # Default residuals = deviance residuals (mgcv default). For Gaussian
        # with prior weights = 1, sign(y-μ)·√((y-μ)²) = (y-μ), so the existing
        # Gaussian RSS-based summaries stay bit-identical.
        self.residuals = self._deviance_residuals(y, mu, self._wt)
        self.sigma = sigma
        self.sigma_squared = sigma_squared
        self.scale = sigma_squared            # mgcv's `$scale`

        # Penalized hat-matrix diagonal h_ii = w_i·(X·A_F⁻¹·X')_ii — mgcv's
        # `m$hat`, sums to edf_total. Plus rstandard.gam-style standardized
        # residuals: r / (σ̂·√(1−h)). For Gaussian-identity fit_F.w is None ⇒
        # unit weights. Cached here so plot_* methods don't recompute.
        w_F = fit_F.w if fit_F.w is not None else np.ones(n)
        HX = X @ A_inv
        self.leverage = (HX * X).sum(axis=1) * w_F
        sigma_for_std = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
        denom = sigma_for_std * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))
        # Pearson residuals need a variance function; ocat (ordered
        # categorical latent variable) has none — mgcv leaves type="pearson"
        # unavailable there too. Fall back to NaN rather than crash.
        try:
            V_mu = self.family.variance(mu)
            pearson_res = (y - mu) * np.sqrt(self._wt / np.maximum(V_mu, 0.0))
        except NotImplementedError:
            pearson_res = np.full(n, np.nan)
        self.std_dev_residuals = self.residuals / denom
        self.std_pearson_residuals = pearson_res / denom
        self.df_residuals = df_resid
        # Family deviance: `_FitState.dev` already holds Σ family.dev_resids
        # (Gaussian path: same as RSS). Keep `m.rss` as an alias for the
        # Gaussian-era name; new code should read `m.deviance`.
        self.deviance = float(fit.dev)
        self.rss = self.deviance              # alias (Gaussian: dev = rss)

        # Null deviance. gam.fit3 is always called with intercept=TRUE
        # (mgcv.r:1667), so the base value uses wtdmu = weighted mean of y
        # for *every* formula — including `- 1` ones (the linkinv(offset)
        # branch in gam.fit3.r:841 is unreachable via gam). estimate.gam
        # then replaces it with the deviance of glm(y ~ offset(off)) when
        # the model has an intercept and a nonzero offset, for non-extended
        # families (mgcv.r:2072-2075). df.null = n.ok − 1 with
        # n.ok = n − #zero-weight rows (gam.fit3.r:843-844; the
        # as.integer(intercept) term is always 1 via gam).
        # For Gaussian (V=1, wt=1, offset=0) this reduces to
        # Σ(y - mean(y))² = tss.
        mu_null_const = float(np.sum(wt * y) / np.sum(wt))
        mu_null = np.full(n, mu_null_const)
        self.null_deviance = float(np.sum(self.family.dev_resids(y, mu_null, wt)))
        if (has_intercept and np.any(self._offset != 0.0)
                and not self._family_mgcv_extended):
            self.null_deviance = self._offset_only_null_deviance(y, wt)
        self.df_null = float(n - int(np.sum(wt == 0.0)) - 1)
        # Extended-family postproc (estimate.gam, mgcv.r:2092-2098):
        # replaces null.deviance with find.null.dev's optimized constant
        # — NOT the weighted-mean value above (≠ at 1e-3 level even at
        # the log link once an offset is present) — and relabels the
        # family display name with the fitted θ ("Tweedie(p=…)",
        # "Negative Binomial(Θ)", "Scaled t(ν,σ)").
        self._postproc: dict = {}
        if self._family_mgcv_extended:
            pp = self.family.postproc(
                y, prior_weights=self._wt, fitted=mu,
                linear_predictors=fit.eta, offset=self._offset,
                intercept=has_intercept,
            )
            self._postproc = pp
            if pp.get("null_deviance") is not None:
                self.null_deviance = float(pp["null_deviance"])
            # betar reports "-2logLik as deviance": its dev_resids omit the
            # saturated reference, so postproc folds 2·saturated_ll into both
            # the deviance and the null deviance (efam.r:3479-3482). Other
            # extended families return no "deviance" key — byte-unchanged.
            if pp.get("deviance") is not None:
                self.deviance = float(pp["deviance"])
                self.rss = self.deviance

        self.Vp = Vp
        self.Ve = Ve
        self._A_inv = A_inv
        self.edf = edf
        self.edf_total = edf_total
        # Per-smooth edf: sum over the block's column range. Multi-block
        # smooths (by=factor) still roll up to a per-label dict — mgcv prints
        # one line per block.
        edf_by_smooth: dict[str, float] = {}
        for b, (a, bcol) in zip(blocks, block_col_ranges):
            edf_by_smooth[b.label] = float(edf[a:bcol].sum())
        self.edf_by_smooth = edf_by_smooth

        # Response-scale residual SS is what mgcv's r.sq is built on (uses
        # `object$y - object$fitted.values`, not deviance residuals — see
        # `summary.gam` line ~4055 in mgcv 1.9). For Gaussian-identity with
        # an intercept, sum(y - μ) = 0 from the unpenalized intercept's score
        # equation, so the variance-based formula reduces algebraically to
        # `1 - rss·(n-1)/(tss·df_resid)`, matching the legacy
        # `1 - (1 - rss/tss)(n-1)/df_resid` exactly.
        ss_resid_response = float(np.sum(wt * (y - mu) ** 2))
        if has_intercept and tss > 0:
            r_squared = 1.0 - ss_resid_response / tss
        elif yty > 0:
            r_squared = 1.0 - ss_resid_response / yty
        else:
            r_squared = float("nan")
        # mgcv's r.sq formula: 1 - var(√w·(y-μ))·(n-1) / (var(√w·(y-mean.y))·df_resid)
        # with var() = unbiased sample variance (denom n-1), matching R's var().
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
        # mgcv summary.gam sets r.sq NULL for families flagging no.r.sq
        # (ocat/ziP — mgcv.r:4055); print then shows Deviance explained
        # alone. NaN here drives the same suppression in hea's summary.
        if getattr(self.family, "no_r_sq", False):
            r_squared = r_squared_adjusted = float("nan")
        self.r_squared = float(r_squared)
        self.r_squared_adjusted = float(r_squared_adjusted)
        # Deviance explained — mgcv: (null.deviance - deviance) / null.deviance.
        if self.null_deviance > 0:
            self.deviance_explained = float(
                (self.null_deviance - self.deviance) / self.null_deviance
            )
        else:
            self.deviance_explained = float("nan")

        # Augmented REML Hessian wrt (ρ, log σ²) — both edf12 (Vr in Vc1
        # and Vc2) and vcomp (CIs on log σ_k) need it. Computed once and
        # cached. For GCV / no-smooth / non-finite σ², leave as None and
        # the consumers fall back to whatever they can do.
        if (
            method in ("REML", "ML", "P-REML", "P-ML")
            and n_sp > 0
            and not self._outer_deriv0
            and np.isfinite(sigma_squared)
            and sigma_squared > 0
        ):
            log_phi_hat_for_aug = (
                self._log_phi_hat
                if self._log_phi_hat is not None
                else float(np.log(sigma_squared))
            )
            # For mgcv-extended families with free θ (tw), the augmented
            # Hessian spans (ρ, log φ, θ_fam) — mgcv's outer.info$hess
            # includes the family parameters, and post.proc's db.drho gets
            # matching θ columns (gam.fit3.r:1018-1031).
            n_th_aug = (
                self.family.n_theta
                if (self._family_mgcv_extended and sp is None
                    and self.family.n_theta > 0)
                else 0
            )
            self._n_theta_aug = n_th_aug
            # mgcv's outer.info$hess spans exactly the optimizer's θ: ρ,
            # plus log φ only when the scale is estimated, plus family θ
            # for tw. Scale-known families (binomial, Poisson) have NO
            # log φ row — appending one shifts every Schur-complement ρρ
            # block (Vr → Vc1/Vc2 → edf2/Vc) off mgcv's, because the
            # (ρ, log φ) cross term −(∂Dp/∂ρ)/φ is nonzero at convergence.
            if self._pearson_scale_criterion:
                # P-REML / P-ML: the Vc sp-uncertainty correction uses the
                # criterion's OWN Hessian (mgcv `object$outer.info$hess` =
                # the P-REML/P-ML REML2 at the Pearson plug-in scale,
                # gam.fit3.r:979) — ρ-only, since the scale is the Pearson
                # plug-in φ_P(ρ), not a free outer parameter. NOT the
                # free-log φ plain-REML Hessian.
                n_th_aug = 0
                self._n_theta_aug = 0
                include_phi_aug = False
                H_aug = self._preml_hessian(rho_hat, fit=fit)
            else:
                include_phi_aug = not self._scale_known_fit
                H_aug = 0.5 * self._reml_hessian(
                    rho_hat, log_phi_hat_for_aug, fit=fit,
                    include_log_phi=include_phi_aug,
                    include_family_theta=n_th_aug > 0,
                )
            # Working-space view: the criterion is optimized over θ
            # (ρ = L·θ), so Vr — and every CI built on H_aug — lives in
            # working coordinates: H_θ = T'·H_ρ·T, T = blockdiag(L, I).
            T_aug = self._T_working((1 if include_phi_aug else 0) + n_th_aug)
            if T_aug is not None:
                H_aug = T_aug.T @ H_aug @ T_aug
            H_aug = 0.5 * (H_aug + H_aug.T)
        else:
            self._n_theta_aug = 0
            H_aug = None
        self._H_aug = H_aug
        # mgcv's df rule (`logLik.gam`): use sum(edf2) when available, where
        # edf2 is the sp-uncertainty-corrected df from Wood 2017 §6.11.3.
        # edf alone systematically under-counts because it conditions on the
        # estimated λ; edf2 = diag((σ²A⁻¹ + Vc1 + Vc2) X'X)/σ² absorbs the
        # extra variance from λ̂. Vc1 = (∂β/∂ρ) Vr (∂β/∂ρ)ᵀ is the obvious
        # bit; Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T accounts for the
        # ρ-dependence of L^{-T} in the Bayesian draw β̃ = β̂ + σ L^{-T} z.
        # edf1 = tr(2F-F²) is the upper bound; cap edf2 at edf1 in total
        # only. sc.p = 1 if scale is estimated, 0 if known (mgcv convention).
        # mgcv computes the sp-uncertainty correction (edf2 / Vc) ONLY on the
        # (RE)ML path — gam.fit3.post.proc / magic.post.proc return edf2 NULL
        # and Vc NULL for GCV/UBRE/GACV (verified: gamma/log GCV.Cp gives
        # sum(edf2)=0 and Vc=NULL), so logLik.gam/AIC fall back to edf there.
        # The magic branch already skipped it for gaussian GCV; extend the
        # skip to EVERY GCV-type fit so the non-magic GCV path (gamma/log,
        # binom, …) neither reports an edf2 mgcv doesn't nor pays for the ~2
        # redundant `_compute_Vr` re-fits (it passes no `fit`, so the
        # `_reml_hessian` fallback at the GCV branch re-solves the PIRLS).
        compute_edf2 = (method in ("REML", "ML", "P-REML", "P-ML")
                        and not self._outer_deriv0)
        if n_sp > 0 and (self._used_magic or not compute_edf2):
            # mgcv's magic.post.proc (mgcv.r:4475-4501): edf1 = 2·edf − diag(FF)
            # only; NO edf2 / Vc. edf2 := edf so logLik.gam/AIC use edf.
            F = A_inv_XtWX
            edf1_per_coef = 2.0 * np.diag(F) - np.einsum("ij,ji->i", F, F)
            self.edf1 = edf1_per_coef
            self.edf2 = edf.copy()
            self.edf1_total = float(edf1_per_coef.sum())
            self.edf2_total = edf_total
            Vc_corr = np.zeros_like(Vp)
        elif n_sp > 0:
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
        # mgcv's `model$Vc`: Vp + sp-uncertainty correction. Returned by
        # `vcov(model, unconditional=TRUE)`. Used by itsadug's plot_diff /
        # get_difference for the simultaneous-CI envelope.
        self.Vc = Vp + Vc_corr

        # edge.correct: recompute the Vc correction at the edge-corrected
        # smoothing parameters (gam.fit3.post.proc's k=2 pass,
        # gam.fit3.r:978-1030). Vb and edf2 keep the fitted-model values
        # (mgcv computes edf2 only at k=1); only Vc is replaced. The k=2
        # prior on Vr is 1e-7, not the k=1 1/10 (gam.fit3.r:1011).
        if (self._edge_theta1 is not None and method in ("REML", "ML")
                and n_sp > 0 and np.isfinite(sigma_squared)
                and sigma_squared > 0):
            n_work = self._work_dim
            th1 = self._edge_theta1
            rho1 = self._rho_full(th1[:n_work])
            has_phi1 = not self._scale_known_fit
            log_phi1 = (float(th1[n_work])
                        if (has_phi1 and th1.size > n_work)
                        else (self._log_phi_hat or 0.0))
            th_base1 = n_work + (1 if has_phi1 else 0)
            n_th_aug = self._n_theta_aug
            theta_fam_saved = self.family.get_theta().copy() if n_th_aug else None
            try:
                if n_th_aug:
                    self.family.set_theta(th1[th_base1:th_base1 + n_th_aug])
                fit1 = self._fit_given_rho(rho1)
                include_phi_aug1 = not self._scale_known_fit
                H_aug1 = 0.5 * self._reml_hessian(
                    rho1, log_phi1, fit=fit1,
                    include_log_phi=include_phi_aug1,
                    include_family_theta=n_th_aug > 0,
                )
                T_aug1 = self._T_working(
                    (1 if include_phi_aug1 else 0) + n_th_aug
                )
                if T_aug1 is not None:
                    H_aug1 = T_aug1.T @ H_aug1 @ T_aug1
                H_aug1 = 0.5 * (H_aug1 + H_aug1.T)
                db1 = self._db_drho(rho1, fit1.beta, fit1.A_chol,
                                    fit1.A_chol_lower)
                if self._L is not None:
                    db1 = db1 @ self._L
                if n_th_aug > 0:
                    db1 = np.hstack([db1, self._db_dtheta_fam(fit1)])
                    Vr1 = self._compute_Vr(rho1, H_aug1, with_theta=True)
                else:
                    Vr1 = self._compute_Vr(rho1, H_aug1)
                Vc1_1 = db1 @ Vr1 @ db1.T
                Vr_reg1 = self._compute_Vr(rho1, H_aug1, prior_var=1e-7)
                Vc2_1 = self._compute_Vc2(rho1, self._fisher_view(fit1),
                                          Vr_reg1, sigma_squared)
                self.Vc = Vp + Vc1_1 + Vc2_1
            finally:
                if theta_fam_saved is not None:
                    self.family.set_theta(theta_fam_saved)

        # AIC / logLik via mgcv's logLik.gam machinery (mgcv.r:4420):
        #   m$aic = family.aic(y, μ, dev1, wt, n) + 2·sum(edf)         (mgcv.r:1843)
        #   logLik(m) = sum(edf) + sc.p − m$aic/2                       (mgcv.r:4428)
        #   df_for_AIC = min(sum(edf2) + sc.p,  p_coef + sc.p)          (mgcv.r:4431-33)
        #   AIC(m) = -2·logLik(m) + 2·df_for_AIC                        (R's AIC.default)
        # `dev1` is family-specific (Gaussian uses dev directly, the Pearson
        # σ̂² is moment-based for the rest); see Family._aic_dev1.
        sc_p = 0.0 if self._scale_known_fit else 1.0
        # mgcv's dev1 scale (gam.fit3.r:848): REML/ML fits use reml.scale
        # = exp(φ̂) — the optimizer's converged scale — NOT the Fletcher
        # scale.est, which only feeds dev1 when reml.scale is NA (GCV.Cp).
        # Gaussian overrides _aic_dev1 to use dev directly either way.
        aic_scale = (float(np.exp(self._log_phi_hat))
                     if self._log_phi_hat is not None else scale)
        if self._scale_known_fit:
            # gam.fit3.r:848's FIRST branch: dev1 = scale·Σwt whenever
            # the scale is known — including a gam(scale=)-fixed
            # gaussian (its dev-based MLE override applies only to the
            # estimated case). Poisson/binomial defaults are the same
            # product as before (scale 1).
            dev1 = float(aic_scale) * float(np.sum(wt))
        else:
            dev1 = self.family._aic_dev1(self.deviance, aic_scale, wt)
        # cbind responses: family$aic gets the trials vector as ``n``
        # (gam.fit3.r:850) — distinct from wt = pw·n when extra prior
        # weights are present.
        n_aic = self._binom_n if self._binom_n is not None else n
        family_aic = float(self.family.aic(y, fit.mu, dev1, wt, n_aic))
        mgcv_aic = family_aic + 2.0 * edf_total                    # mgcv's m$aic
        logLik = sc_p + edf_total - 0.5 * mgcv_aic                 # mgcv's logLik value
        # mgcv leaves edf2 NULL on the GCV/UBRE/GACV path (gam.fit3.post.proc
        # gates on reml.scale), so logLik.gam's df falls back to edf there;
        # hea's GCV/GACV edf2 attribute stays as a best-effort extra but must
        # not leak into AIC/BIC. The Pearson-Laplace criteria (P-REML/P-ML)
        # DO get a Vc/edf2 (reml.scale non-NA), so they use edf2 like REML/ML
        # (logLik.gam: sum(edf2) whenever edf2 is non-NULL, mgcv.r:4431).
        df_base = (self.edf2_total
                   if method in ("REML", "ML", "P-REML", "P-ML")
                   else edf_total)
        df_for_aic = min(df_base + sc_p, float(p) + sc_p)          # capped at np
        # logLik.gam (mgcv.r): extended families add n.theta to the df
        # *after* the np cap (tw: +1 for the free p).
        if self._family_mgcv_extended:
            df_for_aic += float(getattr(self.family, "n_theta", 0) or 0)
        self.loglike = float(logLik)
        self.logLik = self.loglike                                 # alias (mgcv-style name)
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * logLik + 2.0 * df_for_aic
        self.BIC = -2.0 * logLik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)                           # mgcv's m$aic (different from AIC!)

        if self._is_laplace:
            # `_reml` returns -2·V_R (REML/P-REML) or -2·V_ML (ML/P-ML);
            # `summary()`'s `/2` recovers mgcv's `-REML`/`-ML` display value.
            # P-REML/P-ML always use the Pearson-Laplace scale φ = P/(n−Mp)
            # at the converged fit (the criterion's own scale, gam.fit3.r:641),
            # regardless of how ρ̂ was found. Plain REML/ML: known-scale fits
            # substitute log φ = log(scale); estimated-scale fits read the
            # outer-optimizer's (or sp= profile-out) log φ̂.
            if self._pearson_scale_criterion:
                log_phi_hat = float(np.log(max(self._phi_pearson(fit), 1e-300)))
            elif n_sp > 0:
                log_phi_hat = (
                    self._log_phi_hat if self._log_phi_hat is not None
                    else float(np.log(self._scale_fixed_value))
                )
            elif self._scale_known_fit:
                log_phi_hat = float(np.log(self._scale_fixed_value))
            else:
                # No penalties (a purely parametric formula): no outer
                # optimizer ran, but mgcv still reports a (RE)ML score
                # (gam.fit3 with 0 sp). Profile φ̂ exactly as the criterion's
                # reduction-to-Gaussian does — REML: Dp/(n−Mp), ML: Dp/n.
                Dp = fit.dev + fit.pen
                denom = ((float(n) - float(self._Mp)) if self._reml_ind == 1.0
                         else float(n))
                log_phi_hat = float(np.log(Dp / denom))
            score = float(self._reml(rho_hat, log_phi_hat, fit=fit))
            # REML/P-REML → REML_criterion; ML/P-ML → ML_criterion.
            if self._reml_ind == 1.0:
                self.REML_criterion = score
            else:
                self.ML_criterion = score
        elif self.method == "GACV.Cp" and not self._scale_known_fit:
            # GACV.Cp (scale estimated): the optimized GACV score
            # (gam.fit3.r:751). Stored in the GCV_score slot (mgcv's shared
            # `gcv.ubre`). Scale-known GACV.Cp falls through to UBRE below.
            self.GCV_score = float(self._gacv(rho_hat, fit=fit))
        else:
            # GCV/UBRE: `_gcv` estimates φ internally, so the no-penalty
            # case (empty rho) evaluates the same closed form mgcv reports.
            self.GCV_score = float(self._gcv(rho_hat, fit=fit))

        # Variance components: σ² and the implied per-slot std.dev's
        # σ_k = σ/√(sp_k/S.scale_k), with delta-method CIs (REML only).
        # Mirrors mgcv's gam.vcomp at its defaults (rescale=TRUE,
        # conf.lev=0.95); `_compute_vcomp(rescale=False)` gives the
        # fitted-scaling flavor. Cheap to compute eagerly for typical
        # n_sp; users can ignore the attribute if they don't need it.
        self.vcomp = self._compute_vcomp()

        # mgcv's estimated model rank (``oo$rank.est``) — identifiability
        # of the penalized problem at the converged fit. Structural
        # deficiency was handled by the pls_fit1-style column drop at
        # design time (so this re-estimate on the reduced design normally
        # comes back full); a *weight-induced* deficiency appearing only
        # at the converged W still warrants the loud warning.
        # When the working weights never moved from their initial values
        # (gaussian-identity and any other constant-weight canonical fit),
        # the converged-weight augmented system is bit-identical to the one
        # the structural drop above already factored at the SAME tol, and
        # that drop enforced full column rank — so rank.est is necessarily
        # self.p. Skip the redundant n×q QR; only a genuine weight-induced
        # deficiency (weights that actually changed) needs the re-estimate.
        # Exact equality avoids false positives — changed weights differ.
        _fw_now = self._fisher_w if self._fisher_w is not None else np.ones(n)
        if _fw_now.shape == w0.shape and np.array_equal(_fw_now, w0):
            self.rank = self.p
        else:
            self.rank = self._estimate_rank()
        if self.rank < p:
            import warnings as _w
            _w.warn(
                f"model is rank deficient at the converged weights: "
                f"estimated rank {self.rank} < {p} retained coefficients "
                "— estimates and SEs in the deficient directions are not "
                "individually interpretable.",
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------


    def _block_predict_mat(self, block, grid_df) -> np.ndarray:
        """Block basis at grid points, with the identifiability drop
        applied (predict-time bases rebuild the full columns; the fitted
        coefficients live on the reduced ones)."""
        B = np.asarray(block.spec.predict_mat(grid_df), dtype=float)
        if self._block_keep is not None:
            i = next(j for j, b in enumerate(self._blocks) if b is block)
            B = B[:, self._block_keep[i]]
        return B

    def _build_S_lambda(self, rho: np.ndarray) -> np.ndarray:
        """Assemble the full p×p penalty matrix Sλ at log-smoothing-params ρ.

        Each slot's k×k S_j is placed at its block's column range and
        multiplied by λ = exp(ρᵢ). Slots within the same block overlap
        (same col range) and are summed there — that's how tensor smooths
        get multiple penalties per block."""
        return _s_lambda(self._slots, self.p, rho)

    def _rho_full(self, theta_sp: np.ndarray) -> np.ndarray:
        """Working log-sp θ → per-penalty log-sp ρ = L·θ + lsp0 (mgcv's
        ``lsp = L %*% lsp_working + lsp0``). Identity when no smooths
        share an ``id`` and no sp entry is fixed; lsp0 carries the
        folded-out fixed log-sp's (mgcv.r:1513-1538)."""
        if self._L is None:
            rho = theta_sp
        else:
            rho = self._L @ theta_sp
        if self._lsp0 is not None:
            rho = rho + self._lsp0
        return rho

    def _T_working(self, n_extra: int) -> np.ndarray | None:
        """``T = blockdiag(L, I_extra)`` — the Jacobian ∂(ρ, extras)/∂(θ,
        extras) used to chain criterion derivatives to working space:
        ``g_θ = T'·g_ρ``, ``H_θ = T'·H_ρ·T`` (mgcv's newton applies the
        same ``L`` contraction, gam.fit3.r:1335-1340). ``None`` ⇔ identity.
        """
        if self._L is None:
            return None
        n_slots, n_work = self._L.shape
        T = np.zeros((n_slots + n_extra, n_work + n_extra))
        T[:n_slots, :n_work] = self._L
        if n_extra > 0:
            T[n_slots:, n_work:] = np.eye(n_extra)
        return T

    def _initial_sp_rho(self) -> np.ndarray:
        """mgcv's ``initial.spg`` seed for log-smoothing-params
        (mgcv.r:4528-4624) — *every* outer method starts here
        (estimate.gam mgcv.r:1998: ``lsp <- lsp2``, REML/ML included).

        Working weights at the family's starting μ̂ (the patched
        gam-initialize): ``w = wt·μ'(η₀)²/V(μ₀)``; extended families use
        the deviance curvature ``w = ½·Dmu2·μ'²`` (``EDmu2`` if any are
        negative). Then ``initial.sp(√w·X, S, off)``: per penalty

            def.sp[k] = mean(diag(X'WX)[ind]) / mean(diag(S_k)[ind])

        with ``ind`` filtering S_k to its penalised rows/cols
        (``thresh = eps^0.8·max|S_k|`` on row-mean, col-mean and diagonal
        simultaneously), followed by the global ×10 rebalance loop pushing
        ``mean(ldxx/(ldxx+ldss))`` to ~0.4 (mgcv.r:4666-4670). Returns
        log(def.sp) — the full-space ρ seed (the caller maps to working
        space through L when smooths share an id).
        """
        w = self._init_fisher_w(self._y_arr, X=self._X_full)
        Xw = self._X_full * np.sqrt(np.maximum(w, 0.0))[:, None]
        def_sp = _initial_sp(Xw, self._slots)
        return np.log(np.maximum(def_sp, 1e-300))

    def _offset_only_null_deviance(self, y: np.ndarray, wt: np.ndarray) -> float:
        """Deviance of ``glm(y ~ offset(off))`` — the intercept-only model
        refit *with* the offset, which is what mgcv reports as
        ``null.deviance`` for intercept+offset models (estimate.gam,
        mgcv.r:2072-2075 literally calls glm). Fisher IRLS on the single
        free coefficient with glm.fit's control (epsilon=1e-8, maxit=25,
        convergence on |Δdev|/(|dev|+0.1))."""
        family = self.family
        link = family.link
        off = self._offset
        mu = family.initialize(y, wt)
        eta = link.link(mu)
        dev = float(np.sum(family.dev_resids(y, mu, wt)))
        for _ in range(25):
            mu_eta_v = link.mu_eta(eta)
            V = family.variance(mu)
            good = mu_eta_v != 0.0
            safe = np.where(good, mu_eta_v, 1.0)
            z = np.where(good, eta - off + (y - mu) / safe, 0.0)
            w = np.where(good, wt * mu_eta_v ** 2 / V, 0.0)
            b0 = float(np.sum(w * z) / np.sum(w))
            eta = b0 + off
            mu = link.linkinv(eta)
            dev_new = float(np.sum(family.dev_resids(y, mu, wt)))
            done = abs(dev_new - dev) / (abs(dev_new) + 0.1) < 1e-8
            dev = dev_new
            if done:
                break
        return float(dev)

    def _resolve_null_coef(self) -> np.ndarray:
        """mgcv get.null.coef (mgcv.r:1863): ρ-independent null-model
        coefficients used as the PIRLS divergence baseline. Projects a
        constant valid η onto colspan(X); returns the coefficient vector
        passed as ``null_coef`` to `_gam_fit3`/`_gam_fit4`."""
        if self._null_baseline_cache is None:
            family = self.family
            link = family.link
            X = self._X_full
            y = self._y_arr
            wt = self._wt
            off = self._offset
            n, p = self.n, self.p
            fam_hook = getattr(family, "get_null_coef", None)
            if fam_hook is not None:
                # Family-supplied null model (mgcv.r:2022 — gfam's
                # per-member means, gfam.r:569-587).
                nc, _null_scale = fam_hook(X, y, wt, off)
                en = X @ nc
                mn = link.linkinv(en + off)
                self._null_baseline_cache = (nc, en, mn)
                return nc
            mu = (family.gam_initialize(y, wt, n=self._binom_n)
                  if self._binom_n is not None
                  else family.gam_initialize(y, wt))
            mu_null_const = float(np.average(mu, weights=wt))
            eta_null_full = link.link(np.full(n, mu_null_const))
            nc, *_ = np.linalg.lstsq(X, eta_null_full - off, rcond=None)
            en = X @ nc
            mn = link.linkinv(en + off)
            if not (link.valideta(en + off) and family.validmu(mn)):
                nc = np.zeros(p)
                en = np.zeros(n)
                mn = link.linkinv(off)
            self._null_baseline_cache = (nc, en, mn)
        return self._null_baseline_cache[0]

    def _fit_given_rho(self, rho: np.ndarray,
                       efs_scale: float | None = None) -> "_FitState":
        """Penalized IRLS at log-smoothing-params ρ.

        Iterate Newton-form working weights/responses

            αᵢ = 1 + (yᵢ − μᵢ)·(V'(μᵢ)/V(μᵢ) + g''(μᵢ)·dμᵢ/dηᵢ)
            wᵢ = αᵢ · (dμᵢ/dηᵢ)² / V(μᵢ)
            zᵢ = ηᵢ + (yᵢ − μᵢ) / ((dμᵢ/dηᵢ)·αᵢ)

        and solve ``(X'WX + Sλ)β = X'Wz`` by Cholesky each step. Per
        gam.fit3.r:118 the loop weights are Fisher (``α=1``) for canonical
        links and full Newton otherwise; the Newton form makes the
        converged ``H = X'WX + Sλ`` the *observed* penalized Hessian,
        which is what the implicit-function ``∂β̂/∂ρ = -exp(ρ_k) H⁻¹ S_k β̂``
        derivation assumes. For canonical links (incl. Gaussian-identity,
        Poisson-log, Gamma-inverse) ``α ≡ 1`` so Newton == Fisher.

        Control follows gam.fit3 under newton(): maxit=200, ε=1e-8
        (conv.tol/100 cap), the three step-halving inner loops
        (non-finite deviance / invalid η,μ / diverging pdev), and
        convergence on |Δpdev| < ε(|scale|+|pdev|) confirmed by the
        penalized-deviance gradient check (gam.fit3.r:447-462).
        """
        # Dispatch (gam.fit3.r:106-116): extended families (nb/tw/scat)
        # run gam.fit4's inner loop; ordinary exponential families use the
        # free `_gam_fit3`. General (gamlss) families never reach here.
        # ``efs_scale`` (efsudr only) puts gam.fit4 in EFS mode; gam.fit3's
        # EFS differs from REML only in the score tail, which efsudr
        # evaluates itself, so the flag is extended-family-only.
        if self._family_mgcv_extended:
            fit = _gam_fit4(
                self._X_full, self._y_arr, rho,
                slots=self._slots, UrS=self._UrS, reparam_Y=self._reparam_Y,
                keep_cols=self._keep_cols, reparam_cache=self._reparam_cache,
                weights=self._wt, start=self._pirls_start,
                etastart=self._pirls_etastart, mustart=self._pirls_mustart,
                offset=self._offset, family=self.family,
                control=self._control, null_coef=self._resolve_null_coef(),
                warm_eta=self._pirls_warm_eta, pls_lwork=self._pls_lwork,
                efs_scale=efs_scale,
            )
            if fit.converged and np.all(np.isfinite(fit.eta)):
                self._pirls_warm_eta = fit.eta
            return fit
        fit = _gam_fit3(
            self._X_full, self._y_arr, rho,
            slots=self._slots, UrS=self._UrS, reparam_Y=self._reparam_Y,
            keep_cols=self._keep_cols, reparam_cache=self._reparam_cache,
            weights=self._wt, start=self._pirls_start,
            etastart=self._pirls_etastart, mustart=self._pirls_mustart,
            offset=self._offset, family=self.family, control=self._control,
            null_coef=self._resolve_null_coef(), binom_n=self._binom_n,
            warm_eta=self._pirls_warm_eta,
            scale_fixed_value=self._scale_fixed_value,
            scale_known=self._scale_known_fit, pls_lwork=self._pls_lwork,
        )
        # Carry the converged predictor as the next score-eval's warm start
        # (mgcv gam.fit3.r:1366) — only on a finite converged fit.
        if fit.converged and np.all(np.isfinite(fit.eta)):
            self._pirls_warm_eta = fit.eta
        return fit

    def _init_fisher_w(self, y: np.ndarray,
                       X: np.ndarray | None = None) -> np.ndarray:
        """Working weights at the family's starting μ̂ — initial.spg's
        ``w = wt·μ'(η₀)²/V(μ₀)`` (mgcv.r:4595-4602), with the
        extended-family deviance-curvature branch ``w = ½·Dmu2·μ'²``
        (``EDmu2`` if any are negative). Shared by the initial.spg seed
        and the pls_fit1-style identifiability check.

        User starting values follow initial.spg's own precedence
        (mgcv.r:4591-4595): mustart wins outright; else start → η =
        X·start with NO offset (quirk), outranking etastart; else
        etastart → linkinv."""
        family = self.family
        link = family.link
        wt = self._wt
        mustart = (family.gam_initialize(y, wt, n=self._binom_n)
                   if self._binom_n is not None
                   else family.gam_initialize(y, wt))
        mustart_default = mustart
        if self._pirls_mustart is not None:
            mustart = np.asarray(self._pirls_mustart, dtype=float)
        elif self._pirls_start is not None and X is not None:
            mustart = link.linkinv(X @ self._pirls_start)
        elif self._pirls_etastart is not None:
            mustart = link.linkinv(
                np.asarray(self._pirls_etastart, dtype=float))

        def _w_at(mu0):
            eta0 = link.link(mu0)
            if self._family_mgcv_extended:
                dd = family.Dd(y, mu0, family.get_theta(), wt, level=0)
                mu_eta2 = link.mu_eta(eta0) ** 2
                w_ = 0.5 * np.asarray(dd["Dmu2"], dtype=float) * mu_eta2
                if np.any(w_ < 0):
                    w_ = 0.5 * np.asarray(dd["EDmu2"],
                                          dtype=float) * mu_eta2
            else:
                w_ = wt * link.mu_eta(eta0) ** 2 / family.variance(mu0)
            return w_

        with np.errstate(all="ignore"):
            w = _w_at(mustart)
        if mustart is not mustart_default and not np.all(np.isfinite(w)):
            # Invalid USER starting values: the PIRLS shrink loop
            # (gam.fit3.r:286-292) handles recovery for the fit itself;
            # the heuristic initial.spg weights just fall back to the
            # family default instead of NaN-poisoning the seed and the
            # rank check.
            w = _w_at(mustart_default)
        return w

    def _setup_reparam(self, slots: list["_PenaltySlot"] | None = None,
                       p: int | None = None) -> None:
        """Build mgcv's UrS: an orthonormal basis Y for the range of the
        *balanced* total penalty (totalPenaltySpace, gam.fit3.r:2661 —
        eigenvectors of Σ S_k/‖S_k‖_F above max·eps^0.66) and the
        per-penalty square roots projected into it (mini.roots,
        gam.fit3.r:2689). gam.reparam evaluates log|Sλ|+, det1, det2 in
        that fixed q_r-dim space. ``slots``/``p`` default to the model's
        (callers pass the *pre-drop* versions, mgcv semantics). Leaves
        ``self._UrS = None`` (legacy assembled-eigen path) when there are
        no penalties or when the eigen rank disagrees with the structural
        ``p − Mp`` (shouldn't happen for block-diagonal penalties;
        surfacing it beats silently mixing two rank conventions)."""
        if slots is None:
            slots = self._slots
        if p is None:
            p = self.p
        self._UrS = None
        self._reparam_cache = {}
        if not slots:
            return
        St = np.zeros((p, p))
        for slot in slots:
            a, b = slot.col_start, slot.col_end
            nrm = float(np.linalg.norm(slot.S, "fro"))
            if nrm > 0:
                St[a:b, a:b] += slot.S / nrm
        w, V = np.linalg.eigh(St)
        w_max = float(w.max()) if w.size else 0.0
        if w_max <= 0:
            return
        ind = w > w_max * float(np.finfo(float).eps) ** 0.66
        q_r = int(np.sum(ind))
        if q_r != self._penalty_rank:
            import warnings as _w
            _w.warn(
                f"totalPenaltySpace rank {q_r} != structural penalty rank "
                f"{self._penalty_rank}; falling back to the assembled-eigen "
                f"log|S|+ path",
                stacklevel=2,
            )
            return
        Y = V[:, ind]
        UrS = []
        for slot in slots:
            a, b = slot.col_start, slot.col_end
            b_root = _mroot(slot.S)        # estimated-rank columns
            full = np.zeros((p, b_root.shape[1]))
            full[a:b, :] = b_root
            UrS.append(Y.T @ full)
        self._UrS = UrS
        self._reparam_Y = Y

    def _reparam_at(self, rho: np.ndarray) -> dict | None:
        """gam.reparam at ρ (det/det1/det2 of log|Sλ|+ on the fixed total-
        penalty range space), cached for the repeated calls within one
        outer-Newton evaluation (criterion, gradient, Hessian all hit the
        same ρ). None ⇔ reparam basis unavailable (see _setup_reparam)."""
        return _reparam_eval(self._UrS, self._reparam_cache, rho)

    def _penalty_root(self, rho: np.ndarray) -> np.ndarray:
        """Square root E (e × p) of the assembled penalty, ``E'E = Sλ``,
        for the augmented least-squares fit (mgcv's ``Sr``).

        Primary source is gam.reparam's leakage-free root mapped back to
        the original basis: mgcv fits in the transformed basis ``x·T``
        with ``Sr = [rp$E | 0]`` (gam.fit3.r:162-181); since
        ``T = U1·blockdiag(Qs, I)`` is orthogonal, the augmented matrix
        has identical singular values in either basis and hea stays in
        the original one with ``E = rp$E·Qs'·Y'``. With the
        identifiability drop active, the kept-column slice of E is an
        exact root of the reduced Sλ. Falls back to an eigen root of the
        assembled Sλ when no reparam basis exists."""
        return _penalty_root_of(self._slots, self.p, self._UrS,
                                self._reparam_Y, self._keep_cols,
                                self._reparam_cache, rho)

    def _log_det_S_pos(self, rho):
        """Shim → free `_log_det_S_pos`."""
        return _log_det_S_pos(rho, penalty_rank=self._penalty_rank, slots=self._slots, p=self.p)

    def _ml_logdet_adj(self, fit):
        """Shim → free `_ml_logdet_adj` (mgcv gdi1/gdi2 derivative block)."""
        return _ml_logdet_adj(fit, self._Mp)

    def _reml(self, rho, log_phi=0.0, fit=None):
        """Shim → free `_reml`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _reml(rho, log_phi, fit, Mp=self._Mp, wt=self._wt, y=self._y_arr, binom_n=self._binom_n, gamma=self._gamma, family=self.family, family_mgcv_extended=self._family_mgcv_extended, use_ml_proj=self._use_ml_proj, pearson_scale_criterion=self._pearson_scale_criterion, reml_ind=self._reml_ind, penalty_rank=self._penalty_rank, slots=self._slots, p=self.p, UrS=self._UrS, reparam_cache=self._reparam_cache)

    def _profile_log_phi_fixed_sp(self, fit, log_phi0):
        """Shim → free `_profile_log_phi_fixed_sp` (mgcv gdi1/gdi2 derivative block)."""
        return _profile_log_phi_fixed_sp(fit, log_phi0, self._Mp, self._gamma, self._wt, self._y_arr, self.family, self._reml_ind)

    def _reml_grad(self, rho, log_phi=0.0, fit=None, include_log_phi=False, include_family_theta=False):
        """Shim → free `_reml_grad`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        dw_deta = self._dw_deta(fit)
        return _reml_grad(rho, log_phi, fit, include_log_phi, include_family_theta, Mp=self._Mp, X=self._X_full, gamma=self._gamma, slots=self._slots, wt=self._wt, y=self._y_arr, family=self.family, pearson_scale_criterion=self._pearson_scale_criterion, reml_ind=self._reml_ind, use_ml_proj=self._use_ml_proj, p=self.p, family_mgcv_extended=self._family_mgcv_extended, penalty_rank=self._penalty_rank, UrS=self._UrS, reparam_cache=self._reparam_cache, dw_deta=dw_deta)

    def _dW_dtheta_total(self, fit, dd1=None):
        """Shim → free `_dW_dtheta_total`."""
        return _dW_dtheta_total(fit, dd1, X=self._X_full, family=self.family, y=self._y_arr, wt=self._wt, family_mgcv_extended=self._family_mgcv_extended)

    def _dlog_det_H_dtheta(self, fit, dd1=None):
        """Shim → free `_dlog_det_H_dtheta`."""
        return _dlog_det_H_dtheta(fit, dd1, X=self._X_full, family=self.family, y=self._y_arr, wt=self._wt, family_mgcv_extended=self._family_mgcv_extended)

    def _reml_hessian(self, rho, log_phi=0.0, fit=None, include_log_phi=False, include_family_theta=False):
        """Shim → free `_reml_hessian`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        dw_deta = self._dw_deta(fit)
        d2w_deta2 = self._d2w_deta2(fit)
        return _reml_hessian(rho, log_phi, fit, include_log_phi, include_family_theta, X=self._X_full, gamma=self._gamma, slots=self._slots, wt=self._wt, y=self._y_arr, family=self.family, p=self.p, pearson_scale_criterion=self._pearson_scale_criterion, use_ml_proj=self._use_ml_proj, penalty_rank=self._penalty_rank, family_mgcv_extended=self._family_mgcv_extended, Mp=self._Mp, UrS=self._UrS, reparam_cache=self._reparam_cache, dw_deta=dw_deta, d2w_deta2=d2w_deta2)

    def _init_general(self, formulas, data, *, method, sp, family,
                      offset, weights, gamma, select, knots,
                      control=None, start=None,
                      optimizer=("outer", "newton")):
        """estimate.gam's general-family glue (mgcv.r:1893-1924,
        1984-2005, 2060-2092): multi-formula design → Sl.setup +
        initial repara → initial.spg seed → outer Newton over the
        gam.fit5 REML closure → final deriv-2 fit. The post-fit
        surface (gam.fit5.post.proc: Vp/edf/summary/predict/plot) is
        the next §5.3 slice — only fitting attributes are populated.

        ``method`` is coerced to REML like mgcv (mgcv.r:1894-1898,
        silently); ``sp=`` takes the working-length vector and fixes
        every sp (all-or-nothing — the single-formula mixed-sp fold
        hasn't been extended here); ``control=`` supplies the efs and
        newton knobs (edge_correct still raises).
        """
        self._control = gam_control(**(control or {}))
        if self._control["edge_correct"]:
            raise NotImplementedError(
                "edge_correct for general families lands with "
                "gam.fit5.post.proc; not ported for general families."
            )
        if offset is not None:
            raise NotImplementedError(
                "constructor offset= is not supported for formula-list "
                "gam; put offset(...) atoms in the per-LP formulas."
            )
        if not (np.isfinite(gamma) and gamma > 0):
            raise ValueError(
                f"gamma must be a positive finite number, got {gamma!r}")
        if knots is not None and not isinstance(knots, dict):
            raise TypeError(
                "knots must be a dict mapping covariate name -> knot "
                "sequence (mgcv's knots=list(...)), or None")
        if len(formulas) != family.n_lp:
            raise ValueError(
                f"family {family!r} expects {family.n_lp} linear "
                f"predictors; got {len(formulas)} formulas.")
        # general families coerce the method to REML (mgcv.r:1894-1898)
        self.method = "REML"
        self.family = family
        self.formula = list(formulas)
        self.knots = knots
        self._select = bool(select)
        self._gamma = float(gamma)
        self._edge_correct = False
        self._edge_theta1 = None

        md = _prepare_multi_design(
            list(formulas), data, knots, select,
            allow_single=(family.n_lp == 1),
            matrix_response=getattr(family, "matrix_response", False))
        # families with parameters beyond the linear-predictor coefs (mvn's
        # Choleski-factor params) append unpenalized "dummy" columns to the
        # design — mgcv's preinitialize (mvam.r:92-131). They sit in NO
        # linear predictor and carry no penalty; the family's ll reads them
        # as the trailing coefs.
        self._n_extra_coef = int(getattr(family, "n_extra_coef", 0) or 0)
        if self._n_extra_coef:
            md = _append_extra_params(md, self._n_extra_coef)
        self._init_general_from_md(md, data, family=family, sp=sp,
                                   method=method, weights=weights,
                                   start=start, optimizer=optimizer)

    def _init_general_from_md(self, md, data, *, family, sp, method,
                              weights, start,
                              optimizer=("outer", "newton")):
        """The design-agnostic remainder of :meth:`_init_general`: from a
        built multi-LP design bundle ``md`` (dense ``_MultiDesign``, or
        bam's compressed variant whose ``X`` is a ``family.DiscreteX``)
        through Sl.setup/initial-repara, smoothness selection, the final
        gam.fit5 and the whole post-fit surface. Every step below is
        p-space or reaches the design through ``family.ll``/
        ``initialize_coef``/``_gam_fit5``, which dispatch on DiscreteX —
        the one dense-only step (the initial repara of X's columns) is
        deferred to gam.fit5's boundary transforms on the discrete rail.
        Callers must have set ``self.family``/``formula``/``knots``/
        ``_select``/``_gamma``/``_control``/``_n_extra_coef`` first."""
        self._drop_intercept_col = None
        if getattr(family, "drop_intercept", False):
            # cox.ph drops the intercept (drop.intercept=TRUE, coxph.r:355):
            # the partial likelihood is invariant to a constant shift of η
            # (the baseline hazard absorbs it), so the intercept is
            # unidentified and removed from the design. Remember its column
            # so predict's newdata design drops the same one.
            if "(Intercept)" in md.column_names:
                self._drop_intercept_col = md.column_names.index(
                    "(Intercept)")
            md = _drop_general_intercept(md)
        y = np.asarray(md.y, dtype=float)
        n = md.n
        self.n = n
        self.p = md.p
        # Materialize smooth-arg expressions (e.g. ``s(sqrt(protime))``)
        # into ``self.data`` so plot_smooth / partial-residual lookups
        # find the transformed covariate column, matching the single-
        # formula path (gam.py:1104). Each LP carries its own map.
        from ..formula import (_apply_smooth_arg_exprs as _asae,
                               _smooth_arg_expr_map as _saem)
        _expr_map: dict = {}
        for _lp in md.lps:
            _expr_map.update(_saem(_lp.expanded))
        self.data = _asae(data, _expr_map) if _expr_map else data
        if weights is None:
            self._wt = np.ones(n)
        else:
            wt_prior = np.asarray(weights, dtype=float).flatten()
            if wt_prior.shape != (n,):
                raise ValueError(
                    f"weights must have length {n}, got {wt_prior.shape}")
            if not np.all(np.isfinite(wt_prior)):
                raise ValueError("missing or non-finite values in weights")
            if np.any(wt_prior < 0):
                raise ValueError("negative weights not allowed")
            self._wt = wt_prior
        self.prior_weights = self._wt
        self._y_arr = y

        self._md = md
        self._slots = md.slots
        self._L = md.L
        sl = _sl_setup(md.slots, md.p)
        self._sl = sl
        # Discrete rail: the compressed design has no column form to
        # transform — gam.fit5 applies the same block transforms in
        # p-space around family.ll instead (see _gam_fit5's discrete
        # seam), so X passes through untouched.
        X_irp = (md.X if isinstance(md.X, _DiscreteX)
                 else _sl_initial_repara(sl, md.X, both_sides=False))
        # mgcv mvn preinitialize's coefficient seeding (mvam.r:115-125):
        # per-LP magic GCV fits store family$ibeta once; initialize_coef
        # then returns it on every later call (gam.fit5 and initial.spg
        # alike — mgcv's initialize expression, mvam.r:152-155). mgcv
        # reparas G$X (mgcv.r:1902) BEFORE preinitialize runs (:1985), so
        # the seed fit pairs the INITIAL-REPARA'D X with the ORIGINAL
        # G$S penalties — mgcv's own mixed gauge, mirrored exactly; the
        # resulting ibeta is in the irp gauge gam.fit5 consumes.
        pre_g = getattr(family, "preinitialize_general", None)
        if pre_g is not None:
            pre_g(y=y, X=X_irp, lpi=md.lpi, slots=md.slots)

        # G$Mp: total-penalty null-space dimension (mgcv.r:1924) —
        # computed structurally (≡ ncol(totalPenaltySpace$Z), verified).
        Mp = sum(md.nsdf)
        for b, (a, bc) in zip(md.blocks, md.block_col_ranges):
            k = bc - a
            if not b.S:
                Mp += k
                continue
            Mp += k - _sym_rank(np.sum(
                [np.asarray(s_, dtype=float) for s_ in b.S], axis=0))
        # NOTE: the appended covariance params do NOT enter Mp. mgcv fixes
        # G$Mp at gam.setup — before preinitialize appends the dummy
        # columns — so the family's extra params are absent from the REML
        # normalizing constant Mp·log(2π)/2 (verified: mvn REML matches R
        # only with Mp = the mean-design null space, not +d(d+1)/2).
        self._Mp = Mp
        self._penalty_rank = md.p - Mp

        # user start= (mgcv.r:1903): the coefficient vector enters the
        # fitting basis through the forward initial repara
        # (both.sides=FALSE — the vector/coefficient transform).
        start_irp = None
        if start is not None:
            start_arr = np.asarray(start, dtype=float).reshape(-1)
            if start_arr.shape != (md.p,):
                raise ValueError(
                    f"start must have length {md.p} (the stacked "
                    f"coefficient vector), got {start_arr.shape}")
            start_irp = _sl_initial_repara(sl, start_arr,
                                           both_sides=False)

        self._g5 = {
            "X": X_irp, "y": y, "sl": sl, "lpi": md.lpi,
            "offsets": md.offsets, "Mp": Mp, "weights": self._wt,
            "start": start_irp, "gamma": self._gamma,
        }

        # mgcv's estimate.gam general-family branch (mgcv.r:1984+): bundle
        # the general slice of G and run smoothness selection + the final
        # gam.fit5, then assign the returned state onto self.
        seed_start = start_irp
        if start is not None and isinstance(md.X, _DiscreteX):
            # initial.spg reaches the design only through family.ll,
            # which consumes MODEL-basis coefficients on the discrete
            # rail (the irp transform is gam.fit5's boundary seam).
            seed_start = np.asarray(start, dtype=float).reshape(-1)
        gg = _GGeneral(
            n_work=self._work_dim, Mp=Mp, wt=self._wt, family=family,
            control=self._control, gamma=self._gamma, X=X_irp, y=y, sl=sl,
            md_L=md.L, lpi=md.lpi, offsets=md.offsets, p=md.p, slots=md.slots,
            g5=self._g5, optimizer=optimizer, seed_start=seed_start)
        _res = estimate_gam(
            None, sp, method, rho_full=self._rho_full,
            outer_newton=self._outer_newton, outer_bfgs=self._outer_bfgs,
            general=gg, get_outer_info=lambda: self._outer_info)
        fit = _res["fit"]
        self._fit5 = fit
        self._rho_hat = _res["rho_hat"]
        self.sp = _res["sp"]
        self.full_sp = np.exp(np.asarray(_res["rho_hat"], dtype=float))
        self._outer_info = _res["outer_info"]
        self.outer_info = _res["outer_info"]
        self.REML_criterion = _res["REML_criterion"]
        self.converged = _res["converged"]

        from ..R import NamedVector
        coefs = _sl_initial_repara(sl, fit["coefficients"], inverse=True)
        self._beta = coefs
        names = list(md.column_names)
        self.column_names = names
        self.coef = NamedVector(names, coefs)
        self.coefficients = self.coef
        self.bhat = _row_frame(coefs, names)
        self.lpi = [np.asarray(ix, dtype=int) for ix in md.lpi]
        self.fitted_values = fit["fitted_values"]      # (n, n_lp)
        self.fitted = self.fitted_values
        self.linear_predictors = fit["linear_predictors"]
        self.rank = fit["rank"]
        # gam.fit5's scale.est ≡ 1: the scale never enters the outer
        # problem for general families.
        self.scale = 1.0
        self.sigma_squared = 1.0
        # mgcv runs the family postproc on the converged fit
        # (estimate.gam, mgcv.r:2092-2098): family-specific deviance /
        # null-deviance and, for families whose fitted matrix needs a
        # final transform, an in-place `fitted.values` rewrite (gammals
        # exponentiates its log-mean column, gamlss.r:2739). hea returns
        # that as an optional `fitted` key, applied BEFORE residuals so
        # residuals_of/qq see the same matrix mgcv does. r² is skipped
        # (no.r.sq); null_deviance exists only when postproc supplies it.
        # families whose postproc/predict need the design (cox.ph forms
        # the baseline-hazard `a` vectors, absent from the 6-arg postproc
        # signature) get it via an optional context hook, set on the
        # ORIGINAL-basis design + coefficients before postproc runs.
        ctx_hook = getattr(family, "set_fit_context", None)
        if ctx_hook is not None:
            ctx_hook(X=md.X, coef=coefs, offset=md.offsets[0])
        pp = family.postproc(
            y, prior_weights=self._wt, fitted=fit["fitted_values"],
            linear_predictors=fit["linear_predictors"],
            offset=md.offsets, intercept=True,
        )
        fitted_override = pp.pop("fitted", None)
        if fitted_override is not None:
            self.fitted_values = np.asarray(fitted_override, dtype=float)
            self.fitted = self.fitted_values
        self._postproc = pp
        self.residuals = family.residuals(
            y, self.fitted_values, **self._family_residuals_kw())
        self.deviance = (float(pp["deviance"])
                         if pp.get("deviance") is not None
                         else float(np.sum(np.asarray(self.residuals,
                                                      dtype=float) ** 2)))
        self.null_deviance = (float(pp["null_deviance"])
                              if pp.get("null_deviance") is not None
                              else float("nan"))
        # general families carry no r² (mgcv's no.r.sq); deviance
        # explained still prints when the family's postproc gave a null
        # deviance (mgcv print.summary.gam shows exactly that line).
        self.r_squared = float("nan")
        self.r_squared_adjusted = float("nan")
        self.deviance_explained = (
            (self.null_deviance - self.deviance) / self.null_deviance
            if np.isfinite(self.null_deviance) and self.null_deviance != 0
            else float("nan"))
        # summary/anova plumbing: blocks + ranges for the smooth table,
        # per-LP parametric indices for the p.table (mgcv.r:3907-3912).
        self._blocks = md.blocks
        self._block_col_ranges = md.block_col_ranges
        if sum(md.nsdf) > 0:
            self._param_idx = np.concatenate(
                [np.arange(ps, ps + nd)
                 for ps, nd in zip(md.pstart, md.nsdf) if nd > 0]
            ).astype(int)
        else:
            self._param_idx = np.zeros(0, dtype=int)
        self.parametric_columns = [names[i] for i in self._param_idx]

        # ---- gam.fit5.post.proc + the estimate.gam tail ----------------
        mv = self._fit5_post_proc(fit)
        self.Vp = mv["Vp"]
        self.Ve = mv["Ve"]
        self.Vc = mv["Vc"]
        self._V_sp = mv["V_sp"]
        self._R_fit5 = mv["R"]
        self.edf = mv["edf"]
        self.edf1 = mv["edf1"]
        self.edf2 = mv["edf2"]
        self.edf_total = float(np.sum(self.edf))
        self.edf1_total = float(np.sum(self.edf1))
        self.edf2_total = float(np.sum(self.edf2))
        se = np.sqrt(np.maximum(np.diag(self.Vp), 0.0))
        self._se = se
        self._beta_report = coefs
        self._se_report = se
        self.se_bhat = _row_frame(se, names)
        self.df_residuals = float(n) - self.edf_total
        # per-coef Wald (z — the scale is known ≡ 1 for general
        # families, so summary's p.table uses N(0,1) like mgcv).
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(coefs, se,
                                out=np.full_like(coefs, np.nan),
                                where=se > 0)
        self.t_values = _row_frame(t_stats, names)
        self.p_values = _row_frame(
            2 * _nmath.pnorm5_vec(np.abs(t_stats), lower_tail=False), names)
        # mgcv's m$aic = fit5's −2l + 2Σedf (mgcv.r:1843); AIC()'s df is
        # Σedf2 capped at #coef (logLik.gam; sc.p = 0 — scale fixed).
        mgcv_aic = fit["aic"] + 2.0 * self.edf_total
        log_lik = self.edf_total - 0.5 * mgcv_aic        # = fit l
        df_for_aic = min(self.edf2_total, float(self.p))
        self.loglike = float(log_lik)
        self.logLik = self.loglike
        self.npar = float(df_for_aic)
        self.AIC = -2.0 * log_lik + 2.0 * df_for_aic
        self.BIC = -2.0 * log_lik + float(np.log(n)) * df_for_aic
        self._mgcv_aic = float(mgcv_aic)
        # vcomp machinery: the outer REML2 Hessian doubles as mgcv's
        # outer.info$hess (working ρ-only space — the layout
        # _compute_vcomp expects for scale-known fits); σ ≡ 1.
        self.sigma = 1.0
        self._H_aug = (np.asarray(self._outer_info["hess"], dtype=float)
                       if self._outer_info.get("hess") is not None
                       and np.size(self._outer_info.get("hess")) > 0
                       else None)
        self.vcomp = self._compute_vcomp()

    def _fit5_post_proc(self, fit: dict) -> dict:
        """mgcv ``gam.fit5.post.proc`` (gam.fit4.r:1571-1719, the
        edge.correct=FALSE path).

        From the converged gam.fit5 state (fit parameterization,
        possibly with dropped parameters): the unpivoted root R of
        −lbb (nearest-PSD eigen retry when the likelihood Hessian
        isn't +ve definite — which also rebuilds the penalized factor
        and REPLACES fit's L for Vb), Vb = Hp⁻¹ through that factor,
        zero-row reinsertion for dropped coefficients, the
        sp-uncertainty corrections (Vc = Vb + db·hess⁺·db′ + the
        Vb.corr factor-derivative term — "NOTE: unscaled", σ²≡1) with
        V.sp at the 1/50-regularized prior, both reparameterizations
        undone, then F = Vb·R′R, Ve = F·Vb, edf/edf1/edf2 with the
        Σedf2 ≤ Σedf1 cap.
        """
        sl = self._sl
        lbb = -np.asarray(fit["lbb"], dtype=float)
        p = lbb.shape[0]
        D = np.asarray(fit["D"], dtype=float)
        L_fac = fit["L"]
        piv = fit["piv"]
        ipiv = fit["ipiv"]

        # pre-condition lbb before testing rank (gam.fit4.r:1597)
        lbb_pre = D * (D * lbb).T
        R_f, piv_R, rank_R = _pivoted_chol(lbb_pre)
        if rank_R < p:
            # not +ve definite: nearest +ve semi-definite retry,
            # rebuilding the penalized factor as well (1601-1626)
            tol, dtol = 0.0, 1e-7
            ev, V = np.linalg.eigh(lbb_pre)
            mev = float(ev.max())
            while True:
                ev[ev < tol * mev] = tol * mev
                R_f = np.sqrt(ev)[:, None] * V.T
                lbb_pre = R_f.T @ R_f
                Hp = lbb_pre + D * (D * np.asarray(fit["St"],
                                                   dtype=float)).T
                L_new, piv_n, rank_n = _pivoted_chol(Hp)
                if rank_n == p:
                    R_f = R_f / D[None, :]      # R'R = lbb (original)
                    L_fac, piv = L_new, piv_n
                    ipiv = np.empty_like(piv)
                    ipiv[piv] = np.arange(p)
                    break
                tol += dtol
                dtol *= 10.0
        else:
            ipiv_R = np.empty_like(piv_R)
            ipiv_R[piv_R] = np.arange(p)
            R_f = R_f[:, ipiv_R] / D[None, :]   # R'R = lbb (original)

        # Vb = D·Hp_pre⁻¹·D through the (possibly rebuilt) factor
        Dm = np.diag(D)[piv, :]
        sol = solve_triangular(L_fac, Dm, lower=False, trans="T")[ipiv, :]
        Vb = sol.T @ sol

        bdrop = np.asarray(fit["bdrop"], dtype=bool)
        if bdrop.any():                          # reinsert zero rows
            q = bdrop.size
            ibd = ~bdrop
            Vt, Vb = Vb, np.zeros((q, q))
            Vb[np.ix_(ibd, ibd)] = Vt
            Rt, R_f = R_f, np.zeros((q, q))
            R_f[np.ix_(ibd, ibd)] = Rt

        hess = self._outer_info.get("hess")
        have_corr = (hess is not None and np.size(hess) > 0
                     and fit.get("db_drho") is not None)
        V_sp = None
        Vr = None
        Vc_corr = 0.0
        if have_corr:
            hess = np.asarray(hess, dtype=float)
            db = np.asarray(fit["db_drho"], dtype=float)
            if self._L is not None:              # derivs w.r.t. working
                db = db @ self._L
            ev_h, V_h = np.linalg.eigh(hess)
            nonpos = ev_h <= 0
            d = ev_h.copy()
            d[nonpos] = 0.0
            d[~nonpos] = 1.0 / np.sqrt(d[~nonpos])
            db = _sl_inirep(sl, db, lt=1, r=0)    # undo initial repara
            tmp = (d[:, None] * V_h.T) @ db.T
            Vc_corr = tmp.T @ tmp                # first correction
            d2 = ev_h.copy()
            d2[nonpos] = 0.0
            d2 = 1.0 / np.sqrt(d2 + 1.0 / 50.0)  # k=1 prior (1671)
            Vr = (V_h * (d2 * d2)) @ V_h.T
            V_sp = Vr

        Vb = _sl_repara(fit["rp"], Vb, inverse=True)
        Vb = _sl_initial_repara(sl, Vb, inverse=True)
        Vc = Vb + Vc_corr
        R_f = _sl_repa(fit["rp"], R_f, r=1)
        R_f = _sl_initial_repara(sl, R_f, inverse=True,
                                 both_sides=False, cov=False)
        RtR = R_f.T @ R_f
        F = Vb @ RtR
        Ve = F @ Vb
        edf = np.diag(F).copy()
        if have_corr:
            # second correction in the original parameterization —
            # Vb.corr(R, L, lsp0, S, off, w=NULL, lsp, Vr) ≡ the
            # _compute_Vc2 chain at σ² = 1 (1709).
            Vc = Vc + self._vb_corr_fit5(RtR, Vr)
        edf1 = 2.0 * edf - np.einsum("ij,ji->i", F, F)
        edf2 = np.sum(Vc * RtR, axis=1)
        if float(np.sum(edf2)) > float(np.sum(edf1)):
            edf2 = edf1.copy()
        return {"Vc": Vc, "Vp": Vb, "Ve": Ve, "V_sp": V_sp, "edf": edf,
                "edf1": edf1, "edf2": edf2, "R": R_f}

    def _vb_corr_fit5(self, RtR: np.ndarray, Vr: np.ndarray) -> np.ndarray:
        """gam.fit5.post.proc's ``Vb.corr`` call (gam.fit3.r:869-952
        with w=NULL): rebuild H = R'R + Σλ_k S_k in MODEL coordinates,
        Cholesky it (bail to 0 like mgcv when that fails), and reuse
        the `_compute_Vc2` factor-derivative chain (≡ vcorr) — with
        penalty-only dH and NO σ² scaling ("NOTE: unscaled!!")."""
        p = self.p
        A = RtR.copy()
        lam = np.exp(np.asarray(self._rho_hat, dtype=float))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            A[a:b, a:b] += lam[k] * slot.S
        try:
            C = np.linalg.cholesky(0.5 * (A + A.T))
        except np.linalg.LinAlgError:
            return np.zeros((p, p))
        import types
        duck = types.SimpleNamespace(A_chol=C, A_chol_lower=True)
        return self._compute_Vc2(self._rho_hat, duck, Vr, 1.0)

    def sp_vcov(self, edge_correct: bool = True, reg: float = 1e-3):
        """mgcv ``sp.vcov`` (mgcv.r:4221-4234): covariance of the
        (working) log smoothing parameters from the outer optimizer's
        Hessian — ``solve(hess + reg)``, mgcv's literal elementwise
        regularizer. ``None`` when no Hessian is available (GCV/GACV
        fits, fixed sp) — mgcv gates on method ∈ {ML, P-ML, REML,
        P-REML, fREML}; hea's fREML fits carry ``method == "REML"``
        (the bam alias), so the same set is covered by the four
        strings below. ``edge_correct`` mirrors mgcv's formal: its
        branch fires only when the fit stored an edge-corrected
        Hessian (``attr(hess, "hess1")``, written by
        ``gam.control(edge.correct=TRUE)`` fits); hea has no
        edge-corrected fitting, so — exactly like mgcv on a
        non-edge-corrected fit — the argument falls through to
        ``solve(hess + reg)``."""
        H = getattr(self, "_H_aug", None)
        if H is None or self.method not in ("REML", "ML", "P-REML", "P-ML"):
            return None
        return np.linalg.solve(np.asarray(H, dtype=float) + reg,
                               np.eye(H.shape[0]))

    def _score_all(self, rho, log_phi, fit, deriv, scoreType, *,
                   include_log_phi=False, include_family_theta=False,
                   T_work=None):
        """Shim → free `_gam_fit3_score` (mgcv's gam.fit3 score tail). Sources
        the 23 fit-independent args from ``self`` and computes the
        working-weight derivatives POLYMORPHICALLY (``self._dw_deta`` — bam
        overrides to length-p; see SB1) only when a derivative is requested
        for a W-dependent criterion (REML/P-REML)."""
        need_w = scoreType in ("REML", "PREML")
        dw_deta = self._dw_deta(fit) if (need_w and deriv >= 1) else None
        d2w_deta2 = self._d2w_deta2(fit) if (need_w and deriv >= 2) else None
        return _gam_fit3_score(
            rho, log_phi, fit, deriv, scoreType, T_work=T_work,
            include_log_phi=include_log_phi,
            include_family_theta=include_family_theta,
            dw_deta=dw_deta, d2w_deta2=d2w_deta2, Mp=self._Mp, wt=self._wt,
            y=self._y_arr, binom_n=self._binom_n, gamma=self._gamma,
            family=self.family,
            family_mgcv_extended=self._family_mgcv_extended,
            use_ml_proj=self._use_ml_proj,
            pearson_scale_criterion=self._pearson_scale_criterion,
            reml_ind=self._reml_ind, penalty_rank=self._penalty_rank,
            slots=self._slots, p=self.p, UrS=self._UrS,
            reparam_cache=self._reparam_cache, X=self._X_full, XtX=self._XtX,
            n=self.n, scale_fixed_value=self._scale_fixed_value,
            scale_known=self._scale_known_fit, pls_lwork=self._pls_lwork)

    def _outer_newton(
        self, theta0: np.ndarray, *, include_log_phi: bool,
        criterion: str = "REML",
        include_family_theta: bool = False,
        max_iter: int = 200, conv_tol: float = 1e-6,
        max_step: float = 5.0, max_sd_step: float = 2.0,
        max_half: int = 30, qerror_thresh: float = 0.8,
    ) -> np.ndarray:
        """Shim → free `newton` (mgcv gam.fit3.r:1290). Sources data from
        ``self`` and threads the polymorphic fit/score/view machinery as
        callables (bam overrides `_fit_given_rho`/`_dw_deta`), then persists
        the ``outer_info``/``outer_fit``/``edge_theta1`` writes mgcv's newton
        returns in its result list."""
        res = newton(
            theta0, include_log_phi=include_log_phi, criterion=criterion,
            include_family_theta=include_family_theta, max_iter=max_iter,
            conv_tol=conv_tol, max_step=max_step, max_sd_step=max_sd_step,
            max_half=max_half, qerror_thresh=qerror_thresh,
            family=self.family, work_dim=self._work_dim,
            X=getattr(self, "_X_full", None),
            y=getattr(self, "_y_arr", None), wt=getattr(self, "_wt", None),
            n=self.n,
            scale_fixed_value=self._scale_fixed_value, control=self._control,
            g5=getattr(self, "_g5", None), edge_correct=self._edge_correct,
            fit_fn=self._fit_given_rho, score_fn=self._score_all,
            fisher_view_fn=self._fisher_view, rho_full_fn=self._rho_full,
            T_working_fn=self._T_working,
        )
        self._outer_info = res["outer_info"]
        self._outer_fit = res["outer_fit"]
        self._edge_theta1 = res["edge_theta1"]
        return res["theta"]

    def _outer_bfgs(
        self, theta0: np.ndarray, *, conv_tol: float = 1e-6,
        max_Nstep: float = 3.0, max_step: int = 200,
    ) -> np.ndarray:
        """Shim → free `bfgs` (mgcv gam.fit3.r:1722). REML5-only (general
        families); sources `_g5`/family/L from ``self`` and persists the
        ``outer_info`` write."""
        res = bfgs(
            theta0, conv_tol=conv_tol, max_Nstep=max_Nstep, max_step=max_step,
            g5=self._g5, family=self.family, L=self._L,
            rho_full_fn=self._rho_full,
        )
        self._outer_info = res["outer_info"]
        return res["theta"]

    def _outer_nlm_optim(self, theta0: np.ndarray, *, optimizer2: str,
                         criterion: str, include_log_phi: bool,
                         include_family_theta: bool,
                         fscale: float) -> np.ndarray:
        """Shim → free `gam_outer_nlm_optim` (mgcv gam.outer's nlm/optim
        branch, mgcv.r:1692-1717). Sources the fit/score machinery from
        ``self`` like `_outer_newton` and persists the final deriv-0
        ``gam2objective`` fit + the optimizer return (``outer_info``:
        nlm's minimum/estimate/gradient/code/iterations, optim's
        par/value/counts/convergence/message)."""
        res = gam_outer_nlm_optim(
            theta0, optimizer2=optimizer2, criterion=criterion,
            fscale=fscale, control=self._control,
            include_log_phi=include_log_phi,
            include_family_theta=include_family_theta,
            n_work=self._work_dim, family=self.family,
            scale_fixed_value=self._scale_fixed_value,
            fit_fn=self._fit_given_rho, score_fn=self._score_all,
            rho_full_fn=self._rho_full,
            T_working_fn=self._T_working,
        )
        self._outer_info = res["outer_info"]
        self._outer_fit = res["fit"]
        return res["theta"]

    def _outer_efsudr(self, rho0: np.ndarray,
                      log_phi0: float | None) -> dict:
        """Shim → free `efsudr` (mgcv gam.fit4.r:822): sources the data
        args from ``self``, forces ``gamma=1`` in the REML evaluations
        (gam.outer never forwards gamma to efsudr, mgcv.r:1665, and
        efsudr hard-codes ``gamma=1`` in its gam.fit3 calls), and
        persists the ``outer_info``/``outer_fit`` writes like
        `_outer_newton` does."""
        def _reml1(rho_c, lp_c, fit_c):
            return _reml(
                rho_c, lp_c, fit_c, Mp=self._Mp, wt=self._wt,
                y=self._y_arr, binom_n=self._binom_n, gamma=1.0,
                family=self.family,
                family_mgcv_extended=self._family_mgcv_extended,
                use_ml_proj=self._use_ml_proj,
                pearson_scale_criterion=self._pearson_scale_criterion,
                reml_ind=self._reml_ind,
                penalty_rank=self._penalty_rank, slots=self._slots,
                p=self.p, UrS=self._UrS,
                reparam_cache=self._reparam_cache)

        def _scale_est1(fit_c):
            return _fit3_scale_est(
                fit_c, family=self.family, y=self._y_arr, wt=self._wt,
                n=self.n, X=self._X_full, control=self._control,
                fisher_view_fn=self._fisher_view)

        res = efsudr(
            rho0, log_phi0=log_phi0, family=self.family,
            family_mgcv_extended=self._family_mgcv_extended,
            fit_fn=self._fit_given_rho, reml_fn=_reml1,
            scale_est_fn=_scale_est1, fisher_view_fn=self._fisher_view,
            UrS=self._UrS,
            reparam_Y=getattr(self, "_reparam_Y", None),
            reparam_cache=self._reparam_cache, p=self.p, n=self.n,
            control=self._control,
            scale_fixed_value=self._scale_fixed_value,
        )
        self._outer_info = res["outer_info"]
        self._outer_fit = res["fit"]
        return res


    def _S_pinv(self, S_full):
        """Shim → free `_S_pinv` (mgcv gdi1/gdi2 derivative block)."""
        return _S_pinv(S_full, self._penalty_rank)

    def _make_K(self, A_chol, A_chol_lower):
        """Shim → free `_make_K` (mgcv gdi1/gdi2 derivative block)."""
        return _make_K(A_chol, A_chol_lower, self._X_full)

    def _fisher_view(self, fit):
        """Shim → free `_fisher_view`."""
        return _fisher_view(
            fit, family=self.family,
            family_mgcv_extended=self._family_mgcv_extended,
            y=self._y_arr, wt=self._wt, X=self._X_full,
            pls_lwork=self._pls_lwork, n=self.n,
        )

    def _dbeta_drho(self, fit, rho):
        """Shim → free `_dbeta_drho` (mgcv gdi1/gdi2 derivative block)."""
        return _dbeta_drho(fit, rho, self._slots, self.p)

    def _fit_link_derivs(self, fit):
        """Shim → free `_fit_link_derivs` (mgcv gdi1/gdi2 derivative block)."""
        return _fit_link_derivs(fit, self.family)

    def _Dd(self, fit, level):
        """Shim → free `_Dd`."""
        return _Dd(fit, level, family=self.family, y=self._y_arr, wt=self._wt)

    def _dDeta(self, fit, level):
        """Shim → free `_dDeta`."""
        return _dDeta(fit, level, family=self.family, y=self._y_arr, wt=self._wt)

    def _dw_deta(self, fit):
        """Shim → free `_dw_deta`."""
        return _dw_deta(fit, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt)

    def _d2beta_drho_drho(self, fit, rho, db_drho=None, dw_deta=None):
        """Shim → free `_d2beta_drho_drho`."""
        return _d2beta_drho_drho(fit, rho, db_drho, dw_deta, slots=self._slots, p=self.p, X=self._X_full, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt)

    @staticmethod
    def _as_n_nt(arr, nt):
        """Shim → free `_as_n_nt`."""
        return _as_n_nt(arr, nt)

    def _theta2_arr(self, arr, nt, n):
        """Shim → free `_theta2_arr`."""
        return _theta2_arr(arr, nt, n)

    def _d2beta_theta(self, fit, rho, *, db_drho, db_dtheta, dd2):
        """Shim → free `_d2beta_theta`."""
        return _d2beta_theta(fit, rho, db_drho=db_drho, db_dtheta=db_dtheta, dd2=dd2, X=self._X_full, slots=self._slots, family=self.family, p=self.p)

    def _d2w_deta2(self, fit):
        """Shim → free `_d2w_deta2`."""
        return _d2w_deta2(fit, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt)

    def _dlog_det_S_drho(self, rho, S_pinv=None, S_full=None):
        """Shim → free `_dlog_det_S_drho`."""
        return _dlog_det_S_drho(rho, S_pinv, S_full, slots=self._slots, p=self.p, penalty_rank=self._penalty_rank)

    def _d2log_det_S_drho_drho(self, rho, S_pinv=None, S_full=None):
        """Shim → free `_d2log_det_S_drho_drho`."""
        return _d2log_det_S_drho_drho(rho, S_pinv, S_full, slots=self._slots, p=self.p, penalty_rank=self._penalty_rank)

    def _dlog_det_H_drho(self, fit, rho, db_drho=None):
        """Shim → free `_dlog_det_H_drho`."""
        return _dlog_det_H_drho(fit, rho, db_drho, X=self._X_full, slots=self._slots, p=self.p, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt)

    def _dDp_drho(self, fit, rho):
        """Shim → free `_dDp_drho` (mgcv gdi1/gdi2 derivative block)."""
        return _dDp_drho(fit, rho, self._slots)

    def _dbeta_dp_tw(self, fit):
        """Shim → free `_dbeta_dp_tw`."""
        return _dbeta_dp_tw(fit, X=self._X_full, wt=self._wt, y=self._y_arr, family=self.family)

    def _dlog_det_H_dp_tw(self, fit, db_dp=None):
        """Shim → free `_dlog_det_H_dp_tw`."""
        return _dlog_det_H_dp_tw(fit, db_dp, X=self._X_full, wt=self._wt, y=self._y_arr, family=self.family, family_mgcv_extended=self._family_mgcv_extended)

    def _dW_dp_tw_total(self, fit, db_dp=None):
        """Shim → free `_dW_dp_tw_total`."""
        return _dW_dp_tw_total(fit, db_dp, X=self._X_full, wt=self._wt, y=self._y_arr, family=self.family, family_mgcv_extended=self._family_mgcv_extended)

    def _gcv(self, rho, fit=None):
        """Shim → free `_gcv`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gcv(rho, fit, gamma=self._gamma, n=self.n, scale_fixed_value=self._scale_fixed_value, scale_known=self._scale_known_fit, X=self._X_full, XtX=self._XtX, p=self.p, family=self.family, family_mgcv_extended=self._family_mgcv_extended, y=self._y_arr, wt=self._wt, pls_lwork=self._pls_lwork)

    def _fisher_edf(self, fit):
        """Shim → free `_fisher_edf`."""
        return _fisher_edf(fit, X=self._X_full, XtX=self._XtX, p=self.p, family=self.family, family_mgcv_extended=self._family_mgcv_extended, y=self._y_arr, wt=self._wt, pls_lwork=self._pls_lwork, n=self.n)

    def _use_magic(self, criterion: str, include_log_phi: bool,
                   include_family_theta: bool) -> bool:
        """mgcv's ``outer.looping == FALSE`` dispatch (mgcv.r:1932): Gaussian
        family + identity link (``G$am``, mgcv.r:2327), GCV (scale unknown),
        no id-linkage / fixed-sp (so working-sp == per-penalty log-sp)."""
        return (
            criterion == "GCV"
            and not self._scale_known_fit
            and isinstance(self.family, Gaussian)
            and getattr(self.family.link, "name", None) == "identity"
            and self._L is None and self._lsp0 is None
            and not include_log_phi and not include_family_theta
            and len(self._slots) > 0
        )

    def _magic_optimize(self, rho0, tol: float = 1e-7, max_half: int = 15):
        """Shim → free `_magic_optimize`; persists the cached QR + outer_info."""
        res = _magic_optimize(
            rho0, tol=tol, max_half=max_half, wt=self._wt, y=self._y_arr,
            offset=self._offset, struct_R=self._struct_R,
            keep_cols=self._keep_cols, X=self._X_full, gamma=self._gamma,
            slots=self._slots, p=self.p)
        self._magic_R = res["magic_R"]
        self._magic_y0 = res["magic_y0"]
        self._outer_info = res["outer_info"]
        return res["sp"]

    def _magic_fit_state(self, rho):
        """Shim → free `_magic_fit_state`."""
        return _magic_fit_state(
            rho, magic_R=self._magic_R, magic_y0=self._magic_y0,
            fit_given_rho_fn=self._fit_given_rho, family=self.family,
            X=self._X_full, offset=self._offset, y=self._y_arr, wt=self._wt,
            slots=self._slots, p=self.p, UrS=self._UrS,
            reparam_Y=self._reparam_Y, keep_cols=self._keep_cols,
            reparam_cache=self._reparam_cache)

    def _pearson_and_deriv(self, rho, fit, deriv=True):
        """Shim → free `_pearson_and_deriv`."""
        return _pearson_and_deriv(rho, fit, deriv, family=self.family, wt=self._wt, y=self._y_arr, slots=self._slots, X=self._X_full, p=self.p)

    def _pearson_hess(self, fit, rho, *, db_drho=None, d2b=None):
        """Shim → free `_pearson_hess`."""
        return _pearson_hess(fit, rho, db_drho=db_drho, d2b=d2b, X=self._X_full, slots=self._slots, wt=self._wt, y=self._y_arr, family=self.family, p=self.p, family_mgcv_extended=self._family_mgcv_extended)

    def _phi_pearson(self, fit):
        """Shim → free `_phi_pearson`."""
        return _phi_pearson(fit, Mp=self._Mp, n=self.n, family=self.family, wt=self._wt, y=self._y_arr, slots=self._slots, X=self._X_full, p=self.p)

    def _gacv(self, rho, fit=None):
        """Shim → free `_gacv`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gacv(rho, fit, gamma=self._gamma, n=self.n, X=self._X_full, XtX=self._XtX, p=self.p, family=self.family, family_mgcv_extended=self._family_mgcv_extended, y=self._y_arr, wt=self._wt, pls_lwork=self._pls_lwork, slots=self._slots)

    def _gacv_grad(self, rho, fit=None):
        """Shim → free `_gacv_grad`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gacv_grad(rho, fit, gamma=self._gamma, slots=self._slots, n=self.n, X=self._X_full, XtX=self._XtX, family=self.family, p=self.p, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, wt=self._wt, pls_lwork=self._pls_lwork)

    def _preml_grad(self, rho, fit):
        """Shim → free `_preml_grad`."""
        dw_deta = self._dw_deta(fit)
        return _preml_grad(rho, fit, slots=self._slots, Mp=self._Mp, n=self.n, y=self._y_arr, wt=self._wt, family=self.family, reml_ind=self._reml_ind, X=self._X_full, p=self.p, gamma=self._gamma, pearson_scale_criterion=self._pearson_scale_criterion, use_ml_proj=self._use_ml_proj, family_mgcv_extended=self._family_mgcv_extended, penalty_rank=self._penalty_rank, UrS=self._UrS, reparam_cache=self._reparam_cache, dw_deta=dw_deta)

    def _gcv_grad(self, rho, fit=None):
        """Shim → free `_gcv_grad`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gcv_grad(rho, fit, gamma=self._gamma, slots=self._slots, n=self.n, scale_fixed_value=self._scale_fixed_value, scale_known=self._scale_known_fit, X=self._X_full, XtX=self._XtX, family=self.family, p=self.p, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, wt=self._wt, pls_lwork=self._pls_lwork)

    def _gcv_grad_pieces(self, rho, fit):
        """Shim → free `_gcv_grad_pieces`."""
        return _gcv_grad_pieces(rho, fit, X=self._X_full, XtX=self._XtX, slots=self._slots, family=self.family, n=self.n, p=self.p, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, wt=self._wt, pls_lwork=self._pls_lwork)

    def _gcv_hessian(self, rho, fit=None, *, return_pieces=False):
        """Shim → free `_gcv_hessian`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gcv_hessian(rho, fit, return_pieces=return_pieces, X=self._X_full, XtX=self._XtX, gamma=self._gamma, slots=self._slots, n=self.n, p=self.p, scale_fixed_value=self._scale_fixed_value, scale_known=self._scale_known_fit, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt, pls_lwork=self._pls_lwork)

    def _gacv_hessian(self, rho, fit=None):
        """Shim → free `_gacv_hessian`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        return _gacv_hessian(rho, fit, gamma=self._gamma, slots=self._slots, n=self.n, X=self._X_full, XtX=self._XtX, p=self.p, scale_fixed_value=self._scale_fixed_value, scale_known=self._scale_known_fit, y=self._y_arr, family_mgcv_extended=self._family_mgcv_extended, family=self.family, wt=self._wt, pls_lwork=self._pls_lwork)

    def _preml_hessian(self, rho, fit=None):
        """Shim → free `_preml_hessian`."""
        if fit is None:
            fit = self._fit_given_rho(rho)
        dw_deta = self._dw_deta(fit)
        d2w_deta2 = self._d2w_deta2(fit)
        return _preml_hessian(rho, fit, slots=self._slots, Mp=self._Mp, n=self.n, family=self.family, wt=self._wt, y=self._y_arr, X=self._X_full, p=self.p, gamma=self._gamma, pearson_scale_criterion=self._pearson_scale_criterion, use_ml_proj=self._use_ml_proj, penalty_rank=self._penalty_rank, family_mgcv_extended=self._family_mgcv_extended, UrS=self._UrS, reparam_cache=self._reparam_cache, reml_ind=self._reml_ind, dw_deta=dw_deta, d2w_deta2=d2w_deta2)

    def _db_drho(self, rho, beta, A_chol, A_chol_lower):
        """Shim → free `_db_drho` (mgcv gdi1/gdi2 derivative block)."""
        return _db_drho(rho, beta, A_chol, A_chol_lower, self._slots, self.p)

    def _test_stat(
        self,
        X_b: np.ndarray,
        V_b: np.ndarray,
        beta_b: np.ndarray,
        rank: float,
        res_df: float = -1.0,
    ) -> tuple[float, float, float]:
        """mgcv ``testStat`` with ``type = 0`` (summary.r default) —
        Wood (2013) Biometrika 100(1), 221-228. Direct port of
        mgcv.r:3759-3855 including the p-value computation.

        Returns ``(stat, pval, rank_out)``. ``stat`` is the d-statistic,
        ``rank_out`` the (possibly truncated) rank reported as Ref.df.
        ``res_df`` mirrors mgcv's ``res.df``: ``<= 0`` means the scale is
        fixed/known (chi-squared reference); ``> 0`` is the residual d.f.
        used to estimate the scale (F-type reference).

        The "fractional rank" correction blends the k-th and (k+1)-th
        whitened eigenvectors via a 2×2 symmetric square root so the test
        respects a non-integer reference d.f. The statistic is ambiguous
        up to the sign of the first blended column, so mgcv computes both
        variants (``d`` from ``vec``, ``d1`` from ``vec1``) and averages
        the two p-values (the statistics can't be averaged — the mixture
        distribution of the average is unknown). The primary p-value is
        the weighted-chi-squared survival via ``psum_chisq`` (Davies):

            val = [1, …, 1, (rp+√(rp(2−rp)))/2, rp − val[k]],  rp = ν+1
            scale known:     Pr(Σ val_j·χ²_1 > d)
            scale estimated: Pr(Σ val_j·χ²_1 − (d/k0)·χ²_k0 > 0),
                             k0 = max(1, round(res_df))

        The plain ``pchisq(d, rank)`` / ``pf(d/rank, rank, res_df)`` form
        is only the fallback when the mixture p-value is unavailable
        (integer rank sets pval=2 to force it; Davies failure gives NaN).
        """
        # QR on the smooth's design block, then rotate Vp into that basis.
        # (mgcv uses qr(X, tol=0): with tol=0 LINPACK never pivots, so the
        # unpivoted numpy QR is the same operation.)
        _, R = np.linalg.qr(X_b, mode="reduced")
        V_rot = R @ V_b @ R.T
        V_rot = 0.5 * (V_rot + V_rot.T)
        d_eig, U = np.linalg.eigh(V_rot)
        # Descending order, mgcv sign convention (first row >= 0).
        d_eig = d_eig[::-1]
        U = U[:, ::-1]
        siv = np.sign(U[0, :])
        siv = np.where(siv == 0, 1.0, siv)
        U = U * siv

        k = max(0, int(np.floor(rank)))
        nu = abs(rank - k)
        k1 = k + 1 if nu > 0 else k

        # mgcv's effective-rank guard: if eigenvalue tail is below
        # max·eps^0.9, drop them and shrink k1.
        if d_eig.size > 0 and d_eig[0] > 0:
            r_est = int(np.sum(d_eig > d_eig[0] * np.finfo(float).eps ** 0.9))
        else:
            r_est = 0
        if r_est < k1:
            k1 = k = r_est
            nu = 0.0
            rank = float(r_est)

        if k1 == 0 or U.shape[1] == 0:
            return 0.0, 1.0, float(rank)

        vec = U[:, :k1].copy()

        if nu > 0 and k > 0:
            # Whiten cols 0 .. k-2 (R: cols 1..k-1) by 1/sqrt(eigenvalue).
            if k > 1:
                scales = 1.0 / np.sqrt(d_eig[:k - 1])
                vec[:, :k - 1] = vec[:, :k - 1] * scales[np.newaxis, :]
            b12 = 0.5 * nu * (1.0 - nu)
            b12 = float(np.sqrt(max(b12, 0.0)))
            B = np.array([[1.0, b12], [b12, nu]], dtype=float)
            ev = np.diag(d_eig[k - 1:k + 1] ** -0.5)
            B = ev @ B @ ev
            eb_d, eb_v = np.linalg.eigh(B)
            rB = eb_v @ np.diag(np.sqrt(np.maximum(eb_d, 0.0))) @ eb_v.T
            cols_orig = vec[:, k - 1:k + 1].copy()
            # vec1 negates the first of the two cols before rB.
            cols_neg = cols_orig.copy()
            cols_neg[:, 0] = -cols_neg[:, 0]
            vec[:, k - 1:k + 1] = cols_orig @ rB.T
            vec1 = vec.copy()
            vec1[:, k - 1:k + 1] = cols_neg @ rB.T
        else:
            if k == 0:
                # Degenerate: scale all of vec by 1/sqrt(d_eig[0]).
                if d_eig[0] > 0:
                    vec = vec * (1.0 / np.sqrt(d_eig[0]))
            else:
                scales = 1.0 / np.sqrt(d_eig[:k])
                vec = vec * scales[np.newaxis, :]
            vec1 = vec

        Rp = R @ beta_b
        d = float(np.sum((vec.T @ Rp) ** 2))
        d1 = float(np.sum((vec1.T @ Rp) ** 2))

        rank1 = float(rank)            # rank for the fallback below

        if nu > 0:
            # Mixture-of-chi² reference distribution (primary path).
            if k1 == 1:
                rank1 = 1.0
                val = np.ones(1)
            else:
                val = np.ones(k1)
                rp = nu + 1.0
                val[k - 1] = (rp + math.sqrt(rp * (2.0 - rp))) / 2.0
                val[k1 - 1] = rp - val[k - 1]
            if res_df <= 0:
                pval = 0.5 * (psum_chisq(d, val) + psum_chisq(d1, val))
            else:
                k0 = max(1, int(round(res_df)))
                df = np.concatenate(
                    [np.ones(val.size, dtype=int), np.array([k0], dtype=int)]
                )
                pval = 0.5 * (
                    psum_chisq(0.0, np.concatenate([val, [-d / k0]]), df)
                    + psum_chisq(0.0, np.concatenate([val, [-d1 / k0]]), df)
                )
        else:
            pval = 2.0                 # force the fallback (mgcv convention)

        # mgcv's ``pval > 1`` fallback. ``not (pval <= 1)`` also catches a
        # NaN from a Davies/Liu failure. (hea's psum_chisq clips into
        # [0, 1], so unlike mgcv a degraded-but-finite Davies result stays
        # at 1.0 rather than re-routing here — conservative, far lower tail
        # only.)
        if not (pval <= 1.0):
            if res_df <= 0:
                pval = 0.5 * (
                    float(_dist.pchisq(d, rank1, lower_tail=False)) + float(_dist.pchisq(d1, rank1, lower_tail=False))
                )
            else:
                pval = 0.5 * (
                    float(_dist.pf(d / rank1, rank1, res_df, lower_tail=False))
                    + float(_dist.pf(d1 / rank1, rank1, res_df, lower_tail=False))
                )

        return d, float(min(1.0, pval)), float(rank)

    def _recov(
        self, m_idx: int, re_idx: list[int] | tuple[int, ...] = (),
        v_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port of ``mgcv:::recov`` (mgcv.r:3599-3713). Returns ``(Ve, Rm)``.

        ``Ve`` is the frequentist covariance of β̂ under a data
        distribution in which the smooths indexed by ``re_idx`` are
        treated as *fully random* (β_re ~ N(0, σ²·S2⁻)):

            Ve = Vp·R'·L'L·R·Vp / σ²,   L'L = I + R2·S2⁻·R2'

        with R the model R factor (R'R = X'WX; for general families
        gam.fit5.post.proc's root with R'R = −lbb), R2 its random columns
        and S2 the random blocks' penalty. With ``re_idx`` empty this
        reduces to the usual model ``Ve`` (returned symmetrized, as mgcv
        does).

        ``v_scale`` is summary.gam's ``dispersion=`` rescale of the
        object covariances (mgcv.r:3897-3898 replaces ``object$Vp``/
        ``$Ve`` by ``·dispersion/sig2`` before reTest runs; ``sig2``
        itself stays put, so the factor enters exactly where mgcv's
        modified matrices do — quadratically through Vp here, linearly
        in the empty-``re_idx`` model-Ve branch).

        ``Rm`` is an upper-triangular factor whose ``Rm'Rm`` is block
        ``m_idx``'s precision after profiling out everything else —
        stack ``L·R1`` (data factor, random-inflated) on the non-random
        penalty square root, rotate block-m's columns last, unpivoted QR,
        bottom-right block. (Only ``Rm'Rm`` and sign-invariant quadratic
        forms in ``Rm`` are consumed, so a Cholesky stand-in for mgcv's
        fitting-QR ``b$R`` is exact here.)
        """
        p = self.p
        if m_idx in re_idx:
            raise ValueError("m_idx can't be in re_idx")
        sig2 = float(self.sigma_squared) if np.isfinite(self.sigma_squared) \
            and self.sigma_squared > 0 else 1.0
        a_m, b_m = self._block_col_ranges[m_idx]
        k_m = b_m - a_m
        # Per-penalty sp — mgcv reads ``b$full.sp`` when id linkage made
        # the working ``b$sp`` shorter (mgcv.r:3612); exp(ρ̂_full) is that.
        sp = np.exp(np.asarray(self._rho_hat, dtype=float))

        # R factor — mgcv's ``b$R``, consumed verbatim by recov
        # (mgcv.r:3624 ``rbind(b$R, …)``, :3656 ``b$R[, !rind]``). General
        # families: gam.fit5.post.proc's root, R'R = −lbb — exactly mgcv's
        # ``object$R`` on this path (cf. the testStat branch at
        # ``_smooth_significance_rows``). PIRLS: R'R = X'WX from stored
        # Fisher working weights; eigendecomp when XtWX is borderline-PSD
        # (gam.side rank-trim, near-singular weights, etc.).
        if getattr(self, "_md", None) is not None:
            R_factor = self._R_fit5
        else:
            if self._fisher_w is None:
                XtWX = self._XtX
            else:
                Xw = self._X_full * np.sqrt(self._fisher_w)[:, None]
                XtWX = Xw.T @ Xw
            try:
                R_factor = np.linalg.cholesky(XtWX).T
            except np.linalg.LinAlgError:
                ev, U = np.linalg.eigh(0.5 * (XtWX + XtWX.T))
                ev = np.clip(ev, 0.0, None)
                R_factor = (U * np.sqrt(ev)).T

        def _mroot_rows(S: np.ndarray) -> np.ndarray:
            """Rows B' of an eigen square root B·B' = S (rank rows)."""
            if S.size == 0:
                return np.zeros((0, S.shape[1] if S.ndim == 2 else 0))
            ev, U = np.linalg.eigh(0.5 * (S + S.T))
            max_ev = ev.max() if ev.size else 0.0
            keep = ev > max(max_ev, 0.0) * 1e-12
            if not keep.any():
                return np.zeros((0, S.shape[0]), dtype=float)
            return (U[:, keep] * np.sqrt(ev[keep])).T

        def _rm_from(data_rows: np.ndarray, pen_rows: np.ndarray,
                     a: int, b: int) -> np.ndarray:
            """Bottom-right block of the unpivoted QR after rotating cols
            ``a:b`` to the end — mgcv's ``qr.R(qr(LRB, tol=0))[ii, ii]``."""
            LRB = np.vstack([data_rows, pen_rows])
            n_cols = LRB.shape[1]
            target = list(range(a, b))
            other = [j for j in range(n_cols) if j < a or j >= b]
            LRB_perm = LRB[:, other + target]
            _, R_qr = np.linalg.qr(LRB_perm, mode="reduced")
            k = b - a
            return R_qr[-k:, -k:]

        if len(re_idx) == 0:
            # mgcv's re-empty branch: total penalty at λ̂, model Ve.
            S_lam = self._build_S_lambda(self._rho_hat)
            Rm = _rm_from(R_factor, _mroot_rows(S_lam), a_m, b_m)
            return 0.5 * (self.Ve + self.Ve.T) * v_scale, Rm

        # ---- split coefficients into fixed (1) / random (2) -------------
        rind = np.zeros(p, dtype=bool)
        for i in re_idx:
            a, b = self._block_col_ranges[i]
            rind[a:b] = True
        p2 = int(rind.sum())
        p1 = p - p2
        # map[j] = position of coefficient j within its (fixed|random)
        # subset. Blocks are wholly fixed or wholly random, so each
        # block's mapped indices stay contiguous.
        map_idx = np.zeros(p, dtype=int)
        map_idx[rind] = np.arange(p2)
        map_idx[~rind] = np.arange(p1)

        R1 = R_factor[:, ~rind]
        R2 = R_factor[:, rind]

        # Assemble S1 (fixed-part penalty, λ̂-weighted) and S2 (random).
        block_pos = {id(blk): i for i, blk in enumerate(self._blocks)}
        S1 = np.zeros((p1, p1))
        S2 = np.zeros((p2, p2))
        for k, slot in enumerate(self._slots):
            a, b = slot.col_start, slot.col_end
            s0 = int(map_idx[a])
            s1 = s0 + (b - a)
            if block_pos[id(slot.block)] in re_idx:
                S2[s0:s1, s0:s1] += sp[k] * slot.S
            else:
                S1[s0:s1, s0:s1] += sp[k] * slot.S

        # ---- B2 with B2'B2 = S2⁻ (mgcv's three pseudoinvert-root cases) --
        if p2 == 1:
            B2 = np.array([[1.0 / np.sqrt(S2[0, 0])]])
        elif float(np.max(np.abs(np.diag(np.diag(S2)) - S2))) == 0.0:
            # Exactly diagonal S2 (mgcv tests literal equality).
            ds2 = np.diag(S2).copy()
            ind = ds2 > ds2.max() * np.finfo(float).eps ** 0.8
            inv_d = np.zeros_like(ds2)
            inv_d[ind] = 1.0 / ds2[ind]
            B2 = np.diag(np.sqrt(inv_d))
        else:
            ev2, V2 = np.linalg.eigh(0.5 * (S2 + S2.T))
            ind = ev2 > ev2.max() * np.finfo(float).eps ** 0.8
            inv_v = np.zeros_like(ev2)
            inv_v[ind] = 1.0 / ev2[ind]
            B2 = np.sqrt(inv_v)[:, None] * V2.T

        # L'L = I + R2·S2⁻·R2' (R returns the upper factor from chol;
        # numpy's lower Cholesky transposed is the same matrix).
        M2 = B2 @ R2.T                              # (p2, p)
        A_L = np.eye(p) + M2.T @ M2
        L = np.linalg.cholesky(0.5 * (A_L + A_L.T)).T

        Rm = _rm_from(
            L @ R1, _mroot_rows(S1), int(map_idx[a_m]), int(map_idx[a_m]) + k_m,
        )

        G = L @ R_factor @ (self.Vp * v_scale)      # (p, p)
        Ve = (G.T @ G) / sig2
        return Ve, Rm

    def _re_test(
        self, m_idx: int, beta_b: np.ndarray, Vp_b: np.ndarray,
        v_scale: float = 1.0,
    ) -> tuple[float, float, float]:
        """Port of ``mgcv:::reTest`` (mgcv.r:3716-3755). Returns ``(stat,
        pval, rank)``. Uses ``psum_chisq`` (Davies 1980) for the p-value.

        Every *other* random-effect smooth in the model (mgcv's
        ``smooth$random == TRUE`` — set only by ``smooth.construct.re``)
        is treated as fully random via ``_recov``'s re-branch, so the test
        for one ``bs="re"`` term conditions correctly on its siblings.
        ``v_scale`` carries summary.gam's ``dispersion=`` covariance
        rescale into ``_recov`` (sig2 and the scale-estimated p-value
        branch stay on the OBJECT's values, exactly like mgcv's reTest,
        which reads ``b$sig2``/``b$scale.estimated`` untouched by the
        dispersion override).

        - Wood (2013) "On p-values for smooth components of an extended GAM",
          Biometrika 100(1), 221–228.
        - mgcv source: ``R/mgcv.r``.
        """
        sig2 = float(self.sigma_squared) if np.isfinite(self.sigma_squared) \
            and self.sigma_squared > 0 else 1.0
        re_idx = [
            i for i, blk in enumerate(self._blocks)
            if i != m_idx and blk.cls == "re.smooth.spec"
        ]
        Ve_full, Rm = self._recov(m_idx, re_idx, v_scale=v_scale)
        # Ve[ind, ind] half-square-root via eigendecomp.
        a_m, b_m = self._block_col_ranges[m_idx]
        Ve_b = Ve_full[a_m:b_m, a_m:b_m]
        ev_b, U_b = np.linalg.eigh(0.5 * (Ve_b + Ve_b.T))
        ev_b = np.clip(ev_b, 0.0, None)
        B = U_b * np.sqrt(ev_b)
        d = Rm @ beta_b
        stat = float((d * d).sum() / sig2)
        M = Rm @ B
        ev = np.linalg.eigvalsh(0.5 * ((M.T @ M) + (M.T @ M).T) / sig2)
        ev = np.clip(ev, 0.0, None)
        max_ev = ev.max() if ev.size else 0.0
        rank = int(np.sum(ev > max(max_ev, 0.0) * np.finfo(float).eps ** 0.8))
        if self._scale_known_fit:
            pval = psum_chisq(stat, ev) if ev.size else float("nan")
        else:
            k_df = max(1, int(round(self.df_residuals)))
            lb = np.concatenate([ev, np.array([-stat / k_df])])
            df = np.concatenate(
                [np.ones(ev.size, dtype=int), np.array([k_df], dtype=int)]
            )
            pval = psum_chisq(0.0, lb, df) if ev.size else float("nan")
        return stat, float(pval), float(rank)

    def _smooth_significance_rows(
        self, dispersion: float | None = None, re_test: bool = True,
    ) -> list[tuple[str, float, float, float, float]]:
        """Per-smooth test rows ``(label, edf, Ref.df, stat_col, p-value)``
        — the smooth half of mgcv's ``summary.gam`` (mgcv.r:4008-4040),
        shared by :meth:`summary` and ``anova(m)`` (mgcv's single-model
        ``anova.gam`` *is* ``summary.gam`` reclassed, so the tables must
        come from one place).

        Dispatch per smooth: ``reTest`` (Wood 2013) whenever the combined
        penalty is full-rank on the block (``null.space.dim == 0`` — re/fs/
        sz, cyclic bases, and every smooth under ``select=TRUE``);
        ``testStat`` (inverted-Nychka, fractional rank, ``_test_stat``)
        otherwise. ``stat_col`` is the printed statistic: ``Chi.sq`` (the
        raw stat) when ``est.disp`` is FALSE (known-scale family, or any
        family under a ``dispersion=`` override), ``F = stat / Ref.df``
        when the scale is estimated — same for both branches (mgcv prints
        ``chi.sq/df`` under ``est.disp``).

        ``dispersion`` is summary.gam's override (mgcv.r:3895-3899): the
        smooth tests always use ``Vp`` (freq= never reaches them), but a
        supplied dispersion rescales it by ``dispersion/sig2`` and forces
        ``est.disp = FALSE`` (χ² references; testStat ``res.df = -1``).

        ``re_test=False`` (summary.gam's ``re.test`` formal, mgcv.r:3858)
        drops the reTest-path smooths from the table entirely — mgcv sets
        ``res <- NULL`` and skips the row (mgcv.r:4024-4030); it does NOT
        reroute them to testStat.
        """
        scale_known = bool(self._scale_known_fit)
        est_disp = (not scale_known) and dispersion is None
        v_scale = (1.0 if dispersion is None
                   else float(dispersion) / float(self.sigma_squared))
        # mgcv tests against ``object$R`` — the R factor of the QR of
        # √W·X (Fisher working weights), so the statistic's inner product
        # is X'WX, not X'X. hea keeps n-row √W·X blocks instead of the
        # global R factor; ``qr`` inside ``_test_stat`` reduces either to
        # the same R'R = (X'WX)[block]. ``_fisher_w`` is None ↔ W = I
        # (Gaussian-identity), where weighting is a no-op. General-family
        # fits use gam.fit5.post.proc's R directly (R'R = −lbb), exactly
        # mgcv's object$R there.
        if getattr(self, "_md", None) is not None:
            X_w = self._R_fit5
        elif self._fisher_w is not None:
            sqw = np.sqrt(self._fisher_w)
            X_w = self._X_full * sqw[:, None]
        else:
            X_w = self._X_full
        rows: list[tuple[str, float, float, float, float]] = []
        for m_idx, (b, (a, bcol)) in enumerate(
            zip(self._blocks, self._block_col_ranges)
        ):
            beta_b = self._beta[a:bcol]
            Vp_b = self.Vp[a:bcol, a:bcol] * v_scale
            X_b = X_w[:, a:bcol]
            edf_b = float(self.edf[a:bcol].sum())
            edf1_b = (
                float(self.edf1[a:bcol].sum())
                if hasattr(self, "edf1") else edf_b
            )
            p_b = bcol - a
            if b.S:
                S_sum = b.S[0].copy()
                for S_i in b.S[1:]:
                    S_sum = S_sum + S_i
                rank_S = int(np.linalg.matrix_rank(S_sum))
                null_dim = p_b - rank_S
            else:
                null_dim = p_b
            if null_dim == 0:
                # reTest path — penalty is full-rank on the smooth's block.
                if not re_test:
                    continue
                stat, p_val, ref_df = self._re_test(m_idx, beta_b, Vp_b,
                                                    v_scale=v_scale)
            else:
                rank_in = float(min(p_b, edf1_b))
                # mgcv summary.gam: rdf <- residual.df if est.disp else -1.
                # res_df <= 0 in testStat means "scale fixed" (chi² ref).
                res_df = float(self.df_residuals) if est_disp else -1.0
                stat, p_val, ref_df = self._test_stat(
                    X_b, Vp_b, beta_b, rank_in, res_df=res_df,
                )
            col_stat = stat / max(ref_df, 1e-8) if est_disp else stat
            rows.append(
                (b.label, edf_b, float(ref_df), float(col_stat), float(p_val))
            )
        return rows

    def _compute_edf12(self, rho: np.ndarray, fit: "_FitState",
                       sigma_squared: float, A_inv: np.ndarray,
                       A_inv_XtWX: np.ndarray, edf: np.ndarray,
                       H_aug: np.ndarray | None):
        """mgcv's edf1 (frequentist tr(2F−F²) bound) and edf2 (sp-uncertainty
        corrected). Wood 2017 §6.11.3. Returns ``(edf2_per_coef, edf1_per_coef,
        Vc_correction)`` where ``Vc_correction = Vc1 + Vc2`` (the smoothing-
        parameter-uncertainty correction to ``Vp``). Caller adds it to ``Vp``
        to get mgcv's ``model$Vc`` (the ``unconditional=TRUE`` covariance).

        edf2 = diag((σ² A⁻¹ + Vc1 + Vc2) · X'WX) / σ², where

          - Vc1 = (∂β̂/∂ρ) · Vr · (∂β̂/∂ρ)ᵀ     (β̂'s ρ-dependence)
          - Vc2 = σ² Σ_{i,j} Vr[i,j] M_i M_j^T    (Cholesky-derivative bit)

        with M_k = ∂L^{-T}/∂ρ_k. Vr is the marginal covariance of ρ̂,
        taken as the top-left block of pinv(H_aug) (this equals the
        Schur complement of the augmented REML Hessian — same thing as
        inverting the profiled-σ² Hessian, mathematically). Falls back
        to the profiled Hessian when H_aug is unavailable (GCV / no
        smooths). For Gaussian + identity, dw/dρ vanishes so the Vc2
        formula above is the full mgcv expression — matches
        ``gam.fit3.post.proc``'s Vp + Vc1 + Vc2 decomposition.
        """
        F = A_inv_XtWX
        edf1 = 2.0 * np.diag(F) - np.einsum("ij,ji->i", F, F)
        p = F.shape[0]

        n_sp = len(self._slots)
        if n_sp == 0:
            return edf.copy(), edf1, np.zeros((p, p))

        db = self._db_drho(rho, fit.beta, fit.A_chol, fit.A_chol_lower)
        # Working-space chain: ∂β/∂θ = (∂β/∂ρ)·L — mgcv's post-proc does
        # exactly ``db.drho %*% L`` (gam.fit3.r:996-999). Vr is the working
        # θ covariance (H_aug is stored in working space).
        if self._L is not None:
            db = db @ self._L
        # Family-θ columns (tw): mgcv's db.drho spans the family
        # parameters too — Vc1 must propagate θ̂ uncertainty into β
        # (gam.fit3.r:1018: Vc uses the (θ,ρ) marginal of pinv(hess)).
        n_th_aug = getattr(self, "_n_theta_aug", 0)
        if n_th_aug > 0 and H_aug is not None:
            db = np.hstack([db, self._db_dtheta_fam(fit)])
            Vr = self._compute_Vr(rho, H_aug, with_theta=True)
        else:
            Vr = self._compute_Vr(rho, H_aug)
        # mgcv splits Vr by component: Vc1 uses pinv(H_aug) on positive
        # eigenspace; Vc2 uses (H_aug + 0.1·I)^{-1} — a weak prior on log
        # smoothing parameters (gam.fit3.post.proc line 1011). Without
        # this prior on Vc2, edf2 drifts ~1e-3 above mgcv.
        Vr_reg = self._compute_Vr(rho, H_aug, prior_var=0.1)

        # Fisher view once: Vc2's Cholesky seed AND the edf2 metric both
        # live in post.proc's √(object$weights)·X geometry — Fisher-type
        # weights for gam.fit3 and gam.fit4 alike (gam.fit4.r:798); see
        # _compute_Vc2's docstring for the B9 story.
        fit_F_v = self._fisher_view(fit)

        Vc1 = db @ Vr @ db.T
        Vc2 = self._compute_Vc2(rho, fit_F_v, Vr_reg, sigma_squared)

        # diag((σ²A_F⁻¹ + Vc1 + Vc2)·X'W_F X)/σ² = edf + diag((Vc1 + Vc2)·
        # X'W_F X)/σ². Fisher W_F to stay consistent with the edf metric
        # used at gam.fit3.r:644 (and with the Fisher A_inv_XtWX our caller
        # passes in). For Gaussian-identity W_F ≡ I and X'W_F X = X'X.
        # X'W_F X through the Fisher factor (X'WX = C'(K'K)C, K = √W·X·C⁻¹)
        # — the explicit product squares the condition number.
        W_F_view = fit_F_v.w
        if W_F_view is None or np.allclose(W_F_view, 1.0):
            Xw = self._X_full
        else:
            Xw = self._X_full * np.sqrt(np.maximum(W_F_view, 0.0))[:, None]
        C_v = (np.triu(fit_F_v.A_chol) if not fit_F_v.A_chol_lower
               else np.triu(fit_F_v.A_chol.T))
        Kw_v = solve_triangular(C_v, Xw.T, lower=False, trans="T").T
        XtWX = C_v.T @ (Kw_v.T @ Kw_v) @ C_v
        if sigma_squared > 0 and np.isfinite(sigma_squared):
            Vc_corr = Vc1 + Vc2
            edf2 = edf + np.einsum("ij,ij->i", Vc_corr, XtWX) / sigma_squared
        else:
            Vc_corr = np.zeros_like(Vc1)
            edf2 = edf.copy()

        # Total-sum cap only. mgcv's gam.fit3.post.proc deliberately does
        # not cap element-wise — individual edf2[i] can exceed edf1[i] as
        # long as the sum stays ≤ sum(edf1). Element-wise capping was a
        # bug in an earlier version here that pushed sum(edf2) below
        # sum(edf), the wrong direction for an sp-uncertainty correction.
        if edf2.sum() > edf1.sum():
            edf2 = edf1.copy()
        return edf2, edf1, Vc_corr

    def _db_dtheta_fam(self, fit):
        """Shim → free `_db_dtheta_fam`."""
        return _db_dtheta_fam(fit, X=self._X_full, family=self.family, y=self._y_arr, wt=self._wt)

    def _compute_Vr(self, rho: np.ndarray,
                    H_aug: np.ndarray | None,
                    prior_var: float | None = None,
                    with_theta: bool = False) -> np.ndarray:
        """Marginal covariance of ρ̂ — top-left ρρ block of inverse of H_aug.

        ``prior_var=None`` (default): pseudo-inverse with positive-eigenvalue
        projection — used for Vc1 and vcomp CIs. When H_aug is given, this
        is the Schur complement of the augmented Hessian; without it, invert
        the ρ-only profiled Hessian directly. Project onto the positive
        eigenspace before inverting (near sp bounds the surface is locally
        flat and tiny eigenvalues would blow up).

        ``prior_var > 0``: regularized inverse where eigenvalues are
        replaced by ``max(λ, 0) + prior_var`` before inverting — used for
        Vc2 to mirror mgcv's ``1/(d+1/10)`` prior on log smoothing
        parameters (gam.fit3.post.proc line 1011, "exp(4·var^.5) gives
        approx multiplicative range"). Without this, edf2 on bs='re' /
        nested-RE models drifts ~1e-3 above mgcv.
        """
        n_w = self._work_dim
        if H_aug is not None:
            # H_aug is stored in working space — (n_work + 1 [+ n_theta])²
            # with layout (ρ_working…, log φ[, θ_fam…]). ``with_theta``
            # returns the (ρ, θ) marginal (mgcv's rV[, 1:M] — everything
            # but the scale slot); default is the ρ block only.
            sel = np.arange(n_w)
            if with_theta:
                # θ_fam columns start after ρ and after the log φ slot —
                # which only exists when the scale is estimated.
                th_start = n_w + (0 if self._scale_known_fit else 1)
                if H_aug.shape[0] > th_start:
                    sel = np.concatenate([
                        sel, np.arange(th_start, H_aug.shape[0]),
                    ])
            w, V = np.linalg.eigh(H_aug)
            if prior_var is not None:
                d_reg = np.where(w > 0, w, 0.0) + float(prior_var)
                H_inv = (V / d_reg) @ V.T
                return H_inv[np.ix_(sel, sel)]
            w_max = float(w.max()) if w.size > 0 else 0.0
            keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
            if not keep.any():
                return np.zeros((sel.size, sel.size))
            Vk = V[:, keep]
            H_inv = (Vk / w[keep]) @ Vk.T
            return H_inv[np.ix_(sel, sel)]
        # GCV / no-H_aug fallback: ρρ block of the (ρ, log φ) joint Hessian
        # at log φ = 0, chained to working space (T'HT). For
        # Gaussian-identity REML this used to call the Gaussian-profiled
        # `_reml_hessian`; the joint Hessian's ρρ block equals 2× that
        # profiled Hessian up to the rank-1 Schur term, which is fine for
        # the GCV path (mgcv defines edf2 differently for GCV anyway —
        # this is a best-effort sp-uncertainty correction).
        H_full = 0.5 * self._reml_hessian(rho, 0.0, include_log_phi=False)
        if self._L is not None:
            H_full = self._L.T @ H_full @ self._L
        H = 0.5 * (H_full + H_full.T)
        w, V = np.linalg.eigh(H)
        if prior_var is not None:
            d_reg = np.where(w > 0, w, 0.0) + float(prior_var)
            return (V / d_reg) @ V.T
        w_max = float(w.max()) if w.size > 0 else 0.0
        keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
        if not keep.any():
            return np.zeros((n_w, n_w))
        Vk = V[:, keep]
        return (Vk / w[keep]) @ Vk.T

    def _compute_Vc2(self, rho, fit, Vr, sigma_squared):
        """Shim → free `_compute_Vc2`."""
        return _compute_Vc2(rho, fit, Vr, sigma_squared, L_pen=self._L, slots=self._slots, p=self.p)

    def _estimate_rank(self) -> int:
        """mgcv's fitting-rank estimate ``oo$rank.est`` — the
        identifiability check from ``gdiPK`` (gdi.c:1740-1758) on the
        augmented penalized problem.

        Stack the R factor of QR(√W·X) on the *balanced* penalty square
        root Eb (``totalPenaltySpace``, gam.fit3.r:2661:
        ``St = Σ_k S_k/‖S_k‖_F`` — smoothing-parameter independent, so
        identifiability doesn't drift with λ̂), each part divided by its
        own Frobenius norm so neither dominates; pivoted QR of the stack;
        then Cline-condition rank reduction at mgcv's
        ``rank.tol = √eps`` (gam.control default) via :func:`_R_rank`.
        """
        X = self._X_full
        if self._fisher_w is not None:
            Xw = X * np.sqrt(self._fisher_w)[:, None]
        else:
            Xw = X
        rank, _drop, _R = _pls_rank_drop(Xw, self._slots, self.p)
        return rank

    def _compute_vcomp(self, rescale: bool = True,
                       conf_lev: float = 0.95) -> pl.DataFrame:
        """Build the variance-component table mgcv calls ``gam.vcomp``.

        For each smoothing-param slot k, σ_k = σ/√sp_k is the implied
        random-effect std.dev (literal for ``bs='re'``; a parametrization
        for other smooths). CIs come from the delta method on
        log(σ_k) = ½(log σ² − ρ_k) using the joint REML Hessian wrt
        (ρ, log σ²) — only meaningful under REML, so for GCV we return
        point estimates with NaN bounds. Reuses the augmented Hessian
        cached on ``self._H_aug`` (set in ``__init__``).

        ``rescale=True`` (mgcv's default) first divides each sp by its
        penalty's ``S.scale`` — undoing smoothCon's ``scale.penalty``
        rescale so the std.dev's refer to the ORIGINAL penalty scale
        (mgcv.r:4242-4290). Rescaling multiplies σ_k by a constant, so
        the delta-method SEs of log σ_k (hence the CI ratios) are
        unchanged. ``rescale=False`` reports σ_k at the fitted scaling.
        """
        if conf_lev <= 0.0 or conf_lev >= 1.0:   # mgcv's guard
            conf_lev = 0.95
        n_sp = len(self._slots)
        scale_sd = float(self.sigma) if np.isfinite(self.sigma) else float("nan")

        if n_sp == 0:
            return pl.DataFrame({
                "name": ["scale"],
                "std_dev": [scale_sd],
                "lower": [float("nan")],
                "upper": [float("nan")],
            })

        names = [slot.block.label for slot in self._slots] + ["scale"]
        # Per-penalty sp (mgcv ``full.sp``); id-linked slots show the
        # shared value on each of their rows, like mgcv's vcomp output.
        sp_full = np.exp(np.asarray(self._rho_hat, dtype=float))
        if rescale:
            sp_full = sp_full / np.array(
                [slot.S_scale for slot in self._slots], dtype=float)
        sd2 = np.concatenate([
            self.sigma_squared / np.maximum(sp_full, 1e-300),
            [self.sigma_squared],
        ])
        log_sd = 0.5 * np.log(np.clip(sd2, 1e-300, None))
        sd = np.exp(log_sd)

        # GCV / point-estimate-only path: no Hessian-derived CIs.
        H = self._H_aug
        if H is None or self.method not in ("REML", "ML") or not np.isfinite(self.sigma_squared):
            nan_col = [float("nan")] * len(sd)
            return pl.DataFrame({
                "name": names, "std_dev": sd.tolist(),
                "lower": nan_col, "upper": nan_col,
            })

        # Pseudo-invert on the positive eigenspace, same threshold as edf2.
        # ``_H_aug`` lives in working (θ, log σ²) space — (n_work+1)².
        w, V = np.linalg.eigh(H)
        w_max = float(w.max()) if w.size > 0 else 0.0
        keep = (w > w_max * 1e-7) if w_max > 0 else np.zeros_like(w, dtype=bool)
        Hinv = np.zeros_like(H)
        if keep.any():
            Vk = V[:, keep]
            Hinv = (Vk / w[keep]) @ Vk.T

        # J: log(σ_k) = -0.5·ρ_k + 0.5·log σ² per slot k, with
        # ρ = L·θ — so the θ-block of row k is -0.5·L[k, :]; the scale
        # row is 0.5·log σ² only. Columns follow H_aug's layout
        # (ρ_working…, log σ²[, θ_fam…]); family-θ columns are zero (no
        # σ_k depends on θ directly — θ̂ uncertainty still enters through
        # Hinv's cross-covariances).
        n_work = self._work_dim
        J = np.zeros((n_sp + 1, H.shape[0]))
        if self._L is None:
            J[np.arange(n_sp), np.arange(n_sp)] = -0.5
        else:
            J[:n_sp, :n_work] = -0.5 * self._L
        # log σ² column exists only when the scale is estimated (H_aug is
        # ρ-only — or (ρ, θ_fam) for scat — when the scale is known,
        # matching mgcv's hess). Fixed scale ⇒ the +½·log σ² term carries
        # no uncertainty.
        if (not self._scale_known_fit) and H.shape[0] > n_work:
            J[:, n_work] = 0.5

        Vc = J @ Hinv @ J.T
        se = np.sqrt(np.maximum(np.diag(Vc), 0.0))
        z = float(_nmath.qnorm5(1.0 - (1.0 - conf_lev) / 2.0))
        lower = np.exp(log_sd - z * se)
        upper = np.exp(log_sd + z * se)
        return pl.DataFrame({
            "name": names,
            "std_dev": sd.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        })

    # -----------------------------------------------------------------------
    # Public post-fit API
    # -----------------------------------------------------------------------

    def _deviance_residuals(self, y, mu, wt) -> np.ndarray:
        """``sign(y - μ)·√(per-obs deviance)`` — mgcv's default residual.

        Families whose ``dev_resids`` returns ``−2logLik`` rather than a
        proper (≥0) deviance (betar & co) supply a ``residuals_extended``
        hook that folds in the saturated log-lik reference; without it the
        ``√(max(0, −2logLik))`` clamp would zero most residuals."""
        ext = getattr(self.family, "residuals_extended", None)
        if ext is not None:
            return np.asarray(ext(y, mu, wt, "deviance"), dtype=float)
        d_i = self.family.dev_resids(y, mu, wt)
        d_i = np.maximum(d_i, 0.0)            # FP cleanup near zero
        return np.sign(y - mu) * np.sqrt(d_i)

    def _family_residuals_kw(self) -> dict:
        """Extra keyword(s) for a family residuals hook: hooks may
        declare an optional ``prior_weights`` parameter (twlss's
        deviance residuals carry mgcv's ``object$prior.weights``,
        gamlss.r:2541); the engine passes the fit's prior weights when
        the parameter is declared."""
        import inspect
        try:
            params = inspect.signature(self.family.residuals).parameters
        except (TypeError, ValueError):
            return {}
        return ({"prior_weights": self._wt}
                if "prior_weights" in params else {})

    def residuals_of(self, type: str = "deviance") -> np.ndarray:
        """GLM residuals of the requested ``type``.

        Mirrors ``residuals.glm`` / ``residuals.gam`` in R.

        Parameters
        ----------
        type : {"deviance", "pearson", "scaled.pearson", "working", "response"}
            - ``"deviance"`` (default): ``sign(y-μ)·√(per-obs deviance)``.
            - ``"pearson"``: ``(y-μ)·√(wt / V(μ))``.
            - ``"scaled.pearson"``: pearson / √φ̂ (mgcv residuals.gam,
              mgcv.r:3457).
            - ``"working"``: ``(y-μ) · g'(μ)`` (η-scale residual).
            - ``"response"``: ``y - μ``.
        """
        # Family-supplied residuals take precedence (mgcv.r:3429) — the
        # general families (gaulss & co) define deviance/pearson/response
        # from their (n, n_lp) fitted matrix.
        fam_res = getattr(self.family, "residuals", None)
        if fam_res is not None:
            return np.asarray(
                fam_res(self._y_arr, self.fitted_values, type,
                        **self._family_residuals_kw()), dtype=float)
        if type not in ("deviance", "pearson", "scaled.pearson",
                        "working", "response"):
            raise ValueError(
                f"type must be one of 'deviance', 'pearson', "
                f"'scaled.pearson', 'working', 'response'; got {type!r}"
            )
        return self._residuals_for_y(self._y_arr, type)

    def _residuals_for_y(self, y, type: str) -> np.ndarray:
        """Residuals for an arbitrary response vector at the FITTED
        μ̂/weights — qq.gam's ``object$y <- yr`` substitution
        (plots.r:134/158) recomputes residuals exactly this way.
        Family-supplied residuals hooks (general families: gaulss & co,
        with their (n, n_lp) fitted matrix) take precedence exactly as
        in ``residuals_of`` — qq.gam's simulation path needs this for
        multi-LP rd hooks (gaulss rd, gamlss.r:1089)."""
        fam_res = getattr(self.family, "residuals", None)
        if fam_res is not None:
            return np.asarray(
                fam_res(y, self.fitted_values, type,
                        **self._family_residuals_kw()), dtype=float)
        mu = self.fitted_values
        wt = self._wt
        if type == "response":
            # Extended families whose fitted value is NOT the response mean
            # (ziP: fitted = log-Poisson-mean LP; ocat: latent LP) define the
            # response residual themselves (y − E(y) / y − class), mgcv-style.
            ext = getattr(self.family, "residuals_extended", None)
            if ext is not None:
                return np.asarray(ext(y, mu, wt, "response"), dtype=float)
            return y - mu
        if type == "deviance":
            return self._deviance_residuals(y, mu, wt)
        if type in ("pearson", "scaled.pearson"):
            # Variance-less families (gfam defines none, as in mgcv)
            # fail here — mgcv's pearson path errors the same way.
            V = self.family.variance(mu)
            res = (y - mu) * np.sqrt(wt / np.maximum(V, 0.0))
            if type == "scaled.pearson":
                res = res / np.sqrt(self.sigma_squared)
            return res
        # working: (y-μ) · g'(μ) = (y-μ) / (dμ/dη)
        eta = self.linear_predictors
        dmu_deta = self.family.link.mu_eta(eta)
        return (y - mu) / dmu_deta

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
        """Predict from the fitted GAM — :func:`predict.gam` parity.

        ``type='response'`` returns ``μ̂ = g⁻¹(X_new β̂ + offset)``;
        ``type='link'`` returns ``η̂ = X_new β̂ + offset``.

        Both return a ``pl.DataFrame`` with a ``fit`` column; with
        ``se_fit=True`` a second ``se.fit`` column is added. Link-scale SE
        is ``√diag(X · Vp · Xᵀ)`` (offset is constant so it doesn't affect
        SE); response-scale SE multiplies by ``|dμ/dη|`` (delta method,
        same as mgcv).

        ``type='terms'`` returns one column per model term (parametric
        terms by label, then smooths by label), each ``X_term · β̂_term``
        on the link scale; the intercept is never a column (mgcv attaches
        it as the ``"constant"`` attribute — here it's just ``coef[0]``)
        and the model offset is not included. ``se_fit=True`` appends
        ``se.{label}`` columns. ``type='iterms'`` is identical except
        constrained smooths' SEs include the uncertainty about the
        overall mean (mgcv's ``cmX`` construction); ``iterms_type=2``
        restricts that to the fixed-effects mean. ``iterms`` is not
        available for multi-formula fits (warns and falls back to
        ``terms``, like mgcv).

        ``terms=`` / ``exclude=`` (a label or list of labels) select model
        terms for ANY type, exactly like predict.gam: the columns of the
        prediction design belonging to de-selected terms are zeroed (for
        ``terms=`` the intercept is kept only if ``"(Intercept)"`` is
        listed), so link/response predictions become partial linear
        predictors; for ``type='terms'/'iterms'`` the output is
        additionally restricted to the requested columns — with mgcv's
        warn-and-ignore semantics when a requested label doesn't exist
        (the zeroing still applies; only the column selection is
        ignored).

        ``type='lpmatrix'`` returns the linear-predictor design matrix
        ``X_new`` as a raw ``np.ndarray`` — it's the SE building block, not
        a prediction, so the DataFrame wrapper would be misleading.
        ``se_fit=True`` is not allowed with ``type='lpmatrix'``.

        ``Vp`` is the Bayesian posterior covariance (``self.Vp``) — mgcv's
        default for ``se.fit`` since smoothing-parameter shrinkage makes the
        frequentist ``Ve`` over-confident at the posterior mode.
        ``unconditional=True`` uses the smoothing-parameter-uncertainty
        corrected ``self.Vc`` instead (predict.gam's ``unconditional``);
        for GCV fits the correction isn't available — mgcv warns and
        falls back to ``Vp``, and so does this.

        With ``newdata`` and a formula offset, the offset is re-evaluated
        against ``newdata`` (mirrors ``predict.gam``). Pass ``offset=`` to
        override or to add an offset on top of the formula offset.
        """
        if type not in ("link", "response", "lpmatrix", "terms", "iterms"):
            raise ValueError(
                "type must be 'link', 'response', 'lpmatrix', 'terms', or "
                f"'iterms'; got {type!r}"
            )
        if type == "lpmatrix" and se_fit:
            raise ValueError(
                "se_fit=True is not allowed with type='lpmatrix'"
            )
        if isinstance(terms, str):
            terms = [terms]
        if isinstance(exclude, str):
            exclude = [exclude]
        if getattr(self, "_md", None) is not None:
            return self._predict_general(newdata, type, se_fit, offset,
                                         unconditional, terms, exclude)

        if newdata is None:
            X_new = self._X_full
            off_new = self._offset
        else:
            from ..formula import (        # local to avoid cycle
                _apply_smooth_arg_exprs,
                _smooth_arg_expr_map,
                materialize,
                normalize_data,
            )

            # Accept the same dict / DataFrame input as the constructor
            # so matrix-arg smooths can replay on a {name: 2-D ndarray}.
            newdata = normalize_data(newdata)

            # Re-evaluate any smooth-arg expressions on newdata. e.g. if
            # the fit used ``s(I(b.depth^.5))``, the synthesised column
            # ``"I(b.depth^0.5)"`` must be present before the basis
            # evaluator asks for it. ``_smooth_arg_expr_map`` is
            # deterministic in ``self._expanded`` so we rebuild it here
            # rather than caching a copy on the model.
            expr_map = _smooth_arg_expr_map(self._expanded)
            if expr_map:
                newdata = _apply_smooth_arg_exprs(newdata, expr_map)

            # Append stub rows for any fit-time factor level that's
            # missing from ``newdata``. ``materialize``'s droplevels
            # semantics (formula.py:1059-1069) would otherwise collapse
            # the contrast to only the levels present in newdata,
            # returning a design with fewer columns than ``self._beta``.
            # This mirrors mgcv's ``xlevels`` mechanism, which carries
            # the fit-time levels through ``predict.gam`` so the
            # contrast expansion stays consistent. Stubs are appended
            # at the end; we slice them off after building ``X_new``.
            n_user = newdata.height
            newdata, n_stubs = _add_factor_stub_rows(newdata, self.data)

            X_param = materialize(self._expanded, newdata).to_numpy().astype(float)
            cols = [X_param]
            for b in self._blocks:
                if b.spec is None:
                    raise RuntimeError(
                        f"smooth block {b.label!r} (cls={b.cls!r}) has no "
                        "BasisSpec; predict(newdata=...) requires every smooth "
                        "to carry one."
                    )
                cols.append(np.asarray(b.spec.predict_mat(newdata), dtype=float))
            X_new = np.concatenate(cols, axis=1) if len(cols) > 1 else X_param
            if self._keep_cols is not None:
                # predict-time bases rebuild the full columns; apply the
                # identifiability drop so X_new matches the reduced β.
                X_new = X_new[:, self._keep_cols]
            if n_stubs > 0:
                X_new = X_new[:n_user]
            n_new = X_new.shape[0]
            # Re-evaluate any formula offset(...) atoms against newdata
            # — predict.gam does the same. Slice off the stubs so the
            # offset matches the user's row count.
            off_new = np.zeros(n_new)
            for off_node in self._expanded.offsets:
                blk = _eval_atom(off_node, newdata)
                off_full = blk.values.flatten().astype(float)
                if n_stubs > 0:
                    off_full = off_full[:n_user]
                off_new = off_new + off_full
        if offset is not None:
            extra = np.asarray(offset, dtype=float).flatten()
            if extra.shape != off_new.shape:
                raise ValueError(
                    f"offset must have length {off_new.shape[0]}, got {extra.shape}"
                )
            off_new = off_new + extra
        if terms is not None or exclude is not None or type in ("terms",
                                                                "iterms"):
            groups = self._term_column_groups()
        if terms is not None or exclude is not None:
            # predict.gam zeroes the de-selected terms' design columns for
            # every type (mgcv.r:2993-3026) — partial linear predictors.
            X_new = _zero_terms_exclude(X_new, terms, exclude, *groups)
        if type in ("terms", "iterms"):
            return self._terms_frame(
                X_new, self._beta, se_fit,
                self._predict_V(unconditional) if se_fit else None,
                type, iterms_type, terms, exclude, *groups)
        if type == "lpmatrix":
            return X_new
        eta = X_new @ self._beta + off_new

        # Extended-family `predict` hook (mgcv predict.gam, mgcv.r:3171-3198):
        # on type="response" an extended family that defines `predict` (ocat
        # returns the per-class probability matrix, not linkinv(η)) is called
        # with {X, beta, off, Vb} and its {"fit"[, "se_fit"]} used directly.
        fam_predict = getattr(self.family, "predict", None)
        if (type == "response" and fam_predict is not None
                and not isinstance(self.family, GeneralFamily)):
            Vb = self._predict_V(unconditional) if se_fit else None
            ffv = fam_predict(se=se_fit, X=X_new, beta=self._beta,
                              off=off_new, Vb=Vb, eta=None,
                              y=self._gfam_predict_y(newdata), lpi=None)
            return self._general_response_frame(
                ffv["fit"], ffv.get("se_fit") if se_fit else None)

        fit = eta if type == "link" else self.family.link.linkinv(eta)

        if not se_fit:
            return pl.DataFrame({"fit": fit})

        # Var(η̂_i) = X_i · V · X_iᵀ; rowwise via einsum.
        V = self._predict_V(unconditional)
        var_eta = np.einsum("ij,jk,ik->i", X_new, V, X_new)
        se_link = np.sqrt(np.maximum(var_eta, 0.0))
        if type == "link":
            return pl.DataFrame({"fit": fit, "se.fit": se_link})
        # Delta method: Var(μ̂) ≈ (dμ/dη)² · Var(η̂).
        mu_eta_v = self.family.link.mu_eta(eta)
        return pl.DataFrame({"fit": fit, "se.fit": np.abs(mu_eta_v) * se_link})

    def _gfam_predict_y(self, newdata) -> np.ndarray | None:
        """The ``y`` mgcv's predict.gam hands a family ``predict`` hook
        (mgcv.r:3174/3226): for gfam, the family-index vector — the
        second ``cbind`` arg evaluated on newdata (mgcv extracts the
        response from newdata by name, mgcv.r:2819), or the training
        index when predicting on the fit frame. ``None`` when
        unavailable — gfam then requires its stored fi to match the
        prediction length (gfam.r:490-492). Non-gfam families get
        ``None`` (ocat/ziP ignore y)."""
        if not isinstance(self.family, _gfam_family):
            return None
        if newdata is None:
            return self.data["_hea_gfam_fi"].to_numpy().astype(float)
        try:
            cols = set(newdata.columns)
            return (newdata.select(
                _eval_lhs_expr(self._gfam_fi_expr, cols).alias("_v"))
                ["_v"].to_numpy().astype(float))
        except Exception:
            return None

    def _predict_V(self, unconditional: bool) -> np.ndarray:
        """Covariance for prediction SEs: Vp, or Vc when unconditional
        (predict.gam's top-of-function swap; GCV fits warn and keep Vp)."""
        if not unconditional:
            return self.Vp
        if self.method in ("REML", "ML") or getattr(self, "_md", None) \
                is not None:
            return self.Vc
        import warnings as _w
        _w.warn(
            "smoothness-uncertainty corrected covariance not "
            "available for GCV fits; using Vp (mgcv predict.gam "
            "does the same)",
            stacklevel=2,
        )
        return self.Vp

    def _term_column_groups(self):
        """Term → fit-space design-column map for predict's terms machinery.

        Returns ``(plabels, pidx, icols, slabels, sranges)``: parametric
        term labels with their column index arrays (R ``assign`` grouping,
        the same map _pterms_rows tests by), the intercept's columns
        (``assign == 0``), and the smooth labels with their (start, end)
        column ranges. Multi-LP fits suffix LP j ≥ 1 labels with ``.{j}``
        (parametric, matching _pterms_rows; smooth labels already carry
        mgcv's ``s.1(…)`` form from construction).
        """
        md = getattr(self, "_md", None)
        plabels: list[str] = []
        pidx: list[np.ndarray] = []
        icols: list[int] = []
        if md is not None:
            for j, lp in enumerate(md.lps):
                asgn = np.asarray(lp.param_assign or [], dtype=int)
                pstart = int(md.pstart[j])
                icols.extend((pstart + np.flatnonzero(asgn == 0)).tolist())
                for i, t in enumerate(lp.expanded.terms, start=1):
                    plabels.append(t.label if j == 0 else f"{t.label}.{j}")
                    pidx.append(pstart + np.flatnonzero(asgn == i))
            slabels = [b.label for b in md.blocks]
            sranges = list(md.block_col_ranges)
        else:
            asgn = np.asarray(getattr(self, "_param_assign", []) or [],
                              dtype=int)
            if self._keep_cols is not None and asgn.size:
                asgn = asgn[self._keep_cols[:asgn.size]]
            icols.extend(np.flatnonzero(asgn == 0).tolist())
            for i, t in enumerate(self._expanded.terms, start=1):
                plabels.append(t.label)
                pidx.append(np.flatnonzero(asgn == i))
            slabels = [b.label for b in self._blocks]
            sranges = list(self._block_col_ranges)
        return (plabels, pidx, np.asarray(icols, dtype=int), slabels,
                sranges)

    def _terms_frame(self, X, beta, se_fit, V, type_, iterms_type,
                     terms, exclude, plabels, pidx, icols, slabels,
                     sranges):
        """type="terms"/"iterms" assembly (predict.gam mgcv.r:3041-3103 +
        the trailing terms=/exclude= column selection at 3257-3284).

        One link-scale column per term: ``X_term · β_term``; SEs from the
        term's Vp block. iterms widens constrained smooths' SEs by mgcv's
        cmX construction ("carry the intercept"): X1 = rowwise-repeated
        colMeans of the fit design with the smooth's own block patched in,
        se = √rowSums((X1·Vp)∘X1) over the FULL covariance.
        (mgcv's meanL1 rescaling for matrix-argument smooths with constant
        summation weights is not carried — that combination isn't pinned.)
        """
        beta = np.asarray(beta, dtype=float)
        fit_cols: dict[str, np.ndarray] = {}
        se_cols: dict[str, np.ndarray] = {}
        for lab, idx in zip(plabels, pidx):
            if idx.size == 0:
                continue
            Xi = X[:, idx]
            fit_cols[lab] = Xi @ beta[idx]
            if se_fit:
                v = np.einsum("ij,jk,ik->i", Xi, V[np.ix_(idx, idx)], Xi)
                se_cols[lab] = np.sqrt(np.maximum(v, 0.0))
        blocks = (self._md.blocks if getattr(self, "_md", None) is not None
                  else self._blocks)
        for (lab, (a, b)), blk in zip(zip(slabels, sranges), blocks):
            Xs = X[:, a:b]
            fit_cols[lab] = Xs @ beta[a:b]
            if not se_fit:
                continue
            constrained = (blk.spec is not None
                           and blk.spec.absorb is not None)
            if type_ == "iterms" and constrained:
                cmX = getattr(self, "_cmX", None)
                if cmX is None:
                    cmX = np.asarray(self._X_full, dtype=float).mean(axis=0)
                    self._cmX = cmX
                X1 = np.tile(cmX, (X.shape[0], 1))
                if iterms_type == 2:
                    X1[:, self.p_param:] = 0.0
                X1[:, a:b] = Xs
                v = np.einsum("ij,jk,ik->i", X1, V, X1)
            else:
                v = np.einsum("ij,jk,ik->i", Xs, V[a:b, a:b], Xs)
            se_cols[lab] = np.sqrt(np.maximum(v, 0.0))
        # Trailing column selection — mgcv's warn-and-ignore semantics.
        names = list(fit_cols.keys())
        import warnings as _w
        if terms is not None:
            if any(t not in names for t in terms):
                _w.warn("non-existent terms requested - ignoring",
                        stacklevel=3)
            else:
                names = list(terms)
        if exclude is not None:
            if any(e not in fit_cols for e in exclude):
                _w.warn("non-existent exclude terms requested - ignoring",
                        stacklevel=3)
            else:
                names = [n for n in names if n not in exclude]
        out = {n: fit_cols[n] for n in names}
        if se_fit:
            for n in names:
                out[f"se.{n}"] = se_cols[n]
        return pl.DataFrame(out)

    def _predict_general(self, newdata, type, se_fit, offset,
                         unconditional, terms=None, exclude=None):
        """Multi-LP predict (general families): mgcv's predict.gam
        returns an (n, n_lp) matrix per type — hea returns a DataFrame
        with one column per linear predictor, named ``fit``,
        ``fit.{j}`` (and ``se.fit``/``se.fit.{j}``), matching the
        ``.{j}`` coefficient-name suffix convention.

        ``type='lpmatrix'`` returns the stacked design (lpi available
        as ``m.lpi``). Per-LP formula ``offset(...)`` atoms are
        re-evaluated against newdata; the constructor-style ``offset=``
        override is not supported on this path.
        """
        if offset is not None:
            raise NotImplementedError(
                "offset= is not supported for multi-formula gam predict; "
                "put offset(...) atoms in the per-LP formulas."
            )
        md = self._md
        if newdata is None:
            X_new = np.asarray(md.X, dtype=float)
            offs = list(md.offsets)
        else:
            from ..formula import _eval_atom, normalize_data
            newdata_n = normalize_data(newdata)
            X_new, _ = _multi_lpmatrix(md, newdata_n)
            # drop_intercept families (cox.ph) rebuild the newdata design
            # with the intercept; remove the same column the fit dropped.
            if getattr(self, "_drop_intercept_col", None) is not None:
                X_new = np.delete(X_new, self._drop_intercept_col, axis=1)
            offs = []
            for lp in md.lps:
                if lp.expanded.offsets:
                    off_j = np.zeros(newdata_n.height)
                    for off_node in lp.expanded.offsets:
                        blk = _eval_atom(off_node, newdata_n)
                        off_j = off_j + blk.values.flatten().astype(float)
                    offs.append(off_j)
                else:
                    offs.append(None)
        if type == "iterms":
            import warnings as _w
            _w.warn("type iterms not available for multiple predictor "
                    "cases", stacklevel=3)
            type = "terms"
        if terms is not None or exclude is not None or type == "terms":
            groups = self._term_column_groups()
        if terms is not None or exclude is not None:
            X_new = _zero_terms_exclude(X_new, terms, exclude, *groups)
        if type == "terms":
            return self._terms_frame(
                X_new, self._beta, se_fit,
                self._predict_V(unconditional) if se_fit else None,
                "terms", None, terms, exclude, *groups)
        if type == "lpmatrix":
            return X_new

        K = len(md.lpi)
        beta = np.asarray(self._beta, dtype=float)
        V = self.Vp
        if unconditional:
            V = self.Vc        # general fits are always REML
        # mgcv's predict.gam dispatches response-scale prediction to the
        # family's own `predict` hook when it defines one (mgcv.r:3171-3198):
        # e.g. gammals returns (mean, σ) — its mean is e^{η₁}, not the
        # per-LP linkinv — with delta-method SEs, and a hook may emit a
        # different column count than n_lp. Link/terms scales never use it.
        fam_predict = getattr(self.family, "predict", None)
        if type == "response" and fam_predict is not None:
            # families whose response surface depends on the response
            # itself (cox.ph: the survivor function at the new event
            # times) get it via ``y``; in-sample uses the training y, and
            # newdata supplies it through the response column when present.
            if newdata is None:
                y_new = np.asarray(md.y, dtype=float)
            else:
                resp = self.formula[0].split("~", 1)[0].strip()
                y_new = (newdata_n[resp].to_numpy().astype(float)
                         if resp in newdata_n.columns else None)
            ffv = fam_predict(se=se_fit, X=X_new, beta=beta, off=offs,
                              Vb=V, lpi=md.lpi, y=y_new)
            return self._general_response_frame(
                ffv["fit"], ffv.get("se_fit") if se_fit else None)
        fits = []
        ses = []
        for j in range(K):
            cols = np.asarray(md.lpi[j], dtype=int)
            eta_j = X_new[:, cols] @ beta[cols]
            if offs[j] is not None:
                eta_j = eta_j + offs[j]
            if se_fit:
                Xj = X_new[:, cols]
                Vjj = V[np.ix_(cols, cols)]
                var_j = np.einsum("ij,jk,ik->i", Xj, Vjj, Xj)
                se_j = np.sqrt(np.maximum(var_j, 0.0))
            if type == "response":
                mu_j = self.family.links[j].linkinv(eta_j)
                fits.append(mu_j)
                if se_fit:
                    ses.append(np.abs(self.family.links[j].mu_eta(eta_j))
                               * se_j)
            else:
                fits.append(eta_j)
                if se_fit:
                    ses.append(se_j)
        return self._general_response_frame(
            np.column_stack(fits),
            np.column_stack(ses) if se_fit else None)

    @staticmethod
    def _general_response_frame(fit, se) -> "pl.DataFrame":
        """Pack a general-family prediction into hea's per-LP DataFrame
        (``fit``, ``fit.{j}`` + ``se.fit``/``se.fit.{j}``). Accepts a
        1-D or (n, c) ``fit`` — a ``family.predict`` hook may return a
        different column count than ``n_lp`` (mgcv.r:3180)."""
        fit = np.asarray(fit, dtype=float)
        if fit.ndim == 1:
            fit = fit[:, None]
        c = fit.shape[1]
        cols_out: dict = {}
        for j in range(c):
            cols_out["fit" if j == 0 else f"fit.{j}"] = fit[:, j]
        if se is not None:
            se = np.asarray(se, dtype=float)
            if se.ndim == 1:
                se = se[:, None]
            for j in range(c):
                cols_out["se.fit" if j == 0 else f"se.fit.{j}"] = se[:, j]
        return pl.DataFrame(cols_out)

    def vis(
        self,
        view: tuple[str, str] | list[str] | None = None,
        cond: dict | None = None,
        n_grid: int = 30,
        type: str = "link",
        se: bool = False,
        too_far: float = 0.0,
    ) -> "VisResult":
        """2D model-surface viewer — :func:`vis.gam` parity.

        Builds an ``n_grid × n_grid`` grid over two ``view`` covariates, holds
        every other variable at its "typical" value (median for numeric, mode
        for factor — same as mgcv's ``variable.summary``), calls
        :meth:`predict` on the grid, and returns the surface as a
        :class:`VisResult` (which carries a ``.plot()`` method).

        Parameters
        ----------
        view : tuple of 2 str, optional
            Pair of covariate names to vary. If ``None``, picks the first two
            variables in ``self.data`` that have more than one unique value.
        cond : dict, optional
            Override the typical-value default for any non-view variable, e.g.
            ``cond={"sex": "M", "age": 50}``.
        n_grid : int
            Grid resolution per axis (default 30, matching mgcv).
        type : {"link", "response"}
            Scale of the returned fit/SE — ``"link"`` is η̂, ``"response"``
            applies the inverse link.
        se : bool
            If ``True``, also compute pointwise SE on the grid.
        too_far : float
            Mask grid points whose normalized distance to any data point
            exceeds this threshold (replaces fit/se with ``NaN``). 0 = no
            masking. Mirrors mgcv's ``exclude.too.far``.
        """
        if type not in ("link", "response"):
            raise ValueError(
                f"type must be 'link' or 'response'; got {type!r}"
            )

        vs = self._var_summary()

        if view is None:
            view = []
            # Iterate RHS variables in formula order (vs is built that way) —
            # mgcv's vis.gam picks the first two with variation, same idea.
            for name in vs:
                if _has_variation(self.data[name]):
                    view.append(name)
                    if len(view) == 2:
                        break
            if len(view) < 2:
                raise ValueError(
                    "could not auto-pick `view`: need at least two RHS "
                    "variables with more than one unique value"
                )
        else:
            view = list(view)
            if len(view) != 2:
                raise ValueError(
                    f"view must be a pair of variable names; got {view!r}"
                )
            for v in view:
                if v not in self.data.columns:
                    raise ValueError(
                        f"view variable {v!r} not in data; available: "
                        f"{list(self.data.columns)}"
                    )

        m1 = _grid_axis(self.data[view[0]], n_grid)
        m2 = _grid_axis(self.data[view[1]], n_grid)
        n1, n2 = len(m1), len(m2)

        # meshgrid with indexing='ij' so that reshape(n1, n2) puts m1 on axis 0
        # and m2 on axis 1 — i.e. fit[i, j] is the prediction at (m1[i], m2[j]).
        M1, M2 = np.meshgrid(m1, m2, indexing="ij")
        v1 = M1.ravel()
        v2 = M2.ravel()
        n_pts = n1 * n2

        cond = dict(cond or {})
        cols: dict[str, object] = {}
        for name in self.data.columns:
            if name == view[0]:
                cols[name] = v1
            elif name == view[1]:
                cols[name] = v2
            elif name in cond:
                cols[name] = np.repeat(cond[name], n_pts)
            elif name in vs:
                cols[name] = np.repeat(vs[name], n_pts)
            else:
                # Variable wasn't profiled by var_summary (e.g. an offset column
                # or a non-formula column) — leave it out; predict only
                # references columns named in the formula.
                continue

        # Re-impose the original schema (factor levels, dtypes) so PredictMat's
        # factor matching still works on the grid frame.
        new_df = pl.DataFrame(
            {
                k: (v if isinstance(v, pl.Series) else pl.Series(k, v))
                for k, v in cols.items()
            }
        )
        for name in new_df.columns:
            src = self.data[name]
            if src.dtype != new_df[name].dtype:
                new_df = new_df.with_columns(new_df[name].cast(src.dtype))

        if se:
            pred_df = self.predict(new_df, type=type, se_fit=True)
            fit = pred_df["fit"].to_numpy()
            se_arr = pred_df["se.fit"].to_numpy()
        else:
            fit = self.predict(new_df, type=type, se_fit=False)["fit"].to_numpy()
            se_arr = None

        if too_far > 0.0:
            mask = _too_far_mask(
                v1, v2, self.data[view[0]], self.data[view[1]], too_far
            )
            fit = np.array(fit, dtype=float, copy=True)
            fit[mask] = np.nan
            if se_arr is not None:
                se_arr = np.array(se_arr, dtype=float, copy=True)
                se_arr[mask] = np.nan

        fit_grid = np.asarray(fit, dtype=float).reshape(n1, n2)
        se_grid = (
            np.asarray(se_arr, dtype=float).reshape(n1, n2)
            if se_arr is not None
            else None
        )
        return VisResult(
            view=(view[0], view[1]),
            m1=np.asarray(m1),
            m2=np.asarray(m2),
            fit=fit_grid,
            se=se_grid,
            type=type,
        )

    def get_difference(
        self,
        comp: dict,
        cond: dict | None = None,
        rm_ranef: bool | str | list | None = True,
        se: bool = True,
        f: float = 1.96,
        sim_ci: bool = False,
        n_sim: int = 10_000,
        rng: np.random.Generator | int | None = None,
        print_summary: bool = False,
    ) -> "DiffResult":
        """Estimate the difference between two conditions of a fitted GAM —
        :func:`itsadug::get_difference` parity. The numerical engine behind
        :meth:`plot_diff`; call this directly if you want the difference table
        without plotting.

        Builds two prediction grids that differ only in ``comp``, takes the
        link-scale design-matrix difference ``X1 − X2``, and returns
        ``(X1 − X2) β̂`` together with pointwise and (optionally)
        simultaneous confidence bands.

        Parameters
        ----------
        comp : dict
            ``{predictor: (level_a, level_b)}``. The difference is fit-at-A
            minus fit-at-B. itsadug allows ≥ 2 levels and silently keeps the
            first two — same here, with a warning.
        cond : dict, optional
            Other variables held at user-specified values. Length-1 entries
            broadcast across the grid; length-N entries (e.g. the x-axis
            covariate inside :meth:`plot_diff`) define the grid axis. Any
            variables not in ``comp`` or ``cond`` are held at the typical
            value (median for numeric, mode for factor — same as mgcv's
            ``variable.summary``). Variables that overlap with ``comp``
            keys are dropped from ``cond`` with a warning.
        rm_ranef : bool, str, list of str, or None
            Smooth labels whose columns are zeroed in the design matrix
            before computing the difference. ``True`` (default, matching
            itsadug) zeros every smooth with ``null.space.dim == 0`` —
            ``bs="re"`` random effects, ``bs="fs"`` factor smooths, ``bs=
            "sz"`` sum-to-zero interactions. ``False``/``None`` zeros
            nothing. A string or list selects by label-substring AND
            null-space-0 (intersection — itsadug's two-pass grep). Note:
            until GAMM support lands, models won't carry null-space-0
            smooths so ``rm_ranef=True`` is a no-op.
        se : bool
            Compute pointwise CI half-width ``f · √diag((X1−X2) Vp (X1−X2)ᵀ)``.
        f : float
            SE multiplier for the pointwise CI. ``1.96`` ≈ 95%, ``2.58`` ≈ 99%.
            Also drives the ``sim_ci`` envelope's coverage probability via
            ``prob = 1 − round(2·(1 − Φ(f)), 2)`` (itsadug's exact rule —
            the ``round(·, 2)`` snaps 1.96 → 0.95, 2.58 → 0.99).
        sim_ci : bool
            Add a simultaneous CI envelope (Wood 2017 §6.10). Uses
            ``self.Vc`` (mgcv's ``unconditional=TRUE`` covariance) for the
            posterior draws. ``n.grid`` is bumped to ≥ 200 by
            :meth:`plot_diff`; this method itself trusts the caller.
        n_sim : int
            Number of MVN draws for the simultaneous envelope. Default
            10,000, matching itsadug.
        rng : int | RMersenneTwister | RGenerator | numpy Generator | None
            RNG for the simultaneous ``mgcv::rmvn`` draws. ``None`` uses the
            process-global R stream (set via :func:`hea.R.set_seed`, as in
            R+itsadug where ``set.seed()`` precedes the draws). An int seeds a
            fresh R Mersenne-Twister, reproducing R's ``set.seed(int)`` draws.
            A numpy ``Generator`` is also accepted but is *not* R-consistent.
        print_summary : bool
            Print a per-variable summary of the conditions used (mirror of
            itsadug's ``print.summary``).
        """
        if not isinstance(comp, dict) or len(comp) == 0:
            raise ValueError(
                "comp must be a non-empty dict, e.g. comp={'Group': ('A', 'B')}"
            )
        cond = dict(cond) if cond else {}

        # --- comp validation -------------------------------------------------
        cols_data = self.data.columns
        bad = [k for k in comp if k not in cols_data]
        if bad:
            raise ValueError(
                f"Grouping predictor(s) not found in model: {', '.join(bad)}"
            )
        for k, v in list(comp.items()):
            if not hasattr(v, "__len__") or len(v) < 2:
                raise ValueError(
                    f"Provide two levels for {k!r} to calculate the difference."
                )
            if len(v) > 2:
                import warnings as _w
                _w.warn(
                    f"More than two levels provided for predictor {k!r}. "
                    "Only first two levels are being used.",
                    stacklevel=2,
                )

        # cond keys overlapping with comp keys: drop from cond with a warning
        # (itsadug warns and drops; the comp value wins).
        for k in [k for k in cond if k in comp]:
            import warnings as _w
            _w.warn(
                f"Predictor {k!r} specified in comp and cond. "
                "(The value in cond will be ignored.)",
                stacklevel=2,
            )
            cond.pop(k)

        # --- build the two grids ---------------------------------------------
        su = self._var_summary()
        # mgcv's variable.summary ranges over RHS-of-formula variables. For
        # any RHS variable not in comp and not in cond, use the typical value
        # (mode for factor, median for numeric). Variables in cond may be
        # length-N (the x-axis grid that plot_diff prefills) — keep as-is
        # in both grids.
        new_cond1: dict[str, object] = {}
        new_cond2: dict[str, object] = {}
        for var in su:
            if var in comp:
                v = comp[var]
                new_cond1[var] = [v[0]]
                new_cond2[var] = [v[1]]
            elif var in cond:
                vals = cond[var]
                if not hasattr(vals, "__len__") or isinstance(vals, str):
                    vals = [vals]
                new_cond1[var] = list(vals)
                new_cond2[var] = list(vals)
            else:
                typ = su[var]
                new_cond1[var] = [typ]
                new_cond2[var] = [typ]
        # Also honor cond entries for variables outside var.summary (defensive
        # — su covers RHS-of-formula vars; user could pass an extra column
        # name that's still referenced through some indirection).
        for var in cond:
            if var not in new_cond1:
                vals = cond[var]
                if not hasattr(vals, "__len__") or isinstance(vals, str):
                    vals = [vals]
                new_cond1[var] = list(vals)
                new_cond2[var] = list(vals)

        newd1 = _expand_grid(new_cond1)
        newd2 = _expand_grid(new_cond2)
        # Preserve schema from self.data so factor levels and dtypes match
        # what predict() expects.
        newd1 = _coerce_schema(newd1, self.data)
        newd2 = _coerce_schema(newd2, self.data)

        # --- lpmatrices ------------------------------------------------------
        # Predict on a single combined frame to dodge a known limitation:
        # ``materialize`` drops absent factor levels from new data (R's
        # ``droplevels`` semantics — fine at fit time, wrong at predict
        # time, since mgcv stores ``model$xlevels`` and we don't yet). By
        # stacking newd1 + newd2 the comp variables regain both levels in
        # one frame; stub rows then top up any non-comp factor still
        # missing source levels (e.g. sex='F' when 'M' is the mode).
        n1 = newd1.height
        combined = pl.concat([newd1, newd2], how="vertical_relaxed")
        combined, n_stubs = _add_factor_stub_rows(combined, self.data)
        P = np.asarray(self.predict(combined, type="lpmatrix"), dtype=float)
        if n_stubs > 0:
            P = P[:-n_stubs]
        p1 = P[:n1]
        p2 = P[n1:]

        # --- rm.ranef --------------------------------------------------------
        # itsadug treats rm_ranef==False the same as None (no removal).
        if rm_ranef is False:
            rm_ranef = None
        cancelled: list[str] = []
        if rm_ranef is not None:
            # null-space-dim==0 smooths in our codebase: re/fs/sz, mirroring
            # mgcv's bs="re"/"fs"/"sz" (these are the fully penalized,
            # "random-effect-like" smooths).
            ns0_classes = ("re.smooth.spec", "fs.interaction", "sz.interaction")
            ns0_blocks = [
                (b, rng_)
                for b, rng_ in zip(self._blocks, self._block_col_ranges)
                if b.cls in ns0_classes
            ]
            if rm_ranef is True:
                target_labels = [b.label for b, _ in ns0_blocks]
            else:
                if isinstance(rm_ranef, str):
                    rm_ranef_list = [rm_ranef]
                else:
                    rm_ranef_list = list(rm_ranef)
                # itsadug's two-pass grep: keep blocks that are null-space-0
                # AND whose label contains a user-supplied substring.
                target_labels = [
                    b.label
                    for b, _ in ns0_blocks
                    if any(s in b.label for s in rm_ranef_list)
                ]
            for b, (a, bcol) in ns0_blocks:
                if b.label in target_labels:
                    p1[:, a:bcol] = 0.0
                    p2[:, a:bcol] = 0.0
                    cancelled.append(b.label)

        # --- difference + CI -------------------------------------------------
        p = p1 - p2
        diff = p @ self._beta
        ci = None
        if se:
            # √diag(p · Vp · pᵀ) — rowSums((p @ Vp) * p) is the same thing,
            # and is what itsadug writes literally. The einsum is faster.
            var_diff = np.einsum("ij,jk,ik->i", p, self.Vp, p)
            ci = f * np.sqrt(np.maximum(var_diff, 0.0))

        # --- simultaneous CI (Wood 2017 §6.10 / Marra & Wood 2012) ----------
        sim_ci_arr = None
        crit_val = None
        if sim_ci:
            Vb = self.Vc  # unconditional=TRUE covariance
            var_fit = np.einsum("ij,jk,ik->i", p, Vb, p)
            se_fit = np.sqrt(np.maximum(var_fit, 0.0))
            rng_obj = _resolve_sim_rng(rng)
            # itsadug draws MVN(0, Vb) via mgcv::rmvn (a pivoted-Cholesky root
            # on R's MT stream). RGenerator routes that through hea's bit-exact
            # R RNG, so set_seed()/an int seed reproduce R+itsadug's draws to
            # machine precision (only the final GEMM is BLAS-bound). n_sim
            # defaults to 10000, matching itsadug.
            mu0 = np.zeros(Vb.shape[0])
            sim = rng_obj.multivariate_normal(mu0, Vb, size=n_sim)
            # simDev[i, s] = (p · sim[s])[i] — deviation at grid point i for
            # draw s. Standardize by se_fit, take row-wise max, then quantile.
            simDev = p @ sim.T
            absDev = np.abs(simDev / se_fit[:, None])
            masd = absDev.max(axis=0)
            # itsadug's exact prob: 1 − round(2·(1 − Φ(f)), 2). For f=1.96
            # → 0.95; f=2.58 → 0.99. Using R's type-8 quantile (Hyndman-Fan)
            # via numpy's "median_unbiased" method (equivalent).
            prob = 1.0 - round(2.0 * (1.0 - float(_nmath.pnorm5(f))), 2)
            crit_val = float(np.quantile(masd, prob, method="median_unbiased"))
            sim_ci_arr = crit_val * se_fit

        # --- print summary ---------------------------------------------------
        if print_summary:
            print(_format_difference_summary(
                comp=comp, cond=cond, su=su, cancelled=cancelled,
                rm_ranef=rm_ranef, sim_ci=sim_ci, f=f,
            ))

        # --- comp label string ----------------------------------------------
        levels1 = ".".join(str(comp[k][0]) for k in comp)
        levels2 = ".".join(str(comp[k][1]) for k in comp)
        comp_label = f"{', '.join(f'{k}={tuple(v)[:2]}' for k, v in comp.items())}"

        # Output grid: drop comp columns (itsadug does this in the data.frame
        # output — comp is logged separately, not in the per-row table).
        grid_out = newd1.drop(*[c for c in comp if c in newd1.columns])

        return DiffResult(
            xvar=None,
            grid=grid_out,
            difference=diff,
            f=f if se else None,
            ci=ci,
            sim_ci=sim_ci_arr,
            crit=crit_val,
            comp_label=comp_label,
            levels=(levels1, levels2),
            rm_ranef_cancelled=cancelled,
        )

    def plot_diff(
        self,
        view: str,
        comp: dict,
        cond: dict | None = None,
        se: float = 1.96,
        sim_ci: bool = False,
        n_grid: int = 100,
        rm_ranef: bool | str | list | None = True,
        mark_diff: bool = True,
        col: str = "black",
        col_diff: str = "red",
        transform_view=None,
        n_sim: int = 10_000,
        rng: np.random.Generator | int | None = None,
        print_summary: bool = False,
        ax=None,
        figsize: tuple | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        xlab: str | None = None,
        ylab: str | None = None,
        title: str | None = None,
        hide_label: bool = False,
        shade: bool = True,
        alpha: float = 0.25,
    ):
        """Plot the predicted difference between two conditions —
        :func:`itsadug::plot_diff` parity.

        Builds an n_grid grid over ``view``, calls :meth:`get_difference` to
        get the link-scale ``(X1 − X2) β̂`` curve plus its CI, plots the
        curve with a CI band, and (when ``mark_diff=True``) overlays the
        x-windows where the band excludes zero.

        Parameters
        ----------
        view : str
            Name of the x-axis covariate. The grid is
            ``np.linspace(min, max, n_grid)`` over the data column (NaNs
            dropped). itsadug only takes the first element if ``view`` is
            a vector and warns; the 2D analogue is :meth:`plot_diff2`.
        comp : dict
            Same as :meth:`get_difference`: ``{predictor: (level_a, level_b)}``.
        cond : dict, optional
            Other variables to hold fixed. If ``view`` is included here,
            ``cond[view]`` overrides the auto-built grid (with a warning),
            matching itsadug's behavior.
        se : float
            SE multiplier for the pointwise CI band. ``> 0`` draws the band;
            ``≤ 0`` plots only the curve. Default ``1.96`` (≈ 95% pointwise).
        sim_ci : bool
            Use the simultaneous-CI envelope (Wood 2017 §6.10) instead of
            the pointwise band for the visual band and the
            ``mark_diff`` window detection. itsadug bumps ``n_grid`` to
            at least 200 when this is on — same here.
        n_grid : int
            Grid resolution. Bumped to ≥ 200 if ``sim_ci=True`` (matches
            itsadug — fewer points underestimate the simultaneous critical
            value).
        rm_ranef, n_sim, rng, print_summary : passed through.
        mark_diff : bool
            Shade the x-windows where the CI excludes 0 with vertical dotted
            guides + a top-of-axis tick (matching itsadug's
            ``addInterval`` + ``abline`` combo).
        col, col_diff, transform_view, xlim, ylim, xlab, ylab, title,
        hide_label, shade, alpha : visual knobs.

        Returns the matplotlib ``Axes``.
        """
        # itsadug bumps to 200 for adequate sim-ci precision. Same here.
        if sim_ci:
            n_grid = max(n_grid, 200)

        if view not in self.data.columns:
            raise ValueError(
                f"view variable {view!r} not in data; available: "
                f"{list(self.data.columns)}"
            )
        cond = dict(cond) if cond else {}

        # Build the x-axis grid. If view is in cond, itsadug warns and uses
        # cond's values, ignoring the auto-built linspace. Mirror that.
        if view in cond:
            import warnings as _w
            _w.warn(
                f"Predictor {view!r} specified in view and cond. Values in "
                f"cond being used, rather than the whole range of {view!r}.",
                stacklevel=2,
            )
        else:
            col_view = self.data[view].drop_nulls().to_numpy().astype(float)
            if col_view.size == 0:
                raise ValueError(
                    f"view variable {view!r} has no non-null values"
                )
            cond[view] = np.linspace(col_view.min(), col_view.max(), n_grid)
        if xlim is not None:
            if len(xlim) != 2:
                import warnings as _w
                _w.warn(
                    "Invalid xlim values specified. Argument xlim is being ignored.",
                    stacklevel=2,
                )
            else:
                cond[view] = np.linspace(xlim[0], xlim[1], n_grid)

        result = self.get_difference(
            comp=comp, cond=cond, rm_ranef=rm_ranef,
            se=(se > 0), f=(se if se > 0 else 1.96),
            sim_ci=sim_ci, n_sim=n_sim, rng=rng,
            print_summary=print_summary,
        )
        result.xvar = view

        # Optional x-axis transform — itsadug applies `transform.view` to the
        # x values before plotting (for log-scaling, etc.).
        x = np.asarray(result.grid[view].to_numpy(), dtype=float).copy()
        if transform_view is not None:
            try:
                x = np.asarray([transform_view(xi) for xi in x], dtype=float)
            except Exception as exc:
                raise RuntimeError(
                    "Error: the function specified in transform_view cannot be "
                    "applied to x-values, because infinite or missing values "
                    "are not allowed."
                ) from exc
            if not np.all(np.isfinite(x)):
                raise RuntimeError(
                    "Error: the function specified in transform_view cannot be "
                    "applied to x-values, because infinite or missing values "
                    "are not allowed."
                )

        # --- plotting --------------------------------------------------------
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize or (6, 4))

        diff = result.difference
        band = result.sim_ci if (sim_ci and result.sim_ci is not None) else result.ci

        if se > 0 and band is not None and shade:
            ax.fill_between(x, diff - band, diff + band, color=col,
                            alpha=alpha, linewidth=0)
        ax.plot(x, diff, color=col, linewidth=1.5)
        # h=0 reference line — itsadug's `par[["h0"]] <- 0` default.
        ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="-")

        # mark.diff: shade x-windows where the band excludes 0
        regions = result.regions(use_sim_ci=sim_ci) if mark_diff and band is not None else None
        if regions:
            ymin, ymax = ax.get_ylim()
            for (start, end) in regions:
                ax.axvline(start, color=col_diff, linestyle=":", linewidth=1)
                ax.axvline(end, color=col_diff, linestyle=":", linewidth=1)
            # Bottom-of-axis tick bars — itsadug uses
            # ``addInterval(pos=getFigCoords("p")[3], ...)`` and in R's
            # ``c(xleft, xright, ybottom, ytop)`` convention index 3 is
            # ``ybottom``, so the bar sits *along the x-axis*, not at the top.
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for (start, end) in regions:
                ax.plot([start, end], [0.0, 0.0], transform=trans,
                        color=col_diff, linewidth=2.0,
                        clip_on=False, solid_capstyle="butt")

        if title is None:
            title = f"Difference {result.levels[0]} − {result.levels[1]}"
        ax.set_title(title)
        # mgcv stores the response name on the LHS of the formula; pull
        # the formula's lhs for the y-label like itsadug does.
        lhs = self.formula.split("~", 1)[0].strip()
        if ylab is None:
            ylab = f"Est. difference in {lhs}"
        if xlab is None:
            xlab = view
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim)

        if not hide_label:
            label = "difference"
            if rm_ranef not in (None, False) and result.rm_ranef_cancelled:
                label += ", excl. random"
            if sim_ci:
                label += ", simult.CI"
            # itsadug uses ``mtext(side=4)`` (vertical, right margin); we
            # anchor inside the axes' top-right so the label coexists with
            # ``set_title`` and a tight subplot grid without overlap.
            ax.text(0.99, 0.985, label, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#595959")

        if print_summary:
            if regions:
                print(f"\n{view} window(s) of significant difference(s):")
                for (s, e) in regions:
                    print(f"\t{s:f} - {e:f}")
            else:
                print("\nDifference is not significant.")

        return ax

    def plot_diff2(
        self,
        view: tuple[str, str] | list[str],
        comp: dict,
        cond: dict | None = None,
        se: float = 1.96,
        n_grid: int = 30,
        rm_ranef: bool | str | list | None = True,
        plot_ci: bool = False,
        sim_ci: bool = False,
        n_sim: int = 10_000,
        rng: np.random.Generator | int | None = None,
        show_diff: bool = False,
        alpha_diff: float = 0.5,
        col_diff: str = "white",
        color: str = "RdBu_r",
        col: str = "black",
        ci_col: tuple[str, str] = ("red", "green"),
        n_levels: int = 10,
        too_far: float = 0.0,
        print_summary: bool = False,
        ax=None,
        figsize: tuple | None = None,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        zlim: tuple | None = None,
        xlab: str | None = None,
        ylab: str | None = None,
        title: str | None = None,
        hide_label: bool = False,
        add_color_legend: bool = True,
    ):
        """Plot the predicted difference *surface* between two conditions —
        :func:`itsadug::plot_diff2` parity.

        2D analogue of :meth:`plot_diff`: builds an ``n_grid × n_grid``
        grid over ``view``, calls :meth:`get_difference` over the joint
        grid, and renders ``(X1 − X2) β̂`` as a colored heatmap with
        overlaid contour lines. With ``plot_ci=True``, dotted contours of
        ``diff − CI`` (``ci_col[0]``) and ``diff + CI`` (``ci_col[1]``)
        are drawn at the same level set — itsadug's ``plotCI=TRUE`` style.

        Parameters
        ----------
        view : (str, str)
            Pair of x- and y-axis covariate names. Both must be numeric in
            the data. itsadug silently uses only the first two if more are
            given; we mirror that with a warning.
        comp : dict
            Same as :meth:`get_difference`: ``{predictor: (level_a, level_b)}``.
        cond : dict, optional
            Other variables to hold fixed. If a ``view`` name is included
            here, ``cond[v]`` overrides the auto-built linspace (with a
            warning), matching itsadug.
        se : float
            SE multiplier for the pointwise CI used by ``plot_ci``. Default
            ``1.96`` (≈ 95% pointwise). ``≤ 0`` disables CI computation;
            ``plot_ci`` is silently a no-op in that case.
        n_grid : int
            Per-axis grid resolution. Default 30 (matches itsadug).
        rm_ranef : bool, str, list of str, or None
            Same as :meth:`get_difference`.
        plot_ci : bool
            Overlay dotted contours of ``diff ± CI`` at the same levels as
            the bold contour — itsadug's ``plotCI``. Default ``False``.
        sim_ci : bool
            Use a simultaneous CI envelope (Wood 2017 §6.10) instead of the
            pointwise band — itsadug's ``sim.ci``. Wider, controls the
            family-wise type-I rate over the surface. Both ``plot_ci`` and
            the ``show_diff`` mask read this band when ``sim_ci=True``.
        n_sim : int
            Number of MVN draws for the simultaneous envelope. Default
            10,000 (matches itsadug). Ignored when ``sim_ci=False``.
        rng : int | RMersenneTwister | RGenerator | numpy Generator | None
            RNG for the simultaneous ``mgcv::rmvn`` draws (see
            :meth:`get_difference`). ``None`` uses the :func:`hea.R.set_seed`
            global R stream; an int seeds a fresh R Mersenne-Twister.
        show_diff : bool
            Overlay a translucent mask on grid cells where the CI excludes
            0 — itsadug's ``show.diff``. The mask uses the simultaneous
            band when ``sim_ci=True``, the pointwise band otherwise. The
            "significant" region is exactly the union of where ``diff −
            CI > 0`` and ``diff + CI < 0``.
        alpha_diff : float
            Opacity of the ``show_diff`` mask in ``[0, 1]``. Default 0.5.
            Mirrors itsadug's ``alpha.diff``.
        col_diff : str
            Color of the ``show_diff`` mask. Default ``"white"`` so it
            washes out the heatmap's "significant" cells; pass any
            matplotlib color (hex, name, RGB tuple). Mirrors itsadug's
            ``col.diff``.
        color : str
            Matplotlib colormap for the heatmap. Default ``"RdBu_r"``
            (diverging). itsadug's default is ``topo.colors``; pick
            whichever cmap suits the data — diverging is recommended for
            differences so 0 sits at the cmap's neutral.
        col : str
            Color for the bold-contour overlay on the difference itself.
        ci_col : (str, str)
            Lower / upper-band contour colors when ``plot_ci=True`` —
            matches itsadug's ``ci.col`` default.
        n_levels : int
            Approximate number of contour levels. ``f̂`` and ``f̂±CI``
            share a single ``MaxNLocator(nbins=n_levels)`` level set so
            level values line up across the three layers.
        too_far : float
            Mask grid points whose normalized distance to the nearest
            data point exceeds this threshold (mgcv's ``exclude.too.far``).
            ``0`` (default) = no masking. itsadug's ``plot_diff2`` doesn't
            mask either, but it's useful for irregular boundaries.
        print_summary : bool
            Pass-through to :meth:`get_difference`.
        ax : matplotlib Axes | None
            Where to draw. ``None`` builds a new figure / axes.
        figsize : (float, float) | None
            Only used when creating a new figure.
        xlim, ylim : (float, float) | None
            Range for the auto-built grid (mirror of itsadug). ``None``
            (default) uses the data range with NaNs dropped.
        zlim : (float, float) | None
            Color-scale limits. ``None`` (default) uses ``[-m, m]`` if the
            difference straddles 0 (so a diverging cmap is centered);
            otherwise the diff's range.
        xlab, ylab : str | None
            Axis labels — default to the view names.
        title : str | None
            Plot title — default ``"Difference {levels[0]} − {levels[1]}"``.
        hide_label : bool
            Suppress the small "difference [, excl. random]" annotation.
        add_color_legend : bool
            Draw a colorbar to the right. Default ``True``.

        Returns the matplotlib ``Axes``.
        """
        from matplotlib.ticker import MaxNLocator
        import warnings as _w

        view = list(view)
        if len(view) < 2:
            raise ValueError(
                "view must contain two predictor names for plot_diff2"
            )
        if len(view) > 2:
            _w.warn(
                f"view has {len(view)} entries; plot_diff2 only uses the "
                f"first two ({view[0]!r}, {view[1]!r}).",
                stacklevel=2,
            )
            view = view[:2]
        xvar, yvar = view[0], view[1]
        for v in (xvar, yvar):
            if v not in self.data.columns:
                raise ValueError(
                    f"view variable {v!r} not in data; available: "
                    f"{list(self.data.columns)}"
                )
        cond = dict(cond) if cond else {}

        # Build the grid axes. cond[name] (if user-supplied) wins with a
        # warning; otherwise linspace over the data range, with xlim/ylim
        # narrowing it. Mirrors itsadug's plot_diff2 grid construction.
        def _build_axis(name, lim, lim_name):
            if name in cond:
                _w.warn(
                    f"Predictor {name!r} specified in view and cond. Values "
                    f"in cond being used, rather than the whole range of "
                    f"{name!r}.",
                    stacklevel=3,
                )
                return
            arr = self.data[name].drop_nulls().to_numpy().astype(float)
            if arr.size == 0:
                raise ValueError(
                    f"view variable {name!r} has no non-null values"
                )
            cond[name] = np.linspace(arr.min(), arr.max(), n_grid)
            if lim is not None:
                if len(lim) != 2:
                    _w.warn(
                        f"Invalid {lim_name} values specified. Argument "
                        f"{lim_name} is being ignored.",
                        stacklevel=3,
                    )
                else:
                    cond[name] = np.linspace(lim[0], lim[1], n_grid)

        _build_axis(xvar, xlim, "xlim")
        _build_axis(yvar, ylim, "ylim")

        result = self.get_difference(
            comp=comp, cond=cond, rm_ranef=rm_ranef,
            se=(se > 0), f=(se if se > 0 else 1.96),
            sim_ci=sim_ci, n_sim=n_sim, rng=rng,
            print_summary=print_summary,
        )

        # Reshape diff (and CI) onto an Nx × Ny grid in (xvar slow, yvar
        # fast) order. Sorting independent of expand_grid's column order
        # keeps this robust to formula-vs-data column order: get_difference
        # iterates _var_summary keys (data-column order), but lexsort over
        # (xvar, yvar) reorders into the canonical (slow=x, fast=y) layout.
        g = result.grid
        x_arr = g[xvar].to_numpy().astype(float)
        y_arr = g[yvar].to_numpy().astype(float)
        Nx = len(np.asarray(cond[xvar]))
        Ny = len(np.asarray(cond[yvar]))
        if Nx * Ny != len(result.difference):
            raise RuntimeError(
                f"plot_diff2: grid reshape mismatch "
                f"(expected {Nx}*{Ny}={Nx * Ny}, got {len(result.difference)})"
            )
        sort_idx = np.lexsort([y_arr, x_arr])  # primary key is the LAST one
        Z = result.difference[sort_idx].reshape(Nx, Ny)
        # When ``sim_ci=True``, ``result.sim_ci`` carries the simultaneous
        # envelope's half-width (already includes the f multiplier);
        # ``result.ci`` is the pointwise version. The downstream contour
        # overlay and ``show_diff`` mask both read whichever was requested.
        ci_src = result.sim_ci if sim_ci else result.ci
        CI_mat = (
            ci_src[sort_idx].reshape(Nx, Ny)
            if ci_src is not None else None
        )

        x_axis = np.asarray(cond[xvar], dtype=float)
        y_axis = np.asarray(cond[yvar], dtype=float)
        if too_far > 0.0:
            XXm, YYm = np.meshgrid(x_axis, y_axis, indexing="ij")
            mask = _too_far_mask(
                XXm.flatten(), YYm.flatten(),
                self.data[xvar], self.data[yvar], too_far,
            ).reshape(Nx, Ny)
            Z = np.where(mask, np.nan, Z)
            if CI_mat is not None:
                CI_mat = np.where(mask, np.nan, CI_mat)

        # --- plotting --------------------------------------------------------
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize or (6, 5))

        # contour / pcolormesh expect Z indexed [y, x]; transpose Nx×Ny → Ny×Nx.
        XX, YY = np.meshgrid(x_axis, y_axis, indexing="xy")
        Zp = Z.T

        # zlim auto: symmetric around 0 when the diff straddles 0 so a
        # diverging cmap stays centered; otherwise plain [min, max].
        if zlim is None:
            zmin_d = float(np.nanmin(Z)) if np.isfinite(Z).any() else 0.0
            zmax_d = float(np.nanmax(Z)) if np.isfinite(Z).any() else 0.0
            if zmin_d < 0 < zmax_d:
                m = max(-zmin_d, zmax_d)
                zlim_used = (-m, m)
            else:
                zlim_used = (zmin_d, zmax_d)
        else:
            zlim_used = (float(zlim[0]), float(zlim[1]))

        im = ax.pcolormesh(
            XX, YY, Zp, cmap=color, shading="auto",
            vmin=zlim_used[0], vmax=zlim_used[1],
        )

        # Shared contour levels for f̂ and (optionally) f̂±CI — same trick
        # _plot_smooth_2d uses, so the same numeric level lines up across
        # bold / dashed-lower / dotted-upper.
        if plot_ci and CI_mat is not None:
            zmin_b = float(np.nanmin(Z - CI_mat))
            zmax_b = float(np.nanmax(Z + CI_mat))
        else:
            zmin_b = float(np.nanmin(Z)) if np.isfinite(Z).any() else 0.0
            zmax_b = float(np.nanmax(Z)) if np.isfinite(Z).any() else 0.0
        if np.isfinite(zmin_b) and np.isfinite(zmax_b) and zmin_b < zmax_b:
            levels = MaxNLocator(
                nbins=max(int(n_levels), 1), steps=[1, 2, 5, 10],
            ).tick_values(zmin_b, zmax_b)
        else:
            levels = None

        if levels is not None:
            cs = ax.contour(
                XX, YY, Zp, levels=levels,
                colors=col, linestyles="solid", linewidths=0.8,
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
            if plot_ci and CI_mat is not None:
                ax.contour(
                    XX, YY, Zp - CI_mat.T, levels=levels,
                    colors=ci_col[0], linestyles=":", linewidths=0.6,
                )
                ax.contour(
                    XX, YY, Zp + CI_mat.T, levels=levels,
                    colors=ci_col[1], linestyles=":", linewidths=0.6,
                )

        # itsadug ``show.diff``: translucent mask on grid cells where the
        # CI excludes 0. Computed as the union of (Zp − CI > 0) and
        # (Zp + CI < 0); plotted via ``contourf`` on a binary 0/1 field
        # with a single-color cmap, alpha-blended over the heatmap.
        if show_diff and CI_mat is not None:
            sig = ((Zp - CI_mat.T > 0) | (Zp + CI_mat.T < 0)).astype(float)
            # Mask non-significant (0) cells so contourf paints only the
            # 1-valued cells; one-color cmap so ``col_diff`` does the work.
            ax.contourf(
                XX, YY, np.where(sig > 0.5, 1.0, np.nan),
                levels=[0.5, 1.5],
                colors=[col_diff],
                alpha=float(alpha_diff),
            )

        if add_color_legend:
            plt.colorbar(im, ax=ax)

        if title is None:
            title = f"Difference {result.levels[0]} − {result.levels[1]}"
        ax.set_title(title)
        ax.set_xlabel(xlab if xlab is not None else xvar)
        ax.set_ylabel(ylab if ylab is not None else yvar)
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(xlim)
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(ylim)

        if not hide_label:
            label = "difference"
            if rm_ranef not in (None, False) and result.rm_ranef_cancelled:
                label += ", excl. random"
            ax.text(
                0.99, 0.985, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#595959",
            )

        return ax

    def _var_summary(self) -> dict:
        """mgcv ``variable.summary`` parity: typical value per variable.

        Restricted to RHS variables of the formula (so we don't include the
        response or stray data columns). Numeric → median; factor/string →
        modal level.
        """
        from ..formula import referenced_columns  # local to avoid cycle

        rhs_vars = referenced_columns(self._expanded)
        out: dict = {}
        for name in self.data.columns:
            if name not in rhs_vars:
                continue
            col = self.data[name]
            if _is_factor_like_col(col):
                vals = col.drop_nulls()
                if len(vals) == 0:
                    continue
                # Mode: most frequent level. polars `.mode()` returns all ties;
                # take the first to get a deterministic single value.
                out[name] = vals.mode().to_list()[0]
            else:
                arr = col.drop_nulls().to_numpy().astype(float)
                if arr.size == 0:
                    continue
                out[name] = float(np.median(arr))
        return out

    # ------------- printing ------------------------------------------------

    def _family_display_name(self) -> str:
        """Family display string. Extended families' ``postproc`` (run
        once at fit time, stashed on ``self._postproc``) relabels with
        the fitted θ — e.g. mgcv reports ``Scaled t(5,0.3)`` rather
        than ``scat`` once a ``scat()`` fit has converged. Families
        without a ``family_name`` key keep the default ``family.name``.
        """
        pp = getattr(self, "_postproc", None)
        if pp and pp.get("family_name") is not None:
            return str(pp["family_name"])
        return self.family.name

    def _print_score(self) -> tuple[str, float]:
        """The ``{method} score:`` label + value of print.gam (= ``x$method`` +
        ``x$gcv.ubre``). REML/ML report the Laplace criterion (hea stores 2×);
        P-REML/P-ML the Pearson-Laplace variant; bam's fREML keeps its label;
        GACV.Cp (scale est) reports GACV; otherwise UBRE (known scale) or GCV."""
        if self.method == "P-REML":
            return "P-REML", self.REML_criterion / 2.0
        if self.method == "P-ML":
            return "P-ML", self.ML_criterion / 2.0
        if self.method == "REML":
            label = ("fREML" if getattr(self, "_method_in", None) == "fREML"
                     else "REML")
            return label, self.REML_criterion / 2.0
        if self.method == "ML":
            return "ML", self.ML_criterion / 2.0
        if self.method == "GACV.Cp" and not self._scale_known_fit:
            return "GACV", self.GCV_score
        return ("UBRE" if self._scale_known_fit else "GCV"), self.GCV_score

    def __repr__(self) -> str:
        # mgcv print.gam (mgcv.r:2443-2467): family block → Formula → per-smooth
        # estimated edf (round-4/3-sig, 7 per line, ``total =``) → method score
        # → optional rank. Family/link/formula displays mirror summary()'s
        # multi-LP handling (one link name per LP, one formula per line).
        if getattr(self.family, "is_general", False):
            link_disp = " ".join(link.name for link in self.family.links)
        else:
            link_disp = self.family.link.name
        formulas = (self.formula if isinstance(self.formula, (list, tuple))
                    else [self.formula])

        out = [
            "",
            f"Family: {self._family_display_name()} ",
            f"Link function: {link_disp} ",
            "",
            "Formula:",
            *[str(f) for f in formulas],
        ]

        if not self._blocks:
            # cat("Total model degrees of freedom", sum(edf), "\n")
            out.append(
                f"Total model degrees of freedom {_r_cat_num(self.edf_total)} "
            )
        else:
            edf_per = [float(self.edf[a:bcol].sum())
                       for (a, bcol) in self._block_col_ranges]
            edf_str = _format_edf_vector(edf_per)
            out.append("")
            out.append("Estimated degrees of freedom:")
            # each entry + a space, wrapping after 7; the trailing
            # " total = X " rides the final (unwrapped) line.
            line = ""
            for i, s in enumerate(edf_str, start=1):
                line += s + " "
                if i % 7 == 0:
                    out.append(line)
                    line = ""
            line += f" total = {_r_cat_num(round(self.edf_total, 2))} "
            out.append(line)

        label, score = self._print_score()
        score_line = f"{label} score: {_r_cat_num(score)}     "
        p = len(self.coef)
        if self.rank is not None and self.rank < p:
            score_line += f"rank: {self.rank}/{p}"
        out.append("")
        out.append(score_line)
        return "\n".join(out)

    def __str__(self) -> str:
        return self.__repr__()

    def _pterms_rows(self, freq: bool = False,
                     dispersion: float | None = None,
                     ) -> list[tuple[str, int, float, float]]:
        """mgcv summary.gam's pTerms block (mgcv.r:3928-3977): one joint
        Wald test per whole parametric term — a factor's columns are
        tested together, which the per-coefficient p.table can't do.

        Returns ``(label, df, stat, p)`` rows. ``stat`` is Chi.sq with a
        pchisq p-value when ``est.disp`` is FALSE (known scale, or any
        ``dispersion=`` override), else F = Chi.sq/df with
        pf(·, df, residual.df) (mgcv's est.disp dispatch; residual.df =
        n − Σedf). The covariance is ``Vp`` (summary.gam's default
        ``freq=FALSE``) or ``Ve`` under ``freq=True`` (mgcv.r:3890),
        rescaled by ``dispersion/sig2`` when a dispersion is supplied
        (mgcv.r:3895-3899); each term block is pseudo-inverted at
        rank.tol = √eps with the resulting rank as the df
        (:func:`_wald_pinv`), so dropped (rank-deficient) coefficients —
        zero rows in report space — reduce the df exactly like mgcv.

        Written list-generic over linear predictors like mgcv
        (``pterms <- if (is.list(object$pterms)) ... else list(...)``,
        mgcv.r:3930): one entry until §5.3 multi-LP fits land, when
        formula j ≥ 2 terms get mgcv's ``.{j-1}`` label suffix
        (mgcv.r:3939) and per-LP (assign, pstart) blocks feed the same
        loop. The intercept (assign 0) is never a term — mgcv's
        convention. mgcv's printed surface for this table is
        ``anova.gam``, not ``print.summary.gam`` — hea's ``anova()``
        consumes these rows; ``summary()``'s print output is unchanged.
        """
        # Per-LP (term labels, assign vector, pstart): the real list for
        # multi-LP (general-family) fits, a 1-list otherwise — exactly
        # mgcv's `pterms <- if (is.list(object$pterms)) ... else
        # list(...)` dispatch.
        md = getattr(self, "_md", None)
        if md is not None:
            pterms_list = [
                ([t.label for t in lp.expanded.terms],
                 list(lp.param_assign or []),
                 int(md.pstart[j]))
                for j, lp in enumerate(md.lps)
            ]
        else:
            pterms_list = [([t.label for t in self._expanded.terms],
                            list(getattr(self, "_param_assign", []) or []),
                            0)]
        est_disp = (not bool(self._scale_known_fit)) and dispersion is None
        residual_df = float(self.n) - float(self.edf_total)

        # summary.gam's covmat: Ve when freq else Vp (mgcv.r:3890),
        # times dispersion/sig2 under an override — in report
        # (original-p) space: zero rows/cols at dropped columns (mgcv
        # reinserts zeros for dropped coefficients).
        covmat = self.Ve if freq else self.Vp
        if dispersion is not None:
            covmat = covmat * (float(dispersion)
                               / float(self.sigma_squared))
        if self._keep_cols is not None:
            keep = self._keep_cols
            Vp_rep = np.zeros((keep.size, keep.size))
            Vp_rep[np.ix_(keep, keep)] = covmat
        else:
            Vp_rep = covmat
        beta = self._beta_report

        rank_tol = float(np.finfo(float).eps) ** 0.5
        rows: list[tuple[str, int, float, float]] = []
        for j, (labels, asgn, pstart) in enumerate(pterms_list):
            asgn = np.asarray(asgn, dtype=int)
            for i, label in enumerate(labels, start=1):
                idx = pstart + np.flatnonzero(asgn == i)
                if idx.size == 0:
                    continue
                b = beta[idx]
                V = Vp_rep[np.ix_(idx, idx)]
                if idx.size == 1:
                    nb = 1
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi = float(b[0] * b[0] / V[0, 0])
                else:
                    Vi, nb = _wald_pinv(V, idx.size, rank_tol)
                    chi = float(b @ Vi @ b)
                lab = label if j == 0 else f"{label}.{j}"
                if not est_disp:
                    pv = float(_dist.pchisq(chi, nb, lower_tail=False))
                    rows.append((lab, nb, chi, pv))
                else:
                    stat = chi / nb
                    pv = (float(_dist.pf(stat, nb, residual_df, lower_tail=False))
                          if residual_df > 0 else float("nan"))
                    rows.append((lab, nb, stat, pv))
        return rows

    def _se_report_for(self, freq: bool, dispersion: float | None
                       ) -> np.ndarray:
        """Report-space coefficient SEs from summary.gam's covmat choice:
        ``Ve`` when ``freq`` else ``Vp`` (mgcv.r:3890), rescaled by
        ``dispersion/sig2`` under an override (mgcv.r:3895-3899).
        Defaults return the precomputed ``_se_report`` unchanged."""
        if not freq and dispersion is None:
            return self._se_report
        V = self.Ve if freq else self.Vp
        se = np.sqrt(np.clip(np.diag(V), 0.0, None))
        if dispersion is not None:
            se = se * np.sqrt(float(dispersion)
                              / float(self.sigma_squared))
        if self._keep_cols is not None:
            se_rep = np.zeros(self._keep_cols.size)
            se_rep[self._keep_cols] = se
            return se_rep
        return se

    def summary(self, digits: int = 4, freq: bool = False,
                dispersion: float | None = None,
                re_test: bool = True) -> None:
        """mgcv-style summary: parametric table + smooth-edf table + fit stats.

        ``freq=True`` uses the frequentist ``Ve`` instead of the Bayesian
        ``Vp`` for the parametric tables (summary.gam's ``freq``;
        smooth tests always use ``Vp``). ``dispersion=`` overrides the
        scale: every covariance is rescaled by ``dispersion/sig2``, the
        tests switch to their known-scale forms (z / Chi.sq), and the
        printed ``Scale est.`` shows the supplied value — exactly
        ``summary.gam(..., dispersion=)``. ``re_test=False`` omits the
        random-effect/fully-penalized smooths from the significance
        table (summary.gam's ``re.test``; mgcv.r:4024 skips those rows
        rather than rerouting them).
        """
        # multi-LP (general-family) fits: all link names, one formula
        # per line — mgcv's print.summary.gam layout.
        if getattr(self.family, "is_general", False):
            link_disp = " ".join(link.name for link in self.family.links)
        else:
            link_disp = self.family.link.name
        if isinstance(self.formula, (list, tuple)):
            formula_disp = "\n".join(str(f) for f in self.formula)
        else:
            formula_disp = str(self.formula)
        out = [
            "",
            f"Family: {self._family_display_name()}",
            f"Link function: {link_disp}",
            "",
            f"Formula: {formula_disp}",
            "",
        ]
        # mgcv print.summary.gam (mgcv.r:4089): show the rank line only
        # when the model is rank deficient (against the original p).
        p_orig = getattr(self, "_p_orig", None) or self.p
        if self.rank is not None and self.rank < p_orig:
            out.insert(-1, f"Rank: {self.rank}/{p_orig}")

        # -- parametric table (lm-style) -----------------------------------
        # mgcv (summary.gam): when scale.estimated, t/Pr(>|t|) on residual.df;
        # otherwise (binomial/poisson with φ ≡ 1, or any family under a
        # dispersion= override) Wald z/Pr(>|z|). Dropped (rank-deficiency)
        # coefficients show 0 / 0 / NaN like mgcv.
        scale_known = bool(self._scale_known_fit)
        est_disp = (not scale_known) and dispersion is None
        se_report = self._se_report_for(freq, dispersion)
        n_par = len(self.parametric_columns)
        if n_par > 0:
            out.append("Parametric coefficients:")
            # multi-LP fits: the parametric columns sit at each LP's
            # pstart, not a single prefix (mgcv.r:3907-3912's
            # length(nsdf)>1 branch); ``_param_idx`` carries them.
            par_idx = getattr(self, "_param_idx", None)
            if par_idx is not None:
                est = self._beta_report[par_idx]
                se = se_report[par_idx]
            else:
                est = self._beta_report[:n_par]
                se = se_report[:n_par]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = est / se
            if not est_disp:
                pv = 2 * _nmath.pnorm5_vec(np.abs(t_stats), lower_tail=False)
                stat_col = "z value"
                pcol = "Pr(>|z|)"
            elif self.df_residuals > 0 and np.isfinite(self.df_residuals):
                pv = 2 * _dist.pt(np.abs(t_stats), self.df_residuals, lower_tail=False)
                stat_col = "t value"
                pcol = "Pr(>|t|)"
            else:
                pv = np.full_like(t_stats, np.nan)
                stat_col = "t value"
                pcol = "Pr(>|t|)"
            sig = significance_code(pv)
            est_s, se_s = format_signif_jointly([est, se], digits=digits)
            tbl = pl.DataFrame({
                "": self.parametric_columns,
                "Estimate":   est_s,
                "Std. Error": se_s,
                stat_col:     format_signif(t_stats, digits=digits),
                pcol:         format_pval(pv, digits=_dig_tst(digits)),
                " ":          sig,
            })
            out.append(format_df(
                tbl,
                align={c: "right" for c in
                       ("Estimate", "Std. Error", stat_col, pcol)},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- smooth-edf table ----------------------------------------------
        # Rows from ``_smooth_significance_rows`` — reTest / testStat
        # dispatch on null.space.dim, mixture p-values, Chi.sq↔F column by
        # ``family.scale_known``. Ref.df reports the rank used in the test.
        # mgcv's print gates the section on m>0 only (mgcv.r:4084) — with
        # every row skipped (re_test=False, all-re model) R still prints
        # the header over an empty printCoefmat; mirror that.
        if self._blocks:
            out.append("Approximate significance of smooth terms:")
            sm_rows = self._smooth_significance_rows(dispersion=dispersion,
                                                     re_test=re_test)
            rows_label = [r[0] for r in sm_rows]
            rows_edf   = [r[1] for r in sm_rows]
            rows_refdf = [r[2] for r in sm_rows]
            rows_stat  = [r[3] for r in sm_rows]
            rows_p     = [r[4] for r in sm_rows]
            sig = significance_code(rows_p)
            stat_col = "F" if est_disp else "Chi.sq"
            sm_tbl = pl.DataFrame({
                "":        rows_label,
                "edf":     format_signif(rows_edf, digits=digits),
                "Ref.df":  format_signif(rows_refdf, digits=digits),
                stat_col:  format_signif(rows_stat, digits=digits),
                "p-value": format_pval(rows_p, digits=_dig_tst(digits)),
                " ":       sig,
            })
            out.append(format_df(
                sm_tbl,
                align={c: "right" for c in
                       ("edf", "Ref.df", stat_col, "p-value")},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )
            out.append("")

        # -- fit stats ------------------------------------------------------
        # mgcv: `formatC(r.sq, digits=3, width=5)`, `formatC(dev.expl*100,
        # digits=3)`, `formatC(REML/GCV/scale, digits=5)`. Match that.
        # General families have no r² (mgcv's no.r.sq — summary.gam sets
        # r.sq NULL and the print skips it, mgcv.r:4055).
        if np.isfinite(getattr(self, "r_squared_adjusted", float("nan"))):
            out.append(
                f"R-sq.(adj) = {self.r_squared_adjusted:.3g}  "
                f"Deviance explained = {self.deviance_explained * 100:.3g}%"
            )
        else:
            out.append(
                f"Deviance explained = {self.deviance_explained * 100:.3g}%"
            )
        # mgcv summary.gam line 4058: prepend "-" only for "REML"/"ML";
        # "P-REML"/"P-ML" print bare (mgcv leaves them untouched), and
        # "fREML" prints as ``fREML = X``. ``_method_in`` preserves the
        # user's choice across bam's internal fREML→REML rename so the
        # footer label tracks the original.
        method_label = getattr(self, "_method_in", self.method)
        # print.summary.gam shows x$scale — the dispersion override when
        # one was supplied (mgcv.r:3900/4097).
        disp_print = (float(dispersion) if dispersion is not None
                      else self.sigma_squared)
        if self.method == "REML":
            tag = "fREML" if method_label == "fREML" else "-REML"
            out.append(
                f"{tag} = {self.REML_criterion / 2:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        elif self.method == "ML":
            out.append(
                f"-ML = {self.ML_criterion / 2:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        elif self.method == "P-REML":
            out.append(
                f"P-REML = {self.REML_criterion / 2:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        elif self.method == "P-ML":
            out.append(
                f"P-ML = {self.ML_criterion / 2:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        elif self.method == "GACV.Cp" and not self._scale_known_fit:
            out.append(
                f"GACV = {self.GCV_score:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        else:
            # method="GCV.Cp" dispatches by family.scale_known: scale-known
            # (Poisson, Binomial) optimizes UBRE, scale-unknown optimizes
            # GCV. mgcv's summary.gam labels the printed score with the
            # criterion that was actually optimized.
            label = "UBRE" if self._scale_known_fit else "GCV"
            out.append(
                f"{label} = {self.GCV_score:.5g}  "
                f"Scale est. = {disp_print:.5g}  n = {self.n}"
            )
        print("\n".join(out))

    def _k_check(
        self,
        type: str = "deviance",
        subsample: int = 5000,
        n_rep: int = 200,
        seed: int | None = None,
    ) -> pl.DataFrame | None:
        """Port of mgcv's ``k.check`` — basis-dimension test per smooth.

        For each smooth block, pair each residual with neighbours in
        covariate space and compare the mean squared first difference
        against a permutation null. A small ``k-index`` (≪ 1) and small
        p-value indicate the basis is too small to absorb the signal.

        1-D smooths: sort residuals by the covariate, take ``diff``.
        Multi-D smooths: average over the 3 nearest neighbours by
        Euclidean distance in raw covariate space. mgcv additionally
        rescales axes for tensor smooths via ``PredictMat`` gradient
        norms; hea has no PredictMat yet, so tensor (``te``/``ti``/
        ``t2``) k-indexes are not on mgcv's rescaled axes — the
        qualitative "k-index < 1" warning still applies.

        All randomness (the optional subsample and the permutation
        null) goes through R's ``sample()`` via the bit-exact
        ``hea.R.rng`` port, consuming the stream in mgcv's order — so
        ``seed=k`` reproduces ``set.seed(k); k.check(b, ...)``'s
        p-values exactly (k'/edf/k-index are RNG-free).

        Returns a polars DataFrame with columns ``""``, ``"k'"``,
        ``"edf"``, ``"k-index"``, ``"p-value"`` (one row per smooth
        block), or ``None`` if there are no smooths.
        """
        if not self._blocks:
            return None

        rsd = self.residuals_of(type=type)
        n_full = len(rsd)
        if seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        from ..R.rng import RMersenneTwister
        r_rng = RMersenneTwister(seed)

        # Optional subsample (mgcv's `k.sample`). The same row indices
        # subset both residuals and the per-smooth covariate columns so
        # the neighbour graph stays consistent.
        if n_full > subsample:
            idx = r_rng.sample_int(n_full, subsample)
            rsd = rsd[idx]
        else:
            idx = np.arange(n_full)
        nr = len(rsd)
        rsd_sq_mean = float(np.mean(rsd ** 2))
        if rsd_sq_mean <= 0:
            rsd_sq_mean = 1.0

        rows: list[tuple[str, float, float, float, float]] = []
        for b, (a, bcol) in zip(self._blocks, self._block_col_ranges):
            kc = float(bcol - a)
            edf_b = float(self.edf[a:bcol].sum())
            var_names = list(b.term)

            ok = bool(var_names)
            cols: list[np.ndarray] = []
            for v in var_names:
                if v not in self.data.columns:
                    ok = False
                    break
                s = self.data[v]
                if not s.dtype.is_numeric():
                    ok = False
                    break
                cols.append(s.to_numpy().astype(float)[idx])
            if not ok:
                rows.append((b.label, kc, edf_b, float("nan"), float("nan")))
                continue

            # Generate all n_rep permutations via per-iter R ``sample()``
            # draws to consume the stream in mgcv's order — then vectorize
            # the diff/square/mean over the (n_rep, nr) stack. This keeps
            # p-values bit-identical to R's unrolled loop while skipping
            # the Python overhead of n_rep separate invocations.
            shufs = np.empty((n_rep, nr))
            for i in range(n_rep):
                shufs[i] = rsd[r_rng.sample_int(nr, nr)]

            if len(cols) == 1:
                order = np.argsort(cols[0], kind="stable")
                rsd_o = rsd[order]
                v_obs = float(np.mean(np.diff(rsd_o) ** 2) / 2)
                diffs = np.diff(shufs, axis=1)
                ve = np.mean(diffs * diffs, axis=1) / 2
            else:
                from scipy.spatial import cKDTree
                Xnn = np.column_stack(cols)
                nn = 3
                # k=nn+1, skip column 0 (self at distance 0).
                tree = cKDTree(Xnn)
                _, ni = tree.query(Xnn, k=nn + 1)
                ni = ni[:, 1:]
                e_parts = [rsd - rsd[ni[:, j]] for j in range(nn)]
                v_obs = float(np.mean(np.concatenate(e_parts) ** 2) / 2)
                # parts_3d[i, r, j] = shufs[i, r] - shufs[i, ni[r, j]].
                # Transpose to (n_rep, nn, nr) before flattening so the
                # row-major order matches ``np.concatenate(parts)`` (which
                # lays out all j=0 rows first, then j=1, then j=2).
                parts_3d = shufs[:, :, None] - shufs[:, ni]
                parts_flat = parts_3d.transpose(0, 2, 1).reshape(n_rep, -1)
                ve = np.mean(parts_flat * parts_flat, axis=1) / 2

            p_val = float(np.mean(ve < v_obs))
            k_index = v_obs / rsd_sq_mean
            rows.append((b.label, kc, edf_b, float(k_index), p_val))

        return pl.DataFrame({
            "":        [r[0] for r in rows],
            "k'":      [r[1] for r in rows],
            "edf":     [r[2] for r in rows],
            "k-index": [r[3] for r in rows],
            "p-value": [r[4] for r in rows],
        })

    def check(
        self,
        type: str = "deviance",
        k_sample: int = 5000,
        k_rep: int = 200,
        seed: int | None = None,
        plots: bool = True,
        rep: int = 0,
        level: float = 0.9,
        s_rep: int = 10,
    ) -> None:
        """mgcv-style ``gam.check``: diagnostic plots + convergence text +
        ``k.check`` table.

        With ``plots=True`` (default — gam.check always plots) draws
        mgcv's 2×2 panel first (``plot_check``: qq.gam, residuals vs
        linear predictor, residual histogram, response vs fitted), then
        prints:

        - Method / optimizer line.
        - Convergence status, iterations, gradient range.
        - Score and scale at the optimum.
        - Hessian positive-definiteness and eigenvalue range.
        - Per-smooth basis-dimension check table from ``_k_check``.

        Parameters
        ----------
        type : {"deviance", "pearson", "response"}
            Residual type for the plots and ``_k_check``. Default
            matches mgcv.
        k_sample : int
            Maximum residuals to use for the basis check
            (mgcv's ``k.sample``).
        k_rep : int
            Permutation reps for the k-check p-value
            (mgcv's ``k.rep``).
        seed : int | None
            Seeds the permutations/subsample and the qq randomization.
            ``None`` uses fresh randomness each call.
        plots : bool
            Draw the 2×2 diagnostic panel (``plot_check``).
        rep, level, s_rep
            Passed to ``qq_gam`` (see there).
        """
        if plots:
            self.plot_check(type=type, rep=rep, level=level, s_rep=s_rep,
                            seed=seed)
        out: list[str] = []

        # --- method / optimizer header ---
        method_label = self.method
        optimizer_label = ("outer efs"
                           if (self._outer_info or {}).get("conv") in
                           ("full convergence", "iteration limit reached")
                           and (getattr(self.family, "available_derivs",
                                        2) == 0
                                or getattr(self, "optimizer",
                                           ("outer",))[0] == "efs")
                           else "outer newton")
        out.append(f"Method: {method_label}   Optimizer: {optimizer_label}")

        # --- convergence info from _outer_newton ---
        info = self._outer_info
        if info is None:
            if not self._blocks:
                out.append("Model required no smoothing parameter selection")
            else:
                out.append(
                    "Smoothing parameters fixed by user — no outer optimization."
                )
        else:
            iters = info["iter"]
            plural = "" if iters == 1 else "s"
            out.append(f"{info['conv']} after {iters} iteration{plural}.")
            # General-fit efs / fixed-sp infos carry conv+iter only —
            # print whatever diagnostics the optimizer recorded.
            grad = np.asarray(info.get("grad", np.zeros(0)))
            if grad.size > 0:
                out.append(
                    f"Gradient range [{float(grad.min()):.7g},"
                    f"{float(grad.max()):.7g}]"
                )
            score = info.get("score")
            if score is not None:
                scale = self.sigma_squared
                out.append(f"(score {score:.7g} & scale {scale:.7g}).")
            H = np.asarray(info.get("hess")) \
                if info.get("hess") is not None else np.zeros(0)
            if H.size > 0:
                ev = np.linalg.eigvalsh(0.5 * (H + H.T))
                ev_min, ev_max = float(ev.min()), float(ev.max())
                pd_text = (
                    "Hessian positive definite, "
                    if ev_min > 0
                    else "Hessian not positive definite, "
                )
                out.append(
                    f"{pd_text}eigenvalue range [{ev_min:.7g},{ev_max:.7g}]."
                )
        rank_disp = self.rank if self.rank is not None else self.p
        # mgcv reports rank against the *original* parameter count when
        # columns were dropped for identifiability (11 / 12 etc.).
        p_disp = getattr(self, "_p_orig", None) or self.p
        out.append(f"Model rank = {rank_disp} / {p_disp}")
        out.append("")

        # --- basis dimension check ---
        ktab = self._k_check(
            type=type, subsample=k_sample, n_rep=k_rep, seed=seed,
        )
        if ktab is not None:
            out.append(
                "Basis dimension (k) checking results. Low p-value "
                "(k-index<1) may"
            )
            out.append(
                "indicate that k is too low, especially if edf is close to k'."
            )
            out.append("")
            kc_vals  = ktab["k'"].to_list()
            edf_vals = ktab["edf"].to_list()
            ki_vals  = ktab["k-index"].to_list()
            pv_vals  = ktab["p-value"].to_list()
            sig = significance_code(pv_vals)
            disp = pl.DataFrame({
                "":        ktab[""].to_list(),
                "k'":      format_signif(kc_vals,  digits=3, min_decimals=2),
                "edf":     format_signif(edf_vals, digits=3, min_decimals=2),
                "k-index": format_signif(ki_vals,  digits=3, min_decimals=2),
                "p-value": format_pval(pv_vals,    digits=2),
                " ":       sig,
            })
            out.append(format_df(
                disp,
                align={c: "right" for c in ("k'", "edf", "k-index", "p-value")},
            ))
            out.append("---")
            out.append(
                "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
            )

        print("\n".join(out))

    def concurvity(self, full: bool = True):
        """Port of mgcv's ``concurvity`` (mgcv.r:3340-3423).

        Concurvity measures how well each model term could be
        approximated by the others — a generalization of collinearity.
        Three measures in [0, 1] (0 = no problem, 1 = the term lies
        entirely in the span of the rest): ``worst`` (largest singular
        value² of the projected-vs-total ratio — the worst case over
        coefficient directions), ``observed`` (at the fitted
        coefficients), ``estimate`` (the Frobenius/"less pessimistic"
        measure).

        ``full=True`` (default): each term against ALL the rest —
        returns a DataFrame with one row per measure and one column per
        term. ``full=False``: pairwise — a dict of three DataFrames
        (``worst``/``observed``/``estimate``), entry [i, j] measuring
        how much of term j lies in the span of term i alone (diagonal
        1).

        Blocks follow mgcv exactly: each smooth's coefficient range,
        plus a leading ``"para"`` block when parametric columns exist —
        which in mgcv is just the FIRST column (the intercept):
        ``stop <- c(min(start)-1, stop)`` runs after ``start`` was
        prepended with 1, so the para range is ``1:0`` and R's indexing
        quirks collapse it to column 1. Ported bug-for-bug (it's why
        para's three measures coincide — single column ⇒ β cancels).
        Columns in no block (the other parametric columns; LP ≥ 2's
        parametric columns on multi-formula fits) participate in the
        ``full`` complement but not in the pairwise comparison — also
        mgcv's behavior. The design is the unweighted model matrix; on
        rank-dropped fits it is the reduced (identifiable) design.
        """
        from scipy.linalg import solve_triangular as _sla_tri
        if not self._blocks:
            raise ValueError("nothing to do for this model")
        md = getattr(self, "_md", None)
        X = np.asarray(md.X if md is not None else self._X_full,
                       dtype=float)
        # Speed step (mgcv.r:3351): reduce to the p×p R factor — every
        # measure below is quadratic in columns, so spans are preserved.
        Rf = np.linalg.qr(X, mode="r")
        p = Rf.shape[1]
        blocks = [(a, b) for a, b in self._block_col_ranges]
        labels = [b.label for b in self._blocks]
        min_start = min(a for a, _ in blocks)
        if min_start > 0:
            # mgcv.r:3359-3364's "append parametric terms": the 1:0
            # range collapses to column 1 only — see the docstring.
            blocks = [(0, 1)] + blocks
            labels = ["para"] + labels
        m = len(blocks)
        beta = np.asarray(self._beta, dtype=float)

        def _measures(Xi, Xj, bj):
            """The three concurvity measures of Xj's span explained by
            Xi (mgcv.r:3376-3387): unpivoted QR of [Xi Xj], the Xj
            columns' R block split into the projection rows (1:r) and
            the orthogonal remainder."""
            r = Xi.shape[1]
            Rq = np.linalg.qr(np.concatenate([Xi, Xj], axis=1),
                              mode="r")[:, r:]
            Rt = np.linalg.qr(Rq, mode="r")
            M = _sla_tri(Rt.T, Rq[:r].T, lower=True)
            worst = float(np.linalg.svd(M, compute_uv=False)[0] ** 2)
            observed = (float(np.sum((Rq[:r] @ bj) ** 2))
                        / float(np.sum((Rt @ bj) ** 2)))
            estimate = (float(np.sum(Rq[:r] ** 2))
                        / float(np.sum(Rq ** 2)))
            return worst, observed, estimate

        names = ["worst", "observed", "estimate"]
        if full:
            conc = np.zeros((3, m))
            for i, (a, b) in enumerate(blocks):
                others = np.r_[0:a, b:p]
                conc[:, i] = _measures(Rf[:, others], Rf[:, a:b],
                                       beta[a:b])
            return pl.DataFrame(
                {"": names,
                 **{lab: conc[:, i] for i, lab in enumerate(labels)}})
        mats = [np.eye(m), np.eye(m), np.eye(m)]
        for i, (ai, bi) in enumerate(blocks):
            for j, (aj, bj_) in enumerate(blocks):
                if i == j:
                    continue
                w, o, e = _measures(Rf[:, ai:bi], Rf[:, aj:bj_],
                                    beta[aj:bj_])
                mats[0][i, j] = w
                mats[1][i, j] = o
                mats[2][i, j] = e
        return {
            nm: pl.DataFrame(
                {"": labels,
                 **{lab: mats[k][:, jj]
                    for jj, lab in enumerate(labels)}})
            for k, nm in enumerate(names)
        }

    def influence(self) -> np.ndarray:
        """mgcv's ``influence.gam`` (mgcv.r:4415): the penalized
        hat-matrix diagonal ``model$hat`` (sums to the total edf).

        General-family (multi-LP) fits carry no hat values in mgcv
        either (``model$hat`` is NULL there, making influence empty and
        cooks.distance all-NA) — hea raises instead of mirroring the
        silent NULL."""
        lev = getattr(self, "leverage", None)
        if lev is None:
            raise NotImplementedError(
                "mgcv stores no hat values for general-family fits "
                "(model$hat is NULL); influence()/cooks_distance() are "
                "undefined here."
            )
        return np.asarray(lev, dtype=float)

    def cooks_distance(self) -> np.ndarray:
        """mgcv's ``cooks.distance.gam`` (mgcv.r:4212-4218):

            (pearson / (1 − hat))² · hat / (φ̂ · Σedf)

        with the Pearson residuals, the fitted dispersion ``sig2`` and
        the penalized hat diagonal. The same quantity drives
        ``plot_leverage``'s contour labels; this is the per-observation
        accessor."""
        hat = self.influence()
        res = np.asarray(self.residuals_of("pearson"), dtype=float)
        p_edf = float(self.edf_total)
        return ((res / (1.0 - hat)) ** 2 * hat
                / (float(self.sigma_squared) * p_edf))

    def vcov(self, sandwich: bool = False, freq: bool = False,
             dispersion: "float | None" = None,
             unconditional: bool = False) -> np.ndarray:
        """mgcv's ``vcov.gam`` (mgcv.r:4396): the fitted-coefficient
        covariance matrix. Default ``Vp`` (Bayesian, smoothing-parameter
        conditional); ``freq=True`` → ``Ve`` (frequentist); ``unconditional=
        True`` → ``Vc`` (adds smoothing-parameter uncertainty); ``sandwich=
        True`` → the robust sandwich estimator. ``dispersion`` rescales by
        ``dispersion / sig2``. (Vp/Ve/Vc are already exposed as attributes;
        this is the mgcv-parity accessor.)"""
        if sandwich:
            vc = self._gam_sandwich(freq)
        elif freq:
            vc = np.asarray(self.Ve, dtype=float)
        else:
            vc = (np.asarray(self.Vc, dtype=float)
                  if unconditional and getattr(self, "Vc", None) is not None
                  else np.asarray(self.Vp, dtype=float))
        if dispersion is not None:
            vc = dispersion * vc / float(self.scale)
        return vc

    def _gam_sandwich(self, freq: bool = False) -> np.ndarray:
        """mgcv's ``gam.sandwich`` (mgcv.r:4374): sandwich/robust covariance
        ``Vs = m·Vp·(Σ wᵢ² XᵢXᵢᵀ)·Vp + B2``, with ``m = n/(n−Σedf)``,
        ``B2 = Vp−Ve`` (the Bayes squared-bias term, 0 when ``freq``) and, for
        an exponential family, ``wᵢ = μ'(ηᵢ)(yᵢ−μᵢ)/(φ̂·V(μᵢ))``. Extended /
        general families use a family-specific meat mgcv builds from ``dDeta``
        / ``family$sandwich`` (mgcv.r:4379-4386) — not retained post-fit here
        (mgcv itself ``stop()``s for general families lacking ``$sandwich``)."""
        Vp = np.asarray(self.Vp, dtype=float)
        B2 = 0.0 if freq else (Vp - np.asarray(self.Ve, dtype=float))
        fam = self.family
        # mgcv uses model.matrix(b) — the ORIGINAL design (mgcv.r:4377):
        # the stacked multi-LP X for general fits, _X_full otherwise.
        X = (self._md.X if getattr(fam, "is_general", False)
             else self._X_full)
        n = X.shape[0]
        m = n / (n - float(self.edf_total))
        sig2 = float(self.scale)
        mu = np.asarray(self.fitted_values, dtype=float)
        if getattr(fam, "is_general", False):
            # general family: meat from ``family$sandwich`` (mgcv.r:4380-4382)
            # — the per-observation gradient outer-product sum, ll(deriv=1,
            # sandwich=TRUE)$lbb. Families without the slot (cox_ph, mvn,
            # third-party) raise mgcv's "no sandwich estimate available for
            # this model" stop inside :meth:`GeneralFamily.sandwich`.
            meat = fam.sandwich(self._y_arr, X,
                                np.asarray(self.coef.values, dtype=float),
                                self._wt, lpi=self.lpi)
            return m * Vp @ meat @ Vp + B2
        if self._family_mgcv_extended:
            # extended family: meat from the deviance η-derivative
            # ``crossprod(0.5/φ·Deta·X)`` (mgcv.r:4384-4385).
            dd = fam.dDeta(self._y_arr, mu, self._wt, fam.get_theta(), 0)
            Wx = (0.5 / sig2) * dd["Deta"][:, None] * X
            return m * Vp @ (Wx.T @ Wx) @ Vp + B2
        # exponential family: ``wᵢ = μ'(ηᵢ)(yᵢ−μᵢ)/(φ·V(μᵢ))`` (mgcv.r:4388-4390).
        eta = np.asarray(self.linear_predictors, dtype=float)
        w = (fam.link.mu_eta(eta) * (self._y_arr - mu)
             / (sig2 * fam.variance(mu)))
        Wx = w[:, None] * X
        return m * Vp @ (Wx.T @ Wx) @ Vp + B2

    # ----- diagnostic plots -----------------------------------------------
    #
    # Match the graphical half of mgcv's gam.check + R's plot.glm:
    # - x-axis on residual panels = η̂ (linear predictors), labeled
    #   "Predicted values".
    # - panels 1/2/3 use deviance residuals (residuals.gam default).
    # - panel 5 (leverage) uses standardized Pearson residuals on y, with
    #   Cook's-distance contours scaled by edf_total.
    #
    # Per-smooth effect curves (mgcv's plot.gam) and the 2D fitted-surface
    # view (vis.gam) are separate plot methods, added in later passes.

    def plot_observed_fitted(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black", label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        y = self._y_arr
        yhat = self.fitted_values
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=y - yhat, n=label_n)
        ax.set_xlabel("Fitted (μ̂)")
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
        eta = self.linear_predictors
        r = self.residuals_of("deviance")
        ax.scatter(eta, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(eta, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, eta, r, scores=r, n=label_n)
        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")
        return ax

    def plot_qq(self, ax=None, figsize=None, label_n=3):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(
            ax, self.std_dev_residuals, label_n=label_n,
            ylabel="Std. deviance resid.",
        )
        return ax

    def _qq_gam_quantiles(self, type: str = "deviance", rep: int = 0,
                          level: float = 0.9, s_rep: int = 10,
                          seed: int | None = None) -> dict:
        """qq.gam's quantile computation (plots.r:116-163).

        Returns ``{"D", "Dq", "lim", "dm"}``: the residuals, the
        theoretical quantiles (``None`` → caller falls back to a normal
        QQ plot, like mgcv when the family has neither qf nor rd), the
        (2, n) simulation band (``0 < level < 1``), and the sorted
        simulated-residual matrix (returned when ``level >= 1`` so the
        caller can draw per-replicate lines).

        RNG: the direct (default) path randomizes only through R's
        ``sample(U)`` — run through the bit-exact ``hea.R.rng`` port, so
        ``seed=k`` reproduces R's ``set.seed(k); qq.gam(...)`` exactly.
        The ``rep>0`` simulation path draws response deviates through the same
        bit-exact stream (``RGenerator`` over ``RMersenneTwister``), so ``seed=k``
        is **bit-exact to R's ``set.seed(k); qq.gam(rep=)``** for every built-in
        family: gaussian, gaulss, shash, scat, Gamma, poisson, binomial, negbin,
        inverse.gaussian (mgcv's ``rig``) and tweedie (per-jump ``rTweedie``).
        """
        D = np.asarray(self.residuals_of(type), dtype=float)
        n = D.size
        if seed is None:
            seed = int(np.random.default_rng().integers(2**31 - 1))
        fam = self.family
        lim = Dq = dm_out = None
        if rep == 0:
            if getattr(fam, "qf", None) is None:
                rep = 50  # try simulation if no quantile function
            level = 0
        mu = self.fitted_values
        wt = self._wt
        scale = self.sigma_squared
        if rep > 0:  # simulate quantiles via the family's rd
            if getattr(fam, "rd", None) is None:
                return {"D": D, "Dq": None, "lim": None, "dm": None}
            from ..R.rng import RGenerator

            def _simulate(rng):
                out = np.empty((n, rep))
                for i in range(rep):
                    yr = fam.rd(rng, mu, wt, scale)
                    out[:, i] = np.sort(self._residuals_for_y(yr, type))
                return out

            try:
                # R-exact: rd hooks draw from the bit-exact MT stream, so
                # seed=k reproduces R's set.seed(k); qq.gam(rep=) for every
                # built-in family. Safety net: if a custom family's rd reaches an
                # RNG method the facade doesn't cover, fall back to numpy (MC).
                dm = _simulate(RGenerator(seed))
            except (NotImplementedError, AttributeError):
                dm = _simulate(np.random.default_rng(seed))
            Dq = np.quantile(dm.ravel(), (np.arange(1, n + 1) - 0.5) / n)
            alpha = (1.0 - level) / 2.0
            if alpha > 0.5 or alpha < 0:
                alpha = 0.05
            if 0 < level < 1:
                lim = np.quantile(dm, [alpha, 1.0 - alpha], axis=1)
            elif level >= 1:
                dm_out = dm
        else:  # direct: randomized uniform quantiles through qf
            from ..R.rng import RMersenneTwister
            r_rng = RMersenneTwister(seed)
            U = (np.arange(1, n + 1) - 0.5) / n
            dm = np.empty((n, s_rep))
            for i in range(s_rep):
                U = U[r_rng.sample_int(n, n)]   # R: U <- sample(U, n)
                q0 = fam.qf(U, mu, wt, scale)
                dm[:, i] = np.sort(self._residuals_for_y(q0, type))
            Dq = np.sort(dm.mean(axis=1))
        return {"D": D, "Dq": Dq, "lim": lim, "dm": dm_out}

    def qq_gam(self, type: str = "deviance", rep: int = 0,
               level: float = 0.9, s_rep: int = 10,
               seed: int | None = None, ax=None, figsize=None,
               rl_col: str = "red", rep_col: str = "0.8"):
        """mgcv's ``qq.gam`` (plots.r:94): QQ plot of residuals against
        family-correct theoretical quantiles.

        ``rep=0`` uses the family's quantile function directly (averaged
        over ``s_rep`` randomizations of the uniform grid); families
        without one simulate ``rep=50`` datasets via their random-deviate
        hook instead (gaulss takes this path — its rd draws
        ``N(μ̂, (√scale/w)/τ̂)`` from the (n,2) fitted matrix,
        gamlss.r:1089), and with ``rep>0`` simulation is forced with a
        ``level`` reference band (``level>=1`` draws each replicate as a
        line). Families with neither hook fall back to a normal QQ plot
        of the residuals, like mgcv.

        ``seed=k`` reproduces R's ``set.seed(k); qq.gam(...)`` bit-exactly — the
        direct path's ``sample()`` and the ``rep>0`` path's response deviates run
        through the ``hea.R.rng`` port for every built-in family (see
        :meth:`_qq_gam_quantiles`).
        """
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        qq = self._qq_gam_quantiles(type=type, rep=rep, level=level,
                                    s_rep=s_rep, seed=seed)
        D, Dq, lim, dm = qq["D"], qq["Dq"], qq["lim"], qq["dm"]
        ylab = f"{type} residuals"
        if Dq is None:
            # qqnorm fallback: residuals vs N(0,1) quantiles (ppoints).
            n = D.size
            a = 3.0 / 8.0 if n <= 10 else 0.5
            pp = (np.arange(1, n + 1) - a) / (n + 1.0 - 2.0 * a)
            ax.scatter(_nmath.qnorm5_vec(pp), np.sort(D), s=8,
                       facecolor="none", edgecolor="black")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel(ylab)
            ax.set_title("Normal Q-Q Plot")
            return ax
        Ds = np.sort(D)
        if lim is not None:
            ax.fill_between(Dq, lim[0], lim[1], color=rep_col, lw=0)
        elif dm is not None:
            for i in range(dm.shape[1]):
                ax.plot(Dq, dm[:, i], color=rep_col, lw=0.5)
        ax.axline((0.0, 0.0), slope=1.0, color=rl_col, lw=1.0)
        ax.scatter(Dq, Ds, s=8, facecolor="none", edgecolor="black")
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel(ylab)
        ax.set_title("QQ plot of residuals")
        return ax

    def plot_check(self, type: str = "deviance", rep: int = 0,
                   level: float = 0.9, s_rep: int = 10,
                   seed: int | None = None, figsize=None):
        """The graphical half of mgcv's ``gam.check`` (plots.r:277-288):
        qq.gam | residuals vs linear predictor | residual histogram |
        response vs fitted values. Multi-LP fits use the first linear
        predictor / first fitted column on the scatter panels, exactly
        like mgcv when the residual vector is 1-D."""
        fig, axes = plt.subplots(2, 2, figsize=figsize or (10, 8))
        self.qq_gam(type=type, rep=rep, level=level, s_rep=s_rep,
                    seed=seed, ax=axes[0, 0])
        resid = np.asarray(self.residuals_of(type), dtype=float)
        eta = np.asarray(self.linear_predictors, dtype=float)
        if eta.ndim == 2 and resid.ndim == 1:
            eta = eta[:, 0]
        axes[0, 1].scatter(eta, resid, s=8, facecolor="none",
                           edgecolor="black")
        axes[0, 1].set_xlabel("linear predictor")
        axes[0, 1].set_ylabel("residuals")
        axes[0, 1].set_title("Resids vs. linear pred.")
        axes[1, 0].hist(resid, color="0.85", edgecolor="black")
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_title("Histogram of residuals")
        fv = np.asarray(self.fitted_values, dtype=float)
        y = np.asarray(self._y_arr, dtype=float)
        if fv.ndim == 2 and y.ndim == 1:
            fv = fv[:, 0]
        axes[1, 1].scatter(fv, y, s=8, facecolor="none",
                           edgecolor="black")
        axes[1, 1].set_xlabel("Fitted Values")
        axes[1, 1].set_ylabel("Response")
        axes[1, 1].set_title("Response vs. Fitted Values")
        fig.tight_layout()
        return axes

    def plot_scale_location(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        eta = self.linear_predictors
        s = np.sqrt(np.abs(self.std_dev_residuals))
        ax.scatter(eta, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(eta, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, eta, s, scores=self.std_dev_residuals, n=label_n)
        ax.set_xlabel("Predicted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ deviance\ resid.}|}$")
        ax.set_title("Scale-Location")
        return ax

    def plot_leverage(
        self, ax=None, figsize=None,
        facecolor="none", edgecolor="black",
        cook_levels=(0.5, 1.0),
        smooth=True, label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        h = self.leverage
        r = self.std_pearson_residuals
        ax.scatter(h, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if smooth:
            xs, ys = _lowess(h, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        # Cook's contours for GAM: D_i = (r²/k)·h/(1−h), k = edf_total —
        # the GAM analogue of GLM's `rank(X)` for the Bayesian penalized
        # hat matrix. Solving for r: r = ±sqrt(c·k·(1−h)/h).
        k = max(float(self.edf_total), 1.0)
        ymin, ymax = ax.get_ylim()
        h_max = float(np.clip(h.max() * 1.1, 1e-3, 0.999))
        h_grid = np.linspace(1e-3, h_max, 200)
        for c in cook_levels:
            rline = np.sqrt(c * k * (1 - h_grid) / h_grid)
            ax.plot(h_grid, rline, color="red", linestyle="--", linewidth=0.8)
            ax.plot(h_grid, -rline, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        cook = (r ** 2 / k) * h / np.clip(1 - h, 1e-12, None)
        _label_top_n(ax, h, r, scores=cook, n=label_n)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Std. Pearson resid.")
        ax.set_title("Residuals vs. Leverage")
        return ax

    def plot_smooth(
        self,
        select=None,
        scheme=None,
        n_cols=2,
        figsize=None,
        color="black",
        band_color="black",
        band_alpha=0.2,
        rug=True,
        partial_residuals=False,
        n_grid: int = 40,
        too_far: float = 0.1,
        zlim=None,
        xlim=None,
        ylim=None,
        all_terms: bool = False,
        ax=None,
    ):
        """Per-smooth effect plots — the hea port of mgcv's ``plot.gam``.

        Auto-dispatches by smooth dimensionality:

        - **1D** ``s(x)`` → curve of ``f̂(x_i)`` with a 2·SE band, optional
          rug and partial residuals.
        - **2D** ``s(x,y)`` / ``te(x,y)`` → contour plot of ``f̂(x,y)``
          (default, ``scheme=0``) or a 3D persp wireframe (``scheme=1``).
          Contour: bold = f̂, dashed = f̂−SE, dotted = f̂+SE (matches
          mgcv's ``sp.contour`` lty=1/2/3 — note Wood 2017 Fig. 4.14's
          caption inverts the SE assignments relative to the actual mgcv
          code). Persp: white facets, black mesh, ``too_far``-masked
          grid for the irregular boundary (matches mgcv's
          ``plot.gam(scheme=1)`` look used in Wood 2017 Fig. 7.9 bottom
          row).

        With ``all_terms=True``, parametric terms get their own panels
        (mgcv's ``plot.gam(..., all.terms=TRUE)`` behavior):

        - Factor term → horizontal-bar termplot, one bar per level
          (reference level pinned at 0), with ±SE dashed bars and a rug.
        - Numeric term → linear partial effect ``β·x`` with a 2·SE band.

        Multi-block factor-by smooths (e.g. ``s(x, by=g)`` for each level
        of ``g``) appear as separate panels — same as mgcv.

        Parameters
        ----------
        select : int | str | list of int|str | None
            Which panel(s) to plot.

            - ``None`` (default): all plottable panels in their formula
              order.
            - ``int``: 0-indexed position in the plottable list.
            - ``str``: matches ``block.label`` for smooths
              (``"s(dur)"``, ``"ti(gly,bmi)"``) or the term label for
              parametric panels (``all_terms=True`` only). First match
              wins.
            - list of int/str: subset of panels in the given order.

            Required when ``ax=`` is given and the model has more than
            one plottable panel.
        scheme : int | list[int] | None
            Rendering style for 2D smooths, mgcv-style. ``0`` (default)
            = contour; ``1`` = 3D persp (wireframe, white facets, black
            edges, masked by ``too_far``). 1D smooths and parametric
            panels ignore this. A scalar applies to every selected
            panel; a list must have length equal to the number of
            selected panels.
        n_cols : int
            Columns in the grid layout when ``ax`` is None.
        partial_residuals : bool
            (1D only) Overlay partial residuals (working residual + ``f̂_i``).
        rug : bool
            (1D only) Draw a rug of x-values at the bottom of each panel.
        n_grid : int
            (2D only) Per-axis grid resolution. Default 40 (mgcv uses 30).
        too_far : float
            (2D only) Mask grid points whose normalized distance to the
            nearest data point exceeds this threshold (mgcv's
            ``exclude.too.far``). Default 0.1 matches mgcv's plot.gam
            default; set to 0 to disable masking.
        zlim : (float, float) | None
            (``scheme=1`` persp only) Shared z-axis range across all
            persp panels. Default ``None`` lets matplotlib autoscale per
            panel — visually misleading when one term has been shrunk to
            ~0 (the tiny range gets stretched to fill the panel and
            doesn't read as flat). Pass an explicit range (e.g.
            ``(-3, 3)``) to make near-zero terms render as flat plates,
            matching Wood 2017 Fig. 7.9.
        xlim, ylim : (float, float) | list | None
            Per-panel axis limits, applied after each panel is drawn.
            ``None`` (default) leaves matplotlib's autoscaling in place;
            a single ``(lo, hi)`` is applied to every selected panel; a
            list (entries ``(lo, hi)`` or ``None``) sets per-panel
            limits and must have length equal to the number of selected
            panels. The rug is anchored to the axes bottom, so it
            tracks ``ylim`` automatically. mgcv's ``plot.gam`` calls
            these ``xlim``/``ylim`` too.
        all_terms : bool
            Also include parametric terms (factor / numeric, excluding the
            intercept) — Wood 2017 Fig. 4.15 layout.
        ax : matplotlib Axes | None
            If given, draw the (single) selected panel into this axes
            instead of building a new figure. The axes must be a 3D
            ``Axes3D`` (``projection='3d'``) when the panel is a 2D
            smooth with ``scheme=1``; a regular 2D axes otherwise.
            Returns ``ax`` in that case (single-panel return
            convention); otherwise returns ``fig``.

        Returns
        -------
        Figure when building the multi-panel grid; Axes when ``ax=`` is
        provided.

        Notes
        -----
        Smooths of dimension ≥3, factor-smooth interactions (``bs="fs"``),
        and random-effect smooths (``bs="re"``) are still skipped. For ≥3D
        viewing use :meth:`vis` with ``view=`` to pick a 2D slice.
        """
        # Plottable panels: a list of dispatch records, each a tuple where
        # the first element is a discriminator string. Two kinds:
        #   ("smooth", block, a, bcol)
        #   ("param",  term_label, col_indices, kind)  kind ∈ {"factor", "numeric"}
        plottable: list[tuple] = []
        for idx, (b, (a, bcol)) in enumerate(
            zip(self._blocks, self._block_col_ranges)
        ):
            if len(b.term) not in (1, 2):
                continue
            if b.cls in ("re.smooth.spec", "fs.interaction", "sz.interaction"):
                continue
            plottable.append(("smooth", b, a, bcol))

        if all_terms and self._expanded.terms:
            param_cols = self.parametric_columns
            col_index_of = {c: i for i, c in enumerate(param_cols)}
            used = {"(Intercept)"} if "(Intercept)" in col_index_of else set()
            for term in self._expanded.terms:
                label = term.label
                term_cols = [
                    c for c in param_cols
                    if c not in used and (c == label or c.startswith(label))
                ]
                if not term_cols:
                    continue
                used.update(term_cols)
                col_idx = [col_index_of[c] for c in term_cols]
                # Classify the underlying variable: factor (Enum/Categorical/Utf8)
                # vs numeric. Skip terms whose variable can't be resolved
                # (interactions, transformed terms) — those need bespoke
                # rendering and aren't supported here yet.
                if label in self.data.columns:
                    dt = self.data[label].dtype
                    if dt in (pl.Enum, pl.Categorical, pl.Utf8):
                        plottable.append(("param", label, col_idx, "factor"))
                    elif dt.is_numeric():
                        plottable.append(("param", label, col_idx, "numeric"))
                    # else: skip (datetime, list, etc.)

        if not plottable:
            raise ValueError(
                "no plottable panels in this model; "
                "≥3D / fs / re smooths aren't supported here — try vis()"
            )

        sel_idx = self._resolve_plot_select(select, plottable)
        selected = [plottable[i] for i in sel_idx]
        schemes = self._resolve_plot_scheme(scheme, len(selected))
        xlims = self._resolve_plot_lim(xlim, len(selected), "xlim")
        ylims = self._resolve_plot_lim(ylim, len(selected), "ylim")

        wr_all = (
            self.residuals_of("working") if partial_residuals else None
        )

        def draw_panel(ax_, item, sch):
            kind = item[0]
            if kind == "smooth":
                _, block, a, bcol = item
                edf_b = float(self.edf[a:bcol].sum())
                label_inner = block.label.rstrip(")")
                title = f"{label_inner},{round(edf_b, 2):g})"
                if len(block.term) == 1:
                    self._plot_smooth_1d(
                        ax_, block, a, bcol,
                        color=color, band_color=band_color,
                        band_alpha=band_alpha,
                        rug=rug, partial_residuals=partial_residuals,
                        wr_all=wr_all, ylabel=title,
                    )
                elif sch == 1:
                    self._plot_smooth_2d_persp(
                        ax_, block, a, bcol,
                        color=color, n_grid=n_grid, too_far=too_far,
                        zlim=zlim, zlabel=title,
                    )
                else:
                    self._plot_smooth_2d(
                        ax_, block, a, bcol,
                        color=color, n_grid=n_grid, too_far=too_far,
                        title=title,
                    )
            else:  # "param"
                _, term_label, col_idx, term_kind = item
                if term_kind == "factor":
                    self._plot_parametric_factor(
                        ax_, term_label, col_idx,
                        color=color, rug=rug,
                    )
                else:
                    self._plot_parametric_numeric(
                        ax_, term_label, col_idx,
                        color=color, band_color=band_color,
                        band_alpha=band_alpha, rug=rug,
                    )

        # Single-panel target: draw into the user-supplied ax and return it.
        if ax is not None:
            if len(selected) != 1:
                raise ValueError(
                    f"ax= requires exactly one panel; have {len(selected)} "
                    f"selected panel(s). Pass select= to pick one."
                )
            item, sch = selected[0], schemes[0]
            needs_3d = (
                item[0] == "smooth" and len(item[1].term) == 2 and sch == 1
            )
            if needs_3d and not hasattr(ax, "get_zlim"):
                raise TypeError(
                    "scheme=1 (persp) on a 2D smooth requires a 3D Axes; "
                    "pass an axes built with projection='3d'."
                )
            draw_panel(ax, item, sch)
            if xlims[0] is not None:
                ax.set_xlim(xlims[0])
            if ylims[0] is not None:
                ax.set_ylim(ylims[0])
            return ax

        n_plots = len(selected)
        n_cols_eff = 1 if n_plots == 1 else min(n_cols, n_plots)
        n_rows = (n_plots + n_cols_eff - 1) // n_cols_eff
        any_persp = any(
            item[0] == "smooth" and len(item[1].term) == 2 and sch == 1
            for item, sch in zip(selected, schemes)
        )
        if figsize is None:
            # Persp panels need extra width: matplotlib's 3D backend doesn't
            # report the zlabel's bbox to the layout engine, so a vanilla
            # tight/constrained layout clips the rightmost zlabel. Pad both
            # the per-panel width and the inter-panel spacing.
            w = 6.0 if any_persp else 5
            figsize = (w * n_cols_eff, 4 * n_rows)
        fig = plt.figure(figsize=figsize)
        for plot_i, (item, sch) in enumerate(zip(selected, schemes)):
            needs_3d = (
                item[0] == "smooth" and len(item[1].term) == 2 and sch == 1
            )
            proj = "3d" if needs_3d else None
            ax_i = fig.add_subplot(
                n_rows, n_cols_eff, plot_i + 1, projection=proj
            )
            draw_panel(ax_i, item, sch)
            if xlims[plot_i] is not None:
                ax_i.set_xlim(xlims[plot_i])
            if ylims[plot_i] is not None:
                ax_i.set_ylim(ylims[plot_i])

        if any_persp:
            # Hard-coded margins: leave 8% on the right (zlabel of the
            # rightmost panel sits there) and 4% on the other sides;
            # ~25% wspace between panels. Jupyter's inline backend renders
            # with bbox_inches='tight' which can crop zlabels right up to
            # the panel edge — the wider right margin gives them room.
            fig.subplots_adjust(
                left=0.04, right=0.92, bottom=0.06, top=0.96,
                wspace=0.25, hspace=0.25,
            )
        else:
            fig.tight_layout()
        return fig

    def _resolve_plot_select(self, select, plottable):
        """Map ``select=`` (None / int / str / list of those) to a list of
        indices into ``plottable``. String names match ``block.label`` for
        smooth panels and the term label for parametric panels.
        """
        if select is None:
            return list(range(len(plottable)))
        items = select if isinstance(select, (list, tuple)) else [select]
        out = []
        for s in items:
            if isinstance(s, bool):
                raise TypeError(
                    f"select entries must be int or str, got bool ({s!r})"
                )
            if isinstance(s, (int, np.integer)):
                i = int(s)
                if not (0 <= i < len(plottable)):
                    raise IndexError(
                        f"select={i} out of range; have {len(plottable)} "
                        "plottable panel(s)"
                    )
                out.append(i)
            elif isinstance(s, str):
                matched = None
                for j, item in enumerate(plottable):
                    name = item[1].label if item[0] == "smooth" else item[1]
                    if name == s:
                        matched = j
                        break
                if matched is None:
                    avail = [
                        item[1].label if item[0] == "smooth" else item[1]
                        for item in plottable
                    ]
                    raise ValueError(
                        f"select={s!r} doesn't match any plottable panel; "
                        f"have {avail}"
                    )
                out.append(matched)
            else:
                raise TypeError(
                    f"select entries must be int or str, got "
                    f"{type(s).__name__}"
                )
        return out

    def _resolve_plot_scheme(self, scheme, n_panels):
        """Map ``scheme=`` (None / int / list of int) to a list of length
        ``n_panels``.
        """
        if scheme is None:
            return [0] * n_panels
        if isinstance(scheme, (list, tuple)):
            if len(scheme) != n_panels:
                raise ValueError(
                    f"scheme list must have length {n_panels} (one per "
                    f"selected panel); got {len(scheme)}"
                )
            return [int(s) for s in scheme]
        return [int(scheme)] * n_panels

    @staticmethod
    def _resolve_plot_lim(lim, n_panels, name):
        """Map ``xlim=``/``ylim=`` (None / (lo, hi) / list of those) to a
        list of length ``n_panels``. A single ``(lo, hi)`` broadcasts to
        every panel; a list must align with the selection (entries may be
        ``None`` to skip a specific panel).
        """
        if lim is None:
            return [None] * n_panels

        def _is_pair(v):
            return (
                isinstance(v, (list, tuple))
                and len(v) == 2
                and all(isinstance(x, (int, float, np.number)) for x in v)
            )

        if _is_pair(lim):
            return [tuple(lim)] * n_panels
        if isinstance(lim, list):
            if len(lim) != n_panels:
                raise ValueError(
                    f"{name} list must have length {n_panels} (one per "
                    f"selected panel); got {len(lim)}"
                )
            out = []
            for i, v in enumerate(lim):
                if v is None:
                    out.append(None)
                elif _is_pair(v):
                    out.append(tuple(v))
                else:
                    raise TypeError(
                        f"{name}[{i}] must be (lo, hi) or None; got {v!r}"
                    )
            return out
        raise TypeError(
            f"{name}= must be None, (lo, hi), or a list of (lo, hi)/None; "
            f"got {type(lim).__name__}"
        )

    def _plot_smooth_1d(
        self, ax, block, a, bcol, *,
        color, band_color, band_alpha, rug, partial_residuals,
        wr_all, ylabel,
    ):
        """1D smooth panel: curve + 2·SE band + optional rug / partial residuals."""
        cov_name = block.term[0]
        x = self.data[cov_name].to_numpy().astype(float).flatten()
        B = block.X
        beta = self._beta[a:bcol]
        Vp = self.Vp[a:bcol, a:bcol]
        fhat = B @ beta
        # Var(f̂_i) = B_i · Vp · B_iᵀ; rowwise.
        var_f = ((B @ Vp) * B).sum(axis=1)
        se_f = np.sqrt(np.clip(var_f, 0.0, None))

        # Factor-by basis is zero outside the level: filter to where
        # the smooth is actually evaluated, otherwise we get a flat-0
        # line through the masked rows.
        active = np.any(np.abs(B) > 0, axis=1)
        xa = x[active]
        fa = fhat[active]
        sa = se_f[active]

        order = np.argsort(xa)
        xs, fs, ses = xa[order], fa[order], sa[order]

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(
            xs, fs - 2 * ses, fs + 2 * ses,
            color=band_color, alpha=band_alpha, linewidth=0,
        )
        ax.plot(xs, fs, color=color, linewidth=1.0)

        if partial_residuals:
            pr = wr_all[active] + fa
            ax.scatter(
                xa, pr, facecolor="none", edgecolor="grey",
                s=10, alpha=0.5,
            )

        if rug:
            # Anchor at axes-fraction y=0 (data x) so the rug follows any
            # later ylim change instead of stranding at the original ymin.
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(
                xa, np.zeros_like(xa), "|", transform=trans,
                color="black", markersize=6, alpha=0.6,
            )

        ax.set_xlabel(cov_name)
        ax.set_ylabel(ylabel)

    def _plot_smooth_2d(
        self, ax, block, a, bcol, *,
        color, n_grid, too_far, title,
    ):
        """2D smooth panel: three-contour view (estimate / +SE / −SE) plus
        data-location scatter. Mirrors mgcv's ``plot.gam`` for ``s(x,y)`` /
        ``te(x,y)`` smooths: bold = f̂, **dashed = f̂−SE**, **dotted = f̂+SE**
        (matches the lty=1/2/3 assignments in mgcv's ``sp.contour``; note
        Wood 2017 Fig. 4.14's caption swaps these — the code is the truth).
        Levels are shared across the three layers (so the same contour
        value lines up bold/dashed/dotted, ±SE apart) and labeled with
        their numeric value (mgcv default).
        """
        from matplotlib.ticker import MaxNLocator

        x_name, y_name = block.term
        x_data = self.data[x_name].to_numpy().astype(float)
        y_data = self.data[y_name].to_numpy().astype(float)

        x_grid = np.linspace(np.nanmin(x_data), np.nanmax(x_data), n_grid)
        y_grid = np.linspace(np.nanmin(y_data), np.nanmax(y_data), n_grid)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_df = pl.DataFrame({
            x_name: XX.flatten(),
            y_name: YY.flatten(),
        })

        # Smooth-only basis at the grid; β and Vp slices restricted to the
        # block so the contours show f̂(x,y), not the full η.
        B = self._block_predict_mat(block, grid_df)
        beta = self._beta[a:bcol]
        Vp = self.Vp[a:bcol, a:bcol]
        fit = (B @ beta).reshape(XX.shape)
        var_f = np.einsum("ij,jk,ik->i", B, Vp, B).reshape(XX.shape)
        se_f = np.sqrt(np.maximum(var_f, 0.0))

        if too_far > 0.0:
            mask = _too_far_mask(
                XX.flatten(), YY.flatten(),
                self.data[x_name], self.data[y_name], too_far,
            ).reshape(XX.shape)
            fit = np.where(mask, np.nan, fit)
            se_f = np.where(mask, np.nan, se_f)

        # Pick mgcv-style "pretty" round levels covering the union of
        # f̂, f̂+SE and f̂−SE so the same level value renders bold/dashed/
        # dotted across the three layers (mgcv plot.gam convention).
        zmin = float(np.nanmin(fit - se_f))
        zmax = float(np.nanmax(fit + se_f))
        # nbins=15 lets the locator choose a 0.2-spaced step over a [-1, 1]
        # range (matches mgcv's plot.gam default density).
        levels = MaxNLocator(nbins=15, steps=[1, 2, 5, 10]).tick_values(zmin, zmax)

        # ``linestyles="solid"`` overrides matplotlib's default of switching
        # negative-valued contours to dashed (rcParams["contour.negative_
        # linestyle"]) — R's contour() doesn't do that, so the bold lines
        # would otherwise visually mix with the f̂−SE dashed layer.
        cs_fit = ax.contour(XX, YY, fit,        levels=levels,
                            colors=color, linestyles="solid", linewidths=1.4)
        ax.contour(XX, YY, fit - se_f,          levels=levels,
                   colors=color, linestyles="--", linewidths=0.6)  # lty=2
        ax.contour(XX, YY, fit + se_f,          levels=levels,
                   colors=color, linestyles=":",  linewidths=0.6)  # lty=3
        ax.clabel(cs_fit, inline=True, fontsize=8, fmt="%g")
        ax.scatter(x_data, y_data, s=10, color=color)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(title)

    def _plot_smooth_2d_persp(
        self, ax, block, a, bcol, *,
        color, n_grid, too_far, zlim, zlabel,
    ):
        """2D smooth as a 3D persp wireframe — mgcv's ``plot.gam(scheme=1)``.

        White facets, black mesh, masked outside the data convex hull when
        ``too_far > 0`` (NaNs become holes in ``plot_surface``). Used in
        Wood 2017 Fig. 7.9 bottom row.
        """
        x_name, y_name = block.term
        x_data = self.data[x_name].to_numpy().astype(float)
        y_data = self.data[y_name].to_numpy().astype(float)

        x_grid = np.linspace(np.nanmin(x_data), np.nanmax(x_data), n_grid)
        y_grid = np.linspace(np.nanmin(y_data), np.nanmax(y_data), n_grid)
        XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
        grid_df = pl.DataFrame({
            x_name: XX.flatten(),
            y_name: YY.flatten(),
        })

        B = self._block_predict_mat(block, grid_df)
        beta = self._beta[a:bcol]
        Z = (B @ beta).reshape(XX.shape)

        if too_far > 0.0:
            mask = _too_far_mask(
                XX.flatten(), YY.flatten(),
                self.data[x_name], self.data[y_name], too_far,
            ).reshape(XX.shape)
            Z = np.where(mask, np.nan, Z)

        ax.plot_surface(
            XX, YY, Z,
            color="white", edgecolor=color,
            linewidth=0.3, shade=False,
        )
        if zlim is not None:
            ax.set_zlim(*zlim)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(zlabel)

    def _plot_parametric_factor(
        self, ax, label: str, col_idx: list[int], *, color, rug: bool,
    ):
        """Termplot for a factor parametric term — Wood 2017 Fig. 4.15
        right panel. Reference level pinned at 0 (default treatment
        contrasts); other levels show β̂ as a solid horizontal bar with
        ±SE dashed bars. Optional rug along the bottom (one tick per
        observation, aggregated by level).
        """
        series = self.data[label]
        if isinstance(series.dtype, pl.Enum):
            levels = list(series.dtype.categories)
        elif isinstance(series.dtype, pl.Categorical):
            levels = sorted(series.unique().drop_nulls().to_list())
        else:
            # Utf8 fallback — sort alphabetically (matches R's default).
            levels = sorted(series.unique().drop_nulls().to_list())

        ests = [0.0]
        ses = [0.0]
        cols = self.parametric_columns
        for lvl in levels[1:]:
            col_name = f"{label}{lvl}"
            if col_name in cols:
                i = cols.index(col_name)
                ests.append(float(self._beta[i]))
                ses.append(float(np.sqrt(max(self.Vp[i, i], 0.0))))
            else:
                ests.append(float("nan"))
                ses.append(float("nan"))

        half = 0.35
        for i, (est, s) in enumerate(zip(ests, ses)):
            xL, xR = i - half, i + half
            ax.plot([xL, xR], [est, est], color=color, linewidth=1.2)
            if s > 0:
                ax.plot([xL, xR], [est + s, est + s], color=color,
                        linestyle="--", linewidth=0.7)
                ax.plot([xL, xR], [est - s, est - s], color=color,
                        linestyle="--", linewidth=0.7)

        if rug:
            # Spread rug ticks within each level so the count is visible
            # (mgcv's plot.gam uses ``rug(jitter(x))`` for the same effect;
            # we lay them out deterministically across [i±half_rug] instead
            # of jittering randomly).
            pos = {lv: i for i, lv in enumerate(levels)}
            obs_levels = self.data[label].drop_nulls().to_list()
            counts: dict = {}
            for v in obs_levels:
                if v in pos:
                    counts[v] = counts.get(v, 0) + 1
            half_rug = 0.2
            xs_list: list[float] = []
            for lvl in levels:
                n_obs = counts.get(lvl, 0)
                if n_obs == 0:
                    continue
                i = pos[lvl]
                if n_obs == 1:
                    xs_list.append(float(i))
                else:
                    xs_list.extend(
                        np.linspace(i - half_rug, i + half_rug, n_obs).tolist()
                    )
            if xs_list:
                xs = np.asarray(xs_list)
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                ax.plot(xs, np.zeros_like(xs), "|", transform=trans,
                        color="black", markersize=6, alpha=0.6)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(lev) for lev in levels])
        ax.set_xlabel(label)
        ax.set_ylabel(f"Partial for {label}")

    def _plot_parametric_numeric(
        self, ax, label: str, col_idx: list[int], *,
        color, band_color, band_alpha, rug: bool,
    ):
        """Linear partial effect for a numeric parametric term — ``β̂·x``
        with a 2·SE band (mgcv's termplot for non-factor terms).
        """
        i = col_idx[0]
        beta_x = float(self._beta[i])
        se_x = float(np.sqrt(max(self.Vp[i, i], 0.0)))
        x = self.data[label].drop_nulls().to_numpy().astype(float)
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        fhat = beta_x * x_grid
        se_fhat = se_x * np.abs(x_grid)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.fill_between(
            x_grid, fhat - 2 * se_fhat, fhat + 2 * se_fhat,
            color=band_color, alpha=band_alpha, linewidth=0,
        )
        ax.plot(x_grid, fhat, color=color, linewidth=1.0)
        if rug:
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(x, np.zeros_like(x), "|", transform=trans,
                    color="black", markersize=6, alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(f"Partial for {label}")

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel ``plot.lm``-style diagnostic (residuals vs fitted,
        normal QQ, scale-location, leverage) — hea's lm/glm panel applied
        to the GAM. mgcv's gam.check panel is ``plot_check`` (drawn by
        ``check()``); per-smooth effect curves (mgcv's plot.gam) are
        ``plot_smooth``; the 2D fitted-surface viewer (vis.gam) is
        ``vis``.
        """
        if getattr(self, "_md", None) is not None:
            raise NotImplementedError(
                "the lm-style diagnostic panel needs scalar-response GLM "
                "quantities (hat values, standardized residuals) that "
                "multi-LP general-family fits don't define; use check() "
                "/ plot_check() and plot_smooth() instead."
            )
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        self.plot_leverage(ax=axes[1, 1], smooth=smooth, label_n=label_n)
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------
# module-private helpers
# --------------------------------------------------------------------------


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    flat = np.asarray(values, dtype=float).reshape(-1)
    # Row-oriented construction is ~4× faster than the dict-of-singletons
    # build (one Series per column) for these 1×p reporting frames. Fall
    # back to the dict form for the degenerate empty / duplicate-name cases,
    # which the row form shapes (1×0 vs 0×0) or errors on differently.
    if columns and len(set(columns)) == len(columns):
        return pl.DataFrame([flat], schema=list(columns), orient="row")
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


def _add_null_space_penalties(blocks: list[SmoothBlock]) -> list[SmoothBlock]:
    """Mirror mgcv's ``null.space.penalty=TRUE`` (``gam(..., select=TRUE)``).

    For each block, append a rank-``(p − rank_S)`` matrix that penalizes the
    null-space directions of the existing combined penalty ``Σⱼ Sⱼ`` to the
    block's ``S`` list. With this extra penalty plus its own smoothing
    parameter, the term can be shrunk to zero — that's the whole point of
    ``select=TRUE``. After augmentation the per-block combined penalty is
    full-rank, so the smooth's null-space dim is zero and ``_Mp`` collapses
    to ``p_param``.

    Implements the ``need.full`` eigendecomposition branch of mgcv's
    ``smoothCon`` (R/smooth.r): ``St = Σⱼ Sⱼ``, eigendecompose, take the
    eigenvectors ``U`` with eigenvalues below ``max_eig · ε^0.66``, and use
    ``Sf = U Uᵀ`` (the projection onto the null space). Mgcv's fast path
    for ``nsm=1`` plus a diagonal-canonical ``S`` produces the same ``Sf``
    when applicable; this routine takes the eigen path unconditionally,
    which is bit-equal up to LAPACK's choice of basis for repeated
    eigenvalues — and ``U Uᵀ`` is invariant to that choice.

    No rescaling: mgcv assigns ``S.scale = 1`` to ``Sf`` (left at unit
    norm), in contrast to the per-S ``maXX/‖S‖`` rescaling that
    ``_scale_penalty`` applied to the original penalties.
    """
    eps = float(np.finfo(float).eps)
    threshold_factor = eps ** 0.66
    out: list[SmoothBlock] = []
    for b in blocks:
        S_list = [np.asarray(s, dtype=float) for s in b.S]
        if not S_list:
            out.append(b)
            continue
        St = S_list[0].copy()
        for Sj in S_list[1:]:
            St += Sj
        eigvals, eigvecs = np.linalg.eigh(St)
        max_eig = float(eigvals.max()) if eigvals.size else 0.0
        if max_eig <= 0.0:
            out.append(b)
            continue
        null_mask = eigvals < max_eig * threshold_factor
        if not bool(np.any(null_mask)):
            out.append(b)
            continue
        U = eigvecs[:, null_mask]
        Sf = U @ U.T
        Sf = 0.5 * (Sf + Sf.T)
        out.append(SmoothBlock(
            label=b.label, term=b.term, cls=b.cls,
            X=b.X, S=S_list + [Sf], spec=b.spec,
            S_scale=(None if b.S_scale is None
                     else list(b.S_scale) + [1.0]),
        ))
    return out


def _apply_gam_side(blocks: list[SmoothBlock]) -> list[SmoothBlock]:
    """Apply mgcv's ``gam.side`` identifiability surgery.

    For each block ``b`` whose variable set strictly contains another
    block's (e.g. ``te(x1, x2)`` over ``s(x1) + s(x2)``), some columns of
    ``X_b`` are linearly dependent on the union of the smaller smooths'
    designs plus the intercept. mgcv finds those columns via
    ``fixDependence`` (a QR with column pivoting on the residual after
    projecting out the smaller smooths) and **deletes** them — both from
    ``X_b`` and from the rows/cols of each ``S_b[j]``. For a default
    ``te(x1, x2)`` with ``s(x1) + s(x2)`` marginals, this drops exactly 2
    columns (24 → 22), matching ``ncol(model.matrix(m))``.

    Random-effect smooths (``bs='re'``) carry ``side.constrain=FALSE`` in
    mgcv: their identity penalty already identifies the fit even with a
    rank-deficient X, so gam.side neither constrains them nor includes
    them in X1 when constraining other blocks. Replicating that here
    matters for `s(Worker, bs='re') + s(Machine, Worker, bs='re')` style
    nestings — dropping the 6 dependent interaction columns shifts the
    REML surface (different log|A|, log|S|+) and lands at a different
    optimum than mgcv. Skipping the surgery keeps the design at p=27
    (matching mgcv) at the cost of a rank-deficient X that's still PD
    once Sλ = λ·I is added in the re block.
    """
    if len(blocks) < 2:
        return blocks
    var_sets = [frozenset(b.term) for b in blocks]
    n = int(np.asarray(blocks[0].X).shape[0])
    out: list[SmoothBlock] = []
    for i, b in enumerate(blocks):
        if not _side_constrain(b):
            out.append(b)
            continue
        my_vars = var_sets[i]
        Xb = np.asarray(b.X, dtype=float)
        # X1 = intercept + every strict-subset, side-constrained block's
        # design — exactly what `gam.side` builds before `fixDependence`.
        cols_X1 = [np.ones((n, 1))]
        for j, other in enumerate(blocks):
            if i == j or not _side_constrain(other):
                continue
            if var_sets[j] and var_sets[j] < my_vars:
                cols_X1.append(np.asarray(other.X, dtype=float))
        if len(cols_X1) == 1:
            out.append(b)
            continue
        X1 = np.concatenate(cols_X1, axis=1)
        ind = _fix_dependence(X1, Xb)
        if not ind:
            out.append(b)
            continue
        keep = [c for c in range(Xb.shape[1]) if c not in ind]
        new_X = Xb[:, keep]
        new_S = []
        for Sj in b.S:
            Sj = np.asarray(Sj, dtype=float)
            new_S.append(Sj[np.ix_(keep, keep)])
        if b.spec is None:
            new_spec = None
        else:
            # Compose with any prior keep_cols so re-running gam.side is idempotent.
            keep_arr = np.asarray(keep, dtype=np.intp)
            prior = b.spec.keep_cols
            new_keep = keep_arr if prior is None else prior[keep_arr]
            new_spec = BasisSpec(
                raw=b.spec.raw, by=b.spec.by, absorb=b.spec.absorb,
                keep_cols=new_keep,
            )
        out.append(SmoothBlock(
            label=b.label, term=b.term, cls=b.cls, X=new_X, S=new_S,
            spec=new_spec,
        ))
    return out


def _side_constrain(b: SmoothBlock) -> bool:
    """Mirrors mgcv's ``smooth$side.constrain``. Random-effect smooths
    (``re.smooth.spec``) opt out — their identity penalty handles ID."""
    return b.cls != "re.smooth.spec"


def _fix_dependence(X1: np.ndarray, X2: np.ndarray,
                    tol: float = float(np.finfo(float).eps) ** 0.5) -> list[int]:
    """Find columns of ``X2`` that are linearly dependent on ``X1``.

    Mirrors mgcv's ``fixDependence(X1, X2, tol)`` (non-strict mode):

    1. ``Q1 R1 = X1`` (QR of X1).
    2. Project X2 onto the orthogonal complement of X1's column space
       and take the bottom block of ``Q1ᵀ X2`` (rows ``r+1..n``).
    3. QR of that residual *with column pivoting*. Trailing columns
       whose mean abs over the diagonal block falls below
       ``|R1[0,0]| · tol`` are the dependent ones — return their pivot
       indices in X2.
    """
    n, r = X1.shape
    Q1, R1 = np.linalg.qr(X1, mode="complete")
    if R1.size == 0 or n <= r:
        return []
    R11 = abs(R1[0, 0]) if R1.shape[0] > 0 else 1.0
    QtX2 = Q1.T @ X2
    residual = QtX2[r:, :]
    if residual.shape[0] == 0:
        return []
    # column-pivoted QR via scipy (numpy's qr lacks pivoting)
    from scipy.linalg import qr as scipy_qr
    Q2, R2, piv = scipy_qr(residual, mode="economic", pivoting=True)
    nrows = R2.shape[0]
    r_full = nrows
    r0 = r_full
    while r0 > 0 and float(np.mean(np.abs(R2[r0 - 1: r_full, r0 - 1: r_full]))) < R11 * tol:
        r0 -= 1
    r0 += 1
    if r0 > r_full:
        return []
    return [int(p) for p in piv[r0 - 1: r_full]]


def _sym_rank(S: np.ndarray) -> int:
    """Numerical rank of a symmetric matrix via eigendecomposition."""
    if S.size == 0:
        return 0
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    if w.size == 0:
        return 0
    tol = max(1e-12, w.max() * 1e-10) if w.max() > 0 else 1e-12
    return int(np.sum(w > tol))


def _r_cond(R: np.ndarray) -> float:
    """mgcv ``R_cond`` (gdi.c:2851) — ∞-norm condition estimate of an
    upper-triangular ``R``, by the Cline-Moler-Stewart-Wilkinson (1979)
    growth recursion (Golub & Van Loan 1996): solve ``R'y = ±e`` choosing
    each sign to maximize growth, then ``κ ≈ ‖R‖_∞ · ‖y‖_∞``.
    """
    c = R.shape[1]
    if c == 0:
        return 0.0
    y = np.zeros(c)
    p_acc = np.zeros(c)
    y_inf = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for k in range(c - 1, -1, -1):
            yp = (1.0 - p_acc[k]) / R[k, k]
            ym = (-1.0 - p_acc[k]) / R[k, k]
            pp = p_acc[:k] + R[:k, k] * yp
            pm = p_acc[:k] + R[:k, k] * ym
            if abs(yp) + float(np.abs(pp).sum()) >= \
                    abs(ym) + float(np.abs(pm).sum()):
                y[k] = yp
                p_acc[:k] = pp
            else:
                y[k] = ym
                p_acc[:k] = pm
            kappa = abs(y[k])
            if kappa > y_inf:
                y_inf = kappa
        R_inf = float(np.abs(np.triu(R[:c, :c])).sum(axis=1).max())
    return R_inf * y_inf


def _R_rank(R: np.ndarray,
            tol: float = float(np.finfo(float).eps) ** 0.5) -> int:
    """mgcv ``Rrank`` (mgcv.r:4-17): rank of a *pivoted* upper-triangular
    ``R`` by reducing rank until the Cline condition estimate of the
    leading block satisfies ``κ · tol < 1``.

    ``tol`` defaults to ``gam.control(rank.tol = eps^0.5)`` — magic's
    value. The gam.fit3 fitting path overrides to ``eps*100``
    (gam.fit3.r:133) before calling pls_fit1/gdi1 — callers on that path
    pass it explicitly. ``Rrank`` the R function defaults to ``eps^0.9``
    for its own callers.
    """
    rank = min(R.shape[0], R.shape[1])
    while rank > 0:
        rcond = _r_cond(R[:rank, :rank])
        if rcond * tol < 1.0:
            break
        rank -= 1
    return rank


def _pls_rank_drop(Xw: np.ndarray, slots: list["_PenaltySlot"],
                   p: int) -> tuple[int, np.ndarray]:
    """mgcv's fitting-rank determination from ``gdiPK``
    (gdi.c:1740-1775): stack the R factor of QR(√W·X) on the *balanced*
    penalty square root (totalPenaltySpace's E — λ-independent, so
    identifiability doesn't drift with λ̂), each part divided by its own
    Frobenius norm; pivoted QR of the stack; reduce rank by the Cline
    condition estimate at ``rank.tol = √eps``; the dropped columns are
    the trailing pivots (sorted ascending, like gdi.c's qsort).
    Exact-alias columns (bit-identical up to sign) are canonicalized so
    the earliest twin is kept on every BLAS build — see below.
    Returns ``(rank, drop)``."""
    R1 = np.linalg.qr(Xw, mode="r")
    St = np.zeros((p, p))
    for slot in slots:
        a, b = slot.col_start, slot.col_end
        nrm = float(np.sqrt(np.sum(slot.S * slot.S)))
        if nrm > 0:
            St[a:b, a:b] += slot.S / nrm
    if np.any(St):
        ev, Y = np.linalg.eigh(0.5 * (St + St.T))
        keep = ev > ev.max() * np.finfo(float).eps ** 0.66
        E = (Y[:, keep] * np.sqrt(ev[keep])).T        # E'E = St
    else:
        E = np.zeros((0, p))
    R1_norm = float(np.sqrt(np.sum(R1 * R1)))
    parts = [R1 / R1_norm if R1_norm > 0 else R1]
    if E.shape[0] > 0:
        E_norm = float(np.sqrt(np.sum(E * E)))
        if E_norm > 0:
            parts.append(E / E_norm)
    aug = np.vstack(parts)
    # dgeqp3's pivot choice between columns with *tied* partial norms is
    # decided by blocked-kernel rounding noise, so which twin of an
    # exact-alias pair (bit-identical, possibly negated, design columns)
    # lands in the trailing pivots varies by BLAS build — Accelerate and
    # OpenBLAS disagree, and mgcv inherits the very same tie from R's
    # LAPACK. Reference LAPACK (R's default — where the pins were
    # generated) keeps the earliest twin: tied maxima go to the first
    # index, and bit-identical columns stay bit-identical through its
    # unblocked downdates. Canonicalize to that convention: zero the
    # later twins' aug columns — a zero column can never be pivoted
    # ahead of an independent one, so it drops on every platform.
    # Detection runs on (Xw, E), before any factorization has a chance
    # to ULP-split the twins; the exact-norm prefilter means only tied
    # columns are ever byte-compared. Scaled aliases (≠ ±1) tie nothing
    # — dgeqp3 keeps the larger-norm copy deterministically everywhere.
    if p > 1:
        cn = np.einsum("ij,ij->j", Xw, Xw)
        if E.shape[0]:
            cn = cn + np.einsum("ij,ij->j", E, E)
        norm_groups: dict[float, list[int]] = {}
        for j in range(p):
            norm_groups.setdefault(float(cn[j]), []).append(j)
        for group in norm_groups.values():
            if len(group) < 2:
                continue
            seen: dict[tuple[bytes, bytes], int] = {}
            for j in group:
                key = (Xw[:, j].tobytes(), E[:, j].tobytes())
                neg = ((-Xw[:, j]).tobytes(), (-E[:, j]).tobytes())
                if key in seen or neg in seen:
                    aug[:, j] = 0.0
                else:
                    seen[key] = j
    from scipy.linalg import qr as _scipy_qr
    R_piv, piv = _scipy_qr(aug, mode="r", pivoting=True)
    # gam.fit3 overrides the fitting-path rank tolerance to eps*100
    # (gam.fit3.r:133) — both pls_fit1 and gdi1 receive that value, not
    # gam.control's √eps (which magic's GCV path uses).
    rank = _R_rank(R_piv, tol=float(np.finfo(float).eps) * 100.0)
    drop = np.sort(piv[rank:]) if rank < p else np.zeros(0, dtype=int)
    # R1 (the n×q QR factor of √W·X) is returned so the magic path can reuse
    # it for getRpqr instead of factoring √w·X a second time — they are the
    # bit-identical matrix under gaussian-identity (W constant) with no drop.
    return rank, drop, R1


def _mroot(A: np.ndarray, rank: int | None = None,
           method: str = "chol") -> np.ndarray:
    """mgcv ``mroot(A, rank, method)`` (mgcv.r:4444-4470): B with
    ``B @ B.T = A`` and ``rank`` columns. ``method="chol"`` uses pivoted
    Cholesky (LAPACK dpstrf — R's ``chol(pivot=TRUE)``); ``method="svd"``
    uses the symmetric eigendecomposition (mgcv's own shortcut: "same as
    svd for +ve semi def, but faster"), detecting rank from
    ``values > max(values)·eps`` when not supplied. Non-symmetric input
    stops, per mgcv's ``isTRUE(all.equal(A, t(A)))`` guard (mean relative
    difference < 1.5e-8, ``all.equal.numeric`` semantics)."""
    q = A.shape[0]
    if q == 0:
        return np.zeros((0, 0))
    # isTRUE(all.equal(A, t(A))): relative when mean|A| > tol, else absolute.
    asym = float(np.mean(np.abs(A - A.T)))
    a_scale = float(np.mean(np.abs(A)))
    tol = 1.5e-8
    if (asym / a_scale if a_scale > tol else asym) >= tol:
        raise ValueError("Supplied matrix not symmetric")
    if method == "svd":
        vals, vecs = np.linalg.eigh(A)
        vals = vals[::-1]           # R's eigen(): descending order
        vecs = vecs[:, ::-1]
        if rank is None or rank < 1:
            rank = int(np.sum(vals > vals.max() * np.finfo(float).eps))
        if rank == 0:
            raise ValueError(
                "Something wrong - matrix probably not +ve semi definite"
            )
        return vecs[:, :rank] * np.sqrt(vals[:rank])
    elif method == "chol":
        from scipy.linalg.lapack import dpstrf
        c, piv, r_eff, _info = dpstrf(A, lower=0)
        U = np.triu(c)
        if r_eff < q:
            U[r_eff:, :] = 0.0
        if rank is None or rank < 1:
            rank = int(r_eff)
        inv_piv = np.empty(q, dtype=int)
        inv_piv[piv - 1] = np.arange(q)
        return U[:rank, inv_piv].T
    raise ValueError("method not recognised.")


def _qr_ldet_inv(X: np.ndarray, get_inv: bool) -> tuple[float, np.ndarray | None]:
    """mgcv ``qr_ldet_inv`` (gdi.c:257): ``log|X|`` and, optionally, the
    inverse of ``X`` via pivoted QR.

    ``log|X| = sum_i log|R_ii|`` from the column-pivoted QR ``X[:, piv] = Q R``
    (LAPACK dgeqp3, matching mgcv's ``mgcv_qr``). When ``get_inv``, the inverse
    is ``X^{-1} = P R^{-1} Q'`` — ``solve_triangular`` for ``R^{-1} Q'`` then row
    unpivot by ``piv`` (mgcv's column-by-column ``pivot[i]`` permutation).

    Replaces a ``np.linalg.slogdet``/``np.linalg.inv`` (LU) shortcut: LU and QR
    agree to rounding on the log-determinant, but numpy's ``slogdet`` sets the
    divide-by-zero/overflow FP flags (raising spurious RuntimeWarnings) even on
    healthy, well-conditioned ``S``; the QR path mgcv actually uses does not.
    """
    Q, R, piv = qr(X, pivoting=True)
    ldet = float(np.sum(np.log(np.abs(np.diag(R)))))
    Xi = None
    if get_inv:
        RiQt = solve_triangular(R, Q.T, lower=False)
        Xi = np.empty_like(X)
        Xi[piv, :] = RiQt
    return ldet, Xi


def _get_stable_S(rS_list: list[np.ndarray], sp: np.ndarray, deriv: int,
                  d_tol: float, r_tol: float,
                  fixed_penalty: bool = False):
    """mgcv ``get_stableS`` (gdi.c:550-792), line-by-line: similarity
    transform of ``S = Σ λ_i S_i`` (S_i = rS_i rS_i') that prevents
    "dominant machine zero leakage" between penalty components when the
    λ's differ by ≫ 1/eps, plus log|S| and its first/second derivatives
    wrt log(sp) computed in the transformed (stable) space.

    Iteratively: group the components whose ``‖S_i‖_F·λ_i`` is within
    ``d_tol`` of the largest (the dominant set α), eigen-split their
    weighted sum, fold the dominant range into the output basis, and
    recurse on the null-space projection of the sub-dominant set.

    ``rS_list`` are the component square roots **in the total-penalty
    range basis** (mgcv's UrS — see totalPenaltySpace), with the fixed
    penalty's root last when ``fixed_penalty``. Returns
    ``(S, Qf, rS_transformed, det, det1, det2)`` with
    ``S = Qf' S0 Qf``."""
    q = rS_list[0].shape[0]
    Mf = len(rS_list)
    M = Mf - 1 if fixed_penalty else Mf
    spf = np.ones(Mf)
    spf[:M] = np.asarray(sp, dtype=float)
    rS = [np.asarray(r, dtype=float).copy() for r in rS_list]
    Si = [r @ r.T for r in rS]              # shrinking Q×Q blocks
    gamma = np.ones(Mf, dtype=bool)
    K = 0
    Q = q
    S = np.zeros((q, q))
    Qf = np.zeros((q, q))
    it = 0
    while True:
        it += 1
        frob = np.zeros(Mf)
        max_frob = 0.0
        for i in range(Mf):
            if gamma[i]:
                frob[i] = float(np.linalg.norm(Si[i], "fro"))
                if frob[i] * spf[i] > max_frob:
                    max_frob = frob[i] * spf[i]
        alpha = np.zeros(Mf, dtype=bool)
        gamma1 = np.zeros(Mf, dtype=bool)
        for i in range(Mf):
            if gamma[i]:
                if frob[i] * spf[i] > max_frob * d_tol:
                    alpha[i] = True
                else:
                    gamma1[i] = True
        if gamma1.any():
            # rank of the (1/frob-scaled) dominant sum
            Sb = np.zeros((Q, Q))
            for i in range(Mf):
                if alpha[i]:
                    Sb += Si[i] / frob[i]
            ev_asc = np.linalg.eigvalsh(Sb)          # ascending
            r = 1
            while r < Q and ev_asc[Q - r - 1] > ev_asc[Q - 1] * r_tol:
                r += 1
        else:
            r = Q
        if Q == r:
            if it == 1:
                for i in range(Mf):
                    S += spf[i] * Si[i]
                Qf = np.eye(q)
            break
        # dominant term, eigen-decomposed descending
        Sb = np.zeros((Q, Q))
        for i in range(Mf):
            if alpha[i]:
                Sb += spf[i] * Si[i]
        ev, U = np.linalg.eigh(Sb)
        ev = ev[::-1].copy()
        U = np.ascontiguousarray(U[:, ::-1])
        if it == 1:
            Qf = U.copy()                            # Q == q here
        else:
            Qf[:, K:K + Q] = Qf[:, K:K + Q] @ U
        Sg = np.zeros((Q, Q))
        for i in range(Mf):
            if gamma1[i]:
                Sg += spf[i] * Si[i]
        if K > 0:
            Bblk = S[:K, K:K + Q] @ U
            S[:K, K:K + Q] = Bblk
            S[K:K + Q, :K] = Bblk.T
        C = U.T @ Sg @ U
        C[np.arange(r), np.arange(r)] += ev[:r]
        S[K:K + Q, K:K + Q] = C
        # transform the square roots (fixed term's root not needed)
        for k in range(M):
            if alpha[k]:
                Bk = U[:, :r].T @ rS[k][K:K + Q, :]
                rS[k][K:K + r, :] = Bk
                rS[k][K + r:K + Q, :] = 0.0
            elif gamma1[k]:
                rS[k][K:K + Q, :] = U.T @ rS[k][K:K + Q, :]
        # project the sub-dominant Si onto the dominant null space
        Un = U[:, r:]
        for i in range(Mf):
            if gamma1[i]:
                Si[i] = Un.T @ Si[i] @ Un
        K += r
        Q -= r
        gamma = gamma1

    det, B = _qr_ldet_inv(S, get_inv=bool(deriv))
    det1 = None
    det2 = None
    if deriv:
        det1 = np.array([
            float(np.sum((rS[i].T @ B) * rS[i].T)) * spf[i]
            for i in range(M)
        ])
    if deriv == 2:
        P = [B @ rS[i] @ rS[i].T for i in range(M)]
        det2 = np.empty((M, M))
        for i in range(M):
            for j in range(i, M):
                det2[i, j] = det2[j, i] = (
                    -spf[i] * spf[j] * float(np.sum(P[i] * P[j].T))
                )
        det2[np.arange(M), np.arange(M)] += det1
    return S, Qf, rS, float(det), det1, det2


def _gam_reparam(rS_list: list[np.ndarray], lsp: np.ndarray, deriv: int,
                 fixed_penalty: bool = False) -> dict:
    """mgcv ``gam.reparam`` (gam.fit3.r:9-62): get_stableS plus the
    stable square root ``E`` (E'E = S) from diagonally pre-conditioned
    pivoted Cholesky. Returns dict(S, E, Qs, rS, det, det1, det2)."""
    d_tol = float(np.finfo(float).eps) ** 0.3
    r_tol = float(np.finfo(float).eps) ** 0.75
    S, Qs, rS_t, det, det1, det2 = _get_stable_S(
        rS_list, np.exp(np.asarray(lsp, dtype=float)), deriv,
        d_tol, r_tol, fixed_penalty=fixed_penalty,
    )
    S = 0.5 * (S + S.T)
    q = S.shape[0]
    p = np.sqrt(np.abs(np.diag(S)))
    p[p == 0] = 1.0
    St = S / p[:, None] / p[None, :]
    St = 0.5 * (St + St.T)
    E = (_mroot(St, rank=q) * p[:, None]).T
    return {"S": S, "E": E, "Qs": Qs, "rS": rS_t,
            "det": det, "det1": det1, "det2": det2}


# ---------------------------------------------------------------------------
# Sl penalty machinery — mgcv fast-REML.r (Sl.setup / ldetS / appliers).
#
# The block-diagonal penalty representation gam.fit5 (general families) and
# bam's discrete fitter are built on. Port of the DEFAULT path only:
# dense, cholesky=FALSE, no.repara=FALSE — exactly what estimate.gam uses
# for general families (mgcv.r:1899 `Sl.setup(G)`). Out of scope (raise or
# absent, never silent): sparse=TRUE, cholesky=TRUE (ldetSt/iniStrans/
# singleStrans), paraPen blocks, non-linear (`updateS`) blocks, `nl.reg`.
# hea-side simplifications that are exact for hea's smooth zoo: every block
# is linear with repara=TRUE (mgcv sets repara=FALSE only for g.index
# smooths, smooth.r:3819 — none exist here).
#
# Index convention: `start`/`stop` are 0-based half-open (Python slices);
# mgcv's are 1-based inclusive. All other arithmetic is line-by-line.
# ---------------------------------------------------------------------------


class _SlBlock:
    """One block of the block-diagonal total penalty (mgcv Sl[[b]])."""
    __slots__ = ("start", "stop", "S", "rank", "repara", "lam", "D", "Di",
                 "ind", "ldet", "rS", "Srp", "St")

    def __init__(self, *, start: int, stop: int, S: list[np.ndarray],
                 rank: int, repara: bool, lam: np.ndarray,
                 D: np.ndarray | None, Di: np.ndarray | None,
                 ind: np.ndarray, ldet: float = 0.0,
                 rS: list[np.ndarray] | None = None):
        self.start = start          # 0-based first coef of the block
        self.stop = stop            # past-end (Python half-open)
        self.S = S                  # penalties (projected if multi-S)
        self.rank = rank
        self.repara = repara
        self.lam = lam              # per-S λ (setup scaling; ldetS updates)
        self.D = D                  # diag vector or matrix transform
        self.Di = Di
        self.ind = ind              # boolean penalized mask within block
        self.ldet = ldet            # initial-repara log-det correction
        self.rS = rS                # multi-S roots (projected)
        self.Srp = None             # per-term λᵢSᵢ in ldetS repara coords
        self.St = None              # block total penalty at current λ

    @property
    def n_sp(self) -> int:
        return len(self.S)

    def pen_cols(self) -> np.ndarray:
        """Absolute column indices of the penalized coefs of this block —
        mgcv's ``(start:stop)[ind]``."""
        return np.arange(self.start, self.stop)[self.ind]


class _Sl:
    """Container for the Sl block list + Sl.setup attributes."""
    __slots__ = ("blocks", "E", "S", "lam0", "p")

    def __init__(self, blocks: list[_SlBlock], E: np.ndarray,
                 S: np.ndarray, lam0: np.ndarray, p: int):
        self.blocks = blocks
        self.E = E                  # attr(Sl,"E"): E'E = balanced penalty
        self.S = S                  # attr(Sl,"S"): balanced total penalty
        self.lam0 = lam0            # attr(Sl,"lambda")
        self.p = p

    def __len__(self) -> int:
        return len(self.blocks)


def _sl_setup(slots: list["_PenaltySlot"], p: int) -> _Sl:
    """mgcv ``Sl.setup`` (fast-REML.r:68-429), default dense
    non-Cholesky path.

    ``slots`` are hea's per-penalty records; slots sharing one parent
    SmoothBlock and column range form an mgcv "additive block" (tensor
    smooths), everything else is a singleton. Multi-S blocks are split
    into singletons when their penalty footprints don't overlap
    (fast-REML.r:175-225), then each block is reparameterized: singletons
    to partial identity (diagonal shortcut or eigen), multi-S projected
    onto the range space of the unscaled total penalty (eigenvectors U,
    rank-r leading block).
    """
    eps = float(np.finfo(float).eps)
    # ---- group slots into blocks (per-smooth S lists) -------------------
    groups: list[tuple[int, int, list[np.ndarray]]] = []
    for slot in slots:
        if (groups and groups[-1][0] == slot.col_start
                and groups[-1][1] == slot.col_end):
            groups[-1][2].append(np.asarray(slot.S, dtype=float))
        else:
            groups.append((slot.col_start, slot.col_end,
                           [np.asarray(slot.S, dtype=float)]))

    raw_blocks: list[tuple[int, int, list[np.ndarray]]] = []
    for (a, b, S_list) in groups:
        m = len(S_list)
        if m == 1:
            raw_blocks.append((a, b, S_list))
            continue
        # Split test (fast-REML.r:179-204): no overlap in penalty
        # footprints; all-diagonal penalties may interleave.
        nb = S_list[0].shape[0]
        sbdiag = np.zeros(m, dtype=bool)
        sb_start = np.zeros(m, dtype=int)
        sb_stop = np.zeros(m, dtype=int)
        for j, Sj in enumerate(S_list):
            off_diag = Sj - np.diag(np.diag(Sj))
            sbdiag[j] = float(np.sum(np.abs(off_diag))) == 0.0
            nz = np.where(np.sum(np.abs(Sj), axis=1) > 0)[0]
            sb_start[j], sb_stop[j] = int(nz[0]), int(nz[-1])
        split_ok = True
        for j in range(m):
            itot = np.zeros(nb, dtype=bool)
            if np.all(sbdiag):
                for k in range(m):
                    if k != j:
                        itot[np.diag(S_list[k]) != 0] = True
                if np.any(itot[np.diag(S_list[j]) != 0]):
                    split_ok = False
                    break
            else:
                for k in range(m):
                    if k != j:
                        itot[sb_start[k]:sb_stop[k] + 1] = True
                if np.any(itot[sb_start[j]:sb_stop[j] + 1]):
                    split_ok = False
                    break
        if split_ok:
            for j in range(m):
                ind = slice(sb_start[j], sb_stop[j] + 1)
                raw_blocks.append((a + sb_start[j], a + sb_stop[j] + 1,
                                   [S_list[j][ind, ind]]))
        else:
            raw_blocks.append((a, b, S_list))

    # ---- per-block reparameterization + balanced E/S ---------------------
    blocks: list[_SlBlock] = []
    E = np.zeros((p, p))
    S_bal = np.zeros((p, p))
    lam0_parts: list[float] = []
    for (a, b, S_list) in raw_blocks:
        if len(S_list) == 1:
            S1 = S_list[0]
            k = S1.shape[0]
            off_diag = S1 - np.diag(np.diag(S1))
            if float(np.sum(np.abs(off_diag))) == 0.0:
                # Diagonal S: D a vector (fast-REML.r:268-278).
                Dv = np.diag(S1).copy()
                ind = Dv > 0
                rank = int(np.sum(ind))
                Dv[ind] = 1.0 / np.sqrt(Dv[ind])
                Dv[~ind] = 1.0
                blk = _SlBlock(start=a, stop=b, S=[S1], rank=rank,
                               repara=True, lam=np.ones(1), D=Dv, Di=None,
                               ind=ind)
            else:
                # Eigen reparameterization (fast-REML.r:288-302).
                w, U = np.linalg.eigh(0.5 * (S1 + S1.T))
                w = w[::-1].copy()
                U = U[:, ::-1].copy()
                rank = int(np.sum(w > eps ** 0.8 * float(w.max())))
                Dv = w.copy()
                ind = np.zeros(k, dtype=bool)
                ind[:rank] = True
                Dv[ind] = 1.0 / np.sqrt(Dv[ind])
                Dv[~ind] = 1.0
                D = U * Dv[None, :]            # U %*% diag(D)
                Di = U.T / Dv[:, None]         # diag(1/D) %*% t(U)
                blk = _SlBlock(start=a, stop=b, S=[S1], rank=rank,
                               repara=True, lam=np.ones(1), D=D, Di=Di,
                               ind=ind)
            # repara=TRUE contribution: identity at penalized positions
            # (fast-REML.r:317-325).
            pcols = blk.pen_cols()
            E[pcols, pcols] = 1.0
            S_bal[pcols, pcols] = 1.0
            lam0_parts.append(1.0)
            blocks.append(blk)
        else:
            # Multi-S block, non-Cholesky (fast-REML.r:371-404):
            # eigen of the UNSCALED total, project into its range space.
            m = len(S_list)
            St = S_list[0].copy()
            for Sj in S_list[1:]:
                St = St + Sj
            w, U = np.linalg.eigh(0.5 * (St + St.T))
            w = w[::-1].copy()
            U = U[:, ::-1].copy()
            rank = int(np.sum(w > eps ** 0.8 * float(w.max())))
            Ur = U[:, :rank]
            S_proj = []
            rS = []
            for Sj in S_list:
                bob = Ur.T @ Sj @ Ur
                bob = 0.5 * (bob + bob.T)
                S_proj.append(bob)
                rS.append(_mroot(bob, rank))
            ind = np.zeros(S_list[0].shape[0], dtype=bool)
            ind[:rank] = True
            blk = _SlBlock(start=a, stop=b, S=S_proj, rank=rank,
                           repara=True, lam=np.ones(m), D=U, Di=None,
                           ind=ind, rS=rS)
            # Balanced E/S in the NEW (projected) coordinates
            # (fast-REML.r:394-417): Σ S_j/‖S_j‖ over the projected S.
            St2 = np.zeros((rank, rank))
            for Sj in S_proj:
                nrm = float(np.abs(Sj).sum(axis=0).max())  # R one-norm
                St2 = St2 + Sj / nrm
                lam0_parts.append(1.0 / nrm)
            St2 = 0.5 * (St2 + St2.T)
            Sr = _mroot(St2, rank).T
            E[a:a + Sr.shape[0], a:a + Sr.shape[1]] = Sr
            S_bal[a:a + rank, a:a + rank] = St2
            blocks.append(blk)
    return _Sl(blocks, E, S_bal, np.asarray(lam0_parts, dtype=float), p)


def _sl_initial_repara(sl: _Sl, X: np.ndarray, inverse: bool = False,
                       both_sides: bool = True, cov: bool = True):
    """mgcv ``Sl.initial.repara`` (fast-REML.r:517-588): apply (or undo)
    the Sl.setup block transforms. Forward: model matrix → repara'd
    coordinates (X·D per block; both_sides also hits rows — for X'X-like
    inputs). Inverse: coefficient vector / covariance matrix back to the
    original coordinates (``cov=False`` uses Di — the proper inverse —
    for plain matrices). Vector + ``both_sides=False`` + forward means
    "X is a coefficient vector" (transform by Di)."""
    X = np.array(X, dtype=float, copy=True)
    if len(sl) == 0:
        return X
    for blk in sl.blocks:
        if not blk.repara:
            continue
        ind = slice(blk.start, blk.stop)
        D = blk.D
        if inverse:
            if X.ndim == 2:
                if cov:
                    if D.ndim == 2:
                        if both_sides:
                            X[ind, :] = D @ X[ind, :]
                        X[:, ind] = X[:, ind] @ D.T
                    else:
                        X[:, ind] = X[:, ind] * D[None, :]
                        if both_sides:
                            X[ind, :] = D[:, None] * X[ind, :]
                else:
                    if D.ndim == 2:
                        Di = D.T if blk.Di is None else blk.Di
                        if both_sides:
                            X[ind, :] = Di.T @ X[ind, :]
                        X[:, ind] = X[:, ind] @ Di
                    else:
                        Di = 1.0 / D
                        X[:, ind] = X[:, ind] * Di[None, :]
                        if both_sides:
                            X[ind, :] = Di[:, None] * X[ind, :]
            else:
                if D.ndim == 2:
                    X[ind] = D @ X[ind]
                else:
                    X[ind] = D * X[ind]
        else:
            if X.ndim == 2:
                if D.ndim == 2:
                    if both_sides:
                        X[ind, :] = D.T @ X[ind, :]
                    X[:, ind] = X[:, ind] @ D
                else:
                    if both_sides:
                        X[ind, :] = D[:, None] * X[ind, :]
                    X[:, ind] = X[:, ind] * D[None, :]
            else:
                if both_sides:
                    if D.ndim == 2:
                        X[ind] = D.T @ X[ind]
                    else:
                        X[ind] = D * X[ind]
                else:
                    if D.ndim == 2:
                        Di = D.T if blk.Di is None else blk.Di
                        X[ind] = Di @ X[ind]
                    else:
                        X[ind] = X[ind] / D
    return X


def _ldet_s_block(rS_list: list[np.ndarray], rho: np.ndarray,
                  deriv: int = 2, root: bool = False) -> dict:
    """mgcv ``ldetSblock`` (fast-REML.r:593-635): derivatives w.r.t. ρ of
    ``log|S|`` where ``S = Σ_i tcrossprod(rS_i)·exp(ρ_i)``, when S is full
    rank +ve def and no reparameterization is required — the per-block
    engine of ``ldetS(repara=FALSE)`` (Sl.fitChol's log|Sλ|₊ oracle).

    Diagonally pre-conditioned pivoted Cholesky of the summed penalty
    (R's ``chol(pivot=TRUE)`` = dpstrf with its default rank tolerance),
    then per term ``d1[i] = λ_i‖R'⁻¹(rS_i[piv,]/d[piv])‖²_F`` and
    ``d2[i,j] = −λ_iλ_j Σ(M_i∘M_j)`` with ``M_i = tcrossprod(R'⁻¹·)``,
    ``d2[i,i] += d1[i]``. Serial (nt=1) path. ``E`` is the summed penalty
    when ``root=False``, its unpivoted Cholesky root otherwise.
    """
    from scipy.linalg.lapack import dpstrf
    rho = np.asarray(rho, dtype=float)
    lam = np.exp(rho)
    m = len(rS_list)
    S = (rS_list[0] @ rS_list[0].T) * lam[0]
    p = S.shape[1]
    for i in range(1, m):
        S = S + (rS_list[i] @ rS_list[i].T) * lam[i]
    E = None if root else S.copy()
    d = np.diag(S).copy()
    d[d <= 0.0] = 1.0
    d = np.sqrt(d)
    S_pre = (S / d).T / d                          # t(S/d)/d
    c, piv_1based, r, _info = dpstrf(S_pre, lower=0)
    R = np.triu(c)
    piv = np.asarray(piv_1based, dtype=int) - 1
    if r < p:
        R[r:, r:] = 0.0                            # fix chol bug (:611)
    if root:
        rp = np.empty(p, dtype=int)                # reverse pivot (:613)
        rp[piv] = np.arange(p)
        E = R[:, rp] * d[None, :]                  # t(t(R[,rp])*d)
    if r < p:                                      # rank deficiency (:616)
        R = R[:r, :r]
        piv = piv[:r]
    dS1 = np.zeros(m)
    dS2 = np.zeros((m, m))
    RrS: list[np.ndarray] = []
    # dlog|S|/drho_i = lam_i tr(S⁻¹S_i) = tr(R⁻ᵀrS_i rS_i'R⁻¹) etc. (:621)
    for i in range(m):
        # pforwardsolve: R transposed internally — solves R'x = b (:623).
        Xi = solve_triangular(R.T, rS_list[i][piv, :] / d[piv, None],
                              lower=True)
        dS1[i] = float(np.sum(Xi * Xi)) * lam[i]
        if deriv == 2:
            Mi = Xi @ Xi.T                         # tcrossprod (:626)
            RrS.append(Mi)
            for j in range(i + 1):
                v = -float(np.sum(RrS[i] * RrS[j])) * lam[i] * lam[j]
                dS2[i, j] = dS2[j, i] = v
            dS2[i, i] += dS1[i]
    det = 2.0 * float(np.sum(np.log(np.diag(R)) + np.log(d[piv])))
    return {"det": det, "det1": dS1, "det2": dS2, "E": E}


def _ldet_s(sl: _Sl, rho: np.ndarray, fixed: np.ndarray | None = None,
            root: bool = False, stot: bool = False,
            deriv: int = 2, repara: bool = True) -> dict:
    """mgcv ``ldetS`` (fast-REML.r:762-1013), default dense non-Cholesky
    path: log|Sλ|₊ with first/second ρ-derivatives, the per-block multi-S
    reparameterization list ``rp``, the total-penalty root ``E`` (zero
    rows dropped) and total ``S``.

    Singleton blocks contribute ``rank·ρ_k`` analytically (their penalty
    is a partial identity after Sl.setup); multi-S blocks go through
    ``_gam_reparam`` when ``repara=True`` — the gam.fit3 ``gam.reparam``
    similarity transform already pinned against mgcv (§2.2) — or
    :func:`_ldet_s_block` when ``repara=False`` (fast-REML.r:909-910),
    the un-transformed pivoted-Cholesky form ``Sl.fitChol`` uses on the
    initial-repara'd gauge. Updates each block's ``lam``, ``St`` and
    ``Srp`` in place, mirroring the returned ``Sl`` (``repara=False``
    resets ``Srp`` to None and stores ``St = Σ λ_i·rS_i rS_i'``, mgcv's
    ``grp$St <- grp$E``, fast-REML.r:911).
    """
    if root and not repara:
        # fast-REML.r:911/961 would store the ldetSblock ROOT in St for this
        # combination — no live mgcv caller reaches it (Sl.fitChol passes
        # root=FALSE); out of scope: raise, never silent.
        raise NotImplementedError("ldetS root=TRUE with repara=FALSE")
    rho = np.asarray(rho, dtype=float)
    n_sp_total = sum(blk.n_sp for blk in sl.blocks)
    if fixed is None:
        fixed = np.zeros(n_sp_total, dtype=bool)
    n_deriv = int(np.sum(~fixed))
    ldS = 0.0
    d1 = np.zeros(n_deriv)
    d2 = np.zeros((n_deriv, n_deriv))
    rp: list[dict] = []
    E = np.zeros((sl.p, sl.p)) if root else None
    S = np.zeros((sl.p, sl.p)) if stot else None
    k_sp = 0
    k_deriv = 0
    for blk in sl.blocks:
        if blk.n_sp == 1:
            # Linear singleton (fast-REML.r:832-898).
            ldS += blk.ldet + rho[k_sp] * blk.rank
            if not fixed[k_sp]:
                d1[k_deriv] = float(blk.rank)
                k_deriv += 1
            pcols = blk.pen_cols()
            if root:
                E[pcols, pcols] = np.exp(rho[k_sp] * 0.5)
            if stot:
                S[pcols, pcols] = np.exp(rho[k_sp])
            blk.lam = np.array([np.exp(rho[k_sp])])
            k_sp += 1
        else:
            # Linear multi-S block (fast-REML.r:899-1007): gam.reparam
            # (repara=TRUE) or ldetSblock (repara=FALSE), fast-REML.r:909-910.
            m = blk.n_sp
            ind_sp = slice(k_sp, k_sp + m)
            ldS += blk.ldet
            if repara:
                grp = _gam_reparam(blk.rS, rho[ind_sp], deriv)
            else:
                grp = _ldet_s_block(blk.rS, rho[ind_sp], deriv=deriv,
                                    root=False)
            blk.lam = np.exp(rho[ind_sp])
            ldS += grp["det"]
            free = ~fixed[ind_sp]
            if deriv > 0:
                det1 = np.asarray(grp["det1"], dtype=float).reshape(-1)[free]
                nd = det1.size
                if nd > 0:
                    sl_d = slice(k_deriv, k_deriv + nd)
                    d1[sl_d] = det1
                    if deriv > 1:
                        det2 = np.asarray(grp["det2"], dtype=float)
                        d2[sl_d, sl_d] = det2[np.ix_(free, free)]
                    k_deriv += nd
            else:
                k_deriv += int(np.sum(free))
            if repara:
                # Reparameterization info + Srp only when the stabilising
                # transform is applied (fast-REML.r:929-939).
                rp.append({
                    "ind": blk.pen_cols(),
                    "Qs": grp["Qs"],
                    "repara": blk.repara,
                })
                blk.Srp = [
                    blk.lam[i] * (grp["rS"][i] @ grp["rS"][i].T)
                    for i in range(m)
                ]
                blk.St = grp["S"]
            else:
                # ldetSblock returns the summed penalty in E when root==FALSE
                # (fast-REML.r:911 ``grp$St <- grp$E``); no rp entry, and Srp
                # reset so Sl.mult/Sl.termMult use the un-transformed
                # ``lam_i·(S_i·A)`` form.
                blk.Srp = None
                blk.St = grp["E"]
            k_sp += m
            if root:
                Eb = grp["E"]
                E[blk.start:blk.start + Eb.shape[0],
                  blk.start:blk.start + Eb.shape[1]] = Eb
            if stot:
                Stb = blk.St
                S[blk.start:blk.start + Stb.shape[0],
                  blk.start:blk.start + Stb.shape[1]] = Stb
    if root:
        keep = np.sum(np.abs(E), axis=1) != 0
        E = E[keep, :]
    return {"ldetS": ldS, "ldet1": d1, "ldet2": d2, "rp": rp,
            "E": E, "S": S}


def _sl_repara(rp: list[dict], X: np.ndarray, inverse: bool = False,
               both_sides: bool = True):
    """mgcv ``Sl.repara`` (fast-REML.r:1087-1117): apply the ldetS
    multi-S reparameterization. Forward: model matrix columns
    ``X[:, ind] @ Qs`` (or β-vector ``Qs' β``); inverse: coef vector /
    covariance back via ``Qs``."""
    X = np.array(X, dtype=float, copy=True)
    for r in rp:
        if not r["repara"]:
            continue
        ind = r["ind"]
        Qs = r["Qs"]
        if inverse:
            if X.ndim == 2:
                if both_sides:
                    X[ind, :] = Qs @ X[ind, :]
                X[:, ind] = X[:, ind] @ Qs.T
            else:
                X[ind] = Qs @ X[ind]
        else:
            if X.ndim == 2:
                X[:, ind] = X[:, ind] @ Qs
            else:
                X[ind] = Qs.T @ X[ind]
    return X


def _sl_repa(rp: list[dict], X: np.ndarray, lt: int = 0, r: int = 0):
    """mgcv ``Sl.repa`` (fast-REML.r:1062-1085): generalized applier.
    ``lt``/``r`` ∈ {−2,−1,0,1,2}: 0 = skip, 1 = D, 2 = D', −1 = Di,
    −2 = Di', applied to rows (lt) / columns (r). With Qs-only blocks
    (the non-Cholesky path) D = Qs' and Di = Qs."""
    X = np.array(X, dtype=float, copy=True)
    for rec in rp:
        if not rec["repara"]:
            continue
        ind = rec["ind"]
        Qs = rec["Qs"]
        def _T(code):
            # D = t(Qs), Di = Qs (fast-REML.r:1067).
            if code == 1:
                return Qs.T
            if code == 2:
                return Qs
            if code == -1:
                return Qs
            return Qs.T          # code == -2
        if lt:
            T = _T(lt)
            if X.ndim == 2:
                X[ind, :] = T @ X[ind, :]
            else:
                X[ind] = T @ X[ind]
        if r:
            T = _T(r)
            if X.ndim == 2:
                X[:, ind] = X[:, ind] @ T
            else:
                X[ind] = X[ind] @ T
    return X


def _sl_inirep(sl: _Sl, X: np.ndarray, lt: int = 0, r: int = 0):
    """mgcv ``Sl.inirep`` (fast-REML.r:485-520): code-based applier of
    the Sl.setup INITIAL block transforms (the ``Sl.repa`` analog of
    ``Sl.initial.repara``). ``lt``/``r`` ∈ {−2,−1,0,1,2}: 0 = skip,
    1 = D, 2 = D', −1 = Di (D' when Di is None — orthogonal D),
    −2 = Di'. Vector D blocks get the diagonal row/col scaling
    (Sl.initial.repara's vector semantics; mgcv's matrix-only ``%*%``
    would mangle them — no such block reaches this in practice). Note
    mgcv's r-branch tests ``lt`` for the transform choice (a quirk);
    hea keys it on ``r`` — gam.fit5.post.proc only ever calls (1, 0).
    """
    X = np.array(X, dtype=float, copy=True)
    if len(sl) == 0 or (not lt and not r):
        return X
    for blk in sl.blocks:
        if not blk.repara:
            continue
        ind = slice(blk.start, blk.stop)
        D = blk.D

        def _mat(code):
            if D.ndim == 1:
                Dm = np.diag(D)
                Dim = np.diag(1.0 / D)
            else:
                Dm = D
                Dim = (D.T if blk.Di is None else blk.Di)
            if code == 1:
                return Dm
            if code == 2:
                return Dm.T
            if code == -1:
                return Dim
            return Dim.T            # code == -2

        if lt:
            T = _mat(lt)
            if X.ndim == 2:
                X[ind, :] = T @ X[ind, :]
            else:
                X[ind] = T @ X[ind]
        if r:
            T = _mat(r)
            if X.ndim == 2:
                X[:, ind] = X[:, ind] @ T
            else:
                X[ind] = X[ind] @ T
    return X


def _sl_mult(sl: _Sl, A: np.ndarray, k: int | None = None,
             full: bool = True):
    """mgcv ``Sl.mult`` (fast-REML.r:1119-1225): ``Sλ @ A`` (``k=None``)
    or ``λ_k S_k @ A`` for the k-th penalty (0-based here; mgcv's k is
    1-based). ``full=False`` strips the zero rows. Assumes ``_ldet_s``
    has run (block ``lam``/``St``/``Srp`` current)."""
    A = np.asarray(A, dtype=float)
    if len(sl) == 0:
        return np.zeros_like(A)
    if k is None:
        B = np.zeros_like(A)
        for blk in sl.blocks:
            if blk.n_sp == 1:
                pcols = blk.pen_cols()
                B[pcols] = blk.lam[0] * A[pcols]
            else:
                pcols = blk.pen_cols()
                B[pcols] = blk.St @ A[pcols]
        return B
    j = 0
    for blk in sl.blocks:
        for i in range(blk.n_sp):
            if j == k:
                pcols = blk.pen_cols()
                if blk.n_sp == 1:
                    part = blk.lam[0] * A[pcols]
                else:
                    part = (blk.Srp[i] @ A[pcols] if blk.Srp is not None
                            else blk.lam[i] * (blk.S[i] @ A[pcols]))
                if full:
                    B = np.zeros_like(A)
                    B[pcols] = part
                    return B
                return part
            j += 1
    raise IndexError(f"penalty index {k} out of range")


def _sl_term_mult(sl: _Sl, A: np.ndarray, full: bool = False):
    """mgcv ``Sl.termMult`` (fast-REML.r:1227-1327): the list
    ``[λ_i S_i @ A]`` over every penalty. Returns ``(SA, inds)`` —
    mgcv attaches ``ind`` as an attribute; here it's the parallel list
    (None entries when ``full=True``)."""
    A = np.asarray(A, dtype=float)
    SA: list[np.ndarray] = []
    inds: list[np.ndarray | None] = []
    for blk in sl.blocks:
        pcols = blk.pen_cols()
        for i in range(blk.n_sp):
            if blk.n_sp == 1:
                part = blk.lam[0] * A[pcols]
            else:
                part = (blk.Srp[i] @ A[pcols] if blk.Srp is not None
                        else blk.lam[i] * (blk.S[i] @ A[pcols]))
            if full:
                B = np.zeros_like(A)
                B[pcols] = part
                SA.append(B)
                inds.append(None)
            else:
                SA.append(part)
                inds.append(pcols)
    return SA, inds


def _append_extra_params(md: "_MultiDesign", n_extra: int) -> "_MultiDesign":
    """Append ``n_extra`` unpenalized "dummy" columns to a multi-formula
    design — mgcv's ``preinitialize`` for ``mvn`` (mvam.r:103-107): the
    Choleski-factor parameters of the precision matrix get zero design
    columns so they ride the coefficient vector, but they belong to no
    linear predictor and carry no penalty. Only ``X``/``p``/
    ``column_names`` change; ``lpi``/slots/blocks/nsdf/L are untouched
    (the params are pure unpenalized coefficients, like extra parametric
    terms living outside every LP)."""
    n = md.X.shape[0]
    X = np.concatenate([md.X, np.zeros((n, n_extra))], axis=1)
    names = list(md.column_names) + [f"R.{i + 1}" for i in range(n_extra)]
    kw = {k: getattr(md, k) for k in _MultiDesign.__slots__}
    kw.update(X=X, p=X.shape[1], column_names=names)
    return _MultiDesign(**kw)


# ---------------------------------------------------------------------------
# Multi-formula front end — mgcv interpret.gam list branch (mgcv.r:431-498)
# + gam.setup.list (mgcv.r:922-1092). §5.3 prerequisite 4.
#
# A list of formulas — the first with a response, the rest response-less
# (`"~ s(z)"`) — becomes ONE stacked design matrix with `lpi`: per-linear-
# predictor column index lists. Each formula runs the same per-formula
# design pipeline gam.__init__ uses (constraint absorption, select
# penalties, gam.side, id linkage — all WITHIN its own formula, exactly
# like mgcv's per-formula gam.setup calls), then columns are appended in
# formula order. Smooth labels in formula j ≥ 1 get mgcv's textra suffix
# inserted before the first "(" (`s(z)` → `s.1(z)`, interpret.gam0
# mgcv.r:370-374); parametric names get a trailing `.{j}`
# (gam.setup.list mgcv.r:1042).
#
# Out of scope first pass (explicit raise, never silent): mgcv's
# numeric-label shared-term syntax (`1 + 2 ~ s(x)`) and the `olid`
# unidentifiability dropping it requires; multivariate responses (mvn);
# `drop.intercept`. Consumed by gam.fit5 (general families) when §5.3
# proper lands — until then `gam()` raises NotImplementedError on list
# formulas after this assembler is importable for tests.
# ---------------------------------------------------------------------------


class _LpDesign:
    """One linear predictor's design bundle (one formula's gam.setup)."""
    __slots__ = ("formula", "expanded", "data", "X", "blocks",
                 "block_col_ranges", "slots", "column_names", "offset",
                 "nsdf", "L", "n_work", "param_assign")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _MultiDesign:
    """gam.setup.list analog: the stacked multi-LP design."""
    __slots__ = ("lps", "X", "lpi", "y", "blocks", "block_col_ranges",
                 "slots", "column_names", "nsdf", "pstart", "offsets",
                 "n_lp", "L", "n_work", "p", "n")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _suffix_smooth_label(label: str, suffix: str) -> str:
    """mgcv's textra insertion (interpret.gam0, mgcv.r:370-374):
    ``s(z)`` + ``.1`` → ``s.1(z)`` — before the FIRST ``(``."""
    pos = label.find("(")
    if pos < 0:
        return label + suffix
    return label[:pos] + suffix + label[pos:]


def _build_lp_design(formula: str, data, knots: dict | None,
                     select: bool, label_suffix: str | None) -> _LpDesign:
    """One formula through the same design pipeline as gam.__init__
    (prepare_design → smooth-arg materialization → formula offsets →
    materialize_smooths → select penalties → gam.side → column stacking
    → per-formula id L-matrix). Kept in lockstep with the constructor's
    design block — the single-formula path is untouched until §5.3
    unifies them."""
    from ..formula import (_apply_smooth_arg_exprs, _smooth_arg_expr_map,
                           _smooth_id_value)
    d = prepare_design(formula, data)
    expr_map = _smooth_arg_expr_map(d.expanded)
    data_m = _apply_smooth_arg_exprs(d.data, expr_map) if expr_map else d.data
    X_param_df = d.X
    X_param = X_param_df.to_numpy().astype(float)
    n = data_m.height
    if X_param.shape[1] == 0:
        X_param = np.zeros((n, 0))
    off = np.zeros(n)
    has_off = False
    for off_node in d.expanded.offsets:
        blk = _eval_atom(off_node, d.data)
        off = off + blk.values.flatten().astype(float)
        has_off = True

    sb_lists = (materialize_smooths(d.expanded, data_m, knots=knots)
                if d.expanded.smooths else [])
    blocks = [b for group in sb_lists for b in group]
    block_ids: list[str | None] = []
    for call_node, group_blocks in zip(d.expanded.smooths, sb_lists):
        block_ids.extend([_smooth_id_value(call_node)] * len(group_blocks))
    if select:
        blocks = _add_null_space_penalties(blocks)
    blocks = _apply_gam_side(blocks)

    if label_suffix:
        for b in blocks:
            b.label = _suffix_smooth_label(b.label, label_suffix)

    Xs = [X_param]
    slots: list[_PenaltySlot] = []
    ranges: list[tuple[int, int]] = []
    cursor = X_param.shape[1]
    for b in blocks:
        Xb = np.asarray(b.X, dtype=float)
        Xs.append(Xb)
        a, bcol = cursor, cursor + Xb.shape[1]
        ranges.append((a, bcol))
        for j, S_j in enumerate(b.S):
            slots.append(_PenaltySlot(block=b, col_start=a, col_end=bcol,
                                      S=np.asarray(S_j, dtype=float),
                                      S_scale=_block_s_scale(b, j)))
        cursor = bcol
    X = np.concatenate(Xs, axis=1) if len(Xs) > 1 else X_param

    names = list(X_param_df.columns)
    if label_suffix:
        names = [f"{nm}{label_suffix}" for nm in names]
    for b, (a, bcol) in zip(blocks, ranges):
        for i in range(1, bcol - a + 1):
            names.append(f"{b.label}.{i}")

    # Per-formula id L-matrix (same logic as gam.__init__; linkage never
    # crosses formulas — mgcv's gam.setup calls are independent too).
    slot_work_col: list[int] = []
    n_work = 0
    id_first: dict[str, tuple[int, int]] = {}
    for b, bid in zip(blocks, block_ids):
        nS = len(b.S)
        if nS == 0:
            continue
        if bid is None or bid not in id_first:
            start = n_work
            n_work += nS
            if bid is not None:
                id_first[bid] = (start, nS)
        else:
            start, nc = id_first[bid]
            if nS > nc:
                raise ValueError(
                    "Later terms sharing an `id' can not have more "
                    "smoothing parameters than the first such term"
                )
        slot_work_col.extend(range(start, start + nS))
    if n_work == len(slots):
        L = None
    else:
        L = np.zeros((len(slots), n_work))
        L[np.arange(len(slots)), slot_work_col] = 1.0

    return _LpDesign(formula=formula, expanded=d.expanded, data=data_m,
                     X=X, blocks=blocks, block_col_ranges=ranges,
                     slots=slots, column_names=names,
                     offset=(off if has_off else None),
                     nsdf=X_param.shape[1], L=L, n_work=n_work,
                     param_assign=list(d.param_assign or []))


def _prepare_multi_design(formulas: list[str], data,
                          knots: dict | None = None,
                          select: bool = False,
                          allow_single: bool = False,
                          matrix_response: bool = False) -> _MultiDesign:
    """mgcv ``gam.setup.list`` (mgcv.r:922-1092) for hea: a list of
    formula strings → one stacked design with ``lpi``.

    The first formula carries the response; every later one must be
    response-less (``"~ s(z)"`` — mgcv injects the first response to
    keep gam.setup happy, mgcv.r:962-963, and so does this). Columns
    append in formula order; ``lpi[j]`` holds LP j's 0-based column
    indices; offsets are per-LP (``None`` when a formula has no
    ``offset()`` atom); penalties carry global column offsets; the id
    L-matrix is block-diagonal across formulas.

    ``matrix_response=True`` (the ``mvn`` front end): EVERY formula
    carries its own response and they stack column-wise into an
    ``(n, n_lp)`` matrix ``y`` (mgcv stacks the per-formula LHS into the
    matrix response ``G$y``). The designs are still built from the
    per-formula RHS exactly as in the standard case.
    """
    if len(formulas) < 1 or (len(formulas) < 2 and not allow_single):
        raise ValueError(
            "multi-formula gam needs at least 2 formulas; pass a plain "
            "string for single-predictor models"
        )
    first = formulas[0]
    if "~" not in first:
        raise ValueError(f"first formula must contain '~': {first!r}")
    resp = first.split("~", 1)[0].strip()
    if not resp:
        raise ValueError("first formula must have a response on the lhs")
    full_formulas = [first]
    resp_exprs = [resp]
    for j, f in enumerate(formulas[1:], start=1):
        if "~" not in f:
            raise ValueError(f"formula {j} must contain '~': {f!r}")
        lhs = f.split("~", 1)[0].strip()
        if matrix_response:
            # mvn: each formula supplies its own dimension's response.
            if not lhs:
                raise ValueError(
                    "matrix-response family: every formula must carry a "
                    f"response on the lhs; formula {j} is response-less")
            resp_exprs.append(lhs)
            full_formulas.append(f)
            continue
        if lhs:
            raise NotImplementedError(
                "formulas after the first must be response-less "
                f"('~ ...'); got lhs {lhs!r}. mgcv's numeric-label "
                "shared-term syntax ('1 + 2 ~ s(x)') is not supported "
                "yet (multi-formula shared-term front end, out-of-scope "
                "list)."
            )
        full_formulas.append(f"{resp} {f.strip()}")

    lps: list[_LpDesign] = []
    for j, f in enumerate(full_formulas):
        # response-less RHS for design building when each formula has its
        # own LHS (mvn): reuse the shared response label so _build_lp_design
        # finds a valid y to discard.
        build_f = (f"{resp} ~ {f.split('~', 1)[1].strip()}"
                   if matrix_response else f)
        lps.append(_build_lp_design(
            build_f, data, knots, select,
            label_suffix=(f".{j}" if j > 0 else None),
        ))

    n = lps[0].X.shape[0]
    if matrix_response:
        y = np.column_stack([
            prepare_design(f"{r} ~ 1", data).y.to_numpy().astype(float)
            for r in resp_exprs])
    else:
        y = prepare_design(full_formulas[0], data).y.to_numpy().astype(float)
    X = np.concatenate([lp.X for lp in lps], axis=1)
    lpi: list[np.ndarray] = []
    blocks: list[SmoothBlock] = []
    ranges: list[tuple[int, int]] = []
    slots: list[_PenaltySlot] = []
    names: list[str] = []
    nsdf: list[int] = []
    pstart: list[int] = []
    offsets: list[np.ndarray | None] = []
    L_parts: list[np.ndarray | None] = []
    pof = 0
    for lp in lps:
        p_lp = lp.X.shape[1]
        lpi.append(np.arange(pof, pof + p_lp))
        pstart.append(pof)
        nsdf.append(lp.nsdf)
        offsets.append(lp.offset)
        names.extend(lp.column_names)
        blocks.extend(lp.blocks)
        for (a, b) in lp.block_col_ranges:
            ranges.append((a + pof, b + pof))
        for s in lp.slots:
            slots.append(_PenaltySlot(
                block=s.block, col_start=s.col_start + pof,
                col_end=s.col_end + pof, S=s.S, S_scale=s.S_scale,
            ))
        L_parts.append(lp.L if lp.L is not None
                       else (np.eye(len(lp.slots)) if lp.slots else None))
        pof += p_lp

    # Block-diagonal L across formulas; None ⇔ identity everywhere.
    if all(lp.L is None for lp in lps):
        L = None
        n_work = len(slots)
    else:
        sizes = [(len(lp.slots),
                  lp.n_work if lp.L is not None else len(lp.slots))
                 for lp in lps]
        n_work = sum(w for _, w in sizes)
        L = np.zeros((len(slots), n_work))
        r0 = c0 = 0
        for lp, (nr, nc) in zip(lps, sizes):
            if nr:
                Lj = lp.L if lp.L is not None else np.eye(nr)
                L[r0:r0 + nr, c0:c0 + nc] = Lj
            r0 += nr
            c0 += nc

    return _MultiDesign(lps=lps, X=X, lpi=lpi, y=y, blocks=blocks,
                        block_col_ranges=ranges, slots=slots,
                        column_names=names, nsdf=nsdf, pstart=pstart,
                        offsets=offsets, n_lp=len(lps), L=L,
                        n_work=n_work, p=X.shape[1], n=n)


def _drop_general_intercept(md: _MultiDesign) -> _MultiDesign:
    """Drop the ``(Intercept)`` column from a general-family design
    (mgcv ``drop.intercept=TRUE``, used by cox.ph). Implemented for the
    single-LP families that set the flag: the intercept is column 0 of
    LP 0; every later column index shifts down by one."""
    if md.n_lp != 1:
        raise NotImplementedError(
            "drop.intercept is only wired for single-LP general families")
    if "(Intercept)" not in md.column_names:
        return md
    idx = md.column_names.index("(Intercept)")
    keep = [c for c in range(md.p) if c != idx]
    X = md.X[:, keep]

    def _shift(c):
        return c - 1 if c > idx else c

    slots = [_PenaltySlot(block=s.block, col_start=_shift(s.col_start),
                          col_end=_shift(s.col_end), S=s.S, S_scale=s.S_scale)
             for s in md.slots]
    ranges = [(_shift(a), _shift(b)) for (a, b) in md.block_col_ranges]
    names = [nm for k, nm in enumerate(md.column_names) if k != idx]
    nsdf = [md.nsdf[0] - 1]
    lp0 = md.lps[0]
    lp0.X = X
    lp0.nsdf = nsdf[0]
    lp0.column_names = names
    lp0.block_col_ranges = ranges
    lp0.slots = slots
    return _MultiDesign(
        lps=md.lps, X=X, lpi=[np.arange(X.shape[1])], y=md.y,
        blocks=md.blocks, block_col_ranges=ranges, slots=slots,
        column_names=names, nsdf=nsdf, pstart=[0], offsets=md.offsets,
        n_lp=1, L=md.L, n_work=md.n_work, p=X.shape[1], n=md.n)


def _multi_lpmatrix(md: _MultiDesign, newdata) -> tuple[np.ndarray,
                                                        list[np.ndarray]]:
    """Linear-predictor matrix for new data — predict.gam's
    ``type="lpmatrix"`` with the ``lpi`` attribute (mgcv.r:2704, 3173).
    Returns ``(X_new, lpi)``; each LP's columns are rebuilt with the
    same per-formula recipe as gam.predict's newdata branch."""
    from ..formula import (_apply_smooth_arg_exprs, _smooth_arg_expr_map,
                           materialize, normalize_data)
    newdata = normalize_data(newdata)
    cols: list[np.ndarray] = []
    for lp in md.lps:
        nd = newdata
        expr_map = _smooth_arg_expr_map(lp.expanded)
        if expr_map:
            nd = _apply_smooth_arg_exprs(nd, expr_map)
        n_user = nd.height
        nd, n_stubs = _add_factor_stub_rows(nd, lp.data)
        X_param = materialize(lp.expanded, nd).to_numpy().astype(float)
        if X_param.shape[1] == 0:
            X_param = np.zeros((nd.height, 0))
        parts = [X_param]
        for b in lp.blocks:
            if b.spec is None:
                raise RuntimeError(
                    f"smooth block {b.label!r} has no BasisSpec; "
                    "lpmatrix on newdata requires one."
                )
            parts.append(np.asarray(b.spec.predict_mat(nd), dtype=float))
        X_lp = (np.concatenate(parts, axis=1) if len(parts) > 1
                else X_param)
        if n_stubs > 0:
            X_lp = X_lp[:n_user]
        cols.append(X_lp)
    return np.concatenate(cols, axis=1), md.lpi


def _zero_terms_exclude(X, terms, exclude, plabels, pidx, icols, slabels,
                        sranges):
    """predict.gam's terms=/exclude= design zeroing (mgcv.r:2993-3026).

    Returns a copy of ``X`` with the de-selected terms' columns zeroed:
    a term is kept iff (``terms`` is None or its label is listed) and its
    label is not in ``exclude``. The intercept columns (R assign == 0)
    are zeroed when ``"(Intercept)"`` is excluded, or when ``terms`` is
    given without listing it. Smooth blocks zero whole column ranges
    (mgcv skips their PredictMat — value-identical).
    """
    X = np.array(X, dtype=float, copy=True)
    tset = set(terms) if terms is not None else None
    eset = set(exclude) if exclude is not None else set()
    if icols.size and ("(Intercept)" in eset
                       or (tset is not None and "(Intercept)" not in tset)):
        X[:, icols] = 0.0
    for lab, idx in zip(plabels, pidx):
        if idx.size and (lab in eset
                         or (tset is not None and lab not in tset)):
            X[:, idx] = 0.0
    for lab, (a, b) in zip(slabels, sranges):
        if lab in eset or (tset is not None and lab not in tset):
            X[:, a:b] = 0.0
    return X


# ---------------------------------------------------------------------------
# §5.3 gam.fit5 — general-family fitting (gam.fit4.r:941-1477)
#
# Inner Newton on β of the penalized log-likelihood l(β) − β'Sλβ/2 for
# "general families" (several linear predictors, likelihood supplied
# directly via family.ll). mgcv's stabilization protocol, line by line:
#   1. the ldetS (rp) reparameterization is applied to X up front;
#   2. the penalized Hessian is diagonally preconditioned when its
#      diagonal is positive (otherwise it's indefinite anyway and gets
#      the |min D| + √eps·|max D| ridge instead);
#   3. Newton steps through a pivoted Cholesky with an escalating ×100
#      ridge on rank failure; 0.1·‖β‖ step cap; step-halving then
#      steepest-ascent fallback;
#   4. an indefinite Hessian at apparent convergence triggers the
#      saddle-perturbation protocol (≤5 deterministic coef shakes), and
#      from iteration 4 the fundamental-rank check on the BALANCED
#      penalized Hessian — unidentifiable parameters are dropped (lpi
#      reindexed, X/St reduced) and iteration continues;
#   5. all remaining computations run in the reduced space.
# Then implicit differentiation through the preconditioned factor gives
# d1b/d2b, d1ldetH/d2ldetH, d1bSb/d2bSb and the dVkk curvature-check
# matrix, assembled into
#   REML  = −[(l − β'Sβ/2)/γ + log|S|₊/2 − log|H+S|/2 + Mp·log(2π)/2
#             − log(γ)/2]
# with exact first/second log-sp derivatives (gam.fit4.r:1409-1414).
# scale.est ≡ 1: the scale never enters the outer problem. The NCV
# branches (nei) are out of scope.
# ---------------------------------------------------------------------------


def _pivoted_chol(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """R ``chol(A, pivot=TRUE)`` via LAPACK dpstrf: upper ``U``, 0-based
    pivot vector and detected rank, with ``A[piv][:, piv] = U'U`` (rows
    past the rank zeroed)."""
    from scipy.linalg.lapack import dpstrf
    c, piv, r_eff, _info = dpstrf(A, lower=0)
    U = np.triu(c)
    if int(r_eff) < A.shape[0]:
        U[int(r_eff):, :] = 0.0
    return U, piv.astype(int) - 1, int(r_eff)


def _fit5_solve(L: np.ndarray, piv: np.ndarray, ipiv: np.ndarray,
                D: np.ndarray, v: np.ndarray) -> np.ndarray:
    """gam.fit5's preconditioned penalized-Hessian solve:
    ``D·(U⁻¹ U'⁻¹ (D·v)[piv])[ipiv]`` — i.e. Hp⁻¹ v through the pivoted
    upper factor of the preconditioned Hp (gam.fit4.r:1082)."""
    u = D[:, None] * v if v.ndim == 2 else D * v
    u = u[piv]
    t1 = solve_triangular(L, u, lower=False, trans="T")
    t2 = solve_triangular(L, t1, lower=False)
    out = t2[ipiv]
    return D[:, None] * out if v.ndim == 2 else D * out


def _gam_fit5(x, y, lsp, sl: _Sl, *, weights=None, offset=None,
              deriv: int = 2, family, scoreType: str = "REML", Mp: int = -1,
              start=None, gamma: float = 1.0, nei=None, lpi,
              epsilon: float = 1e-7, maxit: int = 200) -> dict:
    """mgcv ``gam.fit5`` (gam.fit4.r:941) — general penalized-likelihood
    fitter for extended families with a formula list (gaulss, mvn, …).

    Signature mirrors mgcv's
    ``gam.fit5(x, y, lsp, Sl, weights, offset, deriv, family, scoreType,
    control, Mp, start, gamma, nei)`` in order and name, with three hea
    deltas: ``lpi`` is threaded explicitly (mgcv's ``Sl``/``G`` carries the
    per-LP column index arrays; hea's ``sl`` does not); ``epsilon``/``maxit``
    stand in for mgcv's ``control`` list; and ``scoreType`` is fixed at
    ``"REML"`` with ``nei`` (NCV neighbourhoods) unsupported — both accepted
    for signature parity and validated below.

    ``x`` must already carry the Sl *initial* reparameterization
    (estimate.gam applies ``Sl.initial.repara`` before fitting,
    mgcv.r:1899-1903); the ldetS (``rp``) reparameterization is applied
    and undone internally, exactly like mgcv. ``lpi``: per-LP 0-based
    column index arrays. ``offset``: per-LP offset arrays (entries may
    be ``None``). ``Mp``: the criterion's prior null-space dimension (a
    setup-time quantity). Returned dict mirrors mgcv's ret list —
    ``coefficients``/``db_drho`` have the rp-reparameterization undone
    (``Sl.repara(inverse)`` / ``Sl.repa(l=-1)``) but NOT the initial
    one; the caller undoes that, like estimate.gam.
    """
    if scoreType != "REML":
        raise NotImplementedError(
            "gam.fit5 general families use scoreType='REML' only "
            f"(mgcv.r:1894-1898 coerces the method); got {scoreType!r}.")
    if nei is not None:
        raise NotImplementedError(
            "gam.fit5 neighbourhood cross-validation (nei) is not ported.")
    # ``x`` may be the compressed multi-LP design (family.DiscreteX) —
    # the discrete general-family driver mgcv never wired (bam.r:2653
    # stops; gamlss.gH's discrete branch was left dormant). The fit's
    # p-space algebra is unchanged; only the n-dependent seams differ
    # (family.ll assembles through the discrete kernels, and the two
    # column reparameterizations become p-space boundary transforms).
    discrete = isinstance(x, _DiscreteX)
    if discrete:
        if deriv > 1:
            raise NotImplementedError(
                "discrete general-family fits carry first-order REML "
                "derivatives only (mgcv gamlss.gH stops at deriv>1, "
                "gamlss.r:777) — sp selection must run EFS or BFGS, "
                "not full Newton.")
        q = x.design.p
    else:
        x = np.asarray(x, dtype=float)
        q = x.shape[1]
    y = np.asarray(y, dtype=float)
    lsp = np.asarray(lsp, dtype=float).reshape(-1)
    nobs = y.shape[0]
    n_sp = int(lsp.size)
    penalized = len(sl) > 0
    warn: list[str] = []
    eps_mach = float(np.finfo(float).eps)

    lpi = [np.asarray(ix, dtype=int) for ix in lpi]
    if weights is None:
        weights = np.ones(nobs)
    if offset is None:
        offset = [None] * len(lpi)

    if penalized:
        # mgcv calls ldetS WITHOUT deriv → its default 2 (gam.fit4.r:969)
        # even for deriv=0 fits: S1 (= ldet1) must exist on every return
        # — efsud's update reads it from deriv-0 fits.
        rp = _ldet_s(sl, lsp, root=True, stot=True, deriv=2)
        if not discrete:
            x = _sl_repara(rp["rp"], x)
        Sb = _sl_repa(rp["rp"], sl.S, lt=-2, r=-1)   # balanced penalty
        St = np.asarray(rp["S"], dtype=float)
        E = rp["E"]
        if start is not None:
            start = _sl_repara(rp["rp"], np.asarray(start, dtype=float))
    else:                       # unpenalized: no derivatives required
        deriv = 0
        rp = {"ldetS": 0.0, "ldet1": np.zeros(0),
              "ldet2": np.zeros((0, 0)), "rp": []}
        St = np.zeros((q, q))
        E = np.zeros((0, q))
        Sb = np.zeros((q, q))

    if discrete:
        # hea seam (mgcv has no discrete gam.fit5 to port): the dense
        # driver reparameterizes X's COLUMNS twice — Sl.initial.repara at
        # the estimate.gam boundary (X·D per block) and Sl.repara(rp)
        # above (X·Qs) — so coef/grad/Hess live in the transformed basis.
        # A compressed design has no column form; the same two block
        # transforms are applied in p-space around family.ll instead
        # (the pattern mgcv's own discrete fitters use on assembled
        # X'WX — never on Xd). With M = D·Qs (fit-basis coef b maps to
        # the model basis as b_x = M·b):
        #   coef/d1b in : b_x = D·(Qs·b)          — Qs then D on rows
        #   fh in       : fh_x = M·fh·M'
        #   lb out      : Qs'·(D'·lb_x)
        #   lbb out     : Qs'·D'·lbb_x·D·Qs
        # The deriv-1 d1H is tr(Hp⁻¹·∂H/∂ρ) — basis-invariant once fh
        # is supplied in the same (model) basis, so it passes through.
        def _to_x(v):
            return _sl_inirep(sl, _sl_repa(rp["rp"], v, lt=-1), lt=1)

        def _from_x_grad(g):
            return _sl_repa(rp["rp"], _sl_inirep(sl, g, lt=2), lt=1)

        def _from_x_hess(h):
            return _sl_repa(rp["rp"], _sl_inirep(sl, h, lt=2, r=1),
                            lt=1, r=2)

        def _fh_x(f):
            return _sl_inirep(sl, _sl_repa(rp["rp"], f, lt=-1, r=-2),
                              lt=1, r=2)

    if start is None:
        # the ldetS root E carries mgcv's use.unscaled attribute here
        # (gam.fit4.r:974) — initializers use it as-is.
        if discrete:
            # E is the St root in the fit basis; the family's discrete
            # initializer (gamlss.r:1035/1051) crossprods it against the
            # model-basis XWXd, so hand it over as E_x = E·M⁻¹ = E·Qs'·Di
            # and map the returned start forward through the same chain
            # the dense flow applies to user starts (Di then Qs').
            E_x = _sl_inirep(sl, _sl_repa(rp["rp"], E, r=-2), r=-1)
            start_x = family.initialize_coef(y, x, lpi, E=E_x,
                                             offset=offset,
                                             use_unscaled=True)
            start = _sl_repara(rp["rp"], _sl_initial_repara(
                sl, np.asarray(start_x, dtype=float), both_sides=False))
        else:
            start = family.initialize_coef(y, x, lpi, E=E, offset=offset,
                                           use_unscaled=True)
    coef = np.asarray(start, dtype=float).reshape(-1).copy()
    start = coef.copy()         # kept for the iconv first-step-fail path

    # the drop protocol's coordinate subset on the discrete rail (set by
    # the fundamental rank check below; None ⇔ full basis). The dense
    # rail cuts x's columns instead — here the SAME reduction lives in
    # the p-space boundary transform: M → M[:, keep], spelled as
    # embed-with-zeros on the way in and restrict on the way out, so the
    # model basis, lpi and X.lpid stay full-q and the family's assembly
    # is exactly mgcv's reduced dense one (tr(M·fh·M'·A) = tr(fh·M'AM)).
    _keep_x: dict = {"keep": None}

    if discrete:
        def llf(b, d, d1b=None, fh=None, **kw):
            k = _keep_x["keep"]
            b = np.asarray(b, dtype=float)
            if d1b is not None:
                d1b = np.asarray(d1b, dtype=float)
            if fh is not None:
                fh = np.asarray(fh, dtype=float)
            if k is not None:
                bf = np.zeros(q)
                bf[k] = b
                b = bf
                if d1b is not None:
                    d1bf = np.zeros((q, d1b.shape[1]))
                    d1bf[k, :] = d1b
                    d1b = d1bf
                if fh is not None:
                    fhf = np.zeros((q, q))
                    fhf[np.ix_(k, k)] = fh
                    fh = fhf
            ret = family.ll(
                y, x, _to_x(b), weights,
                lpi=lpi, offset=offset, deriv=d,
                d1b=None if d1b is None else _to_x(d1b),
                fh=None if fh is None else _fh_x(fh), **kw)
            if ret.get("lb") is not None:
                lb = _from_x_grad(ret["lb"])
                ret["lb"] = lb if k is None else lb[k]
            if ret.get("lbb") is not None:
                lbb = _from_x_hess(ret["lbb"])
                ret["lbb"] = lbb if k is None else lbb[np.ix_(k, k)]
            return ret
    else:
        def llf(b, d, **kw):
            return family.ll(y, x, b, weights, lpi=lpi, offset=offset,
                             deriv=d, **kw)

    ll = llf(coef, 1)
    ll0 = ll["l"] - float(coef @ St @ coef) / 2.0
    grad = ll["lb"] - St @ coef
    iconv = bool(np.max(np.abs(grad)) < epsilon * abs(ll0))
    Hp = -ll["lbb"] + St
    rank_checked = False
    rank = q
    converged = False
    drop = None
    bdrop = np.zeros(q, dtype=bool)
    perturbed = 0
    L = piv = ipiv = D = None
    iter_ = 0

    for iter_ in range(1, 2 * maxit + 1):    # main iteration
        kappaH = float(np.linalg.cond(Hp, 1))
        D = np.diag(Hp).copy()
        if np.sum(~np.isfinite(D)) > 0:
            raise FloatingPointError("non finite values in Hessian")

        if np.min(D) <= 0:      # could be indefinite or +ve semi def
            Dthresh = np.max(D) * np.sqrt(eps_mach)
            if -np.min(D) < Dthresh:
                indefinite = False
                D[D < Dthresh] = Dthresh
            else:
                indefinite = True
        else:
            indefinite = False

        if indefinite:          # Hessian indefinite, for sure
            Ib = np.eye(rank) * abs(np.min(D))
            Ip = np.eye(rank) * abs(np.max(D) * eps_mach ** 0.5)
            Hp = Hp + Ip + Ib
            D = np.ones(Hp.shape[0])
        else:                   # +ve def: cheap pivoted Cholesky
            D = D ** -0.5       # diagonal pre-conditioner
            Hp = D * (D * Hp).T
            Ip = np.eye(rank) * eps_mach ** 0.5
        L, piv, L_rank = _pivoted_chol(Hp)
        while L_rank < rank:    # rank deficient: escalate ridge ×100
            L, piv, L_rank = _pivoted_chol(Hp + Ip)
            Ip = Ip * 100.0
            indefinite = True
        ipiv = np.empty_like(piv)
        ipiv[piv] = np.arange(L.shape[0])

        if converged:
            break               # L and D now match the final Hp

        step = _fit5_solve(L, piv, ipiv, D, grad)
        c_norm = float(np.sum(coef ** 2))
        if c_norm > 0:          # limit step length to .1 of coef length
            s_norm = float(np.sqrt(np.sum(step ** 2)))
            c_norm = float(np.sqrt(c_norm))
            if s_norm > 0.1 * c_norm:
                step = step * (0.1 * c_norm / s_norm)
        s_norm = float(np.sqrt(np.sum(step ** 2)))

        coef1 = coef + step     # try the Newton step
        ll = llf(coef1, 1)
        ll1 = ll["l"] - float(coef1 @ St @ coef1) / 2.0
        khalf = 0
        fac = 2.0
        llold = ll              # keep an lbb slot through step failure
        no_change = 0
        while ((not np.isfinite(ll1)) or ll1 <= ll0) and khalf < 25:
            step = step / fac
            coef1 = coef + step
            ll = llf(coef1, 0)
            ll1 = ll["l"] - float(coef1 @ St @ coef1) / 2.0
            if np.isfinite(ll1) and ll1 >= ll0:   # no worse: get derivs
                ll = llf(coef1, 1)
            if np.isfinite(ll1) and ll1 == ll0:
                no_change += 1
            if (np.max(np.abs(coef - coef1))
                    < np.max(np.abs(coef)) * eps_mach or no_change > 1):
                khalf = 100     # step has gone nowhere — abort halving
            khalf += 1
            if khalf > 5:
                fac = 5.0

        if (not np.isfinite(ll1)) or (ll1 <= ll0 and not iconv):
            # switch to steepest ascent, scaled to Newton step length
            step = grad * (s_norm / float(np.sqrt(np.sum(grad ** 2))))
            khalf = 0

        no_change = 0
        while (((not np.isfinite(ll1)) or (ll1 <= ll0 and not iconv))
               and khalf < 25):
            step = step / 10.0
            coef1 = coef + step
            ll = llf(coef1, 0)
            ll1 = ll["l"] - float(coef1 @ St @ coef1) / 2.0
            if np.isfinite(ll1) and ll1 >= ll0:
                ll = llf(coef1, 1)
            if np.isfinite(ll1) and ll1 == ll0:
                no_change += 1
            if (np.max(np.abs(coef - coef1))
                    < np.max(np.abs(coef)) * eps_mach or no_change > 1):
                khalf = 100
            khalf += 1

        if ((np.isfinite(ll1) and ll1 >= ll0
             and (khalf < 25 or indefinite)) or iter_ == maxit):
            # step ok: accept and test
            coef = coef + step
            grad = ll["lb"] - St @ coef
            Hp = -ll["lbb"] + St
            ok = (iter_ == maxit
                  or np.max(np.abs(grad)) < epsilon * abs(ll0))
            if ok:
                if indefinite:  # not a well defined maximum
                    if perturbed == 5:
                        raise FloatingPointError(
                            "indefinite penalized likelihood in gam.fit5")
                    if iter_ < 4 or rank_checked:
                        perturbed += 1
                        alt = np.resize([0.0, 1.0], coef.size)
                        coef = (coef * (1.0 + (alt * 0.02 - 0.01)
                                        * perturbed)
                                + (alt - 0.5) * np.mean(np.abs(coef))
                                * 1e-5 * perturbed)
                        ll = llf(coef, 1)
                        ll0 = ll["l"] - float(coef @ St @ coef) / 2.0
                        grad = ll["lb"] - St @ coef
                        Hp = -ll["lbb"] + St
                    else:
                        rank_checked = True
                        # fundamental rank check on the balanced penalized
                        # Hessian (gam.fit4.r:1162-1199). Recompute lbb with
                        # the row/col-consistent crossprod (family.gamlss_gH
                        # under deterministic_xwx) so the dropped column is
                        # platform-stable: the hot-path `@` lbb is alignment-
                        # sensitive at the ~1e-13 that decides the QR pivot tie.
                        with _deterministic_xwx():
                            lbb = llf(coef, 1)["lbb"]
                        if penalized:
                            Hb = (-lbb / np.linalg.norm(lbb)
                                  + Sb / np.linalg.norm(Sb))
                        else:
                            Hb = -lbb / np.linalg.norm(lbb)
                        Db = np.abs(np.diag(Hb)).copy()
                        Db[Db < 1e-50] = 1.0
                        Db = Db ** -0.5
                        Hb = (Db * Hb).T * Db
                        from scipy.linalg import qr as _scipy_qr
                        Rq, piv_q = _scipy_qr(Hb, mode="r",
                                              pivoting=True)
                        rank = _R_rank(Rq, tol=eps_mach ** 0.9)
                        if rank < q:
                            # drop unidentifiable params and continue
                            # (gam.fit4.r:1170-1199)
                            drop = np.sort(piv_q[rank:q])
                            bdrop = np.isin(np.arange(q), drop)
                            keep = ~bdrop
                            coef = coef[keep]
                            St = St[np.ix_(keep, keep)]
                            if discrete:
                                # no x columns to cut on a compressed
                                # design: shrink the boundary transform
                                # instead (M → M[:, keep] via llf's
                                # embed/restrict). lpi and X.lpid stay
                                # full — the family keeps assembling in
                                # the full model basis.
                                _keep_x["keep"] = np.flatnonzero(keep)
                            else:
                                x = x[:, keep]
                                ij = np.full(q, -1, dtype=int)
                                ij[keep] = np.arange(int(keep.sum()))
                                lpi = [ij[ix[~np.isin(ix, drop)]]
                                       for ix in lpi]
                            ll = llf(coef, 1)
                            ll0 = (ll["l"]
                                   - float(coef @ St @ coef) / 2.0)
                            grad = ll["lb"] - St @ coef
                            Hp = -ll["lbb"] + St
                else:           # not indefinite: really converged
                    converged = True
                    # don't break: loop top refreshes L and D first
            else:
                ll0 = ll1       # step ok but not converged yet
        else:                   # step failed
            ll = llold          # restore the ll with an lbb slot
            if drop is None:
                bdrop = np.zeros(q, dtype=bool)
            if iconv and iter_ == 1:
                # OK to fail on the first step if apparently converged
                # to start with — but check improvement was impossible,
                # otherwise sp changes can produce no objective change
                converged = True
                coef = start
            else:
                converged = False
                coefp = coef * (1.0 + np.resize([-1.0, 1.0], coef.size)
                                * eps_mach ** 0.9)
                llp = llf(coef, 1)
                gradp = llp["lb"] - St @ coefp
                err = min(1e-3, kappaH * max(
                    1.0, float(np.mean(np.abs(gradp - grad)))
                    / float(np.mean(np.abs(coefp - coef)))) * eps_mach)
                if np.max(np.abs(grad / ll0)) > max(err, epsilon * 2):
                    warn.append(
                        "gam.fit5 step failed: max magnitude relative "
                        f"grad = {np.max(np.abs(grad / ll0))}")
                else:
                    # gradient already at the achievable-accuracy floor
                    # (mgcv's err estimate): a benign step failure —
                    # e.g. a warm start from a neighbouring trial ρ's
                    # optimum. mgcv stays silent here and returns no
                    # converged flag at all; hea's flag mirrors the
                    # warning gate (warned ⇔ not converged).
                    converged = True
            break               # no need to recompute L and D

    if iter_ == 2 * maxit and not converged:
        warn.append("gam.fit5 iteration limit reached: max abs grad = "
                    f"{np.max(np.abs(grad))}")

    ldetHp = (2.0 * float(np.sum(np.log(np.diag(L))))
              - 2.0 * float(np.sum(np.log(D))))

    if drop is not None:        # full coef with zeros for unidentifiable
        fcoef = np.zeros(q)
        fcoef[~bdrop] = coef
    else:
        fcoef = coef

    dVkk = d2l = d1bSb = d2bSb = d1b = d2b = None
    d1ldetH = d2ldetH = None
    llr = None
    keep = ~bdrop
    m = n_sp
    if deriv > 0:               # implicit differentiation for derivs
        d1b = np.zeros((rank, m))
        Sib, _ = _sl_term_mult(sl, fcoef, full=True)
        for i in range(m):
            d1b[:, i] = -_fit5_solve(L, piv, ipiv, D, Sib[i][keep])

        # curvature check matrix (gam.fit4.r:1253)
        dVkk = (L[:, ipiv] @ (d1b / D[:, None])).T @ \
               (L[:, ipiv] @ (d1b / D[:, None]))

        if drop is not None:
            fd1b = np.zeros((q, m))
            fd1b[keep, :] = d1b
        else:
            fd1b = d1b

        # family call for ∂H/∂ρ: trace vector at deriv 1, list above
        invU = solve_triangular(L, np.eye(L.shape[0]), lower=False)
        Hp_inv_perm = invU @ invU.T
        Hp_inv = (D[:, None] * Hp_inv_perm[np.ix_(ipiv, ipiv)]
                  * D[None, :])
        ll = llf(coef, 2 + (1 if deriv > 1 else 0), d1b=d1b, fh=Hp_inv)

        if deriv > 1:           # second derivatives of β̂
            d2b = np.zeros((rank, m * (m + 1) // 2))
            k = 0
            for i in range(m):
                for j in range(i, m):
                    v = (-ll["d1H"][i] @ d1b[:, j]
                         + _sl_mult(sl, fd1b[:, j], k=i)[keep]
                         + _sl_mult(sl, fd1b[:, i], k=j)[keep])
                    d2b[:, k] = -_fit5_solve(L, piv, ipiv, D, v)
                    if i == j:
                        d2b[:, k] = d2b[:, k] + d1b[:, i]
                    k += 1

            # last family call: tr(Hp⁻¹ ∂²H/∂ρᵢ∂ρⱼ)
            llr = llf(coef, 4, d1b=d1b, d2b=d2b, fh=(L, piv), D=D)

            d2l = np.zeros((m, m))
            for i in range(m):
                for j in range(i, m):
                    d2l[j, i] = d2l[i, j] = float(
                        d1b[:, i] @ ll["lbb"] @ d1b[:, j])

    # ----- REML score and its derivatives (gam.fit4.r:1343-1414) -----
    if deriv > 0:
        if deriv == 1 and not isinstance(ll["d1H"], list):
            d1ldetH = -np.asarray(ll["d1H"], dtype=float)
            for i in range(m):
                A = _sl_mult(sl, np.eye(q), k=i, full=True)[
                    np.ix_(keep, keep)]
                bind = np.sum(np.abs(A), axis=1) != 0
                A = A[:, bind]
                A = _fit5_solve(L, piv, ipiv, D, A)
                d1ldetH[i] += float(np.trace(A[bind, :]))
        else:
            d1ldetH = np.zeros(m)
            d1Hp = []
            for i in range(m):
                A = (-ll["d1H"][i]
                     + _sl_mult(sl, np.eye(q), k=i)[np.ix_(keep, keep)])
                d1Hp.append(_fit5_solve(L, piv, ipiv, D, A))
                d1ldetH[i] = float(np.trace(d1Hp[i]))

    if deriv > 1:
        d2ldetH = np.zeros((m, m))
        k = 0
        for i in range(m):
            for j in range(i, m):
                d2ldetH[i, j] = (-float(np.sum(d1Hp[i] * d1Hp[j].T))
                                 - float(llr["trHid2H"][k]))
                if i == j:      # add the smoothing-penalty term
                    A = _sl_mult(sl, np.eye(q), k=i, full=True)[
                        np.ix_(keep, keep)]
                    bind = np.sum(np.abs(A), axis=1) != 0
                    A = A[:, bind]
                    A = _fit5_solve(L, piv, ipiv, D, A)
                    d2ldetH[i, j] += float(np.trace(A[bind, :]))
                else:
                    d2ldetH[j, i] = d2ldetH[i, j]
                k += 1

    if deriv > 0:               # derivatives of β'Sβ
        Skb, _ = _sl_term_mult(sl, fcoef, full=True)
        Skb = [s[keep] for s in Skb]
        d1bSb = np.array([float(np.sum(coef * Skb[i]))
                          for i in range(m)])

    if deriv > 1:
        d2bSb = np.zeros((m, m))
        for i in range(m):
            Sd1b = St @ d1b[:, i]
            for j in range(i, m):
                d2bSb[j, i] = d2bSb[i, j] = 2.0 * float(np.sum(
                    d1b[:, i] * Skb[j] + d1b[:, j] * Skb[i]
                    + d1b[:, j] * Sd1b))
            d2bSb[i, i] += float(np.sum(coef * Skb[i]))

    bSb = float(coef @ St @ coef)
    REML = -float((ll["l"] - bSb / 2.0) / gamma + rp["ldetS"] / 2.0
                  - ldetHp / 2.0 + Mp * (np.log(2.0 * np.pi) / 2.0)
                  - np.log(gamma) / 2.0)
    REML1 = (None if deriv < 1 else
             -(-d1bSb / (2.0 * gamma) + rp["ldet1"] / 2.0
               - d1ldetH / 2.0))
    REML2 = (None if deriv < 2 else
             -((d2l - d2bSb / 2.0) / gamma + rp["ldet2"] / 2.0
               - d2ldetH / 2.0))

    # multiple linear predictors: η and fitted per LP
    K = len(lpi)
    linear_predictors = np.zeros((nobs, K))
    fitted_values = np.zeros((nobs, K))
    if discrete:
        from .bam import Xbd as _Xbd_kernel
        coef_x = _to_x(fcoef)
    for j in range(K):
        eta_j = (_Xbd_kernel(x.design, coef_x, lt=x.lpid[j]) if discrete
                 else x[:, lpi[j]] @ coef[lpi[j]])
        if offset[j] is not None:
            eta_j = eta_j + offset[j]
        linear_predictors[:, j] = eta_j
        fitted_values[:, j] = family.links[j].linkinv(eta_j)

    coef_out = _sl_repara(rp["rp"], fcoef, inverse=True)
    if drop is not None and d1b is not None:
        db_drho = np.zeros((q, d1b.shape[1]))
        db_drho[keep, :] = d1b
    else:
        db_drho = d1b
    if d1b is not None:
        db_drho = _sl_repa(rp["rp"], db_drho, lt=-1)

    return {
        "coefficients": coef_out, "fitted_values": fitted_values,
        "linear_predictors": linear_predictors, "scale_est": 1.0,
        "REML": REML, "REML1": REML1, "REML2": REML2,
        "rank": rank, "aic": -2.0 * ll["l"], "l": ll["l"],
        "lbb": ll["lbb"], "L": L, "piv": piv, "ipiv": ipiv,
        "bdrop": bdrop, "D": D, "St": St, "rp": rp["rp"],
        "db_drho": db_drho, "S1": rp["ldet1"], "iter": iter_,
        "dH": ll.get("d1H"), "dVkk": dVkk, "warn": warn,
        "lpi": lpi, "ldetHp": ldetHp, "ldetS": rp["ldetS"],
        "converged": converged,
    }


def _efsud(x, y, lsp, sl: _Sl, sl_setup: _Sl, *, family, lpi,
           weights=None, offset=None, control=None, Mp: int = -1,
           start=None) -> tuple[dict, np.ndarray, int]:
    """mgcv ``efsud`` (gam.fit4.r:1479-1569): the extended
    Fellner-Schall outer loop for general families. Every gam.fit5
    call runs at deriv=0, so the family only ever needs ``ll`` with
    deriv ≤ 1 — the on-ramp for ``available_derivs == 0`` custom
    families. Update per penalty: λ ← λ·[tr(Sλ⁻Sⱼ) − tr(Vb Sⱼ)]/β'Sⱼβ
    (a = S1·e^{−ρ} − trVS, floored at √eps), with the ×2 extension /
    ÷2 contraction step control and the two stop rules (EFS step small
    + REML flat over 3 steps; or log-lik change < 100·ε·|l| — mgcv's
    ``control$eps`` PARTIAL-MATCHES gam.control's ``epsilon`` in R, no
    eps field exists). The outer cap ``efs_maxit`` mirrors mgcv's
    hard-coded ``for (iter in 1:200)`` (gam.fit4.r:1493) and defaults
    to 200; it is exposed only so hea-native fits of families with two
    or more flat shape directions (which can need >200 EFS steps to
    satisfy ``efs_tol``) can converge — keep it at 200 for mgcv
    cross-engine parity.

    ``sl`` is the fitting penalty object (each fit's ldetS updates its
    λ state in place); ``sl_setup`` must be a pristine Sl.setup copy
    for the termMult traces — mgcv's G$Sl stays at setup state by R
    copy semantics, and Vb here is only rp-undone (still in the
    initial-repara basis sl_setup's penalties live in).

    ``control`` is mgcv's ``gam.control()`` bundle (mgcv threads it into
    efsud, gam.fit4.r:1480); the loop reads ``epsilon`` (inner PIRLS tol +
    the ``100·eps·|l|`` log-lik stop — mgcv's ``control$eps`` partial-matches
    ``epsilon``), ``efs_lspmax``/``efs_tol``, and the ``efs_maxit`` cap
    (mgcv hard-codes ``for iter in 1:200``; hea exposes it — keep at 200
    for cross-engine parity).
    """
    ctrl = control or _GAM_CONTROL_DEFAULTS
    epsilon = ctrl["epsilon"]
    efs_lspmax = ctrl["efs_lspmax"]
    efs_tol = ctrl["efs_tol"]
    efs_maxit = ctrl["efs_maxit"]
    lsp = np.asarray(lsp, dtype=float) + 2.5
    mult = 1.0
    tiny = float(np.finfo(float).eps) ** 0.5

    def fit_at(lsp_arg, st):
        return _gam_fit5(x, y, lsp_arg, sl, family=family, lpi=lpi,
                         weights=weights, offset=offset, Mp=Mp,
                         deriv=0, start=st, gamma=1.0, epsilon=epsilon)

    fit = fit_at(lsp, start)
    score_hist = np.zeros(efs_maxit)
    old_ll = None
    it = 0
    for it in range(1, efs_maxit + 1):
        start = fit["coefficients"]
        L_f, piv, ipiv = fit["L"], fit["piv"], fit["ipiv"]
        D = np.asarray(fit["D"], dtype=float)
        Dm = np.diag(D)[piv, :]
        sol = solve_triangular(L_f, Dm, lower=False, trans="T")[ipiv, :]
        Vb = sol.T @ sol
        bdrop = np.asarray(fit["bdrop"], dtype=bool)
        if bdrop.any():
            q = bdrop.size
            ibd = ~bdrop
            Vt, Vb = Vb, np.zeros((q, q))
            Vb[np.ix_(ibd, ibd)] = Vt
        Vb = _sl_repara(fit["rp"], Vb, inverse=True)
        SVb, inds = _sl_term_mult(sl_setup, Vb, full=False)
        trVS = np.array([float(np.trace(SVb[i][:, inds[i]]))
                         for i in range(len(SVb))])
        st_arr = np.asarray(start, dtype=float)
        Sb, _ = _sl_term_mult(sl_setup, st_arr, full=True)
        bSb = np.array([float(np.sum(st_arr * Sb[i]))
                        for i in range(len(Sb))])

        a = np.maximum(tiny, np.asarray(fit["S1"], dtype=float)
                       * np.exp(-lsp) - trVS)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = a / np.maximum(tiny, bSb)
        r[(a == 0) & (bSb == 0)] = 1.0
        r[~np.isfinite(r)] = 1e6
        lsp1 = np.minimum(lsp + np.log(r) * mult, efs_lspmax)
        max_step = float(np.max(np.abs(lsp1 - lsp)))
        old_reml = fit["REML"]
        fit = fit_at(lsp1, start)

        if fit["REML"] <= old_reml:     # improvement
            if max_step < 0.05:         # consider step extension
                lsp2 = np.minimum(lsp + np.log(r) * mult * 2.0, 12.0)
                fit2 = fit_at(lsp2, start)
                if fit2["REML"] < fit["REML"]:
                    fit, lsp = fit2, lsp2
                    mult = mult * 2.0
                else:
                    lsp = lsp1
            else:
                lsp = lsp1
        else:                           # no improvement: contract
            while fit["REML"] > old_reml and mult > 1.0:
                mult = mult / 2.0
                lsp1 = np.minimum(lsp + np.log(r) * mult, efs_lspmax)
                fit = fit_at(lsp1, start)
            lsp = lsp1
            if mult < 1.0:
                mult = 1.0
        score_hist[it - 1] = fit["REML"]
        # break if EFS step small and REML flat over the last 3 steps
        if (it > 3 and max_step < 0.05
                and float(np.max(np.abs(np.diff(
                    score_hist[it - 4:it])))) < efs_tol):
            break
        # or break if the log likelihood has stopped changing
        if it == 1:
            old_ll = fit["l"]
        else:
            if abs(old_ll - fit["l"]) < 100.0 * epsilon * abs(fit["l"]):
                break
            old_ll = fit["l"]
    return fit, lsp, it


def _single_sp(X, S, target: float = 0.5, tol: float | None = None) -> float:
    """mgcv ``single.sp`` (mgcv.r:4504): the smoothing parameter giving a
    target average e.d.f. per penalized term for a SINGLE-penalty problem —
    ``X`` the model matrix, ``S`` the penalty. Returns ``exp(λ̂)``, or ``-1.0``
    on a backsolve failure (rank-deficient ``X``).

    ``RSR = R⁻ᵀ S R⁻¹`` (R = qr(X).R) has the generalized eigenvalues of
    ``(S, XᵀX)``; ``λ̂`` solves ``mean(1/(1+e^λ·dᵢ)) = target`` by Brent
    root-finding on the bracket mgcv walks out (mgcv's ``uniroot``). mgcv's
    fitter reaches this only through ``initial.sp(expensive=TRUE)``
    (mgcv.r:4663), which the gam/bam fit path never sets; it is otherwise an
    exported utility (parity-tested against ``mgcv:::single.sp``)."""
    from scipy.optimize import brentq

    from ..R.linalg import dqrdc2
    if tol is None:
        tol = float(np.finfo(float).eps) * 100.0
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    # R <- qr.R(qr(X)): R's qr() is dqrdc2 (rank-revealing LINPACK QR,
    # negligible columns cycled last), not LAPACK dgeqrf. The distinction
    # is load-bearing: for rank-deficient X, dqrdc2 leaves an EXACT 0.0 on
    # the trailing diagonal, which is precisely what backsolve's
    # singularity error (bakslv.c checks `== 0`) — and hence the -1
    # return — keys on; an unpivoted BLAS-dispatched dgeqrf leaves
    # CPU-dependent ~1e-17 noise there instead, silently changing the
    # answer. Like qr.R, the triangle is taken in PIVOTED column order and
    # used against the unpivoted S (mgcv's own "### BUG? pivoting?").
    qr_mat, _, _, _ = dqrdc2(X)
    R = np.triu(qr_mat[:min(X.shape), :])
    try:
        RS = solve_triangular(R, S, lower=False, trans="T")
        RSR = solve_triangular(R, RS.T, lower=False, trans="T")
    except (np.linalg.LinAlgError, ValueError):
        return -1.0
    RSR = 0.5 * (RSR + RSR.T)
    d = np.linalg.eigvalsh(RSR)
    d = d[d > d.max() * tol]

    def ff(lam):
        return float(np.mean(1.0 / (1.0 + np.exp(lam) * d)) - target)

    lower = 0.0
    while ff(lower) <= 0.0:
        lower -= 1.0
    upper = lower
    while ff(upper) > 0.0:
        upper += 1.0
    root = brentq(ff, lower, upper, xtol=float(np.finfo(float).eps) ** 0.25)
    return float(np.exp(root))


def _initial_sp_general(X, y, family, slots: list["_PenaltySlot"], lpi,
                        *, weights=None, offsets=None,
                        L=None, start=None) -> np.ndarray:
    """mgcv ``initial.spg``'s general-family branch (mgcv.r:4541-4557
    plus the L-regression tail at :4615-4620): per penalty,
    λᵢ = 0.3·‖Z'H₀Z‖_M / ‖Z'SᵢZ‖_M with H₀ = −lbb at the family's
    ``initialize_coef`` start and Z a pivoted-Cholesky basis of Sᵢ's
    range when Sᵢ is rank-deficient (norm "M" = max |entry|).

    Staged exactly like estimate.gam: ``X`` is the *initial-
    reparameterized* design while the slot penalties stay in MODEL
    coordinates (mgcv passes G$S unchanged — the seed heuristic
    tolerates the mismatch), and the initializer's regularizer E is
    G$Eb — totalPenaltySpace's balanced root Σ Sᵢ/‖Sᵢ‖_F
    (gam.fit3.r:2661-2684), eigen-rooted at eps^0.66. Returns the
    WORKING log-sp vector: with an id-linkage ``L``, mgcv's
    ``lm(lsp ~ L − 1)`` least-squares collapse.
    """
    if isinstance(X, _DiscreteX):
        # compressed design: everything below reaches X only through the
        # family (initialize_coef / ll), which dispatch on DiscreteX.
        p = X.design.p
    else:
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones(y.shape[0])
    eps = float(np.finfo(float).eps)

    St = np.zeros((p, p))
    for s in slots:
        a, b = s.col_start, s.col_end
        nrm = float(np.sqrt(np.sum(s.S * s.S)))
        if nrm > 0:
            St[a:b, a:b] += s.S / nrm
    if np.any(St):
        ev, Y = np.linalg.eigh(0.5 * (St + St.T))
        kept = ev > ev.max() * eps ** 0.66
        Eb = (np.sqrt(ev[kept]) * Y[:, kept]).T
    else:
        Eb = np.zeros((0, p))

    # mgcv evals family$initialize unconditionally here (mgcv.r:4540),
    # but every general family's initialize expression guards on
    # ``is.null(start)`` — so a user start= (already irp-mapped by
    # estimate.gam, mgcv.r:1903, and reaching initial.spg through
    # ``...``) skips the pilot solve and seeds the Hessian directly.
    # hea mirrors: initialize_coef IS the inside of that guard. On the
    # discrete rail the caller passes the MODEL-basis vector instead
    # (family.ll dispatches on DiscreteX in model coordinates; the irp
    # transform is gam.fit5's boundary seam there).
    if start is None:
        start = family.initialize_coef(y, X, lpi, E=Eb, offset=offsets)
    lbb = family.ll(y, X, start, weights, lpi=lpi, offset=offsets,
                    deriv=1)["lbb"]

    lam = np.zeros(len(slots))
    for i, s in enumerate(slots):
        a, b = s.col_start, s.col_end
        S_i = np.asarray(s.S, dtype=float)
        k = S_i.shape[1]
        w_ev = np.linalg.eigvalsh(0.5 * (S_i + S_i.T))
        rank_i = (int(np.sum(w_ev > eps ** 0.8 * float(w_ev.max())))
                  if w_ev.size else 0)
        if rank_i < k:
            # basis for the row/col space of S_i; project into it
            _U, piv_i, _r = _pivoted_chol(S_i)
            Z = S_i[:, piv_i[:rank_i]]
            Z = Z / float(np.abs(Z).sum(axis=0).max())  # R norm(Z), "O"
            ZHZ = -(Z.T @ lbb[a:b, a:b] @ Z)
            ZSZ = Z.T @ S_i @ Z
        else:
            ZHZ = -lbb[a:b, a:b]
            ZSZ = S_i
        lam[i] = (0.3 * float(np.abs(ZHZ).max())
                  / float(np.abs(ZSZ).max()))
    lsp = np.log(lam)
    if L is not None:
        lsp = np.linalg.lstsq(np.asarray(L, dtype=float), lsp,
                              rcond=None)[0]
    return lsp


class _PenaltySlot:
    """One smoothing-param slot: the k×k S matrix and its col range in the
    full design. Each SmoothBlock contributes len(S_list) slots.

    ``S_scale`` is mgcv's ``sm$S.scale`` entry for this penalty — the
    ``maS`` factor ``_scale_penalty`` divided it by (1 where no rescale
    applied, incl. select's appended null-space penalty, smooth.r:4241).
    ``vcomp``'s default ``rescale=True`` divides the slot's sp by it."""
    __slots__ = ("block", "col_start", "col_end", "S", "S_scale")

    def __init__(self, *, block: SmoothBlock, col_start: int, col_end: int,
                 S: np.ndarray, S_scale: float = 1.0):
        self.block = block
        self.col_start = col_start
        self.col_end = col_end
        self.S = S
        self.S_scale = S_scale


def _block_s_scale(b: SmoothBlock, j: int) -> float:
    """``sm$S.scale[j]`` for a block's j-th penalty. Penalties appended
    after construction (select's null-space Sf) sit past the recorded
    list and get mgcv's 1.0 (smooth.r:4241/4259)."""
    if b.S_scale is not None and j < len(b.S_scale):
        return float(b.S_scale[j])
    return 1.0


def _wald_pinv(V: np.ndarray, M: int,
               rank_tol: float) -> tuple[np.ndarray, int]:
    """summary.gam's local ``pinv`` (mgcv.r:3869-3881): eigen
    pseudo-inverse of a Wald-test covariance block, truncated at
    ``rank.tol·λ_max`` and capped at ``M``; returns ``(V⁻, rank)`` —
    the rank is the test's df, so rank-deficient blocks (zero rows from
    dropped coefficients, truncated parametric space) test on reduced
    df exactly like mgcv."""
    vals, vecs = np.linalg.eigh(V)
    vals = vals[::-1]                  # R eigen: descending
    vecs = vecs[:, ::-1]
    M1 = int(np.sum(vals > rank_tol * vals[0])) if vals.size else 0
    if M > M1:
        M = M1
    ivals = np.zeros_like(vals)
    if M > 0:
        ivals[:M] = 1.0 / vals[:M]
    return (vecs * ivals) @ vecs.T, M


class _FitState:
    """Fit-at-one-ρ bundle, populated by either the Gaussian closed-form
    solver or the PIRLS loop. ``rss`` is kept as an alias for ``dev`` so
    the Gaussian-only post-fit code reads cleanly; for non-Gaussian
    families ``rss`` is the deviance (``rss == dev``)."""
    __slots__ = (
        "beta", "eta", "mu", "w", "z", "alpha",
        "dev", "pen", "rss",
        "A_chol", "A_chol_lower", "A_inv",
        "S_full", "log_det_A", "E_aug",
        "is_fisher_fallback", "is_working_gaussian",
        "converged", "boundary", "warn",
        "scale_est",
        "_lderivs", "_dwdeta", "_d2wdeta2", "_ddeta", "_ddraw",
    )

    def __init__(self, *, beta, dev, pen, A_chol, A_chol_lower,
                 S_full, log_det_A,
                 eta=None, mu=None, w=None, z=None, alpha=None,
                 is_fisher_fallback=False, is_working_gaussian=False,
                 converged=True, boundary=False, warn=None,
                 E_aug=None, A_inv=None, scale_est=None):
        self.beta = beta
        self.dev = dev
        self.rss = dev               # back-compat alias for Gaussian path
        self.pen = pen
        self.eta = eta
        self.mu = mu
        self.w = w
        self.z = z
        self.alpha = alpha
        self.A_chol = A_chol
        self.A_chol_lower = A_chol_lower
        # Optional precomputed A⁻¹ in the ORIGINAL basis. When set (bam's
        # discrete POI reuse of Sl.fitChol's PP — bgam.fitd:823), the
        # post-fit reads it directly instead of cho_solve(A_chol); leaving
        # it None preserves the standard A_chol path (gam + all other bam
        # branches). Lets a rank-deficient pseudo-inverse gauge survive
        # instead of the A_chol ridge fallback.
        self.A_inv = A_inv
        self.S_full = S_full
        self.log_det_A = log_det_A
        # PIRLS bookkeeping, mirroring gam.fit3's converged/boundary/warn:
        # intermediate outer-Newton evaluations never print these (mgcv
        # printWarn=FALSE); the constructor surfaces them for the final fit.
        self.converged = converged
        self.boundary = boundary
        self.warn = [] if warn is None else warn
        # Penalty square root used by the augmented-QR solves at this ρ
        # (consumers that refactor with other weights — _fisher_view —
        # reuse it).
        self.E_aug = E_aug
        # gam.fit4's ``scale.est`` (gam.fit4.r:807 — the local scale, i.e.
        # the estimate.theta-updated φ for scale-unknown extended families).
        # Set only by `_gam_fit4` in EFS mode; `efsudr` reads it as mgcv's
        # ``fit$scale`` (R's $ partial-matches scale → scale.est).
        self.scale_est = scale_est
        # True iff PIRLS forced α=1 at convergence because Newton's
        # α formula produced a w<0. In that case dα/dμ is taken as 0
        # for derivative purposes (the analytical α'(μ) is not
        # consistent with the override).
        self.is_fisher_fallback = is_fisher_fallback
        # True for bam's reduced Gaussian working fits (β̂ solved on the
        # compressed (R, f) with the IRLS weights already absorbed into
        # R). mgcv swaps ``G$family <- gaussian()`` / ``G$w <- 1`` before
        # its compressed refits (bam.r:932, 1266-1267), so every
        # criterion-tail quantity is the *working-Gaussian* one there:
        # dW/dη ≡ 0, Pearson ≡ deviance ≡ the working RSS, V ≡ 1. The
        # n-side readers (`_dw_deta`, `_pearson_and_deriv`,
        # `_fit3_scale_est`, ...) consult this flag to read those
        # quantities from the p-space fit instead of recomputing
        # response-scale statistics that don't exist on the reduced
        # problem.
        self.is_working_gaussian = is_working_gaussian
        # Lazily-cached link/variance derivative bundle for the REML weight-
        # derivative chain (set by gam._fit_link_derivs) — the same converged
        # (μ, η) feed _dw_deta, _d2w_deta2 and the gradient, which recomputed
        # variance/dvar/d2link/… independently; mgcv computes them once (gdi1).
        self._lderivs = None
        self._dwdeta = None
        self._d2wdeta2 = None
        self._ddeta = None
        self._ddraw = None


# ---------------------------------------------------------------------------
# vis.gam helpers + result
# ---------------------------------------------------------------------------


def _is_factor_like_col(col: pl.Series) -> bool:
    return col.dtype in (pl.Categorical, pl.Enum, pl.String, pl.Utf8, pl.Object)


def _has_variation(col: pl.Series) -> bool:
    vals = col.drop_nulls()
    if len(vals) <= 1:
        return False
    return vals.n_unique() > 1


def _grid_axis(col: pl.Series, n_grid: int) -> np.ndarray:
    """Build a 1D grid for one ``view`` axis.

    Numeric: ``linspace(min, max, n_grid)``. Factor: the levels (truncated
    to ``n_grid`` if there are more, or each level repeated to fill the
    grid otherwise — same shape as mgcv's ``fac.seq``)."""
    if _is_factor_like_col(col):
        from ..formula import _factor_levels  # local import to avoid cycle

        levels = list(_factor_levels(col))
        fn = len(levels)
        if fn >= n_grid:
            return np.array(levels[:n_grid], dtype=object)
        # Repeat each level ⌊n_grid/fn⌋ times then pad the tail with the
        # last level — mirrors mgcv's fac.seq.
        ln = n_grid // fn
        out = np.array([lev for lev in levels for _ in range(ln)] +
                       [levels[-1]] * (n_grid - ln * fn), dtype=object)
        return out
    arr = col.drop_nulls().to_numpy().astype(float)
    return np.linspace(float(arr.min()), float(arr.max()), n_grid)


def _too_far_mask(
    g1: np.ndarray, g2: np.ndarray,
    d1: pl.Series, d2: pl.Series,
    dist: float,
) -> np.ndarray:
    """Port of mgcv's ``exclude.too.far``.

    Normalize grid + data to the grid's [0, 1] box, compute each grid
    point's nearest-data-point distance, return a boolean mask of grid
    points farther than ``dist``. Factor view axes are not supported by
    mgcv's distance metric — we return all-False for those.
    """
    if _is_factor_like_col(d1) or _is_factor_like_col(d2):
        return np.zeros(g1.shape[0], dtype=bool)

    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    d1 = d1.drop_nulls().to_numpy().astype(float)
    d2 = d2.drop_nulls().to_numpy().astype(float)
    # mgcv normalizes by the grid's range, then both grid + data live in
    # the grid's [0, 1] box (data outside [0, 1] is preserved as-is).
    g1_min, g1_max = g1.min(), g1.max()
    g2_min, g2_max = g2.min(), g2.max()
    g1_span = g1_max - g1_min if g1_max > g1_min else 1.0
    g2_span = g2_max - g2_min if g2_max > g2_min else 1.0
    g1n = (g1 - g1_min) / g1_span
    g2n = (g2 - g2_min) / g2_span
    d1n = (d1 - g1_min) / g1_span
    d2n = (d2 - g2_min) / g2_span
    # Pairwise squared distance — fine for n_grid² ≈ 900 × n data.
    dx = g1n[:, None] - d1n[None, :]
    dy = g2n[:, None] - d2n[None, :]
    min_dist = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return min_dist > dist


class VisResult:
    """Output of :meth:`gam.vis`.

    Attributes
    ----------
    view : (str, str)
        The two covariate names the surface is over.
    m1, m2 : 1D ndarray
        Axis values, length ``n_grid`` each (numeric: linspace; factor: levels).
    fit : (n_grid, n_grid) ndarray
        ``fit[i, j]`` is the prediction at ``(m1[i], m2[j])``. ``NaN`` where
        ``too_far`` masked the grid.
    se : (n_grid, n_grid) ndarray, optional
        Pointwise SE if ``vis(se=True)``; otherwise ``None``.
    type : "link" | "response"
        Scale of fit and se.
    """

    __slots__ = ("view", "m1", "m2", "fit", "se", "type")

    def __init__(self, *, view, m1, m2, fit, se, type):
        self.view = view
        self.m1 = m1
        self.m2 = m2
        self.fit = fit
        self.se = se
        self.type = type

    def __repr__(self) -> str:
        z = self.fit
        return (
            f"VisResult(view={self.view}, n_grid=({len(self.m1)},{len(self.m2)}), "
            f"type={self.type!r}, "
            f"fit range=[{np.nanmin(z):.4g}, {np.nanmax(z):.4g}], "
            f"se={'yes' if self.se is not None else 'no'})"
        )

    def plot(
        self,
        kind: str = "contour",
        ax=None,
        figsize: tuple | None = None,
        cmap: str = "viridis",
        levels: int = 20,
        contour_levels=None,
        vmin: float | None = None,
        vmax: float | None = None,
        extend: str = "neither",
        se_mult: float = 0.0,
        elev: float = 30.0,
        azim: float = -60.0,
        zlabel: str | None = None,
        aspect: str | float | None = "equal",
        colorbar: bool = True,
        clabel: bool = False,
        clabel_kwargs: dict | None = None,
    ):
        """Render the surface.

        ``kind="contour"`` draws a filled contour with overlaid lines;
        ``kind="persp"`` draws a 3D wireframe (mgcv's default). When
        ``se_mult > 0`` and ``se`` is present, persp also draws ±``se_mult``·SE
        envelopes (same convention as ``vis.gam(se=...)``).

        ``aspect`` (contour only): ``"equal"`` (default — one data-unit on
        x takes the same screen length as one on y, so ticks are visually
        the same size), ``"square"`` (square plotting box regardless of
        data ranges), a float (height/width ratio), or ``None`` (matplotlib
        default).
        """
        if kind not in ("contour", "persp"):
            raise ValueError(f"kind must be 'contour' or 'persp'; got {kind!r}")

        x_lab, y_lab = self.view
        z_lab = zlabel or (
            "linear predictor" if self.type == "link" else "response"
        )

        # Numeric coords for plotting — factor axes get plotted at their
        # ordinal positions with the level names as ticks.
        m1_num, m1_ticks = _axis_for_plot(self.m1)
        m2_num, m2_ticks = _axis_for_plot(self.m2)

        if kind == "contour":
            if ax is None:
                _fig, ax = plt.subplots(figsize=figsize or (6, 5))
            # M1 (rows, axis 0) → x; M2 (cols, axis 1) → y; transpose so that
            # contourf's (x, y, Z) call has Z[j, i] for x=m1[i], y=m2[j].
            X, Y = np.meshgrid(m1_num, m2_num, indexing="xy")
            Z = self.fit.T
            # ``vmin``/``vmax`` clip the colormap normalization AND the
            # colorbar range. Matplotlib's ``contourf`` treats those two
            # as separate: ``vmin``/``vmax`` only steer the cmap norm,
            # while the colorbar follows ``levels``. So when the user
            # supplies ``vmin``/``vmax`` but leaves ``levels`` at its
            # int default, derive a level array from [vmin, vmax] so
            # the colorbar matches. Default ``extend`` flips to
            # ``"both"`` in that case so out-of-range Z stays painted
            # at the cmap endpoints rather than going blank.
            extend_used = extend
            if vmin is not None and vmax is not None and isinstance(levels, int):
                n_lev = levels
                levels = np.linspace(float(vmin), float(vmax), n_lev + 1)
                if extend_used == "neither":
                    extend_used = "both"
            cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax, extend=extend_used)
            # Bump line strength when labels are on so the numbers sit on
            # readable lines, not faint guides.
            line_alpha = 0.8 if clabel else 0.5
            line_width = 0.6 if clabel else 0.4
            line_levels = contour_levels if contour_levels is not None else levels
            cs = ax.contour(X, Y, Z, levels=line_levels, colors="black",
                            linewidths=line_width, alpha=line_alpha)
            if clabel:
                kw = dict(inline=True, fontsize=8, fmt="%.2f")
                if clabel_kwargs:
                    kw.update(clabel_kwargs)
                ax.clabel(cs, **kw)
            if colorbar:
                plt.colorbar(cf, ax=ax, label=z_lab)
            ax.set_xlabel(x_lab)
            ax.set_ylabel(y_lab)
            if m1_ticks is not None:
                ax.set_xticks(m1_num)
                ax.set_xticklabels(m1_ticks, rotation=45, ha="right")
            if m2_ticks is not None:
                ax.set_yticks(m2_num)
                ax.set_yticklabels(m2_ticks)
            if aspect == "square":
                ax.set_box_aspect(1)
            elif aspect == "equal":
                ax.set_aspect("equal")
            elif isinstance(aspect, (int, float)):
                ax.set_box_aspect(float(aspect))
            return ax

        # persp: 3D wireframe
        if ax is None:
            fig = plt.figure(figsize=figsize or (7, 6))
            ax = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(m1_num, m2_num, indexing="ij")
        Z = self.fit
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                        linewidth=0.3, edgecolor="black")
        if se_mult > 0 and self.se is not None:
            ax.plot_wireframe(X, Y, Z + se_mult * self.se,
                              color="red", linewidth=0.3, alpha=0.5)
            ax.plot_wireframe(X, Y, Z - se_mult * self.se,
                              color="green", linewidth=0.3, alpha=0.5)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_zlabel(z_lab)
        ax.view_init(elev=elev, azim=azim)
        if m1_ticks is not None:
            ax.set_xticks(m1_num)
            ax.set_xticklabels(m1_ticks)
        if m2_ticks is not None:
            ax.set_yticks(m2_num)
            ax.set_yticklabels(m2_ticks)
        return ax


def _axis_for_plot(m: np.ndarray):
    """Return (numeric_positions, tick_labels_or_None) for a vis axis.

    Factor axes get integer positions and string tick labels; numeric axes
    return themselves and ``None``."""
    if m.dtype.kind in ("U", "S", "O"):
        return np.arange(len(m), dtype=float), [str(v) for v in m]
    return np.asarray(m, dtype=float), None


def _expand_grid(d: dict[str, list]) -> pl.DataFrame:
    """Cartesian product of list-valued columns — R's :func:`expand.grid`.

    Column order matches insertion order of ``d``. R's ``expand.grid``
    iterates the *first* variable fastest; for our use case (one length-N
    column, the rest length 1), the iteration order doesn't matter, but we
    preserve the convention with ``meshgrid(indexing='ij')``.
    """
    if not d:
        return pl.DataFrame()
    keys = list(d.keys())
    arrays = [np.asarray(d[k]) if not isinstance(d[k], np.ndarray) else d[k]
              for k in keys]
    grids = np.meshgrid(*arrays, indexing="ij")
    cols = {k: g.ravel() for k, g in zip(keys, grids)}
    return pl.DataFrame(cols)


def _coerce_schema(grid: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    """Cast each grid column to the matching dtype in ``src`` (factor/string
    columns need to come back as strings, numeric stays numeric). Mirrors
    the schema-restoring loop in :meth:`gam.vis`.
    """
    out = grid
    for name in out.columns:
        if name in src.columns and src[name].dtype != out[name].dtype:
            out = out.with_columns(out[name].cast(src[name].dtype))
    return out


def _add_factor_stub_rows(grid: pl.DataFrame, src: pl.DataFrame):
    """Append one stub row per missing source-factor level so that
    :meth:`gam.predict` (under the hood, :func:`materialize`) sees every
    factor level the model was fit with.

    Rationale: without this, ``materialize``'s droplevels behavior
    (formula.py: line 1031–1037) collapses the contrast to only the levels
    present in the new data, returning a design matrix with fewer columns
    than ``self._beta``. mgcv side-steps this with ``model$xlevels``;
    we'll wire that into predict eventually, but this keeps
    ``get_difference`` correct in the interim.

    Returns ``(grid_with_stubs, n_stubs)``. Stubs are appended at the
    *end* — drop them via ``X[:-n_stubs]`` after predicting.
    """
    if grid.height == 0:
        return grid, 0
    stubs: list[dict] = []
    # Each stub copies the first row's other-column values so the
    # smooth bases evaluate at sensible points.
    template = {col: grid[col][0] for col in grid.columns}
    for name in grid.columns:
        if name not in src.columns:
            continue
        src_col = src[name]
        if not _is_factor_like_col(src_col):
            continue
        from ..formula import _factor_levels  # local to avoid cycle
        src_levels = list(_factor_levels(src_col))
        if len(src_levels) <= 1:
            continue
        present = set(grid[name].drop_nulls().to_list())
        for lv in src_levels:
            if lv not in present:
                row = dict(template)
                row[name] = lv
                stubs.append(row)
                # Track that this stub also adds the level — so
                # downstream factors don't double-stub for it.
                present.add(lv)
    if not stubs:
        return grid, 0
    stub_df = pl.DataFrame(stubs).select(grid.columns)
    for col in stub_df.columns:
        if stub_df[col].dtype != grid[col].dtype:
            stub_df = stub_df.with_columns(stub_df[col].cast(grid[col].dtype))
    return pl.concat([grid, stub_df], how="vertical_relaxed"), len(stubs)


def _resolve_sim_rng(rng):
    """Resolve :meth:`gam.get_difference`'s ``rng`` to an object exposing
    ``multivariate_normal(mean, cov, size)`` for the simultaneous-CI draws.

    ``None`` → the process-global R stream (``hea.R.set_seed``-controlled, as in
    R+itsadug where ``set.seed()`` precedes ``mgcv::rmvn``). An int seeds a
    fresh R ``RMersenneTwister``. An ``RMersenneTwister`` is wrapped in an
    ``RGenerator``; an ``RGenerator`` — or any object already exposing
    ``multivariate_normal`` (e.g. a ``numpy.random.Generator``, non-R) — is used
    as-is."""
    from ..R.rng import RGenerator, RMersenneTwister
    if rng is None:
        from ..R import distributions as _dist
        return RGenerator(_dist._r_rng())
    if isinstance(rng, (int, np.integer)):
        return RGenerator(int(rng))
    if isinstance(rng, RMersenneTwister):
        return RGenerator(rng)
    return rng


def _format_difference_summary(*, comp, cond, su, cancelled, rm_ranef,
                                sim_ci, f) -> str:
    """Itsadug-style ``print.summary`` text for :meth:`gam.get_difference`.

    Lists each variable and the value(s) used: first level vs second level
    for comp predictors, the cond array (or scalar) for cond predictors,
    typical value for the rest. Reports cancelled random-effect labels
    when ``rm_ranef`` is in effect. Not a parser — just an info dump.
    """
    lines = ["Summary:"]
    for k, v in comp.items():
        lines.append(f"\t* {k} : factor; set to the value(s): {v[0]}, {v[1]}.")
    for k, v in cond.items():
        if hasattr(v, "__len__") and not isinstance(v, str) and len(v) > 1:
            lo, hi = float(np.min(v)), float(np.max(v))
            lines.append(
                f"\t* {k} : numeric; range from {lo:.6g} to {hi:.6g} "
                f"(length {len(v)})."
            )
        else:
            lines.append(f"\t* {k} : set to {v}.")
    for k, v in su.items():
        if k in comp or k in cond:
            continue
        lines.append(f"\t* {k} : held at typical value {v}.")
    if rm_ranef not in (None, False):
        if cancelled:
            lines.append(
                "\tNOTE: The following random effects columns are canceled: "
                f"{', '.join(cancelled)}."
            )
        else:
            lines.append("\tNOTE: No random effects in the model to cancel.")
    if sim_ci:
        pct = 100.0 * (1.0 - round(2.0 * (1.0 - float(_nmath.pnorm5(f))), 2))
        lines.append(f"\tSimultaneous {pct:.0f}%-CI used.")
    return "\n".join(lines)


def _find_difference(mean: np.ndarray, se: np.ndarray,
                     x_vals: np.ndarray | None = None,
                     f: float = 1.0) -> dict | None:
    """Return contiguous regions where ``[mean − f·se, mean + f·se]`` excludes 0
    — direct port of itsadug's ``find_difference``.

    Returns ``None`` if no such region (matches R's ``NULL``). Otherwise
    a dict ``{"start": [...], "end": [...], "x_vals": bool}`` — element
    pairs ``(start[i], end[i])`` give the inclusive boundaries of one
    region. With ``x_vals`` provided and length-aligned, boundaries are
    in x-units; otherwise they are zero-based grid indices.

    Matches the R logic: find the indices where 0 is *not* in the band,
    split into runs by ``diff > 1``, take the first index of each run as
    the start and the last as the end.
    """
    if mean.shape != se.shape:
        raise ValueError("mean and se must have the same shape")
    ub = mean + f * se
    lb = mean - f * se
    sig = ~((ub >= 0) & (lb <= 0))
    n = np.where(sig)[0]
    if n.size == 0:
        return None
    diffs = np.diff(n)
    starts_idx = np.concatenate(([0], np.where(diffs > 1)[0] + 1))
    ends_idx = np.concatenate((np.where(diffs > 1)[0], [n.size - 1]))
    starts = n[starts_idx]
    ends = n[ends_idx]
    if x_vals is not None and len(x_vals) == len(mean):
        return {
            "start": np.asarray(x_vals)[starts].tolist(),
            "end":   np.asarray(x_vals)[ends].tolist(),
            "x_vals": True,
        }
    return {
        "start": starts.tolist(),
        "end":   ends.tolist(),
        "x_vals": False,
    }


class DiffResult:
    """Output of :meth:`gam.get_difference` — the numerical table behind a
    :meth:`gam.plot_diff` plot.

    Attributes
    ----------
    xvar : str | None
        Name of the x-axis covariate when this result came from
        :meth:`plot_diff`. ``None`` if the result was produced via
        :meth:`get_difference` directly (use ``grid`` to find the
        varying axis).
    grid : pl.DataFrame
        The condition grid, one row per evaluated point. Comp predictors
        are dropped (they're logged in ``levels``); cond predictors and
        held-at-typical-value predictors stay.
    difference : (n_grid,) ndarray
        Link-scale predicted difference ``(X1 − X2) β̂``. Length matches
        ``grid.height``.
    f : float | None
        SE multiplier used for the pointwise CI (``None`` if ``se=False``).
    ci : (n_grid,) ndarray | None
        Pointwise CI half-width: ``f · √diag((X1−X2) Vp (X1−X2)ᵀ)``.
    sim_ci : (n_grid,) ndarray | None
        Simultaneous CI half-width (Wood 2017 §6.10) when ``sim_ci=True``;
        else ``None``. Built from ``self.Vc`` (the unconditional cov).
    crit : float | None
        The simultaneous critical value (empirical quantile of the max
        absolute standardized deviation).
    comp_label : str
        Human-readable comparison label (e.g. ``"Group=('A', 'B')"``).
    levels : (str, str)
        ``(first, second)`` from the comp dict, joined across multi-key
        comp by '.'.
    rm_ranef_cancelled : list[str]
        Smooth labels whose columns were zeroed.
    """

    __slots__ = (
        "xvar", "grid", "difference", "f", "ci", "sim_ci", "crit",
        "comp_label", "levels", "rm_ranef_cancelled",
    )

    def __init__(self, *, xvar, grid, difference, f, ci, sim_ci, crit,
                 comp_label, levels, rm_ranef_cancelled):
        self.xvar = xvar
        self.grid = grid
        self.difference = difference
        self.f = f
        self.ci = ci
        self.sim_ci = sim_ci
        self.crit = crit
        self.comp_label = comp_label
        self.levels = levels
        self.rm_ranef_cancelled = rm_ranef_cancelled

    def __repr__(self) -> str:
        return (
            f"DiffResult(comp={self.comp_label}, n_grid={len(self.difference)}, "
            f"diff range=[{np.min(self.difference):.4g}, "
            f"{np.max(self.difference):.4g}], "
            f"sim_ci={'yes' if self.sim_ci is not None else 'no'})"
        )

    def regions(self, use_sim_ci: bool = False) -> list[tuple]:
        """Return a list of ``(start, end)`` x-windows where the CI
        excludes 0 — wraps :func:`_find_difference` and returns x-units
        when ``xvar`` is known, grid indices otherwise.
        """
        band = self.sim_ci if use_sim_ci else self.ci
        if band is None:
            return []
        x = None
        if self.xvar is not None and self.xvar in self.grid.columns:
            x = np.asarray(self.grid[self.xvar].to_numpy(), dtype=float)
        # f=1.0 because `band` already includes the f multiplier.
        out = _find_difference(self.difference, band, x_vals=x, f=1.0)
        if out is None:
            return []
        return list(zip(out["start"], out["end"]))
