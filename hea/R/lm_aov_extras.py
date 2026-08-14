"""R's ``lm`` / ``aov`` "extras" — the diagnostics-and-least-squares periphery
of ``stats`` that sits just outside the core ``lm`` / ``glm`` fit:

* **Model accessors** — ``sigma`` (``vcov.R``), ``cov2cor`` (``cor.R``),
  ``weighted.residuals`` (``lm.influence.R``).
* **Influence table** — ``covratio`` and ``influence.measures`` (the
  ``infmat`` / ``is.inf`` bundle, ``lm.influence.R``). These reuse the
  deletion primitives hea already caches (leverage, leave-one-out σ, the
  closed-form dfbeta) in :mod:`hea.R.diagnostics`.
* **Standalone least squares** — ``lsfit`` / ``ls.diag`` / ``ls.print``
  (``lsfit.R``), the pre-``lm`` QR interface. Built directly on
  :func:`hea.R.linalg.Cdqrls` (the same ``dqrls`` LINPACK kernel ``lm.fit``
  uses) and :func:`hea.R.linalg.dqrsl` (``qr.qy`` for the hat diagonal), so
  the fit inherits the documented ≤2-ulp lm-QR/FMA residual and the diagnostics
  match R's ``ls.diag`` exactly.
* **Design replication counts** — ``replications`` (``model.tables.R``),
  self-contained over hea's formula expander (no fitted model needed).

Faithful to base R 4.6.0; verified against live ``Rscript``.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from .diagnostics import influence
from .linalg import Cdqrls, dqrsl
from .model_generics import coef, deviance, nobs, residuals, weights

__all__ = [
    "Infl",
    "cov2cor",
    "covratio",
    "influence_measures",
    "ls_diag",
    "ls_print",
    "lsfit",
    "replications",
    "sigma",
    "weighted_residuals",
]


# --------------------------------------------------------------------------
# Model accessors: sigma / cov2cor / weighted.residuals
# --------------------------------------------------------------------------
def cov2cor(V):
    """R: ``cov2cor(V)`` — covariance matrix → correlation matrix.

    ``r[i,j] = V[i,j] / sqrt(V[i,i]·V[j,j])`` with the diagonal forced to
    exactly 1 (``cor.R``). Non-positive / NA variances give ``NaN`` rows and a
    warning, as in R. Accepts any square array-like; returns an ``ndarray``.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("'V' is not a square numeric matrix")
    p = V.shape[0]
    D = np.diag(V).copy()
    pos = (~np.isnan(D)) & (D > 0)
    Is = np.empty(p, dtype=float)
    Is[pos] = np.sqrt(1.0 / D[pos])
    Is[~pos] = np.nan
    if (not pos.all()) or (not np.all(np.isfinite(Is))):
        warnings.warn(
            "diag(V) had non-positive or NA entries; "
            "the non-finite result may be dubious",
            stacklevel=2,
        )
    r = Is[:, None] * V * Is[None, :]  # D %*% V %*% D, D = diag(Is)
    if p:
        np.fill_diagonal(r, 1.0)  # exact in diagonal
    return r


def _is_mlm(object) -> bool:
    """True for hea's multivariate lm (``cbind(...)`` response)."""
    return hasattr(object, "_mlm_models")


def sigma(object, use_fallback=True):
    """R: ``sigma(object)`` — the residual scale of an ``lm``/``glm``-like fit.

    * ``sigma.default`` — ``sqrt(deviance / (nobs − #non-NA coef))``
      (``vcov.R``); for an ``lm`` this is exactly ``summary(m)$sigma``.
    * ``sigma.glm`` — ``sqrt(dispersion)`` (the ``summary.glm`` dispersion:
      1 for binomial/poisson, the Pearson estimate otherwise).
    * ``sigma.mlm`` — per-response ``sqrt(colSums(resid²) / df.residual)``.
    """
    cls = object.__class__.__name__
    if _is_mlm(object):  # sigma.mlm
        R = np.asarray(object.residuals.to_numpy(), dtype=float)
        dfres = float(object.df_residual)
        return np.sqrt(np.sum(R * R, axis=0) / dfres)
    if cls in ("glm",):  # sigma.glm
        return float(np.sqrt(object.dispersion))
    cf = np.asarray(coef(object), dtype=float).reshape(-1)  # sigma.default
    k = int(np.sum(~np.isnan(cf)))
    return float(np.sqrt(deviance(object) / (nobs(object) - k)))


def _is_glm_like(object) -> bool:
    return object.__class__.__name__ in ("glm", "gam", "bam")


def weighted_residuals(obj, drop0=True):
    """R: ``weighted.residuals(obj, drop0=TRUE)`` (``lm.influence.R``).

    Working residuals scaled by ``sqrt`` of the working weights; for an ``lm``
    that is the response residual times ``sqrt(prior weight)`` (unchanged when
    unweighted). With ``drop0=True`` the zero-(prior-)weight rows are dropped.
    """
    if _is_glm_like(obj):
        w = getattr(obj, "w", None)  # IRLS working weights
        r = np.asarray(residuals(obj, type="working"), dtype=float)
        if w is not None:
            r = r * np.sqrt(np.asarray(w, dtype=float))
        w = getattr(obj, "_prior_w", None)  # prior weights
    else:  # lm
        w = weights(obj)
        r = np.asarray(residuals(obj), dtype=float)
        if w is not None:
            r = r * np.sqrt(np.asarray(w, dtype=float))
    if drop0 and w is not None:
        w = np.asarray(w, dtype=float)
        r = r[w != 0]
    return r


# --------------------------------------------------------------------------
# covratio + influence.measures
# --------------------------------------------------------------------------
def covratio(model, infl=None, res=None):
    """R: ``covratio(model)`` (``lm.influence.R``) — covariance ratio ``COVRATIO_i``.

    ``1 / ((1−h_i)·(((n−p−1)+e*²_i)/(n−p))^p)`` with the internally
    externally-studentized ``e*_i = r_i / (σ_(−i)·√(1−h_i))``.
    """
    if infl is None:
        infl = influence(model)
    h = np.asarray(infl["hat"], dtype=float)
    sigma_i = np.asarray(infl["sigma"], dtype=float)
    if res is None:
        res = weighted_residuals(model)
    res = np.asarray(res, dtype=float)
    n = int(nobs(model))
    p = int(model.rank if hasattr(model, "rank") else model.p)
    omh = 1.0 - h
    with np.errstate(divide="ignore", invalid="ignore"):
        e_star = res / (sigma_i * np.sqrt(omh))
    e_star = np.where(np.isinf(e_star), np.nan, e_star)
    return 1.0 / (omh * (((n - p - 1) + e_star**2) / (n - p)) ** p)


def _abbreviate(names, minlength=4):
    """base R ``abbreviate(names, minlength)`` (default ``use.classes=TRUE``).

    Removes, working from the right and never the first char of a word,
    lower-case vowels, then lower-case consonants, then other characters, until
    each string is ``≤ minlength``. Ports the ``stripchars`` letter-class order
    used for ``influence.measures`` column labels; only names longer than
    ``minlength`` are touched (short design-column names are returned verbatim).
    """

    def strip_one(s):
        if len(s) <= minlength:
            return s
        chars = list(s)
        # word-initial positions (start of string / after a space) are protected
        protected = {0}
        for i in range(1, len(chars)):
            if chars[i - 1] == " ":
                protected.add(i)

        def do_pass(pred):
            while len(chars) > minlength:
                target = None
                for i in range(len(chars) - 1, 0, -1):
                    if i in protected:
                        continue
                    if pred(chars[i]):
                        target = i
                        break
                if target is None:
                    return
                del chars[target]
                # recompute protected positions after deletion
                protected.clear()
                protected.add(0)
                for i in range(1, len(chars)):
                    if chars[i - 1] == " ":
                        protected.add(i)

        do_pass(lambda c: c in "aeiou")  # lower-case vowels
        do_pass(lambda c: c.islower())  # lower-case consonants
        do_pass(lambda c: not c.isspace())  # anything else but spaces
        return "".join(chars)

    return [strip_one(s) for s in names]


class Infl:
    """R's ``"infl"`` object — result of :func:`influence_measures`.

    * ``infmat`` — an ``n × (p+4)`` polars DataFrame: the standardized
      ``dfb.*`` columns, then ``dffit``, ``cov.r``, ``cook.d``, ``hat``.
    * ``is_inf`` — a boolean DataFrame of the same shape flagging each measure
      that crosses R's cut-off.
    * ``call`` — the model formula string (hea has no unevaluated call object).
    """

    def __init__(self, infmat: pl.DataFrame, is_inf: pl.DataFrame, call: str):
        self.infmat = infmat
        self.is_inf = is_inf
        self.call = call

    def __repr__(self) -> str:
        star = self.is_inf.to_numpy().any(axis=1)
        lines = [f"Influence measures of\n\t {self.call} :\n"]
        cols = self.infmat.columns
        header = "    " + " ".join(f"{c:>8s}" for c in cols) + "  inf"
        lines.append(header)
        M = self.infmat.to_numpy()
        for i in range(M.shape[0]):
            row = " ".join(f"{v:8.3g}" for v in M[i])
            lines.append(f"{i + 1:<3d} {row}  {'*' if star[i] else ' '}")
        return "\n".join(lines)


def influence_measures(model, infl=None):
    """R: ``influence.measures(model)`` (``lm.influence.R``).

    Assembles the standard deletion-diagnostic table for a fitted ``lm`` /
    ``glm``: standardized ``dfbetas`` (one column per coefficient), ``dffit``,
    ``cov.r`` (covariance ratio), ``cook.d`` (Cook's distance) and ``hat``
    (leverage), plus the matching ``is.inf`` flag matrix. Returns an
    :class:`Infl`.
    """
    if infl is None:
        infl = influence(model)
    p = int(model.rank if hasattr(model, "rank") else model.p)
    e = np.asarray(weighted_residuals(model), dtype=float)
    from .model_generics import df_residual

    s = float(np.sqrt(np.sum(e**2) / df_residual(model)))
    # (X'X)^-1 — hea's lm caches this as XtXinv (chol2inv of the fit's R factor)
    xxi = np.asarray(model.XtXinv, dtype=float)
    si = np.asarray(infl["sigma"], dtype=float)
    h = np.asarray(infl["hat"], dtype=float)
    cf = infl["coefficients"]  # unstandardized dfbeta
    cf = cf.to_numpy() if isinstance(cf, pl.DataFrame) else np.asarray(cf)

    sd = np.sqrt(np.diag(xxi))
    dfbetas = cf / (si[:, None] * sd[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        dffit = e * np.sqrt(h) / (si * (1.0 - h))
    dffit = np.where(np.isinf(dffit), np.nan, dffit)
    cov_r = (si / s) ** (2 * p) / (1.0 - h)
    with np.errstate(divide="ignore", invalid="ignore"):
        cook_d = ((e / (s * (1.0 - h))) ** 2 * h) / p

    vn = list(model.column_names)
    vn = ["1_" if v == "(Intercept)" else v for v in vn]
    dfb_names = [f"dfb.{a}" for a in _abbreviate(vn)]
    cols = {}
    for j, name in enumerate(dfb_names):
        cols[name] = dfbetas[:, j]
    cols["dffit"] = dffit
    cols["cov.r"] = cov_r
    cols["cook.d"] = cook_d
    cols["hat"] = h
    infmat = pl.DataFrame(cols)
    M = infmat.to_numpy()
    M = np.where(np.isinf(M), np.nan, M)
    infmat = pl.DataFrame({c: M[:, i] for i, c in enumerate(infmat.columns)})

    n_pos = int(np.sum(h > 0))
    k = p
    if n_pos <= k:
        raise ValueError("too few cases i with h_ii > 0), n < k")
    absM = np.abs(M)
    flags = np.empty_like(M, dtype=bool)
    flags[:, :k] = absM[:, :k] > 1  # |dfbetas| > 1
    flags[:, k] = absM[:, k] > 3 * np.sqrt(k / (n_pos - k))  # |dffit|
    flags[:, k + 1] = np.abs(1 - M[:, k + 1]) > (3 * k) / (n_pos - k)  # |1-cov.r|
    from .distributions import pf

    flags[:, k + 2] = np.asarray(pf(M[:, k + 2], k, n_pos - k)) > 0.5  # cook.d
    flags[:, k + 3] = M[:, k + 3] > (3 * k) / n_pos  # hat
    is_inf = pl.DataFrame({c: flags[:, i] for i, c in enumerate(infmat.columns)})
    call = getattr(model, "formula", model.__class__.__name__)
    return Infl(infmat, is_inf, call)


# --------------------------------------------------------------------------
# lsfit / ls.diag / ls.print  (standalone QR least squares)
# --------------------------------------------------------------------------
def _as_matrix(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def lsfit(x, y, wt=None, intercept=True, tolerance=1e-7, yname=None):
    """R: ``lsfit(x, y, wt, intercept, tolerance)`` (``lsfit.R``).

    Least-squares fit via the ``dqrls`` QR (same LINPACK kernel as ``lm.fit``).
    Returns a dict mirroring R's list: ``coefficients`` (named by the design
    columns, ``Intercept`` prepended when ``intercept=True``), ``residuals``,
    optional ``wt``, ``intercept``, and a ``qr`` sub-dict (``qt``/``qr``/
    ``qraux``/``rank``/``pivot``/``tol``) that :func:`ls_diag` / :func:`ls_print`
    consume. ``x`` may be a vector or matrix; a single-column ``x`` is named
    ``"X"`` (else ``X1``, ``X2``, …) when it has no column names.
    """
    x = _as_matrix(x)
    y = _as_matrix(y)
    ncx0 = x.shape[1]
    xnames = [f"X{i + 1}" for i in range(ncx0)] if ncx0 > 1 else ["X"]
    if intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])
        xnames = ["Intercept"] + xnames

    if yname is None and y.shape[1] > 1:
        yname = [f"Y{i + 1}" for i in range(y.shape[1])]

    # complete.cases over x, y, wt
    good = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    if wt is not None:
        wt = np.asarray(wt, dtype=float)
        good = good & np.isfinite(wt)
    dimy = y.shape
    if not good.all():
        ndel = int(np.sum(~good))
        warnings.warn(
            f"{ndel} missing value{'s' if ndel != 1 else ''} deleted", stacklevel=2
        )
        x = x[good]
        y = y[good]
        if wt is not None:
            wt = wt[good]

    nrx, ncx = x.shape
    nry, ncy = y.shape
    if nry != nrx:
        raise ValueError(
            f"'X' matrix has {nrx} cases (rows), 'Y' has {nry} cases (rows)"
        )
    if nry < ncx:
        raise ValueError(f"only {nry} cases, but {ncx} variables")

    invmult = None
    xzero = yzero = None
    if wt is not None:
        if np.any(wt < 0):
            raise ValueError("negative weights not allowed")
        wtmult = np.sqrt(wt)
        if np.any(wt == 0):
            xzero = x[wt == 0]
            yzero = y[wt == 0]
        x = x * wtmult[:, None]
        y = y * wtmult[:, None]
        invmult = 1.0 / np.where(wt == 0, 1.0, wtmult)

    # Cdqrls handles a matrix y column-by-column (R passes y as a matrix).
    coefs = np.zeros((ncx, ncy))
    resid_w = np.zeros((nry, ncy))
    z = None
    for j in range(ncy):
        z = Cdqrls(x, y[:, j], tolerance)
        coefs[:, j] = z["coefficients"]
        resid_w[:, j] = z["residuals"]
    rank = z["rank"]
    pivot = z["pivot"]
    qraux = z["qraux"]
    qr_factor = np.asarray(z["qr"]).reshape(nrx, ncx)
    effects = z["effects"]

    resids = np.full(dimy, np.nan)
    if wt is not None:
        if np.any(wt == 0):
            fitted_zeros = xzero @ coefs
            resid_w[wt == 0, :] = yzero - fitted_zeros
        resid_w = resid_w * invmult[:, None]
    resids[good, :] = resid_w

    # unpivot coefficient order back to xnames order (R keeps xnames order)
    coef_names = list(xnames)
    if dimy[1] == 1 and yname is None:
        resids = resids.reshape(-1)
        coef_out = _NamedVec(coef_names, coefs[:, 0])
    else:
        coef_out = coefs

    if rank != ncx:
        warnings.warn("'X' matrix was collinear", stacklevel=2)

    qr_obj = {
        "qt": effects.reshape(-1) if ncy == 1 else effects,
        "qr": qr_factor,
        "qraux": qraux,
        "rank": rank,
        "pivot": pivot,
        "tol": tolerance,
        "qr_colnames": coef_names,
    }
    out = {
        "coefficients": coef_out,
        "residuals": resids,
        "intercept": intercept,
        "qr": qr_obj,
    }
    if wt is not None:
        wfull = np.full(dimy[0], np.nan)
        wfull[good] = wt
        out["wt"] = wfull
    return out


class _NamedVec(dict):
    """Minimal ordered name→value vector (``lsfit`` coefficients)."""

    def __init__(self, names, values):
        super().__init__(zip(names, [float(v) for v in values]))
        self._names = list(names)
        self._values = np.asarray(values, dtype=float)

    @property
    def values(self):
        return self._values

    def to_numpy(self):
        return self._values


def _qr_qy_matrix(qr_obj, ymat: np.ndarray) -> np.ndarray:
    """``qr.qy(qr_obj, ymat)`` — apply Q (full, n×n) to each column of ymat."""
    qr = np.asarray(qr_obj["qr"], dtype=float)
    n = qr.shape[0]
    k = int(qr_obj["rank"])
    qraux = np.asarray(qr_obj["qraux"], dtype=float)
    ymat = np.asarray(ymat, dtype=float)
    if ymat.ndim == 1:
        ymat = ymat.reshape(-1, 1)
    out = np.empty((n, ymat.shape[1]))
    for j in range(ymat.shape[1]):
        qy, *_ = dqrsl(qr.copy(), n, k, qraux, ymat[:, j], 10000)
        out[:, j] = qy
    return out


def ls_diag(ls_out):
    """R: ``ls.diag(ls.out)`` (``lsfit.R``) — diagnostics from an :func:`lsfit`.

    Returns a dict: ``std.dev`` (residual σ), ``hat`` (leverage), ``std.res``
    (internally studentized), ``stud.res`` (externally studentized), ``cooks``,
    ``dfits``, ``correlation`` / ``cov.scaled`` / ``cov.unscaled`` (coefficient
    covariance) and ``std.err``. Single-response fits return vectors (R's
    ``as.vector`` collapse); the hat diagonal comes from ``qr.qy``.
    """
    resids = _as_matrix(ls_out["residuals"])
    d0 = resids.shape
    qr_obj = ls_out["qr"]

    good = np.all(np.isfinite(resids), axis=1)
    wt = ls_out.get("wt")
    if wt is not None:
        good = good & np.isfinite(np.asarray(wt, dtype=float))
    if not good.all():
        warnings.warn("missing observations deleted", stacklevel=2)
        resids = resids[good]
    if wt is not None:
        wt = np.asarray(wt, dtype=float)
        if np.any(wt[good] == 0):
            warnings.warn(
                "observations with 0 weight not used in calculating standard deviation",
                stacklevel=2,
            )
        resids = resids * np.sqrt(wt[good])[:, None]

    p = int(qr_obj["rank"])
    n = resids.shape[0]
    hatdiag = np.full(d0[0], np.nan)
    ncy = resids.shape[1]

    # hat diagonals: q = qr.qy(qr, rbind(diag(p), 0)); rowSums(q^2)
    e_basis = np.zeros((qr_obj["qr"].shape[0], p))
    e_basis[:p, :] = np.eye(p)
    q = _qr_qy_matrix(qr_obj, e_basis)
    hatdiag[good] = np.sum(q**2, axis=1)

    stddev = np.sqrt(np.sum(resids**2, axis=0) / (n - p))
    stddevmat = np.broadcast_to(stddev, (int(np.sum(good)), ncy))
    hg = hatdiag[good]
    stdres = np.full(d0, np.nan)
    studres = np.full(d0, np.nan)
    dfits = np.full(d0, np.nan)
    cooks = np.full(d0, np.nan)
    sr = resids / (np.sqrt(1 - hg)[:, None] * stddevmat)
    stdres[good] = sr
    studres[good] = (sr * stddevmat) / np.sqrt(
        ((n - p) * stddevmat**2 - resids**2 / (1 - hg)[:, None]) / (n - p - 1)
    )
    dfits[good] = np.sqrt(hg / (1 - hg))[:, None] * studres[good]
    cooks[good] = ((sr**2 * hg[:, None]) / p) / (1 - hg)[:, None]

    # unscaled coefficient covariance: tcrossprod(solve(R))
    R = np.asarray(qr_obj["qr"], dtype=float)[:p, :p].copy()
    R[np.tril_indices(p, -1)] = 0.0
    Rinv = np.linalg.solve(R, np.eye(p))
    covmat_unscaled = Rinv @ Rinv.T
    covmat_scaled = float(np.sum(stddev**2)) * covmat_unscaled
    dg = np.diag(covmat_scaled)
    cormat = covmat_scaled / np.sqrt(np.outer(dg, dg))
    stderr = np.outer(np.sqrt(np.diag(covmat_unscaled)), stddev)

    if ncy == 1:
        stdres = stdres.reshape(-1)
        cooks = cooks.reshape(-1)
        studres = studres.reshape(-1)
        dfits = dfits.reshape(-1)
        stddev = float(stddev[0])
    return {
        "std.dev": stddev,
        "hat": hatdiag,
        "std.res": stdres,
        "stud.res": studres,
        "cooks": cooks,
        "dfits": dfits,
        "correlation": cormat,
        "std.err": stderr,
        "cov.scaled": covmat_scaled,
        "cov.unscaled": covmat_unscaled,
    }


def ls_print(ls_out, digits=4, print_it=True):
    """R: ``ls.print(ls.out, digits, print.it)`` (``lsfit.R``).

    Regression-summary tables for an :func:`lsfit`. Returns a dict with
    ``summary`` (RSE / R² / F / dfs / p) and ``coef.table`` (one entry per
    response: Estimate / Std.Err / t-value / Pr(>|t|)). With ``print_it=True``
    the same is echoed to stdout, matching R's layout.
    """
    from .distributions import pf, pt

    resids = _as_matrix(ls_out["residuals"])
    wt = ls_out.get("wt")
    if wt is not None:
        wt = np.asarray(wt, dtype=float)
        if np.any(wt == 0):
            warnings.warn("observations with 0 weights not used", stacklevel=2)
        resids = resids * np.sqrt(wt)[:, None]
    n = resids.shape[0] - np.sum(np.isnan(resids), axis=0)
    qr_obj = ls_out["qr"]
    p = int(qr_obj["rank"])
    qt = _as_matrix(qr_obj["qt"])

    if ls_out["intercept"]:
        totss = np.sum(qt[1:] ** 2, axis=0)
        degfree = p - 1
    else:
        totss = np.sum(qt**2, axis=0)
        degfree = p

    resss = np.nansum(resids**2, axis=0)
    resse = np.sqrt(resss / (n - p))
    regss = totss - resss
    rsquared = regss / totss
    fstat = (regss / degfree) / (resss / (n - p))
    pvalue = np.asarray(pf(fstat, degfree, n - p, lower_tail=False))

    R = np.asarray(qr_obj["qr"], dtype=float)[:p, :p].copy()
    R[np.tril_indices(p, -1)] = 0.0
    Rinv = np.linalg.solve(R, np.eye(p))
    uVar = np.diag(Rinv @ Rinv.T)

    coef_in = ls_out["coefficients"]
    coef_arr = (
        coef_in.to_numpy().reshape(-1, 1)
        if isinstance(coef_in, _NamedVec)
        else np.asarray(coef_in)
    )
    m_y = resids.shape[1]
    xnames = qr_obj["qr_colnames"]
    ynames = ls_out.get("_ynames")
    coef_table = {}
    for i in range(m_y):
        se = np.sqrt((resss[i] / (n[i] - p)) * uVar)
        est = coef_arr[:, i]
        tval = est / se
        pv = 2 * np.asarray(pt(np.abs(tval), n[i] - p, lower_tail=False))
        tbl = np.column_stack([est, se, tval, pv])
        key = ynames[i] if ynames is not None else i + 1
        coef_table[key] = {
            "rows": xnames,
            "cols": ["Estimate", "Std.Err", "t-value", "Pr(>|t|)"],
            "values": tbl,
        }
        if print_it:
            if m_y > 1:
                print(f"Response: {key}\n")
            print(
                f"Residual Standard Error={round(float(resse[i]), digits)}, "
                f"R-Square={round(float(rsquared[i]), digits)}"
            )
            print(
                f"F-statistic (df={degfree}, {n[i] - p})="
                f"{round(float(fstat[i]), digits)}, "
                f"p-value={round(float(pvalue[i]), digits)}\n"
            )
            hdr = " ".join(f"{c:>10s}" for c in coef_table[key]["cols"])
            print(f"{'':>12s}{hdr}")
            for rname, rowv in zip(xnames, tbl):
                cells = " ".join(f"{round(float(v), digits):>10g}" for v in rowv)
                print(f"{rname:>12s}{cells}")
            print()
    return {
        "summary": {
            "rows": ynames if ynames is not None else ["Y"],
            "resse": resse,
            "rsquared": rsquared,
            "fstat": fstat,
            "df1": degfree,
            "df2": n - p,
            "pvalue": pvalue,
        },
        "coef.table": coef_table,
    }


# --------------------------------------------------------------------------
# replications  (design-balance counts; no fitted model needed)
# --------------------------------------------------------------------------
def replications(formula, data):
    """R: ``replications(formula, data)`` (``model.tables.R``).

    Number of replications of each term in an experimental-design formula. For
    a balanced design returns a name→count dict (R's ``unlist``); if any term
    is unbalanced the whole result is a dict whose unbalanced entries are the
    full cross-tabulation of counts (R returns a list). Non-factor terms are
    ignored with a warning, as in R.
    """
    from ..formula import Name, expand, parse

    f = parse(formula) if isinstance(formula, str) else formula
    ef = expand(f, data_columns=list(data.columns))
    terms = [t for t in ef.terms if t.atoms]  # drop intercept
    labels = [t.label for t in terms]

    # which columns are factors (Enum/categorical/string ⇒ factor)
    def is_factor(colname):
        if colname not in data.columns:
            return False
        dt = data.schema[colname]
        return dt in (pl.Categorical, pl.String, pl.Utf8) or isinstance(dt, pl.Enum)

    # R unlists to a named vector when every term is balanced, else returns a
    # list mixing scalars (balanced terms) and count tables (unbalanced). hea
    # encodes both in one dict: an int per balanced term, a count dict per
    # unbalanced one.
    z = {}
    for term, label in zip(terms, labels):
        if label.startswith("Error"):
            continue
        # variables in this term (main-effect names), in appearance order
        select = []
        for a in term.atoms:
            if isinstance(a, Name):
                if a.ident not in select:
                    select.append(a.ident)
            else:
                # non-Name atom (e.g. a transform) — treat as non-factor
                select = None
                break
        if select is None:
            continue
        notfac = [v for v in select if not is_factor(v)]
        if notfac:
            warnings.warn("non-factors ignored: " + ", ".join(notfac), stacklevel=2)
            continue
        if select:
            tble = data.group_by(select).agg(pl.len().alias("__n__"))
            counts = tble["__n__"].to_numpy()
            # R turns character columns into factors (levels sorted); enumerate
            # the full grid of level combinations so missing cells read as 0.
            level_sets = [sorted(data[v].unique().to_list()) for v in select]
            ncell = int(np.prod([len(s) for s in level_sets]))
            nrep = np.unique(counts)
            if len(nrep) > 1 or len(counts) < ncell:
                grid = {tuple(row[:-1]): row[-1] for row in tble.iter_rows()}
                z[label] = _fill_grid(level_sets, [], grid, single=len(select) == 1)
            else:
                z[label] = int(nrep[0])
        else:
            z[label] = 0
    return z


def _fill_grid(level_sets, prefix, grid, *, single):
    """Enumerate the full factor-combination grid (counts, 0-filled).

    Single-factor terms are keyed by the level; interactions by the tuple of
    levels — both in R's sorted factor-level order.
    """
    out = {}

    def rec(remaining, pre):
        if not remaining:
            key = pre[0] if single else tuple(pre)
            out[key] = int(grid.get(tuple(pre), 0))
            return
        for lv in remaining[0]:
            rec(remaining[1:], pre + [lv])

    rec(level_sets, list(prefix))
    return out
