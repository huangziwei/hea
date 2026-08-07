"""``stat_smooth()`` — fit a smoother and evaluate on a grid for plotting.

Dispatches on ``method``:

* ``"loess"`` (default for n < 1000) — :func:`loess`, defined below;
* ``"lm"`` — ordinary linear model via :class:`hea.lm`;
* ``"glm"`` / ``"gam"`` — defer to :mod:`hea.glm` / :mod:`hea.gam`.

Output columns are ``x``, ``y`` (= fit), ``ymin``, ``ymax`` (CI bounds), and
``se``. ``geom_smooth`` then layers a ribbon (ymin..ymax) plus a path (y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..._dispatch import rs_fn
from ...R import nmath as _nmath
from .._util import to_numeric_aes
from .stat import Stat

# Rust+rayon LOESS kernel (None when the extension is absent or HEA_NO_RS=1);
# the fit and predict share it so they agree bit-for-bit at common points.
_rs_loess = rs_fn("loess_eval")


# ---------------------------------------------------------------------------
# Local polynomial regression — port of R's ``stats::loess``.
#
# Cleveland's LOWESS (1979) and LOESS (1988): fit a low-degree polynomial
# locally around each query point, weighted by the tricube of distance.
# ``family="symmetric"`` adds Tukey-biweight robustness iterations.
#
# Scope (used only by :class:`StatSmooth` for now):
#   * univariate predictor (``x``);
#   * ``span``, ``degree`` (1 or 2), ``family`` ("gaussian" or "symmetric"),
#     ``iterations`` for the robustness M-step;
#   * direct neighbour lookup at each query point — no kd-tree interpolation
#     surface (R's ``loess.control(surface="interpolate")``, which is its
#     default). For n < ~1000 this is fast; geom_smooth defaults to GAM
#     above that anyway.
#   * standard errors at any prediction grid via the WLS variance formula.
#
# Reference: Cleveland WS, Devlin SJ. "Locally Weighted Regression: An
# Approach to Regression Analysis by Local Fitting", JASA 1988.
# ---------------------------------------------------------------------------


@dataclass
class LoessFit:
    """Result of fitting a loess smoother. Holds enough state to predict at
    new x values via :meth:`predict`."""

    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    span: float
    degree: int
    family: str
    iterations: int
    sigma: float  # residual standard deviation
    df_residual: float  # n - trace(S)
    robust_weights: np.ndarray  # final M-step weights (1s for gaussian)

    def predict(self, newx, *, se: bool = False):
        """Evaluate the fitted smoother at ``newx``.

        With ``se=False`` returns the fitted vector. With ``se=True``
        returns ``(fit, se)`` where ``se`` is the pointwise standard error.
        """
        newx = np.asarray(newx, dtype=float)
        fit, var = _loess_eval(
            newx,
            self.x,
            self.y,
            self.span,
            self.degree,
            self.robust_weights,
            want_var=se,
        )
        se_arr = self.sigma * np.sqrt(np.maximum(var, 0.0)) if se else None

        return (fit, se_arr) if se else fit


def loess(
    x,
    y,
    *,
    span: float = 0.75,
    degree: int = 2,
    family: str = "gaussian",
    iterations: int = 4,
) -> LoessFit:
    """Fit a loess smoother. See :class:`LoessFit` for what comes back."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if len(x) != len(y):
        raise ValueError(f"length mismatch: x={len(x)}, y={len(y)}")
    if not 0 < span <= 1:
        raise ValueError(f"span must be in (0, 1], got {span}")
    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2, got {degree}")
    if family not in ("gaussian", "symmetric"):
        raise ValueError(f"family must be 'gaussian' or 'symmetric', got {family!r}")

    n = len(x)
    robust_w = np.ones(n)
    n_robust_iter = iterations if family == "symmetric" else 0

    for _ in range(n_robust_iter + 1):
        fitted, leverages = _loess_eval(x, x, y, span, degree, robust_w, want_var=True)

        residuals = y - fitted

        if family != "symmetric":
            break

        # Tukey biweight M-step: down-weight large residuals.
        s = np.median(np.abs(residuals))
        if s == 0:
            break
        u = residuals / (6.0 * s)
        robust_w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)

    # Approximate trace(S) by summing per-point leverages times the local
    # weight at the query point — an underestimate but in the right order
    # of magnitude. R's loess computes this via a more involved formula
    # (the ``enp`` field). Refine when oracle parity matters.
    eff_p = max(degree + 1, np.sum(leverages))
    df_residual = max(n - eff_p, 1.0)
    sigma = float(np.sqrt(np.sum(residuals**2 * robust_w) / df_residual))

    return LoessFit(
        x=x,
        y=y,
        fitted=fitted,
        residuals=residuals,
        span=span,
        degree=degree,
        family=family,
        iterations=n_robust_iter,
        sigma=sigma,
        df_residual=df_residual,
        robust_weights=robust_w,
    )


def _loess_local_fit(
    x_query: float,
    x: np.ndarray,
    y: np.ndarray,
    span: float,
    degree: int,
    extra_w: np.ndarray,
    *,
    want_var: bool,
):
    """Fit a local polynomial of given degree at ``x_query`` using
    ``span`` fraction of nearest neighbours of ``x``. ``extra_w`` is the
    M-step robustness vector (or all-1 for gaussian). Returns
    ``(beta, var00)`` — coefficients of the centred polynomial and the
    [0, 0] entry of ``(X^T W X)^-1`` (variance of β_0)."""
    n = len(x)
    k = max(int(np.ceil(span * n)), degree + 1)

    d = np.abs(x - x_query)
    if k >= n:
        # Use everything; bandwidth is the max distance. When span > 1
        # R loess (loessc.c) stretches the bandwidth proportionally; we
        # use the simple max-distance bound here.
        h = d.max()
    else:
        h = np.partition(d, k - 1)[k - 1]

    if h <= 0:
        w = (d == 0).astype(float) * extra_w
        if w.sum() == 0:
            return np.zeros(degree + 1), 1e300
        beta = np.zeros(degree + 1)
        beta[0] = np.average(y, weights=w)
        return beta, np.inf

    u = d / h
    tricube = np.where(u < 1, (1 - u**3) ** 3, 0.0)
    w = tricube * extra_w

    if (w > 0).sum() < degree + 1:
        return np.full(degree + 1, np.nan), np.nan

    xc = x - x_query
    X = np.column_stack([xc**p for p in range(degree + 1)])
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    if want_var:
        XtWX = Xw.T @ Xw
        try:
            inv = np.linalg.inv(XtWX)
            var00 = float(inv[0, 0])
        except np.linalg.LinAlgError:
            var00 = np.nan
        return beta, var00
    return beta, np.nan


def _loess_eval(xq, x, y, span, degree, extra_w, want_var):
    """Per-query local fits at ``xq`` against data ``(x, y)``.

    Rust+rayon when the extension is present (the fit and predict both call this,
    so they agree bit-for-bit at shared query points), else the pure-Python loop
    over :func:`_loess_local_fit` (unchanged behaviour under ``HEA_NO_RS=1``).
    Returns ``(fitted, var00)`` arrays aligned with ``xq``.
    """
    xq = np.asarray(xq, dtype=float)
    if _rs_loess is not None:
        fitted, var = _rs_loess(
            np.ascontiguousarray(xq),
            np.ascontiguousarray(np.asarray(x, dtype=float)),
            np.ascontiguousarray(np.asarray(y, dtype=float)),
            float(span),
            int(degree),
            np.ascontiguousarray(np.asarray(extra_w, dtype=float)),
            bool(want_var),
        )
        return np.asarray(fitted), np.asarray(var)
    m = len(xq)
    fitted = np.empty(m)
    var = np.empty(m)
    for i in range(m):
        beta, var00 = _loess_local_fit(
            xq[i], x, y, span, degree, extra_w, want_var=want_var
        )
        fitted[i] = beta[0]
        var[i] = var00
    return fitted, var


@dataclass
class StatSmooth(Stat):
    method: str = "loess"  # "loess" | "lm" | "glm" | "gam"
    formula: str | None = None  # for lm/glm/gam; defaults to "y ~ x"
    se: bool = True
    level: float = 0.95
    span: float = 0.75
    n: int = 80  # grid size
    family: object = None  # for glm — Family instance or name; default Gaussian

    def compute_group(self, data, params):
        # Factor / string columns flow through as integer codes (1..N),
        # matching ggplot2's ``mapped_discrete`` so binary-factor responses
        # like ``y="use"`` work with no upstream coercion.
        x = to_numeric_aes(data["x"])
        y = to_numeric_aes(data["y"])

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return pl.DataFrame(
                {
                    "x": [],
                    "y": [],
                    "ymin": [],
                    "ymax": [],
                    "se": [],
                }
            )

        x_min, x_max = float(x.min()), float(x.max())
        grid = np.linspace(x_min, x_max, self.n)

        yhat, se = _fit_predict(
            self.method,
            self.formula,
            x,
            y,
            grid,
            span=self.span,
            family=self.family,
        )

        z = _nmath.qnorm5(0.5 + self.level / 2)
        ymin = yhat - z * se
        ymax = yhat + z * se

        return pl.DataFrame(
            {
                "x": grid,
                "y": yhat,
                "ymin": ymin,
                "ymax": ymax,
                "se": se,
            }
        )


def _fit_predict(
    method: str,
    formula: str | None,
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    *,
    span: float,
    family=None,
):
    """Returns ``(yhat, se)`` evaluated on ``grid`` for the chosen method."""
    if method == "loess":
        fit = loess(x, y, span=span)
        yhat, se = fit.predict(grid, se=True)
        return yhat, se

    if method == "lm":
        from ...models.lm import lm

        fml = formula or "y ~ x"
        fit = lm(fml, pl.DataFrame({"x": x, "y": y}))
        new = pl.DataFrame({"x": grid})
        # `predict(interval="confidence")` returns columns ("fit", "lwr",
        # "upr"). Back out the standard error from the CI half-width.
        pred = fit.predict(newdata=new, interval="confidence", alpha=0.05)
        yhat = pred["fit"].to_numpy()
        ci_lo = pred["lwr"].to_numpy()
        from scipy.stats import t

        df = float(fit.df_residuals)
        t_crit = t.ppf(0.975, df)
        se = (yhat - ci_lo) / t_crit
        return yhat, se

    if method == "gam":
        from ...models.gam import gam

        fml = formula or "y ~ s(x)"
        fit = gam(fml, pl.DataFrame({"x": x, "y": y}))
        yhat, se = fit.predict(
            newdata=pl.DataFrame({"x": grid}),
            type="response",
            se_fit=True,
        )
        return np.asarray(yhat), np.asarray(se)

    if method == "glm":
        from ...models.glm import glm

        fml = formula or "y ~ x"
        fam = _resolve_family(family)
        fit = glm(fml, pl.DataFrame({"x": x, "y": y}), family=fam)
        yhat, se = fit.predict(
            new=pl.DataFrame({"x": grid}),
            type="response",
            se_fit=True,
        )
        return np.asarray(yhat), np.asarray(se)

    raise ValueError(f"stat_smooth: unknown method {method!r}")


def _resolve_family(family):
    """Coerce ``family`` to a :class:`hea.family.Family` instance.

    Accepts None (defaults to Gaussian — equivalent to OLS, matching
    R's ``glm()`` default), a Family instance, or a name string in
    R's lowercase convention (``"gaussian"``, ``"binomial"``, …).
    """
    from ...family import Family, gaussian

    if family is None:
        return gaussian()
    if isinstance(family, Family):
        return family
    if callable(family):
        # User passed `binomial` (the class) instead of `binomial()` —
        # instantiate with default link.
        result = family()
        if isinstance(result, Family):
            return result
    if isinstance(family, str):
        import hea.family as _f

        name = family.lower().replace(".", "_")
        cls = getattr(_f, name, None)
        if cls is None:
            raise ValueError(
                f"unknown family {family!r}; expected one of gaussian, "
                "binomial, poisson, Gamma, inverse_gaussian, quasi"
            )
        return cls()
    raise TypeError(
        f"family must be None, a Family instance, or a string; "
        f"got {type(family).__name__}"
    )


def stat_smooth(
    *, method="loess", formula=None, se=True, level=0.95, span=0.75, n=80, family=None
):
    return StatSmooth(
        method=method,
        formula=formula,
        se=se,
        level=level,
        span=span,
        n=n,
        family=family,
    )
