"""Local polynomial regression — port of R's ``stats::loess``.

Cleveland's LOWESS (1979) and LOESS (1988): fit a low-degree polynomial
locally around each query point, weighted by the tricube of distance.
``family="symmetric"`` adds Tukey-biweight robustness iterations.

Scope of this port (private, called only by ``hea.ggplot.stat_smooth`` for
now):

* univariate predictor (``x``);
* ``span``, ``degree`` (1 or 2), ``family`` ("gaussian" or "symmetric"),
  ``iterations`` for the robustness M-step;
* direct neighbour lookup at each query point — no kd-tree interpolation
  surface (R's ``loess(surface="interpolate")``). For n < ~1000 this is
  fast; geom_smooth defaults to GAM above that anyway.
* standard errors at any prediction grid via the WLS variance formula.

Reference: Cleveland WS, Devlin SJ. *Locally Weighted Regression: An
Approach to Regression Analysis by Local Fitting*. JASA 1988.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    sigma: float            # residual standard deviation
    df_residual: float       # n - trace(S)
    robust_weights: np.ndarray  # final M-step weights (1s for gaussian)

    def predict(self, newx, *, se: bool = False):
        """Evaluate the fitted smoother at ``newx``.

        With ``se=False`` returns the fitted vector. With ``se=True``
        returns ``(fit, se)`` where ``se`` is the pointwise standard error.
        """
        newx = np.asarray(newx, dtype=float)
        n_new = len(newx)
        fit = np.empty(n_new)
        if se:
            se_arr = np.empty(n_new)

        for i, xi in enumerate(newx):
            beta, var00 = _local_fit(
                xi, self.x, self.y, self.span, self.degree, self.robust_weights,
                want_var=se,
            )
            fit[i] = beta[0]
            if se:
                se_arr[i] = self.sigma * np.sqrt(max(var00, 0.0))

        return (fit, se_arr) if se else fit


def loess(x, y, *, span: float = 0.75, degree: int = 2,
          family: str = "gaussian", iterations: int = 4) -> LoessFit:
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
        fitted = np.empty(n)
        leverages = np.empty(n)
        for i in range(n):
            beta, var00 = _local_fit(
                x[i], x, y, span, degree, robust_w, want_var=True,
            )
            fitted[i] = beta[0]
            leverages[i] = var00  # this is (X^T W X)^-1[0,0]; sum across i is roughly tr(S)

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
    # (the ``enp`` field). Refine when GG-C2 oracle parity matters.
    eff_p = max(degree + 1, np.sum(leverages))
    df_residual = max(n - eff_p, 1.0)
    sigma = float(np.sqrt(np.sum(residuals**2 * robust_w) / df_residual))

    return LoessFit(
        x=x, y=y,
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


def _local_fit(x_query: float, x: np.ndarray, y: np.ndarray, span: float,
               degree: int, extra_w: np.ndarray, *, want_var: bool):
    """Fit a local polynomial of given degree at ``x_query`` using
    ``span`` fraction of nearest neighbours of ``x``. ``extra_w`` is the
    M-step robustness vector (or all-1 for gaussian). Returns
    ``(beta, var00)`` — coefficients of the centred polynomial and the
    [0, 0] entry of ``(X^T W X)^-1`` (variance of β_0)."""
    n = len(x)
    k = max(int(np.ceil(span * n)), degree + 1)

    d = np.abs(x - x_query)
    if k >= n:
        # Use everything; bandwidth is the max distance, possibly stretched
        # past 1 in tricube — R does this in the "global" regime.
        h = d.max()
    else:
        # k-th smallest distance bounds the bandwidth.
        h = np.partition(d, k - 1)[k - 1]

    if h <= 0:
        # All training x's lie at the query point; collapse to weighted mean.
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
        # Not enough non-zero weights to fit the polynomial; return NaN.
        return np.full(degree + 1, np.nan), np.nan

    xc = x - x_query
    X = np.column_stack([xc**p for p in range(degree + 1)])
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    if want_var:
        # (X^T W X)^-1[0, 0] = inv(Xw^T Xw)[0, 0]
        XtWX = Xw.T @ Xw
        try:
            inv = np.linalg.inv(XtWX)
            var00 = float(inv[0, 0])
        except np.linalg.LinAlgError:
            var00 = np.nan
        return beta, var00
    return beta, np.nan
