"""``stat_smooth()`` — fit a smoother and evaluate on a grid for plotting.

Dispatches on ``method``:

* ``"loess"`` (default for n < 1000) — :func:`hea._loess.loess`;
* ``"lm"`` — ordinary linear model via :class:`hea.lm`;
* ``"glm"`` / ``"gam"`` — deferred to a later phase.

Output columns are ``x``, ``y`` (= fit), ``ymin``, ``ymax`` (CI bounds), and
``se``. ``geom_smooth`` then layers a ribbon (ymin..ymax) plus a path (y).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import norm

from .stat import Stat


@dataclass
class StatSmooth(Stat):
    method: str = "loess"  # "loess" | "lm" | "glm" | "gam"
    formula: str | None = None  # for lm/glm/gam; defaults to "y ~ x"
    se: bool = True
    level: float = 0.95
    span: float = 0.75
    n: int = 80  # grid size

    def compute_group(self, data, params):
        x = data["x"].to_numpy().astype(float)
        y = data["y"].to_numpy().astype(float)

        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return pl.DataFrame({
                "x": [], "y": [], "ymin": [], "ymax": [], "se": [],
            })

        x_min, x_max = float(x.min()), float(x.max())
        grid = np.linspace(x_min, x_max, self.n)

        yhat, se = _fit_predict(self.method, self.formula, x, y, grid, self.span)

        z = norm.ppf(0.5 + self.level / 2)
        ymin = yhat - z * se
        ymax = yhat + z * se

        return pl.DataFrame({
            "x": grid,
            "y": yhat,
            "ymin": ymin,
            "ymax": ymax,
            "se": se,
        })


def _fit_predict(method: str, formula: str | None,
                 x: np.ndarray, y: np.ndarray, grid: np.ndarray, span: float):
    """Returns ``(yhat, se)`` evaluated on ``grid`` for the chosen method."""
    if method == "loess":
        from ..._loess import loess
        fit = loess(x, y, span=span)
        yhat, se = fit.predict(grid, se=True)
        return yhat, se

    if method == "lm":
        from ...lm import lm

        fml = formula or "y ~ x"
        fit = lm(fml, pl.DataFrame({"x": x, "y": y}))
        new = pl.DataFrame({"x": grid})
        # `predict(interval="confidence")` returns Fitted plus CI columns named
        # `CI[2.5%]` / `CI[97.5]%` (the closing-bracket asymmetry is hea.lm
        # internal). We back out se from the CI half-width.
        pred = fit.predict(new=new, interval="confidence", alpha=0.05)
        yhat = pred["Fitted"].to_numpy()
        ci_lo = pred[pred.columns[1]].to_numpy()
        # 95% CI half-width / t_crit * sqrt(...) — we re-derive se from
        # (CI - fit) / 1.96. The smoother's `level` then re-inflates to
        # the user's chosen level.
        from scipy.stats import t
        df = float(fit.df_residuals)
        t_crit = t.ppf(0.975, df)
        se = (yhat - ci_lo) / t_crit
        return yhat, se

    if method == "gam":
        from ...gam import gam

        # ggplot2/mgcv default: smooth term on x with mgcv's default basis.
        fml = formula or "y ~ s(x)"
        fit = gam(fml, pl.DataFrame({"x": x, "y": y}))
        yhat, se = fit.predict(
            newdata=pl.DataFrame({"x": grid}),
            type="response",
            se_fit=True,
        )
        return np.asarray(yhat), np.asarray(se)

    if method == "glm":
        raise NotImplementedError(
            f"stat_smooth: method='glm' is not yet implemented; pass "
            "method='gam' for nonlinear fits or method='lm' for linear."
        )

    raise ValueError(f"stat_smooth: unknown method {method!r}")


def stat_smooth(*, method="loess", formula=None, se=True, level=0.95,
                span=0.75, n=80):
    return StatSmooth(method=method, formula=formula, se=se, level=level,
                      span=span, n=n)
