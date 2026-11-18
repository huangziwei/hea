"""Parametric generalized linear models — port of R's ``stats::glm``.

Fisher-scored IRLS using the family/link primitives in :mod:`hea.family`.
Output API mirrors :class:`hea.lm` (so ``bhat``, ``se_bhat``, ``ci_bhat``,
``yhat`` exist as 1-row :class:`polars.DataFrame`s) and adds the GLM-specific
fields that ``summary.glm`` reports: ``deviance``, ``null_deviance``,
``df_residual``, ``df_null``, ``dispersion``, ``aic``, ``iter``, ``converged``.

The module is family-agnostic: it never branches on ``family.name`` /
``link.name``. The only allowed dispatch is on ``family.scale_known``
(controls dispersion estimation and the t-vs-z choice for Wald inference).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .._rlinalg import r_chol2inv, r_matprod, r_qr, r_qr_coef, r_qr_R
from .._rsum import r_sum
from ..family import (
    Binomial,
    Family,
    Gaussian,
    QuasiBinomial,
    _coerce_response,
)
from ..formula import Call, _eval_atom, deparse, materialize, parse, prepare_design
from ..R import distributions as _dist
from ..R import nmath as _nmath
from ..utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    format_signif_jointly,
    significance_code,
)
from .lm import _label_top_n, _lowess, _qq_plot

__all__ = ["glm"]


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    """Build a 1-row pl.DataFrame from a flat numpy array + column names."""
    flat = np.asarray(values).reshape(-1)
    return pl.DataFrame({c: [float(flat[i])] for i, c in enumerate(columns)})


class glm:
    """Generalized linear model — parity target is R's ``stats::glm``.

    Parameters
    ----------
    formula : str
        R-style formula, e.g. ``"y ~ x1 + x2"``. The binomial
        ``cbind(success, failure) ~ ...`` form is rewritten to
        (proportion, ``weights=`` · trials) exactly like R's binomial
        ``initialize``; small LHS expressions (``log(y)``, ``y/100``)
        are also accepted.
    data : polars.DataFrame
        Input data; rows with NA in any referenced column are dropped
        (R's ``na.action = na.omit``).
    family : :class:`hea.family.Family`, optional
        Defaults to :class:`Gaussian` (= identity link). Pass e.g.
        ``Poisson()``, ``Gamma(link="log")``, ``Binomial(link="probit")``.
    weights : array-like or None
        Prior weights. For binomial, this is the binomial size ``m``
        (= 1 for Bernoulli). Defaults to ones.
    offset : array-like or None
        Length-``n`` offset added directly to the linear predictor
        (η = Xβ + offset). Defaults to zeros.
    control : dict
        IRLS control: ``epsilon`` (default 1e-8), ``maxit`` (25). Match R.
    start, etastart, mustart : array-like or None
        R's starting coefficients, linear predictor or means for the IRLS
        (``glm.fit``).

    Attributes
    ----------
    bhat, se_bhat, ci_bhat, t_values, p_values : 1-row pl.DataFrame
        Coefficient estimates and Wald inference. The third column header
        in the printed summary is ``Pr(>|t|)`` for unknown-scale families
        (Gaussian, Gamma, IG) and ``Pr(>|z|)`` for known-scale (Poisson,
        binomial), matching ``summary.glm``.
    yhat : pl.DataFrame
        Single-column ``Fitted`` (= μ̂). Use ``predict`` for new data.
    fitted_values, linear_predictors : numpy.ndarray
        μ̂ and η̂ on the training rows.
    deviance, null_deviance, df_residual, df_null : float / int
        From ``family.dev_resids``; null fit refits an intercept-only model.
    dispersion : float
        Pearson estimator ``Σ w·(y−μ)²/V / df_residual``; ``1.0`` for
        scale-known families. Aliased as ``scale`` and ``sigma_squared``.
    aic, bic, loglike : float
        ``aic = family.aic(...) + 2·k`` with ``k = rank(X) + (1 if not
        scale_known else 0)``; ``BIC = aic - 2k + log(n)·k``;
        ``loglike = -(aic - 2k)/2``.
    iter, converged : int / bool
        Fisher-scoring iteration count and convergence flag.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        family: Family | None = None,
        weights: None | np.ndarray | list = None,
        offset: None | np.ndarray | list = None,
        control: dict | None = None,
        start: None | np.ndarray | list = None,
        etastart: None | np.ndarray | list = None,
        mustart: None | np.ndarray | list = None,
    ):
        self.formula = formula
        self.data = data
        if isinstance(family, type) and issubclass(family, Family):
            family = family()
        self.family = Gaussian() if family is None else family
        ctl = {"epsilon": 1e-8, "maxit": 25}
        if control:
            ctl.update(control)

        f_parsed = parse(formula)
        _cbind = (
            isinstance(f_parsed.lhs, Call)
            and f_parsed.lhs.fn == "cbind"
            and len(f_parsed.lhs.args) == 2
        )
        if _cbind:
            if not isinstance(self.family, (Binomial, QuasiBinomial)):
                raise ValueError(
                    "cbind(success, failure) ~ ... LHS only makes sense "
                    "for Binomial/QuasiBinomial; got family="
                    f"{self.family.name!r}"
                )
            s_blk = _eval_atom(f_parsed.lhs.args[0], data)
            f_blk = _eval_atom(f_parsed.lhs.args[1], data)
            s = s_blk.values.flatten().astype(float)
            f = f_blk.values.flatten().astype(float)
            if np.any(s < 0) or np.any(f < 0):
                raise ValueError("negative counts in cbind() response")
            if np.any(np.abs(s - np.rint(s)) > 0.001) or np.any(
                np.abs(f - np.rint(f)) > 0.001
            ):
                import warnings

                warnings.warn("non-integer counts in a binomial glm!", stacklevel=2)
            tot = s + f
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(tot > 0, s / tot, 0.0)
            p = np.where(np.isnan(tot), np.nan, p)
            data = data.with_columns(
                pl.Series("_hea_cbind_p", p),
                pl.Series("_hea_cbind_n", tot),
            )
            formula = f"_hea_cbind_p ~ {deparse(f_parsed.rhs)}"

        self._basis_state: dict = {}
        d = prepare_design(formula, data, basis_state=self._basis_state)
        self._expanded = d.expanded
        self._design_data = d.data
        self.X = d.X  # pl.DataFrame
        self.y = d.y  # pl.Series

        X = d.X_values
        if X is None:
            X = self.X.to_numpy().astype(float)
        elif X.dtype != np.float64:
            X = X.astype(np.float64)
        y = _coerce_response(self.y, self.family)
        self._y_numeric = y
        n, p = X.shape

        prior_w = (
            np.ones(n)
            if weights is None
            else np.asarray(weights, dtype=float).flatten()
        )
        if prior_w.shape != (n,):
            raise ValueError(f"weights must have length {n}, got {prior_w.shape}")
        if np.any(prior_w < 0):
            raise ValueError("negative weights not allowed")
        self._binom_n = None
        if _cbind:
            self._binom_n = d.data["_hea_cbind_n"].to_numpy().astype(float)
            prior_w = prior_w * self._binom_n
        self._prior_w = prior_w

        off = (
            np.zeros(n) if offset is None else np.asarray(offset, dtype=float).flatten()
        )
        if off.shape != (n,):
            raise ValueError(f"offset must have length {n}, got {off.shape}")
        for off_node in d.expanded.offsets:
            blk = _eval_atom(off_node, d.data)
            off = off + blk.values.flatten().astype(float)
        self._offset = off

        self.column_names = list(self.X.columns)
        self.feature_names = (
            self.column_names[1:]
            if "(Intercept)" in self.column_names
            else self.column_names
        )
        self.has_intercept = "(Intercept)" in self.column_names

        self.n = n
        self.p = p

        from ..R.glm_fit import glm_fit

        fit = glm_fit(
            X,
            y,
            family=self.family,
            weights=prior_w,
            offset=off,
            start=start,
            etastart=etastart,
            mustart=mustart,
            epsilon=ctl["epsilon"],
            maxit=ctl["maxit"],
            intercept=self.has_intercept,
            binom_n=self._binom_n,
        )
        self._fit = fit
        self._X = X
        self._has_offset = offset is not None or len(d.expanded.offsets) > 0
        self.iter = fit["iter"]
        self.converged = fit["converged"]
        self.rank = fit["rank"]
        self.df_residual = fit["df_residual"]
        self.df_residuals = self.df_residual  # lm-style alias

        self._bhat_arr = np.asarray(fit["coefficients"], dtype=float).copy()
        self.bhat = _row_frame(self._bhat_arr, self.column_names)
        from ..R import NamedVector

        self.coef = NamedVector(self.column_names, self._bhat_arr)
        self.coefficients = self.coef

        self.fitted_values = fit["fitted"]
        self.linear_predictors = fit["linear_predictors"]
        self.yhat = pl.DataFrame({"fit": fit["fitted"]})  # lm-style
        self.fitted = self.fitted_values

        self.deviance = fit["deviance"]

        self.dispersion = self._compute_dispersion(fit)
        self.scale = self.dispersion  # gam-style alias
        self.sigma_squared = self.dispersion  # lm-style alias
        self.sigma = float(np.sqrt(self.dispersion))

        self.vcov, self._XtWXinv = self._compute_vcov(fit)
        self.V_bhat = self.vcov  # lm-style alias
        se = np.sqrt(np.diag(self.vcov))
        self._se_bhat_arr = se
        self.se_bhat = _row_frame(se, self.column_names)

        keep = ~np.isnan(np.diag(self._XtWXinv))
        if keep.any():
            Xk = X[:, keep]
            XtWXinv_k = self._XtWXinv[np.ix_(keep, keep)]
            HXk = Xk @ XtWXinv_k
            self.leverage = (HXk * Xk).sum(axis=1) * fit["weights"]
        else:
            self.leverage = np.zeros(n)
        denom = np.sqrt(self.dispersion) * np.sqrt(
            np.clip(1.0 - self.leverage, 1e-12, None)
        )
        self.std_dev_residuals = self.residuals_of("deviance") / denom
        self.std_pearson_residuals = self.residuals_of("pearson") / denom

        self._test_kind = "z" if self.family.scale_known else "t"
        with np.errstate(invalid="ignore", divide="ignore"):
            stat = self._bhat_arr / self._se_bhat_arr
        self._stat_arr = stat
        self.t_values = _row_frame(stat, self.column_names)
        self.z_values = self.t_values  # alias for known-scale
        self.p_values = _row_frame(self._wald_p(stat), self.column_names)
        self.ci_bhat = self._compute_ci(0.05)

        self.null_deviance = self._compute_null_deviance(y, prior_w, off, ctl)
        self.df_null = fit["df_null"]
        self.npar = fit["rank"] + (0 if self.family.scale_known else 1)
        self.aic = fit["aic"]
        self.AIC = self.aic
        self.loglike = float(self.npar) - 0.5 * self.aic
        self.bic = -2.0 * self.loglike + float(np.log(self.n)) * self.npar
        self.BIC = self.bic

    def _compute_dispersion(self, fit: dict) -> float:
        """summary.glm's dispersion: 1 for poisson and binomial, otherwise
        the working weights times the squared working residuals summed over
        the rows of positive weight, over the residual df (NaN without
        residual df)."""
        if self.family.scale_known:
            return 1.0
        if self.df_residual <= 0:
            return float("nan")
        wts = fit["weights"]
        res = fit["residuals"]
        return r_sum((wts * res**2)[wts > 0]) / self.df_residual

    def _compute_vcov(self, fit: dict):
        """vcov.glm: ``chol2inv`` of the leading rank-by-rank block of the
        last IRLS step's QR factor, in pivoted order, times the dispersion;
        the aliased coefficients' rows and columns NA. Returns the scaled
        and the unscaled covariance."""
        p = self.p
        rank = fit["rank"]
        V = np.full((p, p), np.nan)
        U = np.full((p, p), np.nan)
        if rank == 0:
            return V, U
        piv = np.asarray(fit["qr"]["pivot"])[:rank] - 1
        cov_unscaled = r_chol2inv(np.asarray(fit["qr"]["qr"])[:rank, :rank])
        U[np.ix_(piv, piv)] = cov_unscaled
        V[np.ix_(piv, piv)] = self.dispersion * cov_unscaled
        return V, U

    def _wald_p(self, stat: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            if self._test_kind == "z":
                return 2.0 * _nmath.pnorm5_vec(np.abs(stat), lower_tail=False)
            return 2.0 * _dist.pt(np.abs(stat), self.df_residual, lower_tail=False)

    def _compute_ci(self, alpha: float) -> pl.DataFrame:
        q = float(_nmath.qnorm5(1.0 - alpha / 2.0))
        bhat = self._bhat_arr
        se = self._se_bhat_arr
        lo = bhat - q * se
        hi = bhat + q * se
        return pl.DataFrame(
            {
                "coef": self.column_names,
                f"CI[{alpha / 2 * 100}%]": lo,
                f"CI[{100 - alpha / 2 * 100}]%": hi,
            }
        )

    def _compute_null_deviance(self, y, prior_w, offset, ctl) -> float:
        """glm's null deviance: glm.fit's (the weighted mean, or the offset
        alone without an intercept), refitted by glm.fit on the intercept
        from the model's fitted values when there is an offset and an
        intercept (glm.R)."""
        if not (self._has_offset and self.has_intercept):
            return self._fit["null_deviance"]
        from ..R.glm_fit import glm_fit

        icol = self.column_names.index("(Intercept)")
        fit2 = glm_fit(
            self._X[:, [icol]],
            y,
            family=self.family,
            weights=prior_w,
            offset=offset,
            mustart=self._fit["fitted"],
            epsilon=ctl["epsilon"],
            maxit=ctl["maxit"],
            intercept=True,
            binom_n=self._binom_n,
        )
        if not fit2["converged"]:
            import warnings

            warnings.warn(
                "fitting to calculate the null deviance did not converge -- "
                "increase 'maxit'?",
                stacklevel=3,
            )
        return fit2["deviance"]

    def residuals_of(self, type: str = "deviance") -> np.ndarray:
        """Residuals of the requested type.

        ``type`` ∈ ``{"deviance", "pearson", "working", "response"}``;
        defaults to deviance, matching ``residuals.glm``.
        """
        y = self._y_numeric
        mu = self.fitted_values
        prior_w = self._prior_w
        link = self.family.link
        if type == "response":
            return y - mu
        if type == "working":
            mu_eta_v = link.mu_eta(self.linear_predictors)
            return (y - mu) / mu_eta_v
        if type == "pearson":
            V = self.family.variance(mu)
            return np.sqrt(prior_w) * (y - mu) / np.sqrt(V)
        if type == "deviance":
            per_obs = self.family.dev_resids(y, mu, prior_w)
            return np.sign(y - mu) * np.sqrt(np.maximum(per_obs, 0.0))
        raise ValueError(
            f"unknown residual type {type!r}; "
            "expected one of: deviance, pearson, working, response"
        )

    @property
    def residuals(self) -> pl.DataFrame:
        return pl.DataFrame({"residuals": self.residuals_of("deviance")})

    def predict(
        self,
        new: pl.DataFrame | None = None,
        type: str = "response",
        se_fit: bool = False,
        offset: None | np.ndarray | list = None,
        alpha: float = 0.05,
    ):
        """Generate predictions on new data — :func:`predict.glm` parity.

        ``type='response'`` returns μ̂ = linkinv(η̂); ``type='link'`` returns
        η̂. Returns a ``pl.DataFrame`` with column ``fit``; if ``se_fit=True``
        also a column ``se.fit``. Without new data and standard errors the
        fit's own values come back; otherwise ``predict.lm``'s
        ``X[, piv] %*% beta[piv]`` plus the offset, and its standard errors
        from ``X[, piv]`` times the inverse of the QR factor at the scale
        ``sqrt(dispersion)``; on the response scale they are multiplied by
        ``|dμ/dη|``.
        """
        if new is None:
            X_new = self._X
            n_new = self.n
            default_off = self._offset
        else:
            X_new = (
                materialize(self._expanded, new, basis_state=self._basis_state)
                .to_numpy()
                .astype(float)
            )
            n_new = X_new.shape[0]
            default_off = np.zeros(n_new)
            for off_node in self._expanded.offsets:
                blk = _eval_atom(off_node, new)
                default_off = default_off + blk.values.flatten().astype(float)
        off_new = (
            default_off if offset is None else np.asarray(offset, dtype=float).flatten()
        )
        if off_new.shape != (n_new,):
            raise ValueError(f"offset must have length {n_new}")

        if new is None and offset is None and not se_fit:
            # predict.glm without newdata returns the fit's own values.
            if type == "link":
                return pl.DataFrame({"fit": self.linear_predictors})
            if type == "response":
                return pl.DataFrame({"fit": self.fitted_values})
        if type not in ("link", "response"):
            raise ValueError(
                f"unknown predict type {type!r}; expected 'response' or 'link'"
            )
        # predict.lm: X[, piv] %*% beta[piv] plus the offset.
        rank = self.rank
        piv = np.asarray(self._fit["qr"]["pivot"])[:rank] - 1
        if rank > 0:
            eta = (
                r_matprod(
                    np.ascontiguousarray(X_new[:, piv]),
                    self._bhat_arr[piv].reshape(-1, 1),
                )[:, 0]
                + off_new
            )
        else:
            eta = np.zeros(n_new) + off_new
        fit = eta if type == "link" else self.family.link.linkinv(eta)
        if not se_fit:
            return pl.DataFrame({"fit": fit})
        # predict.lm's se.fit at scale sqrt(dispersion): X[, piv] times the
        # inverse of the QR factor's leading block (qr.solve), squared and
        # summed against res.var = scale^2.
        if rank > 0:
            Rp = r_qr_R(self._fit["qr"])[:rank, :rank]
            qrR = r_qr(Rp)
            if qrR["rank"] != rank:
                raise ValueError("singular matrix 'a' in solve")
            Rinv = r_qr_coef(qrR, np.eye(rank))
            Rinv[np.isnan(Rinv)] = 0.0
            XRinv = r_matprod(np.ascontiguousarray(X_new[:, piv]), Rinv)
            scale = float(np.sqrt(self.dispersion))
            res_var = scale * scale
            ip = r_matprod(XRinv * XRinv, np.full((rank, 1), res_var))[:, 0]
        else:
            ip = np.zeros(n_new)
        se_link = np.sqrt(ip)
        if type == "link":
            return pl.DataFrame({"fit": fit, "se.fit": se_link})
        mu_eta_v = self.family.link.mu_eta(eta)
        return pl.DataFrame({"fit": fit, "se.fit": se_link * np.abs(mu_eta_v)})

    def __repr__(self) -> str:
        d = f"glm({self.formula!r}, family={self.family!r})\n\n"
        d += "Coefficients:\n"
        d += format_df(self.bhat)
        return d

    def __str__(self) -> str:
        return self.__repr__()

    def summary(self, digits: int = 4) -> None:
        """Print a ``summary.glm``-styled report."""
        stat_lbl = "z value" if self._test_kind == "z" else "t value"
        p_lbl = "Pr(>|z|)" if self._test_kind == "z" else "Pr(>|t|)"

        p_arr = np.asarray(self.p_values.row(0), dtype=float)
        sig = significance_code(p_arr)
        ci_low_col, ci_hi_col = self.ci_bhat.columns[1], self.ci_bhat.columns[2]
        ci_low_arr = self.ci_bhat[ci_low_col].to_numpy()
        ci_hi_arr = self.ci_bhat[ci_hi_col].to_numpy()
        est_s, se_s = format_signif_jointly(
            [self._bhat_arr, self._se_bhat_arr],
            digits=digits,
        )
        cilo_s, cihi_s = format_signif_jointly(
            [ci_low_arr, ci_hi_arr],
            digits=digits,
        )

        coef_df = pl.DataFrame(
            {
                "": self.column_names,
                "Estimate": est_s,
                "Std. Error": se_s,
                ci_low_col: cilo_s,
                ci_hi_col: cihi_s,
                stat_lbl: format_signif(self._stat_arr, digits=digits),
                p_lbl: format_pval(p_arr, digits=_dig_tst(digits)),
                " ": sig,
            }
        )

        num_align = {
            c: "right"
            for c in ("Estimate", "Std. Error", ci_low_col, ci_hi_col, stat_lbl, p_lbl)
        }
        out = f"Call:\nglm(formula = {self.formula}, family = {self.family})\n\n"
        out += "Coefficients:\n"
        out += format_df(coef_df, align=num_align)
        out += "\n---"
        out += "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"

        disp_word = "fixed" if self.family.scale_known else "estimated"
        out += (
            f"\n(Dispersion parameter for {self.family.name} family "
            f"taken to be {self.dispersion:.{digits}g}; {disp_word})\n"
        )
        out += (
            f"\n    Null deviance: {self.null_deviance:.4f}  on {self.df_null} degrees of freedom"
            f"\nResidual deviance: {self.deviance:.4f}  on {self.df_residual} degrees of freedom"
        )
        out += f"\nAIC: {self.aic:.4f}    BIC: {self.bic:.4f}    logLik: {self.loglike:.4f}"
        out += f"\n\nNumber of Fisher Scoring iterations: {self.iter}"
        out += "" if self.converged else "  (did NOT converge!)"
        print(out)

    def plot_observed_fitted(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        label_n=3,
    ):
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)
        y = self._y_numeric
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
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        smooth=True,
        label_n=3,
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
            ax,
            self.std_dev_residuals,
            label_n=label_n,
            ylabel="Std. deviance resid.",
        )
        return ax

    def plot_scale_location(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        smooth=True,
        label_n=3,
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
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        cook_levels=(0.5, 1.0),
        smooth=True,
        label_n=3,
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
        ymin, ymax = ax.get_ylim()
        h_max = float(np.clip(h.max() * 1.1, 1e-3, 0.999))
        h_grid = np.linspace(1e-3, h_max, 200)
        for c in cook_levels:
            rline = np.sqrt(c * self.rank * (1 - h_grid) / h_grid)
            ax.plot(h_grid, rline, color="red", linestyle="--", linewidth=0.8)
            ax.plot(h_grid, -rline, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        cook = (r**2 / self.rank) * h / np.clip(1 - h, 1e-12, None)
        _label_top_n(ax, h, r, scores=cook, n=label_n)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Std. Pearson resid.")
        ax.set_title("Residuals vs. Leverage")
        return ax

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic, matching R's plot.glm default."""
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        self.plot_leverage(ax=axes[1, 1], smooth=smooth, label_n=label_n)
        fig.tight_layout()
        return fig
