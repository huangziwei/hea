"""Port of R's emmeans package (subset).

emmeans is a standalone CRAN package, not base R; lives inside
:mod:`hea.R` for now because the port is small and shares its real peers
(:mod:`hea.R.htest`, :mod:`hea.R.model_generics`) — promote to a sibling
package when the surface grows beyond the Faraway lmwr coverage.

Initial scope, driven by the Faraway lmwr scripts that use it:

- ``emmeans(model, ~factor)`` — marginal means for ``factor``, averaging
  the design over every other predictor (R's "equal-weighted" reference
  grid default).
- ``emmeans(model, pairwise ~ factor)`` — same plus all pairwise
  differences between factor levels, Tukey-adjusted by default.
- ``emmeans(model, trt.vs.ctrlk ~ factor, ref=N, side="<")`` — each
  non-reference level vs. the reference level (``ref=N`` selects the
  0-based reference level; R's emmeans uses 1-based, hea follows
  Python indexing. ``side="<"`` runs a one-sided lower test).
- ``.emmeans`` / ``.contrasts`` accessors on the return object.
- ``summary(emmGrid.contrasts, infer=True, adjust="bonferroni"/"tukey")``
  re-formats the contrasts with CIs and an alternative adjustment.

What's intentionally **out of scope** for v1:

- ``type="response"`` back-transformation (the ``log(...) ~ ...`` cases).
- Interactions inside ``specs`` (e.g. ``pairwise ~ A | B``).
- ``trt.vs.ctrl`` (vs first level only) — only ``trt.vs.ctrlk`` ("k"
  = your choice via ``ref=``) handled.
- Custom-list contrasts via ``contrast(emmGrid, method=list(...))``.

Reference: https://cran.r-project.org/package=emmeans .
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats as _sps


@dataclass
class EmmGrid:
    """Subset of R's ``emmGrid`` object.

    ``.emmeans`` and ``.contrasts`` are the two tables consumers reach
    for. Item access (``rem["emmeans"]``) is supported for translated R
    that uses ``$`` accessor syntax.
    """

    emmeans: pl.DataFrame
    contrasts: pl.DataFrame
    model: object
    factor: str
    levels: list
    L: np.ndarray
    V: np.ndarray
    df_residuals: float
    adjust: str

    def __getitem__(self, key):
        return getattr(self, key)


def emmeans(
    model,
    specs: str,
    *,
    adjust: str = "tukey",
    ref: int | None = None,
    side: str | None = None,
    level: float = 0.95,
    **_unused,
) -> EmmGrid:
    """Estimated marginal means + optional contrasts.

    Parameters
    ----------
    model
        Fitted ``hea.lm`` / ``hea.glm``.
    specs
        Formula-shaped string. ``"~variety"`` → marginal means only.
        ``"pairwise ~ variety"`` → also all pairwise contrasts.
        ``"trt.vs.ctrlk ~ variety"`` (with ``ref=``) → each level vs.
        the reference level.
    adjust
        ``"tukey"`` (default for pairwise), ``"bonferroni"``, ``"none"``.
    ref
        0-based reference level for ``trt.vs.ctrlk`` contrasts (R /
        emmeans is 1-based; hea follows Python indexing).
    side
        ``"<"`` → one-sided lower-tail test (alternative ``μ_i < μ_ref``).
        ``">"`` → one-sided upper. ``None`` (default) → two-sided.
    level
        Confidence level for the CIs in ``.emmeans``. Default 0.95.

    Returns
    -------
    EmmGrid
    """
    spec_lhs, spec_rhs = _parse_specs(specs)
    target = spec_rhs.strip()

    levels = _factor_levels(model, target)
    L = _reference_grid(model, target, levels)
    beta = np.asarray(model.coef, dtype=float)
    V = np.asarray(model.V_bhat, dtype=float)
    df_resid = float(model.df_residuals)

    means_df = _means_table(target, levels, L, beta, V, df_resid, level)

    contrasts_df = pl.DataFrame()
    if spec_lhs == "pairwise":
        contrasts_df = _pairwise_table(
            target, levels, L, beta, V, df_resid, adjust=adjust
        )
    elif spec_lhs in ("trt.vs.ctrlk", "trt.vs.ctrl"):
        ref_pos = ref if ref is not None else 0
        if not (0 <= ref_pos < len(levels)):
            raise ValueError(
                f"emmeans: ref={ref} out of range for {len(levels)} levels"
            )
        contrasts_df = _vs_control_table(
            target, levels, L, beta, V, df_resid,
            ref_pos=ref_pos, side=side, adjust=adjust,
        )
    elif spec_lhs:
        raise NotImplementedError(
            f"emmeans: contrast type {spec_lhs!r} not yet supported "
            f"(in v1: pairwise, trt.vs.ctrlk)"
        )

    return EmmGrid(
        emmeans=means_df,
        contrasts=contrasts_df,
        model=model,
        factor=target,
        levels=levels,
        L=L,
        V=V,
        df_residuals=df_resid,
        adjust=adjust,
    )


def summary_emmgrid_contrasts(
    contrasts: pl.DataFrame,
    *,
    infer: bool = False,
    adjust: str | None = None,
    level: float = 0.95,
) -> pl.DataFrame:
    """Re-summarise a contrasts table with CIs and/or a different adjustment.

    Mirrors R's ``summary(emmGrid$contrasts, infer=TRUE, adjust="bonferroni")``.
    Re-derives p-values from the stored ``t.ratio`` and ``df`` so we can
    swap adjustment families without re-running the model.
    """
    df = contrasts
    if df.height == 0:
        return df

    t_vals = df["t.ratio"].to_numpy()
    dfs = df["df"].to_numpy().astype(float)
    k = df.height
    # Effective k for studentized range — recover number of levels from
    # the contrast count when adjust='tukey'. Without that we conservatively
    # use ``k`` directly.
    k_levels = _infer_k_levels(k)

    if adjust is not None:
        p_raw = 2.0 * _sps.t.sf(np.abs(t_vals), dfs)
        p_adj = _p_adjust(p_raw, t_vals, dfs, adjust, k_levels)
        df = df.with_columns(pl.Series("p.value", p_adj))
        df = df.with_columns(pl.lit(adjust).alias("adjust"))

    if infer:
        z = _sps.t.ppf((1 + level) / 2, dfs)
        est = df["estimate"].to_numpy()
        se = df["SE"].to_numpy()
        df = df.with_columns([
            pl.Series("lower.CL", est - z * se),
            pl.Series("upper.CL", est + z * se),
        ])
    return df


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


def _parse_specs(specs: str) -> tuple[str, str]:
    """``"pairwise ~ variety"`` → ``("pairwise", "variety")``;
    ``"~variety"`` → ``("", "variety")``."""
    if "~" not in specs:
        raise ValueError(
            f"emmeans: specs must contain '~'; got {specs!r}"
        )
    lhs, rhs = (s.strip() for s in specs.split("~", 1))
    return lhs, rhs


def _factor_levels(model, target: str) -> list:
    """Levels of the target factor, in their natural-sort order.

    Drawn from the model's data column; emmeans needs the *original*
    level set (the design matrix may drop the reference level).
    """
    col = model.data[target]
    levels = sorted(col.unique().to_list())
    return levels


def _reference_grid(model, target: str, levels: list) -> np.ndarray:
    """Build the L matrix: one row per target level.

    Each row sets the intercept to 1, the target's dummy to 1 for that
    level (0 for the reference, which has no dummy), and every other
    column to its data-mean (R's "equal-weighted" default).
    """
    col_names = list(model.column_names)
    X = model.X.to_numpy().astype(float)

    target_cols = [
        (j, col_names[j][len(target):])
        for j, name in enumerate(col_names)
        if name.startswith(target) and name != target and name != "(Intercept)"
    ]
    target_col_idx = {lvl: j for j, lvl in target_cols}
    intercept_idx = (
        col_names.index("(Intercept)") if "(Intercept)" in col_names else None
    )
    other_cols = [
        j for j in range(X.shape[1])
        if j not in target_col_idx.values() and j != intercept_idx
    ]
    other_means = X[:, other_cols].mean(axis=0)

    L = np.zeros((len(levels), X.shape[1]))
    if intercept_idx is not None:
        L[:, intercept_idx] = 1.0
    for j, m in zip(other_cols, other_means):
        L[:, j] = m
    for i, lvl in enumerate(levels):
        key = str(lvl)
        if key in target_col_idx:
            L[i, target_col_idx[key]] = 1.0
        # Otherwise this is the reference level — all target dummies 0.
    return L


def _means_table(
    target, levels, L, beta, V, df_resid, level,
) -> pl.DataFrame:
    """Marginal means with SE and confidence intervals."""
    means = L @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", L, V, L))
    z = _sps.t.ppf((1 + level) / 2, df_resid)
    return pl.DataFrame({
        target: [str(lvl) for lvl in levels],
        "emmean": means,
        "SE": se,
        "df": [df_resid] * len(levels),
        "lower.CL": means - z * se,
        "upper.CL": means + z * se,
    })


def _pairwise_table(
    target, levels, L, beta, V, df_resid, *, adjust,
) -> pl.DataFrame:
    """All C(k, 2) pairwise differences with adjusted p-values."""
    k = len(levels)
    L_arr = np.asarray(L, dtype=float)
    iu, ju = np.triu_indices(k, k=1)
    C = L_arr[iu] - L_arr[ju]                    # (n_pairs, p)
    ests_a = C @ np.asarray(beta, dtype=float)   # (n_pairs,)
    CV = C @ np.asarray(V, dtype=float)
    quad = np.einsum("ip,ip->i", CV, C)
    ses_a = np.sqrt(np.maximum(quad, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ts_a = np.where(ses_a > 0, ests_a / ses_a, np.nan)
    labels = [f"{levels[int(i)]} - {levels[int(j)]}" for i, j in zip(iu, ju)]
    p_raw = 2.0 * _sps.t.sf(np.abs(ts_a), df_resid)
    p_adj = _p_adjust(p_raw, ts_a, np.full_like(ts_a, df_resid), adjust, k)
    return pl.DataFrame({
        "contrast": labels,
        "estimate": ests_a,
        "SE": ses_a,
        "df": np.full(len(labels), df_resid),
        "t.ratio": ts_a,
        "p.value": np.asarray(p_adj),
        "adjust": [adjust] * len(labels),
    })


def _vs_control_table(
    target, levels, L, beta, V, df_resid, *, ref_pos, side, adjust,
) -> pl.DataFrame:
    """Each non-reference level minus the reference (R's ``trt.vs.ctrlk``)."""
    k = len(levels)
    L_arr = np.asarray(L, dtype=float)
    nonref = np.array([i for i in range(k) if i != ref_pos], dtype=int)
    C = L_arr[nonref] - L_arr[ref_pos]            # (k-1, p)
    ests_a = C @ np.asarray(beta, dtype=float)
    CV = C @ np.asarray(V, dtype=float)
    quad = np.einsum("ip,ip->i", CV, C)
    ses_a = np.sqrt(np.maximum(quad, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ts_a = np.where(ses_a > 0, ests_a / ses_a, np.nan)
    labels = [f"{levels[int(i)]} - {levels[ref_pos]}" for i in nonref]
    if side == "<":
        p_raw = _sps.t.cdf(ts_a, df_resid)
    elif side == ">":
        p_raw = _sps.t.sf(ts_a, df_resid)
    else:
        p_raw = 2.0 * _sps.t.sf(np.abs(ts_a), df_resid)
    p_adj = _p_adjust(p_raw, ts_a, np.full_like(ts_a, df_resid), adjust, k - 1)
    return pl.DataFrame({
        "contrast": labels,
        "estimate": ests_a,
        "SE": ses_a,
        "df": np.full(len(labels), df_resid),
        "t.ratio": ts_a,
        "p.value": np.asarray(p_adj),
        "adjust": [adjust] * len(labels),
    })


def _p_adjust(p_raw, t_vals, dfs, method, k_levels) -> np.ndarray:
    """Adjust p-values. ``method`` is ``"none"``, ``"bonferroni"``, or
    ``"tukey"`` — Tukey uses the studentized-range distribution."""
    method = (method or "none").lower()
    if method == "none":
        return np.asarray(p_raw)
    if method == "bonferroni":
        m = len(p_raw)
        return np.minimum(np.asarray(p_raw) * m, 1.0)
    if method == "tukey":
        # Studentized range for ``k_levels`` groups.
        q = np.abs(t_vals) * np.sqrt(2.0)
        try:
            sr = _sps.studentized_range
            return 1.0 - sr.cdf(q, k_levels, dfs)
        except AttributeError:
            # Older scipy fallback — Bonferroni is a conservative proxy.
            m = len(p_raw)
            return np.minimum(np.asarray(p_raw) * m, 1.0)
    raise ValueError(f"emmeans: unknown adjust={method!r}")


def _infer_k_levels(n_contrasts: int) -> int:
    """Back out the level count ``k`` from the contrast count.

    Pairwise contrast count is ``k*(k-1)/2``; solve for ``k``. Used to
    re-derive a Tukey adjustment when only the contrasts table is at
    hand (R's ``summary(...)`` path).
    """
    # Solve k(k-1)/2 = n → k = (1 + sqrt(1 + 8n)) / 2.
    k = (1 + np.sqrt(1 + 8 * n_contrasts)) / 2
    return int(round(k))
