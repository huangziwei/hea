"""R's hypothesis-test family plus the ``htest`` / ``Anova`` print
containers and the rank-based helpers used by Lindeløv-style "tests as
lm" notebook constructions.

Every test function returns an :class:`HTest`; :func:`aov` returns an
:class:`AnovaTable`. R parameter conventions are kept: ``alternative`` ∈
{"two.sided", "greater", "less"}, ``conf_level=0.95``, ``correct=``
(continuity correction) where applicable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats as _sps
from . import distributions as _dist

from ._shared import _as_array, _fmt, _fmt_pval
from ..models.lm import lm


def _avg_rank(a) -> np.ndarray:
    """R's ``rank(x)`` with ``ties.method = "average"`` on a numeric vector —
    stable (mergesort) order, tied runs share their mid-rank. Matches R and
    ``scipy.stats.rankdata(method="average")`` without the scipy dependency."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0  # 1-based average rank of the tie run
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


@dataclass
class HTest:
    """R's ``htest`` class as a Python dataclass.

    Mirrors ``stats:::print.htest``: ``method`` is the title, ``statistic``
    the named scalar, ``parameter`` the df line, plus optional p-value,
    CI, point ``estimate``, and ``alternative``. ``data_name`` is the
    "data:" label R prints before the stats.
    """

    method: str
    statistic: dict = field(default_factory=dict)
    parameter: dict = field(default_factory=dict)
    p_value: Optional[float] = None
    conf_int: Optional[tuple] = None
    estimate: dict = field(default_factory=dict)
    null_value: Optional[Union[float, dict]] = None
    alternative: str = "two.sided"
    data_name: str = ""
    conf_level: float = 0.95

    def __repr__(self) -> str:
        out = ["", f"\t{self.method}", ""]
        if self.data_name:
            out.append(f"data:  {self.data_name}")
        bits = []
        for k, v in self.statistic.items():
            bits.append(f"{k} = {_fmt(v)}")
        for k, v in self.parameter.items():
            bits.append(f"{k} = {_fmt(v)}")
        if self.p_value is not None:
            bits.append(f"p-value = {_fmt_pval(self.p_value)}")
        if bits:
            out.append(", ".join(bits))
        if self.alternative:
            null = self.null_value
            tail = "not equal to"
            if self.alternative == "greater":
                tail = "greater than"
            elif self.alternative == "less":
                tail = "less than"
            if isinstance(null, dict):
                null_str = ", ".join(f"{k} = {_fmt(v)}" for k, v in null.items())
                out.append(f"alternative hypothesis: true {null_str.split(' = ')[0]} is {tail} {null_str.split(' = ')[1]}")
            elif null is not None:
                # name from estimate keys when possible
                nm = next(iter(self.estimate.keys()), "value")
                out.append(f"alternative hypothesis: true {nm} is {tail} {_fmt(null)}")
        if self.conf_int is not None:
            out.append(f"{int(self.conf_level * 100)} percent confidence interval:")
            out.append(f" {_fmt(self.conf_int[0])} {_fmt(self.conf_int[1])}")
        if self.estimate:
            out.append("sample estimates:")
            keys = "  ".join(f"{k}" for k in self.estimate)
            vals = "  ".join(f"{_fmt(v)}" for v in self.estimate.values())
            out.append(keys)
            out.append(vals)
        return "\n".join(out) + "\n"


@dataclass
class AnovaTable:
    """R-style ``Anova`` / ``anova`` table (Type-II by default for ``aov``).

    Stored as a list of rows (term, df, sum_sq, mean_sq, F, p) plus a
    Residuals row. ``__repr__`` formats it close to R's printout.
    """

    response: str
    rows: list  # list of dicts: term, df, sum_sq, mean_sq, F, p
    residual_df: int
    residual_ss: float
    type: str = "II"

    def __repr__(self) -> str:
        out = [f"Anova Table (Type {self.type} tests)", "",
               f"Response: {self.response}",
               f"{'':<12}{'Sum Sq':>10}{'Df':>4}{'F value':>10}{'Pr(>F)':>12}"]
        for r in self.rows:
            out.append(
                f"{r['term']:<12}{_fmt(r['sum_sq']):>10}{r['df']:>4}"
                f"{_fmt(r['F']):>10}{_fmt_pval(r['p']):>12}"
            )
        out.append(
            f"{'Residuals':<12}{_fmt(self.residual_ss):>10}{self.residual_df:>4}"
        )
        return "\n".join(out)


# ---- rank helpers (used by Wilcoxon/Spearman/Lindeløv constructions) -


def rank(x):
    """R's ``rank()`` with ``ties.method = "average"`` (R's default).

    Type-in / type-out: ``pl.Expr`` → ``pl.Expr``; ``pl.Series`` →
    ``pl.Series``; list / tuple / ndarray → ``np.ndarray`` (float, so
    downstream lm() formulas treat it as numeric).
    """
    if isinstance(x, pl.Expr):
        return x.rank("average")
    if isinstance(x, pl.Series):
        return x.rank("average")
    return _avg_rank(_as_array(x))


def signed_rank(x):
    """Lindeløv's ``signed_rank = function(x) sign(x) * rank(abs(x))``.

    Used to turn Wilcoxon signed-rank into an intercept-only ``lm``.
    Dispatches on input like :func:`rank`.
    """
    if isinstance(x, pl.Expr):
        return x.sign() * x.abs().rank("average")
    if isinstance(x, pl.Series):
        return x.sign() * x.abs().rank("average")
    arr = _as_array(x)
    return np.sign(arr) * _avg_rank(np.abs(arr))


# ---- hypothesis tests -----------------------------------------------
#
# Every function returns an :class:`HTest`. R parameter names are
# preserved where possible: ``alternative`` ∈ {"two.sided", "greater",
# "less"}, ``conf_level=0.95``, ``correct=`` (continuity correction)
# where applicable. ``mu`` / ``p`` / ``ratio`` carry their R meanings.

def t_test(
    x,
    y=None,
    *,
    paired: bool = False,
    var_equal: bool = False,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``t.test`` — faithful port of ``stats:::t.test.default``.

    - ``y=None``                       → one-sample t-test on ``x``.
    - ``y`` given, ``paired=True``     → paired t-test on ``x - y``.
    - ``y`` given, ``var_equal=False`` → Welch two-sample (**R's default**).
    - ``y`` given, ``var_equal=True``  → Student's pooled two-sample.

    Statistic, df, the one-/two-sided p-value and the Student-t CI all route
    through nmath (``pt`` / ``qt``) — 0-ulp to R, no scipy. NAs are dropped
    per R (``complete.cases`` for the paired branch).
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"t_test(): unknown alternative {alternative!r}")
    x = _as_array(x)
    has_y = y is not None
    if has_y:
        y = _as_array(y)
        if paired:
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
        else:
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
    else:
        if paired:
            raise ValueError("'y' is missing for paired test")
        x = x[np.isfinite(x)]
    if paired:
        x = x - y
        y = None

    nx = len(x)
    mx = float(np.mean(x))
    vx = float(np.var(x, ddof=1))
    if y is None:
        if nx < 2:
            raise ValueError("not enough 'x' observations")
        df = float(nx - 1)
        stderr = math.sqrt(vx / nx)
        tstat = (mx - mu) / stderr
        method = "Paired t-test" if paired else "One Sample t-test"
        estimate = {"mean difference" if paired else "mean of x": mx}
    else:
        ny = len(y)
        my = float(np.mean(y))
        vy = float(np.var(y, ddof=1))
        method = ("Welch " if not var_equal else "") + "Two Sample t-test"
        estimate = {"mean of x": mx, "mean of y": my}
        if var_equal:
            df = float(nx + ny - 2)
            v = 0.0
            if nx > 1:
                v += (nx - 1) * vx
            if ny > 1:
                v += (ny - 1) * vy
            v /= df
            stderr = math.sqrt(v * (1.0 / nx + 1.0 / ny))
        else:
            sx2, sy2 = vx / nx, vy / ny
            stderr = math.sqrt(sx2 + sy2)
            df = stderr ** 4 / (sx2 ** 2 / (nx - 1) + sy2 ** 2 / (ny - 1))
        tstat = (mx - my - mu) / stderr

    if alternative == "less":
        pval = float(_dist.pt(tstat, df))
        cint = (-math.inf, tstat + float(_dist.qt(conf_level, df)))
    elif alternative == "greater":
        pval = float(_dist.pt(tstat, df, lower_tail=False))
        cint = (tstat - float(_dist.qt(conf_level, df)), math.inf)
    else:
        pval = float(2.0 * _dist.pt(-abs(tstat), df))
        c = float(_dist.qt(1.0 - (1.0 - conf_level) / 2.0, df))
        cint = (tstat - c, tstat + c)

    return HTest(
        method=method,
        statistic={"t": float(tstat)},
        parameter={"df": float(df)},
        p_value=pval,
        conf_int=(mu + cint[0] * stderr, mu + cint[1] * stderr),
        estimate=estimate,
        null_value=mu,
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y" if has_y else "x",
    )


def wilcox_test(
    x,
    y=None,
    *,
    paired: bool = False,
    alternative: str = "two.sided",
    correct: bool = True,
) -> HTest:
    """R's ``wilcox.test``.

    Defaults to continuity correction (``correct=True``). One-sample and
    paired branches use ``scipy.stats.wilcoxon``; the two-sample branch
    uses ``mannwhitneyu`` (R's "Wilcoxon rank-sum" with W statistic).

    PARITY DEBT (blocked): R's *default* uses **exact** p-values for small
    n via the signed-rank / rank-sum distributions (``nmath/signrank.c`` +
    ``nmath/wilcox.c``, not yet ported). Until those land this stays on
    scipy's normal approximation. See r-stats-parity-debt.md §2.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.wilcoxon(x, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        res = _sps.wilcoxon(x, y, alternative=alt, correction=correct, zero_method="wilcox")
        return HTest(
            method="Wilcoxon signed rank test"
            + (" with continuity correction" if correct else ""),
            statistic={"V": float(res.statistic)},
            p_value=float(res.pvalue),
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    res = _sps.mannwhitneyu(
        x, y, alternative=alt, use_continuity=correct, method="asymptotic"
    )
    return HTest(
        method="Wilcoxon rank sum test"
        + (" with continuity correction" if correct else ""),
        statistic={"W": float(res.statistic)},
        p_value=float(res.pvalue),
        null_value=0.0,
        alternative=alternative,
        data_name="x and y",
    )


def _two_sided_min(lo, hi) -> float:
    """R's ``2 * min(p_lower, p_upper)`` two-sided rule, capped at 1."""
    return float(min(2.0 * min(lo, hi), 1.0))


def cor_test(
    x,
    y,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    continuity: bool = False,
) -> HTest:
    """R's ``cor.test`` with ``method`` in {pearson, spearman, kendall}.

    - **pearson**: ``t``, df = n-2, Fisher-z CI — a faithful nmath port
      (``pt`` / ``qnorm``), 0-ulp to R.
    - **spearman**: statistic ``S`` and the **asymptotic** (``exact=FALSE``)
      t-approximation p-value via nmath ``pt``.
    - **kendall**: statistic ``z`` and the **asymptotic** normal p-value
      (tie-corrected variance) via nmath ``pnorm``.

    Parity note: R's *default* for spearman/kendall uses **exact** small-sample
    p-values (AS 89 ``C_pRho`` / ``C_pKendall``); those C kernels
    (``src/prho.c`` / ``src/kendall.c``) are not yet ported, so the
    spearman/kendall p-values here equal R's ``exact = FALSE`` path (which R
    also uses for large n or with ties). See r-stats-parity-debt.md.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"cor_test(): unknown alternative {alternative!r}")
    x = _as_array(x)
    y = _as_array(y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    n = len(x)

    if method == "pearson":
        r = float(np.corrcoef(x, y)[0, 1])
        df = n - 2
        t = math.sqrt(df) * r / math.sqrt(1.0 - r * r)
        if alternative == "less":
            pval = float(_dist.pt(t, df))
        elif alternative == "greater":
            pval = float(_dist.pt(t, df, lower_tail=False))
        else:
            pval = _two_sided_min(float(_dist.pt(t, df)),
                                  float(_dist.pt(t, df, lower_tail=False)))
        conf_int = None
        if n > 3:
            z = math.atanh(r)
            sigma = 1.0 / math.sqrt(n - 3)
            if alternative == "less":
                cint = (-math.inf, z + sigma * float(_dist.qnorm(conf_level)))
            elif alternative == "greater":
                cint = (z - sigma * float(_dist.qnorm(conf_level)), math.inf)
            else:
                d = sigma * float(_dist.qnorm((1.0 + conf_level) / 2.0))
                cint = (z - d, z + d)
            conf_int = (math.tanh(cint[0]), math.tanh(cint[1]))
        return HTest(
            method="Pearson's product-moment correlation",
            statistic={"t": t},
            parameter={"df": df},
            p_value=pval,
            conf_int=conf_int,
            estimate={"cor": r},
            null_value=0.0,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )

    if method == "spearman":
        rho = float(np.corrcoef(_avg_rank(x), _avg_rank(y))[0, 1])
        S = (n ** 3 - n) * (1.0 - rho) / 6.0

        def pspearman(q, lower_tail):
            den = (n ** 3 - n) / 6.0
            if continuity:
                den += 1.0
            rr = 1.0 - q / den
            return float(_dist.pt(rr / math.sqrt((1.0 - rr * rr) / (n - 2)),
                                  df=n - 2, lower_tail=not lower_tail))

        if alternative == "greater":
            pval = pspearman(S, lower_tail=True)
        elif alternative == "less":
            pval = pspearman(S, lower_tail=False)
        else:
            p = (pspearman(S, lower_tail=False) if S > (n ** 3 - n) / 6.0
                 else pspearman(S, lower_tail=True))
            pval = float(min(2.0 * p, 1.0))
        return HTest(
            method="Spearman's rank correlation rho",
            statistic={"S": float(S)},
            p_value=pval,
            estimate={"rho": rho},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )

    if method == "kendall":
        # Kendall score S = sum_{i<j} sign(xi-xj) sign(yi-yj)  (O(n^2)).
        sx = np.sign(np.subtract.outer(x, x))
        sy = np.sign(np.subtract.outer(y, y))
        S = float(np.sum(np.triu(sx * sy, 1)))
        _, cx = np.unique(x, return_counts=True)
        _, cy = np.unique(y, return_counts=True)
        T0 = n * (n - 1) / 2.0
        T1 = float(np.sum(cx * (cx - 1)) / 2.0)
        T2 = float(np.sum(cy * (cy - 1)) / 2.0)
        tau = S / math.sqrt((T0 - T1) * (T0 - T2))
        v0 = n * (n - 1) * (2 * n + 5)
        vt = float(np.sum(cx * (cx - 1) * (2 * cx + 5)))
        vu = float(np.sum(cy * (cy - 1) * (2 * cy + 5)))
        v1 = float(np.sum(cx * (cx - 1)) * np.sum(cy * (cy - 1)))
        v2 = float(np.sum(cx * (cx - 1) * (cx - 2)) * np.sum(cy * (cy - 1) * (cy - 2)))
        var_S = ((v0 - vt - vu) / 18.0
                 + v1 / (2.0 * n * (n - 1))
                 + v2 / (9.0 * n * (n - 1) * (n - 2)))
        Sc = math.copysign(abs(S) - 1.0, S) if continuity else S
        z = Sc / math.sqrt(var_S)
        if alternative == "less":
            pval = float(_dist.pnorm(z))
        elif alternative == "greater":
            pval = float(_dist.pnorm(z, lower_tail=False))
        else:
            pval = _two_sided_min(float(_dist.pnorm(z)),
                                  float(_dist.pnorm(z, lower_tail=False)))
        return HTest(
            method="Kendall's rank correlation tau",
            statistic={"z": z},
            p_value=pval,
            estimate={"tau": tau},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    raise ValueError(f"unknown method: {method}")


def kruskal_test(formula: str, data: pl.DataFrame) -> HTest:
    """R's ``kruskal.test(y ~ group, data)``.

    Only the formula form is supported here — that's what the notebook
    uses. The numeric LHS is grouped by the RHS factor and passed to
    ``scipy.stats.kruskal``.
    """
    if "~" not in formula:
        raise ValueError("formula must look like 'y ~ group'")
    lhs, rhs = [s.strip() for s in formula.split("~", 1)]
    x = data[lhs].to_numpy().astype(float)
    g = np.asarray(data[rhs].to_list())
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    n = len(x)
    r = _avg_rank(x)
    glabels = np.unique(g)
    k = len(glabels)
    # STATISTIC = sum_j (sum of ranks in group j)^2 / n_j, then H with tie corr.
    stat = float(sum(r[g == gl].sum() ** 2 / (g == gl).sum() for gl in glabels))
    _, ties = np.unique(x, return_counts=True)
    ties = ties.astype(float)
    H = ((12.0 * stat / (n * (n + 1)) - 3.0 * (n + 1))
         / (1.0 - np.sum(ties ** 3 - ties) / (n ** 3 - n)))
    pval = float(_dist.pchisq(H, k - 1, lower_tail=False))
    return HTest(
        method="Kruskal-Wallis rank sum test",
        statistic={"Kruskal-Wallis chi-squared": float(H)},
        parameter={"df": int(k - 1)},
        p_value=pval,
        alternative="",
        data_name=f"{lhs} by {rhs}",
    )


def chisq_test(
    x,
    y=None,
    *,
    p=None,
    correct: bool = True,
) -> HTest:
    """R's ``chisq.test``.

    - 1-D ``x`` (and no ``y``)         → goodness-of-fit against ``p`` (uniform if None).
    - 2-D ``x`` (matrix or 2-D array)  → contingency-table test.
    - 1-D ``x`` and 1-D ``y``          → contingency on ``crosstab(x, y)``.
    """
    arr = np.asarray(x)
    if y is not None:
        tbl = _crosstab(x, y)
        return _chisq_table(tbl, correct=correct, name="x and y")
    if arr.ndim == 2:
        return _chisq_table(arr, correct=correct, name="x")
    # goodness of fit
    counts = arr.astype(float)
    if p is None:
        p = np.full_like(counts, 1.0 / len(counts))
    p = np.asarray(p, dtype=float)
    expected = counts.sum() * p
    stat = float(np.sum((counts - expected) ** 2 / expected))
    df = len(counts) - 1
    pval = float(_dist.pchisq(stat, df, lower_tail=False))
    return HTest(
        method="Chi-squared test for given probabilities",
        statistic={"X-squared": stat},
        parameter={"df": df},
        p_value=pval,
        alternative="",
        data_name="x",
    )


def _chisq_stat(tbl: np.ndarray, *, correct: bool) -> tuple[float, int, bool]:
    """Faithful ``chisq.test`` table kernel (nmath, no scipy). Returns
    ``(X-squared, df, yates_applied)``. Yates' correction applies only to a
    2×2 table when ``correct`` is set: ``YATES = min(0.5, min|O - E|)`` and
    ``X² = sum((|O - E| - YATES)² / E)`` — exactly ``stats:::chisq.test``."""
    tbl = np.asarray(tbl, dtype=float)
    nr, nc = tbl.shape
    n = tbl.sum()
    sr = tbl.sum(axis=1, keepdims=True)
    sc = tbl.sum(axis=0, keepdims=True)
    E = sr @ sc / n
    yates = 0.0
    is_2x2 = correct and nr == 2 and nc == 2
    if is_2x2:
        yates = min(0.5, float(np.min(np.abs(tbl - E))))
    stat = float(np.sum((np.abs(tbl - E) - yates) ** 2 / E))
    df = (nr - 1) * (nc - 1)
    return stat, df, is_2x2 and yates > 0


def _chisq_table(tbl: np.ndarray, *, correct: bool, name: str) -> HTest:
    stat, df, yates = _chisq_stat(np.asarray(tbl, dtype=float), correct=correct)
    return HTest(
        method="Pearson's Chi-squared test"
        + (" with Yates' continuity correction" if yates else ""),
        statistic={"X-squared": stat},
        parameter={"df": df},
        p_value=float(_dist.pchisq(stat, df, lower_tail=False)),
        alternative="",
        data_name=name,
    )


def _crosstab(x, y) -> np.ndarray:
    """Build a 2-way contingency table from two 1-D vectors (utf8-cast).

    Internal columns use ``__x__`` / ``__y__`` so user data containing
    string values like ``"x"`` / ``"y"`` doesn't collide with the index
    column name once ``pivot`` spreads the levels of ``y`` into columns.
    """
    x_ser = pl.Series("__x__", x).cast(pl.Utf8)
    y_ser = pl.Series("__y__", y).cast(pl.Utf8)
    return (
        pl.DataFrame({"__x__": x_ser, "__y__": y_ser})
        .group_by(["__x__", "__y__"]).len()
        .pivot(values="len", index="__x__", on="__y__")
        .fill_null(0)
        .drop("__x__")
        .to_numpy()
    )


def fisher_test(
    x,
    y=None,
    *,
    alternative: str = "two.sided",
) -> HTest:
    """R's ``fisher.test`` — Fisher's exact test for a 2×2 contingency table.

    ``x`` may be a 2×2 array/matrix or a 1-D vector paired with ``y``.
    Larger tables (R's Monte-Carlo simulation branch) are not supported.
    Returns the odds ratio as the point estimate; CI is omitted (R uses
    inverse non-central hypergeometric, not yet wired).

    PARITY DEBT (blocked): faithful R parity needs the exact hypergeometric
    (``nmath/dhyper.c``) plus FEXACT for r×c tables and the non-central
    hypergeometric CI (``src/fexact.c``), none yet ported — so this defers
    to ``scipy.stats.fisher_exact`` (2×2 only). See r-stats-parity-debt.md §2.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    if y is not None:
        tbl = _crosstab(x, y)
        name = "x and y"
    else:
        tbl = np.asarray(x)
        name = "x"
    if tbl.shape != (2, 2):
        raise NotImplementedError(
            f"fisher_test(): only 2x2 tables supported (got {tbl.shape})"
        )
    res = _sps.fisher_exact(tbl, alternative=alt)
    odds = float(res.statistic)
    return HTest(
        method="Fisher's Exact Test for Count Data",
        p_value=float(res.pvalue),
        estimate={"odds ratio": odds},
        null_value=1.0,
        alternative=alternative,
        data_name=name,
    )


def prop_test(
    x,
    n=None,
    *,
    p=None,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    correct: bool = True,
) -> HTest:
    """R's ``prop.test`` — chi-squared test on proportions.

    Supports 1-sample (``length(x)==1``, requires ``n``; ``p`` defaults
    to 0.5) and 2-sample equality (``length(x)==2``, ``p=None``). The
    k-sample (k > 2) and ``p`` vectors with ``length>1`` are not yet
    wired.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=int))
    if n is None:
        raise ValueError("prop_test(): n must be provided")
    n_arr = np.atleast_1d(np.asarray(n, dtype=int))
    if x_arr.shape != n_arr.shape:
        raise ValueError("prop_test(): x and n must have the same length")
    k = len(x_arr)
    estimates = {f"prop {i+1}": float(x_arr[i] / n_arr[i]) for i in range(k)}

    if k == 1:
        p_null = 0.5 if p is None else float(np.asarray(p))
        x0, n0 = int(x_arr[0]), int(n_arr[0])
        diff = abs(x0 / n0 - p_null)
        if correct:
            diff = max(diff - 0.5 / n0, 0.0)
        if p_null in (0.0, 1.0):
            stat = float("inf") if diff > 0 else 0.0
        else:
            stat = (diff ** 2) / (p_null * (1 - p_null) / n0)
        df = 1
        pval = float(_dist.pchisq(stat, df, lower_tail=False))
        return HTest(
            method="1-sample test for given proportion"
            + (" with continuity correction" if correct else ""),
            statistic={"X-squared": stat},
            parameter={"df": df},
            p_value=pval,
            estimate={"p": x0 / n0},
            null_value=p_null,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and n",
        )
    if k >= 2 and p is None:
        # k-sample equality of proportions: (k × 2) chi-squared. R applies
        # Yates' continuity correction only for the 2×2 case; for k > 2
        # the correction is silently dropped.
        tbl = np.array([
            [int(x_arr[i]), int(n_arr[i] - x_arr[i])]
            for i in range(k)
        ], dtype=float)
        use_correction = correct and k == 2
        stat, df, _ = _chisq_stat(tbl, correct=use_correction)
        suffix = " with continuity correction" if use_correction else ""
        if k == 2:
            method = f"2-sample test for equality of proportions{suffix}"
        else:
            method = f"{k}-sample test for equality of proportions{suffix}"
        return HTest(
            method=method,
            statistic={"X-squared": stat},
            parameter={"df": df},
            p_value=float(_dist.pchisq(stat, df, lower_tail=False)),
            estimate=estimates,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x out of n",
        )
    raise NotImplementedError(
        "prop_test(): k > 1 with explicit ``p`` (vector hypothesis) "
        "not yet wired"
    )


def binom_test(
    x,
    n=None,
    *,
    p: float = 0.5,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``binom.test`` — exact binomial test for one proportion.

    ``x`` is the success count, or a length-2 ``(successes, failures)``
    vector. ``n`` is the total trials (omitted when ``x`` already has
    both counts).
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"binom_test(): unknown alternative {alternative!r}")
    if n is None:
        x_arr = np.asarray(x, dtype=int)
        if x_arr.shape != (2,):
            raise ValueError(
                "binom_test(): n must be provided unless x = (succ, fail)"
            )
        x_succ = int(x_arr[0])
        n = int(x_arr.sum())
    else:
        x_succ = int(x)
        n = int(n)
    p = float(p)

    # p-value — faithful stats:::binom.test (exact, nmath dbinom/pbinom).
    if alternative == "less":
        pval = float(_dist.pbinom(x_succ, n, p))
    elif alternative == "greater":
        pval = float(_dist.pbinom(x_succ - 1, n, p, lower_tail=False))
    elif p == 0.0:
        pval = 1.0 if x_succ == 0 else 0.0
    elif p == 1.0:
        pval = 1.0 if x_succ == n else 0.0
    else:
        rel_err = 1.0 + 1e-7
        d = float(_dist.dbinom(x_succ, n, p))
        m = n * p
        if x_succ == m:
            pval = 1.0
        elif x_succ < m:
            i = np.arange(math.ceil(m), n + 1)
            yy = int(np.sum(np.asarray(_dist.dbinom(i, n, p)) <= d * rel_err))
            pval = float(_dist.pbinom(x_succ, n, p)
                         + _dist.pbinom(n - yy, n, p, lower_tail=False))
        else:
            i = np.arange(0, math.floor(m) + 1)
            yy = int(np.sum(np.asarray(_dist.dbinom(i, n, p)) <= d * rel_err))
            pval = float(_dist.pbinom(yy - 1, n, p)
                         + _dist.pbinom(x_succ - 1, n, p, lower_tail=False))

    # Clopper-Pearson CI via qbeta.
    def p_L(a):
        return 0.0 if x_succ == 0 else float(_dist.qbeta(a, x_succ, n - x_succ + 1))

    def p_U(a):
        return 1.0 if x_succ == n else float(_dist.qbeta(1 - a, x_succ + 1, n - x_succ))

    if alternative == "less":
        conf_int = (0.0, p_U(1 - conf_level))
    elif alternative == "greater":
        conf_int = (p_L(1 - conf_level), 1.0)
    else:
        a = (1 - conf_level) / 2
        conf_int = (p_L(a), p_U(a))

    return HTest(
        method="Exact binomial test",
        statistic={"number of successes": x_succ},
        parameter={"number of trials": n},
        p_value=float(pval),
        conf_int=conf_int,
        estimate={"probability of success": x_succ / n},
        null_value=p,
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and n",
    )


def var_test(
    x,
    y,
    *,
    ratio: float = 1.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``var.test`` — F-test for equal variances of two samples.

    ``F = (var(x) / var(y)) / ratio``; df = ``(n_x - 1, n_y - 1)``.
    CI is for the variance ratio at the requested confidence level.
    """
    x_arr = _as_array(x)
    y_arr = _as_array(y)
    n1, n2 = len(x_arr), len(y_arr)
    df1, df2 = n1 - 1, n2 - 1
    var_x = float(np.var(x_arr, ddof=1))
    var_y = float(np.var(y_arr, ddof=1))
    if var_y <= 0:
        raise ValueError("var_test(): var(y) must be positive")
    F = (var_x / var_y) / float(ratio)

    if alternative == "two.sided":
        p = 2 * min(_dist.pf(F, df1, df2), _dist.pf(F, df1, df2, lower_tail=False))
    elif alternative == "less":
        p = float(_dist.pf(F, df1, df2))
    elif alternative == "greater":
        p = float(_dist.pf(F, df1, df2, lower_tail=False))
    else:
        raise ValueError(f"var_test(): unknown alternative {alternative!r}")

    alpha = 1 - conf_level
    if alternative == "two.sided":
        lo = F / _dist.qf(1 - alpha / 2, df1, df2)
        hi = F / _dist.qf(alpha / 2, df1, df2)
    elif alternative == "less":
        lo = 0.0
        hi = F / _dist.qf(alpha, df1, df2)
    else:  # greater
        lo = F / _dist.qf(1 - alpha, df1, df2)
        hi = float("inf")

    return HTest(
        method="F test to compare two variances",
        statistic={"F": F},
        parameter={"num df": df1, "denom df": df2},
        p_value=float(p),
        conf_int=(float(lo), float(hi)),
        estimate={"ratio of variances": var_x / var_y},
        null_value=float(ratio),
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y",
    )


def bartlett_test(x, g) -> HTest:
    """R's ``bartlett.test(x, g)`` — Bartlett's test for equal variances.

    ``x`` is the values vector; ``g`` is the parallel group label vector.
    Returns the K² statistic with ``k - 1`` degrees of freedom.
    """
    x_arr = _as_array(x)
    g_arr = np.asarray(g)
    if x_arr.shape != g_arr.shape:
        raise ValueError("bartlett_test(): x and g must have the same length")
    groups = [x_arr[g_arr == val] for val in np.unique(g_arr)]
    k = len(groups)
    if k < 2:
        raise ValueError("bartlett_test(): need at least 2 groups")
    ni = np.array([len(gr) - 1 for gr in groups], dtype=float)  # n_i - 1
    if np.any(ni <= 0):
        raise ValueError("there must be at least 2 observations in each group")
    vi = np.array([np.var(gr, ddof=1) for gr in groups], dtype=float)
    n_total = float(ni.sum())
    v_total = float(np.sum(ni * vi) / n_total)
    # stats:::bartlett.test.default
    stat = ((n_total * math.log(v_total) - np.sum(ni * np.log(vi)))
            / (1.0 + (np.sum(1.0 / ni) - 1.0 / n_total) / (3.0 * (k - 1))))
    return HTest(
        method="Bartlett test of homogeneity of variances",
        statistic={"Bartlett's K-squared": float(stat)},
        parameter={"df": int(k - 1)},
        p_value=float(_dist.pchisq(stat, k - 1, lower_tail=False)),
        alternative="",
        data_name="x by g",
    )


def shapiro_test(x) -> HTest:
    """R's ``shapiro.test`` — Shapiro-Wilk normality test.

    PARITY DEBT (blocked): defers to ``scipy.stats.shapiro``. R's Royston
    AS R94 kernel (``src/swilk.c``) is not yet ported. See r-stats-parity-debt.md §2.
    """
    x_arr = _as_array(x)
    res = _sps.shapiro(x_arr)
    return HTest(
        method="Shapiro-Wilk normality test",
        statistic={"W": float(res.statistic)},
        p_value=float(res.pvalue),
        alternative="",
        data_name="x",
    )


def ks_test(
    x,
    y,
    *,
    alternative: str = "two.sided",
) -> HTest:
    """R's ``ks.test`` — Kolmogorov-Smirnov test.

    ``y`` is either a second sample (two-sample test) or a string naming
    a scipy distribution (one-sample goodness-of-fit). R uses names like
    ``"pnorm"``; we accept either ``"pnorm"`` or scipy's ``"norm"``.

    PARITY DEBT (blocked): defers to ``scipy.stats.kstest`` / ``ks_2samp``
    (asymptotic). R's exact small-n Smirnov / Kolmogorov distributions
    (``src/ks.c`` — ``psmirnov``/``pkolmogorov``) are not yet ported. See
    r-stats-parity-debt.md §2.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x_arr = _as_array(x)
    if isinstance(y, str):
        dist_name = y[1:] if y.startswith("p") and len(y) > 1 else y
        res = _sps.kstest(x_arr, dist_name, alternative=alt)
        method = "One-sample Kolmogorov-Smirnov test"
        data_name = "x"
    else:
        y_arr = _as_array(y)
        res = _sps.ks_2samp(x_arr, y_arr, alternative=alt)
        method = "Two-sample Kolmogorov-Smirnov test"
        data_name = "x and y"
    return HTest(
        method=method,
        statistic={"D": float(res.statistic)},
        p_value=float(res.pvalue),
        alternative=alternative,
        data_name=data_name,
    )


def mcnemar_test(x, y=None, *, correct: bool = True) -> HTest:
    """R's ``mcnemar.test`` — McNemar's chi-squared test on a 2×2 table.

    ``x`` is the 2×2 table or a 1-D vector paired with ``y``. With
    ``correct=True`` (R's default), uses the Yates continuity correction
    ``(|b - c| - 1)² / (b + c)``.
    """
    if y is not None:
        tbl = _crosstab(x, y)
    else:
        tbl = np.asarray(x)
    if tbl.shape != (2, 2):
        raise ValueError(
            f"mcnemar_test(): table must be 2x2 (got {tbl.shape})"
        )
    b = float(tbl[0, 1])
    c = float(tbl[1, 0])
    if b + c == 0:
        stat = 0.0
    elif correct:
        diff = max(abs(b - c) - 1, 0.0)
        stat = diff ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)
    pval = float(_dist.pchisq(stat, 1, lower_tail=False))
    return HTest(
        method="McNemar's Chi-squared test"
        + (" with continuity correction" if correct else ""),
        statistic={"McNemar's chi-squared": stat},
        parameter={"df": 1},
        p_value=pval,
        alternative="",
        data_name="x" if y is None else "x and y",
    )


def friedman_test(y, groups, blocks) -> HTest:
    """R's ``friedman.test(y, groups, blocks)`` — Friedman rank-sum test.

    ``y`` is the value vector, ``groups`` and ``blocks`` are parallel
    label vectors. The data is reshaped into ``(blocks × groups)`` wide
    form before being passed to ``scipy.stats.friedmanchisquare``.
    """
    y_arr = _as_array(y)
    g_arr = np.asarray(groups)
    b_arr = np.asarray(blocks)
    if not (y_arr.shape == g_arr.shape == b_arr.shape):
        raise ValueError(
            "friedman_test(): y, groups, blocks must have the same length"
        )
    # Internal column names use ``__y__`` / ``__g__`` / ``__b__`` so user
    # group/block labels equal to "y" or "g" or "b" don't collide with
    # the temp column names after pivot.
    df = pl.DataFrame({
        "__y__": y_arr,
        "__g__": pl.Series(g_arr).cast(pl.Utf8),
        "__b__": pl.Series(b_arr).cast(pl.Utf8),
    })
    wide = df.pivot(values="__y__", index="__b__", on="__g__")
    cols = [c for c in wide.columns if c != "__b__"]
    # (blocks × groups) matrix; rank within each block-row (R: t(apply(y,1,rank))).
    mat = np.column_stack([wide[c].to_numpy().astype(float) for c in cols])
    n, k = mat.shape
    r = np.vstack([_avg_rank(row) for row in mat])
    # tie correction: sum over rows of sum(t^3 - t) for tied rank groups.
    tie_sum = 0.0
    for row in mat:
        _, cnt = np.unique(row, return_counts=True)
        cnt = cnt.astype(float)
        tie_sum += float(np.sum(cnt ** 3 - cnt))
    stat = (12.0 * np.sum((r.sum(axis=0) - n * (k + 1) / 2.0) ** 2)
            / (n * k * (k + 1) - tie_sum / (k - 1)))
    return HTest(
        method="Friedman rank sum test",
        statistic={"Friedman chi-squared": float(stat)},
        parameter={"df": int(k - 1)},
        p_value=float(_dist.pchisq(stat, k - 1, lower_tail=False)),
        alternative="",
        data_name="y, groups and blocks",
    )


def aov(formula: str, data: pl.DataFrame, *, type: str = "II") -> AnovaTable:
    """R's ``aov`` followed by ``car::Anova(..., type='II')``.

    Computes Type-II sums of squares by dropping one top-level term at a
    time and comparing ``ΔRSS``. Works for either form the notebook uses:
    factor formulas (``value ~ group``) or explicit-dummy formulas
    (``value ~ 1 + group_b + group_c``) — both go through ``hea.lm``,
    so the term grouping comes from the formula's own ``term_labels``.
    """
    fit = lm(formula, data)
    term_labels = list(fit._expanded.term_labels)
    rss_full = float(fit.rss)
    df_full = int(fit.df_residuals)

    lhs = formula.split("~", 1)[0]
    rows = []
    for term in term_labels:
        kept = [t for t in term_labels if t != term]
        reduced_rhs = " + ".join(["1"] + kept) if kept else "1"
        sub_formula = f"{lhs} ~ {reduced_rhs}"
        sub = lm(sub_formula, data)
        ss = float(sub.rss - rss_full)
        df_term = int(sub.df_residuals - df_full)
        F = (ss / df_term) / (rss_full / df_full) if df_term > 0 else None
        p = float(_dist.pf(F, df_term, df_full, lower_tail=False)) if F is not None else None
        rows.append(
            {
                "term": term,
                "df": df_term,
                "sum_sq": ss,
                "mean_sq": ss / df_term if df_term else None,
                "F": F,
                "p": p,
            }
        )
    return AnovaTable(
        response=fit.y.name,
        rows=rows,
        residual_df=df_full,
        residual_ss=rss_full,
        type=type,
    )
