"""R's hypothesis-test family plus the ``htest`` / ``Anova`` print
containers and the rank-based helpers used by Lindeløv-style "tests as
lm" notebook constructions.

Every test function returns an :class:`HTest`; :func:`aov` returns an
:class:`AnovaTable`. R parameter conventions are kept: ``alternative`` ∈
{"two.sided", "greater", "less"}, ``conf_level=0.95``, ``correct=``
(continuity correction) where applicable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy import stats as _sps

from ._shared import _as_array, _fmt, _fmt_pval
from ..models.lm import lm


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
    return _sps.rankdata(_as_array(x), method="average")


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
    return np.sign(arr) * _sps.rankdata(np.abs(arr), method="average")


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
    var_equal: bool = True,
    mu: float = 0.0,
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``t.test``.

    - ``y=None``                     → one-sample t-test on ``x``.
    - ``y`` given, ``paired=True``   → paired t-test on ``x - y``.
    - ``y`` given, ``var_equal=True``→ Student's two-sample (pooled var).
    - ``y`` given, ``var_equal=False``→ Welch's two-sample.

    Always uses Student-t CI on the appropriate df.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    if y is None:
        res = _sps.ttest_1samp(x, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="One Sample t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of x": float(np.mean(x))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x",
        )
    y = _as_array(y)
    if paired:
        d = x - y
        res = _sps.ttest_1samp(d, mu, alternative=alt)
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Paired t-test",
            statistic={"t": float(res.statistic)},
            parameter={"df": float(res.df)},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"mean of the differences": float(np.mean(d))},
            null_value=mu,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    res = _sps.ttest_ind(x, y, equal_var=var_equal, alternative=alt)
    ci = res.confidence_interval(conf_level)
    method = "Two Sample t-test" if var_equal else "Welch Two Sample t-test"
    return HTest(
        method=method,
        statistic={"t": float(res.statistic)},
        parameter={"df": float(res.df)},
        p_value=float(res.pvalue),
        conf_int=(float(ci.low), float(ci.high)),
        estimate={"mean of x": float(np.mean(x)), "mean of y": float(np.mean(y))},
        null_value=mu,
        alternative=alternative,
        conf_level=conf_level,
        data_name="x and y",
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


def cor_test(
    x,
    y,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
) -> HTest:
    """R's ``cor.test`` with ``method`` in {pearson, spearman, kendall}.

    For Pearson, we report ``t``, df = n-2, and Fisher-z CI. Spearman
    reports ``S`` (rank-sum statistic R's ``cor.test`` shows). Kendall
    reports ``z``.
    """
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    x = _as_array(x)
    y = _as_array(y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    n = len(x)
    if method == "pearson":
        res = _sps.pearsonr(x, y, alternative=alt)
        r = float(res.statistic)
        df = n - 2
        t = r * np.sqrt(df / max(1 - r * r, 1e-300))
        ci = res.confidence_interval(conf_level)
        return HTest(
            method="Pearson's product-moment correlation",
            statistic={"t": t},
            parameter={"df": df},
            p_value=float(res.pvalue),
            conf_int=(float(ci.low), float(ci.high)),
            estimate={"cor": r},
            null_value=0.0,
            alternative=alternative,
            conf_level=conf_level,
            data_name="x and y",
        )
    if method == "spearman":
        res = _sps.spearmanr(x, y, alternative=alt)
        rho = float(res.statistic)
        # R reports S = sum((rank(x) - rank(y))^2) for the Spearman test
        S = float(np.sum((_sps.rankdata(x) - _sps.rankdata(y)) ** 2))
        return HTest(
            method="Spearman's rank correlation rho",
            statistic={"S": S},
            p_value=float(res.pvalue),
            estimate={"rho": rho},
            null_value=0.0,
            alternative=alternative,
            data_name="x and y",
        )
    if method == "kendall":
        res = _sps.kendalltau(x, y, alternative=alt)
        return HTest(
            method="Kendall's rank correlation tau",
            statistic={"z": float(res.statistic)},
            p_value=float(res.pvalue),
            estimate={"tau": float(res.statistic)},
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
    groups = [
        data.filter(pl.col(rhs) == g)[lhs].to_numpy().astype(float)
        for g in data[rhs].unique().to_list()
    ]
    res = _sps.kruskal(*groups)
    return HTest(
        method="Kruskal-Wallis rank sum test",
        statistic={"Kruskal-Wallis chi-squared": float(res.statistic)},
        parameter={"df": int(len(groups) - 1)},
        p_value=float(res.pvalue),
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
    pval = float(_sps.chi2.sf(stat, df))
    return HTest(
        method="Chi-squared test for given probabilities",
        statistic={"X-squared": stat},
        parameter={"df": df},
        p_value=pval,
        alternative="",
        data_name="x",
    )


def _chisq_table(tbl: np.ndarray, *, correct: bool, name: str) -> HTest:
    res = _sps.chi2_contingency(tbl, correction=(correct and tbl.shape == (2, 2)))
    return HTest(
        method="Pearson's Chi-squared test"
        + (" with Yates' continuity correction" if (correct and tbl.shape == (2, 2)) else ""),
        statistic={"X-squared": float(res.statistic)},
        parameter={"df": int(res.dof)},
        p_value=float(res.pvalue),
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
        pval = float(_sps.chi2.sf(stat, df))
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
        ])
        use_correction = correct and k == 2
        res = _sps.chi2_contingency(tbl, correction=use_correction)
        suffix = " with continuity correction" if use_correction else ""
        if k == 2:
            method = f"2-sample test for equality of proportions{suffix}"
        else:
            method = f"{k}-sample test for equality of proportions{suffix}"
        return HTest(
            method=method,
            statistic={"X-squared": float(res.statistic)},
            parameter={"df": int(res.dof)},
            p_value=float(res.pvalue),
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
    alt = {"two.sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
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
    res = _sps.binomtest(x_succ, n, float(p), alternative=alt)
    ci = res.proportion_ci(confidence_level=conf_level, method="exact")
    return HTest(
        method="Exact binomial test",
        statistic={"number of successes": x_succ},
        parameter={"number of trials": n},
        p_value=float(res.pvalue),
        conf_int=(float(ci.low), float(ci.high)),
        estimate={"probability of success": x_succ / n},
        null_value=float(p),
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
        p = 2 * min(_sps.f.cdf(F, df1, df2), _sps.f.sf(F, df1, df2))
    elif alternative == "less":
        p = float(_sps.f.cdf(F, df1, df2))
    elif alternative == "greater":
        p = float(_sps.f.sf(F, df1, df2))
    else:
        raise ValueError(f"var_test(): unknown alternative {alternative!r}")

    alpha = 1 - conf_level
    if alternative == "two.sided":
        lo = F / _sps.f.ppf(1 - alpha / 2, df1, df2)
        hi = F / _sps.f.ppf(alpha / 2, df1, df2)
    elif alternative == "less":
        lo = 0.0
        hi = F / _sps.f.ppf(alpha, df1, df2)
    else:  # greater
        lo = F / _sps.f.ppf(1 - alpha, df1, df2)
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
    if len(groups) < 2:
        raise ValueError("bartlett_test(): need at least 2 groups")
    res = _sps.bartlett(*groups)
    return HTest(
        method="Bartlett test of homogeneity of variances",
        statistic={"Bartlett's K-squared": float(res.statistic)},
        parameter={"df": int(len(groups) - 1)},
        p_value=float(res.pvalue),
        alternative="",
        data_name="x by g",
    )


def shapiro_test(x) -> HTest:
    """R's ``shapiro.test`` — Shapiro-Wilk normality test."""
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
    pval = float(_sps.chi2.sf(stat, 1))
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
    samples = [wide[c].to_numpy().astype(float) for c in cols]
    res = _sps.friedmanchisquare(*samples)
    return HTest(
        method="Friedman rank sum test",
        statistic={"Friedman chi-squared": float(res.statistic)},
        parameter={"df": int(len(samples) - 1)},
        p_value=float(res.pvalue),
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
        p = float(_sps.f.sf(F, df_term, df_full)) if F is not None else None
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
