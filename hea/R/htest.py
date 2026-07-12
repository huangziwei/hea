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
from . import _fexact
from . import distributions as _dist
from . import nmath as _nm

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


p_adjust_methods = ("holm", "hochberg", "hommel", "bonferroni",
                    "BH", "BY", "fdr", "none")


def p_adjust(p, method: str = "holm", n: Optional[int] = None):
    """Adjust p-values for multiple comparisons — R's ``p.adjust``.

    Faithful port of ``stats::p.adjust`` (``p.adjust.R``). ``method`` is
    one of :data:`p_adjust_methods` (``"fdr"`` is an alias for ``"BH"``);
    ``n`` is the number of comparisons and defaults to the count of
    non-``NaN`` entries in ``p`` (matching R's lazy-evaluated
    ``n = length(p)`` after ``NA`` removal). ``NaN`` p-values pass through
    unchanged. Returns an :class:`numpy.ndarray`.
    """
    if method not in p_adjust_methods:
        raise ValueError(
            f"p_adjust: 'method' must be one of {p_adjust_methods!r}")
    if method == "fdr":  # back compatibility
        method = "BH"
    p0 = np.asarray(p, dtype=float).ravel().copy()  # output holder
    nna = ~np.isnan(p0)
    all_nna = bool(nna.all())
    pv = p0 if all_nna else p0[nna]
    lp = pv.size
    if n is None:
        n = lp
    if n < lp:
        raise ValueError("p_adjust: n >= length(p) is not TRUE")
    if n <= 1:
        return p0
    if n == 2 and method == "hommel":
        method = "hochberg"

    if method == "bonferroni":
        out = np.minimum(1.0, n * pv)
    elif method == "holm":
        i = np.arange(1, lp + 1)
        o = np.argsort(pv, kind="stable")
        ro = np.argsort(o)
        out = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * pv[o]))[ro]
    elif method == "hommel":
        if n > lp:
            pv = np.concatenate([pv, np.ones(n - lp)])
        i = np.arange(1, n + 1)
        o = np.argsort(pv, kind="stable")
        pv = pv[o]
        ro = np.argsort(o)
        qval = np.min(n * pv / i)
        q = np.full(n, qval)
        pa = q.copy()
        for j in range(n - 1, 1, -1):
            ij = np.arange(0, n - j + 1)
            i2 = np.arange(n - j + 1, n)
            q1 = np.min(j * pv[i2] / np.arange(2, j + 1))
            q[ij] = np.minimum(j * pv[ij], q1)
            q[i2] = q[n - j]
            pa = np.maximum(pa, q)
        res = np.maximum(pa, pv)
        out = res[ro[:lp]] if lp < n else res[ro]
    elif method == "hochberg":
        i = np.arange(lp, 0, -1)
        o = np.argsort(-pv, kind="stable")
        ro = np.argsort(o)
        out = np.minimum(1.0, np.minimum.accumulate((n + 1 - i) * pv[o]))[ro]
    elif method == "BH":
        i = np.arange(lp, 0, -1)
        o = np.argsort(-pv, kind="stable")
        ro = np.argsort(o)
        out = np.minimum(1.0, np.minimum.accumulate(n / i * pv[o]))[ro]
    elif method == "BY":
        i = np.arange(lp, 0, -1)
        o = np.argsort(-pv, kind="stable")
        ro = np.argsort(o)
        qsum = _rsum_ld(1.0 / np.arange(1, n + 1))  # R's sum() is LDOUBLE
        out = np.minimum(1.0, np.minimum.accumulate(qsum * n / i * pv[o]))[ro]
    else:  # "none"
        out = pv

    p0[nna] = out
    return p0


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


# --- Streitberg-Röhmel exact permutation densities (src/permdist.c) ----------
# Used for the exact Wilcoxon p-value in the presence of ties/zeroes, where the
# plain signrank/wilcox distributions no longer apply.

def _rsum_ld(arr) -> float:
    """R's ``sum()`` over a double vector: strict left-to-right accumulation in
    ``LDOUBLE`` (80-bit x87 on x86-64), which ``np.sum`` (pairwise) does not
    reproduce. Needed so the summed permutation-density tail is 0-ulp to R."""
    acc = np.longdouble(0.0)
    for v in np.asarray(arr, dtype=float):
        acc += v
    return float(acc)


def _rmean(arr) -> float:
    """R's ``mean.default`` / cov.c ``MEAN``: two-pass LDOUBLE mean.

    First pass ``tmp = Σx / n`` (LDOUBLE); second pass adds the LDOUBLE
    correction ``Σ(x − tmp) / n``; the result is truncated to ``double``.
    ``np.mean`` (pairwise, no correction) does not reproduce this."""
    a = np.asarray(arr, dtype=float)
    n = a.size
    s = np.longdouble(0.0)
    for v in a:
        s += v
    tmp = s / n
    if np.isfinite(np.float64(tmp)):
        s = np.longdouble(0.0)
        for v in a:
            s += v - tmp
        tmp = tmp + s / n
    return float(tmp)


def _rvar(arr) -> float:
    """R's ``var`` (cov.c self-covariance): ``Σ(x − x̄)² / (n − 1)`` accumulated
    in LDOUBLE, with ``x̄`` the ``double``-truncated two-pass :func:`_rmean`."""
    a = np.asarray(arr, dtype=float)
    n = a.size
    xbar = np.longdouble(_rmean(a))
    s = np.longdouble(0.0)
    for v in a:
        d = np.longdouble(v) - xbar
        s += d * d
    return float(s / (n - 1))


def _dpermdist1(scores) -> list:
    """One-sample permutation density (permdist.c ``dpermdist1``): density of
    the sum of the positive elements over all sign-flips, for integer
    ``scores``. Returns the density over ``0 .. sum(scores)``."""
    n = len(scores)
    sum_a = int(sum(scores))
    dH = [0.0] * (sum_a + 1)
    dH[0] = 1.0
    s_a = 0
    for k in range(n):
        sk = int(scores[k])
        s_a += sk
        for i in range(s_a, sk - 1, -1):
            dH[i] += dH[i - sk]
    msum = 0.0
    for i in range(sum_a + 1):
        msum += dH[i]
    return [v / msum for v in dH]


def _dpermdist2(scores, m: int) -> list:
    """Two-sample permutation density (permdist.c ``dpermdist2``): density of
    ``sum(scores[first m])`` over all splits, for sorted integer ``scores``.
    Returns the density over the support ``min(scores) .. sum(scores)``."""
    n = len(scores)
    sum_a = m
    sum_b = 0
    for i in range(n - sum_a, n):
        sum_b += int(scores[i])
    sum_bp1 = sum_b + 1
    dH = [0.0] * ((sum_a + 1) * sum_bp1)
    dH[0] = 1.0
    s_a = 0
    s_b = 0
    for k in range(n):
        sk = int(scores[k])
        s_a += 1
        s_b += sk
        min_b = min(sum_b, s_b)
        for i in range(min(sum_a, s_a), 0, -1):
            idx = i * sum_bp1
            idx2 = (i - 1) * sum_bp1 - sk
            for j in range(min_b, sk - 1, -1):
                dH[idx + j] += dH[idx2 + j]
    idx = sum_a * sum_bp1 + 1
    ret = [dH[idx + j] for j in range(sum_b)]
    msum = 0.0  # naive sequential double (C's `double msum`); not Python sum()
    for v in ret:
        msum += v
    return [v / msum for v in ret]


def _dsignrank_z(s, z) -> np.ndarray:
    """R's internal ``.dsignrank(s, n, z)`` — signed-rank density on support
    ``s`` given the rank vector ``z`` (ties/zeroes present)."""
    z = np.asarray(z, dtype=float)
    f = 2 - int(np.all(z == np.floor(z)))  # 1 if integer ranks, else 2
    scores = np.sort((f * z).astype(np.int64)).tolist()
    d = _dpermdist1(scores)
    out = np.zeros(len(s))
    xv = f * np.asarray(s, dtype=float)
    for idx, val in enumerate(xv):
        iv = int(round(val))
        if abs(val - iv) < 1e-9 and 0 <= iv < len(d):
            out[idx] = d[iv]
    return out


def _psignrank_z(q, n, z, lower_tail=True) -> float:
    """R's internal ``.psignrank(q, n, z)`` — signed-rank CDF with ties."""
    if z is None:
        return float(_nm.psignrank(q, n, lower_tail, False))
    if np.all(z == np.floor(z)):
        s = np.arange(0, int(n * (n + 1) / 2) + 1, dtype=float)
    else:
        s = np.arange(0, int(n * (n + 1)) + 1, dtype=float) / 2.0
    d = _dsignrank_z(s, z)
    y = _rsum_ld(d[s < q + 1e-8])
    return y if lower_tail else 1.0 - y


def _dwilcox_z(s, m, n, z) -> np.ndarray:
    """R's internal ``.dwilcox(s, m, n, z)`` — rank-sum density with ties."""
    z = np.asarray(z, dtype=float)
    f = 2 - int(np.all(z == np.floor(z)))
    scores = np.sort((f * z).astype(np.int64)).tolist()
    d = _dpermdist2(scores, int(m))
    out = np.zeros(len(s))
    xv = f * (np.asarray(s, dtype=float) + m * (m + 1) / 2.0)
    for idx, val in enumerate(xv):
        iv = int(round(val))
        if abs(val - iv) < 1e-9 and 1 <= iv <= len(d):
            out[idx] = d[iv - 1]
    return out


def _pwilcox_z(q, m, n, z, lower_tail=True) -> float:
    """R's internal ``.pwilcox(q, m, n, z)`` — rank-sum CDF with ties."""
    if z is None:
        return float(_nm.pwilcox(q, m, n, lower_tail, False))
    if np.all(z == np.floor(z)):
        s = np.arange(0, int(m * n) + 1, dtype=float)
    else:
        s = np.arange(0, int(2 * m * n) + 1, dtype=float) / 2.0
    d = _dwilcox_z(s, m, n, z)
    y = _rsum_ld(d[s < q + 1e-8])
    return y if lower_tail else 1.0 - y


def _wilcox_correct_level(correct) -> int:
    """R's ``correct`` normalization: logical TRUE→0 / FALSE→-1 (continuity
    on / off); an integer 0..3 selects the Edgeworth-expansion order."""
    if isinstance(correct, bool):
        return (1 if correct else 0) - 1
    ic = int(correct)
    if ic not in (0, 1, 2, 3):
        raise ValueError("'correct' must be an integer between 0 and 3")
    return ic


def wilcox_test(
    x,
    y=None,
    *,
    mu: float = 0.0,
    paired: bool = False,
    alternative: str = "two.sided",
    exact: Optional[bool] = None,
    correct=True,
) -> HTest:
    """R's ``wilcox.test`` — Wilcoxon signed-rank / rank-sum test.

    Faithful port of ``wilcox.test.default``: statistic (``V`` signed-rank /
    ``W`` rank-sum) and p-value are bit-exact to R. The p-value uses R's default
    **exact** distribution for small samples (``exact = (n < 50)``, resp.
    ``n.x < 50 && n.y < 50``) via the ported nmath ``psignrank``/``pwilcox``;
    with ties or zeroes it uses the exact permutation distribution
    (Streitberg-Röhmel, ``src/permdist.c``). Otherwise it uses R's
    continuity-corrected normal approximation with the tie-corrected variance.

    ``correct`` follows R: ``True`` (default) applies the continuity correction,
    ``False`` disables it, and an integer 0..3 selects the Edgeworth-expansion
    order for the asymptotic tail.

    Note: ``conf.int`` (the Hodges-Lehmann estimate / exact CI) is not computed.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"wilcox_test(): unknown alternative {alternative!r}")
    icorrect = _wilcox_correct_level(correct)
    x = _as_array(x)

    if y is not None:
        y = _as_array(y)
        data_name = "x and y"
        if paired:
            if len(x) != len(y):
                raise ValueError("'x' and 'y' must have the same length")
            ok = ~np.isnan(x) & ~np.isnan(y)
            x = x[ok] - y[ok]
            y = None
        else:
            y = y[~np.isnan(y)]
    else:
        data_name = "x"
        if paired:
            raise ValueError("'y' is missing for paired test")
    x = x[~np.isnan(x)]
    if len(x) < 1:
        raise ValueError("not enough (non-missing) 'x' observations")

    if y is None:  # one-sample / paired: Wilcoxon signed-rank
        method = "Wilcoxon signed rank test"
        n = float(len(x))
        use_exact = (n < 50) if exact is None else exact
        xm = x - mu
        if use_exact:
            zero = np.any(xm == 0)
            r = _avg_rank(np.abs(xm))
            ties = len(np.unique(r)) != len(r)
            V = float(np.sum(r[xm > 0]))
            z = r[xm != 0] if (ties or zero) else None
            method = "Wilcoxon signed rank exact test"
            if alternative == "two.sided":
                m = (float(np.sum(z)) / 2.0) if z is not None else n * (n + 1) / 4
                p = (_psignrank_z(V - 0.25, n, z, lower_tail=False) if V > m
                     else _psignrank_z(V, n, z))
                pval = min(2.0 * p, 1.0)
            elif alternative == "greater":
                pval = _psignrank_z(V - 0.25, n, z, lower_tail=False)
            else:  # less
                pval = _psignrank_z(V, n, z)
        else:
            zero = np.any(xm == 0)
            if zero:
                xm = xm[xm != 0]
                n = float(len(xm))
            r = _avg_rank(np.abs(xm))
            ties = len(np.unique(r)) != len(r)
            V = float(np.sum(r[xm > 0]))
            mean = n * (n + 1) / 4.0
            nties = np.unique(r, return_counts=True)[1].astype(float)
            sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0
                              - np.sum(nties**3 - nties) / 48.0)
            ic = 0 if (icorrect > 0 and (ties or zero)) else icorrect
            pval = _wilcox_pval_asymp(V - mean, sigma, alternative, ic, n=n)
            if icorrect >= 0:
                method += " with continuity correction"
        return HTest(
            method=method,
            statistic={"V": V},
            p_value=float(pval),
            null_value=float(mu),
            alternative=alternative,
            data_name=data_name,
        )

    # two-sample: Wilcoxon rank-sum (Mann-Whitney)
    if len(y) < 1:
        raise ValueError("not enough 'y' observations")
    method = "Wilcoxon rank sum test"
    n_x = float(len(x))
    n_y = float(len(y))
    use_exact = ((n_x < 50) and (n_y < 50)) if exact is None else exact
    r = _avg_rank(np.concatenate([x - mu, y]))
    ties = len(np.unique(r)) != len(r)
    W = float(np.sum(r[:len(x)]) - n_x * (n_x + 1) / 2.0)
    if use_exact:
        z = r if ties else None
        method = "Wilcoxon rank sum exact test"
        if alternative == "two.sided":
            pval = min(2.0 * _pwilcox_z(W, n_x, n_y, z),
                       2.0 * _pwilcox_z(W - 0.25, n_x, n_y, z, lower_tail=False),
                       1.0)
        elif alternative == "greater":
            pval = _pwilcox_z(W - 0.25, n_x, n_y, z, lower_tail=False)
        else:  # less
            pval = _pwilcox_z(W, n_x, n_y, z)
    else:
        mean = n_x * n_y / 2.0
        nties = np.unique(r, return_counts=True)[1].astype(float)
        sigma = math.sqrt((n_x * n_y / 12.0)
                          * ((n_x + n_y + 1)
                             - np.sum(nties**3 - nties)
                             / ((n_x + n_y) * (n_x + n_y - 1))))
        ic = 0 if (icorrect > 0 and ties) else icorrect
        pval = _wilcox_pval_asymp(W - mean, sigma, alternative, ic,
                                  m=n_x, n=n_y)
        if icorrect >= 0:
            method += " with continuity correction"
    return HTest(
        method=method,
        statistic={"W": W},
        p_value=float(pval),
        null_value=float(mu),
        alternative=alternative,
        data_name=data_name,
    )


def _wilcox_pval_asymp(zdiff, sigma, alternative, icorrect, *, n, m=None):
    """R's ``.wilcox_test_*_pval_asymp`` tail: continuity-corrected normal, with
    the Fellingham-Stoker Edgeworth expansion when ``icorrect >= 1``.
    ``m`` distinguishes the two-sample (``m`` set) from one-sample kernel."""
    if icorrect >= 0:
        correction = {"two.sided": math.copysign(0.5, zdiff) if zdiff != 0 else 0.0,
                      "greater": 0.5, "less": -0.5}[alternative]
    else:
        correction = 0.0
    z = (zdiff - correction) / sigma

    def F(zz, lower_tail=True):
        y = float(_nm.pnorm5(zz, 0.0, 1.0, lower_tail, False))
        if icorrect < 1:
            return y
        if m is None:  # one-sample signed-rank Edgeworth
            n4 = 12 * (3 * n**2 + 3 * n - 1)
            d4 = 5 * n * (n + 1) * (2 * n + 1)
            l4 = -n4 / d4
            n6 = 576 * (3 * n**4 + 6 * n**2 - 3 * n + 1)
            d6 = 7 * (n * (n + 1) * (2 * n + 1))**2
            l6 = n6 / d6
        else:  # two-sample rank-sum Edgeworth
            n4 = m**2 + n**2 + m * n + m + n
            d4 = 20 * m * n * (m + n + 1)
            l4 = -n4 / d4
            n6 = (2 * (m**4 + n**4) + 4 * m * n * (m**2 + n**2)
                  + 6 * m**2 * n**2 + 4 * (m**3 + n**3)
                  + 7 * m * n * (m + n) + (m**2 + n**2) + 2 * m * n - (m + n))
            d6 = 210 * m**2 * n**2 * (m + n + 1)**2
            l6 = n6 / d6
        e = l4 / 24 * zz * (zz**2 - 3)
        if icorrect > 1:
            e += l6 / 720 * zz * (zz**4 - 10 * zz**2 + 15)
        if icorrect > 2:
            e += 35 * l4**2 / 40320 * zz * (zz**6 - 21 * zz**4
                                            + 105 * zz**2 - 105)
        return (y - e) if lower_tail else (y + e)

    if alternative == "less":
        return F(z)
    if alternative == "greater":
        return F(z, lower_tail=False)
    p = F(z)
    return 2.0 * min(p, 1.0 - p)


def _two_sided_min(lo, hi) -> float:
    """R's ``2 * min(p_lower, p_upper)`` two-sided rule, capped at 1."""
    return float(min(2.0 * min(lo, hi), 1.0))


# --- Exact Spearman (AS 89, src/prho.c) & Kendall (src/kendall.c) ------------
_PRHO_C = (.2274, .2531, .1745, .0758, .1033, .3932,
           .0879, .0151, .0072, .0831, .0131, 4.6e-4)


def _prho(n, is_, lower_tail):
    """AS 89 (src/prho.c) — Pr[S >= is] (or Pr[S < is] if ``lower_tail``) for the
    Spearman statistic S = (n^3-n)(1-rho)/6. Exact for n <= 9, Edgeworth else."""
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12) = _PRHO_C
    n_small = 9
    pv = 0.0 if lower_tail else 1.0
    if n <= 1:
        return pv
    if is_ <= 0.0:
        return pv
    n3 = float(n)
    n3 *= (n3 * n3 - 1.0) / 3.0  # (n^3 - n)/3
    if is_ > n3:
        return 1 - pv
    if n <= n_small:  # exact by full permutation enumeration
        import itertools
        nfac = math.factorial(n)
        if is_ == n3:
            ifr = 1
        else:
            ifr = 0
            for perm in itertools.permutations(range(1, n + 1)):
                ise = 0
                for i in range(n):
                    d = i + 1 - perm[i]
                    ise += d * d
                if is_ <= ise:
                    ifr += 1
        return (nfac - ifr if lower_tail else ifr) / nfac
    # Edgeworth series expansion
    y = float(n)
    b = 1 / y
    x = (6.0 * (is_ - 1) * b / (y * y - 1) - 1) * math.sqrt(y - 1)
    y = x * x
    u = x * b * (c1 + b * (c2 + c3 * b)
                 + y * (-c4 + b * (c5 + c6 * b)
                        - y * b * (c7 + c8 * b
                                   - y * (c9 - c10 * b + y * b * (c11 - c12 * y)))))
    y = u / math.exp(y / 2.0)
    pv = (-y if lower_tail else y) + _nm.pnorm5(x, 0.0, 1.0, lower_tail, False)
    if pv < 0:
        pv = 0.0
    if pv > 1:
        pv = 1.0
    return pv


def _ckendall(k, n, w):
    """Count permutations of size n with T = k concordant pairs (kendall.c)."""
    u = n * (n - 1) // 2
    if k < 0 or k > u:
        return 0.0
    if w[n] is None:
        w[n] = [-1.0] * (u + 1)
    if w[n][k] < 0:
        if n == 1:
            w[n][k] = 1.0 if k == 0 else 0.0
        else:
            s = 0.0
            for i in range(n):
                s += _ckendall(k - i, n - 1, w)
            w[n][k] = s
    return w[n][k]


def _pkendall(q, n):
    """R's ``pKendall`` (src/kendall.c) — P(T <= q) for Kendall's exact stat."""
    q = math.floor(q + 1e-7)
    if q < 0:
        return 0.0
    if q > n * (n - 1) // 2:
        return 1.0
    w = [None] * (n + 1)
    p = 0.0
    j = 0
    while j <= q:
        p += _ckendall(j, n, w)
        j += 1
    return p / _nm.gammafn(n + 1)


def cor_test(
    x,
    y,
    *,
    method: str = "pearson",
    alternative: str = "two.sided",
    conf_level: float = 0.95,
    continuity: bool = False,
    exact: Optional[bool] = None,
) -> HTest:
    """R's ``cor.test`` with ``method`` in {pearson, spearman, kendall}.

    - **pearson**: ``t``, df = n-2, Fisher-z CI (``pt`` / ``qnorm``).
    - **spearman**: statistic ``S``; R's default **exact** p-value (AS 89
      ``src/prho.c``) for n <= 1290 with no ties, else the asymptotic
      t-approximation via nmath ``pt``.
    - **kendall**: R's default **exact** p-value (``src/kendall.c``) for n < 50
      with no ties — statistic ``T`` (# concordant pairs); else the asymptotic
      normal (tie-corrected variance) with statistic ``z`` via nmath ``pnorm``.

    ``exact`` overrides R's default exact/asymptotic selection.

    The **exact-path p-values are bit-exact** to R (the ``round(S)`` / ``round(T)``
    feeding the exact kernels absorbs any last-ulp noise). The reported
    ``rho``/``S``/``tau`` estimate — and, in rare cases, the *asymptotic*
    p-value — may differ from R by <=1 ulp: R's ``cor`` (``src/cov.c``) centers
    with the system ``sqrtl``, which numpy's long-double sqrt does not reproduce.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"cor_test(): unknown alternative {alternative!r}")
    x = _as_array(x)
    y = _as_array(y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")
    n = len(x)
    ties = min(len(np.unique(x)), len(np.unique(y))) < n

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
        use_exact = True if exact is None else exact
        if ties and use_exact:
            use_exact = False  # cannot compute exact p-value with ties

        def pspearman(q, lower_tail):
            if n <= 1290 and use_exact:
                return float(_prho(n, round(q) + 2 * int(lower_tail), lower_tail))
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
        use_exact = (n < 50) if exact is None else exact
        if use_exact and not ties:
            q = round((tau + 1) * n * (n - 1) / 4)
            if alternative == "two.sided":
                if q > n * (n - 1) / 4:
                    p = 1 - _pkendall(q - 1, n)
                else:
                    p = _pkendall(q, n)
                pval = float(min(2.0 * p, 1.0))
            elif alternative == "greater":
                pval = float(1 - _pkendall(q - 1, n))
            else:  # less
                pval = float(_pkendall(q, n))
            return HTest(
                method="Kendall's rank correlation tau",
                statistic={"T": float(q)},
                p_value=pval,
                estimate={"tau": tau},
                null_value=0.0,
                alternative=alternative,
                data_name="x and y",
            )
        # asymptotic normal with tie-corrected variance
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
    uses. The numeric LHS is grouped by the RHS factor; the tie-corrected H
    statistic and its p-value (nmath ``pchisq``) match ``kruskal.test.R`` —
    bit-exact to R, no scipy.
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
    rescale_p: bool = False,
    simulate_p_value: bool = False,
    B: int = 2000,
) -> HTest:
    """R's ``chisq.test``.

    - 1-D ``x`` (and no ``y``)         → goodness-of-fit against ``p`` (uniform if None).
    - 2-D ``x`` (matrix or 2-D array)  → contingency-table test.
    - 1-D ``x`` and 1-D ``y``          → contingency on ``crosstab(x, y)``.

    ``simulate_p_value=True`` gives a Monte-Carlo p-value (``B`` replicates): for
    a table, random tables with fixed margins (rcont2); for goodness-of-fit,
    weighted ``sample.int``. Both draw from R's MT stream — bit-exact to
    ``set.seed(); chisq.test(..., simulate.p.value=TRUE)``.
    """
    arr = np.asarray(x)
    if y is not None:
        tbl = _crosstab(x, y)
        return _chisq_table(tbl, correct=correct, name="x and y",
                            simulate_p_value=simulate_p_value, B=B)
    if arr.ndim == 2:
        return _chisq_table(arr, correct=correct, name="x",
                            simulate_p_value=simulate_p_value, B=B)
    # goodness of fit
    counts = arr.astype(float)
    if p is None:
        p = np.full_like(counts, 1.0 / len(counts))
    p = np.asarray(p, dtype=float)
    if abs(p.sum() - 1.0) > math.sqrt(_nm._DBL_EPSILON):
        if rescale_p:
            p = p / p.sum()
        else:
            raise ValueError("probabilities must sum to 1.")
    total = counts.sum()
    expected = total * p
    stat = float(np.sum((counts - expected) ** 2 / expected))
    if simulate_p_value:
        # R: sm <- matrix(sample.int(nx, B*n, TRUE, prob=p), nrow=n); per column
        # ss <- sum((tabulate(col) - E)^2 / E); PVAL uses almost.1 * STATISTIC.
        nx = len(counts)
        nn = int(total)
        idx = _dist._r_rng().sample_prob(p, B * nn, replace=True)
        sm = np.asarray(idx).reshape((B, nn))          # each row = one replicate
        almost_1 = 1.0 - 64.0 * _nm._DBL_EPSILON
        ss = np.empty(B, dtype=float)
        for b in range(B):
            tab = np.bincount(sm[b], minlength=nx).astype(float)
            ss[b] = float(np.sum((tab - expected) ** 2 / expected))
        pval = (1.0 + float(np.count_nonzero(ss >= almost_1 * stat))) / (B + 1)
        return HTest(
            method="Chi-squared test for given probabilities with simulated "
            f"p-value\n\t (based on {B} replicates)",
            statistic={"X-squared": stat},
            parameter={},
            p_value=pval,
            alternative="",
            data_name="x",
        )
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


def _chisq_table(tbl: np.ndarray, *, correct: bool, name: str,
                 simulate_p_value: bool = False, B: int = 2000) -> HTest:
    tbl = np.asarray(tbl, dtype=float)
    nr, nc = tbl.shape
    n = tbl.sum()
    sr = tbl.sum(axis=1)
    sc = tbl.sum(axis=0)
    if simulate_p_value and np.all(sr > 0) and np.all(sc > 0):
        e = np.outer(sr, sc) / n
        # STATISTIC: sorted-descending LDOUBLE sum (R's PR#3486 idiom).
        resid2 = ((tbl - e) ** 2 / e).ravel()
        stat = _rsum_ld(np.sort(resid2)[::-1])
        tmp = _chisq_sim(sr, sc, int(B), e, _dist._r_rng())
        almost_1 = 1.0 - 64.0 * _nm._DBL_EPSILON
        pval = (1.0 + float(np.count_nonzero(tmp >= almost_1 * stat))) / (B + 1)
        return HTest(
            method="Pearson's Chi-squared test with simulated p-value\n\t "
            f"(based on {B} replicates)",
            statistic={"X-squared": float(stat)},
            parameter={},                           # df = NA for the MC test
            p_value=pval,
            alternative="",
            data_name=name,
        )
    stat, df, yates = _chisq_stat(tbl, correct=correct)
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


def _mc_fact(n):
    """Cumulative log-factorial ``fact[i] = fact[i-1] + log(i)`` — the exact
    array chisqsim.c/Smirnov_sim build (NOT ``lgammafn``)."""
    fact = [0.0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] + math.log(i)
    return fact


def _chisq_sim(sr, sc, B, expected, rng):
    """R's ``chisq_sim`` (chisqsim.c) — ``B`` Pearson X² statistics from random
    tables (rcont2) with the given margins. Column-major cell traversal and the
    cumulative ``fact`` match the C, so the stream is bit-exact to ``set.seed``."""
    sr = [int(v) for v in sr]
    sc = [int(v) for v in sc]
    n = int(sum(sr))
    fact = _mc_fact(n)
    e = np.asarray(expected, dtype=float)
    nrow = len(sr)
    ncol = len(sc)
    out = np.empty(B, dtype=float)
    for it in range(B):
        obs = rng.rcont2(sr, sc, n, fact)
        chisq = 0.0
        for j in range(ncol):                       # column-major, as C
            for i in range(nrow):
                ev = e[i, j]
                ov = obs[i][j]
                chisq += (ov - ev) * (ov - ev) / ev
        out[it] = chisq
    return out


def _fisher_sim(sr, sc, B, rng):
    """R's ``fisher_sim`` (chisqsim.c) — ``B`` log-probability statistics
    ``-sum(log(obs_ij!))`` from random tables (rcont2). Bit-exact stream."""
    sr = [int(v) for v in sr]
    sc = [int(v) for v in sc]
    n = int(sum(sr))
    fact = _mc_fact(n)
    nrow = len(sr)
    ncol = len(sc)
    out = np.empty(B, dtype=float)
    for it in range(B):
        obs = rng.rcont2(sr, sc, n, fact)
        ans = 0.0
        for j in range(ncol):                       # column-major, as C
            for i in range(nrow):
                ans -= fact[obs[i][j]]
        out[it] = ans
    return out


def _fisher_test_simulate(tbl, name, B):
    """R's ``fisher.test(..., simulate.p.value=TRUE)`` for an r×c table — the
    Monte-Carlo p-value via ``Fisher_sim`` (rcont2). Drops all-zero rows/cols,
    then compares each replicate's log-prob to the observed. Bit-exact stream."""
    tbl = np.asarray(tbl, dtype=float)
    sr = tbl.sum(axis=1)
    sc = tbl.sum(axis=0)
    x2 = tbl[sr > 0][:, sc > 0]                     # drop all-zero margins
    nr, nc = x2.shape
    if nr <= 1:
        raise ValueError("need 2 or more non-zero row marginals")
    if nc <= 1:
        raise ValueError("need 2 or more non-zero column marginals")
    # STATISTIC = -sum(lfactorial(x)) = -sum(lgamma(x+1)); R sums in LDOUBLE.
    stat = -_rsum_ld(np.array([_nm._lgammafn(float(v) + 1.0)
                               for v in x2.ravel()]))
    tmp = _fisher_sim(x2.sum(axis=1), x2.sum(axis=0), B, _dist._r_rng())
    almost_1 = 1.0 + 64.0 * _nm._DBL_EPSILON        # PR#10558: STATISTIC < 0
    pval = (1.0 + float(np.count_nonzero(tmp <= stat / almost_1))) / (B + 1)
    pval = max(0.0, min(1.0, pval))
    return HTest(
        method="Fisher's Exact Test for Count Data with simulated p-value"
        f"\n\t (based on {B} replicates)",
        statistic={},
        parameter={},
        p_value=pval,
        alternative="",
        data_name=name,
    )


def _fisher_test_exact(tbl, name, *, workspace, hybrid, hybrid_pars, mult):
    """R's ``fisher.test`` exact/hybrid p-value for an r×c table via the ported
    FEXACT network algorithm (:func:`hea.R._fexact.fexact`). ``expect < 0`` (the
    default) is the exact p-value; ``hybrid=True`` uses the asymptotic-χ² hybrid
    with ``hybrid_pars = (expect, percent, Emin)``. Bit-exact to R."""
    tbl = np.asarray(tbl, dtype=float)
    if tbl.ndim != 2 or min(tbl.shape) < 2:
        raise ValueError("'x' must have at least 2 rows and columns")
    if np.any(tbl < 0) or not np.all(np.isfinite(tbl)):
        raise ValueError("all entries of 'x' must be nonnegative and finite")
    xi = np.rint(tbl).astype(np.int64)          # R rounds to integer storage
    if np.any(xi > _fexact._INT_MAX):
        raise ValueError("'x' has entries too large to be integer")
    nr, nc = xi.shape
    table = xi.tolist()
    if hybrid:
        expect, percnt, emin = (float(hybrid_pars[0]), float(hybrid_pars[1]),
                                float(hybrid_pars[2]))
        method = ("Fisher's Exact Test for Count Data hybrid using "
                  "asym.chisq. iff (exp=%g, perc=%g, Emin=%g)"
                  % (expect, percnt, emin))
    else:
        expect, percnt, emin = -1.0, 100.0, 0.0
        method = "Fisher's Exact Test for Count Data"
    pval = _fexact.fexact(nr, nc, table, expect, percnt, emin,
                          workspace=int(workspace), mult=int(mult))
    pval = max(0.0, min(1.0, pval))
    return HTest(
        method=method,
        statistic={},
        parameter={},
        p_value=pval,
        alternative="",
        data_name=name,
    )


def fisher_test(
    x,
    y=None,
    *,
    alternative: str = "two.sided",
    or_: float = 1.0,
    conf_int: bool = True,
    conf_level: float = 0.95,
    simulate_p_value: bool = False,
    B: int = 2000,
    workspace: int = 200000,
    hybrid: bool = False,
    hybrid_pars: tuple = (5.0, 80.0, 1.0),
    mult: int = 30,
) -> HTest:
    """R's ``fisher.test`` — Fisher's exact test for contingency tables.

    Faithful port of the 2×2 branch of ``fisher.test.R``: the p-value (via the
    conditional non-central hypergeometric on the ported nmath ``dhyper`` /
    ``phyper``), the conditional-MLE odds ratio, and its confidence interval
    (inverting the non-central hypergeometric with the ported ``uniroot`` /
    Brent ``zeroin``) are all bit-exact to R. ``or_`` is R's ``or`` (the null
    odds ratio, default 1).

    ``x`` is a 2×2 array/matrix, or a 1-D vector paired with ``y``. For **r×c**
    tables the **exact** p-value is computed via the ported FEXACT network
    algorithm (R's ``src/fexact.c``; see :mod:`hea.R._fexact`) — bit-exact to
    ``fisher.test(x)``. ``hybrid=True`` selects R's hybrid asymptotic-χ²
    approximation (``hybrid_pars = (expect, percent, Emin)``); ``workspace`` and
    ``mult`` size the FEXACT hash tables exactly as R's. Alternatively
    ``simulate_p_value=True`` gives the Monte-Carlo p-value (``B`` replicates via
    ``Fisher_sim``/rcont2), bit-exact to ``set.seed(); fisher.test(...,``
    ``simulate.p.value=TRUE)`` — cheaper for large, dense tables.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"fisher_test(): unknown alternative {alternative!r}")
    if y is not None:
        tbl = np.asarray(_crosstab(x, y), dtype=float)
        name = "x and y"
    else:
        tbl = np.asarray(x, dtype=float)
        name = "x"
    if tbl.shape != (2, 2):
        if simulate_p_value:
            return _fisher_test_simulate(tbl, name, int(B))
        return _fisher_test_exact(
            tbl, name, workspace=int(workspace), hybrid=hybrid,
            hybrid_pars=hybrid_pars, mult=int(mult))

    m = tbl[0, 0] + tbl[1, 0]  # sum(x[, 1])
    n = tbl[0, 1] + tbl[1, 1]  # sum(x[, 2])
    k = tbl[0, 0] + tbl[0, 1]  # sum(x[1, ])
    xx = tbl[0, 0]
    lo = max(0.0, k - n)
    hi = min(k, m)
    support = np.arange(lo, hi + 1)
    logdc = np.array([_nm.dhyper(s, m, n, k, True) for s in support])

    def dnhyper(ncp):
        # R's sum() is a sequential LDOUBLE accumulation; np.sum is pairwise —
        # use _rsum_ld so the normalized density (and everything downstream:
        # p-value, MLE, CI via uniroot) is bit-exact to R.
        d = logdc + math.log(ncp) * support
        d = np.exp(d - np.max(d))
        return d / _rsum_ld(d)

    def mnhyper(ncp):
        if ncp == 0:
            return lo
        if ncp == math.inf:
            return hi
        return _rsum_ld(support * dnhyper(ncp))

    def pnhyper(q, ncp=1.0, upper_tail=False):
        if ncp == 1.0:
            if upper_tail:
                return float(_nm.phyper(xx - 1, m, n, k, False, False))
            return float(_nm.phyper(xx, m, n, k, True, False))
        if ncp == 0:
            return float(q <= lo) if upper_tail else float(q >= lo)
        if ncp == math.inf:
            return float(q <= hi) if upper_tail else float(q >= hi)
        d = dnhyper(ncp)
        mask = support >= q if upper_tail else support <= q
        return _rsum_ld(d[mask])

    eps = _nm._DBL_EPSILON
    if alternative == "less":
        pval = pnhyper(xx, or_)
    elif alternative == "greater":
        pval = pnhyper(xx, or_, upper_tail=True)
    elif or_ == 0:
        pval = float(xx == lo)
    elif or_ == math.inf:
        pval = float(xx == hi)
    else:
        rel_err = 1 + 1e-7  # a little fuzz
        d = dnhyper(or_)
        pval = _rsum_ld(d[d <= d[int(xx - lo)] * rel_err])

    def mle(val):
        if val == lo:
            return 0.0
        if val == hi:
            return math.inf
        mu = mnhyper(1.0)
        if mu > val:
            return _nm.uniroot(lambda t: mnhyper(t) - val, 0.0, 1.0)
        if mu < val:
            return 1.0 / _nm.uniroot(lambda t: mnhyper(1.0 / t) - val, eps, 1.0)
        return 1.0

    or_est = mle(xx)

    conf = None
    if conf_int:
        def ncp_u(val, alpha):
            if val == hi:
                return math.inf
            p = pnhyper(val, 1.0)
            if p < alpha:
                return _nm.uniroot(lambda t: pnhyper(val, t) - alpha, 0.0, 1.0)
            if p > alpha:
                return 1.0 / _nm.uniroot(
                    lambda t: pnhyper(val, 1.0 / t) - alpha, eps, 1.0)
            return 1.0

        def ncp_l(val, alpha):
            if val == lo:
                return 0.0
            p = pnhyper(val, 1.0, upper_tail=True)
            if p > alpha:
                return _nm.uniroot(
                    lambda t: pnhyper(val, t, upper_tail=True) - alpha, 0.0, 1.0)
            if p < alpha:
                return 1.0 / _nm.uniroot(
                    lambda t: pnhyper(val, 1.0 / t, upper_tail=True) - alpha,
                    eps, 1.0)
            return 1.0

        if alternative == "less":
            conf = (0.0, ncp_u(xx, 1 - conf_level))
        elif alternative == "greater":
            conf = (ncp_l(xx, 1 - conf_level), math.inf)
        else:
            a = (1 - conf_level) / 2
            conf = (ncp_l(xx, a), ncp_u(xx, a))

    return HTest(
        method="Fisher's Exact Test for Count Data",
        p_value=float(pval),
        conf_int=conf,
        estimate={"odds ratio": float(or_est)},
        null_value=float(or_),
        alternative=alternative,
        conf_level=conf_level,
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


def _swilk_poly(cc, nord: int, x: float) -> float:
    """AS 181.2 — algebraic polynomial of order ``nord-1`` (``src/swilk.c`` ``poly``).
    Zero-order coefficient is ``cc[0]``; Horner from the top down."""
    ret = cc[0]
    if nord > 1:
        p = x * cc[nord - 1]
        for j in range(nord - 2, 0, -1):
            p = (p + cc[j]) * x
        ret += p
    return ret


def _swilk(x: np.ndarray):
    """Royston (1995) AS R94 — Shapiro-Wilk W and its p-value.

    Line-by-line port of ``swilk()`` in R's ``src/library/stats/src/swilk.c``.
    ``x`` must be **sorted ascending** (the R wrapper sorts). Returns
    ``(W, pw, ifault)``;
    ``ifault == 7`` is R's benign "sort order" flag and is ignored by the caller.
    Uses the bit-exact ported ``qnorm5`` / ``pnorm5`` (nmath) for the normal
    scores and the tail probability, so W and p are 0-ulp to R's ``.Call(C_SWilk)``.
    """
    n = int(x.size)
    nn2 = n // 2
    a = [0.0] * (nn2 + 1)  # 1-based, as in the Fortran/C original

    small = 1e-19
    g = (-2.273, 0.459)
    c1 = (0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
    c2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
    c3 = (0.544, -0.39978, 0.025054, -6.714e-4)
    c4 = (1.3822, -0.77857, 0.062767, -0.0020322)
    c5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
    c6 = (-0.4803, -0.082676, 0.0030302)

    pw = 1.0
    if n < 3:
        return 0.0, pw, 1

    an = float(n)

    if n == 3:
        a[1] = 0.70710678  # = sqrt(1/2) (the literal R uses)
    else:
        an25 = an + 0.25
        summ2 = 0.0
        for i in range(1, nn2 + 1):
            a[i] = _nm.qnorm5((i - 0.375) / an25, 0.0, 1.0, True, False)
            summ2 += a[i] * a[i]
        summ2 *= 2.0
        ssumm2 = math.sqrt(summ2)
        rsn = 1.0 / math.sqrt(an)
        a1 = _swilk_poly(c1, 6, rsn) - a[1] / ssumm2

        # Normalize a[]
        if n > 5:
            i1 = 3
            a2 = -a[2] / ssumm2 + _swilk_poly(c2, 6, rsn)
            fac = math.sqrt((summ2 - 2.0 * (a[1] * a[1]) - 2.0 * (a[2] * a[2]))
                            / (1.0 - 2.0 * (a1 * a1) - 2.0 * (a2 * a2)))
            a[2] = a2
        else:
            i1 = 2
            fac = math.sqrt((summ2 - 2.0 * (a[1] * a[1]))
                            / (1.0 - 2.0 * (a1 * a1)))
        a[1] = a1
        for i in range(i1, nn2 + 1):
            a[i] /= -fac

    # Check for zero range
    rng = x[n - 1] - x[0]
    if rng < small:
        return 0.0, pw, 6

    # Check for correct sort order on range - scaled X
    ifault = 0
    xx = x[0] / rng
    sx = xx
    sa = -a[1]
    i = 1
    j = n - 1
    while i < n:
        xi = x[i] / rng
        if xx - xi > small:
            ifault = 7
        sx += xi
        i += 1
        if i != j:
            sa += (1.0 if i > j else -1.0) * a[min(i, j)]
        xx = xi
        j -= 1
    if n > 5000:
        ifault = 2

    # W statistic as squared correlation between data and coefficients
    sa /= n
    sx /= n
    ssa = ssx = sax = 0.0
    for i in range(n):
        j = n - 1 - i
        if i != j:
            asa = (1.0 if i > j else -1.0) * a[1 + min(i, j)] - sa
        else:
            asa = -sa
        xsx = x[i] / rng - sx
        ssa += asa * asa
        ssx += xsx * xsx
        sax += asa * xsx

    # W1 = 1-W, computed this way to avoid rounding error for W very near 1
    ssassx = math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1.0 - w1

    # Significance level for W
    if n == 3:  # exact P value
        pi6 = 1.90985931710274  # = 6/pi
        stqr = 1.04719755119660  # = asin(sqrt(3/4))
        pw = pi6 * (math.asin(math.sqrt(w)) - stqr)
        if pw < 0.0:
            pw = 0.0
        return w, pw, ifault
    y = math.log(w1)
    xx = math.log(an)
    if n <= 11:
        gamma = _swilk_poly(g, 2, an)
        if y >= gamma:
            return w, 1e-99, ifault
        y = -math.log(gamma - y)
        m = _swilk_poly(c3, 4, an)
        s = math.exp(_swilk_poly(c4, 4, an))
    else:  # n >= 12
        m = _swilk_poly(c5, 4, xx)
        s = math.exp(_swilk_poly(c6, 3, xx))

    pw = _nm.pnorm5(y, m, s, False, False)  # upper tail
    return w, pw, ifault


def shapiro_test(x) -> HTest:
    """R's ``shapiro.test`` — Shapiro-Wilk normality test.

    Faithful port of R's Royston AS R94 kernel (``src/swilk.c``) — W and its
    p-value are bit-exact to R's ``.Call(C_SWilk)`` (normal scores + tail via the
    ported nmath ``qnorm5``/``pnorm5``). The wrapper mirrors ``shapiro.test.R``:
    drop NAs, sort, require ``3 <= n <= 5000``, and rescale by the range when it
    is below ``1e-10`` (R's single-precision ``ifault=6`` guard).
    """
    x_arr = np.sort(_as_array(x)[~np.isnan(_as_array(x))])
    n = x_arr.size
    if n < 3 or n > 5000:
        raise ValueError("sample size must be between 3 and 5000")
    rng = x_arr[n - 1] - x_arr[0]
    if rng == 0:
        raise ValueError("all 'x' values are identical")
    if rng < 1e-10:
        x_arr = x_arr / rng  # rescale to avoid ifault=6 with single version
    w, pw, _ = _swilk(x_arr)
    return HTest(
        method="Shapiro-Wilk normality test",
        statistic={"W": float(w)},
        p_value=float(pw),
        alternative="",
        data_name="x",
    )


# --- Kolmogorov-Smirnov exact / asymptotic distributions (src/ks.c) ----------
_KS_M_PI_2 = math.pi / 2.0
_KS_M_PI_4 = math.pi / 4.0


def _R_pow_di(x, n: int) -> float:
    """R's ``R_pow_di(x, n)`` — x**n for integer n by repeated squaring."""
    p = 1.0
    if math.isnan(x):
        return x
    if n != 0:
        if not math.isfinite(x):
            return math.pow(x, float(n))
        is_neg = n < 0
        if is_neg:
            n = -n
        while True:
            if n & 1:
                p *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
        if is_neg:
            p = 1.0 / p
    return p


def _ks_K2l(x, lower, tol):
    """Two-sided asymptotic KS / Smirnov limit distribution (ks.c ``K2l``)."""
    if x <= 0.0:
        return 0.0 if lower else 1.0
    if x < 1.0:
        k_max = int(math.sqrt(2 - math.log(tol)))
        w = math.log(x)
        z = -(_KS_M_PI_2 * _KS_M_PI_4) / (x * x)
        s = 0.0
        k = 1
        while k < k_max:
            s += math.exp(k * k * z - w)
            k += 2
        p = s / _nm._M_1_SQRT_2PI
        if not lower:
            p = 1 - p
    else:
        z = -2 * x * x
        s = -1.0
        if lower:
            k = 1
            new = 1.0
        else:
            k = 2
            new = 2 * math.exp(z)
        old = 0.0
        while abs(old - new) > tol:
            old = new
            new += 2 * s * math.exp(z * k * k)
            s *= -1
            k += 1
        p = new
    return p


def _ks_m_multiply(A, B, m):
    C = [0.0] * (m * m)
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(m):
                s += A[i * m + k] * B[k * m + j]
            C[i * m + j] = s
    return C


def _ks_m_power(A, eA, m, n):
    """Matrix power with scaling (ks.c ``m_power``). Returns (V, eV)."""
    if n == 1:
        return list(A), eA
    V, eV = _ks_m_power(A, eA, m, n // 2)
    B = _ks_m_multiply(V, V, m)
    eB = 2 * eV
    if n % 2 == 0:
        V = B
        eV = eB
    else:
        V = _ks_m_multiply(A, B, m)
        eV = eA + eB
    if V[(m // 2) * m + (m // 2)] > 1e140:
        for i in range(m * m):
            V[i] = V[i] * 1e-140
        eV += 140
    return V, eV


def _ks_K2x(n: int, d: float) -> float:
    """One-sample two-sided exact Kolmogorov distribution (ks.c ``K2x``,
    Marsaglia-Tsang-Wang 2003 matrix method)."""
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    H = [0.0] * (m * m)
    for i in range(m):
        for j in range(m):
            H[i * m + j] = 0.0 if (i - j + 1 < 0) else 1.0
    for i in range(m):
        H[i * m] -= _R_pow_di(h, i + 1)
        H[(m - 1) * m + i] -= _R_pow_di(h, m - i)
    H[(m - 1) * m] += (_R_pow_di(2 * h - 1, m) if (2 * h - 1 > 0) else 0.0)
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 1 + 1):
                    H[i * m + j] /= g
    Q, eQ = _ks_m_power(H, 0, m, n)
    s = Q[(k - 1) * m + k - 1]
    for i in range(1, n + 1):
        s = s * i / n
        if s < 1e-140:
            s *= 1e140
            eQ -= 140
    s *= _R_pow_di(10.0, eQ)
    return s


def _pkolmogorov_one_exact(q, n, lower_tail=True):
    """One-sided one-sample exact Kolmogorov (Birnbaum-Tingey 1951)."""
    jmax = int(math.floor(n * (1 - q)))
    terms = [math.exp(_nm.lchoose(n, j)
                      + (n - j) * math.log(1 - q - j / n)
                      + (j - 1) * math.log(q + j / n))
             for j in range(0, jmax + 1)]
    p = q * _rsum_ld(terms)
    return (1 - p) if lower_tail else p


def _pkolmogorov(q, size, two_sided=True, exact=True, lower_tail=True):
    """R's internal ``pkolmogorov`` — P(D < q), one-sample (ks.test.R)."""
    if math.isnan(q):
        return math.nan
    if q <= 0:
        return 1 - lower_tail
    if q > 1:
        return float(lower_tail)
    if exact:
        if two_sided:
            p = _ks_K2x(int(size), q)
            return p if lower_tail else 1 - p
        return _pkolmogorov_one_exact(q, size, lower_tail)
    if two_sided:
        return _ks_K2l(math.sqrt(size) * q, lower_tail, 1e-6)
    # R: exp(- 2 * n * q^2); q^2 == q*q, unary- binds above * → (-2*n)*(q*q)
    p = math.exp((-2 * size) * (q * q))
    return (1 - p) if lower_tail else p


def _ks_test_two(q, r, s, two):
    return (abs(r - s) >= q) if two else ((r - s) >= q)


def _psmirnov_exact_uniq(q, m, n, two, lower):
    """Two-sample exact Smirnov, distinct values (ks.c uniq_lower/upper)."""
    md = float(m)
    nd = float(n)
    if lower:
        u = [0.0] * (n + 1)
        u[0] = 1.0
        for j in range(1, n + 1):
            u[j] = 0.0 if _ks_test_two(q, 0.0, j / nd, two) else u[j - 1]
        for i in range(1, m + 1):
            w = i / (i + n)
            if _ks_test_two(q, i / md, 0.0, two):
                u[0] = 0.0
            else:
                u[0] = w * u[0]
            for j in range(1, n + 1):
                if _ks_test_two(q, i / md, j / nd, two):
                    u[j] = 0.0
                else:
                    u[j] = w * u[j] + u[j - 1]
        return u[n]
    # upper
    u = [0.0] * (n + 1)
    u[0] = 0.0
    for j in range(1, n + 1):
        u[j] = 1.0 if _ks_test_two(q, 0.0, j / nd, two) else u[j - 1]
    for i in range(1, m + 1):
        if _ks_test_two(q, i / md, 0.0, two):
            u[0] = 1.0
        for j in range(1, n + 1):
            if _ks_test_two(q, i / md, j / nd, two):
                u[j] = 1.0
            else:
                v = i / (i + j)
                w = j / (i + j)
                u[j] = v * u[j] + w * u[j - 1]
    return u[n]


def _psmirnov_exact_ties(q, m, n, z, two, lower):
    """Two-sample exact Smirnov with ties (ks.c ties_lower/upper); ``z`` is the
    length-(m+n+1) integer tie-boundary indicator."""
    md = float(m)
    nd = float(n)
    u = [0.0] * (n + 1)
    if lower:
        u[0] = 1.0
        for j in range(1, n + 1):
            if _ks_test_two(q, 0.0, j / nd, two) and z[j]:
                u[j] = 0.0
            else:
                u[j] = u[j - 1]
        for i in range(1, m + 1):
            w = i / (i + n)
            if _ks_test_two(q, i / md, 0.0, two) and z[i]:
                u[0] = 0.0
            else:
                u[0] = w * u[0]
            for j in range(1, n + 1):
                if _ks_test_two(q, i / md, j / nd, two) and z[i + j]:
                    u[j] = 0.0
                else:
                    u[j] = w * u[j] + u[j - 1]
        return u[n]
    # upper
    u[0] = 0.0
    for j in range(1, n + 1):
        if _ks_test_two(q, 0.0, j / nd, two) and z[j]:
            u[j] = 1.0
        else:
            u[j] = u[j - 1]
    for i in range(1, m + 1):
        if _ks_test_two(q, i / md, 0.0, two) and z[i]:
            u[0] = 1.0
        for j in range(1, n + 1):
            if _ks_test_two(q, i / md, j / nd, two) and z[i + j]:
                u[j] = 1.0
            else:
                v = i / (i + j)
                w = j / (i + j)
                u[j] = v * u[j] + w * u[j - 1]
    return u[n]


def _psmirnov(q, n_x, n_y, w_combined, alternative, exact, lower_tail=True):
    """R's ``psmirnov`` (ks.test.R) — two-sample Smirnov CDF P(D < q).

    ``w_combined`` is the concatenated sample (for the ties path) or ``None``."""
    if q <= 0:
        return 1 - lower_tail
    if q > 1:
        return float(lower_tail)
    two = (alternative == "two.sided")
    n = n_x * n_y / (n_x + n_y)
    if not exact:  # asymptotic
        if two:
            return _ks_K2l(math.sqrt(n) * q, lower_tail, 1e-6)
        ret = -math.expm1((-2 * n) * (q * q))  # R: -expm1(- 2 * n * q^2)
        return ret if lower_tail else 1 - ret
    # exact
    m_, nn = int(n_x), int(n_y)
    if alternative == "less":
        m_, nn = nn, m_
    qa = (0.5 + math.floor(q * m_ * nn - 1e-7)) / (m_ * nn)
    if w_combined is not None:  # ties
        sw = np.sort(w_combined)
        zdiff = (np.diff(sw) != 0).astype(int)
        if zdiff.any():
            z = [0] + list(zdiff) + [1]  # c(0L, z, 1L)
        else:
            z = None
        if z is not None:
            if lower_tail:
                return _psmirnov_exact_ties(qa, m_, nn, z, two, True)
            return _psmirnov_exact_ties(qa, m_, nn, z, two, False)
    if lower_tail:
        return _psmirnov_exact_uniq(qa, m_, nn, two, True)
    return _psmirnov_exact_uniq(qa, m_, nn, two, False)


# --- public Smirnov distribution surface (R: psmirnov/qsmirnov/rsmirnov) -----
_DBL_EPS = 2.220446049250313e-16


def _psmirnov_exact_p(q, n_x, n_y, z, alternative, lower_tail):
    """R's ``psmirnov_exact`` probability (no log) — the exact branch of
    :func:`_psmirnov`, factored for the public ``psmirnov``."""
    two = (alternative == "two.sided")
    m_, nn = int(n_x), int(n_y)
    if alternative == "less":
        m_, nn = nn, m_
    qa = (0.5 + math.floor(q * m_ * nn - 1e-7)) / (m_ * nn)
    if z is not None:
        sw = np.sort(np.asarray(z, dtype=float))
        zdiff = (np.diff(sw) != 0).astype(int)
        if zdiff.any():
            zind = [0] + list(zdiff) + [1]
            return _psmirnov_exact_ties(qa, m_, nn, zind, two, lower_tail)
    return _psmirnov_exact_uniq(qa, m_, nn, two, lower_tail)


def _psmirnov_asymp_r(q, n_x, n_y, alternative, lower_tail, log_p):
    """R's ``psmirnov_asymp`` (ks.test.R) with faithful log/tail handling."""
    n = n_x * n_y / (n_x + n_y)
    if alternative == "two.sided":
        ret = _ks_K2l(math.sqrt(n) * q, lower_tail, 1e-6)
        return math.log(ret) if log_p else ret
    ret = -math.expm1((-2 * n) * (q * q))          # R: -expm1(-2 n q^2)
    if log_p:
        return math.log(ret) if lower_tail else math.log1p(-ret)
    return ret if lower_tail else 1 - ret


def _smirnov_sim(nrowt, ncolt, B, twosided, rng):
    """R's ``Smirnov_sim`` (ks.c) — ``B`` simulated Smirnov D statistics from
    random 2-way tables (rcont2) with the given margins. ``fact`` is the
    *cumulative* log-factorial (``fact[i]=fact[i-1]+log(i)``), matching the C."""
    nrow = len(nrowt)
    n = int(sum(nrowt))
    fact = [0.0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] + math.log(i)
    c0 = ncolt[0]
    c1 = ncolt[1]
    results = np.empty(B, dtype=float)
    for it in range(B):
        observed = rng.rcont2(nrowt, ncolt, n, fact)
        s = 0.0
        cs0 = 0
        cs1 = 0
        for j in range(nrow):
            cs0 += observed[j][0]
            cs1 += observed[j][1]
            diff = cs0 / c0 - cs1 / c1
            if twosided:
                diff = abs(diff)
            if diff > s:
                s = diff
        results[it] = s
    return results


def _psmirnov_ecdf(dsim, q, lower_tail, log_p):
    """R's ``psmirnov_simul`` tail: ``ecdf(Dsim)(q - sqrt(eps))``."""
    thr = q - math.sqrt(_DBL_EPS)
    r = float(np.count_nonzero(dsim <= thr)) / dsim.size
    if log_p:
        return math.log(r) if lower_tail else math.log1p(-r)
    return r if lower_tail else 1 - r


def rsmirnov(n, sizes, z=None, alternative="two.sided"):
    """R's ``rsmirnov(n, sizes, z, alternative)`` (ks.test.R + ks.c
    ``Smirnov_sim``) — ``n`` variates from the two-sample Smirnov distribution,
    on R's MT stream (bit-exact). ``sizes = (n_x, n_y)``; ``z`` is the pooled
    sample when there are ties (else ``None``)."""
    if n is None or int(n) == 0:
        return np.array([])
    if n < 0:
        raise ValueError("invalid arguments")
    B = int(math.floor(n))
    n_x = int(math.floor(sizes[0]))
    n_y = int(math.floor(sizes[1]))
    if n_x < 1:
        raise ValueError("not enough 'x' data")
    if n_y < 1:
        raise ValueError("not enough 'y' data")
    if z is None:
        rt = [1] * (n_x + n_y)                      # rep.int(1L, n_x + n_y)
    else:
        _, counts = np.unique(np.asarray(z), return_counts=True)  # table(z)
        rt = [int(v) for v in counts]
    cols = [n_y, n_x] if alternative == "less" else [n_x, n_y]
    two = (alternative == "two.sided")
    return _smirnov_sim(rt, cols, B, two, _dist._r_rng())


def psmirnov(q, sizes, z=None, alternative="two.sided", exact=True,
             simulate=False, B=2000, lower_tail=True, log_p=False):
    """R's ``psmirnov(q, sizes, z, alternative, exact, simulate, B, lower.tail,
    log.p)`` — the two-sample Smirnov CDF ``P(D < q)`` (ks.test.R). Exact
    (Schröer-Trenkler recursion, incl. ties) and asymptotic branches are
    bit-exact; ``simulate=True`` draws ``B`` Monte-Carlo variates via
    :func:`rsmirnov` (stream bit-exact to R)."""
    qarr = np.atleast_1d(np.asarray(q, dtype=float))
    n_x = int(math.floor(sizes[0]))
    n_y = int(math.floor(sizes[1]))
    exact = exact and not simulate
    dsim = rsmirnov(B, sizes, z, alternative) if simulate else None
    ret = np.empty(qarr.shape, dtype=float)
    for i, qi in enumerate(qarr):
        if math.isnan(qi):
            ret[i] = math.nan
            continue
        if qi <= 0:
            p0 = 1.0 - lower_tail
        elif qi > 1:
            p0 = float(lower_tail)
        else:
            p0 = None
        if p0 is not None:
            ret[i] = ((math.log(p0) if p0 > 0 else -math.inf) if log_p else p0)
        elif simulate:
            ret[i] = _psmirnov_ecdf(dsim, qi, lower_tail, log_p)
        elif not exact:
            ret[i] = _psmirnov_asymp_r(qi, n_x, n_y, alternative, lower_tail, log_p)
        else:
            pp = _psmirnov_exact_p(qi, n_x, n_y, z, alternative, lower_tail)
            if not math.isfinite(pp):               # exact failed → MC fallback
                if dsim is None:
                    dsim = rsmirnov(B, sizes, z, alternative)
                ret[i] = _psmirnov_ecdf(dsim, qi, lower_tail, log_p)
            else:
                ret[i] = math.log(pp) if log_p else pp
    return float(ret[0]) if np.ndim(q) == 0 else ret


def qsmirnov(p, sizes, z=None, alternative="two.sided", exact=True,
             simulate=False, B=2000):
    """R's ``qsmirnov(p, sizes, z, alternative, exact, simulate, B)`` — the
    Smirnov quantile: the smallest support point ``d`` with ``psmirnov(d) >= p``
    (ks.test.R). With ``p=None`` returns the ``{stat, prob}`` support table."""
    n_x = int(math.floor(sizes[0]))
    n_y = int(math.floor(sizes[1]))
    if n_x * n_y < 1e4:
        stat = np.unique(np.subtract.outer(
            np.arange(n_x + 1) / n_x, np.arange(n_y + 1) / n_y).ravel())
    else:
        stat = np.arange(-10000, 10001) / (1e4 + 1)
    if alternative == "two.sided":
        stat = np.abs(stat)
    prb = np.atleast_1d(psmirnov(stat, sizes, z=z, alternative=alternative,
                                 exact=exact, simulate=simulate, B=B,
                                 lower_tail=True, log_p=False))
    if p is None:
        return {"stat": stat, "prob": prb}
    pa = np.atleast_1d(np.asarray(p, dtype=float))
    ret = np.array(pa, dtype=float)
    bad = np.isnan(pa) | (pa < 0) | (pa > 1)
    ret[bad] = np.nan
    for i in range(pa.size):
        if bad[i]:
            continue
        cand = stat[prb >= pa[i]]
        ret[i] = float(np.min(cand)) if cand.size else math.inf
    return float(ret[0]) if np.ndim(p) == 0 else ret


_KS_CDF_NAMES = {
    "pnorm": "pnorm", "punif": "punif", "pexp": "pexp", "pgamma": "pgamma",
    "pbeta": "pbeta", "plnorm": None, "pchisq": "pchisq", "pt": "pt",
    "pf": "pf", "ppois": "ppois", "pbinom": "pbinom", "pcauchy": None,
    "pweibull": None, "plogis": None,
}


def _ks_resolve_cdf(y):
    """Resolve ``y`` (a callable or R-style ``"pnorm"`` name) to a CDF."""
    if callable(y):
        return y
    name = _KS_CDF_NAMES.get(y, None)
    if name is not None and hasattr(_dist, name):
        return getattr(_dist, name)
    raise ValueError(
        f"ks_test(): 'y' must be a second sample, a CDF callable, or a "
        f"supported distribution name; got {y!r}")


def ks_test(
    x,
    y,
    *args,
    alternative: str = "two.sided",
    exact: Optional[bool] = None,
    **kwargs,
) -> HTest:
    """R's ``ks.test`` — Kolmogorov-Smirnov test, faithful to ``ks.test.R``.

    Two-sample (``y`` a numeric sample) or one-sample goodness-of-fit (``y`` a
    CDF callable or an R-style name such as ``"pnorm"``; extra ``*args`` /
    ``**kwargs`` are passed to the CDF, as R passes ``...``). The statistic
    (``D`` / ``D^+`` / ``D^-``) and p-value are bit-exact to R: exact small-n
    distributions via the ported ``src/ks.c`` kernels (Marsaglia-Tsang-Wang
    ``K2x`` one-sample; Schröer-Trenkler / Viehmann recursion two-sample;
    Birnbaum-Tingey one-sided), asymptotic (``K2l``) otherwise. R's default
    ``exact`` selection is used unless overridden.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"ks_test(): unknown alternative {alternative!r}")
    x_arr = _as_array(x).astype(float)
    x_arr = x_arr[~np.isnan(x_arr)]
    n = len(x_arr)
    if n < 1:
        raise ValueError("not enough 'x' data")
    nm_alt = {
        "two.sided": "two-sided",
        "less": "the CDF of x lies below that of y",
        "greater": "the CDF of x lies above that of y",
    }[alternative]
    stat_name = {"two.sided": "D", "greater": "D^+", "less": "D^-"}[alternative]

    if isinstance(y, str) or callable(y):  # one-sample
        cdf = _ks_resolve_cdf(y)
        ties = len(np.unique(x_arr)) < n
        use_exact = ((n < 100) and not ties) if exact is None else exact
        method = ("Exact" if use_exact else "Asymptotic") \
            + " one-sample Kolmogorov-Smirnov test"
        xs = np.asarray(cdf(np.sort(x_arr), *args, **kwargs), float) \
            - np.arange(n) / n
        if alternative == "two.sided":
            stat = float(max(np.max(xs), np.max(1.0 / n - xs)))
        elif alternative == "greater":
            stat = float(np.max(1.0 / n - xs))
        else:
            stat = float(np.max(xs))
        pval = _pkolmogorov(stat, n, two_sided=(alternative == "two.sided"),
                            exact=use_exact, lower_tail=False)
        nm_alt = {
            "two.sided": "two-sided",
            "less": "the CDF of x lies below the null hypothesis",
            "greater": "the CDF of x lies above the null hypothesis",
        }[alternative]
        data_name = "x"
    else:  # two-sample (Smirnov)
        y_arr = _as_array(y).astype(float)
        y_arr = y_arr[~np.isnan(y_arr)]
        n_y = len(y_arr)
        if n_y < 1:
            raise ValueError("not enough 'y' data")
        n_x = n
        use_exact = (n_x * n_y < 10000) if exact is None else exact
        method = ("Exact" if use_exact else "Asymptotic") \
            + " two-sample Kolmogorov-Smirnov test"
        w = np.concatenate([x_arr, y_arr])
        order = np.argsort(w, kind="stable")
        vals = np.where(order < n_x, 1.0 / n_x, -1.0 / n_y)
        # R's cumsum() accumulates in LDOUBLE (src/main/cum.c), storing each
        # partial as a double; np.cumsum is plain double — replicate for parity.
        acc = np.longdouble(0.0)
        z = np.empty(len(vals))
        for _i, _v in enumerate(vals):
            acc += _v
            z[_i] = float(acc)
        ties = len(np.unique(w)) < (n_x + n_y)
        if ties:
            keep = np.concatenate([np.where(np.diff(np.sort(w)) != 0)[0],
                                   [n_x + n_y - 1]])
            z = z[keep]
        if alternative == "two.sided":
            stat = float(np.max(np.abs(z)))
        elif alternative == "greater":
            stat = float(np.max(z))
        else:
            stat = float(-np.min(z))
        pval = _psmirnov(stat, n_x, n_y, w if ties else None,
                         alternative, use_exact, lower_tail=False)
        data_name = "x and y"

    pval = min(1.0, max(0.0, pval))
    return HTest(
        method=method,
        statistic={stat_name: stat},
        p_value=float(pval),
        alternative=nm_alt,
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
    label vectors. The data is reshaped into ``(blocks × groups)`` wide form;
    the tie-corrected Friedman statistic and its p-value (nmath ``pchisq``)
    are computed as in ``friedman.test.R`` — bit-exact to R, no scipy.
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


def _rmedian(arr) -> float:
    """R's ``median.default``: middle order statistic for odd ``n``, else the
    :func:`_rmean` (two-pass) of the two middle values."""
    a = np.sort(np.asarray(arr, dtype=float))
    n = a.size
    half = (n + 1) // 2  # 1-based
    if n % 2 == 1:
        return float(a[half - 1])
    return _rmean([a[half - 1], a[half]])


def oneway_test(formula: str, data: pl.DataFrame, *,
                var_equal: bool = False) -> HTest:
    """R's ``oneway.test`` — test for equal means across groups.

    Welch's ANOVA by default (``var_equal=False``, not assuming equal
    variances); classical one-way ANOVA F-test when ``var_equal=True``.
    Faithful port of ``oneway.test.R`` — F statistic and p-value (nmath
    ``pf``) bit-exact to R.
    """
    if "~" not in formula:
        raise ValueError("formula must look like 'y ~ group'")
    lhs, rhs = (s.strip() for s in formula.split("~", 1))
    y = data[lhs].to_numpy().astype(float)
    g = np.asarray(data[rhs].to_list())
    ok = np.isfinite(y) & np.array([gi is not None for gi in g])
    y, g = y[ok], g[ok]
    glabels = np.unique(g)
    k = len(glabels)
    if k < 2:
        raise ValueError("not enough groups")
    groups = [y[g == gl] for gl in glabels]
    n_i = np.array([len(gr) for gr in groups], dtype=float)
    if np.any(n_i < 2):
        raise ValueError("not enough observations")
    m_i = np.array([_rmean(gr) for gr in groups])
    v_i = np.array([_rvar(gr) for gr in groups])
    w_i = n_i / v_i
    sum_w = _rsum_ld(w_i)
    tmp = _rsum_ld((1 - w_i / sum_w) ** 2 / (n_i - 1)) / (k ** 2 - 1)
    if var_equal:
        n = _rsum_ld(n_i)
        grand = _rmean(y)
        stat = ((_rsum_ld(n_i * (m_i - grand) ** 2) / (k - 1))
                / (_rsum_ld((n_i - 1) * v_i) / (n - k)))
        df1, df2 = float(k - 1), float(n - k)
        method = "One-way analysis of means"
    else:
        m = _rsum_ld(w_i * m_i) / sum_w
        stat = (_rsum_ld(w_i * (m_i - m) ** 2)
                / ((k - 1) * (1 + 2 * (k - 2) * tmp)))
        df1, df2 = float(k - 1), 1.0 / (3 * tmp)
        method = "One-way analysis of means (not assuming equal variances)"
    pval = float(_dist.pf(stat, df1, df2, lower_tail=False))
    return HTest(
        method=method,
        statistic={"F": float(stat)},
        parameter={"num df": df1, "denom df": df2},
        p_value=pval,
        alternative="",
        data_name=f"{lhs} and {rhs}",
    )


def fligner_test(x, g) -> HTest:
    """R's ``fligner.test`` — Fligner-Killeen test of homogeneity of variances.

    ``x`` is the values vector, ``g`` the parallel group labels. Groups are
    median-centred, absolute-rank normal scores are formed, and the χ²
    statistic (nmath ``pchisq``) is returned. Faithful port of
    ``fligner.test.R`` — bit-exact to R.
    """
    x = _as_array(x).astype(float)
    g = np.asarray(g)
    if x.shape != g.shape:
        raise ValueError("fligner_test(): x and g must have the same length")
    ok = np.isfinite(x) & np.array([gi is not None for gi in g])
    x, g = x[ok], g[ok]
    glabels = np.unique(g)
    k = len(glabels)
    if k < 2:
        raise ValueError("all observations are in the same group")
    n = len(x)
    if n < 2:
        raise ValueError("not enough observations")
    # x <- x - tapply(x, g, median)[g]  (centre each group by its median)
    med = {gl: _rmedian(x[g == gl]) for gl in glabels}
    xc = x - np.array([med[gi] for gi in g])
    a = _dist.qnorm((1 + _avg_rank(np.abs(xc)) / (n + 1)) / 2)
    a = a - _rmean(a)
    v = _rsum_ld(a ** 2) / (n - 1)
    stat = _rsum_ld([len(a[g == gl]) * _rmean(a[g == gl]) ** 2
                     for gl in glabels]) / v
    df = k - 1
    return HTest(
        method="Fligner-Killeen test of homogeneity of variances",
        statistic={"Fligner-Killeen:med chi-squared": float(stat)},
        parameter={"df": int(df)},
        p_value=float(_dist.pchisq(stat, df, lower_tail=False)),
        alternative="",
        data_name="x and g",
    )


def mood_test(x, y, alternative: str = "two.sided") -> HTest:
    """R's ``mood.test`` — Mood two-sample test of scale.

    Faithful port of ``mood.test.R``: the no-ties statistic follows Conover
    (1971); with ties, the mid-rank expressions of Mielke (1967). The
    standardised statistic Z and its normal p-value are bit-exact to R.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"mood_test(): unknown alternative {alternative!r}")
    x = _as_array(x).astype(float)
    y = _as_array(y).astype(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    m = len(x)
    n = len(y)
    N = m + n
    if N < 3:
        raise ValueError("not enough observations")
    E = m * (N ** 2 - 1) / 12
    v = (1 / 180) * m * n * (N + 1) * (N + 2) * (N - 2)
    z = np.concatenate([x, y])
    if len(np.unique(z)) == len(z):
        r = _avg_rank(z)
        T = _rsum_ld((r[:m] - (N + 1) / 2) ** 2)
    else:
        u = np.unique(z)  # sort(unique(z))
        a = np.bincount(np.searchsorted(u, x), minlength=len(u)).astype(float)
        t = np.bincount(np.searchsorted(u, z), minlength=len(u)).astype(float)
        p = np.cumsum((np.arange(1, N + 1) - (N + 1) / 2) ** 2)
        ct = np.cumsum(t)
        v = v - (m * n) / (180 * N * (N - 1)) * _rsum_ld(
            t * (t ** 2 - 1) * (t ** 2 - 4 + 15 * (N - 2 * ct + t) ** 2))
        pcum = p[(ct).astype(int) - 1]
        dp = np.diff(np.concatenate([[0.0], pcum]))
        T = _rsum_ld(a * dp / t)
    zstat = (T - E) / math.sqrt(v)
    p = float(_dist.pnorm(zstat))
    if alternative == "less":
        pval = p
    elif alternative == "greater":
        pval = 1 - p
    else:
        pval = 2 * min(p, 1 - p)
    return HTest(
        method="Mood two-sample test of scale",
        statistic={"Z": float(zstat)},
        p_value=pval,
        null_value=None,
        alternative=alternative,
        data_name="x and y",
    )


def quade_test(y, groups=None, blocks=None) -> HTest:
    """R's ``quade.test`` — Quade test for unreplicated block designs.

    Accepts either a ``(blocks × treatments)`` matrix ``y`` or parallel
    ``y``/``groups``/``blocks`` vectors (like :func:`friedman_test`).
    Faithful port of ``quade.test.R`` — the Quade F statistic and its
    p-value (nmath ``pf``) are bit-exact to R; the degenerate ``A == B``
    case returns ``NaN`` with ``PVAL = gamma(k+1)^(1-b)``.
    """
    y_arr = np.asarray(y, dtype=float) if not isinstance(y, np.ndarray) \
        else y.astype(float)
    if y_arr.ndim == 2 and groups is None and blocks is None:
        mat = y_arr
        dname = "y"
    else:
        yv = _as_array(y).astype(float)
        gv = np.asarray(groups)
        bv = np.asarray(blocks)
        if not (yv.shape == gv.shape == bv.shape):
            raise ValueError(
                "quade_test(): y, groups, blocks must have the same length")
        glabels = np.unique(gv)
        blabels = np.unique(bv)
        mat = np.full((len(blabels), len(glabels)), np.nan)
        bpos = {bl: i for i, bl in enumerate(blabels)}
        gpos = {gl: j for j, gl in enumerate(glabels)}
        for yi, gi, bi in zip(yv, gv, bv):
            mat[bpos[bi], gpos[gi]] = yi
        dname = "y, groups and blocks"
    mat = mat[np.all(np.isfinite(mat), axis=1)]
    b, k = mat.shape
    r = np.vstack([_avg_rank(row) for row in mat])
    ranges = np.array([row.max() - row.min() for row in mat])
    q = _avg_rank(ranges)
    s = q[:, None] * (r - (k + 1) / 2)
    A = _rsum_ld(s.ravel() ** 2)
    B = _rsum_ld(s.sum(axis=0) ** 2) / b
    if A == B:
        stat = float("nan")
        df1 = df2 = float("nan")
        pval = float(_nm.gammafn(k + 1) ** (1 - b))
    else:
        stat = (b - 1) * B / (A - B)
        df1, df2 = float(k - 1), float((b - 1) * (k - 1))
        pval = float(_dist.pf(stat, df1, df2, lower_tail=False))
    return HTest(
        method="Quade test",
        statistic={"Quade F": float(stat)},
        parameter={"num df": df1, "denom df": df2},
        p_value=pval,
        alternative="",
        data_name=dname,
    )


def poisson_test(x, T=1.0, r=1.0, alternative: str = "two.sided",
                 conf_level: float = 0.95) -> HTest:
    """R's ``poisson.test`` — exact test for one or two Poisson rates.

    One count (``k = 1``): exact test of the rate against ``r`` on time base
    ``T`` (two-sided uses R's opposite-tail density sum; CI via ``qgamma``).
    Two counts (``k = 2``): comparison of rates, delegated to
    :func:`binom_test` with the rate-ratio reparametrisation. Faithful port
    of ``poisson.test.R`` — bit-exact to R.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"poisson_test(): unknown alternative {alternative!r}")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    lx = len(x)
    if len(T) != lx:
        if len(T) == 1:
            T = np.repeat(T, lx)
        else:
            raise ValueError("'x' and 'T' have incompatible length")
    xr = np.rint(x)
    if np.any(~np.isfinite(x) | (x < 0)) or np.max(np.abs(x - xr)) > 1e-7:
        raise ValueError("'x' must be finite, nonnegative, and integer")
    x = xr
    if np.any(np.isnan(T) | (T < 0)):
        raise ValueError("'T' must be nonnegative")
    k = lx
    if k < 1:
        raise ValueError("not enough data")
    if k > 2:
        raise ValueError("the case k > 2 is unimplemented")
    r = float(r)
    if r < 0:
        raise ValueError("'r' must be a single positive number")

    if k == 2:
        prob = r * T[0] / (r * T[0] + T[1])
        rval = binom_test([int(x[0]), int(x[1])], p=prob,
                          alternative=alternative, conf_level=conf_level)
        pp = rval.conf_int
        ci = (pp[0] / (1 - pp[0]) * T[1] / T[0],
              pp[1] / (1 - pp[1]) * T[1] / T[0])
        return HTest(
            method="Comparison of Poisson rates",
            statistic={"count1": float(x[0])},
            parameter={"expected count1":
                       float(x.sum() * r * T[0] / (T[0] + T[1] * r))},
            p_value=rval.p_value,
            conf_int=ci,
            estimate={"rate ratio": float((x[0] / T[0]) / (x[1] / T[1]))},
            null_value={"rate ratio": r},
            alternative=alternative,
            conf_level=conf_level,
            data_name="x time base: T",
        )

    xx = float(x[0])
    TT = float(T[0])
    m = r * TT
    if alternative == "less":
        pval = float(_dist.ppois(xx, m))
    elif alternative == "greater":
        pval = float(_dist.ppois(xx - 1, m, lower_tail=False))
    else:  # two.sided
        if m == 0:
            pval = float(xx == 0)
        else:
            rel_err = 1 + 1e-7
            d = float(_dist.dpois(xx, m))
            if xx == m:
                pval = 1.0
            elif xx < m:
                N = math.ceil(2 * m - xx)
                while float(_dist.dpois(N, m)) > d:
                    N = 2 * N
                i = np.arange(math.ceil(m), N + 1)
                y = int(np.sum(np.asarray(_dist.dpois(i, m)) <= d * rel_err))
                pval = (float(_dist.ppois(xx, m))
                        + float(_dist.ppois(N - y, m, lower_tail=False)))
            else:  # xx > m
                i = np.arange(0, math.floor(m) + 1)
                y = int(np.sum(np.asarray(_dist.dpois(i, m)) <= d * rel_err))
                pval = (float(_dist.ppois(y - 1, m))
                        + float(_dist.ppois(xx - 1, m, lower_tail=False)))

    def p_L(xv, alpha):
        return 0.0 if xv == 0 else float(_dist.qgamma(alpha, xv))

    def p_U(xv, alpha):
        return float(_dist.qgamma(1 - alpha, xv + 1))

    if alternative == "less":
        ci = (0.0, p_U(xx, 1 - conf_level) / TT)
    elif alternative == "greater":
        ci = (p_L(xx, 1 - conf_level) / TT, float("inf"))
    else:
        alpha = (1 - conf_level) / 2
        ci = (p_L(xx, alpha) / TT, p_U(xx, alpha) / TT)
    return HTest(
        method="Exact Poisson test",
        statistic={"number of events": xx},
        parameter={"time base": TT},
        p_value=float(pval),
        conf_int=ci,
        estimate={"event rate": xx / TT},
        null_value={"event rate": r},
        alternative=alternative,
        conf_level=conf_level,
        data_name="x time base: T",
    )


def prop_trend_test(x, n, score=None) -> HTest:
    """R's ``prop.trend.test`` — chi-squared test for trend in proportions.

    ``x`` successes out of ``n`` trials at ordinal ``score`` (default
    ``1..length(x)``). Faithful port of ``prop.trend.test.R``: the statistic
    is the weighted-regression sum of squares for ``score`` — computed here
    via hea's ``lm`` exactly as R computes ``anova(lm(freq ~ score,
    weights = n/p/(1-p)))["score", "Sum Sq"]``. It therefore inherits the
    documented lm QR ≤1-ulp residual (gfortran ``dqrdc2`` FMA contraction),
    otherwise bit-exact; the p-value is nmath ``pchisq``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = np.asarray(n, dtype=float).ravel()
    if score is None:
        score = np.arange(1, len(x) + 1, dtype=float)
    else:
        score = np.asarray(score, dtype=float).ravel()
    p = _rsum_ld(x) / _rsum_ld(n)
    w = n / p / (1 - p)
    freq = x / n
    fit = lm("freq ~ score",
             pl.DataFrame({"freq": freq, "score": score}), weights=w)
    chisq = float(np.asarray(fit.effects)[1]) ** 2
    score_str = " ".join(
        str(int(s)) if float(s).is_integer() else repr(float(s))
        for s in score)
    return HTest(
        method="Chi-squared Test for Trend in Proportions",
        statistic={"X-squared": chisq},
        parameter={"df": 1},
        p_value=float(_dist.pchisq(chisq, 1, lower_tail=False)),
        alternative="",
        data_name=f"x out of n,\n using scores: {score_str}",
    )


def _cansari(k: int, m: int, n: int, memo: dict) -> float:
    """Count of Ansari-Bradley configurations with statistic ``k`` for group
    sizes ``m``, ``n`` — memoised recursion from ``ansari.c`` ``cansari``."""
    lo = (m + 1) * (m + 1) // 4
    up = lo + m * n // 2
    if k < lo or k > up:
        return 0.0
    key = (m, n, k)
    v = memo.get(key)
    if v is not None:
        return v
    if m == 0:
        v = 1.0 if k == 0 else 0.0
    elif n == 0:
        v = 1.0 if k == lo else 0.0
    else:
        v = (_cansari(k, m, n - 1, memo)
             + _cansari(k - (m + n + 1) // 2, m - 1, n, memo))
    memo[key] = v
    return v


def _pansari(q, m: int, n: int) -> float:
    """Exact Ansari-Bradley CDF ``P(AB <= q)`` — ``ansari.c`` ``pansari``."""
    memo: dict = {}
    lo = (m + 1) * (m + 1) // 4
    up = lo + m * n // 2
    c = _nm.choose(m + n, m)
    qq = math.floor(q + 1e-7)
    if qq < lo:
        return 0.0
    if qq > up:
        return 1.0
    p = 0.0
    for j in range(lo, qq + 1):
        p += _cansari(j, m, n, memo)
    return p / c


def ansari_test(x, y, alternative: str = "two.sided",
                exact: Optional[bool] = None) -> HTest:
    """R's ``ansari.test`` — Ansari-Bradley two-sample test of scale.

    Faithful port of ``ansari.test.R`` (p-value only; ``conf.int`` is not
    computed). The AB statistic is the sum of the folded mid-ranks of ``x``.
    Without ties and with ``exact`` (default when both sizes < 50), the
    p-value uses the exact distribution (``ansari.c`` recursion, ported to
    :func:`_cansari`/:func:`_pansari`); otherwise the normal approximation
    (Conover no-ties variance, or the mid-rank variance with ties). Bit-exact
    to R.
    """
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError(f"ansari_test(): unknown alternative {alternative!r}")
    x = _as_array(x).astype(float)
    y = _as_array(y).astype(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    m = len(x)
    n = len(y)
    if m < 1:
        raise ValueError("not enough 'x' observations")
    if n < 1:
        raise ValueError("not enough 'y' observations")
    N = m + n
    r = _avg_rank(np.concatenate([x, y]))
    ab = np.minimum(r, N - r + 1)
    stat = _rsum_ld(ab[:m])
    ties = len(np.unique(r)) != len(r)
    if exact is None:
        exact = (m < 50) and (n < 50)

    if exact and not ties:
        si = int(round(stat))
        if alternative == "two.sided":
            thr = (m + 1) ** 2 // 4 + ((m * n) // 2) / 2
            if si > thr:
                p = 1 - _pansari(si - 1, m, n)
            else:
                p = _pansari(si, m, n)
            pval = min(2 * p, 1.0)
        elif alternative == "less":
            pval = 1 - _pansari(si - 1, m, n)
        else:  # greater
            pval = _pansari(si, m, n)
    else:
        even = (N % 2) == 0
        if not ties:
            z = stat - (m * (N + 2) / 4 if even else m * (N + 1) ** 2 / (4 * N))
            sigma = (math.sqrt((m * n * (N + 2) * (N - 2)) / (48 * (N - 1)))
                     if even else
                     math.sqrt((m * n * (N + 1) * (3 + N ** 2)) / (48 * N ** 2)))
        else:
            z = stat - m * _rmean(ab)
            sigma = math.sqrt(m * n * _rvar(ab) / N)
        p = float(_dist.pnorm(z / sigma))
        if alternative == "two.sided":
            pval = 2 * min(p, 1 - p)
        elif alternative == "less":
            pval = 1 - p
        else:
            pval = p
        if exact and ties:
            import warnings
            warnings.warn("cannot compute exact p-value with ties")

    return HTest(
        method="Ansari-Bradley test",
        statistic={"AB": float(stat)},
        p_value=float(pval),
        null_value={"ratio of scales": 1.0},
        alternative=alternative,
        data_name="x and y",
    )


def _d2x2xk(K: int, m, n, t):
    """Density of ``S = Σ_k x[1,1,k]`` on its support, by convolution across the
    ``K`` strata of the central product-hypergeometric — port of ``d2x2xk.c``
    ``int_d2x2xk`` (uses nmath ``dhyper``). ``m``/``n``/``t`` are the per-stratum
    column-1, column-2, and row-1 totals."""
    c = [[1.0]]
    length = 0
    for i in range(K):
        y = max(0, int(t[i] - n[i]))
        z = min(int(m[i]), int(t[i]))
        ci = [0.0] * (length + z - y + 1)
        prev = c[i]
        for j in range(z - y + 1):
            u = float(_nm.dhyper(j + y, m[i], n[i], t[i], False))
            for w in range(length + 1):
                ci[w + j] += prev[w] * u
        c.append(ci)
        length = length + z - y
    total = 0.0
    for j in range(length + 1):
        total += c[K][j]
    return np.array([c[K][j] / total for j in range(length + 1)])


def _mh_table_from_factors(x, y, z):
    """Build the ``I×J×K`` count array R's ``table(x, y, z)`` produces from
    parallel factor vectors (levels sorted ascending)."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    xl, yl, zl = np.unique(x), np.unique(y), np.unique(z)
    arr = np.zeros((len(xl), len(yl), len(zl)))
    xi = {v: i for i, v in enumerate(xl)}
    yi = {v: i for i, v in enumerate(yl)}
    zi = {v: i for i, v in enumerate(zl)}
    for a, b, c in zip(x, y, z):
        arr[xi[a], yi[b], zi[c]] += 1
    return arr


def _mantelhaen_exact(arr, ns, alternative, conf_level) -> HTest:
    """Exact conditional 2×2×k test (``mantelhaen.test.R`` ``exact`` branch):
    the common-odds-ratio p-value, MLE and CI from the product-hypergeometric
    density (:func:`_d2x2xk`) inverted with the ported ``uniroot`` — mirrors
    fisher_test's non-central hypergeometric machinery."""
    mn = arr.sum(axis=0)               # (2, K): column totals per stratum
    m_ = mn[0, :]
    n_ = mn[1, :]
    t_ = arr.sum(axis=1)[0, :]         # row-1 totals per stratum
    s = _rsum_ld(arr[0, 0, :])
    lo = _rsum_ld(np.maximum(0.0, t_ - n_))
    hi = _rsum_ld(np.minimum(m_, t_))
    support = np.arange(lo, hi + 1)
    dc = _d2x2xk(ns, m_, n_, t_)
    logdc = np.log(dc)
    eps = _nm._DBL_EPSILON

    def dn(ncp):
        if ncp == 1.0:
            return dc
        d = logdc + math.log(ncp) * support
        d = np.exp(d - np.max(d))
        return d / _rsum_ld(d)

    def mn_(ncp):
        if ncp == 0:
            return lo
        if ncp == math.inf:
            return hi
        return _rsum_ld(support * dn(ncp))

    def pn(q, ncp=1.0, upper_tail=False):
        if ncp == 0:
            return float(q <= lo) if upper_tail else float(q >= lo)
        if ncp == math.inf:
            return float(q <= hi) if upper_tail else float(q >= hi)
        d = dn(ncp)
        mask = support >= q if upper_tail else support <= q
        return _rsum_ld(d[mask])

    if alternative == "less":
        pval = pn(s, 1.0)
    elif alternative == "greater":
        pval = pn(s, 1.0, upper_tail=True)
    else:
        rel_err = 1 + 1e-7
        pval = _rsum_ld(dc[dc <= dc[int(s - lo)] * rel_err])

    def mle(val):
        if val == lo:
            return 0.0
        if val == hi:
            return math.inf
        mu = mn_(1.0)
        if mu > val:
            return _nm.uniroot(lambda tt: mn_(tt) - val, 0.0, 1.0)
        if mu < val:
            return 1.0 / _nm.uniroot(lambda tt: mn_(1.0 / tt) - val, eps, 1.0)
        return 1.0

    estimate = mle(s)

    def ncp_u(val, alpha):
        if val == hi:
            return math.inf
        p = pn(val, 1.0)
        if p < alpha:
            return _nm.uniroot(lambda tt: pn(val, tt) - alpha, 0.0, 1.0)
        if p > alpha:
            return 1.0 / _nm.uniroot(
                lambda tt: pn(val, 1.0 / tt) - alpha, eps, 1.0)
        return 1.0

    def ncp_l(val, alpha):
        if val == lo:
            return 0.0
        p = pn(val, 1.0, upper_tail=True)
        if p > alpha:
            return _nm.uniroot(
                lambda tt: pn(val, tt, upper_tail=True) - alpha, 0.0, 1.0)
        if p < alpha:
            return 1.0 / _nm.uniroot(
                lambda tt: pn(val, 1.0 / tt, upper_tail=True) - alpha, eps, 1.0)
        return 1.0

    if alternative == "less":
        ci = (0.0, ncp_u(s, 1 - conf_level))
    elif alternative == "greater":
        ci = (ncp_l(s, 1 - conf_level), math.inf)
    else:
        alpha = (1 - conf_level) / 2
        ci = (ncp_l(s, alpha), ncp_u(s, alpha))
    return HTest(
        method="Exact conditional test of independence in 2 x 2 x k tables",
        statistic={"S": float(s)},
        p_value=float(pval),
        conf_int=ci,
        estimate={"common odds ratio": float(estimate)},
        null_value={"common odds ratio": 1.0},
        alternative=alternative,
        conf_level=conf_level,
        data_name="x",
    )


def mantelhaen_test(x, y=None, z=None, alternative: str = "two.sided",
                    correct: bool = True, exact: bool = False,
                    conf_level: float = 0.95) -> HTest:
    """R's ``mantelhaen.test`` — Cochran-Mantel-Haenszel test.

    ``x`` is an ``I×J×K`` array of stratified counts (or parallel factor
    vectors ``x``/``y``/``z``). For ``2×2×K`` tables the classical
    Mantel-Haenszel χ² (optional Yates correction) is returned with the
    Mantel-Haenszel common odds-ratio estimate and Robins-Breslow-Greenland
    CI; otherwise the generalized CMH statistic. Faithful port of
    ``mantelhaen.test.R`` — the ``2×2×K`` path is bit-exact; the generalized
    quadratic form inherits a ≤1-ulp residual from the linear solve.
    ``exact=True`` (``2×2×K`` only) gives the exact conditional test via the
    product-hypergeometric density (:func:`_d2x2xk`) — S, p-value, MLE odds
    ratio and CI all bit-exact.
    """
    if y is not None or z is not None:
        arr = _mh_table_from_factors(x, y, z).astype(float)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError("'x' must be a 3-dimensional array")
    if np.any(np.asarray(arr.shape) < 2):
        raise ValueError("each dimension in table must be >= 2")
    if np.any(arr.sum(axis=(0, 1)) < 2):
        raise ValueError("sample size in each stratum must be > 1")
    nr, nc, ns = arr.shape

    if nr == 2 and nc == 2:
        if exact:
            return _mantelhaen_exact(arr, ns, alternative, conf_level)
        s_x = arr.sum(axis=1)          # (2, K): row totals per stratum
        s_y = arr.sum(axis=0)          # (2, K): col totals per stratum
        n = arr.sum(axis=(0, 1))       # (K,): stratum totals
        DELTA = _rsum_ld(arr[0, 0, :] - s_x[0, :] * s_y[0, :] / n)
        YATES = 0.5 if (correct and abs(DELTA) >= 0.5) else 0.0
        denom = _rsum_ld(s_x[0, :] * s_x[1, :] * s_y[0, :] * s_y[1, :]
                         / (n ** 2 * (n - 1)))
        stat = (abs(DELTA) - YATES) ** 2 / denom
        if alternative == "two.sided":
            pval = float(_dist.pchisq(stat, 1, lower_tail=False))
        else:
            zv = math.copysign(math.sqrt(stat), DELTA)
            pval = float(_dist.pnorm(zv, lower_tail=(alternative == "less")))
        method = ("Mantel-Haenszel chi-squared test "
                  + ("with" if YATES else "without")
                  + " continuity correction")
        x11, x12 = arr[0, 0, :], arr[0, 1, :]
        x21, x22 = arr[1, 0, :], arr[1, 1, :]
        s_diag = _rsum_ld(x11 * x22 / n)
        s_offd = _rsum_ld(x12 * x21 / n)
        estimate = s_diag / s_offd
        sd = math.sqrt(
            _rsum_ld((x11 + x22) * x11 * x22 / n ** 2) / (2 * s_diag ** 2)
            + _rsum_ld(((x11 + x22) * x12 * x21 + (x12 + x21) * x11 * x22)
                       / n ** 2) / (2 * s_diag * s_offd)
            + _rsum_ld((x12 + x21) * x12 * x21 / n ** 2) / (2 * s_offd ** 2))
        qn = _dist.qnorm
        if alternative == "less":
            ci = (0.0, estimate * math.exp(float(qn(conf_level)) * sd))
        elif alternative == "greater":
            ci = (estimate * math.exp(float(qn(conf_level, lower_tail=False))
                                      * sd), float("inf"))
        else:
            q = float(qn((1 - conf_level) / 2))
            ci = (estimate * math.exp(q * sd), estimate * math.exp(-q * sd))
        return HTest(
            method=method,
            statistic={"Mantel-Haenszel X-squared": float(stat)},
            parameter={"df": 1},
            p_value=pval,
            conf_int=ci,
            estimate={"common odds ratio": float(estimate)},
            null_value={"common odds ratio": 1.0},
            alternative=alternative,
            conf_level=conf_level,
            data_name="x",
        )

    # Generalized Cochran-Mantel-Haenszel I x J x K test.
    df = (nr - 1) * (nc - 1)
    nvec = np.zeros(df)
    mvec = np.zeros(df)
    V = np.zeros((df, df))
    for k in range(ns):
        f = arr[:, :, k]
        ntot = f.sum()
        rowsums = f.sum(axis=1)[:nr - 1]
        colsums = f.sum(axis=0)[:nc - 1]
        nvec = nvec + f[:nr - 1, :nc - 1].flatten(order="F")
        mvec = mvec + np.outer(rowsums, colsums).flatten(order="F") / ntot
        A_J = np.diag(ntot * colsums) - np.outer(colsums, colsums)
        A_I = np.diag(ntot * rowsums) - np.outer(rowsums, rowsums)
        V = V + np.kron(A_J, A_I) / (ntot ** 2 * (ntot - 1))
    nvec = nvec - mvec
    stat = float(nvec @ np.linalg.solve(V, nvec))
    pval = float(_dist.pchisq(stat, df, lower_tail=False))
    return HTest(
        method="Cochran-Mantel-Haenszel test",
        statistic={"Cochran-Mantel-Haenszel M^2": stat},
        parameter={"df": int(df)},
        p_value=pval,
        alternative="",
        data_name="x",
    )


@dataclass
class PairwiseHTest:
    """R's ``pairwise.htest`` — a table of adjusted pairwise p-values.

    ``p_value`` is the ``(k-1)×(k-1)`` lower-triangular matrix R prints
    (upper triangle ``NaN``); ``row_names``/``col_names`` label it.
    """

    method: str
    p_value: np.ndarray
    row_names: list
    col_names: list
    p_adjust_method: str
    data_name: str = ""

    def __repr__(self) -> str:
        out = ["", f"\tPairwise comparisons using {self.method} ", ""]
        if self.data_name:
            out.append(f"data:  {self.data_name} ")
        out.append("")
        cw = max(8, max((len(str(c)) for c in self.col_names), default=1) + 1)
        rw = max((len(str(r)) for r in self.row_names), default=1)
        out.append(" " * rw + "".join(f"{str(c):>{cw}}" for c in self.col_names))
        for i, rn in enumerate(self.row_names):
            cells = []
            for j in range(len(self.col_names)):
                v = self.p_value[i, j]
                cells.append("-" if (v is None or math.isnan(v))
                             else _fmt_pval(v))
            out.append(f"{str(rn):<{rw}}"
                       + "".join(f"{c:>{cw}}" for c in cells))
        out.append("")
        out.append(f"P value adjustment method: {self.p_adjust_method} ")
        return "\n".join(out)


def _pairwise_table(compare_levels, level_names, method):
    """R's ``pairwise.table``: fill the lower triangle (incl. diagonal) by
    ``compare_levels(row_level, col_level)`` (1-based level indices), then
    :func:`p_adjust` the flattened lower triangle jointly."""
    k = len(level_names)
    pp = np.full((k - 1, k - 1), np.nan)
    for p0 in range(k - 1):
        for q0 in range(k - 1):
            if p0 >= q0:
                pp[p0, q0] = compare_levels(p0 + 2, q0 + 1)
    idx = [(i, j) for j in range(k - 1) for i in range(k - 1) if i >= j]
    adj = p_adjust(np.array([pp[i, j] for i, j in idx]), method)
    for (i, j), v in zip(idx, adj):
        pp[i, j] = v
    return pp


def pairwise_t_test(x, g, p_adjust_method: str = "holm", pool_sd=None,
                    paired: bool = False,
                    alternative: str = "two.sided") -> PairwiseHTest:
    """R's ``pairwise.t.test`` — pairwise t-tests with p-value adjustment.

    ``pool_sd`` (default ``not paired``) uses a single pooled SD across all
    groups; otherwise each pair is a separate :func:`t_test`. Faithful port of
    ``pairwise.t.test`` — the pooled-SD p-values (nmath ``pt``) and the adjusted
    table (:func:`p_adjust`) are bit-exact to R.
    """
    if pool_sd is None:
        pool_sd = not paired
    if paired and pool_sd:
        raise ValueError("pooling of SD is incompatible with paired tests")
    x = _as_array(x).astype(float)
    g = np.asarray(g)
    glabels = np.unique(g)
    if pool_sd:
        method_str = "t tests with pooled SD"
        xbar, s, nn = [], [], []
        for gl in glabels:
            xi = x[g == gl]
            xi = xi[np.isfinite(xi)]
            xbar.append(_rmean(xi))
            s.append(math.sqrt(_rvar(xi)) if len(xi) > 1 else float("nan"))
            nn.append(len(xi))
        xbar = np.array(xbar)
        s = np.array(s)
        nn = np.array(nn, dtype=float)
        degf = nn - 1
        total_degf = _rsum_ld(degf)
        pooled_sd = math.sqrt(
            _rsum_ld(np.where(degf != 0, s ** 2, 0.0) * degf) / total_degf)

        def compare(i, j):
            dif = xbar[i - 1] - xbar[j - 1]
            se = pooled_sd * math.sqrt(1 / nn[i - 1] + 1 / nn[j - 1])
            tval = dif / se
            if alternative == "two.sided":
                return 2 * float(_dist.pt(-abs(tval), total_degf))
            return float(_dist.pt(tval, total_degf,
                                  lower_tail=(alternative == "less")))
    else:
        method_str = "paired t tests" if paired else "t tests with non-pooled SD"

        def compare(i, j):
            xi = x[g == glabels[i - 1]]
            xj = x[g == glabels[j - 1]]
            return t_test(xi, xj, paired=paired,
                          alternative=alternative).p_value

    pp = _pairwise_table(compare, glabels, p_adjust_method)
    return PairwiseHTest(method_str, pp, list(glabels[1:]),
                         list(glabels[:-1]), p_adjust_method, "x and g")


def pairwise_wilcox_test(x, g, p_adjust_method: str = "holm",
                         paired: bool = False, **kwargs) -> PairwiseHTest:
    """R's ``pairwise.wilcox.test`` — pairwise Wilcoxon tests with adjustment.

    Each pair is a :func:`wilcox_test`; the p-value table is :func:`p_adjust`-ed
    jointly. ``**kwargs`` (e.g. ``exact=``, ``correct=``) pass through. Faithful
    port of ``pairwise.wilcox.test`` — bit-exact to R.
    """
    x = _as_array(x).astype(float)
    g = np.asarray(g)
    glabels = np.unique(g)
    holder = [None]

    def compare(i, j):
        xi = x[g == glabels[i - 1]]
        xj = x[g == glabels[j - 1]]
        wt = wilcox_test(xi, xj, paired=paired, **kwargs)
        if holder[0] is None:
            holder[0] = wt.method
        return wt.p_value

    pp = _pairwise_table(compare, glabels, p_adjust_method)
    return PairwiseHTest(holder[0], pp, list(glabels[1:]),
                         list(glabels[:-1]), p_adjust_method, "x and g")


def pairwise_prop_test(x, n=None, p_adjust_method: str = "holm",
                       **kwargs) -> PairwiseHTest:
    """R's ``pairwise.prop.test`` — pairwise comparison of proportions.

    ``x`` is a length-``k`` success vector with trial counts ``n`` (or a
    ``k×2`` matrix of successes/failures). Each pair is a 2-sample
    :func:`prop_test`; the table is :func:`p_adjust`-ed jointly. Faithful port
    of ``pairwise.prop.test`` — inherits ``prop_test``'s ≤2-ulp X²/``pchisq``
    residual, otherwise matches R.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError("'x' must have 2 columns")
        n = x.sum(axis=1).astype(float)
        x = x[:, 0].astype(float)
    else:
        x = x.astype(float)
        n = np.asarray(n, dtype=float)
        if len(x) != len(n):
            raise ValueError("'x' and 'n' must have the same length")
    ok = np.isfinite(x) & np.isfinite(n)
    x, n = x[ok], n[ok]
    if len(x) < 2:
        raise ValueError("too few groups")
    level_names = list(range(1, len(x) + 1))

    def compare(i, j):
        return prop_test([x[i - 1], x[j - 1]], [n[i - 1], n[j - 1]],
                         **kwargs).p_value

    pp = _pairwise_table(compare, level_names, p_adjust_method)
    return PairwiseHTest("Pairwise comparison of proportions", pp,
                         level_names[1:], level_names[:-1],
                         p_adjust_method, "x")


_DBL_XMAX = float(np.finfo(float).max)


def _sign(x: float) -> float:
    return (x > 0) - (x < 0)


def _uniroot_ext(f, lower, upper, extend_int="no", tol=None, maxiter=1000):
    """R's ``uniroot`` including ``extendInt`` (``nlm.R``): extend
    ``[lower, upper]`` until ``f`` changes sign, then Brent-solve via the ported
    ``zeroin2``. ``extend_int`` ∈ {"no","yes","downX","upX"}."""
    if tol is None:
        tol = _nm._DBL_EPSILON ** 0.25
    if not (lower < upper):
        raise ValueError("lower < upper is not fulfilled")
    f_lower = f(lower)
    f_upper = f(upper)
    if math.isnan(f_lower) or math.isnan(f_upper):
        raise ValueError("f() value at an endpoint is NA")
    sig = {"yes": None, "downX": -1, "no": 0, "upX": 1}[extend_int]

    def truncate(x):
        return max(min(x, _DBL_XMAX), -_DBL_XMAX)

    f_low = truncate(f_lower)
    f_upp = truncate(f_upper)
    do_x = ((sig is None and f_low * f_upp > 0)
            or (sig is not None and (sig * f_low > 0 or sig * f_upp < 0)))
    it = 0
    if do_x:
        def delta_of(u):
            return 0.01 * max(1e-4, abs(u))
        if sig is None:
            dl, du = delta_of(lower), delta_of(upper)
            while (f_lower * f_upper > 0
                   and (math.isfinite(lower) or math.isfinite(upper))):
                it += 1
                if it > maxiter:
                    raise ValueError(
                        f"no sign change found in {it - 1} iterations")
                if math.isfinite(lower):
                    ol, of = lower, f_lower
                    lower = lower - dl
                    f_lower = f(lower)
                    if math.isnan(f_lower):
                        lower, f_lower, dl = ol, of, dl / 4
                if math.isfinite(upper):
                    ou, ofu = upper, f_upper
                    upper = upper + du
                    f_upper = f(upper)
                    if math.isnan(f_upper):
                        upper, f_upper, du = ou, ofu, du / 4
                dl, du = 2 * dl, 2 * du
        else:
            d = delta_of(lower)
            while sig * f_lower > 0:
                it += 1
                if it > maxiter:
                    raise ValueError(
                        f"no sign change found in {it - 1} iterations")
                lower = lower - d
                f_lower = f(lower)
                d *= 2
            d = delta_of(upper)
            while sig * f_upper < 0:
                it += 1
                if it > maxiter:
                    raise ValueError(
                        f"no sign change found in {it - 1} iterations")
                upper = upper + d
                f_upper = f(upper)
                d *= 2
    if not (_sign(f_lower) * _sign(f_upper) <= 0):
        raise ValueError("f() values at end points not of opposite sign")
    if do_x and it:
        f_low = truncate(f_lower)
        f_upp = truncate(f_upper)
    return _nm._zeroin2(lower, upper, f_low, f_upp, f, tol, maxiter)


@dataclass
class PowerHTest:
    """R's ``power.htest`` — the result of a power calculation.

    ``params`` holds the solved parameters in R's print order (the one that
    was ``None`` on input is now filled in)."""

    method: str
    params: dict
    note: Optional[str] = None

    def __repr__(self) -> str:
        out = ["", f"     {self.method}", ""]
        for k, v in self.params.items():
            out.append(f"{k:>15} = {_fmt(v) if isinstance(v, float) else v}")
        out.append(f"\nNOTE: {self.note}\n" if self.note else "")
        return "\n".join(out)


def _assert_prob(x, name):
    if x is not None and (not np.isreal(x) or x < 0 or x > 1):
        raise ValueError(f"'{name}' must be numeric in [0, 1]")


def power_t_test(n=None, delta=None, sd=1.0, sig_level=0.05, power=None,
                 type: str = "two.sample", alternative: str = "two.sided",
                 strict: bool = False, tol=None) -> PowerHTest:
    """R's ``power.t.test`` — power of the one/two-sample/paired t-test.

    Exactly one of ``n``, ``delta``, ``sd``, ``power``, ``sig_level`` must be
    ``None`` and is solved for (via the ported ``uniroot`` with ``extendInt``);
    the rest are given. Faithful port of ``power.t.test`` using the ported
    noncentral-t ``pt``/``qt`` — bit-exact to R.
    """
    if sum(v is None for v in (n, delta, sd, power, sig_level)) != 1:
        raise ValueError("exactly one of 'n', 'delta', 'sd', 'power', and "
                         "'sig_level' must be None")
    _assert_prob(sig_level, "sig_level")
    _assert_prob(power, "power")
    tsample = {"one.sample": 1, "two.sample": 2, "paired": 1}[type]
    tside = {"one.sided": 1, "two.sided": 2}[alternative]
    if tside == 2 and delta is not None:
        delta = abs(delta)

    def pbody(n_, delta_, sd_, sig_):
        nu = max(1e-7, n_ - 1) * tsample
        ncp = math.sqrt(n_ / tsample) * delta_ / sd_
        qu = float(_dist.qt(sig_ / tside, nu, lower_tail=False))
        val = float(_dist.pt(qu, nu, ncp=ncp, lower_tail=False))
        if strict and tside == 2:
            val += float(_dist.pt(-qu, nu, ncp=ncp, lower_tail=True))
        return val

    if power is None:
        power = pbody(n, delta, sd, sig_level)
    elif n is None:
        n = _uniroot_ext(lambda v: pbody(v, delta, sd, sig_level) - power,
                         2, 1e7, "upX", tol)
    elif sd is None:
        sd = _uniroot_ext(lambda v: pbody(n, delta, v, sig_level) - power,
                          delta * 1e-7, delta * 1e7, "downX", tol)
    elif delta is None:
        delta = _uniroot_ext(lambda v: pbody(n, v, sd, sig_level) - power,
                             sd * 1e-7, sd * 1e7, "upX", tol)
    else:  # sig_level is None
        sig_level = _uniroot_ext(
            lambda v: pbody(n, delta, sd, v) - power, 1e-10, 1 - 1e-10,
            "yes", tol)
    note = {"paired": "n is number of *pairs*, sd is std.dev. of "
            "*differences* within pairs",
            "two.sample": "n is number in *each* group"}.get(type)
    method = ({"one.sample": "One-sample", "two.sample": "Two-sample",
               "paired": "Paired"}[type] + " t test power calculation")
    return PowerHTest(method, {"n": n, "delta": delta, "sd": sd,
                               "sig.level": sig_level, "power": power,
                               "alternative": alternative}, note)


def power_prop_test(n=None, p1=None, p2=None, sig_level=0.05, power=None,
                    alternative: str = "two.sided", strict: bool = False,
                    tol=None) -> PowerHTest:
    """R's ``power.prop.test`` — power of the two-sample proportion test.

    Exactly one of ``n``, ``p1``, ``p2``, ``power``, ``sig_level`` must be
    ``None`` and is solved for. Faithful port of ``power.prop.test`` using the
    ported ``pnorm``/``qnorm`` — bit-exact to R.
    """
    if sum(v is None for v in (n, p1, p2, power, sig_level)) != 1:
        raise ValueError("exactly one of 'n', 'p1', 'p2', 'power', and "
                         "'sig_level' must be None")
    _assert_prob(sig_level, "sig_level")
    _assert_prob(power, "power")
    tside = {"one.sided": 1, "two.sided": 2}[alternative]

    def pbody(n_, p1_, p2_, sig_):
        qu = float(_dist.qnorm(sig_ / tside, lower_tail=False))
        d = abs(p1_ - p2_)
        if strict and tside == 2:
            pbar = (p1_ + p2_) / 2
            vbar = pbar * (1 - pbar)
            v1 = p1_ * (1 - p1_)
            v2 = p2_ * (1 - p2_)
            denom = math.sqrt(v1 + v2)
            return (float(_dist.pnorm(
                        (math.sqrt(n_) * d - qu * math.sqrt(2 * vbar)) / denom))
                    + float(_dist.pnorm(
                        (math.sqrt(n_) * d + qu * math.sqrt(2 * vbar)) / denom,
                        lower_tail=False)))
        return float(_dist.pnorm(
            (math.sqrt(n_) * d - qu * math.sqrt((p1_ + p2_)
                                                * (1 - (p1_ + p2_) / 2)))
            / math.sqrt(p1_ * (1 - p1_) + p2_ * (1 - p2_))))

    if power is None:
        power = pbody(n, p1, p2, sig_level)
    elif n is None:
        n = _uniroot_ext(lambda v: pbody(v, p1, p2, sig_level) - power,
                         1, 1e7, "upX", tol)
    elif p1 is None:
        p1 = _uniroot_ext(lambda v: pbody(n, v, p2, sig_level) - power,
                          0, p2, "yes", tol)
    elif p2 is None:
        p2 = _uniroot_ext(lambda v: pbody(n, p1, v, sig_level) - power,
                          p1, 1, "yes", tol)
    else:  # sig_level is None
        sig_level = _uniroot_ext(
            lambda v: pbody(n, p1, p2, v) - power, 1e-10, 1 - 1e-10, "upX", tol)
    return PowerHTest(
        "Two-sample comparison of proportions power calculation",
        {"n": n, "p1": p1, "p2": p2, "sig.level": sig_level, "power": power,
         "alternative": alternative}, "n is number in *each* group")


def power_anova_test(groups=None, n=None, between_var=None, within_var=None,
                     sig_level=0.05, power=None) -> PowerHTest:
    """R's ``power.anova.test`` — power of the balanced one-way ANOVA F-test.

    Exactly one of ``groups``, ``n``, ``between_var``, ``within_var``,
    ``power``, ``sig_level`` must be ``None`` and is solved for. Faithful port
    of ``power.anova.test`` using the ported noncentral-F ``pf``/``qf`` —
    bit-exact to R.
    """
    if sum(v is None for v in (groups, n, between_var, within_var, power,
                               sig_level)) != 1:
        raise ValueError("exactly one of 'groups', 'n', 'between_var', "
                         "'within_var', 'power', and 'sig_level' must be None")
    if groups is not None and groups < 2:
        raise ValueError("number of groups must be at least 2")
    if n is not None and n < 2:
        raise ValueError("number of observations in each group must be >= 2")
    _assert_prob(sig_level, "sig_level")
    _assert_prob(power, "power")

    def pbody(groups_, n_, bv, wv, sig_):
        lam = (groups_ - 1) * n_ * (bv / wv)
        qf_ = float(_dist.qf(sig_, groups_ - 1, (n_ - 1) * groups_,
                             lower_tail=False))
        return float(_dist.pf(qf_, groups_ - 1, (n_ - 1) * groups_, ncp=lam,
                              lower_tail=False))

    if power is None:
        power = pbody(groups, n, between_var, within_var, sig_level)
    elif groups is None:
        groups = _uniroot_ext(
            lambda v: pbody(v, n, between_var, within_var, sig_level) - power,
            2, 1e2)
    elif n is None:
        n = _uniroot_ext(
            lambda v: pbody(groups, v, between_var, within_var, sig_level)
            - power, 2, 1e5)
    elif within_var is None:
        within_var = _uniroot_ext(
            lambda v: pbody(groups, n, between_var, v, sig_level) - power,
            between_var * 1e-7, between_var * 1e7)
    elif between_var is None:
        between_var = _uniroot_ext(
            lambda v: pbody(groups, n, v, within_var, sig_level) - power,
            within_var * 1e-7, within_var * 1e7)
    else:  # sig_level is None
        sig_level = _uniroot_ext(
            lambda v: pbody(groups, n, between_var, within_var, v) - power,
            1e-10, 1 - 1e-10)
    return PowerHTest(
        "Balanced one-way analysis of variance power calculation",
        {"groups": groups, "n": n, "between.var": between_var,
         "within.var": within_var, "sig.level": sig_level, "power": power},
        "n is number in each group")


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
