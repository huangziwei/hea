from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.linalg import qr, solve_triangular
from scipy.linalg.lapack import dgeqrf, dormqr
from ..R import distributions as _dist
from ..R import linalg as _linalg

from ..R import nmath as _nmath
from ..formula import (
    Name,
    _eval_atom,
    _lhs_referenced_cols,
    _na_mask_with_matrix_cols,
    expand,
    materialize,
    parse,
    prepare_design,
    referenced_columns,
)
from ..utils import (
    _dig_tst,
    format_df,
    format_pval,
    format_signif,
    format_signif_jointly,
    significance_code,
)

__all__ = ["lm"]


def _row_frame(values: np.ndarray, columns: list[str]) -> pl.DataFrame:
    """Build a 1-row pl.DataFrame from a flat numpy array + column names.

    Constructs straight from the 2-D ``(1, p)`` array (≈2× faster than a
    dict-of-singleton-lists for the per-fit coefficient frames)."""
    arr = np.asarray(values, dtype=float).reshape(1, -1)
    return pl.DataFrame(arr, schema=list(columns))


def _zapsmall(x: np.ndarray, digits: int) -> np.ndarray:
    """R's ``zapsmall(x, digits)`` — round ``x``, zeroing entries negligible
    relative to the largest magnitude: ``round(x, digits − ⌊log10(max|x|)⌋)``
    (clamped at 0 decimals). Used on the residual 5-number summary so a
    numerical-noise quantile prints as ``0`` rather than e.g. ``1.2e-16``."""
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    mx = float(np.max(np.abs(finite)))
    nd = max(0, int(digits - int(np.log10(mx)))) if mx > 0 else digits
    return np.round(x, nd)


def _subset_keep(n: int, subset) -> list[int]:
    """0-based row indices kept by R's ``subset=`` argument.

    Single source of truth for ``lm()`` / ``glm()`` / ``gmm()`` subsetting,
    so the data frame and the ``weights`` vector get filtered identically.
    Three forms: boolean mask (length == nrow), non-negative 0-based
    indices (keep), or negative indices (drop). Negative indices use
    Python's ``range(n) − k`` semantics: ``-1`` is the last row, ``-2``
    the second-to-last, etc. Mixing non-negative and negative isn't
    valid and is rejected.

    R / dplyr ``subset=`` is 1-based; hea follows Python indexing.
    """
    if isinstance(subset, (pl.Series, np.ndarray, list, tuple)):
        arr = np.asarray(subset)
        if arr.dtype == bool:
            return [int(i) for i in np.flatnonzero(arr)]
        ints = arr.astype(int)
    else:
        # scalar
        ints = np.asarray([int(subset)])
    has_nonneg = bool((ints >= 0).any())
    has_neg = bool((ints < 0).any())
    if has_nonneg and has_neg:
        raise ValueError(
            "subset=: cannot mix non-negative and negative indices"
        )
    if has_neg:
        # Negative indices drop the corresponding rows (Python convention:
        # -1 is the last row).
        drop_set = {n + int(idx) for idx in ints.tolist()}
        return [i for i in range(n) if i not in drop_set]
    # Non-negative indices: keep those rows.
    return [int(i) for i in ints.tolist()]


def _apply_subset(data: pl.DataFrame, subset) -> pl.DataFrame:
    """R: ``subset=`` filter for ``lm()`` / ``glm()`` / etc. — see
    :func:`_subset_keep` for the index semantics."""
    return data[_subset_keep(data.height, subset)]


def _referenced_model_cols(formula: str, data: pl.DataFrame) -> list[str]:
    """Columns of ``data`` referenced by the formula (response + RHS), in the
    original column order — R's model-frame variable set."""
    parsed = parse(formula)
    columns = set(data.columns)
    if parsed.lhs is None:
        lhs_cols: set[str] = set()
    elif isinstance(parsed.lhs, Name):
        lhs_cols = {parsed.lhs.ident} & columns
    else:
        lhs_cols = _lhs_referenced_cols(parsed.lhs, columns)
    expanded = expand(parsed, data_columns=list(data.columns))
    ref = (referenced_columns(expanded) | lhs_cols) & columns
    return [c for c in data.columns if c in ref]  # preserve original order


def _model_frame_keep_mask(formula: str, data: pl.DataFrame) -> np.ndarray:
    """Boolean keep-mask matching ``prepare_design``'s ``na.omit`` policy.

    Mirrors R's ``na.action = na.omit`` on the model frame: a row is
    dropped when the response or any RHS-referenced column is NA. Used to
    carry the ``weights`` vector through the *same* row-drops the design
    matrix gets, so weights stay aligned to the rows actually fit (R keeps
    weights inside the model frame, so this is automatic there).

    Reuses ``prepare_design``'s own ``_na_mask_with_matrix_cols`` so the
    mask can't drift from the rows ``prepare_design`` actually keeps.
    """
    na_cols = set(_referenced_model_cols(formula, data))
    if not na_cols:
        return np.ones(data.height, dtype=bool)
    return _na_mask_with_matrix_cols(data, na_cols)


def _resolve_subset(subset, data: pl.DataFrame):
    """Evaluate R's ``subset=`` to a form :func:`_subset_keep` accepts.

    R evaluates ``subset`` as an expression in the data frame
    (``subset = Area > 10``). hea accepts, in addition to a bool mask /
    integer indices (returned untouched):

    * a **polars expression** — ``pl.col("Area") > 10`` (native, full power);
    * an **R-style string** — ``"Area > 10 & Elevation < 200"`` — run through
      hea's R→Python translator first so R's operator precedence (``&`` /
      ``|`` bind *looser* than comparisons, the opposite of Python) is
      honored, then evaluated with the frame's columns bound as polars
      Series. Returns a boolean ndarray for the string / expression forms.
    """
    if isinstance(subset, str):
        from ..translate import translate_r  # local: avoid import cycle
        py_src = translate_r(subset)
        ns = {c: data[c] for c in data.columns}
        result = eval(py_src, {"__builtins__": {}}, ns)  # noqa: S307 — R-style NSE
        return np.asarray(result).astype(bool)
    if isinstance(subset, pl.Expr):
        return data.select(subset.alias("__subset__"))["__subset__"].to_numpy().astype(bool)
    return subset


def _drop_aliased_cols(X_df: pl.DataFrame, tol: float = 1e-7, *,
                       values: np.ndarray | None = None) -> list[str]:
    """Identify linearly-dependent columns in a design matrix.

    With the Rust kernel present, use R's **exact** ``dqrdc2`` pivot/rank
    (``hea._rs`` ``dqrls``, ``ref/r-stats/src/dqrdc2.f``): the columns deferred
    to pivot positions ``rank+1..p`` are precisely the ones R's ``lm`` aliases,
    so hea drops bit-identically what R drops and the rank decision is
    *deterministic* (immune to BLAS-bistable rank flakes). ``tol=1e-7`` is R's
    ``lm.fit`` default.

    Fallback (no Rust / ``HEA_NO_RS``): a left-to-right ``dgeqrf`` +
    relative-tolerance heuristic — the *later* of two collinear columns is
    flagged (``dqrdc2`` prefers earlier columns: intercept first, then RHS terms
    in order), so the intercept isn't dropped when a later predictor is a
    constant + centered copy. Either way ``df_residuals`` reflects effective rank.

    ``values`` is an optional precomputed numpy view of ``X_df`` (the design's
    F-order ``X_values`` fast lane); when supplied, the screen reads it directly
    and skips ``X_df.to_numpy()``. It is read-only here (``dgeqrf`` does not
    overwrite its input), so reusing the buffer shared with ``X_df`` is safe.
    """
    if X_df.height == 0 or X_df.width == 0:
        return []
    X = values if values is not None else X_df.to_numpy().astype(float)
    cols = X_df.columns
    p = X.shape[1]

    # Fast screen (always): the in-order dgeqrf R-diagonal. The common case —
    # a clearly full-rank design — is decided here with no Rust call, so the hot
    # path keeps its speed (the dqrdc2 factorization is ~2-3× the Accelerate one,
    # per the receipt). A column is "suspect" when its pivot diagonal has
    # collapsed relative to the column's original norm (R's dqrdc2 negligibility,
    # relative tol = 1e-7) — only then is an *exact* rank/pivot decision needed.
    qr_c, _tau, _wk, _info = dgeqrf(X)
    diag_abs = np.abs(np.diag(qr_c))
    if diag_abs.size == 0:
        return []
    col_norms = np.linalg.norm(X, axis=0)
    ref = np.where(col_norms > 0.0, col_norms, 1.0)
    suspect = bool(np.any(diag_abs < ref * tol))
    if not suspect:
        return []                                   # clearly full rank — fast path

    if _linalg._rs_dqrls_rank is not None:
        # Rank-deficient → resolve EXACTLY with R's dqrdc2 (Rust): the columns
        # deferred to pivot positions rank+1..p are bit-identically R's aliased
        # set, and the decision is deterministic (no BLAS-bistable rank flake).
        # dqrls_rank is the lean path — rank+pivot only, no QR/coef marshalling.
        rank, pivot = _linalg.dqrls_rank(X, tol)
        if rank >= p:
            return []
        return [cols[i] for i in sorted(int(pivot[j]) - 1 for j in range(rank, p))]
    # no Rust: the strict-tolerance heuristic (later of two collinear cols flagged)
    tol_h = max(float(diag_abs.max()), 1.0) * np.finfo(float).eps * max(X.shape)
    return [cols[i] for i, v in enumerate(diag_abs) if v <= tol_h]


def _lowess(x, y, frac=2 / 3, it=3):
    """Cleveland's LOWESS — local linear smoother with robustness reweighting.

    Matches the smoother R uses in `panel.smooth` for `plot.lm`.
    Returns (x_sorted, y_smooth). O(n^2) memory; pass `smooth=False` if n is huge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return x, y
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    r = max(2, int(np.ceil(frac * n)))

    dists = np.abs(xs[:, None] - xs[None, :])
    bw = np.partition(dists, r - 1, axis=1)[:, r - 1]
    bw = np.where(bw == 0, 1.0, bw)
    W = np.clip(dists / bw[:, None], 0, 1)
    W = (1 - W ** 3) ** 3

    yhat = ys.copy()
    delta = np.ones(n)
    for _ in range(it + 1):
        Wd = W * delta[None, :]
        S = Wd.sum(axis=1)
        Sx = Wd @ xs
        Sxx = Wd @ (xs * xs)
        Sy = Wd @ ys
        Sxy = Wd @ (xs * ys)
        det = S * Sxx - Sx * Sx
        ok = det > 0
        safe_det = np.where(ok, det, 1.0)
        safe_S = np.where(S > 0, S, 1.0)
        a_local = (Sxx * Sy - Sx * Sxy) / safe_det
        b_local = (S * Sxy - Sx * Sy) / safe_det
        wmean = Sy / safe_S
        yhat = np.where(ok, a_local + b_local * xs, wmean)
        resid = ys - yhat
        s = np.median(np.abs(resid))
        if s == 0:
            break
        u = np.clip(resid / (6 * s), -1, 1)
        delta = (1 - u * u) ** 2
    return xs, yhat


def _label_top_n(ax, xs, ys, scores, n=3, indices=None):
    """Annotate the n points with the largest |scores|."""
    if not n:
        return
    n = min(int(n), len(scores))
    if n == 0:
        return
    top = np.argsort(-np.abs(np.asarray(scores)))[:n]
    for i in top:
        label = str(int(indices[i]) if indices is not None else int(i))
        ax.annotate(
            label,
            (xs[i], ys[i]),
            fontsize=8,
            color="black",
            xytext=(3, 3),
            textcoords="offset points",
        )


def _qq_plot(
    ax, vals, labels=None, label_n=3,
    xlabel="Theoretical Quantiles",
    ylabel="Standardized Residuals",
    title="Normal Q-Q",
):
    """Normal Q-Q on ax with quartile-based reference line, label top |vals|.

    `labels` optionally maps an index to a custom annotation string;
    otherwise the integer index is used.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n < 2:
        return
    sort_idx = np.argsort(vals)
    v = vals[sort_idx]
    a = 3.0 / 8.0 if n <= 10 else 0.5
    probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
    q = _nmath.qnorm5_vec(probs)
    ax.scatter(q, v, facecolor="none", edgecolor="black")
    ry1, ry3 = np.quantile(v, [0.25, 0.75])
    qx1, qx3 = _nmath.qnorm5_vec(np.array([0.25, 0.75]))
    slope = (ry3 - ry1) / (qx3 - qx1)
    intercept = ry1 - slope * qx1
    xs = np.array([q.min(), q.max()])
    ax.plot(xs, slope * xs + intercept, color="black", linestyle="--")
    if label_n:
        n_lab = min(int(label_n), n)
        top = np.argsort(-np.abs(vals))[:n_lab]
        rank = np.empty(n, dtype=int)
        rank[sort_idx] = np.arange(n)
        for orig_i in top:
            pos = rank[orig_i]
            text = str(labels[orig_i]) if labels is not None else str(int(orig_i))
            ax.annotate(
                text, (q[pos], v[pos]),
                fontsize=8, color="black",
                xytext=(3, 3), textcoords="offset points",
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


class SummaryLm:
    """R: ``summary.lm()`` return value.

    Mirrors the components R exposes via ``$``:

    - ``.sigma`` (Residual SE), partial-matched as ``["sig"]``
    - ``.r_squared`` / ``.adj_r_squared``
    - ``.fstatistic`` — numpy array ``[F, df1, df2]`` (R's vector shape)
    - ``.coefficients`` — DataFrame with Estimate / Std. Error / t / p
      (R's coefficient matrix)
    - ``.df`` — R's df triple ``(rank, residual_df, total_columns)``
      (total columns counts aliased/dropped columns, like ``NCOL(qr)``)
    - ``.cov_unscaled`` — ``(X'X)^-1`` matrix
    - ``.residuals`` — raw residual Series

    ``__repr__`` prints the R-style summary block. ``__getitem__``
    supports R's ``$`` access including partial-name matching: ``["sig"]``
    resolves to ``sigma`` if it's the unique prefix match.
    """

    def __init__(self, model, *, digits: int = 4, cor: bool = False):
        self._model = model
        self._digits = digits
        self._cor = cor
        # Components in R's summary.lm order — names use ``.`` like R
        # so ``["cov.unscaled"]`` etc. round-trip in translated code.
        self.sigma = float(np.sqrt(model.sigma_squared))
        self.r_squared = float(model.r_squared)
        self.adj_r_squared = float(model.r_squared_adjusted)
        self.fstatistic = (
            np.array([model.fstats, model.df_model, model.df_residuals])
            if model.fstats is not None else None
        )
        # R: ``summary.lm`` sets ``ans$df <- c(p, rdf, NCOL(Qr$qr))`` — that's
        # (rank incl. intercept, residual df, *total* columns incl. aliased),
        # NOT (df_model, rdf, kept_cols). ``model.p`` is the kept-column count,
        # which equals the rank (hea drops aliased columns up front), and
        # ``len(model._full_names)`` is the pre-drop column total.
        self.df = (model.p, model.df_residuals, len(model._full_names))
        self.cov_unscaled = np.asarray(model.XtXinv)
        self.residuals = model.residuals
        # Coefficients matrix — R's ``summary(lm)$coef`` is a numeric
        # matrix; we expose it as a NumPy 2D array with row/col labels
        # held alongside.
        bhat = np.asarray(model._bhat_arr, dtype=float)
        se = np.asarray(model._se_bhat_arr, dtype=float)
        t = np.divide(bhat, se, out=np.full_like(bhat, np.nan), where=se > 0)
        p = 2.0 * _dist.pt(np.abs(t), model.df_residuals, lower_tail=False)
        self.coefficients = np.column_stack([bhat, se, t, p])
        self._coef_rownames = list(model.column_names)
        self._coef_colnames = ("Estimate", "Std. Error", "t value", "Pr(>|t|)")
        # R's ``summary(lm)$aliased`` — a logical over the *full* column set
        # flagging coefficients dropped for singularity (NA in coef()). The
        # ``$coefficients`` matrix above holds only the estimable rows, as in R.
        self.aliased = ~np.isfinite(np.asarray(model._bhat_disp, dtype=float))

    # ---- R-style ``$`` access (partial name matching) ---------------

    _ALIASES = {
        "sig": "sigma",
        "sigma": "sigma",
        "r.squared": "r_squared",
        "adj.r.squared": "adj_r_squared",
        "fstatistic": "fstatistic",
        "df": "df",
        "cov.unscaled": "cov_unscaled",
        "coefficients": "coefficients",
        "coef": "coefficients",
        "residuals": "residuals",
    }

    def __getitem__(self, key):
        # R's partial-name matching: ``$sig`` resolves to ``$sigma`` if
        # exactly one alias has it as a prefix.
        key = str(key)
        if key in self._ALIASES:
            return getattr(self, self._ALIASES[key])
        matches = [a for a in self._ALIASES if a.startswith(key)]
        if len(matches) == 1:
            return getattr(self, self._ALIASES[matches[0]])
        if len(matches) > 1:
            raise KeyError(
                f"summary(lm)[{key!r}] ambiguous; matches: {matches}"
            )
        raise KeyError(f"summary(lm) has no component {key!r}")

    def __getattr__(self, name):
        # ``.sig`` partial-match via attribute access too.
        try:
            return self.__getitem__(name)
        except KeyError as e:
            raise AttributeError(str(e))

    def __repr__(self) -> str:
        return _format_summary_lm(self._model, self._digits, self._cor)


class SummaryMlm:
    """R's ``summary.mlm`` — a list of per-response ``summary.lm`` objects.

    Printed as ``Response <name> :`` blocks; indexable by 1-based position,
    0-based position, or the ``"Response <name>"`` / ``<name>`` key, returning
    the underlying :class:`SummaryLm` for that response column.
    """

    def __init__(self, model, *, digits: int = 4, cor: bool = False):
        self._names = list(model._response_names)
        self._summaries = {
            name: model._mlm_models[name].summary(digits=digits, cor=cor)
            for name in self._names
        }

    def __len__(self) -> int:
        return len(self._names)

    @property
    def names(self) -> list[str]:
        return [f"Response {n}" for n in self._names]

    def __getitem__(self, key):
        if isinstance(key, int):
            # R's summary(m)[[i]] is 1-based; allow Python negatives too.
            return self._summaries[self._names[key - 1 if key > 0 else key]]
        k = key[len("Response "):] if str(key).startswith("Response ") else key
        return self._summaries[k]

    def __iter__(self):
        return iter(self._summaries.values())

    def __repr__(self) -> str:
        return "\n\n".join(
            f"Response {name} :\n\n{self._summaries[name]!r}"
            for name in self._names
        )


def _format_summary_lm(model, digits: int, cor: bool) -> str:
    """Build the R-style ``summary.lm`` print block.

    Pulled out of :meth:`lm.summary` so :class:`SummaryLm` can call it
    from ``__repr__`` without rebuilding the dependency on ``model``.
    """
    docstring = f"Formula: {model.formula}\n\n"
    docstring += "\n".join(model._residuals_lines(digits=digits))
    docstring += "\n\n" + model._coef_header() + "\n"

    # Display-width arrays: equal to the fit columns normally, widened to the
    # full column set with NA for aliased columns when rank-deficient.
    bhat_disp = model._bhat_disp
    se_disp = model._se_disp
    with np.errstate(divide="ignore", invalid="ignore"):
        t_arr = bhat_disp / se_disp
    p_arr = model._p_disp
    sig = significance_code(p_arr)
    ci_low_col, ci_hi_col = model.ci_bhat.columns[1], model.ci_bhat.columns[2]
    ci_low_arr = model.ci_bhat[ci_low_col].to_numpy()
    ci_hi_arr = model.ci_bhat[ci_hi_col].to_numpy()
    est_s, se_s = format_signif_jointly(
        [bhat_disp, se_disp], digits=digits,
    )
    cilo_s, cihi_s = format_signif_jointly(
        [ci_low_arr, ci_hi_arr], digits=digits,
    )
    res = pl.DataFrame({
        "": model._names_disp,
        "Estimate": est_s,
        "Std. Error": se_s,
        ci_low_col: cilo_s,
        ci_hi_col: cihi_s,
        "t value": format_signif(t_arr, digits=digits),
        "Pr(>|t|)": format_pval(p_arr, digits=_dig_tst(digits)),
        " ": sig,
    })
    num_cols = ("Estimate", "Std. Error", ci_low_col, ci_hi_col,
                "t value", "Pr(>|t|)")
    # Aliased rows print "NA" in every numeric column (R's printCoefmat
    # ``na.print="NA"``); the formatters emit "NaN" for those NaN entries.
    aliased = ~np.isfinite(np.asarray(bhat_disp, dtype=float))
    if aliased.any():
        res = res.with_columns([
            pl.when(pl.Series(aliased)).then(pl.lit("NA")).otherwise(pl.col(c)).alias(c)
            for c in num_cols
        ])
    num_align = {c: "right" for c in num_cols}
    docstring += format_df(res, align=num_align)
    docstring += "\n---"
    docstring += "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"

    docstring += (
        f"\n\nn = {model.n}, p = {model.p}, "
        f"Residual SE = {np.sqrt(model.sigma_squared):.3f} "
        f"on {model.df_residuals} DF\n"
    )
    docstring += (
        f"R-Squared = {model.r_squared:.4f}, "
        f"adjusted R-Squared = {model.r_squared_adjusted:.4f}\n"
    )
    if model.fstats is not None:
        fmt = ".2f" if model.f_p_value > 1e-5 else "e"
        docstring += (
            f"F-statistics = {model.fstats:.4f} on "
            f"{model.df_model} and {model.df_residuals} DF, "
            f"p-value: {model.f_p_value:{fmt}}\n\n"
        )
    docstring += (
        f"Log Likelihood = {model.loglike:.4f}, "
        f"AIC = {model.AIC:.4f}, BIC = {model.BIC:.4f}"
    )
    if cor and model.V_bhat.shape[0] >= 2:
        docstring += "\n\nCorrelation of Coefficients:\n"
        docstring += model._correlation_block()
    return docstring


class lm:
    def __new__(
        cls,
        formula: str = None,
        data: pl.DataFrame = None,
        weights=None,
        method: str = "qr",
        subset=None,
        na_action: str = "omit",
        contrasts=None,
        singular_ok: bool = True,
        offset=None,
    ):
        # R: ``lm(..., method="model.frame")`` returns the *model frame*, not a
        # fit. A constructor can't return a non-``lm``, so intercept in
        # ``__new__`` (standard factory idiom): returning a non-instance skips
        # ``__init__``. Every other ``method`` builds an ``lm`` as usual.
        if method == "model.frame":
            return cls._model_frame(
                formula, data, subset=subset, na_action=na_action
            )
        return super().__new__(cls)

    @staticmethod
    def _model_frame(formula, data, *, subset=None, na_action="omit"):
        """R's ``lm(method="model.frame")``: the model frame — the formula's
        referenced columns (response + RHS) after ``subset=`` and the
        ``na.action`` row-drops — with no fit. ``na.fail`` errors on any NA;
        ``omit``/``exclude`` drop NA rows (R keeps NA rows under exclude in the
        frame, but the fit-facing frame is complete-case either way here)."""
        norm = {"omit": "omit", "na.omit": "omit", "exclude": "omit",
                "na.exclude": "omit", "fail": "fail", "na.fail": "fail"}
        na = norm.get(str(na_action))
        if na is None:
            raise ValueError(f"na_action must be omit/exclude/fail; got {na_action!r}")
        if subset is not None:
            keep = _subset_keep(data.height, _resolve_subset(subset, data))
            data = data[keep]
        mask = _model_frame_keep_mask(formula, data)
        if na == "fail" and not mask.all():
            raise ValueError("missing values in object")  # R: na.fail
        mf = data.select(_referenced_model_cols(formula, data))
        return mf.filter(pl.Series(mask)) if not mask.all() else mf

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        weights: Union[None, np.array] = None,
        method: str = "qr",
        subset=None,
        na_action: str = "omit",
        contrasts=None,
        singular_ok: bool = True,
        offset: Union[None, np.array] = None,
    ):

        # R's `cbind(y1, y2, ...) ~ rhs` fits a multivariate linear model
        # (class "mlm") — m independent fits sharing one X/QR. Detect it from
        # the LHS and route to the mlm builder (which wraps m per-column `lm`s).
        from hea.formula import parse as _parse, _multivariate_lhs_specs
        _lhs = _parse(formula).lhs
        _mv = _multivariate_lhs_specs(_lhs) if _lhs is not None else None
        if _mv is not None:
            self._init_mlm(
                formula, data, _mv, weights=weights, method=method,
                subset=subset, na_action=na_action, contrasts=contrasts,
                singular_ok=singular_ok, offset=offset,
            )
            return
        self._is_mlm = False

        # meta
        self.formula = formula
        # R supports only method="qr" for the fit (method="model.frame" is
        # intercepted in __new__). Any other value warns and falls back to qr
        # (lm.R:38-39) — hea keeps no alternative solvers.
        if method != "qr":
            import warnings
            warnings.warn(
                f"method = '{method}' is not supported. Using 'qr'",
                stacklevel=2,
            )
            method = "qr"
        self.method = method
        self.contrasts = contrasts

        # R's ``na.action``: how to treat rows with NA in a referenced
        # column. ``omit`` (default) drops them — hea's established
        # behaviour, matching R's default. ``fail`` errors on any NA.
        # ``exclude`` fits on the complete cases but pads the *accessors*
        # (resid / fitted / predict) back to the model-frame length with NA
        # (see _na_pad). Accept R-style aliases.
        _na_norm = {"omit": "omit", "na.omit": "omit",
                    "exclude": "exclude", "na.exclude": "exclude",
                    "fail": "fail", "na.fail": "fail",
                    "pass": "pass", "na.pass": "pass"}
        self.na_action = _na_norm.get(str(na_action))
        if self.na_action is None:
            raise ValueError(
                "na_action must be one of 'omit' / 'exclude' / 'fail' / 'pass'; "
                f"got {na_action!r}"
            )

        # ``weights`` are validated against the input frame, then carried
        # through the *same* ``subset=`` + ``na.omit`` row-drops the data
        # gets, so they stay aligned to the rows actually fit. R keeps
        # weights inside the model frame, so this alignment is automatic
        # there; hea takes a bare vector, so we replay the row-drops on it.
        if weights is not None:
            w_arr = np.asarray(weights, dtype=float).reshape(-1)
            if w_arr.shape[0] != data.height:
                raise ValueError(
                    "Length of weights should be the same as the number of rows in the dataframe"
                )
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
                # R: lm.wfit() — `any(w < 0 | is.na(w))` is a hard error.
                raise ValueError("missing or negative weights not allowed")
        else:
            w_arr = None

        # ``offset=`` (R's lm ``offset`` argument): a length-n vector added to
        # the linear predictor. R sums it with any in-formula ``offset(...)``
        # term; we carry it through the same ``subset=`` + ``na.omit`` row-drops
        # as ``weights`` so it stays aligned to the fit rows, then fold it into
        # ``self._offset`` (see the offset summation below).
        if offset is not None:
            off_arg = np.asarray(offset, dtype=float).reshape(-1)
            if off_arg.shape[0] != data.height:
                raise ValueError(
                    "Length of offset should be the same as the number of rows in the dataframe"
                )
        else:
            off_arg = None

        # R's ``subset=`` filters rows before fitting. Accepts an R-style
        # expression (string / polars expr) evaluated in the frame, a bool
        # mask, or 0-based keep / negative drop indices (see _resolve_subset
        # / _subset_keep). Filter weights in lockstep through the same index.
        if subset is not None:
            subset = _resolve_subset(subset, data)
            keep = _subset_keep(data.height, subset)
            data = data[keep]
            if w_arr is not None:
                w_arr = w_arr[keep]
            if off_arg is not None:
                off_arg = off_arg[keep]

        # na.action keep-mask over the (post-subset) frame: which rows survive
        # na.omit. Drives ``fail`` (error), ``exclude`` (output padding), and
        # the weight/offset alignment below. ``prepare_design`` recomputes the
        # same na.omit internally, so for the common path (omit/pass with no
        # weights/offset) the lm-side mask is never read — skip it and avoid the
        # duplicate ``_na_mask_with_matrix_cols`` pass (it is the redundant work
        # the fit profile flagged). The ``exclude``-padding readers all guard on
        # ``na_action == "exclude"`` first, so ``None`` is safe otherwise.
        self._n_full = data.height  # model-frame length (post-subset, pre-na)
        need_mask = (
            self.na_action in ("fail", "exclude")
            or w_arr is not None
            or off_arg is not None
        )
        if need_mask:
            na_keep = _model_frame_keep_mask(formula, data)
            if self.na_action == "fail" and not na_keep.all():
                raise ValueError("missing values in object")  # R: na.fail
            if self.na_action == "pass":
                # R's na.pass: keep every row (NA flows into X/y; fit may be NaN).
                na_keep = np.ones(data.height, dtype=bool)
            self._na_mask = na_keep
            if w_arr is not None and not na_keep.all():
                w_arr = w_arr[na_keep]
            if off_arg is not None and not na_keep.all():
                off_arg = off_arg[na_keep]
        else:
            self._na_mask = None

        self.data = data
        self.weights = w_arr
        self._offset_arg = off_arg

        # R's predvars: capture each poly/bs/ns/scale call's training params at
        # fit so predict() replays them on new data instead of recomputing.
        self._basis_state: dict = {}
        d = prepare_design(formula, data, contrasts=contrasts,
                           na_action="pass" if self.na_action == "pass" else "omit",
                           basis_state=self._basis_state)
        self._expanded = d.expanded
        self._design_data = d.data
        self.X = d.X
        self.y = d.y  # pl.Series
        # F-order numpy fast lane: the same buffer ``self.X`` (polars) views, so
        # the rank screen + fit read it directly instead of round-tripping
        # through ``self.X.to_numpy()``. Read-only (mutating it corrupts
        # ``self.X``); ``_qr`` row-scales into a fresh array, never in place.
        # ``None`` if the design is empty; reset below if columns get dropped.
        self._X_values = d.X_values

        # R's na.pass keeps NA rows in the model frame; lm.fit then rejects a
        # non-finite design ("NA/NaN/Inf in 'x'"). Match that intentionally
        # (rather than tripping over it in the QR below).
        if self.na_action == "pass" and not np.all(np.isfinite(self.X.to_numpy())):
            raise ValueError("NA/NaN/Inf in 'x'")

        # Column → term-assignment map (R's model.matrix ``assign``): 0 for
        # the intercept, i for the columns of the i-th RHS term. Captured
        # keyed by name so it survives the rank-deficiency column drop below,
        # and used by predict(type="terms") to group columns into terms.
        _assign = getattr(d, "param_assign", None)
        self._col_assign = (
            {c: int(a) for c, a in zip(self.X.columns, _assign)}
            if _assign is not None else None
        )

        # R's lm() drops linearly-dependent columns from X (via dqrdc2's
        # pivoted QR) before fitting — without this, df_residuals is off by
        # the alias count whenever the design is rank-deficient. Common
        # case: nested factors like `tree` nested in `CO2` (Wood 2017
        # §2.1.1), where the inner factor's dummies absorb the outer one.
        # We drop the aliased columns up front so every downstream df / SE
        # / F-stat reads from the correct effective parameter count.
        # Surface the alias info via summary()/__repr__ rather than a fit-time
        # warning — R does the same (silent at construction, prints "(N not
        # defined because of singularities)" only when you look at the model).
        _full_cols = list(self.X.columns)  # before the rank-deficiency drop
        self._aliased_cols: list[str] = _drop_aliased_cols(
            self.X, values=self._X_values)
        if self._aliased_cols and not singular_ok:
            # R: lm.fit(singular.ok=FALSE) — refuse rank-deficient designs.
            raise ValueError("singular fit encountered")
        # Keep the full (pre-drop) design when rank-deficient: predict() needs
        # it to flag non-estimable newdata rows (rows that leave the training
        # row space). ``None`` for a full-rank fit — no estimability check then.
        self._X_full_train = None
        if self._aliased_cols:
            self._X_full_train = self.X.to_numpy().astype(float)
            keep = [c for c in self.X.columns if c not in self._aliased_cols]
            self.X = self.X.select(keep)
            # The kept-column subset no longer matches the full F-order buffer;
            # the fit below falls back to ``self.X.to_numpy()`` for this design.
            self._X_values = None

        self.column_names = list(self.X.columns)
        # Full (pre-drop) column set + the kept→full index map. The fit runs
        # on the estimable (kept) columns; the *public* coefficient views are
        # widened back to the full set with NA for the aliased columns, so
        # coef(m)["<aliased>"] resolves and summary prints the NA row (R parity).
        self._full_names = _full_cols
        self._kept_idx = [_full_cols.index(c) for c in self.column_names]
        self.feature_names = (
            self.column_names[1:]
            if "(Intercept)" in self.column_names
            else self.column_names
        )

        # Common full-rank path: reuse the F-order design buffer directly (no
        # ``to_numpy`` round-trip). ``_X_values`` is float64 + F-contiguous by
        # construction; the ``dtype`` guard is a cheap no-op safety net.
        if self._X_values is not None:
            X = self._X_values
            if X.dtype != np.float64:
                X = X.astype(np.float64)
        else:
            X = self.X.to_numpy().astype(float)
        y = self.y.to_numpy().astype(float).flatten()

        self.n, self.p = (
            n,
            p,
        ) = X.shape  # n_samples x n_features (intercept included if available)
        # Prior weights as a length-n vector (all-ones when unweighted);
        # ``self.weights`` was already aligned to X's rows above. The
        # solvers / leverage consume the diagonal ``W``.
        self._w = (
            np.ones(n) if self.weights is None
            else np.asarray(self.weights, dtype=float)
        )
        # Effective sample size: R drops zero-weight rows from the fit, so
        # df / logLik / adjusted-R² count only rows with w > 0 (n_eff == n
        # whenever unweighted or all weights are positive).
        self._n_eff = (
            n if self.weights is None else int(np.count_nonzero(self._w > 0.0))
        )
        # NB: R never forms the n×n weight matrix W; lm.wfit row-scales by √w.
        # We carry only the length-n diagonal ``self._w`` (O(n), not O(n²)).

        # model degree of freedom
        self.df_model = self.p - 1 if "(Intercept)" in self.column_names else self.p

        # residual degrees of freedom (n_eff - p)
        self.df_residuals = (
            self._n_eff - self.df_model - 1
            if "(Intercept)" in self.column_names
            else self._n_eff - self.df_model
        )

        # total parameter count (p fixed + 1 residual variance), for the
        # generic AIC() comparison table and AIC/BIC formulas below.
        self.npar = self.p + 1

        # Sum any `offset(...)` atoms from the formula. R's lm() solves
        # (y - offset) ~ X, then adds the offset back to ŷ — so β̂ has the
        # same df as without offset. expanded.offsets holds the inner ASTs.
        off = np.zeros(n)
        for off_node in d.expanded.offsets:
            off = off + _eval_atom(off_node, d.data).values.flatten().astype(float)
        # R's lm() adds the ``offset=`` argument on top of any in-formula
        # offset(...) term (the two sum). ``off_arg`` is already aligned to the
        # fit rows by the subset + na.omit drops above. NOTE: predict() on *new*
        # data re-evaluates only formula offsets (R replays object$call$offset);
        # the constructor ``offset=`` is reflected in fitted values / residuals
        # but not in predict(newdata=...).
        if self._offset_arg is not None:
            off = off + self._offset_arg
        self._offset = off
        y_solve = y - off

        ##############
        # Estimation #
        ##############

        # Single weighted QR (R's lm.wfit / Cdqrls) — yields β̂, the R factor
        # (for chol2inv), the length-n effects (Qᵀ√w·y), and the Householder
        # qraux, all stored below as R's lm components.
        bhat, self._qr_R, self.effects, self.qraux = self.compute_bhat(
            X, y_solve, self._w
        )
        # R's lm$rank / $assign. hea drops aliased columns at fit time, so the
        # kept design is full rank → rank == p; assign is the model.matrix
        # column→term map over the FULL (pre-drop) column set (R parity).
        self.rank = self.p
        self.assign = (
            np.array([self._col_assign[c] for c in self._full_names])
            if self._col_assign is not None else None
        )

        self._bhat_arr = np.asarray(bhat).reshape(-1)
        from ..R import NamedVector

        self.bhat = _row_frame(self._bhat_arr, self.column_names)
        # R-canonical alias: ``m$coef`` is a named numeric vector, so we
        # expose the same on the Python side. ``.bhat`` keeps its 1-row
        # DataFrame shape for internal callers that want a frame.
        self.coef = NamedVector(self.column_names, self._bhat_arr)
        self.coefficients = self.coef

        # compute predicted (fitted values ŷ = Xβ̂)
        self.yhat = self.compute_yhat()
        yhat = self.yhat["fit"].to_numpy().astype(float)

        # compute residuals ϵ̂
        residuals = y - yhat
        self._residuals_arr = residuals
        self.residuals = pl.DataFrame({"residuals": residuals})

        # compute residual sum of squares (RSS). Weighted by the prior
        # weights — R's summary.lm / deviance.lm use Σ wᵢ·rᵢ² for the
        # residual variance, and zero-weight rows drop out naturally.
        # ``self._w`` is all-ones when unweighted, so this is the ordinary
        # Σ rᵢ² in that case.
        self.rss = float(np.sum(self._w * residuals * residuals))

        # compute standard deviation of model coefficients
        # aka Residual SE: σ^2 = RSS / df_residuals. A saturated fit
        # (df_residuals == 0, e.g. n == rank) has no residual variance —
        # R returns NaN (and NaN SEs / t / p) rather than erroring.
        self.sigma_squared = (
            self.rss / self.df_residuals
            if self.df_residuals > 0 else float("nan")
        )
        self.sigma = np.sqrt(self.sigma_squared)

        # compute standard error for β̂
        self.XtXinv = self.compute_XtXinv()

        se_bhat, V_bhat = self.compute_se_bhat()
        self.V_bhat = V_bhat
        self._se_bhat_arr = np.asarray(se_bhat).reshape(-1)
        self.se_bhat = _row_frame(self._se_bhat_arr, self.column_names)

        # hat-matrix diagonal h_ii (leverages) and internally studentized
        # residuals — cached once so plot_qq / plot_scale_location /
        # plot_leverage don't each recompute them.
        HX = X @ self.XtXinv
        h = (HX * X).sum(axis=1)
        w = self._w
        self.leverage = h * w
        denom = self.sigma * np.sqrt(np.clip(1.0 - self.leverage, 1e-12, None))
        self.std_residuals = residuals * np.sqrt(w) / denom

        # compute confidence interval for β̂
        self.ci_bhat = self.compute_ci_bhat()

        # compute t values of model coefficients
        self.t_values = self.compute_t_values()

        # p values
        self.p_values = self.compute_p_values()

        # Display-width coefficient views default to the estimable (fit)
        # columns; widen to the full column set with NA for aliased columns
        # when the design was rank-deficient (R keeps every column in coef()).
        self._bhat_disp = self._bhat_arr
        self._se_disp = self._se_bhat_arr
        self._p_disp = np.asarray(self.p_values.row(0), dtype=float)
        self._names_disp = self.column_names
        if self._aliased_cols:
            self._widen_public_coefs()

        # compute r2 and r2adjusted, aka coefficient of determination
        # aka percentage of variance explained. Noted that the formulae
        # are different for cases with and without intercept
        (
            self.tss,
            self.r_squared,
            self.r_squared_adjusted,
        ) = self.compute_goodness_of_fit()

        # compute F-statistics with scipy.stats.f.sf
        # H0: all coefficients == 0
        # H1: at least one coefficient != 0
        self.fstats, self.f_p_value = self.compute_fstats()

        # compute log-likelihood
        self.loglike = self.compute_loglikelihood()

        # compute AIC (Akaike Information criterion): -2logL + 2p, p is the total number of parameters
        self.AIC = self.compute_AIC()

        # compute BIC (Bayes Information criterian): -2logL + p * log(n)
        self.BIC = self.compute_BIC()

    # ------------------------------------------------------------------
    # Multivariate response (R's "mlm": cbind(y1, y2, ...) ~ rhs)
    # ------------------------------------------------------------------
    def _init_mlm(self, formula, data, mv_specs, *, weights, method, subset,
                  na_action, contrasts, singular_ok, offset):
        """Fit a multivariate linear model: one `lm` per response column, all
        sharing a single jointly-na-omitted model frame (so every column has the
        identical X/QR — R's mlm). Combined accessors (coef p×m, fitted/residuals
        n×m, sigma per-column) delegate to the per-column sub-models."""
        from hea.formula import parse as _parse, deparse as _deparse, _eval_lhs_expr
        self._is_mlm = True
        self.formula = formula
        self.method = method
        self.contrasts = contrasts

        _na_norm = {"omit": "omit", "na.omit": "omit", "exclude": "exclude",
                    "na.exclude": "exclude", "fail": "fail", "na.fail": "fail"}
        self.na_action = _na_norm.get(str(na_action))
        if self.na_action is None:
            raise ValueError("na_action must be one of 'omit' / 'exclude' / "
                             f"'fail'; got {na_action!r}")

        rhs_src = _deparse(_parse(formula).rhs)
        resp_names = [lbl for lbl, _ in mv_specs]
        if len(set(resp_names)) != len(resp_names):
            raise ValueError(f"duplicate response columns in {resp_names}")
        self._response_names = resp_names

        # Replicate lm's subset → na-omit → weights/offset alignment, but
        # na-omit JOINTLY over every response (R's shared model frame).
        w_arr = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        off_arr = None if offset is None else np.asarray(offset, dtype=float).reshape(-1)
        if w_arr is not None:
            if w_arr.shape[0] != data.height:
                raise ValueError("Length of weights should be the same as the "
                                 "number of rows in the dataframe")
            if not np.all(np.isfinite(w_arr)) or np.any(w_arr < 0.0):
                raise ValueError("missing or negative weights not allowed")
        if off_arr is not None and off_arr.shape[0] != data.height:
            raise ValueError("Length of offset should be the same as the number "
                             "of rows in the dataframe")
        if subset is not None:
            keep = _subset_keep(data.height, _resolve_subset(subset, data))
            data = data[keep]
            if w_arr is not None:
                w_arr = w_arr[keep]
            if off_arr is not None:
                off_arr = off_arr[keep]
        na_keep = _model_frame_keep_mask(formula, data)
        if self.na_action == "fail" and not na_keep.all():
            raise ValueError("missing values in object")
        # Model-frame length (post-subset, pre-na) + the joint keep-mask, so
        # na.action="exclude" can pad the combined n×m accessors back to full
        # length with NA (R's naresid on an mlm). ``omit`` ignores both.
        self._n_full = data.height
        self._na_mask = na_keep
        if not na_keep.all():
            data = data.filter(pl.Series(na_keep))
            if w_arr is not None:
                w_arr = w_arr[na_keep]
            if off_arr is not None:
                off_arr = off_arr[na_keep]

        # Materialize any expression responses (cbind(log(a), b)) into named
        # columns so each per-column sub-formula can reference them.
        cols = set(data.columns)
        add = {}
        for lbl, node in mv_specs:
            if lbl not in cols:
                add[lbl] = data.select(_eval_lhs_expr(node, cols).alias(lbl))[lbl].to_numpy()
        if add:
            data = data.with_columns([pl.Series(k, v) for k, v in add.items()])

        # Per-column fits on the shared clean frame (na-omit now a no-op).
        self._mlm_models = {
            lbl: lm(f"`{lbl}` ~ {rhs_src}", data,
                    weights=w_arr, method=method, contrasts=contrasts,
                    singular_ok=singular_ok, offset=off_arr, na_action="omit")
            for lbl in resp_names
        }
        self.response_models = self._mlm_models
        m0 = self._mlm_models[resp_names[0]]

        # Shared design pieces (identical across columns).
        self.data = data
        self.column_names = m0.column_names
        self.X = m0.X
        self._aliased_cols = m0._aliased_cols
        self.df_residuals = m0.df_residuals
        self.df_residual = m0.df_residuals

        # Combined accessors.
        self._coef_matrix = np.column_stack(
            [self._mlm_models[r]._bhat_arr for r in resp_names])
        self.coef = self._mlm_matrix_frame(
            {r: self._mlm_models[r]._bhat_arr for r in resp_names}, index="term",
            index_vals=self.column_names)
        self.coefficients = self.coef
        self.bhat = self.coef
        self.fitted = self._mlm_matrix_frame(
            {r: self._mlm_pad(self._mlm_models[r].yhat["fit"].to_numpy())
             for r in resp_names})
        self.yhat = self.fitted
        self.residuals = self._mlm_matrix_frame(
            {r: self._mlm_pad(self._mlm_models[r].residuals.to_series(0).to_numpy())
             for r in resp_names})
        self.sigma = np.array([self._mlm_models[r].sigma for r in resp_names])

    def _mlm_pad(self, arr: np.ndarray) -> np.ndarray:
        """na.action="exclude": pad a per-response fit-row vector back to the
        model-frame length with NA (R's naresid for an mlm). No-op otherwise."""
        arr = np.asarray(arr, dtype=float)
        if self.na_action != "exclude" or self._na_mask.all():
            return arr
        out = np.full(self._n_full, np.nan, dtype=float)
        out[self._na_mask] = arr
        return out

    def _mlm_matrix_frame(self, col_map, index=None, index_vals=None):
        """Bundle per-response arrays into a hea.DataFrame (one column per
        response). With ``index=``, prepend a label column (used by coef)."""
        from ..tidy import DataFrame as _DF
        cols = {}
        if index is not None:
            cols[index] = list(index_vals)
        for name, arr in col_map.items():
            cols[name] = np.asarray(arr, dtype=float)
        return _DF(pl.DataFrame(cols))

    def _mlm_predict(self, newdata, interval, se_fit, type_):
        from ..tidy import DataFrame as _DF
        if interval is not None or se_fit or type_ != "response":
            raise NotImplementedError(
                "predict() on a multivariate (mlm) fit returns point predictions "
                "only (type='response', no interval/se_fit) — matching R's "
                "predict.mlm. For per-column inference use "
                ".response_models['<name>'].predict(...).")
        return _DF(pl.DataFrame({
            r: self._mlm_models[r].predict(newdata)["fit"].to_numpy()
            for r in self._response_names
        }))

    def __repr__(self):
        if getattr(self, "_is_mlm", False):
            out = f"Formula: {self.formula}\n\nCoefficients:\n"
            return out + format_df(self.coef)

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += self._coef_header() + "\n"
        docstring += format_df(self.bhat)

        return docstring

    def _coef_header(self) -> str:
        """R-style 'Coefficients:' header — adds '(N not defined because of
        singularities)' when the design was rank-deficient and aliased columns
        had to be dropped from ``self.X``."""
        if self._aliased_cols:
            return f"Coefficients: ({len(self._aliased_cols)} not defined because of singularities)"
        return "Coefficients:"

    def __str__(self):

        return self.__repr__()

    def compute_XtXinv(self):
        """(XᵀWX)⁻¹ via ``chol2inv`` of the QR ``R`` factor — R's ``summary.lm``
        (``chol2inv(Qr$qr[p1,p1])``, lm.R:326). More accurate than inverting an
        explicitly-formed XᵀWX (which squares the condition number) and matches
        R's path; this is R's ``cov.unscaled`` (``V_bhat = σ²·XtXinv``)."""
        return _chol2inv(self._qr_R)

    def compute_bhat(self, X, y, w):
        """β̂ by weighted-least-squares QR — R's ``lm.wfit`` (the only method
        R supports). ``w`` is the length-n prior-weight vector. Returns
        ``(β̂, R, effects, qraux)`` (see :func:`_qr`)."""
        return _qr(X, y, w)

    def compute_se_bhat(self):
        V_bhat = self.sigma_squared * self.XtXinv
        se_bhat = np.sqrt(np.diag(V_bhat))[:, None]
        return se_bhat, V_bhat

    def compute_ci_bhat(self, alpha=0.05):

        se_bhat = self._se_bhat_arr[:, None]
        bhat = self._bhat_arr[:, None]
        ci = (
            _dist.qt(1 - alpha / 2, self.df_residuals) * se_bhat * np.array([-1, 1]) + bhat
        )
        return pl.DataFrame(
            {
                "coef": self.column_names,
                f"CI[{alpha/2*100}%]": ci[:, 0],
                f"CI[{100-alpha/2*100}]%": ci[:, 1],
            }
        )

    def compute_ci_bhat_bootstrap(self, num_bootstrap=4000, alpha=0.05,
                                  *, seed=None):
        """Residual bootstrap CI for the coefficients.

        hea-original (``stats::lm`` has no bootstrap-CI method), so this is
        **not an R-parity target** — it draws from numpy, not ``hea.R.rng``.
        ``seed=None`` (default) uses numpy's global RNG, so results vary across
        runs; pass an ``int`` or ``numpy.random.Generator`` for a reproducible
        bootstrap.
        """

        X = self.X.to_numpy().astype(float)
        sw = np.sqrt(self._w)            # √w row-scaling (was cholesky(diag(w)))
        bhat = self._bhat_arr[:, None]
        residuals = self._residuals_arr
        n = len(residuals)

        # X and w are constant across draws, so the QR-WLS factorisation
        # (√w·X, qr) is constant too — the per-iter ``compute_bhat`` call
        # would otherwise redo it on every draw. Hoist it and keep the
        # per-iter ``√w·y_star``, ``Q.T @ ·``, ``solve_triangular`` exactly as
        # ``_qr`` does, so each bhat_star matches the loop's arithmetic
        # bit-for-bit. (cholesky(diag(w)) == diag(√w), so this is identical to
        # the former L.T@· form for positive weights, and robust to zero w.)
        Xhat = sw[:, None] * X
        Q, R = qr(Xhat, mode="economic")
        X_bhat_flat = (X @ bhat).flatten()

        # ``choice(..., size=(B, n))`` produces the same draws and advances the
        # RNG by the same amount as B sequential ``size=n`` calls (verified on
        # the legacy MT19937), so this batched sample is RNG-byte-equivalent to
        # the unrolled loop. ``seed=None`` keeps the legacy global-``np.random``
        # path; an int / Generator gives a reproducible local stream.
        if seed is None:
            gen = np.random
        elif isinstance(seed, np.random.Generator):
            gen = seed
        else:
            gen = np.random.default_rng(seed)
        residuals_star = gen.choice(
            residuals, size=(num_bootstrap, n), replace=True
        )

        bhat_stars = np.zeros([num_bootstrap, self.p])
        for i in range(num_bootstrap):
            y_flat = X_bhat_flat + residuals_star[i]
            f = Q.T @ (sw * y_flat)
            bhat_stars[i] = solve_triangular(R, f)

        quantiles = np.quantile(
            bhat_stars, q=[alpha / 2, 1 - alpha / 2], axis=0
        ).T
        ci_bhat_bootstrap = pl.DataFrame(
            {
                "coef": self.column_names,
                f"CI[{alpha/2*100}%]": quantiles[:, 0],
                f"CI[{100-alpha/2*100}]%": quantiles[:, 1],
            }
        )
        self.ci_bhat_bootstrap = ci_bhat_bootstrap
        self.bhat_bootstrap = pl.DataFrame(
            {c: bhat_stars[:, i] for i, c in enumerate(self.column_names)}
        )

        return ci_bhat_bootstrap

    def compute_yhat(self, Xnew=None, interval=None, alpha=0.05, se_fit=False,
                     res_var=None, df_q=None, pred_var=None):
        # ``res_var`` / ``df_q`` override the residual variance σ² and the
        # interval-quantile df (R's ``scale²`` / ``df``); ``None`` → the fit's
        # own σ² / df.residual. ``pred_var`` overrides the prediction-interval
        # variance (R's ``pred.var``; default = ``res_var``). (``sigma_squared``
        # isn't set yet during the fit-time ``compute_yhat()`` call, so it's
        # only read in the ``se_fit`` / interval branches below.)
        if Xnew is None:
            X = self.X.to_numpy().astype(float)
            off = self._offset
        else:
            X = materialize(self._expanded, Xnew, basis_state=self._basis_state).select(self.column_names).to_numpy().astype(float)
            # Re-evaluate formula offsets against newdata, mirroring R's
            # predict.lm. Offsets are zero unless the formula uses offset(...).
            off = np.zeros(X.shape[0])
            for off_node in self._expanded.offsets:
                off = off + _eval_atom(off_node, Xnew).values.flatten().astype(float)
        # compute predicted or fitted values ŷ = Xβ̂ + offset
        bhat = self._bhat_arr[:, None]
        yhat_vals = (X @ bhat).flatten() + off
        yhat = pl.DataFrame({"fit": yhat_vals})

        if se_fit:
            # R's predict.lm `se.fit`: SE of the fitted mean,
            # √(res_var·xᵀ(XᵀWX)⁻¹x). Same quantity the confidence interval
            # uses (and reported even when interval="prediction").
            rv = self.sigma_squared if res_var is None else res_var
            var_mean = np.einsum(
                "ij,jk,ik->i", X, self.XtXinv, X
            ) * rv
            yhat = yhat.with_columns(
                pl.Series("se.fit", np.sqrt(np.maximum(var_mean, 0.0)))
            )

        match interval:
            case None:
                return yhat
            case True:
                # Both CI and PI in one frame — column names are prefixed
                # so the two interval kinds don't collide.
                ci_yhat = self.compute_ci_yhat(
                    yhat, Xnew, alpha, res_var, df_q
                ).rename({"lwr": "ci_lwr", "upr": "ci_upr"})
                pi_yhat = self.compute_pi_yhat(
                    yhat, Xnew, alpha, res_var, df_q, pred_var
                ).rename({"lwr": "pi_lwr", "upr": "pi_upr"})
                return pl.concat([yhat, ci_yhat, pi_yhat], how="horizontal")
            case "prediction":
                pi_yhat = self.compute_pi_yhat(
                    yhat, Xnew, alpha, res_var, df_q, pred_var
                )
                return pl.concat([yhat, pi_yhat], how="horizontal")
            case "confidence":
                ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha, res_var, df_q)
                return pl.concat([yhat, ci_yhat], how="horizontal")
            case _:
                raise ValueError(
                    "Please enter a valid value: [None, True, 'prediction', 'confidence']"
                )

    def compute_ci_yhat(self, yhat, Xnew=None, alpha=0.05, res_var=None,
                        df_q=None):
        rv = self.sigma_squared if res_var is None else res_var
        dq = self.df_residuals if df_q is None else df_q
        if Xnew is None:
            X = self.X.to_numpy().astype(float)
        else:
            X = materialize(self._expanded, Xnew, basis_state=self._basis_state).select(self.column_names).to_numpy().astype(float)

        # Var(ŷ) = res_var · x'(X'X)⁻¹x  ⇒  se = √diag(res_var · X(X'X)⁻¹Xᵀ).
        var_mean = np.einsum("ij,jk,ik->i", X, self.XtXinv, X) * rv
        se_yhat_mean = np.sqrt(np.maximum(var_mean, 0.0))
        yhat_vals = yhat["fit"].to_numpy().astype(float)[:, None]
        ci = (
            _dist.qt(1 - alpha / 2, dq)
            * se_yhat_mean[:, None]
            * np.array([-1, 1])
            + yhat_vals
        )
        return pl.DataFrame(
            {
                "lwr": ci[:, 0],
                "upr": ci[:, 1],
            }
        )

    def compute_pi_yhat(self, yhat, Xnew=None, alpha=0.05, res_var=None,
                        df_q=None, pred_var=None):
        rv = self.sigma_squared if res_var is None else res_var
        dq = self.df_residuals if df_q is None else df_q
        if Xnew is None:
            X = self.X.to_numpy().astype(float)
        else:
            X = materialize(self._expanded, Xnew, basis_state=self._basis_state).select(self.column_names).to_numpy().astype(float)

        # Var(y_new − ŷ) = pred_var + res_var·x'(X'X)⁻¹x  (R: ip + pred.var,
        # pred.var defaulting to res_var).
        var_mean = np.einsum("ij,jk,ik->i", X, self.XtXinv, X) * rv
        pv = rv if pred_var is None else pred_var
        se_yhat = np.sqrt(pv + np.maximum(var_mean, 0.0))
        yhat_vals = yhat["fit"].to_numpy().astype(float)[:, None]
        pi = (
            _dist.qt(1 - alpha / 2, dq)
            * se_yhat[:, None]
            * np.array([-1, 1])
            + yhat_vals
        )
        return pl.DataFrame(
            {
                "lwr": pi[:, 0],
                "upr": pi[:, 1],
            }
        )

    def compute_t_values(self):

        t_values = self._bhat_arr / self._se_bhat_arr

        return _row_frame(t_values, self.column_names)

    def compute_p_values(self):
        # compute p values of model coefficients with scipy.stats.t.sf
        # H0: βi==0
        # H1: βi!=0
        t_arr = self._bhat_arr / self._se_bhat_arr
        p_values = 2 * _dist.pt(np.abs(t_arr), self.df_residuals, lower_tail=False)
        return _row_frame(p_values, self.column_names)

    def _expand_to_full(self, kept_vals) -> np.ndarray:
        """Scatter estimable-column values into the full column set, with NA
        (NaN) in the aliased slots — the inverse of the rank-deficiency drop."""
        out = np.full(len(self._full_names), np.nan, dtype=float)
        out[self._kept_idx] = np.asarray(kept_vals, dtype=float)
        return out

    def _widen_public_coefs(self) -> None:
        """Rank-deficient parity: present every original column in the public
        coefficient views (``coef`` / ``bhat`` / ``se_bhat`` / ``t_values`` /
        ``p_values`` / ``ci_bhat``), with NA for the aliased columns. The fit
        itself stays on the estimable columns — ``_bhat_arr`` / ``XtXinv`` /
        ``df_residuals`` are unchanged — so only the *display* widens, exactly
        like R (``coef(m)`` carries NA rows; ``summary`` prints them)."""
        from ..R import NamedVector

        names = self._full_names
        bhat_f = self._expand_to_full(self._bhat_arr)
        se_f = self._expand_to_full(self._se_bhat_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_f = bhat_f / se_f
        p_f = 2.0 * _dist.pt(np.abs(t_f), self.df_residuals, lower_tail=False)  # NaN stays NaN
        tcrit = _dist.qt(1 - 0.05 / 2, self.df_residuals)
        cilo = bhat_f - tcrit * se_f
        cihi = bhat_f + tcrit * se_f

        self._bhat_disp = bhat_f
        self._se_disp = se_f
        self._p_disp = p_f
        self._names_disp = names

        self.bhat = _row_frame(bhat_f, names)
        self.coef = NamedVector(names, bhat_f)
        self.coefficients = self.coef
        self.se_bhat = _row_frame(se_f, names)
        self.t_values = _row_frame(t_f, names)
        self.p_values = _row_frame(p_f, names)
        lo_col, hi_col = self.ci_bhat.columns[1], self.ci_bhat.columns[2]
        self.ci_bhat = pl.DataFrame({"coef": names, lo_col: cilo, hi_col: cihi})

    def compute_goodness_of_fit(self):

        y = self.y.to_numpy().astype(float)
        # Weighted TSS — R's summary.lm uses the weighted mean ȳ_w =
        # Σwy/Σw and tss = Σ w (y − ȳ_w)², so that tss = mss + rss and
        # R² = 1 − rss/tss = mss/(mss+rss) matches R exactly. With the
        # all-ones weight vector this is the ordinary (unweighted) TSS.
        w = self._w

        # Adjusted R² divides by df_residuals — NaN for a saturated fit
        # (df_residuals == 0), matching R.
        df_ok = self.df_residuals > 0
        if "(Intercept)" in self.column_names:
            ybar = float(np.sum(w * y) / np.sum(w))
            tss = float(np.sum(w * (y - ybar) ** 2))
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum(w (ŷ - yi)**2) / sum(w (y - ȳ)**2)
            r_squared = float(1 - self.rss / tss)
            # Eq: r2adj = 1 - (1 - r2) * (n_eff - 1) / df_residuals
            r_squared_adjusted = (
                1 - (1 - r_squared) * (self._n_eff - 1) / self.df_residuals
                if df_ok else float("nan")
            )
        else:
            tss = float(np.sum(w * y**2))
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum(w (ŷ - yi)**2) / sum(w y**2)
            r_squared = float(1 - self.rss / tss)
            # Eq: r2adj = 1 - (1 - r2) * n_eff / df_residuals
            r_squared_adjusted = (
                1 - (1 - r_squared) * self._n_eff / self.df_residuals
                if df_ok else float("nan")
            )

        return tss, r_squared, r_squared_adjusted

    def compute_fstats(self):
        # No F-statistic for an intercept-only model (df_model == 0) or a
        # saturated fit (df_residuals == 0, where the denominator vanishes).
        if self.df_model != 0 and self.df_residuals > 0:
            fstats = float(
                ((self.tss - self.rss) / self.df_model) / (self.rss / self.df_residuals)
            )
            f_p_value = float(_dist.pf(fstats, self.df_model, self.df_residuals, lower_tail=False))
        else:
            fstats, f_p_value = None, None
        return fstats, f_p_value

    def compute_loglikelihood(self):
        # R's logLik.lm (Gaussian):
        #   0.5·(Σ log wᵢ − N·(log 2π + 1 − log N + log Σ wᵢ rᵢ²))
        # with N = #{wᵢ ≠ 0} (zero-weight rows excluded) and Σ wᵢ rᵢ² the
        # weighted RSS. Reduces to −0.5·n·(log(rss/n)+log 2π+1) when w ≡ 1.
        n = self._n_eff
        if self.rss <= 0.0:
            # Perfect (saturated) fit: the Gaussian log-likelihood diverges;
            # R returns +Inf. Guard the log(0) so construction stays warning-free.
            return float("inf")
        if self.weights is None:
            sum_log_w = 0.0
        else:
            nz = self._w > 0.0
            sum_log_w = float(np.sum(np.log(self._w[nz])))
        return float(
            0.5 * (sum_log_w
                   - n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(self.rss)))
        )

    def compute_AIC(self):
        # npar = p + 1 (residual variance) — matches R, see
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + 2 * self.npar

    def compute_BIC(self):
        # R's BIC uses log(nobs) with nobs = N = #{wᵢ ≠ 0}.
        return -2 * self.loglike + np.log(self._n_eff) * self.npar

    def predict(self, newdata=None, interval=None, alpha=0.05, *,
                level=None, se_fit=False, type="response", terms=None,
                rankdeficient="warnif", scale=None, df=None, pred_var=None):
        """R: ``predict.lm`` — fitted/predicted values on ``newdata``.

        ``interval`` ∈ ``{None, "confidence", "prediction", True}`` adds
        ``lwr``/``upr`` columns (``True`` returns both, ``ci_*``/``pi_*``).
        ``level=`` is R's name for the interval level — an alias for
        ``1 − alpha`` (``alpha`` still works; ``level`` wins if both given).

        ``se_fit=True`` adds an ``se.fit`` column: the SE of the fitted
        mean, ``√(res_var·xᵀ(XᵀWX)⁻¹x)`` (reported even with
        ``interval="prediction"``, matching R). The returned frame also carries
        R's other ``se.fit`` list elements as attributes: ``.df`` (the quantile
        df) and ``.residual_scale`` (``√res_var``).

        ``scale=`` / ``df=`` override the residual scale and quantile df used
        for ``se.fit`` and the intervals — R's ``scale`` / ``df`` (``res.var =
        scale²``; ``df`` defaults to ∞ ⇒ normal quantile). When ``scale`` is
        None they default to the fit's own ``σ`` and ``df.residual``.
        ``pred_var=`` overrides the prediction-interval variance (R's
        ``pred.var``; default ``res_var``) — e.g. a known future-obs weight.

        ``type="terms"`` returns one *centered* column per RHS term — R's
        ``predict(type="terms")`` — with the overall constant attached as
        the ``.constant`` attribute (so ``fit = constant + rowSums``) and
        ``se.{term}`` columns when ``se_fit=True``. ``terms=`` (a label or
        list) selects a subset. See :meth:`_predict_terms`.

        With ``na_action="exclude"`` and no ``newdata``, the result is padded
        back to the model-frame length with NA at the omitted rows (R's
        ``napredict``), so it lines up with the original frame.

        For a rank-deficient fit, ``rankdeficient=`` controls handling of
        *non-estimable* ``newdata`` rows (those leaving the training row span,
        where the dropped aliased columns would matter) — R's ``rankdeficient=``:
        ``"warnif"`` (default) warn + attach the flagged row indices as
        ``.non_estim``; ``"non-estim"`` attach silently; ``"simple"`` warn only;
        ``"NA"`` set those rows to NA; ``"NAwarn"`` NA + warn.

        For a multivariate (mlm) fit the result is an ``n × m`` frame of point
        predictions (one column per response); ``interval``/``se_fit`` are not
        supported there (R's ``predict.mlm`` returns points only).
        """
        if getattr(self, "_is_mlm", False):
            return self._mlm_predict(newdata, interval, se_fit, type)
        if level is not None:
            alpha = 1.0 - level
        # R: scale= overrides res.var (=scale²) and df= the quantile df
        # (default ∞ ⇒ normal). With scale=None, R uses σ² and df.residual
        # (the df= argument is ignored), so leave them None to pick the
        # fit's own values downstream.
        if scale is not None:
            res_var = float(scale) ** 2
            df_q = float("inf") if df is None else float(df)
        else:
            res_var = None
            df_q = None
        if type == "terms":
            out = self._predict_terms(Xnew=newdata, se_fit=se_fit, terms=terms)
        elif type == "response":
            out = self.compute_yhat(
                Xnew=newdata, interval=interval, alpha=alpha, se_fit=se_fit,
                res_var=res_var, df_q=df_q, pred_var=pred_var,
            )
        else:
            raise ValueError(
                f"predict(): type must be 'response' or 'terms', got {type!r}"
            )
        # Rank-deficient fit + new data: flag / NA non-estimable rows (R's
        # rankdeficient=). type="terms" is exempt, matching R (its FIXME).
        if type == "response" and newdata is not None and self._aliased_cols:
            out = self._apply_rankdeficient(out, newdata, rankdeficient)
        if newdata is None and self.na_action == "exclude" and not self._na_mask.all():
            out = self._na_pad_frame(out)
        # R's se.fit return is list(fit, se.fit, df, residual.scale); hea keeps
        # the frame and carries df / residual.scale as attributes (an additive
        # improvement on R's list shape).
        if se_fit and type == "response":
            from ..tidy import DataFrame as _DF  # local: avoid import cycle
            if not isinstance(out, _DF):
                out = _DF(out)
            out.df = self.df_residuals if df_q is None else df_q
            out.residual_scale = float(
                np.sqrt(self.sigma_squared if res_var is None else res_var)
            )
        return out

    def _nonestimable_mask(self, newdata, tol=1e-6):
        """Boolean mask over ``newdata`` rows that are *non-estimable* from a
        rank-deficient fit: the full (pre-drop) design row leaves the training
        row space, so the dropped aliased columns would change the prediction.

        R's ``predict.lm`` uses the QR null-space basis and flags rows where
        ``tol·‖x‖ ≤ ‖N′x‖``. Equivalently we project each newdata row onto the
        training row space (``P = X⁺X``) and flag rows whose out-of-span part
        ``‖x·(I − P)‖`` exceeds ``tol·‖x‖``. ``None`` for a full-rank fit.
        """
        if not self._aliased_cols or self._X_full_train is None:
            return None
        Xtr = self._X_full_train  # n × p_full (pre-drop)
        P = np.linalg.pinv(Xtr) @ Xtr  # projector onto the training row space
        Xnew = (
            materialize(self._expanded, newdata, basis_state=self._basis_state)
            .select(self._full_names).to_numpy().astype(float)
        )
        resid_norm = np.linalg.norm(Xnew - Xnew @ P, axis=1)
        x_norm = np.linalg.norm(Xnew, axis=1)
        x_norm = np.where(x_norm == 0.0, 1.0, x_norm)
        return resid_norm > tol * x_norm

    def _apply_rankdeficient(self, out, newdata, mode):
        """Apply R's ``rankdeficient=`` policy to a rank-deficient predict
        frame — see :meth:`predict`. No-op when every newdata row is estimable
        (so estimable prediction from a rank-deficient fit stays warning-free,
        matching R's ``"warnif"``)."""
        valid = ("warnif", "simple", "non-estim", "NA", "NAwarn")
        if mode not in valid:
            raise ValueError(
                f"predict(): rankdeficient must be one of {valid}, got {mode!r}"
            )
        mask = self._nonestimable_mask(newdata)
        if mask is None or not mask.any():
            return out
        import warnings

        msg = "prediction from rank-deficient fit"
        if mode == "simple":
            warnings.warn(f'{msg}; consider predict(rankdeficient="NA")')
            return out
        if mode in ("NA", "NAwarn"):
            cols = {}
            for c in out.columns:
                arr = out[c].to_numpy().astype(float).copy()
                arr[mask] = np.nan
                cols[c] = arr
            if mode == "NAwarn":
                warnings.warn(f"{msg}: NAs produced for non-estimable cases")
            return pl.DataFrame(cols)
        # "warnif" / "non-estim": attach the flagged 0-based row indices as
        # ``.non_estim`` (R's attr(*, "non-estim")); "warnif" also warns.
        if mode == "warnif":
            warnings.warn(f'{msg}; .non_estim has doubtful cases')
        from ..tidy import DataFrame as _DF  # local: avoid import cycle

        out = _DF(out)
        out.non_estim = np.flatnonzero(mask)
        return out

    def _na_pad(self, values) -> np.ndarray:
        """R's ``naresid`` / ``napredict`` for ``na.action="exclude"``:
        scatter complete-case ``values`` into the model-frame length with NA
        at the omitted rows. Identity for ``omit`` / ``fail`` (no rows to
        restore). Used by the ``resid()`` / ``fitted()`` accessors."""
        arr = np.asarray(values, dtype=float)
        if self.na_action != "exclude" or self._na_mask.all():
            return arr
        out = np.full(self._n_full, np.nan, dtype=float)
        out[self._na_mask] = arr
        return out

    def _na_pad_frame(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Row-pad a (complete-case) predict frame to the model-frame length
        with NA at the omitted rows — the ``na.action="exclude"`` form of
        :meth:`_na_pad` for the multi-column predict output."""
        idx = np.flatnonzero(self._na_mask)
        cols = {}
        for c in frame.columns:
            out = np.full(self._n_full, np.nan, dtype=float)
            out[idx] = frame[c].to_numpy().astype(float)
            cols[c] = out
        padded = pl.DataFrame(cols)
        constant = getattr(frame, "constant", None)
        if constant is not None:
            from ..tidy import DataFrame as _DF
            padded = _DF(padded)
            padded.constant = constant
        return padded

    def _predict_terms(self, Xnew=None, se_fit=False, terms=None):
        """R: ``predict.lm(type="terms")`` — per-term contributions.

        Each column is the centered contribution of one RHS term,
        ``(X[:, cols] − colMeans(Xtrain)[cols]) · β̂[cols]`` — R centers by
        the *training* design's (unweighted) column means. The overall
        constant ``Σ colMeans·β̂`` (= mean fitted value) is attached as the
        ``.constant`` attribute, so ``fit = constant + rowSums(terms)``.
        ``se_fit=True`` appends ``se.{label}`` columns,
        ``√(σ²·diag(X_c[:,cols]·(XᵀWX)⁻¹[cols,cols]·X_cᵀ))``; ``terms=``
        (label or list) selects a subset of term labels.
        """
        if self._col_assign is None:
            raise RuntimeError(
                "predict(type='terms') needs the column→term map "
                "(no param_assign on the design)"
            )
        if Xnew is None:
            X = self.X.to_numpy().astype(float)
        else:
            X = materialize(self._expanded, Xnew, basis_state=self._basis_state).select(
                self.column_names
            ).to_numpy().astype(float)

        beta = self._bhat_arr
        avx = self.X.to_numpy().astype(float).mean(axis=0)  # R: colMeans(mm)
        Xc = X - avx
        assign = np.array([self._col_assign[c] for c in self.column_names])
        constant = float(np.sum(avx * beta))

        XtXinv = np.asarray(self.XtXinv)
        res_var = self.sigma_squared
        present = [a for a in sorted(set(assign.tolist())) if a != 0]

        data: dict[str, np.ndarray] = {}
        se_data: dict[str, np.ndarray] = {}
        for a in present:
            label = self._expanded.terms[a - 1].label
            idx = np.where(assign == a)[0]
            data[label] = Xc[:, idx] @ beta[idx]
            if se_fit:
                sub = XtXinv[np.ix_(idx, idx)]
                var_t = np.einsum(
                    "ij,jk,ik->i", Xc[:, idx], sub, Xc[:, idx]
                ) * res_var
                se_data[label] = np.sqrt(np.maximum(var_t, 0.0))

        if terms is not None:
            sel = [terms] if isinstance(terms, str) else list(terms)
            # R warns and ignores unknown labels; we just intersect.
            data = {k: v for k, v in data.items() if k in sel}
            se_data = {k: v for k, v in se_data.items() if k in sel}

        if se_fit:
            for k, v in se_data.items():
                data[f"se.{k}"] = v

        from ..tidy import DataFrame as _DF  # local: avoid import cycle
        out = _DF(data)
        # R attaches the term constant as attr(., "constant"); hea's
        # DataFrame subclass carries it as a plain attribute.
        out.constant = constant
        return out

    def _residuals_lines(self, digits: int = 4) -> list[str]:
        # R's print.summary.lm shows *weighted* residuals (√wᵢ·rᵢ) under a
        # "Weighted Residuals:" header when the weights vary; otherwise the
        # raw residuals under "Residuals:". The body has three forms keyed on
        # the residual df (rdf): the 5-number summary for rdf > 5, each
        # residual for 0 < rdf ≤ 5, and a perfect-fit note for rdf == 0.
        r = self._residuals_arr
        header = "Residuals:"
        if self.weights is not None and float(np.ptp(self._w)) > 0.0:
            r = np.sqrt(self._w) * r
            header = "Weighted Residuals:"

        rdf = self.df_residuals
        if rdf == 0:
            # R: "ALL <rank> residuals are 0: no residual degrees of freedom!"
            return [
                header,
                f"ALL {self.p} residuals are 0: no residual degrees of freedom!",
            ]
        if rdf <= 5:
            # R prints each residual, index-labelled (0-based here; R 1-based).
            labels = [str(i) for i in range(len(r))]
            vals = format_signif(r, digits=digits)
        else:
            # R: 5-number summary, zapsmall'd to (digits+1) so a numerically
            # negligible quantile (e.g. a ~1e-16 median) prints as 0, not noise.
            qs = _zapsmall(np.quantile(r, [0.0, 0.25, 0.5, 0.75, 1.0]), digits + 1)
            labels = ["Min", "1Q", "Median", "3Q", "Max"]
            vals = format_signif(qs, digits=digits)
        widths = [max(len(lab), len(v)) for lab, v in zip(labels, vals)]
        hdr = " ".join(lab.rjust(w) for lab, w in zip(labels, widths))
        row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return [header, hdr, row]

    def _correlation_block(self) -> str:
        # cov2cor(vcov(m)) — correlation between coefficient estimates,
        # including the intercept. Layout mirrors R's print.summary.lm:
        # strict lower triangle, first row and last column dropped.
        sd = np.sqrt(np.diag(self.V_bhat))
        corr = self.V_bhat / np.outer(sd, sd)
        names = self.column_names
        rows, cols = names[1:], names[:-1]
        # R's format() pads non-negatives with a leading space iff the
        # displayed values contain any negative (so signs line up).
        tri_vals = [corr[i + 1, j] for i in range(len(rows)) for j in range(i + 1)]
        pad_pos = any(v < 0 for v in tri_vals)
        def fmt(v: float) -> str:
            s = f"{v:.2f}"
            return s if s.startswith("-") or not pad_pos else " " + s
        cells = [
            [fmt(corr[i + 1, j]) if j <= i else "" for j in range(len(cols))]
            for i in range(len(rows))
        ]
        col_w = [
            max(len(cols[j]), max(len(cells[i][j]) for i in range(len(rows))))
            for j in range(len(cols))
        ]
        row_w = max(len(r) for r in rows)
        hdr = " " * row_w + " " + " ".join(c.ljust(w) for c, w in zip(cols, col_w))
        lines = [hdr.rstrip()]
        for i, r in enumerate(rows):
            line = r.ljust(row_w) + " " + " ".join(
                cells[i][j].ljust(col_w[j]) for j in range(len(cols))
            )
            lines.append(line.rstrip())
        return "\n".join(lines)

    def summary(self, digits=4, cor=False):
        """R: ``summary.lm()`` — returns a :class:`SummaryLm` whose
        ``__repr__`` is the R-style print block and whose attributes
        (``.sigma``, ``.r_squared``, ``.fstatistic``, …) mirror the
        components of R's ``summary.lm`` return value.

        For a multivariate (mlm) fit, returns a :class:`SummaryMlm` — R's
        ``summary.mlm``: a list of per-response ``summary.lm`` objects, printed
        and indexable as ``Response <name>``.
        """
        if getattr(self, "_is_mlm", False):
            return SummaryMlm(self, digits=digits, cor=cor)
        return SummaryLm(self, digits=digits, cor=cor)

    def plot_observed_fitted(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        y = self.y.to_numpy().astype(float)
        yhat = self.yhat["fit"].to_numpy().astype(float)
        ax.scatter(yhat, y, facecolor=facecolor, edgecolor=edgecolor)
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--")
        _label_top_n(ax, yhat, y, scores=self._residuals_arr, n=label_n)
        ax.set_xlabel("Fitted")
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
            fig, ax = plt.subplots(figsize=figsize)
        yhat = self.yhat["fit"].to_numpy().astype(float)
        r = self._residuals_arr
        ax.scatter(yhat, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--")
        if smooth:
            xs, ys = _lowess(yhat, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, r, scores=r, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. Fitted Plot")
        return ax

    def plot_qq(
        self,
        ax=None,
        figsize=None,
        facecolor="none",
        edgecolor="black",
        label_n=3,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        _qq_plot(ax, self.std_residuals, label_n=label_n)
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
            fig, ax = plt.subplots(figsize=figsize)
        yhat = self.yhat["fit"].to_numpy().astype(float)
        s = np.sqrt(np.abs(self.std_residuals))
        ax.scatter(yhat, s, facecolor=facecolor, edgecolor=edgecolor)
        if smooth:
            xs, ys = _lowess(yhat, s)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        _label_top_n(ax, yhat, s, scores=self.std_residuals, n=label_n)
        ax.set_xlabel("Fitted")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Std.\ Residuals}|}$")
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
            fig, ax = plt.subplots(figsize=figsize)
        h = self.leverage
        r = self.std_residuals

        # R's plot.lm swaps panel 5 to a factor-level stripchart when hat
        # values are essentially constant (pure-ANOVA, balanced one-way
        # design, etc.). The standard Residuals-vs-Leverage view is
        # degenerate then — every point stacks at the same h, and the
        # Cook's contours sweep over leverages that don't exist in the
        # data, looking informative but meaning nothing.
        h_mean = float(np.mean(h))
        if h_mean > 0 and bool(np.all(np.abs(h - h_mean) < 1e-10 * h_mean)):
            return self._plot_constant_leverage(
                ax=ax, r=r, facecolor=facecolor, edgecolor=edgecolor,
                label_n=label_n,
            )

        ax.scatter(h, r, facecolor=facecolor, edgecolor=edgecolor)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if smooth:
            xs, ys = _lowess(h, r)
            ax.plot(xs, ys, color="red", linewidth=1.0)
        # Cook's contours: D_i = (r_i^2 / p) · h_i / (1 - h_i)
        # ⇒ r = ±sqrt(c · p · (1 - h) / h)
        ymin, ymax = ax.get_ylim()
        h_max = float(np.clip(h.max() * 1.1, 1e-3, 0.999))
        h_grid = np.linspace(1e-3, h_max, 200)
        for c in cook_levels:
            rline = np.sqrt(c * self.p * (1 - h_grid) / h_grid)
            ax.plot(h_grid, rline, color="red", linestyle="--", linewidth=0.8)
            ax.plot(h_grid, -rline, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylim(ymin, ymax)
        # label by Cook's distance
        cook = (r ** 2 / self.p) * h / np.clip(1 - h, 1e-12, None)
        _label_top_n(ax, h, r, scores=cook, n=label_n)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Residuals vs. Leverage")
        return ax

    def _plot_constant_leverage(self, *, ax, r, facecolor, edgecolor, label_n):
        """Stripchart of standardized residuals vs factor-level combinations
        — R's fallback for plot.lm panel 5 when leverage is constant.
        Levels of every categorical RHS predictor are concatenated with
        ``:`` (matching R's ``apply(..., paste, collapse=":")``); models
        with no categorical predictor fall back to observation index."""
        from ..formula import referenced_columns

        referenced = referenced_columns(self._expanded)
        factor_cols = [
            c for c in self._design_data.columns
            if c in referenced
            and self._design_data[c].dtype in (pl.Enum, pl.Categorical, pl.Utf8)
        ]

        if factor_cols:
            keys = self._design_data.select(factor_cols).to_numpy().astype(str)
            level_labels = [":".join(row) for row in keys]
            unique_levels = list(dict.fromkeys(level_labels))
            level_to_x = {lab: i for i, lab in enumerate(unique_levels)}
            x_pos = np.array([level_to_x[lab] for lab in level_labels],
                             dtype=float)
            ax.scatter(x_pos, r, facecolor=facecolor, edgecolor=edgecolor)
            ax.set_xticks(range(len(unique_levels)))
            ax.set_xticklabels(unique_levels)
            ax.set_xlim(-0.5, len(unique_levels) - 0.5)
            ax.set_xlabel("Factor Level Combinations")
        else:
            x_pos = np.arange(len(r), dtype=float)
            ax.scatter(x_pos, r, facecolor=facecolor, edgecolor=edgecolor)
            ax.set_xlabel("Obs. number")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if label_n:
            _label_top_n(ax, x_pos, r, scores=np.abs(r), n=label_n)
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Constant Leverage:\nResiduals vs Factor Levels")
        return ax

    def plot(self, figsize=None, smooth=True, label_n=3):
        """4-panel diagnostic display: Residuals, Q-Q, Scale-Location, Leverage."""
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        self.plot_residuals(ax=axes[0, 0], smooth=smooth, label_n=label_n)
        self.plot_qq(ax=axes[0, 1], label_n=label_n)
        self.plot_scale_location(ax=axes[1, 0], smooth=smooth, label_n=label_n)
        self.plot_leverage(ax=axes[1, 1], smooth=smooth, label_n=label_n)
        fig.tight_layout()
        return fig

    def plot_contrast(
        self, features=None, figsize=None, subplots=None, away_from="median"
    ):

        """Visreg style contrast plot.

        "It showed the effect of changing Xj away from an arbitrary point xj*;
        the choice of xj* thereby determines the intercept, as the line by definition passes through (xj*, 0.)
        The equation of this line is y = (x - xj*)bj.

        Ref:
            Breheny & Burchett (2017)
        Note:
            https://stats.stackexchange.com/questions/520774/questions-concerning-visualizing-model-results-with-the-r-package-visreg
        """

        if features is None:
            num_subplots = self.df_model
            features = self.feature_names
        else:
            if type(features) is str:
                num_subplots = 1
                features = [features]
            else:
                num_subplots = len(features)

        if figsize is None:
            figsize = np.array([4 * num_subplots, 3])

        if subplots is None:
            subplots = (1, num_subplots)

        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()

        for i, name in enumerate(features):

            xx = self.X[name].to_numpy().astype(float)

            if away_from == "median":
                xxbar = float(np.median(xx))
            elif away_from == "mean":
                xxbar = float(np.mean(xx))
            elif away_from == "0":
                xxbar = 0.0
            else:
                raise ValueError(f'The Input value for "{away_from}" is not supported.')

            X_arr = self.X.with_columns(pl.lit(xxbar).alias(name)).to_numpy().astype(float)
            rj = self.y.to_numpy().astype(float) - (X_arr @ self._bhat_arr)
            ax[i].scatter(xx, rj, color="gray", facecolor="none", edgecolor="black")
            ax[i].set_xlabel(name)
            ax[i].set_ylabel("Δ" + self.y.name)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)

            se_scalar = float(self.se_bhat[name].item())
            Vx = (xx - xxbar) ** 2 * se_scalar ** 2
            se = np.sqrt(Vx)

            tt = _dist.qt(1 - 0.05 / 2, self.df_residuals)
            yy = (xx - xxbar) * float(self.bhat[name].item())
            idx_sorted = np.argsort(xx)
            ax[i].plot(xx[idx_sorted], yy[idx_sorted], color="black")
            ax[i].fill_between(
                xx[idx_sorted],
                yy[idx_sorted] + tt * se[idx_sorted],
                yy[idx_sorted] - tt * se[idx_sorted],
                alpha=0.5,
            )

        fig.tight_layout()
        return fig

    def plot_conditional(
        self, features=None, figsize=None, subplots=None, away_from="median"
    ):

        """Visreg style conditional plot.

        It showed the relationship between E(Y) and Xj while holding other variables constant (mean or median).

        Ref:
            Breheny & Burchett (2017)
        Note:
            https://stats.stackexchange.com/questions/520774/questions-concerning-visualizing-model-results-with-the-r-package-visreg
        """

        if features is None:
            num_subplots = self.df_model
            features = self.feature_names
        else:
            if type(features) is str:
                num_subplots = 1
                features = [features]
            else:
                num_subplots = len(features)

        if figsize is None:
            figsize = np.array([4 * num_subplots, 3])

        if subplots is None:
            subplots = (1, num_subplots)

        if num_subplots > 1:
            fig, ax = plt.subplots(subplots[0], subplots[1], figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        ax = np.array([ax]).flatten()

        for i, name in enumerate(features):

            xx = self.X[name].to_numpy().astype(float)

            if away_from == "median":
                repl = {
                    name1: float(self.X[name1].median())
                    for name1 in self.column_names
                    if name1 != name
                }
            elif away_from == "mean":
                repl = {
                    name1: float(self.X[name1].mean())
                    for name1 in self.column_names
                    if name1 != name
                }
            elif away_from == "0":
                repl = {name1: 0.0 for name1 in self.column_names if name1 != name}
            else:
                raise ValueError('The Input value for "away_from" is not supported.')

            Xnew = self.X.with_columns(
                [pl.lit(v).alias(k) for k, v in repl.items()]
            ).to_numpy().astype(float)

            rj = self._residuals_arr + (Xnew @ self._bhat_arr)
            ax[i].scatter(xx, rj, color="gray", facecolor="none", edgecolor="black")
            ax[i].set_xlabel(name)
            ax[i].set_ylabel(self.y.name)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)

            Vx = Xnew @ self.V_bhat @ Xnew.T
            se = np.sqrt(np.diag(Vx))

            tt = _dist.qt(1 - 0.05 / 2, self.df_residuals)
            yy = (Xnew @ self._bhat_arr).flatten()

            ax[i].plot(xx[np.argsort(xx)], yy[np.argsort(xx)], color="black")
            ax[i].fill_between(
                xx[np.argsort(xx)],
                yy[np.argsort(xx)] + tt * se[np.argsort(xx)],
                yy[np.argsort(xx)] - tt * se[np.argsort(xx)],
                alpha=0.5,
            )

        fig.tight_layout()
        return fig


#################
# Estimate bhat #
#################


def _chol2inv(R: np.ndarray) -> np.ndarray:
    """R's ``chol2inv``: ``(RᵀR)⁻¹`` from an upper-triangular factor ``R``,
    via LAPACK ``dpotri`` — the exact routine R calls. Used on the QR ``R``
    factor (where ``RᵀR == XᵀWX``) to get ``(XᵀWX)⁻¹`` the way ``summary.lm``
    does (``chol2inv(Qr$qr[p1,p1])``, lm.R:326). The sign of ``R``'s diagonal
    is irrelevant (``dpotri`` inverts then forms ``R⁻¹R⁻ᵀ``)."""
    from scipy.linalg.lapack import dpotri

    inv, info = dpotri(np.ascontiguousarray(R), lower=0)
    if info != 0:
        # singular factor — fall back to the triangular-solve form (same value)
        Rinv = solve_triangular(R, np.eye(R.shape[0]))
        return Rinv @ Rinv.T
    return np.triu(inv) + np.triu(inv, 1).T   # dpotri fills one triangle; mirror


def _qty_householder(qr_c: np.ndarray, tau: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Apply ``Qᵀ`` (from a ``dgeqrf`` compact factorization) to a vector ``c``
    via LAPACK ``dormqr`` — the full length-n ``Qᵀc`` in O(np), without ever
    forming the n×n ``Q`` (this is how R's ``dqrsl`` produces the ``effects``
    vector cheaply; forming ``Q`` would reintroduce the O(n²) we just removed)."""
    c2 = np.asfortranarray(c.reshape(-1, 1))
    lwork = int(np.asarray(dormqr("L", "T", qr_c, tau, c2, -1)[1]).reshape(-1)[0].real)
    out, _, info = dormqr("L", "T", qr_c, tau, c2, lwork)
    if info != 0:
        raise np.linalg.LinAlgError(f"dormqr failed (info={info})")
    return out.reshape(-1)


def _qr(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """QR weighted least squares — R's ``lm.wfit`` (the only method R supports).

    ``w`` is the length-n prior-weight vector. We whiten by √w row-scaling
    (R fits on ``√w·X`` / ``√w·y``) rather than ``cholesky(W)`` so zero
    weights don't break the factorization (a zero on the diagonal makes W
    only positive-*semi*definite).

    Returns ``(β̂, R, effects, qraux)`` mirroring R's ``Cdqrls`` outputs:
    ``R`` is the upper-triangular factor (``RᵀR == XᵀWX``, fed to ``chol2inv``);
    ``effects`` is the length-n ``Qᵀ(√w·y)`` (R's ``$effects``, first ``p``
    named, used by ``anova``/``effects``); ``qraux`` are the Householder scalar
    factors (R's ``$qr$qraux``). β̂ is the triangular solve on the first ``p``
    effects — bit-identical to the former ``solve_triangular(R, Qᵀy)`` path."""
    wts = np.sqrt(w)
    Xhat = wts[:, None] * X
    yhat = wts * y

    qr_c, qraux, _work, info = dgeqrf(Xhat)
    if info != 0:
        raise np.linalg.LinAlgError(f"dgeqrf failed (info={info})")
    p = X.shape[1]
    R = np.triu(qr_c[:p, :p])
    effects = _qty_householder(qr_c, qraux, yhat)
    b = solve_triangular(R, effects[:p])
    return b, R, effects, qraux
