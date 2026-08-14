"""Base-R ``stats`` formula / model-frame helpers.

Faithful ports of the formula-manipulation, NA-handling and orthogonal-
polynomial helpers from R's ``stats`` package:

* **Formula algebra** (``models.R``, ``update.R``) — :func:`reformulate`,
  :func:`as_formula`, :func:`update_formula`, :func:`delete_response`,
  :func:`drop_terms`, :func:`DF2formula`, :func:`get_all_vars`. These operate
  on hea's canonical formula *strings* (hea represents formulas as strings, so
  every function returns the deparsed formula text rather than an R language
  object). They reuse hea's own formula parser / term-expander
  (:mod:`hea.formula`), so ``update_formula`` simplifies exactly like R's
  ``terms.formula(simplify=TRUE)`` (expands ``*``, drops ``-`` terms, dedups,
  and re-orders main-effects-before-interactions).

* **NA handling** (``nafns.R``) — :func:`na_pass`, :func:`na_fail`,
  :func:`na_exclude`, :func:`na_action`, and the pad/reconstruct primitives
  :func:`naresid` / :func:`napredict` / :func:`naprint` over an
  :class:`NAAction` index object (``na_omit`` / ``complete_cases`` already live
  in :mod:`hea.R.shape`).

* **Model-frame class** (``models.R``) — :func:`MFclass` (R's ``.MFclass``).

* **Orthogonal polynomials** (``contr.poly.R``) — :func:`poly` / :func:`polym`
  / :func:`predict_poly`, returning a :class:`Poly` ndarray that carries the
  ``coefs`` needed for safe prediction. The non-raw *fit* path runs through R's
  exact ``qr()`` (``dqrdc2`` + ``qr.qy``, :mod:`hea.R.linalg`), so it inherits
  the documented ≤2-ulp lm-QR/FMA residual; the ``raw=True`` and prediction
  (three-term recurrence, no QR) paths are 0-ulp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..formula import (
    BinOp,
    Call,
    Dot,
    Empty,
    Formula,
    Literal,
    Name,
    Paren,
    Subscript,
    UnaryOp,
    deparse,
    expand,
    parse,
)
from .linalg import dqrdc2, dqrsl

__all__ = [
    "DF2formula",
    "MFclass",
    "NAAction",
    "Poly",
    "as_formula",
    "delete_response",
    "drop_terms",
    "get_all_vars",
    "na_action",
    "na_exclude",
    "na_fail",
    "na_pass",
    "napredict",
    "naprint",
    "naresid",
    "poly",
    "polym",
    "predict_poly",
    "reformulate",
    "update_formula",
]


# ---------------------------------------------------------------------------
# Formula algebra — operate on hea's canonical formula strings.
# ---------------------------------------------------------------------------


def _as_formula_node(obj) -> Formula:
    """Coerce ``obj`` (a formula string or already-parsed :class:`Formula`) to a
    :class:`hea.formula.Formula` AST."""
    if isinstance(obj, Formula):
        return obj
    if isinstance(obj, str):
        return parse(obj)
    raise TypeError(f"expected a formula string, got {type(obj).__name__}")


def _deparse_formula(f: Formula) -> str:
    """R's formula deparse: two-sided ``lhs ~ rhs`` (spaces around ``~``),
    one-sided ``~rhs`` (no leading space)."""
    if f.lhs is None:
        return "~" + deparse(f.rhs)
    return f"{deparse(f.lhs)} ~ {deparse(f.rhs)}"


def as_formula(object, env=None) -> str:
    """R: ``as.formula(object)`` — coerce to a (canonical) formula.

    ``object`` is a formula string (or an already-parsed :class:`Formula`);
    the return is the canonicalized formula text (R returns a formula object,
    which prints as exactly this string). No term simplification is done —
    ``as_formula("y ~ a*b")`` stays ``"y ~ a * b"`` — matching R. ``env`` is
    accepted for signature compatibility and ignored (hea formulas carry no
    environment).
    """
    return _deparse_formula(_as_formula_node(object))


def reformulate(termlabels, response=None, intercept=True, env=None) -> str:
    """R: ``reformulate(termlabels, response, intercept)`` — build a formula
    from a character vector of term labels.

    ``termlabels`` is a string or list of strings; they are joined with ``+``
    (R keeps them as written, so ``"log(x)"`` / ``"a:b"`` pass through). With
    ``intercept=False`` a ``- 1`` is appended; an empty ``termlabels`` with
    ``intercept=True`` yields ``~1``. ``response`` (a string — a name or a call
    like ``"Surv(t, e)"``) becomes the LHS; ``None`` gives a one-sided formula.
    Returns the deparsed formula string.
    """
    if isinstance(termlabels, str):
        labels = [termlabels]
    else:
        labels = list(termlabels)
        if not all(isinstance(t, str) for t in labels):
            raise TypeError("'termlabels' must be a character vector")
    if intercept and not labels:
        labels = ["1"]
    termtext = "+".join(labels)
    if not intercept:
        termtext = f"{termtext} - 1"
    # str2lang(termtext): parse the RHS, then canonicalize via deparse.
    rhs_str = deparse(parse(termtext).rhs)
    if response is None:
        return f"~{rhs_str}"
    if not isinstance(response, str):
        raise TypeError("'response' must be a character string")
    resp_str = deparse(parse(response).rhs)
    return f"{resp_str} ~ {rhs_str}"


def delete_response(formula) -> str:
    """R: ``delete.response(terms(formula))`` — drop the LHS, keep the RHS
    verbatim (no simplification). ``delete_response("y ~ a*b")`` → ``"~a * b"``.
    """
    f = _as_formula_node(formula)
    return "~" + deparse(f.rhs)


def _rebuild_from_expanded(lhs_node, term_labels, intercept) -> str:
    """Reconstruct a formula string from expanded term labels + intercept flag,
    mirroring R's ``terms.formula(simplify=TRUE)`` / ``fixFormulaObject``:
    ``rhs = "1"`` when empty, and ``- 1`` appended when the intercept is
    dropped (so an empty no-intercept model deparses to ``1 - 1``, as in R)."""
    rhs = " + ".join(term_labels) if term_labels else "1"
    if not intercept:
        rhs = f"{rhs} - 1"
    if lhs_node is None:
        return "~" + rhs
    return f"{deparse(lhs_node)} ~ {rhs}"


def drop_terms(formula, dropx=None, keep_response=False) -> str:
    """R: ``drop.terms(termobj, dropx, keep.response)`` — drop RHS terms by
    (1-based) position.

    ``dropx`` is a 1-based index or list of indices into the *expanded* term
    labels (so ``terms(y~a*b)`` has labels ``a, b, a:b``). With no ``dropx``:
    the response is kept (``keep_response=True``) or dropped
    (:func:`delete_response`). Returns the deparsed formula string.
    """
    f = _as_formula_node(formula)
    if not dropx:
        if keep_response:
            return _deparse_formula(f)
        return delete_response(f)
    ef = expand(f)
    labels = ef.term_labels
    drop = {dropx} if isinstance(dropx, int) else set(dropx)
    kept = [lab for i, lab in enumerate(labels, start=1) if i not in drop]
    response = deparse(f.lhs) if (keep_response and f.lhs is not None) else None
    return reformulate(kept, response=response, intercept=ef.intercept)


def DF2formula(x) -> str:
    """R: ``DF2formula(x)`` — first column ``~`` the rest, from a data frame's
    column names. A single-column frame gives the one-sided ``~col``. ``x`` is a
    polars DataFrame (or any object with a ``.columns`` list of names)."""
    if isinstance(x, pl.DataFrame) or hasattr(x, "columns"):
        names = list(x.columns)
    else:
        names = list(x)
    if not names:
        raise ValueError("cannot create a formula from a zero-column data frame")
    if len(names) == 1:
        return f"~{names[0]}"
    return f"{names[0]} ~ {' + '.join(names[1:])}"


def _subst_dot(node, repl):
    """Return a copy of ``node``'s AST with every :class:`Dot` replaced by the
    ``repl`` node (R's ``.`` substitution in ``update.formula``). The parse
    cache is shared/immutable, so this never mutates in place."""
    if node is None:
        return None
    if isinstance(node, Dot):
        return repl
    if isinstance(node, (Name, Literal, Empty)):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _subst_dot(node.operand, repl))
    if isinstance(node, BinOp):
        return BinOp(node.op, _subst_dot(node.left, repl), _subst_dot(node.right, repl))
    if isinstance(node, Call):
        return Call(
            node.fn,
            [_subst_dot(a, repl) for a in node.args],
            {k: _subst_dot(v, repl) for k, v in node.kwargs.items()},
        )
    if isinstance(node, Paren):
        return Paren(_subst_dot(node.expr, repl))
    if isinstance(node, Subscript):
        return Subscript(
            _subst_dot(node.obj, repl), [_subst_dot(i, repl) for i in node.idx]
        )
    return node


def update_formula(old, new) -> str:
    """R: ``update.formula(old, new)`` — substitute ``.`` then simplify.

    Each ``.`` in ``new`` is replaced by the matching side of ``old`` (LHS with
    LHS, RHS with RHS); a one-sided ``new`` keeps ``old``'s response. The result
    is then run through hea's term expander — exactly R's
    ``terms.formula(simplify=TRUE)`` — so ``*`` expands, ``-`` removes terms,
    duplicates collapse, and terms sort main-effects-first. Returns the
    simplified formula string, e.g. ``update_formula("y ~ a*b", ". ~ . - a:b")``
    → ``"y ~ a + b"``.
    """
    of = _as_formula_node(old)
    nf = _as_formula_node(new)
    lhs = of.lhs if nf.lhs is None else _subst_dot(nf.lhs, of.lhs)
    rhs = _subst_dot(nf.rhs, of.rhs)
    ef = expand(Formula(lhs=lhs, rhs=rhs))
    return _rebuild_from_expanded(lhs, ef.term_labels, ef.intercept)


def _all_vars(node, out: list[str]) -> None:
    """Ordered, deduplicated variable-name collector (R's ``all.vars``): first
    appearance wins, ``data$col`` contributes only ``col``."""
    if node is None:
        return
    if isinstance(node, Name):
        if node.ident not in out:
            out.append(node.ident)
    elif isinstance(node, BinOp):
        if node.op == "$":
            if isinstance(node.right, Name) and node.right.ident not in out:
                out.append(node.right.ident)
            return
        _all_vars(node.left, out)
        _all_vars(node.right, out)
    elif isinstance(node, UnaryOp):
        _all_vars(node.operand, out)
    elif isinstance(node, Paren):
        _all_vars(node.expr, out)
    elif isinstance(node, Call):
        for a in node.args:
            _all_vars(a, out)
        for v in node.kwargs.values():
            _all_vars(v, out)
    elif isinstance(node, Subscript):
        _all_vars(node.obj, out)
        for i in node.idx:
            _all_vars(i, out)


def get_all_vars(formula, data) -> pl.DataFrame:
    """R: ``get_all_vars(formula, data)`` — the sub-frame of every base variable
    the formula references.

    Uses ``all.vars`` semantics (``log(x2)`` contributes the raw column ``x2``,
    not the transformed value), in first-appearance order (LHS then RHS).
    ``data`` is a polars DataFrame; raises if a referenced variable is absent.
    """
    f = _as_formula_node(formula)
    names: list[str] = []
    _all_vars(f.lhs, names)
    _all_vars(f.rhs, names)
    missing = [n for n in names if n not in data.columns]
    if missing:
        raise KeyError(f"variable(s) {missing} not found in data (get_all_vars)")
    return data.select(names)


# ---------------------------------------------------------------------------
# NA handling — nafns.R
# ---------------------------------------------------------------------------


@dataclass
class NAAction:
    """R's ``na.action`` index object (class ``"omit"`` / ``"exclude"``).

    ``omit`` holds the 0-based positions of the dropped rows in the *original*
    data (R stores 1-based); ``names`` holds their row labels. Only the
    ``"exclude"`` variant pads results back to full length
    (:func:`naresid` / :func:`napredict`); both print the same via
    :func:`naprint`.
    """

    omit: np.ndarray
    kind: str = "exclude"
    names: list | None = None

    def __post_init__(self):
        self.omit = np.asarray(self.omit, dtype=int).reshape(-1)

    def __len__(self) -> int:
        return int(self.omit.size)


def na_pass(object, **kwargs):
    """R: ``na.pass(object)`` — return ``object`` unchanged (keep the NAs)."""
    return object


def na_fail(object, **kwargs):
    """R: ``na.fail(object)`` — return ``object`` if it has no missing values,
    otherwise raise. ``object`` is a polars DataFrame / Series or an ndarray."""
    from .shape import complete_cases

    ok = complete_cases(object)
    ok = ok.to_numpy() if isinstance(ok, pl.Series) else np.asarray(ok)
    if not ok.all():
        raise ValueError("missing values in object")
    return object


def _na_positions(object) -> tuple[np.ndarray, list]:
    """0-based positions of incomplete rows + their labels (row index as str)."""
    from .shape import complete_cases

    ok = complete_cases(object)
    ok = ok.to_numpy() if isinstance(ok, pl.Series) else np.asarray(ok)
    omit = np.nonzero(~ok)[0]
    return omit, [str(i) for i in omit]


def na_exclude(object, **kwargs):
    """R: ``na.exclude(object)`` — drop rows with any NA, returning an
    ``(cleaned, na_action)`` pair.

    ``cleaned`` is ``object`` with the incomplete rows removed (a polars
    DataFrame / Series, or an ndarray); ``na_action`` is the :class:`NAAction`
    of class ``"exclude"`` describing the dropped rows — feed it to
    :func:`naresid` / :func:`napredict` to pad predictions/residuals back to the
    original length. (Unlike R, which attaches the action as an attribute, hea
    returns it explicitly since polars frames carry no attributes.) When nothing
    is dropped, ``na_action`` is ``None``.
    """
    omit, names = _na_positions(object)
    if isinstance(object, (pl.DataFrame, pl.Series)):
        cleaned = object.drop_nulls()
    else:
        arr = np.asarray(object)
        keep = np.ones(arr.shape[0], dtype=bool)
        keep[omit] = False
        cleaned = arr[keep]
    action = None if omit.size == 0 else NAAction(omit, kind="exclude", names=names)
    return cleaned, action


def na_action(object):
    """R: ``na.action(object)`` — the ``na.action`` carried by a fitted model
    (hea models stash it on ``._na_action``), else ``None``."""
    if isinstance(object, NAAction):
        return object
    return getattr(object, "_na_action", None)


def naresid(omit, x):
    """R: ``naresid(omit, x)`` — reconstruct residuals to the original length.

    For an ``"exclude"`` :class:`NAAction`, pad ``x`` (one value per kept row)
    back to ``len(x) + len(omit)`` with NaN at the dropped positions (R's
    ``keep <- rep(NA, n+len(omit)); keep[-omit] <- 1:n; x[keep]``). For an
    ``"omit"`` action (or ``None``) ``x`` is returned unchanged. Works on a
    1-D vector or a 2-D array (rows padded)."""
    return _na_reconstruct(omit, x)


def napredict(omit, x):
    """R: ``napredict(omit, x)`` — reconstruct fitted values to the original
    length. Identical to :func:`naresid` (``napredict.exclude`` calls it)."""
    return _na_reconstruct(omit, x)


def _na_reconstruct(omit, x):
    if x is None or omit is None:
        return x
    if not isinstance(omit, NAAction) or omit.kind != "exclude":
        return x
    x = np.asarray(x, dtype=float)
    drop = omit.omit
    n = x.shape[0]
    total = n + drop.size
    keep = np.ones(total, dtype=bool)
    keep[drop] = False
    if x.ndim == 1:
        out = np.full(total, np.nan)
        out[keep] = x
    else:
        out = np.full((total,) + x.shape[1:], np.nan)
        out[keep] = x
    return out


def naprint(x, **kwargs) -> str:
    """R: ``naprint(x)`` — the "*N* observations deleted due to missingness"
    message for an :class:`NAAction` (or an integer count). Any other argument
    (including ``None``) gives ``""`` — R's ``naprint.default``."""
    if isinstance(x, NAAction):
        n = len(x)
    elif isinstance(x, (int, np.integer)):
        n = int(x)
    else:
        return ""
    noun = "observation" if n == 1 else "observations"
    return f"{n} {noun} deleted due to missingness"


# ---------------------------------------------------------------------------
# Model-frame class — models.R
# ---------------------------------------------------------------------------


def _series_is_ordered(x: pl.Series) -> bool:
    """Whether a hea factor Series is an R-style *ordered* factor — the
    ``_hea_ordered`` local marker (from ``ordered()``) or the column name in the
    formula engine's ordered-column context."""
    if getattr(x, "_hea_ordered", False):
        return True
    name = getattr(x, "name", None)
    if name:
        from ..formula import _ORDERED_COLS_CV

        return name in _ORDERED_COLS_CV.get()
    return False


def MFclass(x) -> str:
    """R: ``.MFclass(x)`` — the model-frame class of a variable, the label
    ``model.matrix`` keys its handling off: ``"logical"``, ``"ordered"``,
    ``"factor"``, ``"character"``, ``"nmatrix.<ncol>"`` for a numeric matrix,
    ``"numeric"`` (integers included), else ``"other"``.

    Accepts a polars Series (its dtype decides), a numpy array, or a Python
    sequence.
    """
    if isinstance(x, pl.Series):
        dt = x.dtype
        if dt == pl.Boolean:
            return "logical"
        if isinstance(dt, (pl.Categorical, pl.Enum)):
            # hea represents *both* ordered and unordered factors as a polars
            # Enum; the ordered signal is the ``_hea_ordered`` local marker
            # (set by ``ordered()``) or the column name in the ``_ORDERED_COLS``
            # context — matching how the formula engine detects ordered factors.
            if _series_is_ordered(x):
                return "ordered"
            return "factor"
        if dt == pl.String:
            return "character"
        if dt.is_numeric():
            return "numeric"
        return "other"
    arr = np.asarray(x)
    if arr.dtype == bool:
        return "logical"
    if arr.dtype.kind in "US":
        return "character"
    if arr.dtype.kind in "iuf" or arr.dtype.kind == "c":
        if arr.ndim == 2:
            return f"nmatrix.{arr.shape[1]}"
        return "numeric"
    return "other"


# ---------------------------------------------------------------------------
# Orthogonal polynomials — contr.poly.R
# ---------------------------------------------------------------------------


class Poly(np.ndarray):
    """R ``poly()`` result — the (n × degree) polynomial basis matrix carrying
    the ``coefs`` (``alpha`` / ``norm2``), ``degree`` vector, and column names
    needed for safe prediction (:func:`predict_poly`)."""

    def __new__(cls, arr, *, coefs=None, degree=None, colnames=None, raw=False):
        obj = np.asarray(arr, dtype=float).view(cls)
        obj.coefs = coefs
        obj.degree = degree
        obj.colnames = colnames
        obj.raw = raw
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.coefs = getattr(obj, "coefs", None)
        self.degree = getattr(obj, "degree", None)
        self.colnames = getattr(obj, "colnames", None)
        self.raw = getattr(obj, "raw", False)


def _r_mean(x: np.ndarray) -> float:
    """R's ``mean.default`` — two-pass long-double mean, cast to double."""
    xl = np.asarray(x, dtype=np.longdouble)
    n = xl.size
    tmp = xl.sum() / n
    tmp = tmp + (xl - tmp).sum() / n
    return float(tmp)


def _poly_fit(x: np.ndarray, degree: int):
    """Non-raw ``poly`` fit: R's ``qr()`` (``dqrdc2`` + ``qr.qy``) on the
    centered Vandermonde, then column normalization + the ``alpha`` recurrence
    coefficients. Returns ``(Z, alpha, norm2)`` with ``Z`` the n × degree
    orthonormal basis (constant column dropped)."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("missing values are not allowed in 'poly'")
    nu = np.unique(x).size
    if degree >= nu:
        raise ValueError("'degree' must be less than number of unique points")
    xbar = _r_mean(x)
    xc = x - xbar
    n = x.size
    p = degree + 1
    X = np.column_stack([xc**d for d in range(p)])  # outer(x, 0:degree, ^)
    qr, qraux, _jpvt, rank = dqrdc2(X, 1e-7)
    if rank < degree:
        raise ValueError("'degree' must be less than number of unique points")
    Z = np.empty((n, p))
    for j in range(p):  # z <- QR$qr*(row==col)
        col = np.zeros(n)
        col[j] = qr[j, j]
        qy, *_ = dqrsl(qr, n, rank, qraux, col, 10000)  # qr.qy(QR, z)
        Z[:, j] = qy
    Zl = Z.astype(np.longdouble)
    norm2 = np.asarray((Zl**2).sum(axis=0), dtype=np.longdouble)  # colSums LD
    alpha = np.asarray(
        (xc.astype(np.longdouble)[:, None] * Zl**2).sum(axis=0) / norm2
        + np.longdouble(xbar),
        dtype=float,
    )[:degree]
    norm2 = np.concatenate([[1.0], np.asarray(norm2, dtype=float)])
    Zn = Z / np.sqrt(norm2[1:])
    return Zn[:, 1:], alpha, norm2


def _poly_predict(x: np.ndarray, degree: int, coefs: dict) -> np.ndarray:
    """Non-raw ``poly`` prediction: the three-term recurrence using stored
    ``alpha`` / ``norm2`` (no QR — 0-ulp vs R)."""
    x = np.asarray(x, dtype=float)
    alpha = np.asarray(coefs["alpha"], dtype=float)
    norm2 = np.asarray(coefs["norm2"], dtype=float)
    n = x.size
    Z = np.ones((n, degree + 1))
    Z[:, 1] = x - alpha[0]
    for i in range(2, degree + 1):
        Z[:, i] = (x - alpha[i - 1]) * Z[:, i - 1] - (norm2[i] / norm2[i - 1]) * Z[
            :, i - 2
        ]
    Z = Z / np.sqrt(norm2[1:])
    return Z[:, 1:]


def poly(x, *args, degree=1, coefs=None, raw=False, simple=False):
    """R: ``poly(x, degree, raw=, coefs=)`` — orthogonal (or raw) polynomials.

    Returns a :class:`Poly` (an ndarray with the ``coefs`` / ``degree`` /
    ``colnames`` R attaches). Forms mirror R:

    * ``poly(x, 3)`` / ``poly(x, degree=3)`` — degree-3 orthogonal basis.
    * ``poly(x, raw=True)`` — the raw powers ``x, x², …`` (0-ulp).
    * ``poly(x, degree=d, coefs=<from a fit>)`` — safe prediction via the
      three-term recurrence (0-ulp).
    * ``poly(x1, x2, degree=d)`` or a 2-D ``x`` — delegates to :func:`polym`.

    The non-raw *fit* path runs R's exact ``qr()`` so it carries the documented
    ≤2-ulp lm-QR/FMA residual; ``raw`` and prediction are 0-ulp. ``simple=True``
    returns a bare ndarray (no ``coefs`` / class), as in R.
    """
    # `...`: an unnamed scalar is the degree; anything else → polym.
    if args:
        if len(args) == 1 and np.ndim(args[0]) == 0:
            degree = args[0]
        else:
            return polym(x, *args, degree=degree, coefs=coefs, raw=raw)
    x = np.asarray(x)
    if x.ndim == 2:  # matrix x → polym
        cols = [x[:, j] for j in range(x.shape[1])]
        return polym(*cols, degree=degree, coefs=coefs, raw=raw)
    degree = int(degree)
    if degree < 1:
        raise ValueError("'degree' must be at least 1")
    x = x.astype(float)
    if raw:
        Z = np.column_stack([x**d for d in range(1, degree + 1)])
        names = [str(d) for d in range(1, degree + 1)]
        if simple:
            return Z
        return Poly(
            Z, coefs=None, degree=np.arange(1, degree + 1), colnames=names, raw=True
        )
    if coefs is None:
        Z, alpha, norm2 = _poly_fit(x, degree)
        co = {"alpha": alpha, "norm2": norm2}
    else:
        Z = _poly_predict(x, degree, coefs)
        co = coefs
    names = [str(d) for d in range(1, degree + 1)]
    if simple:
        return Z
    return Poly(Z, coefs=co, degree=np.arange(1, degree + 1), colnames=names)


def _expand_grid_degrees(nd: int, degree: int) -> np.ndarray:
    """R's ``expand.grid(rep(list(0:degree), nd))`` row order — the first
    variable varies fastest. Returns an ``(m, nd)`` integer array."""
    vals = np.arange(degree + 1)
    grids = np.meshgrid(*([vals] * nd), indexing="ij")
    # meshgrid(indexing="ij") makes the LAST axis vary fastest; expand.grid
    # varies the FIRST fastest, so reverse the column stack order accordingly.
    cols = [g.reshape(-1, order="F") for g in grids]
    return np.column_stack(cols)


def polym(*vecs, degree=1, coefs=None, raw=False):
    """R: ``polym(..., degree=, raw=, coefs=)`` — multivariate polynomials
    (tensor product of per-variable :func:`poly` bases, keeping only total
    degrees ``1..degree``).

    Called for ``poly`` with several vectors or a matrix argument. Returns a
    :class:`Poly` whose ``coefs`` is the list of per-variable coef dicts (for
    prediction). Column names are the ``"d1.d2.…"`` degree tuples, as in R.
    """
    dots = [np.asarray(v, dtype=float) for v in vecs]
    nd = len(coefs) if coefs is not None else len(dots)
    if nd == 0:
        raise ValueError("must supply one or more vectors")
    degree = int(degree)
    z = _expand_grid_degrees(nd, degree)
    s = z.sum(axis=1)
    keep = (s > 0) & (s <= degree)
    z = z[keep]
    s = s[keep]

    if coefs is None:
        a_poly = poly(dots[0], degree=degree, raw=raw, simple=(raw and nd > 1))
        if nd == 1:
            return a_poly
        n = [v.size for v in dots]
        if any(m != n[0] for m in n):
            raise ValueError("arguments must have the same length")
        co_list = None if raw else [_poly_coefs(a_poly)]
        cb = np.column_stack([np.ones(n[0]), np.asarray(a_poly)])
        res = cb[:, z[:, 0]]
        for i in range(1, nd):
            a_i = poly(dots[i], degree=degree, raw=raw, simple=raw)
            cb = np.column_stack([np.ones(n[0]), np.asarray(a_i)])
            res = res * cb[:, z[:, i]]
            if not raw:
                co_list.append(_poly_coefs(a_i))
        names = [".".join(str(v) for v in row) for row in z]
        return Poly(res, coefs=co_list, degree=s, colnames=names, raw=raw)
    else:  # prediction
        n = dots[0].size
        res = np.ones(n)
        for i in range(nd):
            a_i = poly(dots[i], degree=degree, coefs=coefs[i], simple=True)
            cb = np.column_stack([np.ones(n), np.asarray(a_i)])
            res = res * cb[:, z[:, i]]
        names = [".".join(str(v) for v in row) for row in z]
        return Poly(res, coefs=None, degree=s, colnames=names)


def _poly_coefs(p):
    """The ``coefs`` dict carried by a :class:`Poly` (from a non-raw fit)."""
    return getattr(p, "coefs", None)


def predict_poly(object: Poly, newdata):
    """R: ``predict.poly(object, newdata)`` — evaluate a fitted :class:`Poly`
    basis on new data via its stored ``coefs`` (raw bases re-power ``newdata``).
    Returns a bare ndarray (``simple=TRUE``), matching R."""
    deg = int(np.max(object.degree))
    if getattr(object, "coefs", None) is None:
        return poly(newdata, degree=deg, raw=True, simple=True)
    return poly(newdata, degree=deg, coefs=object.coefs, simple=True)
