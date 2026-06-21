"""hea.R.distance — base-R ``stats`` distance layer.

Mechanical port of R's ``src/library/stats/R/dist.R`` and the compiled core
``src/library/stats/src/distance.c`` (``C_Cdist``). This is the metric-space
layer: it has **no clustering dependency** and is independently useful
(``cmdscale``, ``mahalanobis`` and a precomputed-matrix ``hclust`` all consume a
``Dist``). ``hea.R.clustering`` imports from here one-way (acyclic).

Surface
-------
* :class:`Dist` — R's ``"dist"`` object: the packed **lower-triangle** vector
  (column-major, exactly R's layout) plus the ``Size``/``Labels``/``Diag``/
  ``Upper``/``method``/``p`` attributes.
* :func:`dist` — the six metrics (``euclidean``, ``maximum``, ``manhattan``,
  ``canberra``, ``binary``, ``minkowski``), NA-aware, bit-exact to ``stats::dist``.
* :func:`as_dist` — square matrix → ``Dist`` (the other input path to ``hclust``).
* :func:`as_matrix_dist` — ``Dist`` → full symmetric matrix.
* :func:`format_dist`, :func:`labels_dist`, :func:`print_dist` — text accessors.

Bit-exactness
-------------
``C_Cdist`` accumulates each pair's reduction **sequentially over columns**
(``dist += dev*dev`` for ``j = 0 .. nc-1``). Floating-point ``+`` is not
associative, so the column order is load-bearing. We reproduce it by keeping the
reduction loop over columns sequential while vectorizing across the independent
pairs (``for j: dist += dev_j**2`` over the whole pair vector) — identical
arithmetic order to the C loop, so the pure-Python kernel is the 0-ulp spec the
Rust kernel (end goal, see the plan) is checked against.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._dispatch import rs_fn

__all__ = [
    "Dist",
    "cmdscale",
    "dist",
    "as_dist",
    "as_matrix_dist",
    "format_dist",
    "labels_dist",
    "mahalanobis",
    "print_dist",
]

# Order matches the R function ``dist`` (the C ``enum`` is 1-based off this).
_METHODS = ("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")

# <float.h> constants used verbatim by distance.c.
_DBL_MIN = 2.2250738585072014e-308
_DBL_MAX = 1.7976931348623157e308

# Rust seam (plan build-order step 10): a ``cdist`` kernel mirroring this module's
# pure-Python ``_cdist`` 1:1 (rayon over independent pairs, sequential per-pair
# column reduction → parallel == serial bit-for-bit). ``None`` until built ⇒ the
# pure-Python path below runs and stays the spec/oracle.
_rs_cdist = rs_fn("cdist")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pmatch(x, table):
    """R ``pmatch(x, table)`` for a single string: exact, else unique prefix.

    Returns the 0-based index, or ``None`` for no/ambiguous match.
    """
    for i, t in enumerate(table):
        if t == x:
            return i
    hits = [i for i, t in enumerate(table) if t.startswith(x)]
    return hits[0] if len(hits) == 1 else None


def _lower_tri_ij(n):
    """Row/column index arrays for R's ``dist`` packing: the strict lower
    triangle in **column-major** order — for ``j = 0..n-2``, ``i = j+1..n-1``.
    """
    if n < 2:
        z = np.empty(0, dtype=np.intp)
        return z, z
    rows = np.concatenate([np.arange(j + 1, n) for j in range(n - 1)])
    cols = np.concatenate([np.full(n - 1 - j, j, dtype=np.intp) for j in range(n - 1)])
    return rows.astype(np.intp), cols


def _as_matrix(x):
    """Coerce ``x`` to a 2-D float ndarray (R ``as.matrix``); a 1-D vector
    becomes an ``n x 1`` column. Row labels are not wired yet (numeric parity
    does not need them); ``Labels`` is ``None`` until a labelled consumer lands.
    """
    if not isinstance(x, np.ndarray) and hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("'x' must be a matrix or a vector")
    return arr, None


def _finish_scaled(dist, count, nc):
    """Shared tail for euclidean/manhattan/canberra/minkowski:
    ``count==0 -> NA``; ``count!=nc -> dist /= count/nc`` (the C up-scaling for
    columns dropped to NA).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = dist / (count / nc)
        out = np.where(count != nc, scaled, dist)
        out = np.where(count != 0, out, np.nan)
    return out


# --------------------------------------------------------------------------- #
# the six metric kernels (distance.c), pair-vectorized, column-sequential
# --------------------------------------------------------------------------- #
def _cdist(x, mi, p):
    """Packed lower-triangle distance vector for method index ``mi`` (0-based
    into :data:`_METHODS`). Mirrors ``R_distance`` / the per-metric C functions.
    """
    n, nc = x.shape
    rows, cols = _lower_tri_ij(n)
    m = rows.size
    if m == 0:
        return np.empty(0, dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if mi == 0:  # R_euclidean
            dist = np.zeros(m)
            count = np.zeros(m, dtype=np.int64)
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                dev = a - b
                use = ok & ~np.isnan(dev)
                dist += np.where(use, dev * dev, 0.0)
                count += use
            return np.sqrt(_finish_scaled(dist, count, nc))

        if mi == 1:  # R_maximum
            dist = np.full(m, -_DBL_MAX)
            count = np.zeros(m, dtype=np.int64)
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                dev = np.abs(a - b)
                use = ok & ~np.isnan(dev)
                dist = np.where(use, np.maximum(dist, dev), dist)
                count += use
            return np.where(count != 0, dist, np.nan)

        if mi == 2:  # R_manhattan
            dist = np.zeros(m)
            count = np.zeros(m, dtype=np.int64)
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                dev = np.abs(a - b)
                use = ok & ~np.isnan(dev)
                dist += np.where(use, dev, 0.0)
                count += use
            return _finish_scaled(dist, count, nc)

        if mi == 3:  # R_canberra
            dist = np.zeros(m)
            count = np.zeros(m, dtype=np.int64)
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                s = np.abs(a) + np.abs(b)
                diff = np.abs(a - b)
                outer = ok & ((s > _DBL_MIN) | (diff > _DBL_MIN))
                dev = diff / s
                # second clause: Inf/Inf with diff==sum is the limit x->oo, dev:=1
                special = (~np.isfinite(diff)) & (diff == s)
                accept = outer & (~np.isnan(dev) | special)
                dev_used = np.where(np.isnan(dev), 1.0, dev)
                dist += np.where(accept, dev_used, 0.0)
                count += accept
            return _finish_scaled(dist, count, nc)

        if mi == 4:  # R_dist_binary
            dist = np.zeros(m, dtype=np.int64)
            count = np.zeros(m, dtype=np.int64)
            total = np.zeros(m, dtype=np.int64)
            warned = False
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                nn = ~(np.isnan(a) | np.isnan(b))  # both_non_NA
                fin = np.isfinite(a) & np.isfinite(b)  # both_FINITE
                if np.any(nn & ~fin):
                    warned = True
                valid = nn & fin
                either = (a != 0.0) | (b != 0.0)
                both = (a != 0.0) & (b != 0.0)
                cnt = valid & either
                count += cnt
                dist += (cnt & ~both).astype(np.int64)
                total += valid
            if warned:
                warnings.warn("treating non-finite values as NA", stacklevel=2)
            ratio = dist / count
            out = np.where(count != 0, ratio, 0.0)
            return np.where(total != 0, out, np.nan)

        if mi == 5:  # R_minkowski
            dist = np.zeros(m)
            count = np.zeros(m, dtype=np.int64)
            for j in range(nc):
                a, b = x[rows, j], x[cols, j]
                ok = ~(np.isnan(a) | np.isnan(b))
                dev = a - b
                use = ok & ~np.isnan(dev)
                dist += np.where(use, np.power(np.abs(dev), p), 0.0)
                count += use
            return np.power(_finish_scaled(dist, count, nc), 1.0 / p)

    raise ValueError("distance(): invalid distance")  # pragma: no cover


# --------------------------------------------------------------------------- #
# the Dist object
# --------------------------------------------------------------------------- #
class Dist:
    """R's ``"dist"`` object — a packed lower-triangle distance vector + attrs.

    ``data`` is the column-major strict-lower-triangle vector (length
    ``Size*(Size-1)/2``); the object is vector-like (``len``, iteration,
    ``np.asarray``) so ``as.vector``/``format`` style callers see the raw values,
    exactly as in R.
    """

    __slots__ = ("data", "Size", "Labels", "Diag", "Upper", "method", "p", "call")

    def __init__(self, data, Size, Labels=None, Diag=False, Upper=False,
                 method=None, p=None, call=None):
        self.data = np.ascontiguousarray(data, dtype=float)
        self.Size = int(Size)
        self.Labels = list(Labels) if Labels is not None else None
        self.Diag = bool(Diag)
        self.Upper = bool(Upper)
        self.method = method
        self.p = p
        self.call = call

    def __array__(self, dtype=None, copy=None):
        if copy:
            return self.data.astype(dtype) if dtype is not None else self.data.copy()
        return self.data.astype(dtype) if dtype is not None else self.data

    def __len__(self):
        return int(self.data.size)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return print_dist(self, _return=True)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def dist(x, method="euclidean", diag=False, upper=False, p=2):
    """R ``stats::dist(x, method, diag, upper, p)`` — pairwise distances.

    Returns a :class:`Dist` (packed lower triangle). ``method`` is partial-matched
    against ``euclidean``/``maximum``/``manhattan``/``canberra``/``binary``/
    ``minkowski`` (with R's ``euclidian`` misspelling alias). NA-aware: columns
    where either value is NA are skipped and the result up-scaled, per R.
    """
    if _pmatch(method, ("euclidian",)) is not None:
        method = "euclidean"
    mi = _pmatch(method, _METHODS)
    if mi is None:
        raise ValueError("invalid distance method")

    arr, labels = _as_matrix(x)
    n = arr.shape[0]
    if mi == 5 and (not np.isfinite(p) or p <= 0):
        raise ValueError("distance(): invalid p")

    if _rs_cdist is not None:  # Rust accelerator (step 10); pure-Python is the spec
        data = _rs_cdist(np.ascontiguousarray(arr, dtype=float), mi, float(p))
    else:
        data = _cdist(arr, mi, float(p))

    return Dist(data, Size=n, Labels=labels, Diag=diag, Upper=upper,
                method=_METHODS[mi], p=(float(p) if mi == 5 else None))


_MISSING = object()


def as_dist(m, diag=_MISSING, upper=_MISSING):
    """R ``stats::as.dist(m, diag, upper)`` — coerce a square matrix to a
    :class:`Dist` (its strict lower triangle, column-major). If ``m`` is already
    a :class:`Dist` it is returned (with ``Diag``/``Upper`` updated when given).
    """
    dg = False if diag is _MISSING else bool(diag)
    up = False if upper is _MISSING else bool(upper)

    if isinstance(m, Dist):
        out = Dist(m.data.copy(), m.Size, m.Labels, m.Diag, m.Upper, m.method, m.p)
        if diag is not _MISSING:
            out.Diag = dg
        if upper is not _MISSING:
            out.Upper = up
        return out

    if not isinstance(m, np.ndarray) and hasattr(m, "to_numpy"):
        m = m.to_numpy()
    mat = np.asarray(m, dtype=float)
    if mat.ndim != 2:
        mat = np.atleast_2d(mat)
    p = mat.shape[0]
    if mat.shape[1] != p:
        warnings.warn("non-square matrix", stacklevel=2)
    rows, cols = _lower_tri_ij(p)
    data = mat[rows, cols] if p > 1 else np.empty(0, dtype=float)
    return Dist(data, Size=p, Labels=None, Diag=dg, Upper=up)


def as_matrix_dist(x):
    """R ``as.matrix(<dist>)`` — expand a :class:`Dist` to a full ``n x n``
    symmetric matrix with a zero diagonal."""
    n = x.Size
    mat = np.zeros((n, n), dtype=float)
    if n > 1:
        rows, cols = _lower_tri_ij(n)
        mat[rows, cols] = x.data
        mat[cols, rows] = x.data
    return mat


def _double_centre(a):
    """Port of ``C_DoubleCentre`` (``cluster_dblcen.c``): centre the rows, then
    the columns, of a square matrix. Vectorized (mean over ``n``); cmdscale's
    downstream eigendecomposition is LAPACK-bound, so its parity is tolerance,
    not 0-ulp, and the exact summation order here does not matter."""
    a = np.array(a, dtype=float)  # copy (C mutates in place; we don't)
    a -= a.mean(axis=1, keepdims=True)  # row centring: sum over columns / n
    a -= a.mean(axis=0, keepdims=True)  # then column centring
    return a


def cmdscale(d, k=2, eig=False, add=False, x_ret=False, list_=None):
    """R ``stats::cmdscale(d, k, eig, add, x.ret)`` — classical (metric) MDS.

    Mirrors ``cmdscale.R:19``: square the distances, double-centre
    (``B = -½·J·D²·J``), eigen-decompose ``-B/2`` (``np.linalg.eigh``, reversed to
    R's descending order), and scale the top-``k`` positive eigenvectors by
    ``sqrt(eig)``. ``add=True`` solves the additive-constant problem via the
    ``2n×2n`` block eigenproblem.

    Returns the ``points`` array, or (when ``list_``/``eig``/``add``/``x_ret``)
    a dict ``{points, eig, x, ac, GOF}`` like R's list. Parity is LAPACK
    tolerance and **eigenvector signs are arbitrary** (compare up to per-column
    sign); degenerate eigenvalues additionally admit rotation within the
    eigenspace.
    """
    if list_ is None:
        list_ = eig or add or x_ret

    if isinstance(d, Dist):
        if np.any(np.isnan(d.data)):
            raise ValueError("NA values not allowed in 'd'")
        n = d.Size
        rows, cols = _lower_tri_ij(n)
        x = np.zeros((n, n))
        x[rows, cols] = d.data ** 2
        x = x + x.T
        dfull = None
        if add:
            d0 = np.zeros((n, n))
            d0[rows, cols] = d.data
            dfull = d0 + d0.T
    else:
        mat = np.asarray(d, dtype=float)
        if np.any(np.isnan(mat)):
            raise ValueError("NA values not allowed in 'd'")
        dfull = mat if add else None
        x = mat ** 2
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError("distances must be result of 'dist' or a square matrix")
        n = x.shape[0]

    n = int(n)
    if n > 46340:
        raise ValueError("invalid value of 'n'")
    k = int(k)
    if k > n - 1 or k < 1:
        raise ValueError("'k' must be in {1, 2, ..  n - 1}")

    x = _double_centre(x)

    add_c = 0.0
    if add:
        # additive constant = largest eigenvalue of the 2n x 2n block matrix Z
        z = np.zeros((2 * n, 2 * n))
        z[:n, n:] = -x
        z[n:, :n] = -np.eye(n)
        z[n:, n:] = _double_centre(2.0 * dfull)
        add_c = float(np.max(np.linalg.eigvals(z).real))
        x2 = np.zeros((n, n))
        non_diag = ~np.eye(n, dtype=bool)
        x2[non_diag] = (dfull[non_diag] + add_c) ** 2
        x = _double_centre(x2)

    w, v = np.linalg.eigh(-x / 2.0)
    evalues = w[::-1]        # R eigen(symmetric=TRUE): descending eigenvalues
    evectors = v[:, ::-1]
    ev = evalues[:k]
    evec = evectors[:, :k]
    k1 = int(np.sum(ev > 0))
    if k1 < k:
        warnings.warn(
            f"only {k1} of the first {k} eigenvalues are > 0", stacklevel=2)
        evec = evec[:, ev > 0]
        ev = ev[ev > 0]
    points = evec * np.sqrt(ev)

    if list_:
        return {
            "points": points,
            "eig": evalues if eig else None,
            "x": x if x_ret else None,
            "ac": add_c if add else 0,
            "GOF": np.array([
                ev.sum() / np.abs(evalues).sum(),
                ev.sum() / np.maximum(evalues, 0).sum(),
            ]),
        }
    return points


def mahalanobis(x, center, cov, inverted=False):
    """R ``stats::mahalanobis(x, center, cov, inverted)`` — squared Mahalanobis
    distance of each row of ``x`` from ``center`` under covariance ``cov``.

    Mirrors ``mahalanobis.R:31``: subtract ``center`` row-wise (skipped when
    ``center is False``, R's ``isFALSE(center)``), invert ``cov`` unless
    ``inverted`` says it is already the precision matrix, then
    ``rowSums((x %*% cov) * x)``. Returns a 1-D ndarray of length ``nrow(x)``.

    Parity is at LAPACK/BLAS tolerance (``solve``/matmul), not 0-ulp — R and
    numpy may use different BLAS builds.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)  # is.vector(x): matrix(x, ncol=length(x))
    else:
        x = np.atleast_2d(x)
    if center is not False:  # R: if(!isFALSE(center)) x <- sweep(x, 2L, center)
        x = x - np.asarray(center, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not inverted:
        cov = np.linalg.inv(cov)  # R: cov <- solve(cov)
    return np.sum((x @ cov) * x, axis=1)


def labels_dist(x):
    """R ``labels(<dist>)`` — the ``Labels`` attribute (or ``None``)."""
    return x.Labels


def format_dist(x, **kwargs):
    """R ``format(<dist>)`` — ``format(as.vector(x))``; the formatted packed
    vector as a string ndarray."""
    return np.array([_format_num(v) for v in x.data], dtype=object)


def _format_num(v):
    if np.isnan(v):
        return "NA"
    return np.format_float_positional(v, trim="-", unique=True)


def print_dist(x, diag=None, upper=None, _return=False):
    """R ``print(<dist>)`` — the lower-triangular text layout.

    The numeric layout (which cells are shown, the labels) matches R; exact
    column spacing is not yet pinned byte-for-byte to R's ``print`` (a
    follow-up — the numeric verbs are the parity gate).
    """
    n = x.Size
    if len(x) == 0:
        s = "dist(0)\n"
        if _return:
            return s
        print(s, end="")
        return None

    if diag is None:
        diag = x.Diag
    if upper is None:
        upper = x.Upper

    mat = as_matrix_dist(x)
    labels = x.Labels if x.Labels is not None else [str(i + 1) for i in range(n)]
    cells = [[_format_num(mat[i, j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if not upper and i < j:
                cells[i][j] = ""
            if not diag and i == j:
                cells[i][j] = ""

    if diag or upper:
        ri, ci = range(n), range(n)
    else:
        ri, ci = range(1, n), range(0, n - 1)  # drop empty first row / last col

    widths = [max(len(labels[j]), max((len(cells[i][j]) for i in ri), default=0))
              for j in ci]
    rowlab_w = max(len(labels[i]) for i in ri)

    lines = [" " * rowlab_w + " " + " ".join(
        labels[j].rjust(widths[k]) for k, j in enumerate(ci))]
    for i in ri:
        lines.append(labels[i].rjust(rowlab_w) + " " + " ".join(
            cells[i][j].rjust(widths[k]) for k, j in enumerate(ci)))
    s = "\n".join(lines) + "\n"
    if _return:
        return s
    print(s, end="")
    return None
