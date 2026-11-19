"""mgcv's small C helpers (mat.c, gdi.c) as R's arm64 mgcv.so computes them:
the Python twin of the extension's ``mgcv_c.rs``, for the ports of gdi1,
gdi2 and pls_fit1 in :mod:`hea.models._gdi`.

Matrices are flat, column-major float64 arrays, and a routine writes its
result into the array it is handed, as the C writes through its pointer
arguments; the C's ``x + k`` is the view ``x[k:]``. The BLAS calls go to the
routines the extension links, with the C's own arguments, leading dimensions
and operand addresses included (Accelerate's ``ddot``, for one, rounds by
where its operands start): where that is Accelerate
(``hea._dispatch.ACCELERATE``), its Fortran interface (``dgemm_`` and the
rest, which R's vecLib BLAS forwards to; the interface SciPy's
``cython_blas`` reaches rounds some products differently), otherwise the
BLAS SciPy is built against. The LAPACK ones go to R's LAPACK through
:mod:`hea._rlinalg`.

clang fuses every ``x += a*b`` and ``x -= a*b`` (``_rfma``: one rounding on
arm64, two elsewhere). A reduction over contiguous memory it vectorizes, 8
products a pass rounded and added in order and the remainder fused
(:func:`dot_s8`); one over a strided operand it keeps scalar and fused.
"""

from __future__ import annotations

import ctypes

import numpy as np
from scipy.linalg import cython_blas

from .._dispatch import ACCELERATE
from .._rfma import _rfma, _rfma_vec
from .._rlinalg import mgcv_svd_full as _svd_full
from .._rlinalg import r_dgeqp3, r_dormqr_left

_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)
_C = ctypes.c_char_p


def _blas_fn(name: str, *argtypes):
    """The BLAS routine ``name`` as the extension calls it, as a ctypes
    function taking Fortran's by-reference arguments: Accelerate's
    ``name_`` where the BLAS is Accelerate, else the pointer in SciPy's
    ``cython_blas`` capsule."""
    if ACCELERATE:
        lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        fn = getattr(lib, name + "_")
        fn.restype = None
        fn.argtypes = list(argtypes)
        return fn
    cap = cython_blas.__pyx_capi__[name]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    get_ptr.restype = ctypes.c_void_p
    get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.CFUNCTYPE(None, *argtypes)(get_ptr(cap, get_name(cap)))


_dgemm = _blas_fn("dgemm", _C, _C, _I, _I, _I, _D, _D, _I, _D, _I, _D, _D, _I)
_dgemv = _blas_fn("dgemv", _C, _I, _I, _D, _D, _I, _D, _I, _D, _D, _I)
_dsyrk = _blas_fn("dsyrk", _C, _C, _I, _I, _D, _D, _I, _D, _D, _I)
_dtrsm = _blas_fn("dtrsm", _C, _C, _C, _C, _I, _I, _D, _D, _I, _D, _I)


def _i(v):
    return ctypes.byref(ctypes.c_int(int(v)))


def _d(v):
    return ctypes.byref(ctypes.c_double(float(v)))


def _p(a: np.ndarray):
    """The address of ``a``'s first element, for the BLAS. ``a`` must be a
    contiguous float64 array (a flat buffer or a view into one)."""
    if a.dtype != np.float64 or not a.flags.c_contiguous:
        raise TypeError("the BLAS takes contiguous float64 buffers")
    return a.ctypes.data_as(_D)


def _addr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


def dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) -> None:
    """BLAS ``dgemm`` with its Fortran arguments, the arrays by address."""
    _dgemm(
        ta,
        tb,
        _i(m),
        _i(n),
        _i(k),
        _d(alpha),
        _p(a),
        _i(lda),
        _p(b),
        _i(ldb),
        _d(beta),
        _p(c),
        _i(ldc),
    )


def dgemv(ta, m, n, alpha, a, lda, x, incx, beta, y, incy) -> None:
    """BLAS ``dgemv`` with its Fortran arguments, the arrays by address."""
    _dgemv(
        ta,
        _i(m),
        _i(n),
        _d(alpha),
        _p(a),
        _i(lda),
        _p(x),
        _i(incx),
        _d(beta),
        _p(y),
        _i(incy),
    )


def dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc) -> None:
    """BLAS ``dsyrk`` with its Fortran arguments, the arrays by address."""
    _dsyrk(
        uplo, trans, _i(n), _i(k), _d(alpha), _p(a), _i(lda), _d(beta), _p(c), _i(ldc)
    )


def dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb) -> None:
    """BLAS ``dtrsm`` with its Fortran arguments, the arrays by address."""
    _dtrsm(
        side,
        uplo,
        transa,
        diag,
        _i(m),
        _i(n),
        _d(alpha),
        _p(a),
        _i(lda),
        _p(b),
        _i(ldb),
    )


def mat(x: np.ndarray, r: int, c: int) -> np.ndarray:
    """The ``r``-by-``c`` matrix at the start of the flat ``x``, as a
    column-major view."""
    return x[: r * c].reshape((r, c), order="F")


def _inorder(v: np.ndarray, s: float = 0.0) -> float:
    """``for (k) s += v[k];``, rounded at each step."""
    if v.size == 0:
        return float(s)
    return float(np.cumsum(np.concatenate(([s], v)))[-1])


def dot_s8_acc(s: float, a: np.ndarray, b: np.ndarray) -> float:
    """``s + sum a[i] b[i]`` accumulated in a local as clang vectorizes it:
    for ``n >= 8`` the first ``8 floor(n/8)`` products rounded and added in
    order, the rest fused."""
    n = len(a)
    nv = n // 8 * 8 if n >= 8 else 0
    s = _inorder(np.asarray(a[:nv]) * np.asarray(b[:nv]), s)
    for x, y in zip(np.asarray(a[nv:n]).tolist(), np.asarray(b[nv:n]).tolist()):
        s = _rfma(x, y, s)
    return s


def dot_s8(a: np.ndarray, b: np.ndarray) -> float:
    """``sum a[i] b[i]`` as clang vectorizes it (:func:`dot_s8_acc` from
    zero)."""
    return dot_s8_acc(0.0, a, b)


def dot_s8_from(a: np.ndarray, b: np.ndarray, min_n: int) -> float:
    """``sum a[i] b[i]`` accumulated through memory clang cannot prove
    distinct from ``a`` and ``b``: below ``min_n`` terms every product fused
    into the sum; from ``min_n`` on (after its run-time overlap check
    passes) as :func:`dot_s8`."""
    if len(a) >= min_n:
        return dot_s8(a, b)
    s = 0.0
    for x, y in zip(np.asarray(a).tolist(), np.asarray(b).tolist()):
        s = _rfma(x, y, s)
    return s


def sumsq_s4(a: np.ndarray) -> float:
    """``sum a[i]^2`` as clang unrolls it: for ``n > 3`` the first ``4
    floor(n/4)`` squares rounded and added in order, the rest fused."""
    a = np.asarray(a)
    n = a.size
    nv = n // 4 * 4 if n > 3 else 0
    s = _inorder(a[:nv] * a[:nv])
    for x in a[nv:].tolist():
        s = _rfma(x, x, s)
    return s


def get_xtx(xtx: np.ndarray, x: np.ndarray, r: int, c: int) -> None:
    """mgcv's ``getXtX``: ``X'X`` for the ``r``-by-``c`` ``x`` by
    ``dsyrk('L', 'T')``, the upper triangle copied from the lower."""
    dsyrk(b"L", b"T", c, r, 1.0, x, r, 0.0, xtx, c)
    m = mat(xtx, c, c)
    iu = np.triu_indices(c, 1)
    m[iu] = m.T[iu]


def get_xxt(xxt: np.ndarray, x: np.ndarray, r: int, c: int) -> None:
    """mgcv's ``getXXt``: ``X X'`` for the ``r``-by-``c`` ``x`` by
    ``dsyrk('L', 'N')``, the upper triangle copied from the lower."""
    dsyrk(b"L", b"N", r, c, 1.0, x, r, 0.0, xxt, r)
    m = mat(xxt, r, r)
    iu = np.triu_indices(r, 1)
    m[iu] = m.T[iu]


def mgcv_mmult(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bt: bool,
    ct: bool,
    r: int,
    cc: int,
    n: int,
) -> None:
    """mgcv's ``mgcv_mmult``: ``A = op(B) op(C)`` (``r``-by-``cc``, common
    dimension ``n``) by one ``dgemm``, or by ``getXtX``/``getXXt`` when ``B``
    and ``C`` are the same matrix (the same address) and the product is its
    cross product."""
    if r <= 0 or cc <= 0 or n <= 0:
        return
    if _addr(b) == _addr(c):
        if bt and not ct and r == cc:
            get_xtx(a, b, n, r)
            return
        if ct and not bt and r == cc:
            get_xxt(a, b, cc, n)
            return
    ta, lda = (b"T", n) if bt else (b"N", r)
    tb, ldb = (b"T", cc) if ct else (b"N", n)
    dgemm(ta, tb, r, cc, n, 1.0, b, lda, c, ldb, 0.0, a, r)


def mgcv_svd_full(x: np.ndarray, vt: np.ndarray, d: np.ndarray, r: int, c: int) -> None:
    """mgcv's ``mgcv_svd_full``: ``dgesdd`` with ``JOBZ = 'O'`` on the
    ``r``-by-``c`` ``x`` (``r >= c``): ``U`` over ``x``, ``V'`` in ``vt``,
    the singular values in ``d``."""
    u, dd, v = _svd_full(mat(x, r, c).copy(order="F"))
    x[: r * c] = np.asarray(u).ravel(order="F")
    vt[: c * c] = np.asarray(v).ravel(order="F")
    d[:c] = dd


def mgcv_qr(x: np.ndarray, r: int, c: int, pivot: np.ndarray, tau: np.ndarray) -> None:
    """mgcv's ``mgcv_qr``: ``dgeqp3`` on the ``r``-by-``c`` ``x`` with the
    workspace its query returns, the pivots then made 0-based. ``pivot``
    goes in as ``JPVT``: an entry non-zero on entry fixes that column
    (gdiPK's second call passes the first call's pivots)."""
    qr, tq, jpvt = r_dgeqp3(mat(x, r, c), pivot[:c])
    x[: r * c] = np.asarray(qr).ravel(order="F")
    tau[: tq.size] = tq
    pivot[:c] = np.asarray(jpvt) - 1


def mgcv_qrqy_left(
    b: np.ndarray, a: np.ndarray, tau: np.ndarray, r: int, c: int, k: int, tp: bool
) -> None:
    """mgcv's ``mgcv_qrqy`` from the left: ``b`` (``r``-by-``c``) times the
    ``Q`` of the ``k`` reflectors in ``a`` and ``tau`` (``tp``: its
    transpose), by ``dormqr`` with the workspace its query returns."""
    if r == 0 or c == 0:
        return
    out = r_dormqr_left(mat(a, r, k), tau[:k], mat(b, r, c), tp)
    b[: r * c] = np.asarray(out).ravel(order="F")


def mgcv_pqrqy(
    b: np.ndarray, a: np.ndarray, tau: np.ndarray, r: int, c: int, cb: int, tp: bool
) -> None:
    """mgcv's ``mgcv_pqrqy`` with one thread: ``b``, physically
    ``r``-by-``cb``, holds a ``c``-by-``cb`` matrix on entry when ``tp`` is
    false (spread to ``r`` rows first) and on exit when it is true (packed
    back); the ``Q`` of ``mgcv_pqr`` applied from the left."""
    if not tp:
        # each column moved down into place from the end, its source cleared
        packed = mat(b, c, cb).copy(order="F")
        b[: c * cb] = 0.0
        mat(b, r, cb)[:c, :] = packed
    mgcv_qrqy_left(b, a, tau, r, cb, c, tp)
    if tp:
        b[: c * cb] = mat(b, r, cb)[:c, :].ravel(order="F")


def get_rpqr(rm: np.ndarray, x: np.ndarray, r: int, c: int, rr: int) -> None:
    """mgcv's ``getRpqr``: the ``c``-by-``c`` upper triangle of the
    ``r``-by-``c`` QR in ``x`` into the ``rr``-row ``rm``, zeros below the
    diagonal."""
    rows = min(c, rr)
    mat(rm, rr, c)[:rows, :] = np.triu(mat(x, r, c)[:rows, :])


def mgcv_backsolve(
    rm: np.ndarray,
    r: int,
    c: int,
    b: np.ndarray,
    cm: np.ndarray,
    bc: int,
    right: bool,
) -> None:
    """mgcv's ``mgcv_backsolve``: ``C = R^{-1} B`` (``right``: ``C = B
    R^{-1}``) for the ``c``-by-``c`` upper triangle of the ``r``-row ``R``,
    ``B`` ``c``-by-``bc`` (``bc``-by-``c``), by ``dtrsm``."""
    side, m, n = (b"R", bc, c) if right else (b"L", c, bc)
    cm[: bc * c] = b[: bc * c]
    dtrsm(side, b"U", b"N", b"N", m, n, 1.0, rm, r, cm, m)


def mgcv_forwardsolve(
    rm: np.ndarray,
    r: int,
    c: int,
    b: np.ndarray,
    cm: np.ndarray,
    bc: int,
    right: bool,
) -> None:
    """mgcv's ``mgcv_forwardsolve``: ``C = R^{-T} B`` (``right``: ``C = B
    R^{-T}``), as :func:`mgcv_backsolve` with ``R`` transposed."""
    side, m, n = (b"R", bc, c) if right else (b"L", c, bc)
    cm[: bc * c] = b[: bc * c]
    dtrsm(side, b"U", b"T", b"N", m, n, 1.0, rm, r, cm, m)


def get_xtwx(xtwx: np.ndarray, x: np.ndarray, w: np.ndarray, r: int, c: int) -> None:
    """mgcv's ``getXtWX``: ``X'WX`` for the ``r``-by-``c`` ``x`` and ``W =
    diag(w)``, a column at a time (``dgemv('T')`` of ``X[, 1:i]`` on ``w *
    X[, i]`` into the start of ``xtwx``, then moved to its column), the
    upper triangle copied to the lower."""
    work = np.zeros(max(r, 1))
    xx = 0.0
    for i in range(c):
        work[:r] = x[r * i : r * i + r] * w[:r]
        dgemv(b"T", r, i + 1, 1.0, x, r, work, 1, 0.0, xtwx, 1)
        if i == 0:
            xx = xtwx[0]
        else:
            xtwx[i * c : i * c + i + 1] = xtwx[: i + 1]
    if r * c > 0:
        xtwx[0] = xx
    m = mat(xtwx, c, c)
    il = np.tril_indices(c, -1)
    m[il] = m.T[il]


def get_xtmx(xtmx: np.ndarray, x: np.ndarray, m: np.ndarray, r: int, c: int) -> None:
    """mgcv's ``getXtMX``: ``X'MX`` for the ``r``-by-``c`` ``x`` and
    symmetric ``r``-by-``r`` ``m``: ``M X[, i]`` a fused column sweep, each
    entry an 8-wide dot."""
    for i in range(c):
        xi = x[r * i : r * i + r]
        work = xi[0] * m[:r]
        for j in range(1, r):
            work = _rfma_vec(xi[j], m[r * j : r * j + r], work)
        for j in range(i + 1):
            xx = dot_s8(work, x[r * j : r * j + r])
            xtmx[i * c + j] = xx
            xtmx[j * c + i] = xx


def pivoter(
    x: np.ndarray, r: int, c: int, pivot: np.ndarray, col: bool, reverse: bool
) -> None:
    """gdi.c's ``pivoter``: pivot (``reverse``: unpivot) the rows, or with
    ``col`` the columns, of the ``r``-by-``c`` ``x`` by the 0-based
    ``pivot``; pivoted element ``i`` is unpivoted element ``pivot[i]``."""
    X = mat(x, r, c)
    p = np.asarray(pivot[: c if col else r], dtype=np.int64)
    if col:
        if reverse:
            Y = np.empty_like(X)
            Y[:, p] = X
        else:
            Y = X[:, p]
    elif reverse:
        Y = np.empty_like(X)
        Y[p, :] = X
    else:
        Y = X[p, :]
    x[: r * c] = Y.ravel(order="F")


def drop_cols(x: np.ndarray, r: int, c: int, drop) -> None:
    """gdi.c's ``drop_cols``: remove the columns ``drop`` (ascending) of the
    ``r``-by-``c`` ``x``, packing the rest to the front."""
    if len(drop) == 0:
        return
    keep = np.setdiff1d(np.arange(c), np.asarray(drop, dtype=np.int64))
    x[: r * keep.size] = mat(x, r, c)[:, keep].ravel(order="F")


def drop_rows(x: np.ndarray, r: int, c: int, drop) -> None:
    """gdi.c's ``drop_rows``: remove the rows ``drop`` (ascending) of the
    ``r``-by-``c`` ``x``, packing it as ``(r - n_drop)``-by-``c``."""
    if len(drop) == 0:
        return
    keep = np.setdiff1d(np.arange(r), np.asarray(drop, dtype=np.int64))
    x[: keep.size * c] = mat(x, r, c)[keep, :].ravel(order="F")


def undrop_rows(x: np.ndarray, r: int, c: int, drop) -> None:
    """gdi.c's ``undrop_rows``: spread the ``(r - n_drop)``-by-``c`` ``x``
    back to ``r``-by-``c`` with zero rows at ``drop`` (ascending)."""
    if len(drop) == 0:
        return
    keep = np.setdiff1d(np.arange(r), np.asarray(drop, dtype=np.int64))
    Y = np.zeros((r, c), order="F")
    Y[keep, :] = mat(x, keep.size, c)
    x[: r * c] = Y.ravel(order="F")


def rinv(out: np.ndarray, rm: np.ndarray, c: int, r: int, ri: int) -> None:
    """gdi.c's ``Rinv``: the inverse of the ``c``-by-``c`` upper triangle of
    the ``r``-row ``rm`` into the top of the ``ri``-row ``out``, by columns
    of back substitution. The inner sum runs along a row of ``rm``, strided,
    so clang keeps it scalar and fused."""
    for i in range(c):
        o = ri * i
        eye = 1.0
        for k in range(i, -1, -1):
            xx = 0.0
            for j in range(k + 1, i + 1):
                xx = _rfma(float(rm[k + j * r]), float(out[o + j]), xx)
            out[o + k] = (eye - xx) / rm[k + k * r]
            eye = 0.0
        out[o + i + 1 : o + c] = 0.0


def rc_prod(y: np.ndarray, z: np.ndarray, x: np.ndarray, xcol: int, n: int) -> None:
    """gdi.c's ``rc_prod``: ``y = diag(z) x`` for the ``n``-row ``x`` of
    ``xcol`` columns."""
    y[: n * xcol] = (np.asarray(z[:n])[:, None] * mat(x, n, xcol)).ravel(order="F")


def frobenius_norm(x: np.ndarray, r: int, c: int) -> float:
    """gdi.c's ``frobenius_norm``: ``sqrt(sum x^2)`` over the ``r``-by-``c``
    ``x``."""
    return float(np.sqrt(sumsq_s4(x[: r * c])))


def diag_abt(d: np.ndarray, a: np.ndarray, b: np.ndarray, r: int, c: int) -> float:
    """gdi.c's ``diagABt``: ``d = diag(A B')`` for ``r``-by-``c`` ``A`` and
    ``B`` (the first column's products rounded, the rest fused in),
    returning its sum."""
    if c == 0:
        return 0.0
    dd = a[:r] * b[:r]
    for j in range(1, c):
        dd = _rfma_vec(a[r * j : r * j + r], b[r * j : r * j + r], dd)
    d[:r] = dd
    return _inorder(d[:r])


def r_cond(rm: np.ndarray, r: int, c: int) -> float:
    """gdi.c's ``R_cond``: the Cline, Moler, Stewart and Wilkinson estimate
    of the condition number of the ``c``-by-``c`` upper triangle of the
    ``r``-row ``rm``."""
    pp = [0.0] * c
    pm = [0.0] * c
    y = [0.0] * c
    p = [0.0] * c
    y_inf = r_inf = 0.0
    for k in range(c - 1, -1, -1):
        rkk = float(rm[k + r * k])
        yp = (1.0 - p[k]) / rkk
        ym = (-1.0 - p[k]) / rkk
        col = rm[r * k : r * k + k].tolist()
        pp_norm = 0.0
        for i in range(k):
            pp[i] = _rfma(col[i], yp, p[i])
            pp_norm += abs(pp[i])
        pm_norm = 0.0
        for i in range(k):
            pm[i] = _rfma(col[i], ym, p[i])
            pm_norm += abs(pm[i])
        if abs(yp) + pp_norm >= abs(ym) + pm_norm:
            y[k] = yp
            p[:k] = pp[:k]
        else:
            y[k] = ym
            p[:k] = pm[:k]
        y_inf = max(y_inf, abs(y[k]))
    for i in range(c):
        kappa = 0.0
        for j in range(i, c):
            kappa += abs(float(rm[i + r * j]))
        r_inf = max(r_inf, kappa)
    return r_inf * y_inf
