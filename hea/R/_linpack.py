"""Bit-exact ports of the BLAS level-1 / LINPACK kernels used by R's
optimizers (``nlm`` via ``src/appl/uncmin.c``, ``optim(method="L-BFGS-B")``
via ``src/appl/lbfgsb.c``).

What "R's BLAS" means here is what R actually *links*, which differs by
platform. The CRAN macOS build symlinks ``libRblas.dylib ->
libRblas.vecLib.dylib``, a forwarder to Apple **Accelerate** — NOT the
reference ``blas.f``; on Linux, R binds a plain-ordered BLAS (Debian/
Ubuntu's reference ``libblas3``, or R's own bundled ``blas.f``) whose
``ddot`` is strictly sequential accumulation at *every* length (the
netlib mod-5 unrolled loop preserves left-to-right order). These
kernels therefore emulate Accelerate's observed behavior only on
darwin/arm64 — where it was probed empirically via ``ctypes`` against
``libRblas.dylib`` — and stay plain sequential elsewhere (the
``_ACCEL_PAIR4`` gate below, the BLAS-ordering analogue of
``_shared._rfma``'s per-arch contraction policy):

* ``_ddot`` — Accelerate is plain sequential accumulation for n ≤ 3 and
  the pair tree ``(s0+s2) + (s1+s3)`` at n = 4 (verified bit-exact over
  adversarial mixed-magnitude probes, alignment-independent). From five
  terms on its kernel rounds by its own reduction and by where its second
  operand starts, so where the BLAS is Accelerate
  (``hea._dispatch.ACCELERATE``) the call goes to Accelerate's ``ddot_``
  itself when both operands are contiguous, as the columns of R's
  column-major arrays are; a strided operand keeps the sequential loop,
  as the extension's ``incx != 1`` path does. The callers lay their
  arrays out as R does (column-major, the L-BFGS-B workspace in one block
  carved at ``setulb``'s offsets) so that each operand starts where R's
  does. On reference-BLAS platforms (Linux CI) the sequential order is
  exact at every length.
* ``_daxpy`` — Accelerate fuses ``y + a*x`` per element (probed: fma
  matches, plain does not); mirrored via ``_rfma`` (fused on arm64,
  plain on x86-64, matching R-as-built per arch).
* ``_dnrm2`` — Accelerate matches ``sqrt(seq-dot)`` for n ≤ 2; n ≥ 3
  unidentified (used only inside secant-update *skip comparisons* on
  R's nlm path, where a 1-ulp norm difference is measure-zero). The
  reference ``dnrm2`` (LAPACK ≥ 3.10's three-accumulator version) also
  reduces to ``sqrt`` of the sequential self-dot for medium-magnitude
  inputs, so the same code is exact on Linux.
* ``_dtrsl`` / ``_dpofa`` — LINPACK sources compiled *into* libR by
  gfortran; their internal ``ddot``/``daxpy`` calls resolve to the same
  Accelerate BLAS (routed through ``_ddot``/``_daxpy`` here), and
  gfortran's default ``-ffp-contract=fast`` fuses ``s = s + t*t`` in
  ``dpofa`` (``_rfma``). R's 2002 ``dpofa`` modification (pivot
  tolerance ``s <= 1e-14*|a_jj|``) is kept.

All routines mutate their numpy arguments in place exactly like the
Fortran; matrix arguments are ``(n, n)``-indexed ``[row, col]`` (numpy
views of a larger array are fine — that reproduces Fortran leading-
dimension submatrix calls).
"""

from __future__ import annotations

import ctypes
import math
import platform
import sys

import numpy as np

from .._dispatch import ACCELERATE
from ._shared import _rfma

_ACCEL_PAIR4 = sys.platform == "darwin" and platform.machine() == "arm64"


def _accelerate_ddot():
    """Accelerate's Fortran ``ddot_``, the routine R's ``ddot`` reaches on
    darwin, when the extension's BLAS is Accelerate; ``None`` otherwise."""
    if not ACCELERATE:
        return None
    lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    fn = lib.ddot_
    fn.restype = ctypes.c_double
    pi = ctypes.POINTER(ctypes.c_int)
    fn.argtypes = [pi, ctypes.c_void_p, pi, ctypes.c_void_p, pi]
    return fn


_ACCEL_DDOT = _accelerate_ddot()


def _ddot(n, dx, dy, ox=0, oy=0):
    """R-linked ``ddot`` over ``n`` entries of ``dx[ox:]``/``dy[oy:]``:
    sequential, except Accelerate's pair tree at n = 4 on darwin/arm64 and,
    where the BLAS is Accelerate, Accelerate's own call from five contiguous
    terms on (see module docstring)."""
    if n == 4 and _ACCEL_PAIR4:
        s0 = float(dx[ox]) * float(dy[oy])
        s1 = float(dx[ox + 1]) * float(dy[oy + 1])
        s2 = float(dx[ox + 2]) * float(dy[oy + 2])
        s3 = float(dx[ox + 3]) * float(dy[oy + 3])
        return (s0 + s2) + (s1 + s3)
    if n >= 5 and _ACCEL_DDOT is not None:
        x = dx[ox : ox + n]
        y = dy[oy : oy + n]
        if (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and x.dtype == np.float64
            and y.dtype == np.float64
            and x.strides == (8,)
            and y.strides == (8,)
        ):
            nn, one = ctypes.c_int(n), ctypes.c_int(1)
            return float(
                _ACCEL_DDOT(
                    ctypes.byref(nn),
                    x.ctypes.data,
                    ctypes.byref(one),
                    y.ctypes.data,
                    ctypes.byref(one),
                )
            )
    dtemp = 0.0
    for i in range(n):
        dtemp += float(dx[ox + i]) * float(dy[oy + i])
    return dtemp


def _dnrm2(n, x, ox=0):
    """R-linked (Accelerate) ``dnrm2``: ``sqrt`` of the ``_ddot``
    self-product (bit-exact vs Accelerate for n ≤ 2; see module
    docstring for the n ≥ 3 boundary)."""
    if n <= 0:
        return 0.0
    return math.sqrt(_ddot(n, x, x, ox=ox, oy=ox))


def _dscal(n, da, dx, ox=0):
    """``dscal``: ``dx[ox:ox+n] *= da`` elementwise, in place (a single
    rounding per element — no ordering freedom)."""
    for i in range(n):
        dx[ox + i] = float(da) * float(dx[ox + i])


def _daxpy(n, da, dx, dy, ox=0, oy=0):
    """R-linked (Accelerate) ``daxpy``: ``dy[i] = fma(da, dx[i],
    dy[i])`` per element (probed: Accelerate fuses; ``_rfma`` keeps the
    per-arch R-parity policy)."""
    if n <= 0 or da == 0.0:
        return
    for i in range(n):
        dy[oy + i] = _rfma(float(da), float(dx[ox + i]), float(dy[oy + i]))


def _dcopy(n, dx, dy, ox=0, oy=0):
    """``dcopy``: ``dy[oy:oy+n] = dx[ox:ox+n]`` (forward loop, matching
    the Fortran for overlapping shifts)."""
    for i in range(n):
        dy[oy + i] = float(dx[ox + i])


def _dtrsl(t, n, b, job):
    """LINPACK ``dtrsl`` (src/appl/dtrsl.f): solve ``t*x=b`` (or
    ``t'x=b``) for triangular ``t``, overwriting ``b`` with the solution.
    """
    for info in range(1, n + 1):
        if float(t[info - 1, info - 1]) == 0.0:
            return info
    kase = 1
    if job % 10 != 0:
        kase = 2
    if (job % 100) // 10 != 0:
        kase += 2
    if kase == 1:
        b[0] = float(b[0]) / float(t[0, 0])
        for j in range(2, n + 1):
            temp = -float(b[j - 2])
            _daxpy(n - j + 1, temp, t[:, j - 2], b, ox=j - 1, oy=j - 1)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    elif kase == 2:
        b[n - 1] = float(b[n - 1]) / float(t[n - 1, n - 1])
        for jj in range(2, n + 1):
            j = n - jj + 1
            temp = -float(b[j])
            _daxpy(j, temp, t[:, j], b)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    elif kase == 3:
        b[n - 1] = float(b[n - 1]) / float(t[n - 1, n - 1])
        for jj in range(2, n + 1):
            j = n - jj + 1
            b[j - 1] = float(b[j - 1]) - _ddot(jj - 1, t[:, j - 1], b, ox=j, oy=j)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    else:
        b[0] = float(b[0]) / float(t[0, 0])
        for j in range(2, n + 1):
            b[j - 1] = float(b[j - 1]) - _ddot(j - 1, t[:, j - 1], b)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    return 0


def _dpofa(a, n):
    """LINPACK ``dpofa`` (src/appl/dpofa.f, with R's 2002 positive-
    definiteness tolerance): factor the symmetric matrix whose diagonal
    and upper triangle are in ``a`` as ``a = r'r``, overwriting the upper
    triangle with ``r``. Returns ``info`` (0 normal; k if the order-k
    leading minor fails ``s > 1e-14*|a_kk|``). ``s = s + t*t`` is
    gfortran-contracted in R's build → ``_rfma``."""
    eps = 1e-14
    for j in range(1, n + 1):
        s = 0.0
        for k in range(1, j):
            t = float(a[k - 1, j - 1]) - _ddot(k - 1, a[:, k - 1], a[:, j - 1])
            t /= float(a[k - 1, k - 1])
            a[k - 1, j - 1] = t
            s = _rfma(t, t, s)
        s = float(a[j - 1, j - 1]) - s
        if s <= eps * abs(float(a[j - 1, j - 1])):
            return j
        a[j - 1, j - 1] = math.sqrt(s)
    return 0


def _dasum(n, dx, ox=0):
    """``dasum``: the sum of ``|dx[ox:ox+n]|`` accumulated left to right,
    which is the reference BLAS order (its unrolled loop adds in sequence).
    Accelerate's reduction order for this kernel has not been probed, so on
    darwin/arm64 the result can differ from R's in the last bit."""
    s = 0.0
    for i in range(n):
        s += abs(float(dx[ox + i]))
    return s


def _dtrco(t, n, job):
    """LINPACK ``dtrco`` (src/appl/dtrco.f): an estimate of the reciprocal
    1-norm condition number of the triangular ``t`` (``job = 0`` lower,
    otherwise upper), returned as ``rcond``. It solves ``t'y = e`` with the
    signs of ``e`` chosen for local growth, then ``t z = y``, rescaling to
    avoid overflow, and reports ``‖z‖₁ / (‖t‖₁ ‖y‖₁)``. The products folded
    into a sum are fused through ``_rfma``, as gfortran contracts them."""
    lower = job == 0
    tnorm = 0.0
    for j in range(1, n + 1):
        ln = n + 1 - j if lower else j
        i1 = j if lower else 1
        tnorm = max(tnorm, _dasum(ln, t[:, j - 1], ox=i1 - 1))

    # solve trans(t)*y = e
    ek = 1.0
    z = [0.0] * n
    for kk in range(1, n + 1):
        k = n + 1 - kk if lower else kk
        km = k - 1
        tkk = float(t[km, km])
        if z[km] != 0.0:
            ek = math.copysign(ek, -z[km])
        if abs(ek - z[km]) > abs(tkk):
            s = abs(tkk) / abs(ek - z[km])
            _dscal(n, s, z)
            ek = s * ek
        wk = ek - z[km]
        wkm = -ek - z[km]
        s = abs(wk)
        sm = abs(wkm)
        if tkk != 0.0:
            wk = wk / tkk
            wkm = wkm / tkk
        else:
            wk = 1.0
            wkm = 1.0
        if kk != n:
            j1 = 1 if lower else k + 1
            j2 = k - 1 if lower else n
            for j in range(j1, j2 + 1):
                tkj = float(t[km, j - 1])
                sm = sm + abs(_rfma(wkm, tkj, z[j - 1]))
                z[j - 1] = _rfma(wk, tkj, z[j - 1])
                s = s + abs(z[j - 1])
            if s < sm:
                w = wkm - wk
                wk = wkm
                for j in range(j1, j2 + 1):
                    z[j - 1] = _rfma(w, float(t[km, j - 1]), z[j - 1])
        z[km] = wk
    s = 1.0 / _dasum(n, z)
    _dscal(n, s, z)

    ynorm = 1.0
    # solve t*z = y
    for kk in range(1, n + 1):
        k = kk if lower else n + 1 - kk
        km = k - 1
        tkk = float(t[km, km])
        if abs(z[km]) > abs(tkk):
            s = abs(tkk) / abs(z[km])
            _dscal(n, s, z)
            ynorm = s * ynorm
        if tkk != 0.0:
            z[km] = z[km] / tkk
        if tkk == 0.0:
            z[km] = 1.0
        i1 = k + 1 if lower else 1
        if kk < n:
            w = -z[km]
            _daxpy(n - kk, w, t[:, km], z, ox=i1 - 1, oy=i1 - 1)
    s = 1.0 / _dasum(n, z)
    _dscal(n, s, z)
    ynorm = s * ynorm

    return ynorm / tnorm if tnorm != 0.0 else 0.0
