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
  adversarial mixed-magnitude probes, alignment-independent). For n ≥ 5
  Accelerate switches to a kernel whose reduction is NOT any sum tree
  over the rounded products (exhaustive 17M-tree search), nor an
  FMA-fold lane family, nor a compensated/correctly-rounded dot — it is
  behind several dyld-shared-cache dispatch hops and was left
  unidentified. Here n ≥ 5 falls back to sequential order, so on
  darwin/arm64 trajectory bit-parity with R holds wherever every dot
  has length ≤ 4: all of R's ``nlm``/uncmin (its solves stay n-sized;
  verified 72/72 bit-exact against R's compiled ``optif9`` for n ≤ 4)
  and L-BFGS-B with ≤ 2 parameters (verified 48/48 against R's
  compiled ``lbfgsb`` driver); L-BFGS-B with 3-4 parameters hits
  length-5..7 dots inside ``dtrsl`` on the ``2·col`` system and drifts
  ~1 ulp per such call. On reference-BLAS platforms (Linux CI) the
  sequential order is exact at every length, so no such boundary
  exists there.
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

import math
import platform
import sys

from ._shared import _rfma

# Accelerate is R's BLAS only on macOS, and its n=4 pair tree was
# probed on arm64; everywhere else R's ddot is plain sequential at
# every n (reference BLAS), so the tree must not be applied there.
_ACCEL_PAIR4 = sys.platform == "darwin" and platform.machine() == "arm64"


def _ddot(n, dx, dy, ox=0, oy=0):
    """R-linked ``ddot`` over ``n`` entries of ``dx[ox:]``/``dy[oy:]``:
    sequential everywhere, except the probed Accelerate pair tree at
    n = 4 on darwin/arm64 (see module docstring)."""
    if n == 4 and _ACCEL_PAIR4:
        s0 = float(dx[ox]) * float(dy[oy])
        s1 = float(dx[ox + 1]) * float(dy[oy + 1])
        s2 = float(dx[ox + 2]) * float(dy[oy + 2])
        s3 = float(dx[ox + 3]) * float(dy[oy + 3])
        return (s0 + s2) + (s1 + s3)
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

    ``t`` is an ``[row, col]``-indexed matrix (a numpy view is fine — that
    is the leading-dimension submatrix idiom); only the ``n×n`` leading
    block is referenced. ``job``: 0 = lower, 1 = upper, 10 = lower
    transposed, 11 = upper transposed. Returns ``info`` (0, or the
    1-based index of the first zero diagonal; ``b`` untouched then).
    The forward solves are the column-oriented ``daxpy`` sweeps of the
    original — the BLAS calls route through the Accelerate-emulating
    ``_daxpy``/``_ddot`` above, exactly as R's compiled ``dtrsl`` does."""
    for info in range(1, n + 1):
        if float(t[info - 1, info - 1]) == 0.0:
            return info
    kase = 1
    if job % 10 != 0:
        kase = 2
    if (job % 100) // 10 != 0:
        kase += 2
    if kase == 1:
        # solve t*x=b, t lower triangular
        b[0] = float(b[0]) / float(t[0, 0])
        for j in range(2, n + 1):
            temp = -float(b[j - 2])
            # daxpy(n-j+1, temp, t(j, j-1), 1, b(j), 1)
            _daxpy(n - j + 1, temp, t[:, j - 2], b, ox=j - 1, oy=j - 1)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    elif kase == 2:
        # solve t*x=b, t upper triangular
        b[n - 1] = float(b[n - 1]) / float(t[n - 1, n - 1])
        for jj in range(2, n + 1):
            j = n - jj + 1
            temp = -float(b[j])
            # daxpy(j, temp, t(1, j+1), 1, b(1), 1)
            _daxpy(j, temp, t[:, j], b)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    elif kase == 3:
        # solve trans(t)*x=b, t lower triangular
        b[n - 1] = float(b[n - 1]) / float(t[n - 1, n - 1])
        for jj in range(2, n + 1):
            j = n - jj + 1
            # b(j) -= ddot(jj-1, t(j+1, j), 1, b(j+1), 1)
            b[j - 1] = float(b[j - 1]) - _ddot(jj - 1, t[:, j - 1], b, ox=j, oy=j)
            b[j - 1] = float(b[j - 1]) / float(t[j - 1, j - 1])
    else:
        # solve trans(t)*x=b, t upper triangular
        b[0] = float(b[0]) / float(t[0, 0])
        for j in range(2, n + 1):
            # b(j) -= ddot(j-1, t(1, j), 1, b(1), 1)
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
            # t = a(k,j) - ddot(k-1, a(1,k), 1, a(1,j), 1)
            t = float(a[k - 1, j - 1]) - _ddot(k - 1, a[:, k - 1], a[:, j - 1])
            t /= float(a[k - 1, k - 1])
            a[k - 1, j - 1] = t
            s = _rfma(t, t, s)
        s = float(a[j - 1, j - 1]) - s
        if s <= eps * abs(float(a[j - 1, j - 1])):
            return j
        a[j - 1, j - 1] = math.sqrt(s)
    return 0
