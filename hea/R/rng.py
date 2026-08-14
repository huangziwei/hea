"""R's random number generator — a bit-exact port.

Reproduces R >= 3.6 defaults: ``RNGkind("Mersenne-Twister", normal.kind =
"Inversion", sample.kind = "Rejection")``. ``RMersenneTwister(seed)``
matches ``set.seed(seed)`` and then draws the identical stream R would,
so R results that depend on ``runif()`` / ``sample()`` can be pinned
bit-for-bit from Python.

Direct port of R's ``src/main/RNG.c`` and ``src/main/random.c``:

* ``RNG_Init`` — 50× LCG warm-up of the user seed, then 625 further LCG
  draws fill ``i_seed[0..624]``; ``FixupSeeds(initial=TRUE)`` overwrites
  ``i_seed[0]`` (the ``mti`` slot) with N=624 so the first draw
  regenerates the whole state.
* ``MT_genrand`` — MT19937 twist + R's tempering masks, scaled by the
  IEEE-754 nearest-double to 2⁻³². The twist here is vectorized over
  the 624-word block (tempering is a pure function of the stored word,
  so block-tempering equals R's per-draw tempering bit-for-bit).
* ``unif_rand`` — the MT output through R's ``fixup`` (keeps draws
  strictly inside (0, 1)).
* ``R_unif_index`` (``Sample_kind = REJECTION``) — ``bits =
  ceil(log2(dn))``, draw ``rbits(bits)`` until < dn.
* ``sample(n, k, replace=FALSE)`` — ``do_sample``'s shrinking-pool walk;
  ``replace=TRUE`` — one ``R_unif_index(n)`` per draw.

Index results are 0-based (R returns 1-based; callers adjust).

In-tree consumers: tp/ds/sos knot subsampling (``temp.seed(1)`` +
``sample()``, smooth.r:1286 via hea.formula), qq.gam's ``sample(U)``
randomization and k.check's permutation null (hea.models.gam), and
bam's ``compress.df`` / ``discrete.mf`` (``temp.seed(8547)``,
hea.models.bam).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg.lapack import dpstrf

from .._dispatch import rs as _rs_mod
from ._shared import _rfma, _rfma_vec

__all__ = ["RGenerator", "RMersenneTwister"]


def _make_impl(seed):
    """The compiled Rust MT (``hea._rs.RsMt``) for ``seed``, or ``None`` when the
    extension is unavailable / disabled — in which case the pure-Python stream
    runs. Both are bit-exact to R; the Rust path just skips the per-draw Python
    overhead in the rejection-sampling loops."""
    m = _rs_mod()
    return m.RsMt(int(seed)) if m is not None else None


def _R_pow_di(x: float, n: int) -> float:
    """R's ``R_pow_di(x, n)`` (arithmetic.c): integer power by repeated squaring.
    Deliberately NOT ``x ** n`` (libm ``pow``): R uses this in ``rbinom``'s
    ``qn = q^n`` and the two differ by up to hundreds of ulp, which can flip a
    rejection-sampling result. Bit-exact mirror of R's loop."""
    pow_ = 1.0
    if math.isnan(x):
        return x
    if n != 0:
        if not math.isfinite(x):
            return x ** float(n)  # R: R_pow(x, (double)n)
        is_neg = n < 0
        if is_neg:
            n = -n
        while True:
            if n & 1:
                pow_ *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
        if is_neg:
            pow_ = 1.0 / pow_
    return pow_


# Period parameters (RNG.c:646-650).
_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
# MT_genrand scale factor (RNG.c:722) — IEEE-754 nearest-double to 2^-32.
_INV_2P32 = 2.3283064365386963e-10
# fixup() boundary epsilon (RNG.c:86) — R's i2_32m1, a distinct constant
# (1/(2^32-1)-ish); only reachable when a tempered word is exactly 0.
_I2_32M1 = 2.328306437080797e-10
# rbeta (rbeta.c) overflow guard: expmax = DBL_MAX_EXP * M_LN2, and DBL_MAX.
_EXPMAX = 1024 * 0.6931471805599453
_DBL_MAX = 1.7976931348623157e308
_INT_MAX = 2147483647  # C INT_MAX — rhyper's large-n threshold

# qnorm5 — R's normal quantile (nmath/qnorm.c, Wichura 1988 AS-241): the exact
# rational-approx coefficients + Horner nesting R uses. norm_rand's Inversion
# case calls this on the combined-uniform argument, so a bit-exact port (not
# SciPy's ndtri, which differs ~1e-12) makes rnorm 0-ulp to R. Three branches
# keyed on |p-0.5|: central, near-tail (r<=5), far-tail.
_QN_A = (
    2509.0809287301226727,
    33430.575583588128105,
    67265.770927008700853,
    45921.953931549871457,
    13731.693765509461125,
    1971.5909503065514427,
    133.14166789178437745,
    3.387132872796366608,
)
_QN_B = (
    5226.495278852854561,
    28729.085735721942674,
    39307.89580009271061,
    21213.794301586595867,
    5394.1960214247511077,
    687.1870074920579083,
    42.313330701600911252,
    1.0,
)
_QN_C = (
    7.7454501427834140764e-4,
    0.0227238449892691845833,
    0.24178072517745061177,
    1.27045825245236838258,
    3.64784832476320460504,
    5.7694972214606914055,
    4.6303378461565452959,
    1.42343711074968357734,
)
_QN_D = (
    1.05075007164441684324e-9,
    5.475938084995344946e-4,
    0.0151986665636164571966,
    0.14810397642748007459,
    0.68976733498510000455,
    1.6763848301838038494,
    2.05319162663775882187,
    1.0,
)
_QN_E = (
    2.01033439929228813265e-7,
    2.71155556874348757815e-5,
    0.0012426609473880784386,
    0.026532189526576123093,
    0.29656057182850489123,
    1.7848265399172913358,
    5.4637849111641143699,
    6.6579046435011037772,
)
_QN_F = (
    2.04426310338993978564e-15,
    1.4215117583164458887e-7,
    1.8463183175100546818e-5,
    7.868691311456132591e-4,
    0.0148753612908506148525,
    0.13692988092273580531,
    0.59983220655588793769,
    1.0,
)


def _qn_horner(r, c, fma=_rfma):
    # `((c[0]*r + c[1])*r + …)` — R fuses each step to fmadd on arm64. `fma` is
    # `_rfma` (scalar) or `_rfma_vec` (array); both per-arch.
    v = c[0]
    for k in c[1:]:
        v = fma(v, r, k)
    return v


def _qnorm5(p: float) -> float:
    """R's ``qnorm5(p, mu=0, sigma=1, lower_tail=TRUE, log_p=FALSE)``
    (nmath/qnorm.c), bit-exact: same AS-241 coefficients and Horner nesting R
    uses. Boundaries return ±Inf at 0/1 and NaN outside [0, 1] / for NaN."""
    if math.isnan(p):
        return math.nan
    if p <= 0.0:
        return -math.inf if p == 0.0 else math.nan
    if p >= 1.0:
        return math.inf if p == 1.0 else math.nan
    q = p - 0.5
    if abs(q) <= 0.425:
        r = _rfma(-q, q, 0.180625)
        return q * _qn_horner(r, _QN_A) / _qn_horner(r, _QN_B)
    r = (1.0 - p) if q > 0.0 else p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r += -1.6
        val = _qn_horner(r, _QN_C) / _qn_horner(r, _QN_D)
    else:
        r += -5.0
        val = _qn_horner(r, _QN_E) / _qn_horner(r, _QN_F)
    return -val if q < 0.0 else val


def _qnorm5_vec(p: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_qnorm5` for ``p`` strictly in (0, 1) — the only range
    the Inversion deviates feed it. Bit-identical to the scalar version
    elementwise (same AS-241 coefficients, Horner nesting, and op order;
    ``_qn_horner`` is array-polymorphic). Both tail branches are evaluated and
    selected with ``np.where``; for any ``p`` in (0, 1) all three branches are
    finite (``min(p, 1-p)`` in (0, 0.5] ⇒ ``sqrt(-log)`` is real), so no NaNs
    leak through the discarded lanes."""
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    # Central: r = 0.180625 - q^2 (finite for all q).
    rc = _rfma_vec(-q, q, 0.180625)
    val_c = q * _qn_horner(rc, _QN_A, _rfma_vec) / _qn_horner(rc, _QN_B, _rfma_vec)
    # Tails: r = sqrt(-log(min(p, 1-p))).
    rt = np.sqrt(-np.log(np.where(q > 0.0, 1.0 - p, p)))
    rn = rt - 1.6
    rf = rt - 5.0
    val_t = np.where(
        rt <= 5.0,
        _qn_horner(rn, _QN_C, _rfma_vec) / _qn_horner(rn, _QN_D, _rfma_vec),
        _qn_horner(rf, _QN_E, _rfma_vec) / _qn_horner(rf, _QN_F, _rfma_vec),
    )
    val_t = np.where(q < 0.0, -val_t, val_t)
    return np.where(np.abs(q) <= 0.425, val_c, val_t)


def _beta_vw(AA: float, u1: float, beta: float) -> tuple[float, float]:
    """Cheng's ``v`` / ``w`` step shared by rbeta's BB and BC branches
    (rbeta.c ``v_w_from__u1_bet``), with R's overflow guard."""
    v = beta * math.log(u1 / (1.0 - u1))
    if v <= _EXPMAX:
        w = AA * math.exp(v)
        if not math.isfinite(w):
            w = _DBL_MAX
    else:
        w = _DBL_MAX
    return v, w


def _revsort(a: np.ndarray, ib: np.ndarray) -> None:
    """R's ``revsort`` (sort.c): heapsort ``a`` into **descending** order,
    applying the same permutation to ``ib``. In place; reproduces R's exact
    (non-stable) tie order so weighted ``sample`` is bit-exact."""
    n = len(a)
    if n <= 1:
        return
    A = [0.0] + list(a)  # 1-based: A[1..n]
    B = [0] + [int(x) for x in ib]
    lo = (n >> 1) + 1
    ir = n
    while True:
        if lo > 1:
            lo -= 1
            ra = A[lo]
            ii = B[lo]
        else:
            ra = A[ir]
            ii = B[ir]
            A[ir] = A[1]
            B[ir] = B[1]
            ir -= 1
            if ir == 1:
                A[1] = ra
                B[1] = ii
                break
        i = lo
        j = lo + lo
        while j <= ir:
            if j < ir and A[j] > A[j + 1]:
                j += 1
            if ra > A[j]:
                A[i] = A[j]
                B[i] = B[j]
                i = j
                j += j
            else:
                j = ir + 1
        A[i] = ra
        B[i] = ii
    for idx in range(n):
        a[idx] = A[idx + 1]
        ib[idx] = B[idx + 1]


def _mroot_chol(V: np.ndarray) -> np.ndarray:
    """``mgcv::mroot(V, rank=ncol(V), method="chol")`` — a pivoted-Cholesky
    matrix square root ``R`` (p×p) with ``R @ R.T == V``.

    Uses LAPACK ``dpstrf`` (the exact routine R's ``chol(pivot=TRUE, tol=0)``
    calls), then un-pivots the columns and transposes, mirroring ``mroot``'s R
    source line-for-line. Verified bit-identical to ``mgcv::mroot`` (the pivot
    order and every factor entry match). Used by :meth:`RMersenneTwister.rmvn`."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("V must be a square matrix")
    p = V.shape[1]
    # Upper factor U (p×p) and 1-based pivot s.t. U' U == V[piv, piv]. tol=0
    # matches mroot's chol(..., tol=0): stop only on a non-positive pivot.
    c, piv, rank, info = dpstrf(V, lower=0, tol=0.0)
    if info < 0:
        raise ValueError(f"dpstrf: illegal argument {-info}")
    U = np.triu(c[:p, :p])
    r = int(rank)
    if r < p:
        U[r:p, r:p] = 0.0  # mroot: zero the trailing block
    oo = np.argsort(piv, kind="stable")  # order(attr(L, "pivot")), 0-based
    Lp = U[:, oo]  # un-pivot columns: t(Lp) Lp == V
    return Lp.T.copy()  # t(L[1:rank,]) with rank == ncol(V)


# log(sqrt(2*pi)) — R's M_LN_SQRT_2PI, used by rhyper's Stirling afc().
_M_LN_SQRT_2PI = 0.918938533204672741780329736406

# ln(i!) table for i = 0..7 (rhyper.c `afc`), exact to the printed digits.
_AFC_AL = (
    0.0,
    0.0,
    0.69314718055994530941723212145817,
    1.79175946922805500081247735838070,
    3.17805383034794561964694160129705,
    4.78749174278204599424770093452324,
    6.57925121201010099506017829290394,
    8.52516136106541430016553103634712,
)


def _afc(i: int) -> float:
    """``afc(i) = ln(i!)`` (rhyper.c) — table lookup for i <= 7, else Stirling."""
    if i <= 7:
        return _AFC_AL[i]
    di = float(i)
    i2 = di * di
    return (
        (di + 0.5) * math.log(di)
        - di
        + _M_LN_SQRT_2PI
        + (0.0833333333333333 - 0.00277777777777778 / i2) / di
    )


class RMersenneTwister:
    """R's default RNG after ``set.seed(seed)``, bit-exact and
    platform-independent."""

    __slots__ = ("_buf", "_impl", "_mt", "_pos")

    def __init__(self, seed: int, *, force_py: bool = False):
        self._impl = None
        self.set_seed(seed, force_py=force_py)

    def set_seed(self, seed: int, *, force_py: bool = False) -> None:
        """R's ``set.seed(seed)`` (Mersenne-Twister kind).

        When the Rust extension is present (and ``force_py`` is False) the whole
        stream — uniforms, normals, and every ``r*`` sampler — runs natively via
        ``self._impl`` (an ``hea._rs.RsMt``). Otherwise the pure-Python state set
        up below is used; it is the bit-exact reference and fallback. The Python
        methods that aren't individually ported (weighted ``sample_prob``,
        ``rmvn``) still work in either mode because they consume the stream only
        through the delegating primitives. ``force_py=True`` pins the pure-Python
        path (used by the 3-way Rust/Python/R parity gate)."""
        self._impl = None if force_py else _make_impl(seed)
        if self._impl is not None:
            return
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        fills = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            fills[j] = s
        # fills[0] is i_seed[0] (the mti slot) which FixupSeeds discards;
        # the remaining 624 words are the MT state.
        self._mt = fills[1:].copy()
        self._buf = np.empty(0)
        self._pos = 0

    def _refill(self) -> None:
        # One MT19937 twist of the 624-word state, vectorized in
        # dependency-free slices. Wrap-around: the last word's y pairs
        # old mt[623] with the freshly updated mt[0] (the C loop has
        # already overwritten it by then).
        mt = self._mt
        y = (mt[: _N - 1] & _UPPER) | (mt[1:] & _LOWER)
        mag = np.where((y & np.uint32(1)) != 0, _MATRIX_A, np.uint32(0))
        yshift = (y >> np.uint32(1)) ^ mag
        new = np.empty(_N, dtype=np.uint32)
        new[: _N - _M] = mt[_M:] ^ yshift[: _N - _M]
        new[_N - _M : 2 * (_N - _M)] = new[: _N - _M] ^ yshift[_N - _M : 2 * (_N - _M)]
        new[2 * (_N - _M) : _N - 1] = new[_N - _M : _M - 1] ^ yshift[2 * (_N - _M) :]
        y_last = (mt[_N - 1] & _UPPER) | (new[0] & _LOWER)
        mag_last = _MATRIX_A if (int(y_last) & 1) else np.uint32(0)
        new[_N - 1] = new[_M - 1] ^ (y_last >> np.uint32(1)) ^ mag_last
        self._mt = new
        # Tempering (RNG.c:716-719) + fixup to the open interval (0, 1).
        t = new.copy()
        t ^= t >> np.uint32(11)
        t ^= (t << np.uint32(7)) & np.uint32(0x9D2C5680)
        t ^= (t << np.uint32(15)) & np.uint32(0xEFC60000)
        t ^= t >> np.uint32(18)
        u = t.astype(np.float64) * _INV_2P32
        u = np.where(u <= 0.0, 0.5 * _I2_32M1, u)
        u = np.where(1.0 - u <= 0.0, 1.0 - 0.5 * _I2_32M1, u)
        self._buf = u
        self._pos = 0

    def unif_rand(self, n: int | None = None):
        """R's ``runif`` stream: one draw (``n=None``) or a length-``n``
        array — consuming the identical sequence either way."""
        if self._impl is not None:
            return (
                self._impl.unif_rand() if n is None else self._impl.unif_rand_n(int(n))
            )
        if n is None:
            if self._pos >= self._buf.size:
                self._refill()
            v = float(self._buf[self._pos])
            self._pos += 1
            return v
        # Bulk-copy whole buffer slices instead of n scalar pulls — identical
        # values (the buffer is already fixup'd in _refill), refilling at the
        # same 624-word boundaries.
        n = int(n)
        out = np.empty(n)
        filled = 0
        while filled < n:
            if self._pos >= self._buf.size:
                self._refill()
            take = min(n - filled, self._buf.size - self._pos)
            out[filled : filled + take] = self._buf[self._pos : self._pos + take]
            self._pos += take
            filled += take
        return out

    def norm_rand(self) -> float:
        """R's ``norm_rand`` with ``normal.kind = "Inversion"`` (R's default).

        One standard normal = ``qnorm`` of two combined uniforms for 53-bit
        precision (``snorm.c`` INVERSION case): ``u = floor(2^27·u1) + u2``,
        return ``qnorm(u / 2^27)``. So each normal consumes **two**
        ``unif_rand`` draws. The MT uniforms are bit-exact to R, and ``qnorm``
        here is a bit-exact port of R's ``qnorm5`` (AS-241, see ``_qnorm5``),
        so each normal is 0-ulp to ``set.seed(); rnorm()``.
        """
        if self._impl is not None:
            return self._impl.norm_rand()
        big = 134217728.0  # 2^27
        u1 = self.unif_rand()
        u1 = float(int(big * u1)) + self.unif_rand()
        return _qnorm5(u1 / big)

    def rnorm(self, n: int | None = None, mean: float = 0.0, sd: float = 1.0):
        """R's ``rnorm(n, mean, sd)`` on R's MT stream (Inversion normals).

        ``n=None`` returns one draw; otherwise a length-``n`` array consuming
        the identical sequence. Use this (not numpy / ``hea.R.rnorm``) whenever
        the draws must line up with R's ``set.seed(); rnorm()``.

        Vectorized: the 2n Inversion uniforms are drawn in one batch (same
        stream as 2n scalar draws) and run through ``_qnorm5_vec`` — bit-
        identical to the per-draw scalar path, but without the Python loop.
        """
        if self._impl is not None:
            if n is None:
                return mean + sd * self._impl.norm_rand()
            return mean + sd * self._impl.rnorm_n(int(n))
        if n is None:
            return mean + sd * self.norm_rand()
        big = 134217728.0  # 2^27
        u = self.unif_rand(2 * int(n))
        comb = np.floor(big * u[0::2]) + u[1::2]
        return mean + sd * _qnorm5_vec(comb / big)

    def _rbits(self, bits: int) -> int:
        # rbits (RNG.c:875-885): 16 bits per unif_rand draw.
        v = 0
        nb = 0
        while nb <= bits:
            v1 = int(np.floor(self.unif_rand() * 65536.0))
            v = 65536 * v + v1
            nb += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """``R_unif_index(dn)``, ``Sample_kind = REJECTION`` — an integer
        in [0, dn). R computes ``bits = ceil(log2(dn))`` in C-double
        arithmetic; kept literally for fidelity."""
        if self._impl is not None:
            return self._impl.unif_index(int(dn))
        if dn <= 0:
            return 0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_int(self, n: int, k: int, replace: bool = False) -> np.ndarray:
        """R's ``sample(1:n, k, replace=)`` as 0-based indices.

        Without replacement: ``do_sample``'s shrinking-pool walk
        (src/main/random.c). With replacement: independent
        ``R_unif_index(n)`` per draw."""
        if self._impl is not None:
            return self._impl.sample_int(int(n), int(k), bool(replace))
        if replace:
            out = np.empty(k, dtype=np.int64)
            for i in range(k):
                out[i] = self.unif_index(n)
            return out
        if k < 0 or k > n:
            raise ValueError(f"k={k} not in [0, n={n}]")
        pool = np.arange(n, dtype=np.int64)
        out = np.empty(k, dtype=np.int64)
        m = n
        for i in range(k):
            j = self.unif_index(m)
            out[i] = pool[j]
            m -= 1
            pool[j] = pool[m]
        return out

    # bam-era aliases, kept for the established call sites.
    def sample_no_replace(self, n: int, k: int) -> np.ndarray:
        return self.sample_int(n, k)

    def sample_replace(self, n: int, k: int) -> np.ndarray:
        return self.sample_int(n, k, replace=True)

    def permute(self, x: np.ndarray) -> np.ndarray:
        """R's ``sample(x)`` — a full permutation of ``x``."""
        x = np.asarray(x)
        return x[self.sample_int(len(x), len(x))]

    # ------------------------------------------------------------------
    # Family samplers — ports of R's nmath ``r*`` on this bit-exact stream
    # (sexp.c / rpois.c / rbinom.c / rgamma.c). Used by simulate.merMod.
    # ------------------------------------------------------------------

    # cumulative ln(2)^k / k! — R sexp.c
    _EXP_Q = (
        0.6931471805599453,
        0.9333736875190459,
        0.9888777961838675,
        0.9984959252914960040,
        0.9998292811061389,
        0.9999833164100727,
        0.9999985691438767,
        0.9999998906925558,
        0.9999999924734159,
        0.9999999995283275,
        0.9999999999728814,
        0.9999999999985598,
        0.9999999999999289,
        0.9999999999999968,
        0.9999999999999999,
        1.0000000000000000,
    )

    def exp_rand(self) -> float:
        """R's ``exp_rand`` (standard exponential) — sexp.c (Ahrens-Dieter)."""
        if self._impl is not None:
            return self._impl.exp_rand()
        q = self._EXP_Q
        a = 0.0
        u = self.unif_rand()
        while u <= 0.0 or u >= 1.0:
            u = self.unif_rand()
        while True:
            u += u
            if u > 1.0:
                break
            a += q[0]
        u -= 1.0
        if u <= q[0]:
            return a + u
        i = 0
        ustar = self.unif_rand()
        umin = ustar
        while True:
            ustar = self.unif_rand()
            umin = min(umin, ustar)
            i += 1
            if u <= q[i]:
                break
        return _rfma(umin, q[0], a)

    def rpois(self, mu: float) -> float:
        """R's ``rpois(mu)`` — rpois.c. Inversion for ``mu < 10`` (one uniform
        per draw, CDF walk), transformed rejection (PTRS) for ``mu >= 10``."""
        if self._impl is not None:
            return self._impl.rpois(float(mu))
        if mu <= 0.0:
            return 0.0
        if mu < 10.0:  # inversion (consumes 1 uniform)
            u = self.unif_rand()
            p = q = p0 = math.exp(-mu)
            if u <= p0:
                return 0.0
            k = 0
            while k < 35:
                k += 1
                p *= mu / k
                q += p
                if u <= q:
                    return float(k)
            # tail: keep walking (rare)
            while True:
                k += 1
                p *= mu / k
                q += p
                if u <= q or p == 0.0:
                    return float(k)
        # big mu (>= 10): Ahrens-Dieter (1982) "PD" algorithm — R rpois.c.
        M_1_SQRT_2PI = 0.398942280401432677939946059934
        a0, a1, a2, a3 = -0.5, 0.3333333, -0.2500068, 0.2000118
        a4, a5, a6, a7 = -0.1661269, 0.1421878, -0.1384794, 0.1250060
        fact = (1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0)
        one_7, one_12, one_24 = (
            0.1428571428571428571,
            0.0833333333333333333,
            0.0416666666666666667,
        )
        s = math.sqrt(mu)
        d = 6.0 * mu * mu
        big_l = math.floor(mu - 1.1484)
        omega = M_1_SQRT_2PI / s
        b1 = one_24 / mu
        b2 = 0.3 * b1 * b1
        c3 = one_7 * b1 * b2
        c2 = b2 - 15.0 * c3
        c1 = b1 - 6.0 * b2 + 45.0 * c3
        c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3
        c = 0.1069 / mu

        def step_f(pois, fk, difmuk):
            if pois < 10:
                px = -mu
                py = mu**pois / fact[int(pois)]
            else:
                del_ = one_12 / fk
                del_ = del_ * (1.0 - 4.8 * del_ * del_)
                v = difmuk / fk
                if abs(v) <= 0.25:
                    px = (
                        fk
                        * v
                        * v
                        * (
                            (
                                (
                                    ((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
                                    + a2
                                )
                                * v
                                + a1
                            )
                            * v
                            + a0
                        )
                        - del_
                    )
                else:
                    px = fk * math.log(1.0 + v) - difmuk - del_
                py = M_1_SQRT_2PI / math.sqrt(fk)
            x = (0.5 - difmuk) / s
            xx = x * x
            fx = -0.5 * xx
            fy = omega * (((c3 * xx + c2) * xx + c1) * xx + c0)
            return px, py, fx, fy

        # Step N — normal candidate (immediate / squeeze acceptance)
        g = mu + s * self.norm_rand()
        if g >= 0.0:
            pois = math.floor(g)
            if pois >= big_l:
                return float(pois)
            fk = float(pois)
            difmuk = mu - fk
            u = self.unif_rand()
            if d * u >= difmuk * difmuk * difmuk:
                return float(pois)
            px, py, fx, fy = step_f(pois, fk, difmuk)
            if fy - u * fy <= py * math.exp(px - fx):
                return float(pois)
        # Step E — exponential candidates
        while True:
            e = self.exp_rand()
            u = 2.0 * self.unif_rand() - 1.0
            t = 1.8 + math.copysign(e, u)
            if t <= -0.6744:
                continue
            pois = math.floor(mu + s * t)
            fk = float(pois)
            difmuk = mu - fk
            px, py, fx, fy = step_f(pois, fk, difmuk)
            if c * abs(u) <= py * math.exp(px + e) - fy * math.exp(fx + e):
                return float(pois)

    def rbinom(self, size: float, prob: float) -> float:
        """R's ``rbinom(size, prob)`` — rbinom.c. Inversion for small
        ``size·min(p,1-p)``, BTPE rejection otherwise."""
        if self._impl is not None:
            return self._impl.rbinom(float(size), float(prob))
        n = round(size)
        if n == 0 or prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return float(n)
        p = min(prob, 1.0 - prob)
        q = 1.0 - p
        np_ = n * p
        if np_ < 30.0:  # inversion (BINV)
            qn = _R_pow_di(q, n)  # R_pow_di, NOT q**n (libm pow); see helper
            r = p / q
            g = r * (n + 1)
            while True:
                ix = 0
                f = qn
                u = self.unif_rand()
                while True:
                    if u < f:
                        return float(ix if prob <= 0.5 else n - ix)
                    if ix > 110:
                        break
                    u -= f
                    ix += 1
                    f *= g / ix - r
        # BTPE (Kachitvichyanukul & Schmeiser)
        ffm = np_ + p
        m = int(ffm)
        fm = m
        npq = np_ * q
        p1 = math.floor(2.195 * math.sqrt(npq) - 4.6 * q) + 0.5
        xm = fm + 0.5
        xl = xm - p1
        xr = xm + p1
        c = 0.134 + 20.5 / (15.3 + fm)
        al = (ffm - xl) / (ffm - xl * p)
        xll = al * (1.0 + 0.5 * al)
        al = (xr - ffm) / (xr * q)
        xlr = al * (1.0 + 0.5 * al)
        p2 = p1 * (1.0 + c + c)
        p3 = p2 + c / xll
        p4 = p3 + c / xlr
        while True:
            u = self.unif_rand() * p4
            v = self.unif_rand()
            if u <= p1:
                ix = int(xm - p1 * v + u)
                return float(ix if prob <= 0.5 else n - ix)
            if u <= p2:
                x = xl + (u - p1) / c
                v = v * c + 1.0 - abs(xm - x) / p1
                if v > 1.0 or v <= 0.0:
                    continue
                ix = int(x)
            elif u <= p3:
                ix = int(xl + math.log(v) / xll)
                if ix < 0:
                    continue
                v = v * (u - p2) * xll
            else:
                ix = int(xr - math.log(v) / xlr)
                if ix > n:
                    continue
                v = v * (u - p3) * xlr
            k = abs(ix - m)
            if k <= 20 or k >= npq / 2 - 1:
                f = 1.0
                r = p / q
                g = (n + 1) * r
                if m < ix:
                    for i in range(m + 1, ix + 1):
                        f *= g / i - r
                elif m > ix:
                    for i in range(ix + 1, m + 1):
                        f /= g / i - r
                if v <= f:
                    return float(ix if prob <= 0.5 else n - ix)
                continue
            amaxp = (k / npq) * ((k * (k / 3.0 + 0.625) + 0.1666666666666) / npq + 0.5)
            ynorm = -k * k / (2.0 * npq)
            alv = math.log(v)
            if alv < ynorm - amaxp:
                return float(ix if prob <= 0.5 else n - ix)
            if alv > ynorm + amaxp:
                continue
            x1 = ix + 1
            f1 = fm + 1.0
            z = n + 1 - fm
            w = n - ix + 1.0
            z2 = z * z
            x2 = x1 * x1
            f2 = f1 * f1
            w2 = w * w
            t = (
                xm * math.log(f1 / x1)
                + (n - m + 0.5) * math.log(z / w)
                + (ix - m) * math.log(w * p / (x1 * q))
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2)
                / f1
                / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2)
                / z
                / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2)
                / x1
                / 166320.0
                + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2)
                / w
                / 166320.0
            )
            if alv <= t:
                return float(ix if prob <= 0.5 else n - ix)

    # rgamma coefficients (R rgamma.c)
    _GA_Q = (
        0.04166669,
        0.02083148,
        0.00801191,
        0.00144121,
        -7.388e-5,
        2.4511e-4,
        2.424e-4,
    )
    _GA_A = (
        0.3333333,
        -0.250003,
        0.2000062,
        -0.1662921,
        0.1423657,
        -0.1367177,
        0.1233795,
    )

    def rgamma(self, shape: float, scale: float = 1.0) -> float:
        """R's ``rgamma(shape, scale=)`` — rgamma.c (GD for a>=1, GS for a<1)."""
        if self._impl is not None:
            return self._impl.rgamma(float(shape), float(scale))
        a = float(shape)
        if a < 1.0:  # GS algorithm
            if a == 0.0:
                return 0.0
            e1 = 0.36787944117144232159  # exp(-1)
            e = _rfma(e1, a, 1.0)
            while True:
                p = e * self.unif_rand()
                if p >= 1.0:
                    x = -math.log((e - p) / a)
                    if self.exp_rand() >= (1.0 - a) * math.log(x):
                        break
                else:
                    x = math.exp(math.log(p) / a)
                    if self.exp_rand() >= x:
                        break
            return scale * x
        # GD algorithm (a >= 1)
        sqrt32 = 5.656854
        s2 = a - 0.5
        s = math.sqrt(s2)
        d = _rfma(-12.0, s, sqrt32)
        t = self.norm_rand()
        x = _rfma(0.5, t, s)
        ret = x * x
        if t >= 0.0:
            return scale * ret
        u = self.unif_rand()
        if d * u <= t * t * t:
            return scale * ret
        r = 1.0 / a
        q = self._GA_Q
        q0 = _qn_horner(r, q[::-1]) * r
        if a <= 3.686:
            b = _rfma(0.178, s2, 0.463 + s)
            si = 1.235
            c = _rfma(0.16, s, 0.195 / s - 0.079)
        elif a <= 13.022:
            b = _rfma(0.0076, s2, 1.654)
            si = 1.68 / s + 0.275
            c = 0.062 / s + 0.024
        else:
            b = 1.77
            si = 0.75
            c = 0.1515 / s
        aa = self._GA_A
        if x > 0.0:
            v = t / (s + s)
            if abs(v) <= 0.25:
                qq = _rfma(0.5 * t * t * _qn_horner(v, aa[::-1]), v, q0)
            else:
                qq = _rfma(
                    s2 + s2, math.log(1.0 + v), _rfma(0.25 * t, t, _rfma(-s, t, q0))
                )
            if math.log(1.0 - u) <= qq:
                return scale * ret
        while True:
            e = self.exp_rand()
            u = 2.0 * self.unif_rand() - 1.0
            # R: `t = b - si*e` / `b + si*e` (clang fuses to fma on arm64);
            # copysign(si*e, u) rounds si*e first → diverges. Match R's fused form.
            t = _rfma(-si, e, b) if u < 0.0 else _rfma(si, e, b)
            if t >= -0.71874483771719:
                v = t / (s + s)
                if abs(v) <= 0.25:
                    qq = _rfma(0.5 * t * t * _qn_horner(v, aa[::-1]), v, q0)
                else:
                    qq = _rfma(
                        s2 + s2, math.log(1.0 + v), _rfma(0.25 * t, t, _rfma(-s, t, q0))
                    )
                if qq > 0.0:
                    w = math.expm1(qq)
                    if c * abs(u) <= w * math.exp(_rfma(-(0.5 * t), t, e)):
                        break
        x = _rfma(0.5, t, s)
        return scale * x * x

    def rnbinom(self, size: float, mu: float) -> float:
        """R's ``rnbinom(size, mu=)`` — a Poisson-Gamma mixture
        ``rpois(rgamma(size, scale=mu/size))`` (rnbinom.c)."""
        if self._impl is not None:
            return self._impl.rnbinom(float(size), float(mu))
        if mu <= 0.0:
            return 0.0
        return self.rpois(self.rgamma(size, scale=mu / size))

    # ------------------------------------------------------------------
    # Composed continuous families (R's rchisq/rt/rf built from the already
    # bit-exact rgamma/rnorm/rpois) and Cheng's rbeta. Central rt/rf are
    # per-draw scalars; the noncentral *block* ordering lives in
    # hea.R.distributions, where R applies it at the vector level.
    # ------------------------------------------------------------------

    def rchisq(self, df: float, ncp: float = 0.0) -> float:
        """R's ``rchisq(df, ncp=0)`` — central ``rgamma(df/2, 2)``; noncentral
        ``rnchisq`` = ``rpois(ncp/2)`` → ``rchisq(2·that)`` + ``rgamma(df/2, 2)``
        (rchisq.c / rnchisq.c)."""
        if self._impl is not None:
            return self._impl.rchisq(float(df), float(ncp))
        if ncp == 0.0:
            return 0.0 if df == 0.0 else self.rgamma(df / 2.0, scale=2.0)
        r = self.rpois(ncp / 2.0)
        out = self.rgamma(r, scale=2.0) if r > 0.0 else 0.0  # rchisq(2r)=rgamma(r,2)
        if df > 0.0:
            out += self.rgamma(df / 2.0, scale=2.0)
        return out

    def rt(self, df: float) -> float:
        """R's central ``rt(df)`` = ``norm_rand() / sqrt(rchisq(df)/df)`` (rt.c).
        Noncentral t is block-ordered at the vector level (see distributions)."""
        if self._impl is not None:
            return self._impl.rt(float(df))
        if not math.isfinite(df):
            return self.norm_rand()
        return self.norm_rand() / math.sqrt(self.rchisq(df) / df)

    def rf(self, df1: float, df2: float) -> float:
        """R's central ``rf(df1, df2)`` = ``(rchisq(df1)/df1)/(rchisq(df2)/df2)``
        (rf.c). Noncentral F is block-ordered at the vector level."""
        if self._impl is not None:
            return self._impl.rf(float(df1), float(df2))
        v1 = self.rchisq(df1) / df1 if math.isfinite(df1) else 1.0
        v2 = self.rchisq(df2) / df2 if math.isfinite(df2) else 1.0
        return v1 / v2

    def rbeta(self, aa: float, bb: float) -> float:
        """R's ``rbeta(aa, bb)`` — Cheng's BB (min > 1) / BC (min <= 1)
        algorithm (rbeta.c) on R's uniform stream."""
        if self._impl is not None:
            return self._impl.rbeta(float(aa), float(bb))
        if aa < 0.0 or bb < 0.0:
            raise ValueError("rbeta: shapes must be >= 0")
        if aa == 0.0 and bb == 0.0:
            return 0.0 if self.unif_rand() < 0.5 else 1.0
        a = min(bb, aa)  # min(aa, bb)
        b = max(aa, bb)  # max(aa, bb)
        alpha = a + b
        if a <= 1.0:  # --- Algorithm BC ---
            beta = 1.0 / a
            delta = 1.0 + b - a
            k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778)
            k2 = 0.25 + (0.5 + 0.25 / delta) * a
            while True:
                u1 = self.unif_rand()
                u2 = self.unif_rand()
                if u1 < 0.5:
                    y = u1 * u2
                    z = u1 * y
                    if 0.25 * u2 + z - y >= k1:
                        continue
                else:
                    z = u1 * u1 * u2
                    if z <= 0.25:
                        v, w = _beta_vw(b, u1, beta)
                        break
                    if z >= k2:
                        continue
                v, w = _beta_vw(b, u1, beta)
                if alpha * (math.log(alpha / (a + w)) + v) - 1.3862944 >= math.log(z):
                    break
            return a / (a + w) if aa == a else w / (a + w)
        # --- Algorithm BB ---
        beta = math.sqrt((alpha - 2.0) / (2.0 * a * b - alpha))
        gamma = a + 1.0 / beta
        while True:
            u1 = self.unif_rand()
            u2 = self.unif_rand()
            v, w = _beta_vw(a, u1, beta)
            z = u1 * u1 * u2
            r = gamma * v - 1.3862944
            s = a + r - w
            if s + 2.609438 >= 5.0 * z:
                break
            t = math.log(z)
            if s > t:
                break
            if r + alpha * math.log(alpha / (b + w)) >= t:
                break
        return b / (b + w) if aa != a else w / (b + w)

    def rnbinom_prob(self, size: float, prob: float) -> float:
        """R's ``rnbinom(size, prob)`` — Poisson-Gamma mixture
        ``rpois(rgamma(size, (1-prob)/prob))`` (rnbinom.c). The mu-parameterised
        variant is :meth:`rnbinom`."""
        if (
            not math.isfinite(prob)
            or math.isnan(size)
            or size <= 0.0
            or prob <= 0.0
            or prob > 1.0
        ):
            return math.nan
        if not math.isfinite(size):
            size = _DBL_MAX / 2.0
        if prob == 1.0:
            return 0.0
        return self.rpois(self.rgamma(size, scale=(1.0 - prob) / prob))

    # ------------------------------------------------------------------
    # Rank-statistic + hypergeometric + multinomial variates — ports of R's
    # rsignrank/rwilcox (signrank.c/wilcox.c), rhyper (rhyper.c, H2PE) and
    # rmultinom (rmultinom.c). Each consumes the uniform stream in the same
    # order as the C code (via the primitive draws above), so set.seed() is
    # bit-exact. No _impl branch: the primitives already route through Rust.
    # ------------------------------------------------------------------

    def rsignrank(self, n: float) -> float:
        """R's ``rsignrank(nn)`` — one Wilcoxon signed-rank variate,
        ``sum_{i=1}^{n} i * floor(unif_rand() + 0.5)`` (signrank.c)."""
        if math.isnan(n):
            return n
        n = float(np.rint(n))
        if n < 0.0:
            return math.nan
        if n == 0.0:
            return 0.0
        r = 0.0
        for i in range(1, int(n) + 1):
            r += i * math.floor(self.unif_rand() + 0.5)
        return r

    def rwilcox(self, m: float, n: float) -> float:
        """R's ``rwilcox(m, n)`` — one Wilcoxon rank-sum variate via a partial
        Fisher-Yates draw of ``n`` ranks from ``0..m+n-1`` (wilcox.c)."""
        if math.isnan(m) or math.isnan(n):
            return m + n
        m = float(np.rint(m))
        n = float(np.rint(n))
        if m < 0.0 or n < 0.0:
            return math.nan
        if m == 0.0 or n == 0.0:
            return 0.0
        r = 0.0
        k = int(m + n)
        x = list(range(k))
        nn = int(n)
        for _ in range(nn):
            j = self.unif_index(k)
            r += x[j]
            k -= 1
            x[j] = x[k]
        return r - n * (n - 1.0) / 2.0

    def rmultinom(self, n: int, size: int, prob) -> np.ndarray:
        """R's ``rmultinom(n, size, prob)`` — a (K x n) integer matrix whose
        columns are independent Multinomial(size, prob) draws (rmultinom.c).
        ``prob`` is normalised via ``FixupProb`` (plain-double sum of the
        positive entries), then each column fills the first K-1 cells with
        sequential ``rbinom`` on the shrinking remainder."""
        p = np.asarray(prob, dtype=float).ravel()
        K = p.size
        s = 0.0  # FixupProb: sum only p[i] > 0
        for x in p:
            if not math.isfinite(x):
                raise ValueError("NA in probability vector")
            if x < 0.0:
                raise ValueError("negative probability")
            if x > 0.0:
                s += x
        if s == 0.0:
            raise ValueError("no positive probabilities")
        p = p / s
        nn = int(n)
        out = np.zeros((K, nn), dtype=np.int64)
        for col in range(nn):
            out[:, col] = self._rmultinom_col(int(size), p, K)
        return out

    def _rmultinom_col(self, size: int, prob: np.ndarray, K: int) -> np.ndarray:
        """One Multinomial column: rN[0:K], sum == size (rmultinom.c inner)."""
        rN = [0] * K
        p_tot = np.longdouble(0.0)  # R accumulates in LDOUBLE
        for k in range(K):
            p_tot += np.longdouble(prob[k])
        n = size
        if n == 0:
            return np.array(rN, dtype=np.int64)
        if K == 1 and p_tot == 0.0:
            return np.array(rN, dtype=np.int64)
        for k in range(K - 1):
            if prob[k] != 0.0:
                pp = float(np.longdouble(prob[k]) / p_tot)
                rN[k] = n if pp >= 1.0 else int(self.rbinom(n, pp))
                n -= rN[k]
            else:
                rN[k] = 0
            if n <= 0:  # all drawn
                return np.array(rN, dtype=np.int64)
            p_tot -= np.longdouble(prob[k])
        rN[K - 1] = n
        return np.array(rN, dtype=np.int64)

    def rhyper(self, nn1in: float, nn2in: float, kkin: float) -> float:
        """R's ``rhyper(m, n, k)`` — number of white balls when ``k`` are drawn
        from an urn of ``m`` white + ``n`` black (rhyper.c, Kachitvichyanukul-
        Schmeiser H2PE). Consumes the uniform stream in the same order as C."""
        if not (math.isfinite(nn1in) and math.isfinite(nn2in) and math.isfinite(kkin)):
            return math.nan
        nn1in = float(np.rint(nn1in))
        nn2in = float(np.rint(nn2in))
        kkin = float(np.rint(kkin))
        if nn1in < 0 or nn2in < 0 or kkin < 0 or kkin > nn1in + nn2in:
            return math.nan
        if nn1in >= _INT_MAX or nn2in >= _INT_MAX or kkin >= _INT_MAX:
            # large n: evade int overflow / inappropriate algorithms
            if kkin == 1.0:
                return self.rbinom(kkin, nn1in / (nn1in + nn2in))
            from . import nmath as _nm

            return _nm.qhyper(self.unif_rand(), nn1in, nn2in, kkin, False, False)

        nn1 = int(nn1in)
        nn2 = int(nn2in)
        kk = int(kkin)
        # --- setup (always, on fresh parameters) ---
        N = nn1 + float(nn2)
        if nn1 <= nn2:
            n1, n2 = nn1, nn2
        else:
            n1, n2 = nn2, nn1
        k = int(N - kk) if (kk + kk >= N) else kk  # now k < N/2
        m = int((k + 1.0) * (n1 + 1.0) / (N + 2.0))  # floor(adjusted mean)
        minjx = max(0, k - n2)
        maxjx = min(n1, k)

        if minjx == maxjx:  # I: degenerate
            ix = maxjx
        elif m - minjx < 10:  # II: scaled HIN inverse
            con = 57.5646273248511421
            scale = 1e25
            if k < n2:
                lw = _afc(n2) + _afc(n1 + n2 - k) - _afc(n2 - k) - _afc(n1 + n2)
            else:
                lw = _afc(n1) + _afc(k) - _afc(k - n2) - _afc(n1 + n2)
            w = math.exp(lw + con)
            while True:  # L10
                p = w
                ix = minjx
                u = self.unif_rand() * scale
                resample = False
                while u > p:
                    u -= p
                    p *= (float(n1) - ix) * (k - ix)
                    ix += 1
                    p = p / ix / (n2 - k + ix)
                    if ix > maxjx:
                        resample = True
                        break
                if not resample:
                    break
        else:  # III: H2PE
            s = math.sqrt((N - k) * k * n1 * n2 / (N - 1) / N / N)
            d = float(int(1.5 * s)) + 0.5
            xl = m - d + 0.5
            xr = m + d + 0.5
            a = _afc(m) + _afc(n1 - m) + _afc(k - m) + _afc(n2 - k + m)
            kl = math.exp(
                a
                - _afc(int(xl))
                - _afc(int(n1 - xl))
                - _afc(int(k - xl))
                - _afc(int(n2 - k + xl))
            )
            kr = math.exp(
                a
                - _afc(int(xr - 1))
                - _afc(int(n1 - xr + 1))
                - _afc(int(k - xr + 1))
                - _afc(int(n2 - k + xr - 1))
            )
            lamdl = -math.log(xl * (n2 - k + xl) / (n1 - xl + 1) / (k - xl + 1))
            lamdr = -math.log((n1 - xr + 1) * (k - xr + 1) / xr / (n2 - k + xr))
            p1 = d + d
            p2 = p1 + kl / lamdl
            p3 = p2 + kr / lamdr
            n_uv = 0
            while True:  # L30: accept/reject
                u = self.unif_rand() * p3
                v = self.unif_rand()
                n_uv += 1
                if n_uv >= 10000:
                    return math.nan
                if u < p1:  # rectangular
                    ix = int(xl + u)
                elif u <= p2:  # left tail
                    ix = int(xl + math.log(v) / lamdl)
                    if ix < minjx:
                        continue
                    v = v * (u - p1) * lamdl
                else:  # right tail
                    ix = int(xr - math.log(v) / lamdr)
                    if ix > maxjx:
                        continue
                    v = v * (u - p2) * lamdr
                if m < 100 or ix <= 50:  # explicit evaluation
                    f = 1.0
                    if m < ix:
                        for i in range(m + 1, ix + 1):
                            f = f * (n1 - i + 1) * (k - i + 1) / (n2 - k + i) / i
                    elif m > ix:
                        for i in range(ix + 1, m + 1):
                            f = f * i * (n2 - k + i) / (n1 - i + 1) / (k - i + 1)
                    if v <= f:
                        break
                    continue
                # squeeze using upper and lower bounds
                deltal = 0.0078
                deltau = 0.0034
                y = float(ix)
                y1 = y + 1.0
                ym = y - m
                yn = n1 - y + 1.0
                yk = k - y + 1.0
                nk = n2 - k + y1
                r = -ym / y1
                sq = ym / yn
                t = ym / yk
                e = -ym / nk
                g = yn * yk / (y1 * nk) - 1.0
                dg = 1.0 + g if g < 0.0 else 1.0
                gu = g * (1.0 + g * (-0.5 + g / 3.0))
                gl = gu - 0.25 * (g * g * g * g) / dg
                xm = m + 0.5
                xn = n1 - m + 0.5
                xk = k - m + 0.5
                nm = n2 - k + xm
                ub = (
                    y * gu
                    - m * gl
                    + deltau
                    + xm * r * (1.0 + r * (-0.5 + r / 3.0))
                    + xn * sq * (1.0 + sq * (-0.5 + sq / 3.0))
                    + xk * t * (1.0 + t * (-0.5 + t / 3.0))
                    + nm * e * (1.0 + e * (-0.5 + e / 3.0))
                )
                alv = math.log(v)
                if alv > ub:  # test upper bound
                    continue
                dr = xm * (r * r * r * r)
                if r < 0.0:
                    dr /= 1.0 + r
                ds = xn * (sq * sq * sq * sq)
                if sq < 0.0:
                    ds /= 1.0 + sq
                dt = xk * (t * t * t * t)
                if t < 0.0:
                    dt /= 1.0 + t
                de = nm * (e * e * e * e)
                if e < 0.0:
                    de /= 1.0 + e
                if alv < ub - 0.25 * (dr + ds + dt + de) + (y + m) * (gl - gu) - deltal:
                    break  # test lower bound
                # Stirling to machine accuracy
                if alv <= (
                    a - _afc(ix) - _afc(n1 - ix) - _afc(k - ix) - _afc(n2 - k + ix)
                ):
                    break
                # else reject → redraw

        # --- L_finis: map ix back to the original parameterisation ---
        if kk + kk >= N:
            ix = (kk - nn2 + ix) if (nn1 > nn2) else (nn1 - ix)
        elif nn1 > nn2:
            ix = kk - ix
        return float(ix)

    # ------------------------------------------------------------------
    # Multivariate variates: rcont2 (AS 159, backs r2dtable) and the
    # standardized Wishart Bartlett factor (backs rWishart). ``fact`` (the
    # log-factorial table) is supplied by the caller so this stays nmath-free.
    # ------------------------------------------------------------------

    def rcont2(self, nrowt, ncolt, ntotal, fact):
        """R's ``rcont2`` (rcont.c, AS 159) — one random 2-way table with the
        given row/column margins. ``fact[i] = lgamma(i+1)``. Returns an
        nrow×ncol integer matrix; consumes the uniform stream as the C code."""
        nrowt = [int(v) for v in nrowt]
        ncolt = [int(v) for v in ncolt]
        nrow = len(nrowt)
        ncol = len(ncolt)
        nr_1 = nrow - 1
        nc_1 = ncol - 1
        jwork = [0] * ncol
        for j in range(nc_1):
            jwork[j] = ncolt[j]
        matrix = [[0] * ncol for _ in range(nrow)]
        ib = 0
        jc = ntotal
        for lr in range(nr_1):  # rows 0..nrow-2
            ia = nrowt[lr]
            ic = jc
            jc -= ia
            for m in range(nc_1):
                id_ = jwork[m]
                ie = ic
                ib = ie - ia
                ii = ib - id_
                ic -= id_
                if ie == 0:  # row full → zero the rest
                    for j in range(m, nc_1):
                        matrix[lr][j] = 0
                    ia = 0
                    break
                u = self.unif_rand()
                nlm = self._rcont2_cell(ia, id_, ie, ii, ib, ic, u, fact)
                matrix[lr][m] = nlm
                ia -= nlm
                jwork[m] -= nlm
            matrix[lr][nc_1] = ia  # last column of row lr
        for m in range(nc_1):  # last row = leftover margins
            matrix[nr_1][m] = jwork[m]
        matrix[nr_1][nc_1] = ib - matrix[nr_1][nc_1 - 1]
        return matrix

    def _rcont2_cell(self, ia, id_, ie, ii, ib, ic, u, fact):
        """The AS 159 inner search for one cell value (rcont.c 'Outer Loop')."""
        while True:  # (A) outer loop
            nlm = int(ia * (id_ / float(ie)) + 0.5)
            x = math.exp(
                fact[ia]
                + fact[ib]
                + fact[ic]
                + fact[id_]
                - fact[ie]
                - fact[nlm]
                - fact[id_ - nlm]
                - fact[ia - nlm]
                - fact[ii + nlm]
            )
            if x >= u:
                return nlm
            if x == 0.0:
                raise RuntimeError("rcont2: exp underflow to 0; algorithm failure")
            sumprb = x
            y = x
            nll = nlm
            lsp = False
            while not lsp:  # (B) do..while(!lsp)
                j = (id_ - nlm) * float(ia - nlm)
                lsp = (nlm == ia) or (nlm == id_)
                if not lsp:
                    nlm += 1
                    x *= j / (float(nlm) * (ii + nlm))
                    sumprb += x
                    if sumprb >= u:
                        return nlm
                lsm = False
                while not lsm:  # (C) do..while(!lsm)
                    j = nll * float(ii + nll)
                    lsm = nll == 0
                    if not lsm:
                        nll -= 1
                        y *= j / (float(id_ - nll) * (ia - nll))
                        sumprb += y
                        if sumprb >= u:
                            return nll  # nlm = nll; goto L160
                        if not lsp:
                            break  # back to (B) condition
            u = sumprb * self.unif_rand()

    def std_rwishart_factor(self, nu: float, p: int) -> np.ndarray:
        """R's ``std_rWishart_factor(nu, p, upper=1)`` (rWishart.c) — a p×p
        upper-triangular Bartlett factor: diagonal ``sqrt(rchisq(nu-j))``,
        strict-upper ``norm_rand()``. Draw order matches the C column sweep."""
        if nu < float(p) or p <= 0:
            raise ValueError("inconsistent degrees of freedom and dimension")
        ans = np.zeros((p, p), dtype=float)  # column-major (i, j) → ans[i, j]
        for j in range(p):  # jth column
            ans[j, j] = math.sqrt(self.rchisq(nu - float(j)))
            for i in range(j):
                ans[i, j] = self.norm_rand()  # upper triangle
        return ans

    # ------------------------------------------------------------------
    # Batch samplers — the whole per-element loop runs in one call (Rust when
    # available, else a Python list-comp), saving n Python↔Rust crossings for
    # the family ``$rd`` hooks and the public ``hea.R.r*`` vector draws. Each
    # element is bit-identical to the scalar method and the draw order is the
    # same as n scalar calls.
    # ------------------------------------------------------------------
    def rpois_n(self, mu) -> np.ndarray:
        mu = np.ascontiguousarray(mu, dtype=float)
        if self._impl is not None:
            return self._impl.rpois_n(mu)
        return np.array([self.rpois(float(m)) for m in mu])

    def rbinom_n(self, size, prob) -> np.ndarray:
        size = np.ascontiguousarray(size, dtype=float)
        prob = np.ascontiguousarray(prob, dtype=float)
        if self._impl is not None:
            return self._impl.rbinom_n(size, prob)
        return np.array(
            [self.rbinom(round(float(s)), float(p)) for s, p in zip(size, prob)]
        )

    def rgamma_n(self, shape, scale) -> np.ndarray:
        shape = np.ascontiguousarray(shape, dtype=float)
        scale = np.ascontiguousarray(scale, dtype=float)
        if self._impl is not None:
            return self._impl.rgamma_n(shape, scale)
        return np.array(
            [self.rgamma(float(a), scale=float(c)) for a, c in zip(shape, scale)]
        )

    def rt_n(self, df) -> np.ndarray:
        df = np.ascontiguousarray(df, dtype=float)
        if self._impl is not None:
            return self._impl.rt_n(df)
        return np.array([self.rt(float(d)) for d in df])

    def rf_n(self, df1, df2) -> np.ndarray:
        df1 = np.ascontiguousarray(df1, dtype=float)
        df2 = np.ascontiguousarray(df2, dtype=float)
        if self._impl is not None:
            return self._impl.rf_n(df1, df2)
        return np.array([self.rf(float(a), float(b)) for a, b in zip(df1, df2)])

    def rchisq_n(self, df, ncp) -> np.ndarray:
        df = np.ascontiguousarray(df, dtype=float)
        ncp = np.ascontiguousarray(ncp, dtype=float)
        if self._impl is not None:
            return self._impl.rchisq_n(df, ncp)
        return np.array([self.rchisq(float(d), float(c)) for d, c in zip(df, ncp)])

    def rnbinom_n(self, size, mu) -> np.ndarray:
        size = np.ascontiguousarray(size, dtype=float)
        mu = np.ascontiguousarray(mu, dtype=float)
        if self._impl is not None:
            return self._impl.rnbinom_n(size, mu)
        return np.array([self.rnbinom(float(s), float(m)) for s, m in zip(size, mu)])

    def rbeta_n(self, aa, bb) -> np.ndarray:
        aa = np.ascontiguousarray(aa, dtype=float)
        bb = np.ascontiguousarray(bb, dtype=float)
        if self._impl is not None:
            return self._impl.rbeta_n(aa, bb)
        return np.array([self.rbeta(float(a), float(b)) for a, b in zip(aa, bb)])

    def exp_rand_n(self, n: int) -> np.ndarray:
        if self._impl is not None:
            return self._impl.exp_rand_n(int(n))
        return np.array([self.exp_rand() for _ in range(int(n))])

    def sample_prob(self, prob, k: int, replace: bool = False) -> np.ndarray:
        """R's weighted ``sample(n, k, replace=, prob=)`` as 0-based indices.
        Replicates ``FixupProb`` + ``revsort`` + ``ProbSample[No]Replace`` and,
        for replacement when >200 weights are sizeable, the Walker alias method
        — matching R's algorithm selection (``do_sample``, random.c)."""
        p = np.asarray(prob, dtype=float).copy()
        n = p.size
        if not np.all(np.isfinite(p)):
            raise ValueError("sample: NA/Inf in prob")
        if np.any(p < 0.0):
            raise ValueError("sample: negative prob")
        npos = int(np.count_nonzero(p > 0.0))
        if npos == 0 or (not replace and k > npos):
            raise ValueError("sample: too few positive probabilities")
        p /= float(p.sum())
        perm = np.arange(n, dtype=np.int64)
        if replace:
            if int(np.count_nonzero(n * p > 0.1)) > 200:
                return self._walker_sample(p, k)
            pw = p.copy()
            pm = perm.copy()
            _revsort(pw, pm)
            cs = np.cumsum(pw)
            out = np.empty(k, dtype=np.int64)
            nm1 = n - 1
            for i in range(k):
                rU = self.unif_rand()
                j = 0
                while j < nm1 and rU > cs[j]:
                    j += 1
                out[i] = pm[j]
            return out
        # ProbSampleNoReplace — revsort once, then shrinking-pool walk.
        pw = p.copy()
        pm = perm.copy()
        _revsort(pw, pm)
        out = np.empty(k, dtype=np.int64)
        totalmass = 1.0
        n1 = n - 1
        for i in range(k):
            rT = totalmass * self.unif_rand()
            mass = 0.0
            j = 0
            while j < n1:
                mass += pw[j]
                if rT <= mass:
                    break
                j += 1
            out[i] = pm[j]
            totalmass -= pw[j]
            for kk in range(j, n1):
                pw[kk] = pw[kk + 1]
                pm[kk] = pm[kk + 1]
            n1 -= 1
        return out

    def _walker_sample(self, p: np.ndarray, k: int) -> np.ndarray:
        """Walker alias method (``walker_ProbSampleReplace``, random.c) — R's
        with-replacement weighted path when >200 weights are sizeable."""
        n = p.size
        q = np.empty(n)
        a = np.zeros(n, dtype=np.int64)
        HL = np.empty(n, dtype=np.int64)
        hpos = -1  # H = HL-1; ``*++H`` fills HL[0..] (low, q<1)
        lpos = n  # L = HL+n; ``*--L`` fills HL[n-1..] (high, q>=1)
        for i in range(n):
            q[i] = p[i] * n
            if q[i] < 1.0:
                hpos += 1
                HL[hpos] = i
            else:
                lpos -= 1
                HL[lpos] = i
        if hpos >= 0 and lpos < n:
            for kk in range(n - 1):
                i = int(HL[kk])
                j = int(HL[lpos])
                a[i] = j
                q[j] += q[i] - 1.0
                if q[j] < 1.0:
                    lpos += 1
                if lpos >= n:
                    break
        for i in range(n):
            q[i] += i
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            # Sample_kind = REJECTION (R's default): index via R_unif_index,
            # then one more unif_rand for the within-cell test (random.c).
            kk = int(self.unif_index(n))
            rU = kk + self.unif_rand()
            out[i] = kk if rU < q[kk] else int(a[kk])
        return out

    def rmvn(self, n: int, mu, V) -> np.ndarray:
        """``mgcv::rmvn(n, mu, V)`` on R's MT stream — multivariate normals via
        the pivoted-Cholesky root :func:`_mroot_chol`.

        ``mu`` is either a length-``p`` vector — returns an ``(n, p)`` array
        (a length-``p`` vector when ``n == 1``, matching R's ``as.numeric``) —
        or an ``n×p`` matrix, returning ``(n, p)``. The ``p*n`` standard
        normals are drawn with bit-exact :meth:`rnorm` in R's column-major
        ``matrix(rnorm(p*n), ...)`` order, so ``set.seed(k)`` reproduces R's
        draws. The trailing ``R @ Z`` GEMM matches R to machine precision only
        (BLAS associativity), which downstream quantiles/rounding absorb."""
        V = np.asarray(V, dtype=float)
        p = V.shape[1]
        R = _mroot_chol(V)
        mu = np.asarray(mu, dtype=float)
        n = int(n)
        z = self.rnorm(p * n)
        if mu.ndim == 2:
            # matrix-mu: z <- matrix(rnorm(p*n), n, p) %*% t(R) + mu
            if mu.shape != (n, p):
                raise ValueError("mu dimensions wrong")
            return z.reshape((n, p), order="F") @ R.T + mu
        # vector-mu: z <- t(R %*% matrix(rnorm(p*n), p, n) + as.numeric(mu))
        if mu.shape[0] != p:
            raise ValueError("mu dimensions wrong")
        out = (R @ z.reshape((p, n), order="F") + mu[:, None]).T
        return out.ravel() if n == 1 else out


def _rgen_resolve(size, *params):
    """Shared shape logic for :class:`RGenerator`: returns ``(n, scalar, cols)``
    where each ``col`` is the param broadcast to a length-``n`` 1-D array, and
    ``scalar`` flags numpy's "all-scalar input, no size → return a scalar"
    convention so the facade matches ``numpy.random.Generator``'s return type."""
    arrs = [np.asarray(p, dtype=float) for p in params]
    if size is None:
        b = np.broadcast(*arrs)
        n = b.size
        scalar = b.ndim == 0
    else:
        n = int(size)
        scalar = False
    cols = [np.ravel(np.broadcast_to(a, (n,))) for a in arrs]
    return n, scalar, cols


class RGenerator:
    """A numpy-``Generator``-compatible facade over :class:`RMersenneTwister`.

    Code written against numpy's RNG API — notably ``family.py``'s ``rd`` hooks,
    which call ``rng.normal`` / ``gamma`` / ``poisson`` / ``binomial`` /
    ``standard_t`` / ``uniform`` — draws from R's bit-exact stream instead when
    handed one of these. Each method maps to the matching R sampler in R's
    vectorised (per-element) order and mirrors numpy's scalar-vs-array return,
    so ``set.seed(k)``-pinned R results (e.g. ``qq.gam(rep>0)``) reproduce
    bit-for-bit. (Inverse-Gaussian deviates use mgcv's ``rig`` — n ``normal`` +
    n ``uniform`` — and tweedie uses ``poisson`` + ``gamma``, so no dedicated
    ``wald``/``rTweedie`` method is needed.) ``multivariate_normal`` maps to
    ``mgcv::rmvn`` for itsadug's simultaneous-CI draws."""

    __slots__ = ("mt",)

    def __init__(self, seed_or_mt):
        self.mt = (
            seed_or_mt
            if isinstance(seed_or_mt, RMersenneTwister)
            else RMersenneTwister(int(seed_or_mt))
        )

    def normal(self, loc=0.0, scale=1.0, size=None):
        n, scalar, (loc, scale) = _rgen_resolve(size, loc, scale)
        z = self.mt.rnorm(n)  # vectorized standard normals
        out = loc + scale * z
        return float(out[0]) if scalar else out

    def gamma(self, shape, scale=1.0, size=None):
        _n, scalar, (shape, scale) = _rgen_resolve(size, shape, scale)
        out = self.mt.rgamma_n(shape, scale)
        return float(out[0]) if scalar else out

    def poisson(self, lam=1.0, size=None):
        _n, scalar, (lam,) = _rgen_resolve(size, lam)
        out = self.mt.rpois_n(lam)
        return float(out[0]) if scalar else out

    def binomial(self, n, p, size=None):
        _m, scalar, (nt, pp) = _rgen_resolve(size, n, p)
        out = self.mt.rbinom_n(nt, pp)  # rounds size per-element
        return float(out[0]) if scalar else out

    def standard_t(self, df, size=None):
        _n, scalar, (df,) = _rgen_resolve(size, df)
        out = self.mt.rt_n(df)
        return float(out[0]) if scalar else out

    def uniform(self, low=0.0, high=1.0, size=None):
        n, scalar, (low, high) = _rgen_resolve(size, low, high)
        out = low + (high - low) * self.mt.unif_rand(n)
        return float(out[0]) if scalar else out

    def multivariate_normal(self, mean, cov, size=None, method=None):
        """Facade over :meth:`RMersenneTwister.rmvn` (``mgcv::rmvn``) matching
        ``numpy.random.Generator.multivariate_normal``'s signature/return: a
        ``(p,)`` vector when ``size is None``, else ``(size, p)``. ``method`` is
        accepted for drop-in compatibility and ignored — ``rmvn`` always uses
        the pivoted-Cholesky root, as itsadug does. Routes draws through R's
        bit-exact stream so ``set_seed(k)`` reproduces R+itsadug's MVN draws."""
        mean = np.asarray(mean, dtype=float)
        if size is None:
            return self.mt.rmvn(1, mean, cov)
        return self.mt.rmvn(int(size), mean, cov)
