"""Bit-exact port of R's Mersenne-Twister RNG and ``sample()``.

Reproduces the output of R 3.6+ default ``RNGkind("Mersenne-Twister",
"Inversion", "Rejection")`` so that mgcv's ``temp.seed(8547)`` +
``sample()`` calls inside ``compress.df`` / ``discrete.mf`` can be
matched bit-exactly from Python.

Direct port of the relevant pieces of R's ``src/main/RNG.c`` and
``src/main/random.c``:

* ``RNG_Init`` — 50× LCG (``seed = 69069*seed + 1``) warm-up, then 625
  more LCG iterations to fill ``i_seed[0..624]``. ``FixupSeeds`` with
  ``initial=1`` then overwrites ``i_seed[0]`` (= mti) with N=624 so the
  first ``MT_genrand`` regenerates the state.
* ``MT_genrand`` — standard MT19937 step plus R's tempering masks,
  scaled to a double via ``y * 2.3283064365386963e-10``.
* ``unif_rand`` — wraps MT output through R's ``fixup`` to avoid 0.0
  and 1.0 exactly.
* ``R_unif_index`` (Sample_kind = REJECTION) — bits = ceil(log2(dn)),
  draws ``rbits(bits)`` until result < dn.
* ``rbits`` — packs the low ``bits`` bits from successive
  ``floor(unif_rand() * 65536)`` chunks (one chunk for ``bits<=15``,
  two for 16–31).
* ``sample(n, k, replace=TRUE)`` — independent ``R_unif_index(n)`` per
  draw.
* ``sample(n, k, replace=FALSE)`` — partial Fisher-Yates with
  swap-and-pop using ``R_unif_index(m)`` for the decreasing remaining
  population ``m``.

We return 0-based indices throughout (R returns 1-based — caller
adjusts if needed).
"""
from __future__ import annotations

import numpy as np

# Period parameters (RNG.c:646-650).
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF

# MT_genrand scale factor (RNG.c:722). Note: this is the literal R uses,
# which is the IEEE-754 nearest-double to 2^-32. ``float()`` of the
# string reproduces the same bit pattern.
_INV_2P32 = 2.3283064365386963e-10
# fixup() boundary epsilon (RNG.c:86, used to step away from 0/1).
_I2_32M1 = 2.328306437080797e-10


class RMersenneTwister:
    """R's default RNG, reproducible bit-exactly across platforms."""

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        # ``RNG_Init``: warm 50 LCG steps, then 625 more to fill
        # ``i_seed[0..624]``. ``FixupSeeds(initial=1)`` overwrites
        # ``i_seed[0]`` with ``N=624`` so the first ``MT_genrand`` call
        # regenerates the state.
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        state = [0] * _N
        # ``state[0]`` is i_seed[0] which gets thrown away; we still need
        # to advance the LCG so the actual 624 state words match R.
        s = (69069 * s + 1) & 0xFFFFFFFF
        for j in range(_N):
            s = (69069 * s + 1) & 0xFFFFFFFF
            state[j] = s
        self._mt = state
        self._mti = _N  # force regen on first genrand

    def _genrand_int32(self) -> int:
        if self._mti >= _N:
            mt = self._mt
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
            y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
            self._mti = 0
        y = self._mt[self._mti]
        self._mti += 1
        # Tempering (RNG.c:716-719).
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def unif_rand(self) -> float:
        u = self._genrand_int32() * _INV_2P32
        # ``fixup`` (RNG.c:100-105). MT output is always in (0, 1) for
        # any non-zero state, so these branches almost never fire — but
        # they *can* fire for degenerate seeds; keep them for parity.
        if u <= 0.0:
            return 0.5 * _I2_32M1
        if (1.0 - u) <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return u

    def _rbits(self, bits: int) -> int:
        """``rbits(bits)`` — RNG.c:875-885. Returns int in [0, 2^bits)."""
        v = 0
        n = 0
        while n <= bits:
            v1 = int(self.unif_rand() * 65536)
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """``R_unif_index(dn)`` for ``Sample_kind=REJECTION`` (R 3.6+
        default). Returns int in [0, dn)."""
        if dn <= 0:
            return 0
        # ``ceil(log2(dn))``: for dn=1 use 0; otherwise ``(dn-1).bit_length()``.
        bits = (dn - 1).bit_length() if dn > 1 else 0
        while True:
            dv = self._rbits(bits)
            if dv < dn:
                return dv

    def sample_replace(self, n: int, k: int) -> np.ndarray:
        """``sample(n, k, replace=TRUE)`` — 0-based indices in [0, n)."""
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            out[i] = self.unif_index(n)
        return out

    def sample_no_replace(self, n: int, k: int) -> np.ndarray:
        """``sample(n, k, replace=FALSE)`` — 0-based indices in [0, n).

        R uses the ``replace`` branch for ``k < 2`` (no allocation), but
        the resulting draw is identical to a single ``unif_index(n)`` so
        we route k=1 through the FY path uniformly. For k >= 2 this is
        partial Fisher-Yates with swap-and-pop:

            x[0..n-1] = 0..n-1
            for i in 0..k-1:
                j = unif_index(n_remaining)
                out[i] = x[j]
                x[j] = x[--n_remaining]
        """
        if k < 0 or k > n:
            raise ValueError(f"k={k} not in [0, n={n}]")
        x = list(range(n))
        m = n
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            j = self.unif_index(m)
            out[i] = x[j]
            m -= 1
            x[j] = x[m]
        return out
