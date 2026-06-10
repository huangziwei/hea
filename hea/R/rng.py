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

import numpy as np

__all__ = ["RMersenneTwister"]

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


class RMersenneTwister:
    """R's default RNG after ``set.seed(seed)``, bit-exact and
    platform-independent."""

    __slots__ = ("_mt", "_buf", "_pos")

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """R's ``set.seed(seed)`` (Mersenne-Twister kind)."""
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
        y = (mt[:_N - 1] & _UPPER) | (mt[1:] & _LOWER)
        mag = np.where((y & np.uint32(1)) != 0, _MATRIX_A, np.uint32(0))
        yshift = (y >> np.uint32(1)) ^ mag
        new = np.empty(_N, dtype=np.uint32)
        new[:_N - _M] = mt[_M:] ^ yshift[:_N - _M]
        new[_N - _M:2 * (_N - _M)] = (new[:_N - _M]
                                      ^ yshift[_N - _M:2 * (_N - _M)])
        new[2 * (_N - _M):_N - 1] = (new[_N - _M:_M - 1]
                                     ^ yshift[2 * (_N - _M):])
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
        if n is None:
            if self._pos >= self._buf.size:
                self._refill()
            v = float(self._buf[self._pos])
            self._pos += 1
            return v
        out = np.empty(int(n))
        for i in range(int(n)):
            out[i] = self.unif_rand()
        return out

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
