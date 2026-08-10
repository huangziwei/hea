//! METIS's random number generator.
//!
//!   * `GKlib/random.c`      → [`Rng::init`], [`Rng::randint64`], [`Rng::randint32`]
//!   * `GKlib/gk_mkrandom.h` → [`Rng::isrand`], [`Rng::irand`],
//!     [`Rng::irand_in_range`], [`Rng::irand_array_permute`]
//!   * `libmetis/util.c:21`  → [`Rng::init_random`]
//!
//! `random.c` only reaches the Mersenne Twister when `USE_GKRAND` is defined —
//! otherwise it is libc's `srand`/`rand`, which differs between platforms.
//! `CHOLMOD/Partition/cholmod_metis_wrapper.h:15-16` defines it unconditionally,
//! so the ordering CHOLMOD computes is the same everywhere, and reproducing it
//! is a well-posed target rather than a per-libc one.
//!
//! `gklib.c:39` instantiates the template as `GK_MKRANDOM(i, idx_t, idx_t)`, so
//! both the range type and the array element type are [`Idx`].

use super::Idx;

/// `random.c:59-61` — MT19937-64's parameters.
const NN: usize = 312;
const MM: usize = 156;
const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
/// Most significant 33 bits.
const UM: u64 = 0xFFFF_FFFF_8000_0000;
/// Least significant 31 bits.
const LM: u64 = 0x7FFF_FFFF;

/// The generator state. In the C this is a pair of file statics in `random.c`
/// (`mt`, `mti`), reachable from every `irand*` call; here it is owned by
/// [`super::ctrl::Ctrl`], which every call site already has in scope. What has
/// to match is the *order* of the draws, and that follows the call order.
pub struct Rng {
    mt: [u64; NN],
    /// `mti == NN + 1` means `mt` has never been seeded (`random.c:88`).
    mti: usize,
}

impl Default for Rng {
    fn default() -> Self {
        Rng {
            mt: [0; NN],
            mti: NN + 1,
        }
    }
}

impl Rng {
    /// `gk_randinit` (`random.c:101-110`).
    pub fn init(&mut self, seed: u64) {
        self.mt[0] = seed;
        for i in 1..NN {
            self.mt[i] = 6_364_136_223_846_793_005u64
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        self.mti = NN;
    }

    /// `InitRandom` (`util.c:21-24`) — the only seeding METIS does.
    /// `SetupCtrl` calls it with `ctrl->seed`, which defaults to `-1`
    /// (`options.c:81`), so in practice the seed is always 4321.
    pub fn init_random(&mut self, seed: Idx) {
        self.isrand(if seed == -1 { 4321 } else { seed });
    }

    /// `isrand` (`gk_mkrandom.h:27-30`).
    pub fn isrand(&mut self, seed: Idx) {
        self.init(seed as u64);
    }

    /// `gk_randint64` (`random.c:114-150`). Note the `& 0x7FFF...` on the way
    /// out: the result is always non-negative when reinterpreted as `i64`.
    pub fn randint64(&mut self) -> u64 {
        const MAG01: [u64; 2] = [0, MATRIX_A];

        if self.mti >= NN {
            // "if init_genrand64() has not been called, a default initial seed
            // is used" — unreachable through `METIS_NodeND`, which seeds in
            // `SetupCtrl`, but it is what the C does.
            if self.mti == NN + 1 {
                self.init(5489);
            }

            for i in 0..NN - MM {
                let x = (self.mt[i] & UM) | (self.mt[i + 1] & LM);
                self.mt[i] = self.mt[i + MM] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
            }
            for i in NN - MM..NN - 1 {
                let x = (self.mt[i] & UM) | (self.mt[i + 1] & LM);
                self.mt[i] = self.mt[i + MM - NN] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
            }
            let x = (self.mt[NN - 1] & UM) | (self.mt[0] & LM);
            self.mt[NN - 1] = self.mt[MM - 1] ^ (x >> 1) ^ MAG01[(x & 1) as usize];

            self.mti = 0;
        }

        let mut x = self.mt[self.mti];
        self.mti += 1;

        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;

        x & 0x7FFF_FFFF_FFFF_FFFF
    }

    /// `gk_randint32` (`random.c:154-160`). Never called on the `METIS_NodeND`
    /// path — `gklib.c:39`'s instantiation is 64-bit — but it is half of what
    /// `random.c` exports and the gate walks it.
    #[allow(dead_code)]
    pub fn randint32(&mut self) -> u32 {
        (self.randint64() & 0x7FFF_FFFF) as u32
    }

    /// `irand` (`gk_mkrandom.h:36-41`). The template picks the 64-bit generator
    /// when `sizeof (RNGT) > sizeof (int32_t)`, and `idx_t` is `int64_t` in the
    /// build CHOLMOD compiles (`metis.h:34`, `IDXTYPEWIDTH 64`).
    pub fn irand(&mut self) -> Idx {
        self.randint64() as Idx
    }

    /// `irandInRange` (`gk_mkrandom.h:48-51`).
    pub fn irand_in_range(&mut self, max: Idx) -> Idx {
        self.irand() % max
    }

    /// `irandArrayPermute` (`gk_mkrandom.h:59-90`).
    ///
    /// Two regimes, and they draw different numbers of randoms: below 10
    /// elements it is `n` independent swap pairs, at 10 and above it is
    /// `nshuffles` rounds that each draw twice and swap four *crossed* pairs
    /// (`p[v+0]<->p[u+2]`, not `p[v+0]<->p[u+0]` — the straight version is
    /// commented out upstream). Drawing the wrong number of randoms
    /// desynchronises the generator for everything that follows.
    pub fn irand_array_permute(&mut self, n: Idx, p: &mut [Idx], nshuffles: Idx, flag: i32) {
        if flag == 1 {
            for (i, pi) in p.iter_mut().take(n as usize).enumerate() {
                *pi = i as Idx;
            }
        }

        if n < 10 {
            for _ in 0..n {
                let v = self.irand_in_range(n) as usize;
                let u = self.irand_in_range(n) as usize;
                p.swap(v, u);
            }
        } else {
            for _ in 0..nshuffles {
                let v = self.irand_in_range(n - 3) as usize;
                let u = self.irand_in_range(n - 3) as usize;
                p.swap(v, u + 2);
                p.swap(v + 1, u + 3);
                p.swap(v + 2, u);
                p.swap(v + 3, u + 1);
            }
        }
    }
}
