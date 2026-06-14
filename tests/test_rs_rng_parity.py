"""3-way bit-exact parity gate for ``hea.R.rng`` — Rust vs pure-Python vs **R**.

Every case draws the identical Mersenne-Twister stream three ways:

* ``RMersenneTwister(seed)`` — the Rust path (``hea._rs.RsMt``) when the
  extension is present;
* ``RMersenneTwister(seed, force_py=True)`` — the pure-Python reference;
* live R (``set.seed(seed); <rcall>``, tests/scripts/rng_r_oracle.R).

All three must agree bit-for-bit. R and the Rust/Python kernels go through the
same scalar libm, so on macOS this is 0-ulp with no committed pins (same
rationale as tests/test_rs_parity.py); the gate is macOS-only and needs R.

This gate is the one that found (and the fix that keeps pinned): rng.py's
``rbinom`` used ``q ** n`` (libm pow) where R uses ``R_pow_di`` (integer power),
diverging by up to ~200 ulp in ``qn`` — enough to flip an inversion draw.
"""
import sys

import numpy as np
import pytest

from conftest import have_rscript, run_rng_r_oracle

rs = pytest.importorskip("hea._rs")
from hea.R.rng import RGenerator, RMersenneTwister  # noqa: E402

if sys.platform != "darwin":
    pytest.skip("RNG Rust/Python/R 0-ulp gate is macOS-only (glibc libm diverges "
                "a few ulp from R); the Linux matrix covers it via pinned R tests.",
                allow_module_level=True)
if not have_rscript():
    pytest.skip("Rscript not on PATH (install R)", allow_module_level=True)

SEEDS = (1, 42, 4357)
_g = np.random.default_rng(20260614)


def _bits(v):
    return np.float64(v).view(np.int64).item()


def _assert_bit_exact(got, exp, label):
    got = np.atleast_1d(np.asarray(got, dtype=float))
    exp = np.atleast_1d(np.asarray(exp, dtype=float))
    assert got.shape == exp.shape, f"{label}: shape {got.shape} != {exp.shape}"
    for i, (g, e) in enumerate(zip(got.ravel(), exp.ravel())):
        if np.isnan(e):
            assert np.isnan(g), f"{label}[{i}]: expected NaN, got {g!r}"
        elif g == 0.0 and e == 0.0:
            continue
        else:
            assert _bits(g) == _bits(e), f"{label}[{i}]: {g!r} != R {e!r}"


def _tile(values, reps):
    return np.asarray(values * reps, dtype=float)


def _build_cases():
    """Each case: (name, rcall, [param arrays], py_draw(mt)->ndarray)."""
    C = []

    def add(name, rcall, params, py):
        C.append((name, rcall, [np.asarray(p, float) for p in params], py))

    # --- uniforms / normals / exponential -----------------------------------
    add("runif", "runif(4000)", [], lambda mt: mt.unif_rand(4000))
    add("rnorm", "rnorm(4000)", [], lambda mt: mt.rnorm(4000))
    add("rnorm_scaled", "rnorm(4000, 2.5, 0.5)", [],
        lambda mt: mt.rnorm(4000, 2.5, 0.5))
    add("rexp", "rexp(4000)", [],
        lambda mt: np.array([mt.exp_rand() for _ in range(4000)]))

    # --- rpois: runs of equal mu (exercises R's static cache) + pure-vary ----
    mu_inv = np.repeat([0.5, 2.0, 5.0, 9.0, 9.9], 300)         # <10 inversion
    mu_rej = np.repeat([10.0, 15.0, 40.0, 200.0, 1000.0], 300)  # >=10 rejection
    mu_mix = np.concatenate([mu_inv, mu_rej])
    add("rpois", "rpois(length(p0), p0)", [mu_mix],
        lambda mt, m=mu_mix: np.array([mt.rpois(x) for x in m]))
    mu_vary = _g.uniform(0.2, 60.0, 1500)
    add("rpois_vary", "rpois(length(p0), p0)", [mu_vary],
        lambda mt, m=mu_vary: np.array([mt.rpois(x) for x in m]))

    # --- rbinom: BINV (small n*p) + BTPE (large n*p) -------------------------
    sizes = [2, 5, 20, 40, 100, 500, 1000]
    probs = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    bsz = np.array([s for s in sizes for _ in probs] * 25, float)
    bpr = np.array([p for _ in sizes for p in probs] * 25, float)
    add("rbinom", "rbinom(length(p0), p0, p1)", [bsz, bpr],
        lambda mt, s=bsz, p=bpr: np.array([mt.rbinom(a, b) for a, b in zip(s, p)]))
    # all-BINV stress (the q**n vs R_pow_di path): many small binomials.
    isz = np.array([3, 5, 7, 9, 11, 13, 17, 21, 25, 29] * 300, float)
    ipr = np.array([0.5] * isz.size, float)
    add("rbinom_binv", "rbinom(length(p0), p0, p1)", [isz, ipr],
        lambda mt, s=isz, p=ipr: np.array([mt.rbinom(a, b) for a, b in zip(s, p)]))

    # --- rgamma: GS (a<1) + GD (a>=1) across the b/si/c regime splits --------
    gsh_l = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.5, 5.0, 10.0, 13.0, 13.5,
             20.0, 50.0]
    gsc_l = [1.0, 2.5]
    gsh = np.array([s for s in gsh_l for _ in gsc_l] * 25, float)
    gsc = np.array([c for _ in gsh_l for c in gsc_l] * 25, float)
    add("rgamma", "rgamma(length(p0), p0, scale=p1)", [gsh, gsc],
        lambda mt, s=gsh, c=gsc: np.array([mt.rgamma(a, b) for a, b in zip(s, c)]))

    # --- rnbinom (Poisson-Gamma) --------------------------------------------
    nbz = np.array([s for s in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
                    for _ in [0.5, 2.0, 10.0, 50.0]] * 30, float)
    nbm = np.array([m for _ in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
                    for m in [0.5, 2.0, 10.0, 50.0]] * 30, float)
    add("rnbinom", "rnbinom(length(p0), size=p0, mu=p1)", [nbz, nbm],
        lambda mt, s=nbz, m=nbm: np.array([mt.rnbinom(a, b) for a, b in zip(s, m)]))

    # --- rchisq central + noncentral ----------------------------------------
    cdf = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0] * 100, float)
    add("rchisq", "rchisq(length(p0), p0)", [cdf],
        lambda mt, d=cdf: np.array([mt.rchisq(x) for x in d]))
    ncdf = np.array([d for d in [1.0, 2.0, 5.0, 10.0] for _ in [0.5, 2.0, 8.0]] * 80,
                    float)
    ncp = np.array([c for _ in [1.0, 2.0, 5.0, 10.0] for c in [0.5, 2.0, 8.0]] * 80,
                   float)
    add("rchisq_nc", "rchisq(length(p0), p0, ncp=p1)", [ncdf, ncp],
        lambda mt, d=ncdf, c=ncp: np.array([mt.rchisq(a, b) for a, b in zip(d, c)]))

    # --- rt / rf -------------------------------------------------------------
    tdf = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 0.5] * 120, float)
    add("rt", "rt(length(p0), p0)", [tdf],
        lambda mt, d=tdf: np.array([mt.rt(x) for x in d]))
    f1 = np.array([a for a in [1.0, 5.0, 30.0, 100.0] for _ in [2.0, 10.0, 50.0]] * 60,
                  float)
    f2 = np.array([b for _ in [1.0, 5.0, 30.0, 100.0] for b in [2.0, 10.0, 50.0]] * 60,
                  float)
    add("rf", "rf(length(p0), p0, p1)", [f1, f2],
        lambda mt, a=f1, b=f2: np.array([mt.rf(x, y) for x, y in zip(a, b)]))

    # --- rbeta: BC (min<=1) + BB (min>1) ------------------------------------
    ba = np.array([a for a in [0.3, 0.5, 1.0, 2.0, 5.0] for _ in [0.5, 1.0, 3.0]] * 60,
                  float)
    bb = np.array([b for _ in [0.3, 0.5, 1.0, 2.0, 5.0] for b in [0.5, 1.0, 3.0]] * 60,
                  float)
    add("rbeta", "rbeta(length(p0), p0, p1)", [ba, bb],
        lambda mt, a=ba, b=bb: np.array([mt.rbeta(x, y) for x, y in zip(a, b)]))

    # --- sample.int (unweighted): shrinking-pool / replace / permutation -----
    add("sample_norep", "sample.int(2000, 800)", [],
        lambda mt: mt.sample_int(2000, 800, False) + 1)
    add("sample_rep", "sample.int(60, 1000, replace=TRUE)", [],
        lambda mt: mt.sample_int(60, 1000, True) + 1)
    add("sample_perm", "sample.int(1500)", [],
        lambda mt: mt.sample_int(1500, 1500, False) + 1)

    # --- weighted sample (ProbSample{No,}Replace + Walker alias) -------------
    w = _g.uniform(0.0, 1.0, 500)
    w[::7] = 0.0                                  # some zero weights
    add("sample_prob_norep", "sample.int(500, 100, prob=p0)", [w],
        lambda mt, p=w: mt.sample_prob(p, 100, False) + 1)
    add("sample_prob_walker", "sample.int(500, 800, replace=TRUE, prob=p0)", [w],
        lambda mt, p=w: mt.sample_prob(p, 800, True) + 1)
    w2 = np.zeros(500)
    w2[:40] = _g.uniform(0.1, 1.0, 40)            # <200 sizeable ⇒ non-walker
    add("sample_prob_rep", "sample.int(500, 800, replace=TRUE, prob=p0)", [w2],
        lambda mt, p=w2: mt.sample_prob(p, 800, True) + 1)

    # --- RGenerator facade (the family `$rd` batch path) vs R -----------------
    rg_mu = np.repeat([0.5, 3.0, 12.0, 80.0], 400).astype(float)
    add("rgen_poisson", "rpois(length(p0), p0)", [rg_mu],
        lambda mt, m=rg_mu: RGenerator(mt).poisson(lam=m))
    rg_sh = np.repeat([0.5, 2.0, 7.0, 20.0], 300).astype(float)
    rg_sc = np.tile([1.0, 2.0], 600).astype(float)
    add("rgen_gamma", "rgamma(length(p0), p0, scale=p1)", [rg_sh, rg_sc],
        lambda mt, s=rg_sh, c=rg_sc: RGenerator(mt).gamma(shape=s, scale=c))
    rg_n = np.repeat([5, 30, 100, 800], 250).astype(float)
    rg_p = np.tile([0.2, 0.5, 0.8, 0.95], 250).astype(float)
    add("rgen_binomial", "rbinom(length(p0), p0, p1)", [rg_n, rg_p],
        lambda mt, n=rg_n, p=rg_p: RGenerator(mt).binomial(n=n, p=p))
    rg_df = np.repeat([1.0, 5.0, 30.0], 400).astype(float)
    add("rgen_t", "rt(length(p0), p0)", [rg_df],
        lambda mt, d=rg_df: RGenerator(mt).standard_t(df=d))

    # --- remaining batch methods (rf_n/rchisq_n/rbeta_n/rnbinom_n/exp_rand_n) -
    bf1 = np.repeat([1.0, 5.0, 30.0], 300).astype(float)
    bf2 = np.tile([2.0, 10.0, 50.0], 300).astype(float)
    add("rf_n", "rf(length(p0), p0, p1)", [bf1, bf2],
        lambda mt, a=bf1, b=bf2: mt.rf_n(a, b))
    bcdf = np.repeat([1.0, 5.0, 10.0], 300).astype(float)
    bncp = np.tile([0.0, 2.0, 8.0], 300).astype(float)
    add("rchisq_n", "rchisq(length(p0), p0, ncp=p1)", [bcdf, bncp],
        lambda mt, d=bcdf, c=bncp: mt.rchisq_n(d, c))
    bba = np.repeat([0.5, 2.0, 5.0], 300).astype(float)
    bbb = np.tile([0.5, 1.0, 3.0], 300).astype(float)
    add("rbeta_n", "rbeta(length(p0), p0, p1)", [bba, bbb],
        lambda mt, a=bba, b=bbb: mt.rbeta_n(a, b))
    bnz = np.repeat([1.0, 5.0, 20.0], 300).astype(float)
    bnm = np.tile([0.5, 5.0, 30.0], 300).astype(float)
    add("rnbinom_n", "rnbinom(length(p0), size=p0, mu=p1)", [bnz, bnm],
        lambda mt, s=bnz, m=bnm: mt.rnbinom_n(s, m))
    add("exp_rand_n", "rexp(3000)", [],
        lambda mt: mt.exp_rand_n(3000))

    return C


CASES = _build_cases()


@pytest.fixture(scope="module")
def r_oracles(tmp_path_factory):
    """{seed: {case name: R values}} — one Rscript invocation per seed."""
    out = {}
    for seed in SEEDS:
        workdir = tmp_path_factory.mktemp(f"rng_r_oracle_{seed}")
        out[seed] = run_rng_r_oracle(seed, CASES, workdir)
    return out


def test_rust_path_is_active():
    """Guard: with the extension enabled, RMersenneTwister must use it (no silent
    fallback). Under HEA_NO_RS the pure-Python path is intentional, so skip."""
    if RMersenneTwister(SEEDS[0])._impl is None:
        pytest.skip("hea._rs disabled (HEA_NO_RS) — pure-Python path is intentional")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rng_3way_bit_exact(case, seed, r_oracles):
    name, rcall, params, py = case
    r_vals = r_oracles[seed][name]
    rust_vals = py(RMersenneTwister(seed))                  # Rust path
    py_vals = py(RMersenneTwister(seed, force_py=True))     # pure-Python path
    _assert_bit_exact(rust_vals, r_vals, f"{name}/rust-vs-R/seed={seed}")
    _assert_bit_exact(py_vals, r_vals, f"{name}/python-vs-R/seed={seed}")
