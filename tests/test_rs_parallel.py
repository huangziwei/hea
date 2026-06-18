"""rayon parallel-path determinism gate.

Every vectorized ``hea._rs`` d/p/q kernel maps an element-wise function over its
inputs. Above ``PAR_THRESHOLD`` (2048, see ``rust/src/par.rs``) the map runs on
rayon worker threads under ``py.allow_threads``; below it, serially. The two
branches share the *exact* per-element float ops and rayon's indexed ``collect``
preserves order, so they MUST be bit-for-bit equal — this test locks that in.

Unlike ``test_rs_parity`` (Rust == R, libm-dependent → macOS-only), this is an
internal invariant of the crate, so it runs on **every** platform / in CI: it is
what guards the parallel path on Linux, where the strict R gate is skipped.
"""
import numpy as np
import pytest

rs = pytest.importorskip("hea._rs")

# N is above PAR_THRESHOLD (2048) → parallel branch; CHUNK is below it → serial
# branch. Margins on both sides keep both branches exercised if the threshold is
# retuned modestly.
N = 3000
CHUNK = 512


def _build_cases():
    g = np.random.default_rng(0)
    n = N
    xp = g.uniform(0.1, 20.0, n)       # x > 0
    al = g.uniform(0.5, 8.0, n)        # shape / a
    sc = g.uniform(0.5, 3.0, n)        # scale
    p01 = g.uniform(1e-4, 1 - 1e-4, n)
    xr = g.normal(0, 3, n)             # real line
    df = g.uniform(1.0, 40.0, n)
    df2 = g.uniform(1.0, 40.0, n)
    bb = g.uniform(0.5, 8.0, n)
    lam = g.uniform(0.5, 50.0, n)
    nb = g.integers(1, 200, n).astype(float)
    kb = np.round(g.random(n) * nb)
    pb = g.uniform(0.05, 0.95, n)
    sx = g.uniform(-0.4, 0.4, n)       # small |x| for pow1p
    Z = np.zeros(n)
    ONE = np.ones(n)
    # (rs name, [inputs in native order], (trailing flag bools))
    return [
        ("pnorm", [xr, Z, ONE], (True, False)),
        ("qnorm", [p01, Z, ONE], (True, False)),
        ("dnorm", [xr, Z, ONE], (False,)),
        ("lgammafn", [al], ()),
        ("gammafn", [al], ()),
        ("stirlerr", [al], ()),
        ("bd0", [xp, al], ()),
        ("pow1p", [sx, df], ()),
        ("dpois_raw", [kb, lam], (True,)),
        ("dbinom_raw", [kb, nb, pb, 1.0 - pb], (True,)),
        ("pgamma", [xp, al, sc], (True, False)),
        ("dgamma", [xp, al, sc], (False,)),
        ("qgamma", [p01, al, sc], (True, False)),
        ("pbeta", [p01, al, bb], (True, False)),
        ("lbeta", [al, bb], ()),
        ("qbeta", [p01, al, bb], (True, False)),
        ("ppois", [kb, lam], (True, False)),
        ("dpois", [kb, lam], (False,)),
        ("qpois", [p01, lam], (True, False)),
        ("pbinom", [kb, nb, pb], (True, False)),
        ("dbinom", [kb, nb, pb], (False,)),
        ("qbinom", [p01, nb, pb], (True, False)),
        ("dbeta", [p01, al, bb], (False,)),
        ("pt", [xr, df], (True, False)),
        ("qt", [p01, df], (True, False)),
        ("dt", [xr, df], (False,)),
        ("pf", [xp, df, df2], (True, False)),
        ("qf", [p01, df, df2], (True, False)),
        ("dexp", [xp, sc], (False,)),
        ("pexp", [xp, sc], (True, False)),
        ("qexp", [p01, sc], (True, False)),
    ]


CASES = _build_cases()


def _call(fn, arrs, fl, sl):
    return fn(*[np.ascontiguousarray(a[sl]) for a in arrs], *fl)


def _serial(fn, arrs, fl):
    n = arrs[0].size
    return np.concatenate(
        [_call(fn, arrs, fl, slice(i, i + CHUNK)) for i in range(0, n, CHUNK)]
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_parallel_matches_serial(case):
    name, arrs, fl = case
    fn = getattr(rs, name)
    par = _call(fn, arrs, fl, slice(None))   # N=3000 → parallel branch
    ser = _serial(fn, arrs, fl)              # CHUNK=512 → serial branch
    # Same Rust kernel both ways → identical output bits (incl. sign-of-zero and
    # NaN payloads), so a raw byte compare is the right 0-ulp check here.
    assert par.tobytes() == ser.tobytes(), f"{name}: parallel != serial"
