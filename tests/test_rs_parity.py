"""Bit-exact parity gate: the compiled ``hea._rs`` (Rust) d/p/q kernels must
equal **R** bit-for-bit, compared against live R on the *same machine*.

Why live R, not the Python ``nmath`` reference or committed pins:

* The Rust kernels and R both evaluate transcendentals through the platform's
  scalar libm. On macOS (Apple libm) that is 0-ulp; on Linux glibc the
  transcendental-heavy kernels (pbeta/qbeta/pgamma/…) drift a few ulp because
  Rust-std-math and R's glibc path don't agree bit-for-bit even on one machine
  (the cross-platform libm floor — 0-ulp to an arbitrary R is not promisable).
  So the strict 0-ulp form runs on macOS (Intel & arm64); off-macOS (Linux, incl.
  CI when R is installed) the same cases run as a *tolerance* check — see
  ``_STRICT`` below — which still catches gross / platform-specific Rust bugs.
* ``numpy``'s vectorized transcendentals are NOT bit-identical to scalar libm
  and drift by a few ulp across numpy builds — so the old ``rs == nmath``
  differential gate failed on some Linux/numpy combinations even though Rust was
  the *correct* (R-faithful) side. The pure-Python ``nmath`` path is a fallback
  slated for deprecation; it is no longer the oracle.
* Committed R pins can't be cross-platform bit-exact either (glibc-R differs
  from Apple-libm-R at the last ulp), so the reference is regenerated live.

R exposes only the public d/p/q surface; the internal saddlepoint primitives
(``bd0``/``stirlerr``/``ebd0``/``dpois_raw``/``dbinom_raw``/``pow1p``) have no
R entry point and are covered transitively (``dpois``/``dbinom``/``pgamma`` in
the large-count/large-shape regime exercise the whole chain).

Skips when ``hea._rs`` isn't compiled (sdist / no toolchain) or when ``Rscript``
is absent; off-macOS it relaxes to a tolerance check (the libm floor) not 0-ulp.
"""
import sys

import numpy as np
import pytest

from conftest import have_rscript, run_rs_r_oracle

rs = pytest.importorskip("hea._rs")

if not have_rscript():
    pytest.skip("Rscript not on PATH (install R)", allow_module_level=True)

# Bit-exactness to R holds only where hea and R share the platform's scalar libm:
# macOS (Apple libm) → 0-ulp on BOTH Intel & arm64. On Linux/glibc the
# transcendental-heavy kernels drift a few ulp (Rust-std-math vs R's glibc path —
# see the module docstring), so off-macOS this runs as a TOLERANCE check: it still
# catches gross / platform-specific Rust regressions (notably the x86-64 `rfma`
# plain-path, otherwise only gated on a local Intel Mac), just not at the last ulp.
_STRICT = sys.platform == "darwin"
# DIAGNOSTIC: rtol=0 makes each off-macOS kernel report its "Max relative difference"
# (the true glibc libm floor) in the CI log — read those, then set the calibrated
# tolerance (~2–4× the worst kernel, or an ULP-count bound). Intentionally turns the
# Linux job RED for this one measurement run.
_LINUX_RTOL = 0.0


def _bits(v: float) -> int:
    return np.float64(v).view(np.int64).item()


def _assert_bit_exact(got, exp):
    """Bit-for-bit (0-ulp) equality, NaN-aware and sign-of-zero-agnostic.

    ±0.0 are the same real number; for a probability/density/quantile the sign
    bit on a zero result is an arithmetic byproduct (R and nmath disagree on it
    in a few log_p / zero-quantile cases) carrying no numerical meaning, so the
    gate treats them as equal — exactly how ulp-equality is conventionally
    defined. Every non-zero value is still required to match R bit-for-bit.
    """
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    assert got.shape == exp.shape, f"shape {got.shape} != {exp.shape}"
    if not _STRICT:
        # Off-macOS: glibc libm floor → a few-ulp tolerance (NaN/±Inf-aware;
        # atol covers the underflow corner where R rounds to 0).
        np.testing.assert_allclose(got, exp, rtol=_LINUX_RTOL, atol=1e-300,
                                   equal_nan=True)
        return
    for g, e in zip(got.ravel(), exp.ravel()):
        if np.isnan(e):
            assert np.isnan(g), f"expected NaN, got {g!r}"
        elif g == 0.0 and e == 0.0:
            continue  # +0.0 == -0.0 as real numbers
        else:
            assert _bits(g) == _bits(e), f"bit mismatch: rs={g!r} R={e!r}"


# ---------------------------------------------------------------------------
# Input grids — chosen to stress every internal branch of each kernel.
# ---------------------------------------------------------------------------
def _norm_grid() -> np.ndarray:
    # pnorm_both branches: central, mid, far tail, the log_p/tail cutoffs,
    # tiny, zero, non-finite.
    return np.array([
        -50.0, -40.0, -38.4674, -8.2924, -5.657, -5.0, -1.0, -0.6744,
        -1e-8, -1e-300, 0.0, 1e-300, 1e-8, 0.5, 0.6744, 0.67448975, 1.0,
        5.0, 5.657, 8.2924, 38.0, 40.0, 50.0, 1e170, 1e171,
        np.inf, -np.inf, np.nan,
    ], dtype=float)


_PG_X = np.array([0.5, 0.001, 5.0, 20.0, 50.0, 100.0, 2.0, 1e5, 0.3, 1000.0,
                  1e-8, 3.0, 15.0, 0.0, np.inf, 1e-300])
_PG_A = np.array([2.0, 0.5, 20.0, 5.0, 50.0, 2.0, 100.0, 1e4, 0.2, 1000.0,
                  1.0, 3.0, 7.0, 2.0, 2.0, 0.5])


def _beta_grid():
    ab = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]
    xs = [1e-8, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1 - 1e-9]
    X, A, B = [], [], []
    for a in ab:
        for b in ab:
            for x in xs:
                X.append(x)
                A.append(a)
                B.append(b)
    return np.array(X), np.array(A), np.array(B)


def _qbeta_grid():
    pq = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
    al = [1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999, 1 - 1e-10]
    AL, P, Q = [], [], []
    for p in pq:
        for q in pq:
            for a in al:
                AL.append(a)
                P.append(p)
                Q.append(q)
    return np.array(AL), np.array(P), np.array(Q)


# ---------------------------------------------------------------------------
# Cases: (name, rs/R kernel name, [inputs in hea order], [trailing flag bools]).
# rs is called ``getattr(rs, fn)(*inputs, *flags)``; the R oracle calls the
# matching ``hea_<fn>`` wrapper (tests/scripts/nmath_r_oracle.R).
# ---------------------------------------------------------------------------
def _build_cases():
    C = []

    def add(name, fn, arrays, flags=()):
        C.append((name, fn, [np.asarray(a, dtype=float) for a in arrays],
                  list(flags)))

    # --- normal: pnorm / qnorm / dnorm ---
    g = _norm_grid()
    for mu, sigma in [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5)]:
        for lt in (True, False):
            for lp in (True, False):
                add(f"pnorm_{mu}_{sigma}_{lt}_{lp}", "pnorm",
                    [g, np.full_like(g, mu), np.full_like(g, sigma)], (lt, lp))
    xd = np.array([-1.0, 0.0, 1.0, 2.0])
    for sig in (0.0, -1.0):
        for lt in (True, False):
            for lp in (True, False):
                add(f"pnorm_deg_{sig}_{lt}_{lp}", "pnorm",
                    [xd, np.zeros_like(xd), np.full_like(xd, sig)], (lt, lp))
    p_log = np.array([-700., -300., -100., -50., -10., -1., -0.5, -1e-3,
                      -1e-8, np.log(0.5)])
    p_lin = np.array([1e-300, 1e-50, 1e-10, 1e-4, 0.001, 0.01, 0.1, 0.25, 0.5,
                      0.75, 0.9, 0.99, 0.999, 1 - 1e-10, 1 - 1e-16, 0.0, 1.0])
    for lt in (True, False):
        for lp in (True, False):
            p = p_log if lp else p_lin
            add(f"qnorm_{lt}_{lp}", "qnorm",
                [p, np.zeros_like(p), np.ones_like(p)], (lt, lp))
    xn = np.array([-40, -6, -5, -1, 0, 0.5, 1, 5, 6, 38, 40,
                   np.inf, -np.inf, np.nan], dtype=float)
    for gl in (True, False):
        add(f"dnorm_{gl}", "dnorm",
            [xn, np.zeros_like(xn), np.ones_like(xn)], (gl,))

    # --- lgamma / gamma ---
    xg = np.array([-9.3, -2.5, -0.5, -1e-8, 1e-307, 1e-200, 0.1, 0.5, 1.0, 1.5,
                   2.0, 3.7, 9.99, 10.0, 10.5, 50.0, 100.0, 4934721.0, 1e17,
                   1e18, 170.0, -170.5, np.nan, np.inf])
    add("lgammafn", "lgammafn", [xg])
    xg2 = xg[(np.abs(xg) < 171) | ~np.isfinite(xg)]
    add("gammafn", "gammafn", [xg2])

    # --- gamma: pgamma / dgamma / qgamma ---
    for scale in (1.0, 2.5):
        for lt in (True, False):
            for lp in (True, False):
                add(f"pgamma_{scale}_{lt}_{lp}", "pgamma",
                    [_PG_X, _PG_A, np.full_like(_PG_X, scale)], (lt, lp))
    gx = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 0.001, 100.0, 1e-8])
    gsh = np.array([0.5, 2.0, 1.0, 3.0, 0.7, 0.5, 50.0, 2.0])
    for gl in (True, False):
        add(f"dgamma_{gl}", "dgamma", [gx, gsh, np.ones_like(gx)], (gl,))
    qgp = np.array([1e-10, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9999, 1 - 1e-12])
    for alpha in (0.5, 2.0, 50.0, 1e-11):
        add(f"qgamma_{alpha}", "qgamma",
            [qgp, np.full_like(qgp, alpha), np.ones_like(qgp)], (True, False))

    # --- beta: pbeta / qbeta / lbeta ---
    Xb, Ab, Bb = _beta_grid()
    for lt in (True, False):
        for lp in (True, False):
            add(f"pbeta_{lt}_{lp}", "pbeta", [Xb, Ab, Bb], (lt, lp))
    ab = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]
    add("lbeta", "lbeta",
        [np.array([x for x in ab for _ in ab]),
         np.array([y for _ in ab for y in ab])])
    ALq, Pq, Qq = _qbeta_grid()
    for lt in (True, False):
        add(f"qbeta_{lt}", "qbeta", [ALq, Pq, Qq], (lt, False))
        add(f"qbeta_log_{lt}", "qbeta", [np.log(ALq), Pq, Qq], (lt, True))

    # --- t / F ---
    xt = np.array([-50, -3, -1, -0.1, 0, 0.1, 1, 3, 50, 1e8], dtype=float)
    nt = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 0.5, 0.7, 100.0, 1e21, 3.0])
    for lt in (True, False):
        add(f"pt_{lt}", "pt", [xt, nt], (lt, False))
    for gl in (True, False):
        add(f"dt_{gl}", "dt", [xt, nt], (gl,))
    tp = np.array([1e-8, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999])
    nt2 = np.array([1.0, 2.0, 2.0, 5.0, 0.5, 10.0, 30.0, 0.7, 100.0, 3.0])
    for lt in (True, False):
        add(f"qt_{lt}", "qt", [tp, nt2], (lt, False))
    xf = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 0.3, 3.0, 100.0])
    f1 = np.array([1.0, 2.0, 5.0, 10.0, 3.0, 1.0, 8.0, 0.5, 1e6, 4.0])
    f2 = np.array([1.0, 10.0, 5.0, 2.0, 30.0, 1e6, 8.0, 3.0, 5.0, 20.0])
    for lt in (True, False):
        add(f"pf_{lt}", "pf", [xf, f1, f2], (lt, False))
    fp = np.array([1e-8, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999])
    qf1 = np.array([1.0, 2.0, 5.0, 10.0, 3.0, 4e5 + 1, 8.0, 0.5, 4.0, 1.0])
    qf2 = np.array([1.0, 10.0, 5.0, 2.0, 30.0, 8.0, 4e5 + 1, 3.0, 20.0, 1e6])
    for lt in (True, False):
        add(f"qf_{lt}", "qf", [fp, qf1, qf2], (lt, False))

    # --- discrete: ppois / qpois / pbinom / qbinom / dpois / dbinom / dbeta ---
    px = np.array([0., 1., 2., 5., 10., 50., 3., 7., 0., 100.])
    pl = np.array([3., 3., 3., 4.5, 10., 40., 0.5, 7., 0., 1e3])
    for lt in (True, False):
        add(f"ppois_{lt}", "ppois", [px, pl], (lt, False))
    qpp = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.5, 0.3, 1 - 1e-9])
    qpl = np.array([3., 3., 10., 4.5, 40., 100., 1e3, 0.5, 7., 5.])
    for lt in (True, False):
        add(f"qpois_{lt}", "qpois", [qpp, qpl], (lt, False))
    bx = np.array([0., 5., 10., 3., 20., 7., 50., 0., 15., 2.])
    bn = np.array([20., 20., 20., 20., 20., 10., 100., 5., 30., 2.])
    bp = np.array([.3, .3, .3, .5, 1., .7, .1, .5, .4, .5])
    for lt in (True, False):
        add(f"pbinom_{lt}", "pbinom", [bx, bn, bp], (lt, False))
    qbp = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.5, 0.3, 1 - 1e-9])
    qbn = np.array([20., 20., 50., 20., 100., 1000., 500., 10., 30., 5.])
    qbpr = np.array([.3, .5, .3, .5, .1, .5, .7, .5, .4, .5])
    for lt in (True, False):
        add(f"qbinom_{lt}", "qbinom", [qbp, qbn, qbpr], (lt, False))
    dbex = np.array([0., 0.01, 0.1, 0.5, 0.9, 0.99, 1., 0.3, 0.5, 0.7])
    dbea = np.array([2., 0.5, 2., 3., 0.7, 5., 2., 1., 3., 0.5])
    dbeb = np.array([3., 2., 0.5, 3., 2., 1., 0.7, 1., 5., 0.5])
    for gl in (True, False):
        add(f"dpois_{gl}", "dpois", [px, pl], (gl,))
        add(f"dbinom_{gl}", "dbinom", [bx, bn, bp], (gl,))
        add(f"dbeta_{gl}", "dbeta", [dbex, dbea, dbeb], (gl,))

    # --- saddlepoint regime (exercises bd0/stirlerr/ebd0/dpois_wrap) ---
    kk = np.arange(120.0, 260.0)
    lams = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    add("dpois_saddle", "dpois",
        [np.repeat(kk, lams.size), np.tile(lams, kk.size)], (False,))
    xsp = np.linspace(0.5, 400.0, 400)
    for shape in (20.0, 50.0, 100.0, 200.0):
        add(f"pgamma_saddle_{shape}", "pgamma",
            [xsp, np.full_like(xsp, shape), np.ones_like(xsp)], (True, False))
    psp = np.linspace(1e-4, 1 - 1e-4, 400)
    for alpha in (50.0, 100.0, 200.0):
        add(f"qgamma_saddle_{alpha}", "qgamma",
            [psp, np.full_like(psp, alpha), np.ones_like(psp)], (True, False))

    # --- exponential (hea/nmath parameterise by scale; R by rate=1/scale) ---
    xe = np.array([0., 0.1, 1., 5., 20., 0., 2., 100., 0.5, 1e-8])
    se = np.array([1., 1., 2., 0.5, 1., 3., 1., 2., 0.7, 1.])
    for gl in (True, False):
        add(f"dexp_{gl}", "dexp", [xe, se], (gl,))
    qpe = np.array([1e-8, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.3, 0.7, 1 - 1e-9])
    for lt in (True, False):
        add(f"pexp_{lt}", "pexp", [xe, se], (lt, False))
        add(f"qexp_{lt}", "qexp", [qpe, se], (lt, False))

    return C


CASES = _build_cases()


@pytest.fixture(scope="session")
def r_oracle(tmp_path_factory):
    """Run R once for the whole module; return ``{case name: R values}``."""
    workdir = tmp_path_factory.mktemp("rs_r_oracle")
    return run_rs_r_oracle(CASES, workdir)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rs_matches_r(case, r_oracle):
    name, fn, arrays, flags = case
    got = getattr(rs, fn)(*arrays, *flags)
    _assert_bit_exact(got, r_oracle[name])
