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

from conftest import have_rscript, r_scalar_values, run_rs_r_oracle

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
# Calibrated from the diagnostic CI run (glibc, Python 3.10–3.14): off-macOS, 14
# kernels drift and the rest are 0-ulp; the worst is ~1.8e-15 (≈8 ulp — pgamma_saddle
# / pbeta). 1e-14 sits ~5× above that floor: it absorbs the glibc libm floor + numpy-
# build variation, while still catching any real Rust regression (those are ≫1e-14).
_LINUX_RTOL = 1e-14


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

    # --- FMA-contraction / platform-libm materialization points --------------
    # R's CRAN build (clang, default -ffp-contract=on) fuses `a*b + c` written
    # in one C expression to fmadd on arm64, and links platform-libm
    # __sinpi/__tanpi/lgamma in cospi.c / stirlerr.c:120 / lbeta.c:76. Each
    # point below materialized a last-ulp divergence until the port mirrored
    # the exact contraction/symbol; keep them pinned against live R.
    # bpser `z = a*log(x) - betaln(a,b)` — the original traced point, plus
    # bfrac/basym/bgrat census hits at large shapes.
    add("fma_pbeta_log_upper", "pbeta",
        [np.array([0.9852216748768474, 0.08636146016681467,
                   0.2903716807336236]),
         np.array([2.0, 9.589203880616369, 2855.5232943161823]),
         np.array([100001.0, 7819.912855463742, 13825.861560595382])],
        (False, True))
    add("fma_pbeta_plain", "pbeta",
        [np.array([0.3678134730867635, 0.6511511075950023]),
         np.array([8739.554082286646, 1124.8701718194927]),
         np.array([8682.928425324666, 3.230859039365317])],
        (True, False))
    # pd_lower_series `sum += term * f` (pgamma.c:498) via the qgamma Newton.
    add("fma_qgamma_upper", "qgamma",
        [np.array([0.4355139643795751, 0.39636761591153125,
                   0.13513636327126444]),
         np.array([2.6669191376302344, 2.536168158840434,
                   2.502400760154327]),
         np.ones(3)], (False, False))
    # gammafn/lgammafn negative x — platform __sinpi (cospi.c HAVE___SINPI).
    add("fma_gammafn_sinpi", "gammafn",
        [np.array([-77.24812758594393, -22.139344079349083,
                   -30.668781877071297, -11.047609531649897])])
    add("fma_lgammafn_sinpi", "lgammafn",
        [np.array([-77.24812758594393, -22.139344079349083,
                   -30.668781877071297, -11.047609531649897])])
    # stirlerr.c:120 libm lgamma (MM2, 1<=n<=5.25) + lgamma1p (n<1), through
    # dgamma's dpois_raw at non-integer shape.
    add("fma_dgamma_stirlerr", "dgamma",
        [np.array([0.0160354563943336, 0.13706135587904972]),
         np.array([2.035706520012819, 0.07697169460569216]),
         np.ones(2)], (True,))
    # lbeta.c:76 libm lgamma (p < 1e-306).
    add("fma_lbeta_tiny", "lbeta",
        [np.array([1e-307, 5e-307]), np.array([5.0, 2.5])])
    # pt.c `1 + (x/n)*x` / `n + x*x`; pf.c `df2 + df1*x`; dbeta.c lval.
    add("fma_pt_log_upper", "pt",
        [np.array([32.590139391805508, 117.24654876041654]),
         np.array([2088.0050570848925, 1496.6629770688498])],
        (False, True))
    add("fma_pf_log_upper", "pf",
        [np.array([422.04704175918579, 1.306547315051616,
                   2.4685076649519151]),
         np.array([4.8002726733580481, 305.1263899607892,
                   918.50946023950348]),
         np.array([306.96100608680996, 3.3554946569535344,
                   0.86639519425105682])],
        (False, True))
    add("fma_dbeta_lval", "dbeta",
        [np.array([0.62571187908681525, 0.13513636327126444]),
         np.array([0.4617840216011842, 1.9180888552809936]),
         np.array([0.12184792400934256, 0.74393942619217313])],
        (False,))
    # qbeta swapped-tail u = R_Log1_Exp(0) — C99 log(±0) = -Inf, no exception.
    add("fma_qbeta_log1exp_edge", "qbeta",
        [np.array([0.23074964248420893, 0.5373735010591706]),
         np.array([0.4617840216011842, 0.18892481378342565]),
         np.array([0.12184792400934256, 0.11337311528534429])],
        (True, False))
    # qt df<1: bisection `(ux-lx)/fabs(nx)` at nx == 0 (C99 Inf, no raise) and
    # the pt overflow lane `fma(x/n, x, 1)` -> Inf during bracket doubling.
    add("fma_qt_df_lt_1", "qt",
        [np.array([0.5, 0.9999, 0.13513636327126444]),
         np.array([0.4, 0.4, 0.9])], (True, False))
    # dnorm.c:52 log path `M_LN_SQRT_2PI + 0.5*x*x + log(sigma)` — the outer
    # mul of 0.5*x*x fuses into the add (materialized by the bcg family
    # census; dnorm was not in the original 26-config referee sweep). First
    # point is the traced bcg drift z.
    add("fma_dnorm_log", "dnorm",
        [np.array([1.0034637489050912, 0.6294080605981774,
                   2.3306046596904781, 31.415092653589793]),
         np.zeros(4), np.ones(4)], (True,))
    # dnorm.c:85 big-x split `(-0.5*x2 - x1)*x2` — fma(-0.5, x2, -x1).
    add("fma_dnorm_bigx", "dnorm",
        [np.array([5.7183098861837907, 12.566370614359172,
                   26.535897932384626]),
         np.zeros(3), np.ones(3)], (False,))

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


# ---------------------------------------------------------------------------
# Noncentral / studentized-range / hypergeometric _rs kernels — these
# accumulate their AS-275 / AS-226 / AS-243 / Copenhaver-Holland series in f64
# where R uses 80-bit LDOUBLE (Rust std has no 80-bit float). So unlike the
# strict gate above they match R to a TIGHT TOLERANCE, not 0-ulp; this pins port
# correctness (catches any O(1) transcription bug or gross regression). The
# 0-ulp Python reference for these is pinned in test_R.py; per-kernel ulp
# characterization is documented in the rust-perf plan.
# ---------------------------------------------------------------------------
def _grid(*axes):
    out = [[]]
    for ax in axes:
        out = [row + [v] for row in out for v in ax]
    cols = list(zip(*out))
    return [np.array(c, dtype=float) for c in cols], out


def _build_f64_cases():
    cases = []
    # noncentral chi-square (bulk grid)
    (X, D, N), g = _grid([0.5, 3.0, 10.0, 30.0, 60.0], [2.0, 5.0, 12.0], [2.0, 8.0, 40.0])
    cases.append(("pnchisq", "pnchisq", (X, D, N), (True, False),
                  [f"pchisq({x},{d},ncp={n})" for x, d, n in g]))
    cases.append(("dnchisq", "dnchisq", (X, D, N), (False,),
                  [f"dchisq({x},{d},ncp={n})" for x, d, n in g]))
    (P, D, N), g = _grid([0.1, 0.3, 0.5, 0.7, 0.9], [2.0, 5.0, 12.0], [2.0, 8.0, 40.0])
    cases.append(("qnchisq", "qnchisq", (P, D, N), (True, False),
                  [f"qchisq({p},{d},ncp={n})" for p, d, n in g]))
    # noncentral t (bulk — extreme far tails excluded; they degrade in f64)
    (T, D, N), g = _grid([-2.0, -0.5, 0.5, 2.0, 5.0], [5.0, 12.0, 40.0], [1.0, 3.0, 8.0])
    cases.append(("pnt", "pnt", (T, D, N), (True, False),
                  [f"pt({t},{d},ncp={n})" for t, d, n in g]))
    (P, D, N), g = _grid([0.1, 0.3, 0.5, 0.7, 0.9], [5.0, 12.0, 40.0], [1.0, 3.0, 8.0])
    cases.append(("qnt", "qnt", (P, D, N), (True, False),
                  [f"qt({p},{d},ncp={n})" for p, d, n in g]))
    # noncentral F
    (X, A, B, N), g = _grid([0.5, 1.0, 2.5], [3.0, 10.0], [5.0, 20.0], [2.0, 8.0])
    cases.append(("pnf", "pnf", (X, A, B, N), (True, False),
                  [f"pf({x},{a},{b},ncp={n})" for x, a, b, n in g]))
    (P, A, B, N), g = _grid([0.1, 0.5, 0.9], [3.0, 10.0], [5.0, 20.0], [2.0, 8.0])
    cases.append(("qnf", "qnf", (P, A, B, N), (True, False),
                  [f"qf({p},{a},{b},ncp={n})" for p, a, b, n in g]))
    # noncentral beta
    (X, A, B, N), g = _grid([0.1, 0.3, 0.6, 0.9], [2.0, 5.0], [2.0, 5.0], [2.0, 20.0])
    cases.append(("pnbeta", "pnbeta", (X, A, B, N), (True, False),
                  [f"pbeta({x},{a},{b},ncp={n})" for x, a, b, n in g]))
    # studentized range (rr = nranges = 1)
    (Q, R_, C, D), g = _grid([2.0, 3.0, 4.0], [1.0], [3.0, 5.0, 10.0], [10.0, 20.0, 60.0])
    cases.append(("ptukey", "ptukey", (Q, R_, C, D), (True, False),
                  [f"ptukey({q},{c},{d})" for q, _r, c, d in g]))
    (P, R_, C, D), g = _grid([0.5, 0.9, 0.95], [1.0], [3.0, 5.0, 10.0], [10.0, 20.0, 60.0])
    cases.append(("qtukey", "qtukey", (P, R_, C, D), (True, False),
                  [f"qtukey({p},{c},{d})" for p, _r, c, d in g]))
    # hypergeometric (m=20 red, n=25 black, k=15 drawn)
    xs = [float(x) for x in range(0, 16)]
    X = np.array(xs)

    def ones(v):
        return np.full(len(xs), float(v))
    cases.append(("dhyper", "dhyper", (X, ones(20), ones(25), ones(15)), (False,),
                  [f"dhyper({int(x)},20,25,15)" for x in xs]))
    cases.append(("phyper", "phyper", (X, ones(20), ones(25), ones(15)), (True, False),
                  [f"phyper({int(x)},20,25,15)" for x in xs]))
    return cases


_F64_CASES = _build_f64_cases()


@pytest.mark.parametrize("case", _F64_CASES, ids=[c[0] for c in _F64_CASES])
def test_rs_noncentral_f64_matches_r_tol(case):
    _label, fn, arrays, flags, exprs = case
    got = np.asarray(getattr(rs, fn)(*[np.ascontiguousarray(a) for a in arrays], *flags),
                     dtype=float)
    ref = r_scalar_values(exprs)
    exp = np.array([ref[e] for e in exprs])
    # f64 accumulators vs R's 80-bit LDOUBLE — tight tolerance, not 0-ulp.
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-9, equal_nan=True)


_FMA_CASES = [c for c in CASES if c[0].startswith("fma_")]


@pytest.mark.parametrize("case", _FMA_CASES, ids=[c[0] for c in _FMA_CASES])
def test_fma_cases_python_matches_rs(case):
    """Pure-Python nmath scalars ≡ rust 0-ulp at the FMA-materialization
    points. Both sides share the per-arch ``_rfma``/``rfma`` contraction and
    platform-libm (``__sinpi``/``lgamma``) decisions, so unlike the live-R
    gate this holds bit-for-bit on every platform; it pins the Python twins
    of every contraction site the ``fma_*`` R cases pin for rust."""
    from hea.R import nmath as nm
    py_fn = {
        "pbeta": nm.pbeta, "qbeta": nm.qbeta, "qgamma": nm.qgamma,
        "dgamma": nm.dgamma, "gammafn": nm.gammafn,
        "lgammafn": nm._lgammafn, "lbeta": nm.lbeta, "pt": nm.pt,
        "pf": nm.pf, "dbeta": nm.dbeta, "qt": nm.qt, "dnorm": nm.dnorm5,
    }
    name, fn, arrays, flags = case
    got_rs = np.asarray(getattr(rs, fn)(*arrays, *flags))
    got_py = np.array([py_fn[fn](*(float(v) for v in args), *flags)
                       for args in zip(*arrays)])
    assert got_py.shape == got_rs.shape
    for g, e in zip(got_py, got_rs):
        if np.isnan(e):
            assert np.isnan(g)
        else:
            assert _bits(g) == _bits(e), f"py={g!r} rs={e!r}"


# ---------------------------------------------------------------------------
# dqrls (R's lm.fit QR kernel) — 3-way parity: Rust ≡ pure-Python ≡ live R.
#
# Linear algebra, not transcendental libm, so this is a *tolerance* gate (BLAS
# reduction order differs across Accelerate/OpenBLAS/our in-order Rust), but
# rank + pivot must match R EXACTLY (the whole point of porting dqrdc2: a
# deterministic, R-faithful rank/pivot, immune to BLAS-bistable flakes).
# ---------------------------------------------------------------------------
import subprocess  # noqa: E402

from hea.R import linalg  # noqa: E402

_DQRLS_R = r"""
args <- commandArgs(trailingOnly=TRUE)
x <- as.matrix(read.csv(args[1], header=FALSE))
y <- as.numeric(read.csv(args[2], header=FALSE)[[1]])
z <- .lm.fit(x, y)
writeLines(c(
  paste("rank", z$rank),
  paste("pivot", paste(z$pivot, collapse=" ")),
  paste("coef", paste(sprintf("%.17g", z$coefficients), collapse=" ")),
  paste("effects", paste(sprintf("%.17g", z$effects), collapse=" ")),
  paste("resid", paste(sprintf("%.17g", z$residuals), collapse=" "))
), args[3])
"""


def _dqrls_cases():
    rng = np.random.default_rng(11)
    a = rng.standard_normal(12)
    b = rng.standard_normal(15)
    return {
        # full-rank well-conditioned
        "full_rank": (np.c_[np.ones(20), rng.standard_normal((20, 4))],
                      rng.standard_normal(20)),
        # one alias: col3 == 2·col2
        "rank_def": (np.c_[np.ones(12), a, 2.0 * a, rng.standard_normal(12)],
                     rng.standard_normal(12)),
        # two aliases: col3 == 3·col2, col5 == col2 − col1  → rank 3 of 5
        "two_alias": (np.c_[np.ones(15), b, 3.0 * b, rng.standard_normal(15),
                            b - 1.0], rng.standard_normal(15)),
    }


def _r_lmfit(x, y, tmp_path):
    xf, yf, of = tmp_path / "x.csv", tmp_path / "y.csv", tmp_path / "o.txt"
    rf = tmp_path / "f.R"
    rf.write_text(_DQRLS_R)
    np.savetxt(xf, x, delimiter=",")
    np.savetxt(yf, np.asarray(y), delimiter=",")
    subprocess.run(["Rscript", str(rf), str(xf), str(yf), str(of)],
                   check=True, stdin=subprocess.DEVNULL,
                   capture_output=True, text=True)
    res = {}
    for line in of.read_text().splitlines():
        key, _, rest = line.partition(" ")
        if key == "rank":
            res[key] = int(rest)
        elif key == "pivot":
            res[key] = np.array([int(v) for v in rest.split()])
        else:
            res[key] = np.array([float(v) for v in rest.split()])
    return res


@pytest.mark.parametrize("name", list(_dqrls_cases()))
def test_dqrls_3way_parity(name, tmp_path):
    """Rust dqrls ≡ pure-Python dqrls ≡ R .lm.fit (rank/pivot exact; the rest
    within a BLAS tolerance)."""
    x, y = _dqrls_cases()[name]
    rust = linalg.Cdqrls(x, y)                          # Rust active path
    qr, coef, rsd, qty, k, jpvt, qraux = linalg.dqrls(x.copy(), y)  # pure-Python oracle
    R = _r_lmfit(x, y, tmp_path)

    # rank + pivot: EXACT, all three
    assert rust["rank"] == k == R["rank"]
    assert np.array_equal(rust["pivot"], jpvt)
    assert np.array_equal(rust["pivot"], R["pivot"])

    rk = R["rank"]
    # Rust ≡ pure-Python: deterministic in-order BLAS both sides → tight
    np.testing.assert_allclose(rust["coefficients"], coef, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rust["effects"], qty, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rust["residuals"], rsd, rtol=0, atol=1e-10)
    # Rust ≡ R: BLAS tolerance, but the USED (first-rank) effects/coef match tightly
    np.testing.assert_allclose(rust["effects"][:rk], R["effects"][:rk], atol=1e-9)
    np.testing.assert_allclose(rust["residuals"], R["resid"], atol=1e-9)
    # coefficients are in pivoted order on both sides
    np.testing.assert_allclose(rust["coefficients"][:rk], R["coef"][:rk], atol=1e-9)


# ---------------------------------------------------------------------------
# tp basis kernel eval (XBuild + tpsE) — Rust ≡ pure-Python, BIT-EXACT.
#
# Pure element-wise fast_eta + polynomial powers (NO matmul inside), so the Rust
# build of b=[E|T] and the knot matrix E are byte-identical to the numpy
# `_tp_fast_eta_vec`/`_tp_T` build (the `b @ UZ` matmul stays in numpy). d=1 is
# the common `s(x)`; d=2 exercises the even-d log branch, d=3 the odd-d √ branch
# + the degree-2 polynomial null space (powi vs `**`). Sizes ≥256 hit rayon.
# ---------------------------------------------------------------------------
from hea.formula import (  # noqa: E402
    _tp_eta_const, _tp_fast_eta_vec, _tp_gen_poly_powers, _tp_null_space_dim,
    _tp_T,
)
from hea.R._shared import _rfma, _rfma_vec  # noqa: E402


def _rsq_rfma(diff):
    """Squared distance by the same per-arch ``r += z*z`` → fma fold that the
    rust ``tp_eval_*`` kernels and the numpy ``_tp_E``/``_tp_eval_X_raw`` build
    use (tprs.c:92 tpsE / :591 XBuild). Mirroring it here keeps the reference
    0-ulp to both on arm64 (where clang fuses the accumulate to ``fmadd``)."""
    rsq = np.zeros(diff.shape[:-1])
    for k in range(diff.shape[-1]):
        zk = diff[..., k]
        rsq = _rfma_vec(zk, zk, rsq)
    return rsq


def _assert_eta_parity(d, got, want):
    """Rust ``tp_eval_*`` vs the numpy ``_tp_fast_eta_vec`` build.

    For ODD d the radial kernel is ``f0·(r²)^k·√(r²)`` — only ``sqrt`` (an
    IEEE correctly-rounded operation, so bit-identical across every libm) plus
    integer-power multiplies, hence byte-exact on every platform.

    For EVEN d it is ``f0·log(r²)·…`` and ``log`` is NOT a correctly-rounded
    IEEE operation: rust's scalar ``f64::ln`` and numpy's vectorised ``np.log``
    are two conformant-but-distinct implementations. They coincide bit-for-bit
    on the darwin capture box (both Intel and arm64), but a Linux numpy wheel's
    SIMD ``log`` can disagree by ≤1 ULP (a py3.12 CI runner did). That is far
    below the mgcv-oracle fixture tolerance (5e-5) the basis is actually gated
    on, so off-darwin the even-d branch uses the shared libm-floor tolerance —
    same ``_STRICT``/``_LINUX_RTOL`` split as :func:`_assert_bit_exact`."""
    if d % 2 == 0 and not _STRICT:
        np.testing.assert_allclose(got, want, rtol=_LINUX_RTOL, atol=1e-300,
                                   equal_nan=True)
    else:
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("d,m", [(1, 2), (2, 2), (3, 3)])
def test_tp_eval_b_bit_exact(d, m):
    rng = np.random.default_rng(d)
    x_c = rng.uniform(-2.0, 2.0, (500, d))
    Xu = rng.uniform(-2.0, 2.0, (60, d))
    eta0 = _tp_eta_const(m, d)
    M = _tp_null_space_dim(d, m)
    pp = np.ascontiguousarray(_tp_gen_poly_powers(M, m, d).astype(np.int64))
    b_rs = rs.tp_eval_b(np.ascontiguousarray(x_c), np.ascontiguousarray(Xu),
                        int(m), int(d), float(eta0), pp)
    diff = x_c[:, None, :] - Xu[None, :, :]
    rsq = _rsq_rfma(diff)
    b_np = np.hstack([_tp_fast_eta_vec(m, d, rsq, eta0), _tp_T(x_c, m, d)])
    _assert_eta_parity(d, b_rs, b_np)


@pytest.mark.parametrize("d,m", [(1, 2), (2, 2), (3, 3)])
def test_tp_eval_E_bit_exact(d, m):
    rng = np.random.default_rng(d + 10)
    Xu = rng.uniform(-2.0, 2.0, (300, d))
    eta0 = _tp_eta_const(m, d)
    E_rs = rs.tp_eval_E(np.ascontiguousarray(Xu), int(m), int(d), float(eta0))
    diff = Xu[:, None, :] - Xu[None, :, :]
    rsq = _rsq_rfma(diff)
    E_np = _tp_fast_eta_vec(m, d, rsq, eta0)
    _assert_eta_parity(d, E_rs, E_np)


def _rw_matrix_ref(stop, row, weight, X, trans):
    """Scalar mirror of mgcv ``rwMatrix`` (misc.c:731-745): the i-outer/j-inner
    fold the rust ``rw_matrix`` kernel ports, accumulating ``*X1p += weight *
    *Xp`` via the per-arch ``_rfma`` (the C ``fmadd`` on arm64). Pins the kernel
    here without an R round-trip; separately verified 0-ulp to live
    ``mgcv:::rwMatrix`` on arm64."""
    stop = np.asarray(stop, dtype=int) - 1
    row = np.asarray(row, dtype=int) - 1
    weight = np.asarray(weight, dtype=float)
    X = np.asarray(X, dtype=float)
    is_mat = X.ndim == 2
    Xm = X if is_mat else X.reshape(-1, 1)
    n, p = Xm.shape
    out = np.zeros((n, p))
    start = 0
    for i in range(n):
        end = int(stop[i]) + 1
        for j in range(start, end):
            rj, w = int(row[j]), float(weight[j])
            src, dst = (i, rj) if trans else (rj, i)
            for c in range(p):
                out[dst, c] = _rfma(w, Xm[src, c], out[dst, c])
        start = end
    return out if is_mat else out.ravel()


@pytest.mark.parametrize("n,p,arstart", [
    (50, 4, None), (200, 1, None), (30, 3, (10, 21)),
    (2, 2, None), (1, 5, None),
])
@pytest.mark.parametrize("rho", [0.3, 0.7, 0.9])
@pytest.mark.parametrize("trans", [False, True])
def test_rw_matrix_bit_exact(n, p, arstart, rho, trans):
    """Rust ``rw_matrix`` (misc.c:710-748, the AR1 row-recombine) ≡ the scalar
    per-arch fma fold of misc.c, bit-for-bit. Its only arithmetic is ``rfma``
    (no transcendental), so the equality holds on every arch — arm64 fuses both
    sides to ``fmadd``, x86 leaves both as ``a*b+c``. Separately verified 0-ulp
    to live ``mgcv:::rwMatrix`` on arm64."""
    from hea.models.bam import _ar1_rwmatrix_indices, _rw_matrix
    rng = np.random.default_rng(n + p + int(rho * 100))
    ld = 1.0 / np.sqrt(1.0 - rho ** 2)
    sd = -rho * ld
    ar_block = None
    if arstart is not None:
        ar_block = np.zeros(n, dtype=bool)
        for k in arstart:               # 1-based AR-restart event -> 0-based
            ar_block[k - 1] = True
    stop, row, weight = _ar1_rwmatrix_indices(n, ld, sd, ar_block)
    X = rng.standard_normal(n) if p == 1 else rng.standard_normal((n, p))
    got = _rw_matrix(stop, row, weight, X, trans=trans)
    want = _rw_matrix_ref(stop, row, weight, X, trans=trans)
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# coxlpl — mgcv's Cox partial-likelihood kernel (coxph.c:141). The rust single-
# pass risk-set sweep vs the numpy cumsum oracle: agree to ~n·eps, NOT 0-ulp
# (sequential vs pairwise reduction — a DIFFERENT algorithm, per the user's
# "diverged from numpy, fine" rule). The rust kernel's per-statement fma + the
# mult/div order are pinned bit-for-bit to the COMPILED coxph.c separately (the
# d1H/d2H emit fma trees were verified 4000/6000-for-4000/6000 vs `clang -O2
# -arch arm64` of the exact C emit); the live-R gate is the functional cox.ph
# gam fixture (tests/test_gam.py::test_cox_ph_through_gam_matches_mgcv).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deriv", [0, 1])
def test_coxlpl_kernel_parity(deriv):
    import hea.family as fam

    rng = np.random.default_rng(40 + deriv)
    n, p = 250, 5
    X = rng.standard_normal((n, p))
    eta = X @ (rng.standard_normal(p) * 0.4)
    time = rng.uniform(0.0, 12.0, n)
    d = (rng.uniform(size=n) < 0.65).astype(int)
    kw = {"d1b": rng.standard_normal((p, 3)) * 0.25} if deriv == 1 else {}

    rust = fam._coxlpl(eta, X, d, time, deriv, **kw)               # rust active
    saved = (fam._rs_cox_l, fam._rs_cox_lpl0, fam._rs_cox_lpl_d1, fam._rs_cox_d2h)
    fam._rs_cox_l = fam._rs_cox_lpl0 = fam._rs_cox_lpl_d1 = fam._rs_cox_d2h = None
    try:
        npy = fam._coxlpl(eta, X, d, time, deriv, **kw)           # numpy oracle
    finally:
        (fam._rs_cox_l, fam._rs_cox_lpl0,
         fam._rs_cox_lpl_d1, fam._rs_cox_d2h) = saved

    assert abs(rust["l"] - npy["l"]) < 1e-11
    np.testing.assert_allclose(rust["lb"], npy["lb"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(rust["lbb"], npy["lbb"], rtol=0, atol=1e-11)
    if deriv == 1:
        np.testing.assert_allclose(rust["d1H"], npy["d1H"], rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------
# pls_fit1 — mgcv's penalized least-squares inner solve (rust TSQR + neg-weight
# eigen correction) vs the numpy QR/eigh oracle (== hea gam._pls_qr's pure path).
# Not 0-ulp (different QR — TSQR vs LAPACK), but every returned quantity (β, the
# Cholesky factor of X'WX+Sλ, log|X'WX+Sλ|) is QR-convention-invariant so they
# agree to the BLAS floor on well-conditioned problems.
from scipy.linalg import solve_triangular  # noqa: E402


def _pls_oracle(X, w, E, z=None, Xtwz=None):
    neg = w < 0.0
    sqw = np.sqrt(np.abs(w))
    aug = np.vstack([X * sqw[:, None], E])
    Q, R = np.linalg.qr(aug)
    diag = np.diag(R)
    XWz = Xtwz if Xtwz is not None else X.T @ (w * z)
    if not neg.any():
        c = solve_triangular(R, XWz, lower=False, trans="T")
        beta = solve_triangular(R, c, lower=False)
        sgn = np.where(diag < 0, -1.0, 1.0)
        return beta, R * sgn[:, None], 2 * np.sum(np.log(np.abs(diag)))
    A = sqw[neg][:, None] * X[neg]
    Y = solve_triangular(R, A.T @ A, lower=False, trans="T")
    Z = solve_triangular(R, Y.T, lower=False, trans="T").T
    evals, V = np.linalg.eigh(0.5 * (Z + Z.T))
    d2 = 1 - 2 * evals
    if np.any(d2 <= 0.0):
        return None  # penalized Hessian indefinite (Fisher-retry signal)
    Vt = V.T
    c = Vt @ solve_triangular(R, XWz, lower=False, trans="T")
    beta = solve_triangular(R, Vt.T @ (c / d2), lower=False)
    M = np.sqrt(d2)[:, None] * (Vt @ R)
    Rc = np.linalg.qr(M, mode="r")
    sgn = np.where(np.diag(Rc) < 0, -1.0, 1.0)
    return beta, Rc * sgn[:, None], (2 * np.sum(np.log(np.abs(diag)))
                                     + np.sum(np.log(d2)))


@pytest.mark.parametrize("n,p,nneg", [(2000, 10, 0), (2000, 10, 200),
                                      (1500, 14, 700), (1200, 8, 0)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_pls_fit1_parity(n, p, nneg, order):
    rng = np.random.default_rng(n + p + nneg)
    X = rng.standard_normal((n, p))
    X = np.ascontiguousarray(X) if order == "C" else np.asfortranarray(X)
    w = np.abs(rng.standard_normal(n)) * rng.uniform(0.2, 3.0)
    if nneg:
        idx = rng.choice(n, nneg, replace=False)
        w[idx] = -w[idx]
    E = (np.asfortranarray if order == "F" else np.ascontiguousarray)(
        rng.standard_normal((p, p)) * 0.7)
    z = rng.standard_normal(n)
    oracle = _pls_oracle(np.asarray(X), w, np.asarray(E), z=z)
    ok, beta, R, ld = rs.pls_fit1(X, w, E, z, np.empty(0), False)
    if oracle is None:  # indefinite penalized Hessian — both must decline
        assert not ok
        return
    b0, R0, ld0 = oracle
    assert ok
    np.testing.assert_allclose(beta, b0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(R), R0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(ld, ld0, rtol=0, atol=1e-9)


def test_pls_fit1_xtwz_mode_parity():
    rng = np.random.default_rng(99)
    n, p = 1800, 11
    X = np.asfortranarray(rng.standard_normal((n, p)))
    w = np.abs(rng.standard_normal(n)) + 0.1
    E = np.asfortranarray(rng.standard_normal((p, p)) * 0.5)
    wz = rng.standard_normal(n)
    Xtwz = np.asarray(X).T @ wz
    b0, R0, ld0 = _pls_oracle(np.asarray(X), w, np.asarray(E), Xtwz=Xtwz)
    ok, beta, R, ld = rs.pls_fit1(X, w, E, np.empty(0), Xtwz, True)
    assert ok
    np.testing.assert_allclose(beta, b0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ld, ld0, rtol=0, atol=1e-9)


# pls_fit1 — kernel-level R arm: the rust solve vs mgcv's OWN `pls_fit1` C
# routine (.C(C_pls_fit1), gdi.c:2895, the same call gam.fit3.r:334 makes), and
# its returned factor/log-det vs R's exact `chol`/`determinant` of the penalized
# Hessian X'WX+Sλ. Not 0-ulp: mgcv solves via LAPACK `dgeqrf` (libRlapack) over
# Accelerate BLAS leaf reductions while the rust kernel is a row-blocked TSQR,
# so this gates rs==R at the LAPACK floor (~1e-14 on the factor, ~1e-15 on β).
# Together with `test_pls_fit1_parity` (rs==numpy) this is the 3-way gate.
_PLS_R = r"""
suppressMessages(library(mgcv))
a <- commandArgs(trailingOnly=TRUE)
rd <- function(f) as.matrix(read.csv(f, header=FALSE))
n <- as.integer(a[5]); p <- as.integer(a[6])
X <- rd(a[1]); dim(X) <- c(n, p)
w <- as.numeric(rd(a[2])); z <- as.numeric(rd(a[3]))
E <- rd(a[4]); dim(E) <- c(p, p)
oo <- .C("pls_fit1", y=as.double(z), X=as.double(X), w=as.double(w),
         wy=as.double(w*z), E=as.double(E), Es=as.double(E),
         n=as.integer(n), q=as.integer(p), rE=as.integer(p),
         eta=as.double(z), penalty=as.double(1),
         rank.tol=as.double(.Machine$double.eps^0.5),
         nt=as.integer(1), use.wy=as.integer(0), PACKAGE="mgcv")
out <- paste("nsig", oo$n)
if (oo$n >= 0) {
  M <- t(X) %*% (w * X) + t(E) %*% E
  Rc <- tryCatch(chol(M), error=function(e) NULL)
  lines <- c(out,
    paste("beta", paste(sprintf("%.17g", oo$y[1:p]), collapse=" ")),
    paste("penalty", sprintf("%.17g", oo$penalty)))
  if (!is.null(Rc)) lines <- c(lines,
    paste("pd", 1),
    paste("rfac", paste(sprintf("%.17g", as.numeric(t(Rc))), collapse=" ")),
    paste("logdet", sprintf("%.17g",
          as.numeric(determinant(M, logarithm=TRUE)$modulus))))
  else lines <- c(lines, paste("pd", 0))
  out <- lines
}
writeLines(out, a[7])
"""


def _r_pls_fit1(X, w, z, E, tmp_path):
    n, p = X.shape
    xf, wf, zf, ef = (tmp_path / f"{s}.csv" for s in "xwze")
    of, rf = tmp_path / "o.txt", tmp_path / "f.R"
    rf.write_text(_PLS_R)
    np.savetxt(xf, np.asarray(X), delimiter=",")
    np.savetxt(wf, w, delimiter=",")
    np.savetxt(zf, z, delimiter=",")
    np.savetxt(ef, np.asarray(E), delimiter=",")
    subprocess.run(["Rscript", str(rf), str(xf), str(wf), str(zf), str(ef),
                    str(n), str(p), str(of)],
                   check=True, stdin=subprocess.DEVNULL,
                   capture_output=True, text=True)
    res = {}
    for line in of.read_text().splitlines():
        key, _, rest = line.partition(" ")
        if key in ("nsig", "pd"):
            res[key] = int(rest)
        elif key in ("penalty", "logdet"):
            res[key] = float(rest)
        else:
            res[key] = np.array([float(v) for v in rest.split()])
    return res


@pytest.mark.parametrize("n,p,nneg", [(2000, 10, 0), (2000, 10, 200),
                                      (1500, 14, 700), (200, 6, 40)])
def test_pls_fit1_matches_r(n, p, nneg, tmp_path):
    """Rust ``pls_fit1`` ≡ mgcv's own ``.C(C_pls_fit1)`` (β, penalty) and its
    factor/log-det ≡ R ``chol``/``determinant`` of X'WX+Sλ, at the LAPACK floor.
    The indefinite case must be declined by BOTH (mgcv's ``n<0`` Fisher-retry
    signal)."""
    rng = np.random.default_rng(n + p + nneg)
    X = np.asfortranarray(rng.standard_normal((n, p)))
    w = np.abs(rng.standard_normal(n)) * rng.uniform(0.2, 3.0)
    if nneg:
        idx = rng.choice(n, nneg, replace=False)
        w[idx] = -w[idx]
    E = np.asfortranarray(rng.standard_normal((p, p)) * 0.7)
    z = rng.standard_normal(n)
    R = _r_pls_fit1(X, w, z, E, tmp_path)
    ok, beta, Rfac, ld = rs.pls_fit1(X, w, E, z, np.empty(0), False)
    if R["nsig"] < 0:                       # X'WX+Sλ indefinite
        assert not ok                       # rust must decline too
        return
    assert ok
    # β vs mgcv's actual pls_fit1; penalty β'Sλβ = ‖Eβ‖² vs its `penalty` out.
    np.testing.assert_allclose(beta, R["beta"], rtol=0, atol=1e-10)
    pen_rust = float(np.asarray(E) @ beta @ (np.asarray(E) @ beta))
    np.testing.assert_allclose(pen_rust, R["penalty"], rtol=1e-9, atol=1e-12)
    if R.get("pd"):                         # factor + log-det vs R linear algebra
        Rc = R["rfac"].reshape(p, p)        # chol(M), positive-diagonal upper
        np.testing.assert_allclose(np.asarray(Rfac), Rc, rtol=0, atol=1e-9)
        np.testing.assert_allclose(ld, R["logdet"], rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# gamlss_xwx — gamlss.gH's Hessian-block crossprod `Σ_k X_i[k,r]·WX_j[k,c]`
# (family.gamlss_gH under deterministic_xwx, gam.fit5's rank check). Not 0-ulp
# to numpy `@`/einsum (a different, fixed reduction order) but agrees to the
# BLAS floor. The property that matters — and that numpy `@` fails — is row/col
# consistency: bit-identical input columns must give bit-identical output rows
# AND cols, else gam.fit5's QR rank-check drops a duplicate column platform-
# dependently (the arm64 gevlss bug this kernel fixes).
@pytest.mark.parametrize("n,p", [(300, 14), (2000, 41), (5000, 23)])
def test_gamlss_xwx_parity(n, p):
    rng = np.random.default_rng(n + p)
    Xi = np.ascontiguousarray(rng.standard_normal((n, p)))
    WXj = np.ascontiguousarray(rng.standard_normal(n)[:, None]
                               * rng.standard_normal((n, p)))
    A = np.asarray(rs.gamlss_xwx(Xi, WXj))
    # agrees with the einsum oracle to the summation-order floor
    np.testing.assert_allclose(A, np.einsum("kr,kc->rc", Xi, WXj),
                               rtol=0, atol=1e-9)
    # deterministic: identical across runs
    np.testing.assert_array_equal(A, np.asarray(rs.gamlss_xwx(Xi, WXj)))


def test_gamlss_xwx_row_col_consistent():
    # Duplicate an input column (rank-deficient design) → its two output
    # rows/cols must be bit-identical, the construction property `@` lacks.
    rng = np.random.default_rng(7)
    n, p = 4000, 18
    Xi = np.ascontiguousarray(rng.standard_normal((n, p)))
    Xi[:, 11] = Xi[:, 4]                       # column 11 ≡ column 4
    WXj = np.ascontiguousarray(rng.standard_normal(n)[:, None] * Xi)
    A = np.asarray(rs.gamlss_xwx(Xi, WXj))
    np.testing.assert_array_equal(A[4], A[11])     # rows bit-identical
    np.testing.assert_array_equal(A[:, 4], A[:, 11])  # cols bit-identical


# ---------------------------------------------------------------------------
# tweedie_series — mgcv `tweedious` (misc.c:170) scalar-p series via the rust
# per-row sweep. The returned 7 columns are (log_a, E[j], Var[j], E[jψ]) plus
# mgcv's three p-param working-derivative accumulators (m_wp1, m_comb, m_dwpp,
# misc.c:346-503). Two gates: (1) rust vs the numpy dense-matrix oracle (==
# `_tweedie_log_a_vec` forced off) — not 0-ulp, the sweep reduces in a
# different order + uses `rfma` for the wp1²+wp2 combine; (2) the whole
# `_ld_tweedie_work` vs LIVE R `ldTweedie` (the new R arm).
import hea.family as _fam  # noqa: E402


@pytest.mark.parametrize("p", [1.05, 1.1, 1.5, 1.93, 1.99])
@pytest.mark.parametrize("phi", [0.3, 1.0, 5.0])
def test_tweedie_series_parity(p, phi):
    if _fam._rs_tweedie_series is None:
        pytest.skip("hea._rs.tweedie_series unavailable")
    rng = np.random.default_rng(int(p * 100) + int(phi * 10))
    y = rng.uniform(0.01, 500.0, 1500)
    out_rs = _fam._tweedie_log_a_vec(y, phi, p)
    orig = _fam._rs_tweedie_series
    _fam._rs_tweedie_series = None
    try:
        out_np = _fam._tweedie_log_a_vec(y, phi, p)
    finally:
        _fam._rs_tweedie_series = orig
    for a, b in zip(out_rs, out_np):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("p", [1.05, 1.1, 1.5, 1.93, 1.99])
@pytest.mark.parametrize("phi", [0.3, 1.0, 5.0])
def test_tweedious_work_rs_matches_py(p, phi):
    """The faithful tweedious/tweedious2 twins: rust ``tweedious_work`` /
    ``tweedious_work_pv`` vs the pure-Python reference ports. Both use plain
    ops (no fma) and the same nmath scalars, so they are BIT-IDENTICAL (0 ulp) —
    unlike the moment kernels' `rfma` combine, which forces the 1e-8 gate."""
    if _fam._rs_tweedious_work is None:
        pytest.skip("hea._rs.tweedious_work unavailable")
    rng = np.random.default_rng(int(p * 100) + int(phi * 10))
    y = rng.uniform(0.01, 500.0, 800)
    rho = float(np.log(phi))
    eth = np.exp(-abs(0.3))       # θ = 0.3 (scalar buffer path)
    a, b = 1.001, 1.999
    dpth1 = eth * (b - a) / (1 + eth) ** 2
    dpth2 = ((a - b) * eth + (b - a) * eth * eth) / (1 + eth) ** 3
    # scalar path
    sr = _fam._tweedious_work_scalar(y, rho, p, dpth1, dpth2)
    sp = _fam._tweedious_work_scalar_py(y, rho, p, dpth1, dpth2)
    np.testing.assert_array_equal(sr, sp)
    # vector path: per-row θ/ρ derived from a jittered p/φ
    thv = rng.uniform(-1.0, 1.2, y.size)
    rhv = rng.uniform(-0.6, 0.5, y.size)
    ev = np.exp(-np.abs(thv))
    pv = np.where(thv > 0, (b + a * ev) / (1 + ev), (b * ev + a) / (ev + 1))
    d1v = ev * (b - a) / (1 + ev) ** 2
    d2v = np.where(thv > 0, ((a - b) * ev + (b - a) * ev * ev) / (1 + ev) ** 3,
                   ((a - b) * ev * ev + (b - a) * ev) / (1 + ev) ** 3)
    pr = _fam._tweedious_work_pv(y, rhv, pv, d1v, d2v)
    pp = _fam._tweedious_work_pv_py(y, rhv, pv, d1v, d2v)
    np.testing.assert_array_equal(pr, pp)


@pytest.mark.parametrize("theta,rho", [(-0.5, 0.3), (0.0, -0.4), (0.7, 0.8)])
def test_tweedie_ldwork_matches_r(theta, rho):
    """R arm: hea ``_ld_tweedie_work`` vs live mgcv ``ldTweedie`` in the
    (θ, ρ) working parameterisation (``all.derivs=TRUE``). The log-density,
    1st derivatives and the closed-form μ columns hit the libm floor; the 2nd
    p-derivatives (cols 2/4/5) carry an inherent saddle+series cancellation
    that mgcv shares (R itself is ~1e-9 off the float128 truth for large y), so
    those gate at the cancellation floor on moderate y. nmath (R's Rmath)
    special functions + mgcv's working-derivative reduction back the tight arms."""
    import subprocess

    from hea.R import runif, set_seed
    set_seed(abs(int(theta * 100) + int(rho * 10)) + 11)
    y = np.concatenate([[0.0, 0.0], runif(18, 0.1, 60.0)])
    mu = np.concatenate([[0.5, 1.5], runif(18, 0.2, 50.0)])
    ld = _fam._ld_tweedie_work(y, mu, np.full_like(y, theta),
                               np.full_like(y, rho), a=1.001, b=1.999)
    ys = ",".join(map(repr, y.tolist()))
    mus = ",".join(map(repr, mu.tolist()))
    src = (f"suppressMessages(library(mgcv)); y<-c({ys}); mu<-c({mus}); "
           f"M<-ldTweedie(y,mu,rho={rho!r},theta={theta!r},a=1.001,b=1.999,"
           "all.derivs=TRUE); "
           "write.table(format(M,digits=17),stdout(),row.names=F,"
           "col.names=F,quote=F)")
    out = subprocess.run(["Rscript", "-e", src], capture_output=True,
                         text=True, check=True).stdout
    R = np.array([[float(v) for v in ln.split()]
                  for ln in out.splitlines() if ln.strip()])
    # cols 0 (l), 1 (d/dρ), 3 (d/dθ), 6-9 (μ) — libm floor.
    for c in (0, 1, 3, 6, 7, 8, 9):
        np.testing.assert_allclose(ld[:, c], R[:, c], rtol=1e-9, atol=1e-10)
    # cols 2/4/5 (2nd p-derivatives) — saddle+series cancellation floor.
    for c in (2, 4, 5):
        np.testing.assert_allclose(ld[:, c], R[:, c], rtol=1e-6, atol=1e-7)


# --- psigamma (R dpsifn) — rust vs the pure-Python oracle --------------------
from hea.R import nmath as _nmath  # noqa: E402


@pytest.mark.parametrize("deriv", [0, 1, 2, 3])
def test_psigamma_parity_positive(deriv):
    """x > 0 (the betar/nb/scat domain) — bit-exact on darwin, libm-floor off.
    Spans the three dpsifn regimes (backward-recursion / series / asymptotic)."""
    if _nmath._rs_psigamma is None:
        pytest.skip("hea._rs.psigamma unavailable")
    rng = np.random.default_rng(deriv + 1)
    x = np.concatenate([rng.uniform(1e-3, 0.5, 200), rng.uniform(0.5, 9.0, 200),
                        rng.uniform(9.0, 1e4, 200)])
    got = _nmath.psigamma_vec(x, deriv)
    orig = _nmath._rs_psigamma
    _nmath._rs_psigamma = None
    try:
        want = _nmath.psigamma_vec(x, deriv)
    finally:
        _nmath._rs_psigamma = orig
    _assert_bit_exact(got, want)


@pytest.mark.parametrize("deriv", [0, 1])
def test_psigamma_parity_reflection(deriv):
    """x < 0 reflection (twlss's deriv=1 negative-arg trigamma). The A&S
    reflection cancels badly, so a last-bit libm sin/cos difference between
    rust-std and python-math gets amplified to ~1 ulp even on shared-libm
    darwin — so this is a tolerance gate (rtol=1e-9, the reduction precedent)
    on all platforms, not bit-exact. Both sides are ~5-ulp vs R here."""
    if _nmath._rs_psigamma is None:
        pytest.skip("hea._rs.psigamma unavailable")
    rng = np.random.default_rng(deriv + 7)
    x = -rng.uniform(0.01, 8.0, 400)
    got = _nmath.psigamma_vec(x, deriv)
    orig = _nmath._rs_psigamma
    _nmath._rs_psigamma = None
    try:
        want = _nmath.psigamma_vec(x, deriv)
    finally:
        _nmath._rs_psigamma = orig
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300, equal_nan=True)


# --- discrete X'WX smooth×smooth block — rust vs the numpy oracle ------------
def test_xwx_smooth_block_parity():
    """``hea._rs.xwx_smooth_block`` (the matrix-arg / tensor X'WX accumulation,
    mgcv XWXijs) must equal the numpy ``_smooth_smooth_block`` oracle. The two
    sum the final-marginal weight table in different orders (rust scatters per
    row; numpy bincounts per (s,t)), so this is a tight tolerance gate, not
    0-ulp. Exercises both a matrix-argument by= tensor and its diagonal."""
    import importlib
    _bam = importlib.import_module("hea.models.bam")
    if _bam._rs_xwx_smooth_block is None:
        pytest.skip("hea._rs.xwx_smooth_block unavailable")
    import polars as pl
    from hea.family import Poisson
    rng = np.random.default_rng(0)
    nm, mm = 600, 5
    pm10 = rng.uniform(0, 5, (nm, mm))
    lag = np.tile(np.arange(mm, dtype=float), (nm, 1))
    stim = rng.standard_normal((nm, mm))
    y = rng.poisson(1.0, nm).astype(float)
    df = pl.DataFrame({"y": y, "Lag": lag, "Xc": pm10, "Stim": stim})
    m = _bam.bam("y ~ te(Lag, Xc, by=Stim, k=c(4,3))", df,
                 family=Poisson(), discrete=True)
    d = m._discrete_design
    w = rng.uniform(0.1, 2.0, nm)
    got = _bam.XWXd(d, w)
    orig = _bam._rs_xwx_smooth_block
    _bam._rs_xwx_smooth_block = None
    try:
        want = _bam.XWXd(d, w)
    finally:
        _bam._rs_xwx_smooth_block = orig
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_xwx_smooth_block_largep_indreduce_parity():
    """rust == numpy on the !acc_w large-``p`` ``indReduce`` dense branch
    (bam.py ``min(p)>15`` gate / discrete.rs). A discretised functional covariate
    bounds the final-marginal grid (``msize ≤ cap``) so the dense-W̄ path is taken
    with p≈19 > 15 — the k=c(4,3) case above stays on the small-p factor path, so
    this is the branch the existing parity test does not reach."""
    import importlib
    _bam = importlib.import_module("hea.models.bam")
    if _bam._rs_xwx_smooth_block is None:
        pytest.skip("hea._rs.xwx_smooth_block unavailable")
    import polars as pl
    from hea.family import Gaussian
    rng = np.random.default_rng(5)
    nm, L = 2500, 5
    grid = np.linspace(0, 5, 70)
    Xc = grid[rng.integers(0, 70, (nm, L))]      # discretised → bounded grid
    lag = np.tile(np.linspace(0, 1, L), (nm, 1))
    stim = rng.standard_normal((nm, L))
    df = pl.DataFrame({"y": rng.standard_normal(nm), "Lag": lag,
                       "Xc": Xc, "Stim": stim})
    m = _bam.bam("y ~ te(Lag, Xc, by=Stim, k=c(4,20))", df,
                 family=Gaussian(), discrete=True)
    d = m._discrete_design
    w = rng.uniform(0.1, 2.0, d.n)
    got = _bam.XWXd(d, w)
    orig = _bam._rs_xwx_smooth_block
    _bam._rs_xwx_smooth_block = None
    try:
        want = _bam.XWXd(d, w)
    finally:
        _bam._rs_xwx_smooth_block = orig
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_wbar_contract_indreduce_dense_branch():
    """``_wbar_contract``'s !acc_w large-``p`` branch (the dense-W̄ ``indReduce``
    equivalent, mgcv discrete.c:1884/1922) equals the brute-force ``Xim' W̄ Xjm``
    to the contraction floor — and the perf-guard (``n ≪ msize``), memory-cap
    (``msize > cap``) and small-``p`` cases all fall back to the factor path and
    stay exact. Verifies the dense gate matches across all four regimes."""
    import importlib
    _bam = importlib.import_module("hea.models.bam")
    cap = _bam._XWX_DENSE_MSIZE_CAP

    def brute(Ki_l, Kj_l, vl, Xim, Xjm):
        W = np.zeros((Xim.shape[0], Xjm.shape[0]))
        for Ki, Kj, v in zip(Ki_l, Kj_l, vl):
            np.add.at(W, (Ki, Kj), v)
        return Xim.T @ W @ Xjm

    def check(n, mim, mjm, p, ss, seed, expect_dense):
        rng = np.random.default_rng(seed)
        Ki = [rng.integers(0, mim, n).astype(np.int64) for _ in range(ss)]
        Kj = [rng.integers(0, mjm, n).astype(np.int64) for _ in range(ss)]
        vl = [rng.standard_normal(n) for _ in range(ss)]
        Xim = rng.standard_normal((mim, p))
        Xjm = rng.standard_normal((mjm, p))
        msize = mim * mjm
        dense = (n > msize
                 or (min(p, p) > 15 and msize <= cap and msize <= 16 * ss * n))
        assert dense is expect_dense
        got = _bam._wbar_contract(Ki, Kj, vl, Xim, Xjm)
        np.testing.assert_allclose(got, brute(Ki, Kj, vl, Xim, Xjm),
                                   rtol=0, atol=1e-9)

    check(3000, 80, 80, 20, 4, 1, expect_dense=True)     # indReduce dense branch
    check(400, 150, 150, 20, 1, 2, expect_dense=False)   # n≪msize → factor guard
    check(2500, 2100, 2100, 18, 1, 3, expect_dense=False)  # msize>cap → factor
    check(1000, 40, 40, 10, 3, 4, expect_dense=False)    # !acc_w p≤15 → factor


def test_xwx_smooth_block_ar1_tri_parity():
    """rust == numpy on the AR1 ``tri`` path (the rust kernel does the diagonal +
    super/sub tridiagonal scatters internally, mgcv XWXijs tri branches
    discrete.c:1843-1880). Before this, general AR1 blocks were forced onto the
    numpy per-(s,t) loop; now they take the rust pass. Covers both a plain tensor
    ``te()`` (s_i=1, nd_i>1) and a signal-regression ``te(…, by=)`` (s_i·s_j=L²,
    the case the rust pass is meant to win)."""
    import importlib
    _bam = importlib.import_module("hea.models.bam")
    if _bam._rs_xwx_smooth_block is None:
        pytest.skip("hea._rs.xwx_smooth_block unavailable")
    import polars as pl
    from hea.family import Gaussian
    from hea.models.bam import _ar1_rwmatrix_indices

    def check(formula, df):
        m = _bam.bam(formula, df, family=Gaussian(), discrete=True)
        d = m._discrete_design
        rng = np.random.default_rng(0)
        w = rng.uniform(0.1, 2.0, d.n)
        ld = 1.0 / np.sqrt(1 - 0.5 ** 2)
        _, _, wt = _ar1_rwmatrix_indices(d.n, ld, -0.5 * ld, None)
        got = _bam.XWXd(d, w, ar_weights=wt)
        orig = _bam._rs_xwx_smooth_block
        _bam._rs_xwx_smooth_block = None
        try:
            want = _bam.XWXd(d, w, ar_weights=wt)
        finally:
            _bam._rs_xwx_smooth_block = orig
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)

    rng = np.random.default_rng(11)
    n = 800
    check("y ~ te(x, z, k=c(5,5))",      # plain tensor AR1 (s_i=1, nd_i>1)
          pl.DataFrame({"y": rng.standard_normal(n),
                        "x": rng.uniform(0, 1, n), "z": rng.uniform(0, 1, n)}))
    nm, L = 2000, 6                       # signal-regression AR1 (s_i·s_j=36)
    grid = np.linspace(0, 5, 50)
    Xc = grid[rng.integers(0, 50, (nm, L))]
    check("y ~ te(Lag, Xc, by=Stim, k=c(4,5))",
          pl.DataFrame({"y": rng.standard_normal(nm),
                        "Lag": np.tile(np.linspace(0, 1, L), (nm, 1)),
                        "Xc": Xc, "Stim": rng.standard_normal((nm, L))}))


# ---------------------------------------------------------------------------
# R-optimizer ports (uncmin / L-BFGS-B): rs == python differential.
# Unlike the d/p/q kernels, the Python optimizers involve no numpy
# transcendentals — pure arithmetic + libm sqrt/hypot/pow — so the pure-
# Python modules ARE the bit-exact oracle here (they are pinned to live R
# both by tests/test_r_optimize.py and by the ctypes trajectory oracles
# against libR's compiled optif9/lbfgsb documented in hea/R/uncmin.py).
# The Rust port must reproduce them bit-for-bit including the evaluation
# trajectory.


def test_optif9_rs_python_parity():
    from hea.R.uncmin import optif9 as py_optif9

    def make(track):
        def f_val(x):
            track.append(np.array(x, dtype=float, copy=True))
            v = 0.0
            for i in range(len(x) - 1):
                v += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 \
                     + (1.0 - x[i]) ** 2
            return float(v)

        def f_grad(x):
            n = len(x)
            g = np.zeros(n)
            for i in range(n - 1):
                t = x[i + 1] - x[i] * x[i]
                g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i])
                g[i + 1] += 200.0 * t
            return g
        return f_val, f_grad

    rng = np.random.default_rng(42)
    for n in (2, 3, 4):
        for rep in range(3):
            x0 = rng.standard_normal(n) * (1.0 + rep)
            ts = np.abs(rng.standard_normal(n)) + 0.1
            for msg, ndig, smx, stol, ilim in (
                    (9, 12, 1000.0, 1e-6, 100),
                    (15, 7, 2.0, 1e-4, 200)):
                tr_py, tr_rs = [], []
                fv, fg = make(tr_py)
                py = py_optif9(n, x0.copy(), fv, fg, lambda x, a: None,
                               ts.copy(), 1.0, 1, 1, msg, ndig, ilim,
                               1, 0, 1.0, 1e-6, smx, stol)
                fv2, fg2 = make(tr_rs)
                rs_out = rs.optif9(x0.copy(), fv2, fg2, None, ts.copy(),
                                   1.0, 1, 1, msg, ndig, ilim, 1, 0,
                                   1.0, 1e-6, smx, stol)
                assert len(tr_py) == len(tr_rs)
                for a, b in zip(tr_py, tr_rs):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(py[0], rs_out[0])  # xpls
                assert py[1] == rs_out[1]                        # fpls
                np.testing.assert_array_equal(py[2], rs_out[2])  # gpls
                assert py[3:] == tuple(rs_out[3:])   # itrmcd/itncnt/msg


def test_lbfgsb_rs_python_parity():
    import math

    from hea.R.lbfgsb import lbfgsb as py_lbfgsb

    def f_val(x):
        v = 0.0
        for i in range(len(x) - 1):
            v += 100.0 * (x[i + 1] - x[i] * x[i]) ** 2 \
                 + (1.0 - x[i]) ** 2
        return float(v)

    def f_grad(x):
        n = len(x)
        g = np.zeros(n)
        for i in range(n - 1):
            t = x[i + 1] - x[i] * x[i]
            g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i])
            g[i + 1] += 200.0 * t
        return g

    rng = np.random.default_rng(7)
    for n in (2, 3, 4):
        for rep in range(3):
            x0 = rng.standard_normal(n) * (1 + rep)
            for lo, up, nbd in (
                    ([-math.inf] * n, [math.inf] * n, [0] * n),
                    (list(x0 - 0.7), list(x0 + 0.9), [2] * n),
                    ([-1.0] * n, [math.inf] * n,
                     [1 if i % 2 == 0 else 0 for i in range(n)])):
                m = min(5, n)
                tr_py, tr_rs = [], []

                def fminfn(x, tr=tr_py):
                    tr.append(np.array(x, dtype=float, copy=True))
                    return f_val(x)

                def fmingr(x, g):
                    g[:] = f_grad(x)

                xp = np.array(x0, dtype=float)
                py = py_lbfgsb(n, m, xp, np.array(lo), np.array(up),
                               np.array(nbd, dtype=np.int64), fminfn,
                               fmingr, 1e7, 0.0, 100, 0, 10)

                def fminfn2(x, tr=tr_rs):
                    tr.append(np.array(x, dtype=float, copy=True))
                    return f_val(x)

                def gr_ret(x):
                    return f_grad(x)

                xr, val, fail, fnc, grc, msg = rs.lbfgsb_drive(
                    n, m, np.array(x0, dtype=float), np.array(lo),
                    np.array(up), np.array(nbd, dtype=np.int64),
                    fminfn2, gr_ret, 1e7, 0.0, 100)
                assert len(tr_py) == len(tr_rs)
                for a, b in zip(tr_py, tr_rs):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(xp, xr)
                assert py[0] == val and py[1] == fail
                assert py[2] == fnc and py[3] == grc
                assert py[4] == msg
