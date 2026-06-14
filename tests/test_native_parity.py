"""Differential parity gate: the compiled ``hea._native`` kernels must equal the
pure-Python ``hea.R.nmath`` kernels **bit-for-bit**.

``nmath.py`` is pinned 0-ulp to R (tests/test_R.py), so ``native == python`` here
transitively guarantees ``native == R`` without needing R in CI. If the native
extension is not built, these tests skip (the Python fallback still runs).

T0 covers ``pnorm`` only; extend per kernel as Tier 1 lands.
"""
import numpy as np
import pytest

from hea.R import nmath

# Skip the whole module when the extension isn't compiled in (sdist / no toolchain).
native = pytest.importorskip("hea._native")


def _bits(v: float) -> int:
    return np.float64(v).view(np.int64).item()


def _assert_bit_exact(got, exp):
    """Bit-for-bit equality, NaN- and signed-zero-aware."""
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    assert got.shape == exp.shape
    for g, e in zip(got.ravel(), exp.ravel()):
        if np.isnan(e):
            assert np.isnan(g), f"expected NaN, got {g!r}"
        else:
            assert _bits(g) == _bits(e), f"bit mismatch: native={g!r} python={e!r}"


def _grid() -> np.ndarray:
    # Stress every branch of pnorm_both: central (|x|<=0.6745), mid
    # (<=sqrt(32)), far tail (>sqrt(32)), the cutoffs that gate log_p/tail,
    # tiny (|x|<=eps), zero, and the non-finite lanes.
    pts = [
        -50.0, -40.0, -38.4674, -8.2924, -5.657, -5.0, -1.0, -0.6744,
        -1e-8, -1e-300, 0.0, 1e-300, 1e-8, 0.5, 0.6744, 0.67448975, 1.0,
        5.0, 5.657, 8.2924, 38.0, 40.0, 50.0, 1e170, 1e171,
        np.inf, -np.inf, np.nan,
    ]
    return np.array(pts, dtype=float)


@pytest.mark.parametrize("lower_tail", [True, False])
@pytest.mark.parametrize("log_p", [True, False])
@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5)])
def test_pnorm_bit_exact(lower_tail, log_p, mu, sigma):
    x = _grid()
    got = native.pnorm(x, np.full_like(x, mu), np.full_like(x, sigma), lower_tail, log_p)
    exp = np.array(
        [nmath.pnorm5(float(xi), mu, sigma, lower_tail, log_p) for xi in x]
    )
    _assert_bit_exact(got, exp)


def test_pnorm_degenerate_sigma():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    for sig in (0.0, -1.0):
        for lt in (True, False):
            for lp in (True, False):
                got = native.pnorm(x, np.zeros_like(x), np.full_like(x, sig), lt, lp)
                exp = np.array([nmath.pnorm5(float(xi), 0.0, sig, lt, lp) for xi in x])
                _assert_bit_exact(got, exp)


# ============================================================================
# Tier 1 — lgamma / loader (saddlepoint) foundation
# ============================================================================

def test_lgammafn_gammafn():
    xg = np.array([-9.3, -2.5, -0.5, -1e-8, 1e-307, 1e-200, 0.1, 0.5, 1.0, 1.5,
                   2.0, 3.7, 9.99, 10.0, 10.5, 50.0, 100.0, 4934721.0, 1e17,
                   1e18, 170.0, -170.5, 0.0, -3.0, np.nan, np.inf])
    _assert_bit_exact(native.lgammafn(xg), [nmath._lgammafn(float(v)) for v in xg])
    xg2 = xg[(np.abs(xg) < 171) | ~np.isfinite(xg)]
    _assert_bit_exact(native.gammafn(xg2), [nmath.gammafn(float(v)) for v in xg2])


def test_stirlerr_all_branches():
    ns = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 5.0, 5.25, 5.3, 6.0, 6.1,
                   6.2, 6.6, 6.7, 7.3, 7.4, 8.9, 9.0, 12.3, 12.4, 12.8, 12.9,
                   15.0, 15.5, 16.0, 23.5, 23.6, 24.0, 27.0, 27.1, 86.0, 86.1,
                   205.0, 205.1, 6180.0, 6181.0, 1.57e7, 1.6e7, 2.0e7])
    _assert_bit_exact(native.stirlerr(ns), [float(nmath._stirlerr(float(v))) for v in ns])


def test_bd0_pow1p():
    bx = np.array([5.0, 10.0, 10.0, 0.001, 1e-10, 100.0, 3.0, 1e6, 50.0, 0.5])
    bn = np.array([5.0, 10.0, 9.8, 10.0, 1e-9, 100.5, 30.0, 1.0001e6, 49.0, 5.0])
    _assert_bit_exact(native.bd0(bx, bn), [nmath._bd0(float(a), float(b)) for a, b in zip(bx, bn)])
    px = np.array([0.0, 1e-12, -1e-10, 0.3, -0.7, 2.0, 0.1, -0.05, 1e-8])
    py = np.array([3.0, 100.0, 50.0, 2.5, 4.0, 0.5, 1.0, 1000.0, np.nan])
    _assert_bit_exact(native.pow1p(px, py),
                      [float(nmath._pow1p(float(a), float(b))) for a, b in zip(px, py)])


@pytest.mark.parametrize("give_log", [True, False])
def test_dpois_raw(give_log):
    dx = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 3.0, 0.0, 1e300])
    dl = np.array([3.0, 3.0, 3.0, 4.5, 10.0, 100.0, 1000.0, 0.0, 0.0, 1.0])
    _assert_bit_exact(native.dpois_raw(dx, dl, give_log),
                      [nmath._dpois_raw(float(a), float(b), give_log) for a, b in zip(dx, dl)])


@pytest.mark.parametrize("give_log", [True, False])
def test_dbinom_raw(give_log):
    ex = np.array([0.0, 5.0, 10.0, 3.0, 0.0, 20.0, 7.0, 50.0])
    en = np.array([20.0, 20.0, 20.0, 20.0, 0.0, 20.0, 10.0, 100.0])
    ep = np.array([0.3, 0.3, 0.3, 0.5, 0.5, 1.0, 0.7, 0.1])
    eq = 1.0 - ep
    _assert_bit_exact(
        native.dbinom_raw(ex, en, ep, eq, give_log),
        [nmath._dbinom_raw(float(a), float(b), float(c), float(d), give_log)
         for a, b, c, d in zip(ex, en, ep, eq)])


# ============================================================================
# Tier 1 — gamma family (qnorm5 / dnorm5 / pgamma / dgamma / qgamma)
# ============================================================================

@pytest.mark.parametrize("lower_tail", [True, False])
@pytest.mark.parametrize("log_p", [True, False])
def test_qnorm5(lower_tail, log_p):
    if log_p:
        p = np.array([-700.0, -300.0, -100.0, -50.0, -10.0, -1.0, -0.5, -1e-3,
                      -1e-8, np.log(0.5)])
    else:
        p = np.array([1e-300, 1e-50, 1e-10, 1e-4, 0.001, 0.01, 0.1, 0.25, 0.5,
                      0.75, 0.9, 0.99, 0.999, 1 - 1e-10, 1 - 1e-16, 0.0, 1.0, np.nan])
    _assert_bit_exact(native.qnorm(p, np.zeros_like(p), np.ones_like(p), lower_tail, log_p),
                      [nmath.qnorm5(float(v), 0, 1, lower_tail, log_p) for v in p])


@pytest.mark.parametrize("give_log", [True, False])
def test_dnorm5(give_log):
    x = np.array([-40, -6, -5, -1, 0, 0.5, 1, 5, 6, 38, 40, np.inf, -np.inf, np.nan])
    _assert_bit_exact(native.dnorm(x, np.zeros_like(x), np.ones_like(x), give_log),
                      [nmath.dnorm5(float(v), 0, 1, give_log) for v in x])


# pgamma stress grid hits all 4 branches: smallx (x<1), upper-series, lower, asymp
_PG_X = np.array([0.5, 0.001, 5.0, 20.0, 50.0, 100.0, 2.0, 1e5, 0.3, 1000.0,
                  1e-8, 3.0, 15.0, 0.0, np.inf, 1e-300])
_PG_A = np.array([2.0, 0.5, 20.0, 5.0, 50.0, 2.0, 100.0, 1e4, 0.2, 1000.0,
                  1.0, 3.0, 7.0, 2.0, 2.0, 0.5])


@pytest.mark.parametrize("lower_tail", [True, False])
@pytest.mark.parametrize("log_p", [True, False])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_pgamma(lower_tail, log_p, scale):
    sc = np.full_like(_PG_X, scale)
    _assert_bit_exact(
        native.pgamma(_PG_X, _PG_A, sc, lower_tail, log_p),
        [nmath.pgamma(float(x), float(a), scale, lower_tail, log_p)
         for x, a in zip(_PG_X, _PG_A)])


@pytest.mark.parametrize("give_log", [True, False])
def test_dgamma(give_log):
    x = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 0.001, 100.0, 1e-8, -1.0])
    sh = np.array([0.5, 2.0, 1.0, 3.0, 0.7, 0.5, 50.0, 2.0, 2.0])
    _assert_bit_exact(native.dgamma(x, sh, np.ones_like(x), give_log),
                      [nmath.dgamma(float(a), float(b), 1.0, give_log) for a, b in zip(x, sh)])


@pytest.mark.parametrize("alpha", [0.5, 2.0, 50.0, 1e-11])
def test_qgamma(alpha):
    p = np.array([1e-10, 1e-4, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9999, 1 - 1e-12])
    aa = np.full_like(p, alpha)
    _assert_bit_exact(native.qgamma(p, aa, np.ones_like(p), True, False),
                      [nmath.qgamma(float(v), alpha, 1.0, True, False) for v in p])


# ============================================================================
# Tier 1 — beta (toms708 bratio): pbeta / lbeta
# ============================================================================

def _beta_grid():
    ab = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]
    xs = [1e-8, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1 - 1e-9]
    X, A, B = [], [], []
    for a in ab:
        for b in ab:
            for x in xs:
                X.append(x); A.append(a); B.append(b)
    return np.array(X), np.array(A), np.array(B)


@pytest.mark.parametrize("lower_tail", [True, False])
@pytest.mark.parametrize("log_p", [True, False])
def test_pbeta(lower_tail, log_p):
    X, A, B = _beta_grid()
    _assert_bit_exact(
        native.pbeta(X, A, B, lower_tail, log_p),
        [nmath.pbeta(float(x), float(a), float(b), lower_tail, log_p)
         for x, a, b in zip(X, A, B)])


def test_lbeta():
    ab = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]
    a = np.array([x for x in ab for _ in ab])
    b = np.array([y for _ in ab for y in ab])
    _assert_bit_exact(native.lbeta(a, b),
                      [nmath.lbeta(float(x), float(y)) for x, y in zip(a, b)])


def _qbeta_grid():
    pq = [0.5, 1.0, 2.0, 5.0, 20.0, 100.0]
    al = [1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999, 1 - 1e-10]
    AL, P, Q = [], [], []
    for p in pq:
        for q in pq:
            for a in al:
                AL.append(a); P.append(p); Q.append(q)
    return np.array(AL), np.array(P), np.array(Q)


@pytest.mark.parametrize("lower_tail", [True, False])
def test_qbeta(lower_tail):
    AL, P, Q = _qbeta_grid()
    _assert_bit_exact(
        native.qbeta(AL, P, Q, lower_tail, False),
        [nmath.qbeta(float(a), float(p), float(q), lower_tail, False)
         for a, p, q in zip(AL, P, Q)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_qbeta_log(lower_tail):
    AL, P, Q = _qbeta_grid()
    lAL = np.log(AL)
    _assert_bit_exact(
        native.qbeta(lAL, P, Q, lower_tail, True),
        [nmath.qbeta(float(a), float(p), float(q), lower_tail, True)
         for a, p, q in zip(lAL, P, Q)])


# ============================================================================
# Tier 1 — t / F (pt/pf/dt/qt/qf), discrete (ppois/pbinom/dpois/dbinom/dbeta/
# qpois/qbinom), exponential (dexp/pexp/qexp)
# ============================================================================

@pytest.mark.parametrize("lower_tail", [True, False])
def test_pt(lower_tail):
    x = np.array([-50, -3, -1, -0.1, 0, 0.1, 1, 3, 50, 1e8])
    n = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 0.5, 0.7, 100.0, 1e21, 3.0])
    _assert_bit_exact(native.pt(x, n, lower_tail, False),
                      [nmath.pt(float(a), float(b), lower_tail, False) for a, b in zip(x, n)])


@pytest.mark.parametrize("give_log", [True, False])
def test_dt(give_log):
    x = np.array([-50, -3, -1, -0.1, 0, 0.1, 1, 3, 50, 1e8])
    n = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 0.5, 0.7, 100.0, 1e21, 3.0])
    _assert_bit_exact(native.dt(x, n, give_log),
                      [nmath.dt(float(a), float(b), give_log) for a, b in zip(x, n)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_qt(lower_tail):
    p = np.array([1e-8, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999])
    n = np.array([1.0, 2.0, 2.0, 5.0, 0.5, 10.0, 30.0, 0.7, 100.0, 3.0])
    _assert_bit_exact(native.qt(p, n, lower_tail, False),
                      [nmath.qt(float(a), float(b), lower_tail, False) for a, b in zip(p, n)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_pf(lower_tail):
    x = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 0.3, 3.0, 100.0])
    f1 = np.array([1.0, 2.0, 5.0, 10.0, 3.0, 1.0, 8.0, 0.5, 1e6, 4.0])
    f2 = np.array([1.0, 10.0, 5.0, 2.0, 30.0, 1e6, 8.0, 3.0, 5.0, 20.0])
    _assert_bit_exact(native.pf(x, f1, f2, lower_tail, False),
                      [nmath.pf(float(a), float(b), float(c), lower_tail, False)
                       for a, b, c in zip(x, f1, f2)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_qf(lower_tail):
    p = np.array([1e-8, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999])
    f1 = np.array([1.0, 2.0, 5.0, 10.0, 3.0, 4e5 + 1, 8.0, 0.5, 4.0, 1.0])
    f2 = np.array([1.0, 10.0, 5.0, 2.0, 30.0, 8.0, 4e5 + 1, 3.0, 20.0, 1e6])
    _assert_bit_exact(native.qf(p, f1, f2, lower_tail, False),
                      [nmath.qf(float(a), float(b), float(c), lower_tail, False)
                       for a, b, c in zip(p, f1, f2)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_ppois_qpois(lower_tail):
    x = np.array([0., 1., 2., 5., 10., 50., 3., 7., 0., 100.])
    lam = np.array([3., 3., 3., 4.5, 10., 40., 0.5, 7., 0., 1e3])
    _assert_bit_exact(native.ppois(x, lam, lower_tail, False),
                      [nmath.ppois(float(a), float(b), lower_tail, False) for a, b in zip(x, lam)])
    qp = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.5, 0.3, 1 - 1e-9])
    ql = np.array([3., 3., 10., 4.5, 40., 100., 1e3, 0.5, 7., 5.])
    _assert_bit_exact(native.qpois(qp, ql, lower_tail, False),
                      [nmath.qpois(float(a), float(b), lower_tail, False) for a, b in zip(qp, ql)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_pbinom_qbinom(lower_tail):
    x = np.array([0., 5., 10., 3., 20., 7., 50., 0., 15., 2.])
    n = np.array([20., 20., 20., 20., 20., 10., 100., 5., 30., 2.])
    p = np.array([.3, .3, .3, .5, 1., .7, .1, .5, .4, .5])
    _assert_bit_exact(native.pbinom(x, n, p, lower_tail, False),
                      [nmath.pbinom(float(a), float(b), float(c), lower_tail, False)
                       for a, b, c in zip(x, n, p)])
    qp = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.5, 0.3, 1 - 1e-9])
    qn = np.array([20., 20., 50., 20., 100., 1000., 500., 10., 30., 5.])
    qpr = np.array([.3, .5, .3, .5, .1, .5, .7, .5, .4, .5])
    _assert_bit_exact(native.qbinom(qp, qn, qpr, lower_tail, False),
                      [nmath.qbinom(float(a), float(b), float(c), lower_tail, False)
                       for a, b, c in zip(qp, qn, qpr)])


@pytest.mark.parametrize("give_log", [True, False])
def test_dpois_dbinom_dbeta(give_log):
    px = np.array([0., 1., 2., 5., 10., 50., 3., 7., 0., 100.])
    pl = np.array([3., 3., 3., 4.5, 10., 40., 0.5, 7., 0., 1e3])
    _assert_bit_exact(native.dpois(px, pl, give_log),
                      [nmath.dpois(float(a), float(b), give_log) for a, b in zip(px, pl)])
    bx = np.array([0., 5., 10., 3., 20., 7., 50., 0., 15., 2.])
    bn = np.array([20., 20., 20., 20., 20., 10., 100., 5., 30., 2.])
    bp = np.array([.3, .3, .3, .5, 1., .7, .1, .5, .4, .5])
    _assert_bit_exact(native.dbinom(bx, bn, bp, give_log),
                      [nmath.dbinom(float(a), float(b), float(c), give_log)
                       for a, b, c in zip(bx, bn, bp)])
    dx = np.array([0., 0.01, 0.1, 0.5, 0.9, 0.99, 1., 0.3, 0.5, 0.7])
    da = np.array([2., 0.5, 2., 3., 0.7, 5., 2., 1., 3., 0.5])
    db = np.array([3., 2., 0.5, 3., 2., 1., 0.7, 1., 5., 0.5])
    _assert_bit_exact(native.dbeta(dx, da, db, give_log),
                      [nmath.dbeta(float(a), float(b), float(c), give_log)
                       for a, b, c in zip(dx, da, db)])


@pytest.mark.parametrize("lower_tail", [True, False])
def test_exp(lower_tail):
    x = np.array([0., 0.1, 1., 5., 20., 0., 2., 100., 0.5, 1e-8])
    s = np.array([1., 1., 2., 0.5, 1., 3., 1., 2., 0.7, 1.])
    _assert_bit_exact(native.dexp(x, s, False),
                      [nmath.dexp(float(a), float(b), False) for a, b in zip(x, s)])
    _assert_bit_exact(native.pexp(x, s, lower_tail, False),
                      [nmath.pexp(float(a), float(b), lower_tail, False) for a, b in zip(x, s)])
    qp = np.array([1e-8, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.3, 0.7, 1 - 1e-9])
    _assert_bit_exact(native.qexp(qp, s, lower_tail, False),
                      [nmath.qexp(float(a), float(b), lower_tail, False) for a, b in zip(qp, s)])
