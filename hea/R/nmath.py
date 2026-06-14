"""Bit-exact ports of R's ``src/nmath/`` probability kernels.

R's distribution CDF / quantile / density functions are NOT scipy's: R ships its
own algorithms in ``src/nmath/`` (Cody 1993 for ``pnorm``, Wichura AS-241 for
``qnorm``, Pearson/continued-fraction + saddlepoint for ``pgamma``, TOMS 708 for
``pbeta``, Loader 2000 ``bd0``/``stirlerr`` for the discrete densities). scipy's
Cephes/Boost backends differ from these at the 1-3 ulp level (worse in tails),
which is why hea's d/p/q surface was never bit-exact to R.

This module ports those kernels directly from the R 4.6.0 C source so the public
``hea.R`` d/p/q surface, the probit/cauchit links, the shash family, and every
p-value / quantile that routes through them can be 0-ulp to R. Each function is a
line-by-line translation of the corresponding ``src/nmath/*.c`` file; verify with
the R-oracle harness against live R, requiring 0-ulp (not "rel < 1e-9").

The ``q*`` normal coefficients live in :mod:`hea.R.rng` (``_QN_A`` .. ``_QN_F``,
shared with the ``rnorm`` Inversion path) and are imported here rather than
duplicated.
"""
from __future__ import annotations

import math

import numpy as np

from .rng import _QN_A, _QN_B, _QN_C, _QN_D, _QN_E, _QN_F, _qn_horner
from ._dispatch import rs_fn

# Rust kernels — None when the extension is absent/disabled, in which
# case the pure-Python kernels below run unchanged (bit-identical, just slower).
_rs_pnorm = rs_fn("pnorm")
_rs_qnorm = rs_fn("qnorm")
_rs_dnorm = rs_fn("dnorm")

# name -> numpy-vectorized pure-Python kernel, used by _disp as the no-native
# fallback (populated at end of module, after the kernels are defined).
_PY_VEC = {}


def _norm_rs(kern, x, mu, sigma, flags):
    """Broadcast (x, mu, sigma), run the native norm kernel, reshape. The native
    norm kernels take mu/sigma as arrays (uniform with the rest of the surface),
    so array mean/sd route through Rust too — no scalar-loop special case."""
    xa, ma, sa = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float))
    return kern(
        np.ascontiguousarray(xa.reshape(-1)), np.ascontiguousarray(ma.reshape(-1)),
        np.ascontiguousarray(sa.reshape(-1)), *flags).reshape(xa.shape)

# --- R constants (Rmath.h) ----------------------------------------------------
_M_SQRT_32 = 5.656854249492380195206754896838      # sqrt(32)
_M_1_SQRT_2PI = 0.398942280401432677939946059934   # 1/sqrt(2pi)
_M_LN_SQRT_2PI = 0.918938533204672741780329736406  # log(sqrt(2pi))
_M_2PI = 6.283185307179586476925286766559          # 2*pi
_M_SQRT2 = 1.414213562373095048801688724210        # sqrt(2)
_M_LN2 = 0.693147180559945309417232121458          # ln(2)
_M_LN_2PI = 1.8378770664093454835606594728112352798  # log(2*pi)
_M_SQRT_2PI = 2.50662827463100050241576528481104525301  # sqrt(2*pi)
_X_LRG = 2.86111748575702815380240589208115399625e307  # 2^1023 / pi
_DBL_EPSILON = 2.220446049250313080847e-16
_INF = math.inf
_NEGINF = -math.inf
_NAN = math.nan


def _disp(name, scalar_fn, num_args, flags=()):
    """Native-accelerated vectorised dispatch with pure-Python fallback.

    ``name`` is the :mod:`hea._rs` kernel (e.g. ``"pgamma"``); ``scalar_fn``
    is the matching :mod:`hea.R.nmath` scalar kernel; ``num_args`` are the
    numeric (array-or-scalar) arguments in native-call order; ``flags`` are the
    trailing bool flags (lower_tail/log_p/give_log). When the extension is built
    the kernel runs in Rust (broadcast → flat → reshape, 0-ulp to the Python
    path); otherwise the scalar loop runs. Scalar inputs → Python float.
    """
    kern = rs_fn(name)
    arrs = [np.asarray(a, dtype=float) for a in num_args]
    scalar = all(a.ndim == 0 for a in arrs)
    if kern is not None:
        if scalar:
            r = kern(*[a.reshape(1) for a in arrs], *flags)
            return float(r[0])
        barr = np.broadcast_arrays(*arrs)
        shape = barr[0].shape
        flat = [np.ascontiguousarray(a.reshape(-1)) for a in barr]
        return kern(*flat, *flags).reshape(shape)
    # No native extension: use the numpy-vectorized pure-Python kernel if one is
    # registered (bit-identical, ~C-speed); else the scalar loop (the bratio /
    # Newton-quantile kernels are not vectorizable cheaply — see plan §5).
    py_vec = _PY_VEC.get(name)
    if py_vec is not None:
        r = py_vec(*num_args, *flags)
        return float(r) if scalar else r
    return _vec(lambda *a: scalar_fn(*a, *flags), *num_args)


def _vec(fn, *args):
    """Apply scalar nmath ``fn`` over R-recycled/broadcast ``args``.

    Scalar args + scalar result → Python float (matches scipy's scalar return);
    otherwise broadcast to a numpy array, mirroring R's vectorised recycling.
    """
    arrs = [np.asarray(a, dtype=float) for a in args]
    if all(a.ndim == 0 for a in arrs):
        return float(fn(*[float(a) for a in arrs]))
    barr = np.broadcast_arrays(*arrs)
    shape = barr[0].shape
    flat = [a.ravel() for a in barr]
    out = np.empty(flat[0].size, dtype=float)
    for i in range(out.size):
        out[i] = fn(*[float(f[i]) for f in flat])
    return out.reshape(shape)


# === qnorm5 — normal quantile (nmath/qnorm.c, Wichura AS-241) =================
def qnorm5(p: float, mu: float = 0.0, sigma: float = 1.0,
           lower_tail: bool = True, log_p: bool = False) -> float:
    """R's ``qnorm5(p, mu, sigma, lower_tail, log_p)`` — full semantics, bit-exact.

    Unlike :func:`hea.R.rng._qnorm5` (the ``lower_tail=TRUE, log_p=FALSE`` fast
    path feeding ``rnorm``), this replicates R's ``R_DT_qIv`` handling
    (``0.5 - p + 0.5`` idiom), the ``log_p`` tail formulas, and the ``r > 27``
    asymptotic expansion for the extreme tail.
    """
    if math.isnan(p) or math.isnan(mu) or math.isnan(sigma):
        return p + mu + sigma
    # R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF)
    if log_p:
        if p > 0:
            return _NAN
        if p == 0:
            return _INF if lower_tail else _NEGINF
        if p == _NEGINF:
            return _NEGINF if lower_tail else _INF
    else:
        if p < 0 or p > 1:
            return _NAN
        if p == 0:
            return _NEGINF if lower_tail else _INF
        if p == 1:
            return _INF if lower_tail else _NEGINF
    if sigma < 0:
        return _NAN
    if sigma == 0:
        return mu

    # p_ = R_DT_qIv(p): real lower-tail probability
    if log_p:
        p_ = math.exp(p) if lower_tail else -math.expm1(p)
    else:
        p_ = p if lower_tail else (0.5 - p + 0.5)
    q = p_ - 0.5

    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        val = q * _qn_horner(r, _QN_A) / _qn_horner(r, _QN_B)
        return mu + sigma * val

    # closer than 0.075 from {0,1}: r := log(min(p, 1-p))
    if log_p and ((lower_tail and q <= 0) or ((not lower_tail) and q > 0)):
        lp = p
    else:
        if q > 0:
            # R_DT_CIv(p) == 1 - p
            if log_p:
                civ = -math.expm1(p) if lower_tail else math.exp(p)
            else:
                civ = (0.5 - p + 0.5) if lower_tail else p
            lp = math.log(civ)
        else:
            lp = math.log(p_)
    r = math.sqrt(-lp)

    if r <= 5.0:
        r += -1.6
        val = _qn_horner(r, _QN_C) / _qn_horner(r, _QN_D)
    elif r <= 27.0:
        r += -5.0
        val = _qn_horner(r, _QN_E) / _qn_horner(r, _QN_F)
    else:  # r > 27: extreme tail asymptotic (practically only log_p=TRUE)
        if r >= 6.4e8:
            val = r * _M_SQRT2
        else:
            s2 = -math.ldexp(lp, 1)            # = 2s
            x2 = s2 - math.log(_M_2PI * s2)    # xs_1
            if r < 36000.0:
                x2 = s2 - math.log(_M_2PI * x2) - 2.0 / (2.0 + x2)  # xs_2
                if r < 840.0:
                    x2 = (s2 - math.log(_M_2PI * x2)
                          + 2.0 * math.log1p(-(1 - 1 / (4 + x2)) / (2.0 + x2)))
                    if r < 109.0:
                        x2 = (s2 - math.log(_M_2PI * x2) + 2.0 * math.log1p(
                            -(1 - (1 - 5 / (6 + x2)) / (4.0 + x2)) / (2.0 + x2)))
                        if r < 55.0:
                            x2 = (s2 - math.log(_M_2PI * x2) + 2.0 * math.log1p(
                                -(1 - (1 - (5 - 9 / (8.0 + x2)) / (6.0 + x2))
                                  / (4.0 + x2)) / (2.0 + x2)))
            val = math.sqrt(x2)
    if q < 0.0:
        val = -val
    return mu + sigma * val


# === dnorm5 — normal density (nmath/dnorm.c) =================================
# sqrt(-2*ln2*(DBL_MIN_EXP+1-DBL_MANT_DIG)) with DBL_MIN_EXP=-1021, MANT_DIG=53
_DNORM_BIG = math.sqrt(-2.0 * _M_LN2 * (-1021 + 1 - 53))
_TWO_SQRT_DBL_MAX = 2.0 * math.sqrt(1.7976931348623157e308)


def dnorm5(x: float, mu: float = 0.0, sigma: float = 1.0,
           give_log: bool = False) -> float:
    """R's ``dnorm4/dnorm5`` (nmath/dnorm.c), bit-exact (non-FAST variant)."""
    if math.isnan(x) or math.isnan(mu) or math.isnan(sigma):
        return x + mu + sigma
    rd0 = _NEGINF if give_log else 0.0
    if math.isinf(sigma):
        return rd0
    if math.isinf(x) and mu == x:
        return _NAN
    if sigma <= 0:
        if sigma < 0:
            return _NAN
        return _INF if x == mu else rd0
    x = (x - mu) / sigma
    if math.isinf(x):
        return rd0
    x = abs(x)
    if x >= _TWO_SQRT_DBL_MAX:
        return rd0
    if give_log:
        return -(_M_LN_SQRT_2PI + 0.5 * x * x + math.log(sigma))
    if x < 5.0:
        return _M_1_SQRT_2PI * math.exp(-0.5 * x * x) / sigma
    if x > _DNORM_BIG:
        return 0.0
    x1 = math.ldexp(round(math.ldexp(x, 16)), -16)
    x2 = x - x1
    return _M_1_SQRT_2PI / sigma * (math.exp(-0.5 * x1 * x1)
                                    * math.exp((-0.5 * x2 - x1) * x2))


def dnorm5_vec(x, mu=0.0, sigma=1.0, give_log=False):
    """Vectorised :func:`dnorm5`; bit-identical to the scalar version."""
    if _rs_dnorm is not None:
        return _norm_rs(_rs_dnorm, x, mu, sigma, (bool(give_log),))
    if np.ndim(mu) or np.ndim(sigma):  # array params w/o native → scalar loop
        return _vec(lambda v, m, s: dnorm5(v, m, s, give_log), x, mu, sigma)
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=float)
    rd0 = _NEGINF if give_log else 0.0
    if math.isnan(mu) or math.isnan(sigma) or sigma <= 0 or math.isinf(sigma):
        it = np.nditer(x, flags=["multi_index"])
        for v in it:
            out[it.multi_index] = dnorm5(float(v), mu, sigma, give_log)
        return out
    z = np.abs((x - mu) / sigma)
    if give_log:
        out = -(_M_LN_SQRT_2PI + 0.5 * z * z + math.log(sigma))
        out = np.where(np.isnan(x), x, out)
        out = np.where(np.isinf(z), rd0, out)
        return out
    out = np.full(x.shape, np.nan, dtype=float)
    small = z < 5.0
    out[small] = _M_1_SQRT_2PI * np.exp(-0.5 * z[small] * z[small]) / sigma
    big = (~small) & (z <= _DNORM_BIG) & np.isfinite(z)
    if big.any():
        xb = z[big]
        x1 = np.ldexp(np.rint(np.ldexp(xb, 16)), -16)
        x2 = xb - x1
        out[big] = _M_1_SQRT_2PI / sigma * (np.exp(-0.5 * x1 * x1)
                                            * np.exp((-0.5 * x2 - x1) * x2))
    out = np.where((~small) & (z > _DNORM_BIG) & np.isfinite(z), 0.0, out)
    out = np.where(np.isinf(z), rd0, out)
    out = np.where(np.isnan(x), x, out)
    return out


# === pnorm5 — normal CDF (nmath/pnorm.c, Cody 1993) ==========================
_PN_A = (2.2352520354606839287, 161.02823106855587881, 1067.6894854603709582,
         18154.981253343561249, 0.065682337918207449113)
_PN_B = (47.20258190468824187, 976.09855173777669322, 10260.932208618978205,
         45507.789335026729956)
_PN_C = (0.39894151208813466764, 8.8831497943883759412, 93.506656132177855979,
         597.27027639480026226, 2494.5375852903726711, 6848.1904505362823326,
         11602.651437647350124, 9842.7148383839780218, 1.0765576773720192317e-8)
_PN_D = (22.266688044328115691, 235.38790178262499861, 1519.377599407554805,
         6485.558298266760755, 18615.571640885098091, 34900.952721145977266,
         38912.003286093271411, 19685.429676859990727)
_PN_P = (0.21589853405795699, 0.1274011611602473639, 0.022235277870649807,
         0.001421619193227893466, 2.9112874951168792e-5, 0.02307344176494017303)
_PN_Q = (1.28426009614491121, 0.468238212480865118, 0.0659881378689285515,
         0.00378239633202758244, 7.29751555083966205e-5)


def _pnorm_both(x: float, i_tail: int, log_p: bool):
    """R's ``pnorm_both(x, &cum, &ccum, i_tail, log_p)`` (nmath/pnorm.c).

    ``i_tail`` in {0,1,2} = {lower, upper, both}. Returns ``(cum, ccum)``;
    the entry not requested by ``i_tail`` may be left as ``nan``.
    """
    if math.isnan(x):
        return x, x
    eps = _DBL_EPSILON * 0.5
    lower = i_tail != 1
    upper = i_tail != 0
    cum = _NAN
    ccum = _NAN
    y = abs(x)
    if y <= 0.67448975:
        if y > eps:
            xsq = x * x
            xnum = _PN_A[4] * xsq
            xden = xsq
            for i in range(3):
                xnum = (xnum + _PN_A[i]) * xsq
                xden = (xden + _PN_B[i]) * xsq
        else:
            xnum = xden = 0.0
        temp = x * (xnum + _PN_A[3]) / (xden + _PN_B[3])
        if lower:
            cum = 0.5 + temp
        if upper:
            ccum = 0.5 - temp
        if log_p:
            if lower:
                cum = math.log(cum)
            if upper:
                ccum = math.log(ccum)
        return cum, ccum

    if y <= _M_SQRT_32:
        # qnorm(3/4) < |x| <= sqrt(32) ~ 5.657
        xnum = _PN_C[8] * y
        xden = y
        for i in range(7):
            xnum = (xnum + _PN_C[i]) * y
            xden = (xden + _PN_D[i]) * y
        temp = (xnum + _PN_C[7]) / (xden + _PN_D[7])
        # do_del(y); swap_tail
        xsq = math.ldexp(math.trunc(math.ldexp(y, 4)), -4)
        del_ = (y - xsq) * (y + xsq)
        if log_p:
            cum = (-xsq * math.ldexp(xsq, -1)) - math.ldexp(del_, -1) + math.log(temp)
            if (lower and x > 0.0) or (upper and x <= 0.0):
                ccum = math.log1p(-math.exp(-xsq * math.ldexp(xsq, -1))
                                  * math.exp(-math.ldexp(del_, -1)) * temp)
        else:
            cum = (math.exp(-xsq * math.ldexp(xsq, -1))
                   * math.exp(-math.ldexp(del_, -1)) * temp)
            ccum = 1.0 - cum
        if x > 0.0:  # swap_tail
            temp = cum
            if lower:
                cum = ccum
            ccum = temp
        return cum, ccum

    # |x| > sqrt(32)
    if ((log_p and y < 1e170)
            or (lower and -38.4674 < x < 8.2924)
            or (upper and -8.2924 < x < 38.4674)):
        xsq = 1.0 / (x * x)
        xnum = _PN_P[5] * xsq
        xden = xsq
        for i in range(4):
            xnum = (xnum + _PN_P[i]) * xsq
            xden = (xden + _PN_Q[i]) * xsq
        temp = xsq * (xnum + _PN_P[4]) / (xden + _PN_Q[4])
        temp = (_M_1_SQRT_2PI - temp) / y
        # do_del(x); swap_tail
        xsq = math.ldexp(math.trunc(math.ldexp(x, 4)), -4)
        del_ = (x - xsq) * (x + xsq)
        if log_p:
            cum = (-xsq * math.ldexp(xsq, -1)) - math.ldexp(del_, -1) + math.log(temp)
            if (lower and x > 0.0) or (upper and x <= 0.0):
                ccum = math.log1p(-math.exp(-xsq * math.ldexp(xsq, -1))
                                  * math.exp(-math.ldexp(del_, -1)) * temp)
        else:
            cum = (math.exp(-xsq * math.ldexp(xsq, -1))
                   * math.exp(-math.ldexp(del_, -1)) * temp)
            ccum = 1.0 - cum
        if x > 0.0:  # swap_tail
            temp = cum
            if lower:
                cum = ccum
            ccum = temp
        return cum, ccum

    # large |x|: probs are 0 or 1.  R_D__1 = log_p?0:1 ; R_D__0 = log_p?-inf:0
    rd0 = _NEGINF if log_p else 0.0
    rd1 = 0.0 if log_p else 1.0
    if x > 0:
        cum, ccum = rd1, rd0
    else:
        cum, ccum = rd0, rd1
    return cum, ccum


def pnorm5(x: float, mu: float = 0.0, sigma: float = 1.0,
           lower_tail: bool = True, log_p: bool = False) -> float:
    """R's ``pnorm5(x, mu, sigma, lower_tail, log_p)`` (nmath/pnorm.c), bit-exact."""
    if math.isnan(x) or math.isnan(mu) or math.isnan(sigma):
        return x + mu + sigma
    if math.isinf(x) and mu == x:
        return _NAN
    if sigma <= 0:
        if sigma < 0:
            return _NAN
        # sigma == 0 : return (x < mu) ? R_DT_0 : R_DT_1
        return _dt0(lower_tail, log_p) if x < mu else _dt1(lower_tail, log_p)
    p = (x - mu) / sigma
    if math.isinf(p):
        below = x < mu
        # R_DT_0 if below else R_DT_1
        if below:
            return _dt0(lower_tail, log_p)
        return _dt1(lower_tail, log_p)
    x = p
    cum, ccum = _pnorm_both(x, 0 if lower_tail else 1, log_p)
    return cum if lower_tail else ccum


def _dt0(lower_tail, log_p):
    # R_DT_0 = lower_tail ? R_D__0 : R_D__1
    if lower_tail:
        return _NEGINF if log_p else 0.0
    return 0.0 if log_p else 1.0


def _dt1(lower_tail, log_p):
    # R_DT_1 = lower_tail ? R_D__1 : R_D__0
    if lower_tail:
        return 0.0 if log_p else 1.0
    return _NEGINF if log_p else 0.0


# === Vectorised fast paths (numpy) ===========================================
# The common case (log_p=False, finite/in-range argument, non-extreme tail) is
# evaluated with numpy across the whole array; the rare lanes (nan, boundaries,
# log_p=True, qnorm r>27 extreme tail) fall back to the bit-exact scalar
# functions above, so the result is identical to a scalar loop but ~elementwise.

def qnorm5_vec(p, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Vectorised :func:`qnorm5`; bit-identical to the scalar version."""
    if _rs_qnorm is not None:
        return _norm_rs(_rs_qnorm, p, mu, sigma, (bool(lower_tail), bool(log_p)))
    if np.ndim(mu) or np.ndim(sigma):  # array params w/o native → scalar loop
        return _vec(lambda v, m, s: qnorm5(v, m, s, lower_tail, log_p), p, mu, sigma)
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    if log_p:  # rare — scalar fallback for the whole array
        it = np.nditer(p, flags=["multi_index"])
        for v in it:
            out[it.multi_index] = qnorm5(float(v), mu, sigma, lower_tail, True)
        return out
    reg = (p > 0.0) & (p < 1.0)            # regular open-interval lanes
    pr = p[reg]
    p_ = pr if lower_tail else (0.5 - pr + 0.5)
    q = p_ - 0.5
    res = np.full(pr.shape, np.nan, dtype=float)
    central = np.abs(q) <= 0.425
    qc = q[central]
    rc = 0.180625 - qc * qc
    res[central] = qc * _qn_horner(rc, _QN_A) / _qn_horner(rc, _QN_B)
    tail = ~central
    qt = q[tail]
    civ = (0.5 - pr[tail] + 0.5) if lower_tail else pr[tail]   # R_DT_CIv (log_p=F)
    lp = np.log(np.where(qt > 0.0, civ, p_[tail]))
    r = np.sqrt(-lp)
    valt = np.full(qt.shape, np.nan, dtype=float)   # r>27 lanes stay nan
    near = r <= 5.0
    rn = r[near] - 1.6
    valt[near] = _qn_horner(rn, _QN_C) / _qn_horner(rn, _QN_D)
    far = (~near) & (r <= 27.0)
    rfr = r[far] - 5.0
    valt[far] = _qn_horner(rfr, _QN_E) / _qn_horner(rfr, _QN_F)
    valt = np.where(qt < 0.0, -valt, valt)
    res[tail] = valt
    out[reg] = mu + sigma * res
    # fallback (scalar, bit-exact) for nan lanes: p<=0, p>=1, nan input, or the
    # r>27 extreme-tail lanes left nan above.
    fb = np.isnan(out)
    if fb.any():
        pb = p[fb]
        out[fb] = np.array([qnorm5(float(pp), mu, sigma, lower_tail, False)
                            for pp in pb])
    return out


def pnorm5_vec(x, mu=0.0, sigma=1.0, lower_tail=True, log_p=False):
    """Vectorised :func:`pnorm5`; bit-identical to the scalar version."""
    if _rs_pnorm is not None:
        return _norm_rs(_rs_pnorm, x, mu, sigma, (bool(lower_tail), bool(log_p)))
    if np.ndim(mu) or np.ndim(sigma):  # array params w/o native → scalar loop
        return _vec(lambda v, m, s: pnorm5(v, m, s, lower_tail, log_p), x, mu, sigma)
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=float)
    if log_p:  # rare — scalar fallback for the whole array
        it = np.nditer(x, flags=["multi_index"])
        for v in it:
            out[it.multi_index] = pnorm5(float(v), mu, sigma, lower_tail, True)
        return out
    if sigma <= 0 or not np.isfinite(mu) or np.isnan(x).any():
        it = np.nditer(x, flags=["multi_index"])
        for v in it:
            out[it.multi_index] = pnorm5(float(v), mu, sigma, lower_tail, False)
        return out
    z = (x - mu) / sigma
    lower = lower_tail
    y = np.abs(z)
    eps = _DBL_EPSILON * 0.5
    out[:] = np.nan

    # region 1: y <= 0.67448975 (no swap_tail)
    m1 = y <= 0.67448975
    if m1.any():
        x1 = z[m1]
        xsq = x1 * x1
        big = y[m1] > eps
        xnum = np.where(big, _PN_A[4] * xsq, 0.0)
        xden = np.where(big, xsq, 0.0)
        for i in range(3):
            xnum = np.where(big, (xnum + _PN_A[i]) * xsq, xnum)
            xden = np.where(big, (xden + _PN_B[i]) * xsq, xden)
        temp = x1 * (xnum + _PN_A[3]) / (xden + _PN_B[3])
        out[m1] = (0.5 + temp) if lower else (0.5 - temp)

    # do_del epilogue (non-log path) returning (c, cc) = (cum0, ccum0)
    def _do_del(xv, temp):
        xsq = np.ldexp(np.trunc(np.ldexp(xv, 4)), -4)
        del_ = (xv - xsq) * (xv + xsq)
        c = np.exp(-xsq * np.ldexp(xsq, -1)) * np.exp(-np.ldexp(del_, -1)) * temp
        return c, 1.0 - c

    # swap_tail picks the requested tail: lower -> where(x>0, cc, c);
    #                                     upper -> where(x>0, c, cc)
    def _sel(xv, c, cc):
        sw = xv > 0.0
        return np.where(sw, cc, c) if lower else np.where(sw, c, cc)

    # region 2: 0.674.. < y <= sqrt(32)
    m2 = (~m1) & (y <= _M_SQRT_32)
    if m2.any():
        yv = y[m2]
        xnum = _PN_C[8] * yv
        xden = yv
        for i in range(7):
            xnum = (xnum + _PN_C[i]) * yv
            xden = (xden + _PN_D[i]) * yv
        temp = (xnum + _PN_C[7]) / (xden + _PN_D[7])
        c, cc = _do_del(yv, temp)
        out[m2] = _sel(z[m2], c, cc)

    # region 3: y > sqrt(32) within finite range for the requested tail
    m_rest = (~m1) & (~m2)
    in_rng = ((-38.4674 < z) & (z < 8.2924)) if lower \
        else ((-8.2924 < z) & (z < 38.4674))
    m3 = m_rest & in_rng
    if m3.any():
        xv = z[m3]
        xsqi = 1.0 / (xv * xv)
        xnum = _PN_P[5] * xsqi
        xden = xsqi
        for i in range(4):
            xnum = (xnum + _PN_P[i]) * xsqi
            xden = (xden + _PN_Q[i]) * xsqi
        temp = xsqi * (xnum + _PN_P[4]) / (xden + _PN_Q[4])
        temp = (_M_1_SQRT_2PI - temp) / y[m3]
        c, cc = _do_del(xv, temp)
        out[m3] = _sel(xv, c, cc)

    # region 4: large |x| -> probs 0/1
    m4 = m_rest & (~in_rng)
    if m4.any():
        pos = z[m4] > 0
        if lower:
            out[m4] = np.where(pos, 1.0, 0.0)
        else:
            out[m4] = np.where(pos, 0.0, 1.0)
    return out


# === Loader saddlepoint kernels (stirlerr/bd0/ebd0 -> dpois_raw/
# dbinom_raw), moved here from family.py so nmath stays the leaf module
# that family/distributions build on. Bit-exact ports of nmath
# stirlerr.c / bd0.c / dpois.c / dbinom.c. =========================

# ---------------------------------------------------------------------------
# R nmath ports — bit-exact ``dpois`` / ``dbinom`` (saddlepoint algorithm,
# Loader 1999). Used by ``Poisson.aic`` and ``Binomial.aic`` so that the
# Laplace deviance reported by hea matches ``rho$resp$aic()`` from lme4 at
# the ULP level. scipy's ``poisson.logpmf`` / ``binom.logpmf`` use the
# direct formula ``y·log(μ) - μ - lgamma(y+1)`` (and analog for binomial),
# which differs from R's ``dpois`` / ``dbinom`` by ~1 ULP per call — and
# that 1 ULP compounded over n obs is what propagates into deriv12's
# numerator and produces visible SE / vcov gaps against R.
#
# Sources (R 4.5):
# - /tmp/R-src/src/nmath/stirlerr.c
# - /tmp/R-src/src/nmath/bd0.c   (both bd0 and ebd0)
# - /tmp/R-src/src/nmath/dpois.c
# - /tmp/R-src/src/nmath/dbinom.c
# ---------------------------------------------------------------------------


# stirlerr(n) = log(n!) - log(sqrt(2πn)·(n/e)ⁿ)
# Exact table for half-integer arguments 0, 0.5, 1.0, …, 15.0
# (stirlerr.c:78-110).
_STIRLERR_HALVES = (
    0.0,                              # n=0 — placeholder, never used
    0.1534264097200273452913848,      # 0.5
    0.0810614667953272582196702,      # 1.0
    0.0548141210519176538961390,      # 1.5
    0.0413406959554092940938221,      # 2.0
    0.03316287351993628748511048,     # 2.5
    0.02767792568499833914878929,     # 3.0
    0.02374616365629749597132920,     # 3.5
    0.02079067210376509311152277,     # 4.0
    0.01848845053267318523077934,     # 4.5
    0.01664469118982119216319487,     # 5.0
    0.01513497322191737887351255,     # 5.5
    0.01387612882307074799874573,     # 6.0
    0.01281046524292022692424986,     # 6.5
    0.01189670994589177009505572,     # 7.0
    0.01110455975820691732662991,     # 7.5
    0.010411265261972096497478567,    # 8.0
    0.009799416126158803298389475,    # 8.5
    0.009255462182712732917728637,    # 9.0
    0.008768700134139385462952823,    # 9.5
    0.008330563433362871256469318,    # 10.0
    0.007934114564314020547248100,    # 10.5
    0.007573675487951840794972024,    # 11.0
    0.007244554301320383179543912,    # 11.5
    0.006942840107209529865664152,    # 12.0
    0.006665247032707682442354394,    # 12.5
    0.006408994188004207068439631,    # 13.0
    0.006171712263039457647532867,    # 13.5
    0.005951370112758847735624416,    # 14.0
    0.005746216513010115682023589,    # 14.5
    0.005554733551962801371038690,    # 15.0
)

# Asymptotic-series coefficients (stirlerr.c:56-72).
_S0  = 0.083333333333333333333          # 1/12
_S1  = 0.00277777777777777777778        # 1/360
_S2  = 0.00079365079365079365079365     # 1/1260
_S3  = 0.000595238095238095238095238    # 1/1680
_S4  = 0.0008417508417508417508417508   # 1/1188
_S5  = 0.0019175269175269175269175262   # 691/360360
_S6  = 0.0064102564102564102564102561   # 1/156
_S7  = 0.029550653594771241830065352    # 3617/122400
_S8  = 0.17964437236883057316493850     # 43867/244188
_S9  = 1.3924322169059011164274315      # 174611/125400
_S10 = 13.402864044168391994478957      # 77683/5796
_S11 = 156.84828462600201730636509      # 236364091/1506960
_S12 = 2193.1033333333333333333333      # 657931/300
_S13 = 36108.771253724989357173269      # 3392780147/93960
_S14 = 691472.26885131306710839498      # 1723168255201/2492028
_S15 = 15238221.539407416192283370      # 7709321041217/505920
_S16 = 382900751.39141414141414141      # 151628697551/396



_STIRLERR_HALVES_ARR = np.array(_STIRLERR_HALVES, dtype=float)


def _stirlerr(n):
    """Port of nmath ``stirlerr(n)`` (stirlerr.c). Vectorized over ``n``.

    Returns log(n!) - log(sqrt(2πn)·(n/e)ⁿ). The error term in
    Stirling's formula. Used by Loader's saddlepoint algorithm for
    dpois/dbinom. Accepts a scalar or array; returns the same shape.
    Bit-identical to the scalar Fortran source — branches via
    ``np.where``, all arithmetic ops in the same order.
    """
    n = np.asarray(n, dtype=float)
    scalar_input = (n.ndim == 0)
    n = np.atleast_1d(n)

    out = np.empty_like(n)
    nn2 = n + n
    nn2_int = np.rint(nn2).astype(np.int64)

    # ---- n <= 23.5 ----
    le_235 = n <= 23.5
    # Table path: n <= 15.0 and 2n is integer.
    table_mask = le_235 & (n <= 15.0) & (nn2 == nn2_int)
    if np.any(table_mask):
        idx = nn2_int[table_mask]
        out[table_mask] = _STIRLERR_HALVES_ARR[idx]

    # MM2 (n>=1, n<=5.25, not in table)
    mm2_mask = le_235 & ~table_mask & (n <= 5.25) & (n >= 1.0)
    if np.any(mm2_mask):
        nm = n[mm2_mask]
        l_n = np.log(nm)
        out[mm2_mask] = (_lgammafn_arr(nm) + nm * (1.0 - l_n)
                         + (l_n - _M_LN_2PI) * 0.5)

    # n < 1, not in table
    lt1_mask = le_235 & ~table_mask & ~mm2_mask & (n < 1.0)
    if np.any(lt1_mask):
        nm = n[lt1_mask]
        out[lt1_mask] = (_lgammafn_arr(1.0 + nm) - (nm + 0.5) * np.log(nm)
                         + nm - _M_LN_SQRT_2PI)

    # 5.25 < n <= 23.5 — asymptotic series, branches by n threshold.
    series_mask = le_235 & ~table_mask & ~mm2_mask & ~lt1_mask
    if np.any(series_mask):
        nm = n[series_mask]
        nn = nm * nm
        # We need different series lengths per element. Compute the longest
        # branch (k=16) and shorter ones; np.where picks per element.
        s_k7  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - _S6 / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k8  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - _S7 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k9  = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - _S8 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k11 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - _S10 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k13 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - _S12 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k15 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - (_S12 - (_S13 - _S14 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        s_k16 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - (_S5 - (_S6 - (_S7 - (_S8 - (_S9 - (_S10 - (_S11 - (_S12 - (_S13 - (_S14 - (_S15 - _S16 / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nn) / nm
        # Select per-element by threshold.
        ser = np.where(nm > 12.8, s_k7,
              np.where(nm > 12.3, s_k8,
              np.where(nm > 8.9,  s_k9,
              np.where(nm > 7.3,  s_k11,
              np.where(nm > 6.6,  s_k13,
              np.where(nm > 6.1,  s_k15, s_k16))))))
        out[series_mask] = ser

    # ---- n > 23.5 ----
    gt235 = ~le_235
    if np.any(gt235):
        nm = n[gt235]
        nn = nm * nm
        a_k1 = _S0 / nm
        a_k2 = (_S0 - _S1 / nn) / nm
        a_k3 = (_S0 - (_S1 - _S2 / nn) / nn) / nm
        a_k4 = (_S0 - (_S1 - (_S2 - _S3 / nn) / nn) / nn) / nm
        a_k5 = (_S0 - (_S1 - (_S2 - (_S3 - _S4 / nn) / nn) / nn) / nn) / nm
        a_k6 = (_S0 - (_S1 - (_S2 - (_S3 - (_S4 - _S5 / nn) / nn) / nn) / nn) / nn) / nm
        a = np.where(nm > 15.7e6, a_k1,
            np.where(nm > 6180.0, a_k2,
            np.where(nm > 205.0,  a_k3,
            np.where(nm > 86.0,   a_k4,
            np.where(nm > 27.0,   a_k5, a_k6)))))
        out[gt235] = a

    return float(out[0]) if scalar_input else out


def _bd0(x, np_):
    """Port of nmath ``bd0(x, np)`` (bd0.c:48-87). Vectorized.

    Evaluates ``M·D₀(x/M) = x·log(x/M) + M - x`` (where ``M = np_``) with
    small relative error even when ``x/M ≈ 1``. Bit-identical per element
    to the scalar Fortran source — Taylor series for the close branch,
    direct evaluation otherwise.
    """
    x = np.asarray(x, dtype=float)
    np_ = np.asarray(np_, dtype=float)
    scalar = (x.ndim == 0 and np_.ndim == 0)
    x = np.atleast_1d(x)
    np_ = np.atleast_1d(np.broadcast_to(np_, x.shape).copy())

    out = np.empty_like(x)
    out[:] = np.nan
    valid = np.isfinite(x) & np.isfinite(np_) & (np_ != 0.0)

    close = valid & (np.abs(x - np_) < 0.1 * (x + np_))
    far = valid & ~close

    # Far branch: direct formula.
    if np.any(far):
        xf, nf = x[far], np_[far]
        xnp = xf / nf
        # Safe log: fall back to log(x) - log(np_) if xnp non-finite.
        with np.errstate(invalid="ignore"):
            lg_x_n = np.where(np.isfinite(xnp),
                              np.log(np.where(np.isfinite(xnp), xnp, 1.0)),
                              np.log(xf) - np.log(nf))
        out[far] = np.where(xf > nf,
                            xf * (lg_x_n - 1.0) + nf,
                            xf * lg_x_n + nf - xf)

    # Close branch: Taylor series with per-element early exit.
    if np.any(close):
        xc, nc = x[close], np_[close]
        d = xc - nc
        v = d / (xc + nc)
        # Underflow fix: scale by 2^-2 to avoid x+np overflow path.
        underflow = (d != 0.0) & (v == 0.0)
        if np.any(underflow):
            x_ = np.ldexp(xc[underflow], -2)
            n_ = np.ldexp(nc[underflow], -2)
            v_uf = (x_ - n_) / (x_ + n_)
            v[underflow] = v_uf
        s = np.ldexp(d, -1) * v
        # Underflow early-return: ldexp(s, 1) < tiny.
        s2 = np.ldexp(s, 1)
        early = np.abs(s2) < np.finfo(float).tiny
        ej = xc * v
        v2 = v * v
        # Iterate Taylor series; mask out converged/early-returned elements.
        active = ~early
        for j in range(1, 1000):
            if not np.any(active):
                break
            ej_a = ej[active] * v2[active]
            ej[active] = ej_a
            s_old = s[active].copy()
            s_new = s[active] + ej_a / ((j << 1) + 1)
            s[active] = s_new
            still_changed = s_new != s_old
            # Re-build active mask
            idx = np.where(active)[0]
            active = np.zeros_like(active)
            active[idx[still_changed]] = True
        # Return 2*s for converged; 2*early-s for early.
        out[close] = np.where(early, s2, np.ldexp(s, 1))

    return float(out[0]) if scalar else out


# === lgammafn / gammafn (nmath gamma.c / lgamma.c / lgammacor.c) =============
# scipy.special.gammaln is NOT bit-exact to R's lgammafn at small args
# (1-6 ulp at 0.5/1.1/1.5; large near the x=1 root), which leaks into
# stirlerr's small-n branches, lgamma1p, and dpois_wrap -> pgamma. Port R's
# Fullerton/Chebyshev lgammafn so the whole gamma family is 0-ulp.
_M_LN_SQRT_PId2 = 0.225791352644727432363097614947  # log(sqrt(pi/2))

_GAMCS = (
    +.8571195590989331421920062399942e-2, +.4415381324841006757191315771652e-2,
    +.5685043681599363378632664588789e-1, -.4219835396418560501012500186624e-2,
    +.1326808181212460220584006796352e-2, -.1893024529798880432523947023886e-3,
    +.3606925327441245256578082217225e-4, -.6056761904460864218485548290365e-5,
    +.1055829546302283344731823509093e-5, -.1811967365542384048291855891166e-6,
    +.3117724964715322277790254593169e-7, -.5354219639019687140874081024347e-8,
    +.9193275519859588946887786825940e-9, -.1577941280288339761767423273953e-9,
    +.2707980622934954543266540433089e-10, -.4646818653825730144081661058933e-11,
    +.7973350192007419656460767175359e-12, -.1368078209830916025799499172309e-12,
    +.2347319486563800657233471771688e-13, -.4027432614949066932766570534699e-14,
    +.6910051747372100912138336975257e-15, -.1185584500221992907052387126192e-15,
    +.2034148542496373955201026051932e-16, -.3490054341717405849274012949108e-17,
    +.5987993856485305567135051066026e-18, -.1027378057872228074490069778431e-18,
    +.1762702816060529824942759660748e-19, -.3024320653735306260958772112042e-20,
    +.5188914660218397839717833550506e-21, -.8902770842456576692449251601066e-22,
    +.1527474068493342602274596891306e-22, -.2620731256187362900257328332799e-23,
    +.4496464047830538670331046570666e-24, -.7714712731336877911703901525333e-25,
    +.1323635453126044036486572714666e-25, -.2270999412942928816702313813333e-26,
    +.3896418998003991449320816639999e-27, -.6685198115125953327792127999999e-28,
    +.1146998663140024384347613866666e-28, -.1967938586345134677295103999999e-29,
    +.3376448816585338090334890666666e-30, -.5793070335782135784625493333333e-31,
)
_NGAM = 22
_GAM_XMIN = -170.5674972726612
_GAM_XMAX = 171.61447887182298
_GAM_XSML = 2.2474362225598545e-308

_ALGMCS = (
    +.1666389480451863247205729650822e+0, -.1384948176067563840732986059135e-4,
    +.9810825646924729426157171547487e-8, -.1809129475572494194263306266719e-10,
    +.6221098041892605227126015543416e-13, -.3399615005417721944303330599666e-15,
    +.2683181998482698748957538846666e-17, -.2868042435334643284144622399999e-19,
    +.3962837061046434803679306666666e-21, -.6831888753985766870111999999999e-23,
    +.1429227355942498147573333333333e-24, -.3547598158101070547199999999999e-26,
    +.1025680058010470912000000000000e-27, -.3401102254316748799999999999999e-29,
    +.1276642195630062933333333333333e-30,
)
_NALGM = 5
_LGC_XBIG = 94906265.62425156
_LGM_XMAX = 2.5327372760800758e+305


def _chebyshev_eval(x, a, n):
    """R's ``chebyshev_eval`` (chebyshev.c) — n-term Chebyshev series at x."""
    if n < 1 or n > 1000:
        return _NAN
    if x < -1.1 or x > 1.1:
        return _NAN
    twox = x * 2
    b2 = b1 = 0.0
    b0 = 0.0
    for i in range(1, n + 1):
        b2 = b1
        b1 = b0
        b0 = twox * b1 - b2 + a[n - i]
    return (b0 - b2) * 0.5


def _lgammacor(x):
    """R's ``lgammacor(x)`` (lgammacor.c) — log-gamma correction, x >= 10."""
    if x < 10:
        return _NAN
    if x < _LGC_XBIG:
        tmp = 10 / x
        return _chebyshev_eval(tmp * tmp * 2 - 1, _ALGMCS, _NALGM) / x
    return 1 / (x * 12)


def _sinpi(x):
    """R's portable ``sinpi(x) = sin(pi*x)`` (cospi.c fallback branch)."""
    if math.isnan(x):
        return x
    if not math.isfinite(x):
        return _NAN
    x = math.fmod(x, 2.0)
    if x <= -1:
        x += 2.0
    elif x > 1.0:
        x -= 2.0
    if x == 0.0 or x == 1.0:
        return 0.0
    if x == 0.5:
        return 1.0
    if x == -0.5:
        return -1.0
    return math.sin(math.pi * x)


def gammafn(x):
    """R's ``gammafn(x)`` (gamma.c, Fullerton), bit-exact."""
    if math.isnan(x):
        return x
    if x == 0 or (x < 0 and x == round(x)):
        return _NAN
    y = abs(x)
    if y <= 10:
        n = int(x)
        if x < 0:
            n -= 1
        y = x - n
        n -= 1
        value = _chebyshev_eval(y * 2 - 1, _GAMCS, _NGAM) + .9375
        if n == 0:
            return value
        if n < 0:
            if y < _GAM_XSML:
                return _INF if x > 0 else _NEGINF
            n = -n
            for i in range(n):
                value /= (x + i)
            return value
        for i in range(1, n + 1):
            value *= (y + i)
        return value
    # y = |x| > 10
    if x > _GAM_XMAX:
        return _INF
    if x < _GAM_XMIN:
        return 0.0
    if y <= 50 and y == int(y):
        value = 1.0
        for i in range(2, int(y)):
            value *= i
    else:
        corr = _stirlerr(y) if (2 * y == int(2 * y)) else _lgammacor(y)
        value = math.exp((y - 0.5) * math.log(y) - y + _M_LN_SQRT_2PI + corr)
    if x > 0:
        return value
    sinpiy = _sinpi(y)
    if sinpiy == 0:
        return _INF
    return -math.pi / (y * sinpiy * value)


def _lgammafn(x):
    """R's ``lgammafn(x) = log|gamma(x)|`` (lgamma.c), bit-exact."""
    if math.isnan(x):
        return x
    if x <= 0 and x == math.trunc(x):
        return _INF
    y = abs(x)
    if y < 1e-306:
        return -math.log(y)
    if y <= 10:
        return math.log(abs(gammafn(x)))
    if y > _LGM_XMAX:
        return _INF
    if x > 0:
        if x > 1e17:
            return x * (math.log(x) - 1.0)
        elif x > 4934720.0:
            return _M_LN_SQRT_2PI + (x - 0.5) * math.log(x) - x
        return _M_LN_SQRT_2PI + (x - 0.5) * math.log(x) - x + _lgammacor(x)
    sinpiy = abs(_sinpi(y))
    return (_M_LN_SQRT_PId2 + (x - 0.5) * math.log(y) - x
            - math.log(sinpiy) - _lgammacor(y))


def _lgammafn_arr(x):
    """Element-wise :func:`_lgammafn` over an array (scalar -> float)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return _lgammafn(float(x))
    out = np.empty(x.shape, dtype=float)
    flat = x.ravel()
    o = out.ravel()
    for i in range(flat.size):
        o[i] = _lgammafn(float(flat[i]))
    return out


# --- gamma-family scalar foundations (pgamma.c) ------------------------------
_PG_SCALEFACTOR = 2.0 ** 256        # (2^32)^8
_M_CUTOFF = _M_LN2 * 1024 / _DBL_EPSILON   # = 3.196577e18
_DBL_MIN = 2.2250738585072014e-308


def _logcf(x, i, d, eps):
    """R's ``logcf`` (pgamma.c) — continued fraction aux for log1pmx/lgamma1p."""
    c1 = 2 * d
    c2 = i + d
    c4 = c2 + d
    a1 = c2
    b1 = i * (c2 - i * x)
    b2 = d * d * x
    a2 = c4 * c2 - b2
    b2 = c4 * b1 - i * b2
    sf = _PG_SCALEFACTOR
    while abs(a2 * b1 - a1 * b2) > abs(eps * b1 * b2):
        c3 = c2 * c2 * x
        c2 += d
        c4 += d
        a1 = c4 * a2 - c3 * a1
        b1 = c4 * b2 - c3 * b1
        c3 = c1 * c1 * x
        c1 += d
        c4 += d
        a2 = c4 * a1 - c3 * a2
        b2 = c4 * b1 - c3 * b2
        if abs(b2) > sf:
            a1 /= sf
            b1 /= sf
            a2 /= sf
            b2 /= sf
        elif abs(b2) < 1 / sf:
            a1 *= sf
            b1 *= sf
            a2 *= sf
            b2 *= sf
    return a2 / b2


def _log1pmx(x):
    """R's ``log1pmx(x) = log(1+x) - x`` (pgamma.c), accurate for small |x|."""
    minLog1Value = -0.79149064
    if x > 1 or x < minLog1Value:
        return math.log1p(x) - x
    r = x / (2 + x)
    y = r * r
    if abs(x) < 1e-2:
        two = 2.0
        return r * ((((two / 9 * y + two / 7) * y + two / 5) * y
                     + two / 3) * y - x)
    tol_logcf = 1e-14
    return r * (2 * y * _logcf(y, 3, 2, tol_logcf) - x)


# coeffs[i] = (zeta(i+2)-1)/(i+2), i=0..39  (pgamma.c lgamma1p)
_LGAMMA1P_COEFFS = (
    0.3224670334241132182362075833230126e-0,
    0.6735230105319809513324605383715000e-1,
    0.2058080842778454787900092413529198e-1,
    0.7385551028673985266273097291406834e-2,
    0.2890510330741523285752988298486755e-2,
    0.1192753911703260977113935692828109e-2,
    0.5096695247430424223356548135815582e-3,
    0.2231547584535793797614188036013401e-3,
    0.9945751278180853371459589003190170e-4,
    0.4492623673813314170020750240635786e-4,
    0.2050721277567069155316650397830591e-4,
    0.9439488275268395903987425104415055e-5,
    0.4374866789907487804181793223952411e-5,
    0.2039215753801366236781900709670839e-5,
    0.9551412130407419832857179772951265e-6,
    0.4492469198764566043294290331193655e-6,
    0.2120718480555466586923135901077628e-6,
    0.1004322482396809960872083050053344e-6,
    0.4769810169363980565760193417246730e-7,
    0.2271109460894316491031998116062124e-7,
    0.1083865921489695409107491757968159e-7,
    0.5183475041970046655121248647057669e-8,
    0.2483674543802478317185008663991718e-8,
    0.1192140140586091207442548202774640e-8,
    0.5731367241678862013330194857961011e-9,
    0.2759522885124233145178149692816341e-9,
    0.1330476437424448948149715720858008e-9,
    0.6422964563838100022082448087644648e-10,
    0.3104424774732227276239215783404066e-10,
    0.1502138408075414217093301048780668e-10,
    0.7275974480239079662504549924814047e-11,
    0.3527742476575915083615072228655483e-11,
    0.1711991790559617908601084114443031e-11,
    0.8315385841420284819798357793954418e-12,
    0.4042200525289440065536008957032895e-12,
    0.1966475631096616490411045679010286e-12,
    0.9573630387838555763782200936508615e-13,
    0.4664076026428374224576492565974577e-13,
    0.2273736960065972320633279596737272e-13,
    0.1109139947083452201658320007192334e-13,
)


def _lgamma1p(a):
    """R's ``lgamma1p(a) = log(gamma(a+1))`` (pgamma.c), accurate for small a."""
    if abs(a) >= 0.5:
        return _lgammafn(a + 1)
    eulers_const = 0.5772156649015328606065120900824024
    N = 40
    c = 0.2273736845824652515226821577978691e-12  # zeta(N+2)-1
    tol_logcf = 1e-14
    lgam = c * _logcf(-a / 2, N + 2, 1, tol_logcf)
    for i in range(N - 1, -1, -1):
        lgam = _LGAMMA1P_COEFFS[i] - a * lgam
    return (a * lgam - eulers_const) * a - _log1pmx(a)


def _R_Log1_Exp(x):
    """R's ``R_Log1_Exp(x) = log(1 - exp(x))`` (dpq.h), stable form."""
    return math.log(-math.expm1(x)) if x > -_M_LN2 else math.log1p(-math.exp(x))


def _logspace_add(logx, logy):
    return max(logx, logy) + math.log1p(math.exp(-abs(logx - logy)))


def _logspace_sub(logx, logy):
    return logx + _R_Log1_Exp(logy - logx)


# ebd0 (extended bd0) — Welinder's improved-precision version used by R
# dpois. The 128-entry log table from bd0.c:102-231 (each row: 4 floats
# encoding log(p/1024) where p = floor(1024/(0.5+i/256)+0.5), p ≈ 1024 to
# 2048). Decoded from hex-float to plain double values.

# Hex-float decoder: each entry "+0x1.62e430p-1" → that float value.
def _hex_to_float(s: str) -> float:
    return float.fromhex(s)


# Hex-float table from bd0.c:102-231. Reproduced verbatim so this file
# can be diffed against the C source. Each tuple is the 4 float parts
# (a high-bit chunk + three corrections) of one log value.
_BD0_SCALE_HEX = (
    ("+0x1.62e430p-1", "-0x1.05c610p-29", "-0x1.950d88p-54", "+0x1.d9cc02p-79"),
    ("+0x1.5ee02cp-1", "-0x1.6dbe98p-25", "-0x1.51e540p-50", "+0x1.2bfa48p-74"),
    ("+0x1.5ad404p-1", "+0x1.86b3e4p-26", "+0x1.9f6534p-50", "+0x1.54be04p-74"),
    ("+0x1.570124p-1", "-0x1.9ed750p-25", "-0x1.f37dd0p-51", "+0x1.10b770p-77"),
    ("+0x1.5326e4p-1", "-0x1.9b9874p-25", "-0x1.378194p-49", "+0x1.56feb2p-74"),
    ("+0x1.4f4528p-1", "+0x1.aca70cp-28", "+0x1.103e74p-53", "+0x1.9c410ap-81"),
    ("+0x1.4b5bd8p-1", "-0x1.6a91d8p-25", "-0x1.8e43d0p-50", "-0x1.afba9ep-77"),
    ("+0x1.47ae54p-1", "-0x1.abb51cp-25", "+0x1.19b798p-51", "+0x1.45e09cp-76"),
    ("+0x1.43fa00p-1", "-0x1.d06318p-25", "-0x1.8858d8p-49", "-0x1.1927c4p-75"),
    ("+0x1.3ffa40p-1", "+0x1.1a427cp-25", "+0x1.151640p-53", "-0x1.4f5606p-77"),
    ("+0x1.3c7c80p-1", "-0x1.19bf48p-34", "+0x1.05fc94p-58", "-0x1.c096fcp-82"),
    ("+0x1.38b320p-1", "+0x1.6b5778p-25", "+0x1.be38d0p-50", "-0x1.075e96p-74"),
    ("+0x1.34e288p-1", "+0x1.d9ce1cp-25", "+0x1.316eb8p-49", "+0x1.2d885cp-73"),
    ("+0x1.315124p-1", "+0x1.c2fc60p-29", "-0x1.4396fcp-53", "+0x1.acf376p-78"),
    ("+0x1.2db954p-1", "+0x1.720de4p-25", "-0x1.d39b04p-49", "-0x1.f11176p-76"),
    ("+0x1.2a1b08p-1", "-0x1.562494p-25", "+0x1.a7863cp-49", "+0x1.85dd64p-73"),
    ("+0x1.267620p-1", "+0x1.3430e0p-29", "-0x1.96a958p-56", "+0x1.f8e636p-82"),
    ("+0x1.23130cp-1", "+0x1.7bebf4p-25", "+0x1.416f1cp-52", "-0x1.78dd36p-77"),
    ("+0x1.1faa34p-1", "+0x1.70e128p-26", "+0x1.81817cp-50", "-0x1.c2179cp-76"),
    ("+0x1.1bf204p-1", "+0x1.3a9620p-28", "+0x1.2f94c0p-52", "+0x1.9096c0p-76"),
    ("+0x1.187ce4p-1", "-0x1.077870p-27", "+0x1.655a80p-51", "+0x1.eaafd6p-78"),
    ("+0x1.1501c0p-1", "-0x1.406cacp-25", "-0x1.e72290p-49", "+0x1.5dd800p-73"),
    ("+0x1.11cb80p-1", "+0x1.787cd0p-25", "-0x1.efdc78p-51", "-0x1.5380cep-77"),
    ("+0x1.0e4498p-1", "+0x1.747324p-27", "-0x1.024548p-51", "+0x1.77a5a6p-75"),
    ("+0x1.0b036cp-1", "+0x1.690c74p-25", "+0x1.5d0cc4p-50", "-0x1.c0e23cp-76"),
    ("+0x1.077070p-1", "-0x1.a769bcp-27", "+0x1.452234p-52", "+0x1.6ba668p-76"),
    ("+0x1.04240cp-1", "-0x1.a686acp-27", "-0x1.ef46b0p-52", "-0x1.5ce10cp-76"),
    ("+0x1.00d22cp-1", "+0x1.fc0e10p-25", "+0x1.6ee034p-50", "-0x1.19a2ccp-74"),
    ("+0x1.faf588p-2", "+0x1.ef1e64p-27", "-0x1.26504cp-54", "-0x1.b15792p-82"),
    ("+0x1.f4d87cp-2", "+0x1.d7b980p-26", "-0x1.a114d8p-50", "+0x1.9758c6p-75"),
    ("+0x1.ee1414p-2", "+0x1.2ec060p-26", "+0x1.dc00fcp-52", "+0x1.f8833cp-76"),
    ("+0x1.e7e32cp-2", "-0x1.ac796cp-27", "-0x1.a68818p-54", "+0x1.235d02p-78"),
    ("+0x1.e108a0p-2", "-0x1.768ba4p-28", "-0x1.f050a8p-52", "+0x1.00d632p-82"),
    ("+0x1.dac354p-2", "-0x1.d3a6acp-30", "+0x1.18734cp-57", "-0x1.f97902p-83"),
    ("+0x1.d47424p-2", "+0x1.7dbbacp-31", "-0x1.d5ada4p-56", "+0x1.56fcaap-81"),
    ("+0x1.ce1af0p-2", "+0x1.70be7cp-27", "+0x1.6f6fa4p-51", "+0x1.7955a2p-75"),
    ("+0x1.c7b798p-2", "+0x1.ec36ecp-26", "-0x1.07e294p-50", "-0x1.ca183cp-75"),
    ("+0x1.c1ef04p-2", "+0x1.c1dfd4p-26", "+0x1.888eecp-50", "-0x1.fd6b86p-75"),
    ("+0x1.bb7810p-2", "+0x1.478bfcp-26", "+0x1.245b8cp-50", "+0x1.ea9d52p-74"),
    ("+0x1.b59da0p-2", "-0x1.882b08p-27", "+0x1.31573cp-53", "-0x1.8c249ap-77"),
    ("+0x1.af1294p-2", "-0x1.b710f4p-27", "+0x1.622670p-51", "+0x1.128578p-76"),
    ("+0x1.a925d4p-2", "-0x1.0ae750p-27", "+0x1.574ed4p-51", "+0x1.084996p-75"),
    ("+0x1.a33040p-2", "+0x1.027d30p-29", "+0x1.b9a550p-53", "-0x1.b2e38ap-78"),
    ("+0x1.9d31c0p-2", "-0x1.5ec12cp-26", "-0x1.5245e0p-52", "+0x1.2522d0p-79"),
    ("+0x1.972a34p-2", "+0x1.135158p-30", "+0x1.a5c09cp-56", "+0x1.24b70ep-80"),
    ("+0x1.911984p-2", "+0x1.0995d4p-26", "+0x1.3bfb5cp-50", "+0x1.2c9dd6p-75"),
    ("+0x1.8bad98p-2", "-0x1.1d6144p-29", "+0x1.5b9208p-53", "+0x1.1ec158p-77"),
    ("+0x1.858b58p-2", "-0x1.1b4678p-27", "+0x1.56cab4p-53", "-0x1.2fdc0cp-78"),
    ("+0x1.7f5fa0p-2", "+0x1.3aaf48p-27", "+0x1.461964p-51", "+0x1.4ae476p-75"),
    ("+0x1.79db68p-2", "-0x1.7e5054p-26", "+0x1.673750p-51", "-0x1.a11f7ap-76"),
    ("+0x1.744f88p-2", "-0x1.cc0e18p-26", "-0x1.1e9d18p-50", "-0x1.6c06bcp-78"),
    ("+0x1.6e08ecp-2", "-0x1.5d45e0p-26", "-0x1.c73ec8p-50", "+0x1.318d72p-74"),
    ("+0x1.686c80p-2", "+0x1.e9b14cp-26", "-0x1.13bbd4p-50", "-0x1.efeb1cp-78"),
    ("+0x1.62c830p-2", "-0x1.a8c70cp-27", "-0x1.5a1214p-51", "-0x1.bab3fcp-79"),
    ("+0x1.5d1bdcp-2", "-0x1.4fec6cp-31", "+0x1.423638p-56", "+0x1.ee3feep-83"),
    ("+0x1.576770p-2", "+0x1.7455a8p-26", "-0x1.3ab654p-50", "-0x1.26be4cp-75"),
    ("+0x1.5262e0p-2", "-0x1.146778p-26", "-0x1.b9f708p-52", "-0x1.294018p-77"),
    ("+0x1.4c9f08p-2", "+0x1.e152c4p-26", "-0x1.dde710p-53", "+0x1.fd2208p-77"),
    ("+0x1.46d2d8p-2", "+0x1.c28058p-26", "-0x1.936284p-50", "+0x1.9fdd68p-74"),
    ("+0x1.41b940p-2", "+0x1.cce0c0p-26", "-0x1.1a4050p-50", "+0x1.bc0376p-76"),
    ("+0x1.3bdd24p-2", "+0x1.d6296cp-27", "+0x1.425b48p-51", "-0x1.cddb2cp-77"),
    ("+0x1.36b578p-2", "-0x1.287ddcp-27", "-0x1.2d0f4cp-51", "+0x1.38447ep-75"),
    ("+0x1.31871cp-2", "+0x1.2a8830p-27", "+0x1.3eae54p-52", "-0x1.898136p-77"),
    ("+0x1.2b9304p-2", "-0x1.51d8b8p-28", "+0x1.27694cp-52", "-0x1.fd852ap-76"),
    ("+0x1.265620p-2", "-0x1.d98f3cp-27", "+0x1.a44338p-51", "-0x1.56e85ep-78"),
    ("+0x1.211254p-2", "+0x1.986160p-26", "+0x1.73c5d0p-51", "+0x1.4a861ep-75"),
    ("+0x1.1bc794p-2", "+0x1.fa3918p-27", "+0x1.879c5cp-51", "+0x1.16107cp-78"),
    ("+0x1.1675ccp-2", "-0x1.4545a0p-26", "+0x1.c07398p-51", "+0x1.f55c42p-76"),
    ("+0x1.111ce4p-2", "+0x1.f72670p-37", "-0x1.b84b5cp-61", "+0x1.a4a4dcp-85"),
    ("+0x1.0c81d4p-2", "+0x1.0c150cp-27", "+0x1.218600p-51", "-0x1.d17312p-76"),
    ("+0x1.071b84p-2", "+0x1.fcd590p-26", "+0x1.a3a2e0p-51", "+0x1.fe5ef8p-76"),
    ("+0x1.01ade4p-2", "-0x1.bb1844p-28", "+0x1.db3cccp-52", "+0x1.1f56fcp-77"),
    ("+0x1.fa01c4p-3", "-0x1.12a0d0p-29", "-0x1.f71fb0p-54", "+0x1.e287a4p-78"),
    ("+0x1.ef0adcp-3", "+0x1.7b8b28p-28", "-0x1.35bce4p-52", "-0x1.abc8f8p-79"),
    ("+0x1.e598ecp-3", "+0x1.5a87e4p-27", "-0x1.134bd0p-51", "+0x1.c2cebep-76"),
    ("+0x1.da85d8p-3", "-0x1.df31b0p-27", "+0x1.94c16cp-57", "+0x1.8fd7eap-82"),
    ("+0x1.d0fb80p-3", "-0x1.bb5434p-28", "-0x1.ea5640p-52", "-0x1.8ceca4p-77"),
    ("+0x1.c765b8p-3", "+0x1.e4d68cp-27", "+0x1.5b59b4p-51", "+0x1.76f6c4p-76"),
    ("+0x1.bdc46cp-3", "-0x1.1cbb50p-27", "+0x1.2da010p-51", "+0x1.eb282cp-75"),
    ("+0x1.b27980p-3", "-0x1.1b9ce0p-27", "+0x1.7756f8p-52", "+0x1.2ff572p-76"),
    ("+0x1.a8bed0p-3", "-0x1.bbe874p-30", "+0x1.85cf20p-56", "+0x1.b9cf18p-80"),
    ("+0x1.9ef83cp-3", "+0x1.2769a4p-27", "-0x1.85bda0p-52", "+0x1.8c8018p-79"),
    ("+0x1.9525a8p-3", "+0x1.cf456cp-27", "-0x1.7137d8p-52", "-0x1.f158e8p-76"),
    ("+0x1.8b46f8p-3", "+0x1.11b12cp-30", "+0x1.9f2104p-54", "-0x1.22836ep-78"),
    ("+0x1.83040cp-3", "+0x1.2379e4p-28", "+0x1.b71c70p-52", "-0x1.990cdep-76"),
    ("+0x1.790ed4p-3", "+0x1.dc4c68p-28", "-0x1.910ac8p-52", "+0x1.dd1bd6p-76"),
    ("+0x1.6f0d28p-3", "+0x1.5cad68p-28", "+0x1.737c94p-52", "-0x1.9184bap-77"),
    ("+0x1.64fee8p-3", "+0x1.04bf88p-28", "+0x1.6fca28p-52", "+0x1.8884a8p-76"),
    ("+0x1.5c9400p-3", "+0x1.d65cb0p-29", "-0x1.b2919cp-53", "+0x1.b99bcep-77"),
    ("+0x1.526e60p-3", "-0x1.c5e4bcp-27", "-0x1.0ba380p-52", "+0x1.d6e3ccp-79"),
    ("+0x1.483bccp-3", "+0x1.9cdc7cp-28", "-0x1.5ad8dcp-54", "-0x1.392d3cp-83"),
    ("+0x1.3fb25cp-3", "-0x1.a6ad74p-27", "+0x1.5be6b4p-52", "-0x1.4e0114p-77"),
    ("+0x1.371fc4p-3", "-0x1.fe1708p-27", "-0x1.78864cp-52", "-0x1.27543ap-76"),
    ("+0x1.2cca10p-3", "-0x1.4141b4p-28", "-0x1.ef191cp-52", "+0x1.00ee08p-76"),
    ("+0x1.242310p-3", "+0x1.3ba510p-27", "-0x1.d003c8p-51", "+0x1.162640p-76"),
    ("+0x1.1b72acp-3", "+0x1.52f67cp-27", "-0x1.fd6fa0p-51", "+0x1.1a3966p-77"),
    ("+0x1.10f8e4p-3", "+0x1.129cd8p-30", "+0x1.31ef30p-55", "+0x1.a73e38p-79"),
    ("+0x1.08338cp-3", "-0x1.005d7cp-27", "-0x1.661a9cp-51", "+0x1.1f138ap-79"),
    ("+0x1.fec914p-4", "-0x1.c482a8p-29", "-0x1.55746cp-54", "+0x1.99f932p-80"),
    ("+0x1.ed1794p-4", "+0x1.d06f00p-29", "+0x1.75e45cp-53", "-0x1.d0483ep-78"),
    ("+0x1.db5270p-4", "+0x1.87d928p-32", "-0x1.0f52a4p-57", "+0x1.81f4a6p-84"),
    ("+0x1.c97978p-4", "+0x1.af1d24p-29", "-0x1.0977d0p-60", "-0x1.8839d0p-84"),
    ("+0x1.b78c84p-4", "-0x1.44f124p-28", "-0x1.ef7bc4p-52", "+0x1.9e0650p-78"),
    ("+0x1.a58b60p-4", "+0x1.856464p-29", "+0x1.c651d0p-55", "+0x1.b06b0cp-79"),
    ("+0x1.9375e4p-4", "+0x1.5595ecp-28", "+0x1.dc3738p-52", "+0x1.86c89ap-81"),
    ("+0x1.814be4p-4", "-0x1.c073fcp-28", "-0x1.371f88p-53", "-0x1.5f4080p-77"),
    ("+0x1.6f0d28p-4", "+0x1.5cad68p-29", "+0x1.737c94p-53", "-0x1.9184bap-78"),
    ("+0x1.60658cp-4", "-0x1.6c8af4p-28", "+0x1.d8ef74p-55", "+0x1.c4f792p-80"),
    ("+0x1.4e0110p-4", "+0x1.146b5cp-29", "+0x1.73f7ccp-54", "-0x1.d28db8p-79"),
    ("+0x1.3b8758p-4", "+0x1.8b1b70p-28", "-0x1.20aca4p-52", "-0x1.651894p-76"),
    ("+0x1.28f834p-4", "+0x1.43b6a4p-30", "-0x1.452af8p-55", "+0x1.976892p-80"),
    ("+0x1.1a0fbcp-4", "-0x1.e4075cp-28", "+0x1.1fe618p-52", "+0x1.9d6dc2p-77"),
    ("+0x1.075984p-4", "-0x1.4ce370p-29", "-0x1.d9fc98p-53", "+0x1.4ccf12p-77"),
    ("+0x1.f0a30cp-5", "+0x1.162a68p-37", "-0x1.e83368p-61", "-0x1.d222a6p-86"),
    ("+0x1.cae730p-5", "-0x1.1a8f7cp-31", "-0x1.5f9014p-55", "+0x1.2720c0p-79"),
    ("+0x1.ac9724p-5", "-0x1.e8ee08p-29", "+0x1.a7de04p-54", "-0x1.9bba74p-78"),
    ("+0x1.868a84p-5", "-0x1.ef8128p-30", "+0x1.dc5eccp-54", "-0x1.58d250p-79"),
    ("+0x1.67f950p-5", "-0x1.ed684cp-30", "-0x1.f060c0p-55", "-0x1.b1294cp-80"),
    ("+0x1.494accp-5", "+0x1.a6c890p-32", "-0x1.c3ad48p-56", "-0x1.6dc66cp-84"),
    ("+0x1.22c71cp-5", "-0x1.8abe2cp-32", "-0x1.7e7078p-56", "-0x1.ddc3dcp-86"),
    ("+0x1.03d5d8p-5", "+0x1.79cfbcp-31", "-0x1.da7c4cp-58", "+0x1.4e7582p-83"),
    ("+0x1.c98d18p-6", "+0x1.a01904p-31", "-0x1.854164p-55", "+0x1.883c36p-79"),
    ("+0x1.8b31fcp-6", "-0x1.356500p-30", "+0x1.c3ab48p-55", "+0x1.b69bdap-80"),
    ("+0x1.3cea44p-6", "+0x1.a352bcp-33", "-0x1.8865acp-57", "-0x1.48159cp-81"),
    ("+0x1.fc0a8cp-7", "-0x1.e07f84p-32", "+0x1.e7cf6cp-58", "+0x1.3a69c0p-82"),
    ("+0x1.7dc474p-7", "+0x1.f810a8p-31", "-0x1.245b5cp-56", "-0x1.a1f4f8p-80"),
    ("+0x1.fe02a8p-8", "-0x1.4ef988p-32", "+0x1.1f86ecp-57", "+0x1.20723cp-81"),
    ("+0x1.ff00acp-9", "-0x1.d4ef44p-33", "+0x1.2821acp-63", "+0x1.5a6d32p-87"),
    ("0",              "0",               "0",               "0"),  # log(1) = 0
)
_BD0_SCALE = tuple(tuple(_hex_to_float(s) for s in row) for row in _BD0_SCALE_HEX)
_BD0_SCALE_NP = np.array(_BD0_SCALE, dtype=float)  # shape (129, 4) for vectorized lookup


def _ebd0(x, M):
    """Port of nmath ``ebd0(x, M)`` (bd0.c:241-355). Vectorized.

    Computes ``x·log(x/M) + (M - x)`` with extended precision. Returns
    ``(yh, yl)`` arrays such that ``yh + yl`` is the value. Welinder's
    improved algorithm (R Bugzilla PR#15628).
    """
    Sb = 10
    S = 1 << Sb  # = 1024
    N = 128

    x = np.asarray(x, dtype=float)
    M = np.asarray(M, dtype=float)
    scalar = (x.ndim == 0 and M.ndim == 0)
    x = np.atleast_1d(x)
    M = np.atleast_1d(np.broadcast_to(M, x.shape).copy())

    yh = np.zeros_like(x)
    yl = np.zeros_like(x)

    # Edge cases.
    eq = x == M
    x_zero = ~eq & (x == 0.0)
    M_zero = ~eq & ~x_zero & (M == 0.0)
    yh[x_zero] = M[x_zero]
    yh[M_zero] = np.inf

    # M/x → ∞ (M >> x).
    Mox = np.where(eq | x_zero | M_zero, 1.0, M / np.where(x == 0.0, 1.0, x))
    inf_Mox = ~eq & ~x_zero & ~M_zero & (Mox == np.inf)
    yh[inf_Mox] = M[inf_Mox]

    active = ~(eq | x_zero | M_zero | inf_Mox)
    if not np.any(active):
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    xa = x[active]
    Ma = M[active]
    Mox_a = Ma / xa

    # M/x = r · 2^e
    r, e = np.frexp(Mox_a)

    # Overflow check (rare): M_LN2 * (-e) > 1 + DBL_MAX/x → yh = +inf
    with np.errstate(over="ignore"):
        overflow = _M_LN2 * (-e.astype(float)) > (1.0 + np.finfo(float).max / xa)
    if np.any(overflow):
        active_idx = np.where(active)[0]
        yh[active_idx[overflow]] = np.inf
        good = ~overflow
        xa = xa[good]
        Ma = Ma[good]
        r = r[good]
        e = e[good]
        active_idx = active_idx[good]
    else:
        active_idx = np.where(active)[0]

    if xa.size == 0:
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    i = np.floor((r - 0.5) * (2 * N) + 0.5).astype(np.int64)
    f = np.floor(S / (0.5 + i / (2.0 * N)) + 0.5)
    fg = np.ldexp(f, -(e + Sb))

    inf_fg = fg == np.inf
    if np.any(inf_fg):
        yh[active_idx[inf_fg]] = np.inf
        good = ~inf_fg
        xa = xa[good]
        Ma = Ma[good]
        fg = fg[good]
        i = i[good]
        e = e[good]
        active_idx = active_idx[good]

    if xa.size == 0:
        return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)

    # Local accumulators (we update yh/yl only via these arrays).
    lh = np.zeros_like(xa)
    ll = np.zeros_like(xa)

    def add1(d_arr):
        d1 = np.floor(d_arr + 0.5)
        d2 = d_arr - d1
        np.add(lh, d1, out=lh)
        np.add(ll, d2, out=ll)

    # ADD1(-x * log1pmx((M*fg - x) / x))
    arg = (Ma * fg - xa) / xa
    log1pmx_val = _log1pmx_vec(arg)  # R's ebd0 uses the accurate log1pmx, not log1p(arg)-arg
    add1(-xa * log1pmx_val)

    fg_ne_1 = fg != 1.0
    if np.any(fg_ne_1):
        # Process the 4-iteration table corrections only where fg != 1.
        # We compute updates for the WHOLE active set; for fg==1 elements
        # the increments are 0 (since x * 0 = 0 with proper masking).
        for j in range(4):
            tbl_i = _BD0_SCALE_NP[i, j]
            tbl_0 = _BD0_SCALE_NP[0, j]
            inc1 = np.where(fg_ne_1, xa * tbl_i, 0.0)
            inc2 = np.where(fg_ne_1, -xa * tbl_0 * e, 0.0)
            add1(inc1)
            add1(inc2)
            # Per-iter overflow check: any !isfinite → set to inf and freeze.
            nonfinite = ~np.isfinite(lh)
            if np.any(nonfinite):
                lh[nonfinite] = np.inf
                ll[nonfinite] = 0.0
                fg_ne_1 = fg_ne_1 & ~nonfinite

    # ADD1(M); ADD1(-M·fg) only where fg != 1; for fg==1, the original
    # scalar code returns early before these — match that exactly.
    # But: the scalar code returns IMMEDIATELY for fg==1 after the first
    # add1(-x·log1pmx). For fg==1, lh/ll already have the right value,
    # so skip the M / -M·fg adds.
    fg_eq_1 = fg == 1.0
    fg_ne_1 = ~fg_eq_1
    if np.any(fg_ne_1):
        # Apply M / -M·fg adds only for fg != 1 (otherwise scalar returns
        # early so we shouldn't add).
        i_ne = np.where(fg_ne_1)[0]
        d = Ma[i_ne]
        d1 = np.floor(d + 0.5)
        lh[i_ne] = lh[i_ne] + d1
        ll[i_ne] = ll[i_ne] + (d - d1)
        d = -Ma[i_ne] * fg[i_ne]
        d1 = np.floor(d + 0.5)
        lh[i_ne] = lh[i_ne] + d1
        ll[i_ne] = ll[i_ne] + (d - d1)

    yh[active_idx] = lh
    yl[active_idx] = ll
    return (float(yh[0]), float(yl[0])) if scalar else (yh, yl)


def _pow1p(x, y):
    """R's ``pow1p(x, y) = (1+x)^y`` (dbinom.c), accurate for ``|x| << 1``.
    Vectorised; matches R's branch logic element-wise."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    out = np.empty(x.shape, dtype=float)
    # small non-negative integer y in {0,1,2,3,4}: exact polynomial
    is_int = (y == np.trunc(y)) & (y >= 0) & (y <= 4.0)
    done = np.zeros(x.shape, dtype=bool)
    for k, poly in enumerate((
            lambda xx: np.ones_like(xx),
            lambda xx: xx + 1.0,
            lambda xx: xx * (xx + 2.0) + 1.0,
            lambda xx: xx * (xx * (xx + 3.0) + 3.0) + 1.0,
            lambda xx: xx * (xx * (xx * (xx + 4.0) + 6.0) + 4.0) + 1.0)):
        m = is_int & (y == k) & ~done
        if m.any():
            out[m] = poly(x[m])
            done |= m
    rest = ~done
    if rest.any():
        xr = x[rest]
        yr = y[rest]
        xp1 = xr + 1.0
        x_ = xp1 - 1.0
        naive = (x_ == xr) | (np.abs(xr) > 0.5) | np.isnan(xr)
        res = np.where(naive, np.power(xp1, yr), np.exp(yr * np.log1p(xr)))
        out[rest] = res
    # NaN y handling: (0+1)^NaN := 1 ; else y
    nan_y = np.isnan(y)
    if nan_y.any():
        out[nan_y] = np.where(x[nan_y] == 0.0, 1.0, y[nan_y])
    return out


def _dpois_raw(x, lambda_, give_log: bool = True):
    """Port of nmath ``dpois_raw(x, lambda, give_log)`` (dpois.c:43-69).

    Vectorized over ``x`` and ``lambda``. Uses Loader's saddlepoint with
    ebd0 (R 4.5). Returns the same shape as the broadcast of inputs.
    """
    x_in = np.asarray(x, dtype=float)
    l_in = np.asarray(lambda_, dtype=float)
    scalar = (x_in.ndim == 0 and l_in.ndim == 0)
    x = np.atleast_1d(x_in.copy())
    lam = np.atleast_1d(np.broadcast_to(l_in, x.shape).copy())

    NEG_INF = float('-inf')
    out = np.empty_like(x)

    # Edge cases (rare in PIRLS; cheap to test).
    lam_zero = (lam == 0.0)
    lam_inf  = ~np.isfinite(lam)
    x_neg    = x < 0
    tiny = np.finfo(float).tiny
    x_le_lt = (x <= lam * tiny) & ~lam_zero & ~lam_inf & ~x_neg
    lam_lt_xt = (lam < x * tiny) & ~lam_zero & ~lam_inf & ~x_neg & ~x_le_lt
    main = ~(lam_zero | lam_inf | x_neg | x_le_lt | lam_lt_xt)

    # lam == 0: x==0 → log(1)=0; else -inf
    if np.any(lam_zero):
        out[lam_zero] = np.where(x[lam_zero] == 0.0, 0.0, NEG_INF)
    if np.any(lam_inf):
        out[lam_inf] = NEG_INF
    if np.any(x_neg):
        out[x_neg] = NEG_INF
    if np.any(x_le_lt):
        out[x_le_lt] = -lam[x_le_lt]
    if np.any(lam_lt_xt):
        sub = lam_lt_xt
        xn = x[sub]
        ln = lam[sub]
        out[sub] = np.where(~np.isfinite(xn),
                            NEG_INF,
                            -ln + xn * np.log(ln) - _lgammafn_arr(xn + 1.0))

    # Common (saddlepoint) path.
    m_yl = m_yh = m_r = m_Lrg = None
    if np.any(main):
        xm = x[main]
        lm = lam[main]
        yh, yl = _ebd0(xm, lm)
        yl_total = yl + _stirlerr(xm)
        Lrg = xm >= _X_LRG
        r = np.where(Lrg, _M_SQRT_2PI * np.sqrt(xm), _M_2PI * xm)
        log_correction = np.where(Lrg, np.log(r), 0.5 * np.log(r))
        out[main] = -yl_total - yh - log_correction
        m_yl, m_yh, m_r, m_Lrg = yl_total, yh, r, Lrg

    if not give_log:
        # Edges map to R_D_exp / R_D__0/__1, i.e. exp(log-value); only the
        # saddlepoint lanes need R's exact  exp(-yl)*exp(-yh)/sqrt(r)  form
        # (dpois.c:68), which differs from exp(log-result) at the last ulp.
        out = np.exp(out)
        if m_yl is not None:
            out[main] = (np.exp(-m_yl) * np.exp(-m_yh)
                         / np.where(m_Lrg, m_r, np.sqrt(m_r)))
    return float(out[0]) if scalar else out


def _dbinom_raw(x, n, p, q, give_log: bool = True):
    """Port of nmath ``dbinom_raw(x, n, p, q, give_log)`` (dbinom.c:72-118).

    Vectorized. Uses Loader's saddlepoint with the older (non-extended)
    ``bd0`` — matches dbinom.c which calls ``bd0(...)`` not ``ebd0(...)``.
    """
    x_in = np.asarray(x, dtype=float)
    n_in = np.asarray(n, dtype=float)
    p_in = np.asarray(p, dtype=float)
    q_in = np.asarray(q, dtype=float)
    scalar = (x_in.ndim == 0 and n_in.ndim == 0 and p_in.ndim == 0 and q_in.ndim == 0)
    # Broadcast to common shape.
    shape = np.broadcast_shapes(x_in.shape, n_in.shape, p_in.shape, q_in.shape)
    x = np.broadcast_to(x_in, shape).astype(float).copy()
    n = np.broadcast_to(n_in, shape).astype(float).copy()
    p = np.broadcast_to(p_in, shape).astype(float).copy()
    q = np.broadcast_to(q_in, shape).astype(float).copy()

    NEG_INF = float('-inf')
    out = np.empty(shape, dtype=float)

    p_zero = p == 0.0
    q_zero = q == 0.0
    x_zero = x == 0.0
    x_eq_n = x == n
    x_oob = (x < 0) | (x > n)

    edge_p0 = p_zero
    edge_q0 = q_zero & ~p_zero
    edge_x0 = x_zero & ~p_zero & ~q_zero
    edge_xn = x_eq_n & ~p_zero & ~q_zero & ~x_zero
    edge_oob = x_oob & ~p_zero & ~q_zero & ~x_zero & ~x_eq_n
    main = ~(edge_p0 | edge_q0 | edge_x0 | edge_xn | edge_oob)

    if np.any(edge_p0):
        out[edge_p0] = np.where(x[edge_p0] == 0.0, 0.0, NEG_INF)
    if np.any(edge_q0):
        out[edge_q0] = np.where(x[edge_q0] == n[edge_q0], 0.0, NEG_INF)
    if np.any(edge_x0):
        n0 = n[edge_x0]
        p0 = p[edge_x0]
        q0 = q[edge_x0]
        # n == 0 → log(1) = 0
        n_is_0 = n0 == 0.0
        # else: n*log(q) if p>q, n*log1p(-p) otherwise.
        big_p = (p0 > q0) & ~n_is_0
        big_q = ~big_p & ~n_is_0
        val = np.empty_like(n0)
        val[n_is_0] = 0.0
        if np.any(big_p):
            val[big_p] = n0[big_p] * np.log(q0[big_p])
        if np.any(big_q):
            val[big_q] = n0[big_q] * np.log1p(-p0[big_q])
        out[edge_x0] = val
    if np.any(edge_xn):
        n0 = n[edge_xn]
        p0 = p[edge_xn]
        q0 = q[edge_xn]
        big_p = p0 > q0
        val = np.empty_like(n0)
        if np.any(big_p):
            val[big_p] = n0[big_p] * np.log1p(-q0[big_p])
        if np.any(~big_p):
            val[~big_p] = n0[~big_p] * np.log(p0[~big_p])
        out[edge_xn] = val
    if np.any(edge_oob):
        out[edge_oob] = NEG_INF

    if np.any(main):
        xm = x[main]
        nm = n[main]
        pm = p[main]
        qm = q[main]
        lc = (_stirlerr(nm) - _stirlerr(xm) - _stirlerr(nm - xm)
              - _bd0(xm, nm * pm) - _bd0(nm - xm, nm * qm))
        lf = _M_LN_2PI + np.log(xm) + np.log1p(-xm / nm)
        out[main] = lc - 0.5 * lf

    if not give_log:
        # main + {p,q}==0 + oob edges are R_D_exp(log-value); only the x==0 /
        # x==n edges use R's pow/pow1p (dbinom.c:79-91) for full accuracy.
        out = np.asarray(np.exp(out))  # keep 0-d scalar case as an ndarray
        if np.any(edge_x0):
            n0, p0, q0 = n[edge_x0], p[edge_x0], q[edge_x0]
            out[edge_x0] = np.where(
                n0 == 0.0, 1.0,
                np.where(p0 > q0, np.power(q0, n0), _pow1p(-p0, n0)))
        if np.any(edge_xn):
            n0, p0, q0 = n[edge_xn], p[edge_xn], q[edge_xn]
            out[edge_xn] = np.where(
                p0 > q0, _pow1p(-q0, n0), np.power(p0, n0))
    return float(out.reshape(())) if scalar else out




# === pgamma — gamma/chi-square CDF (nmath/pgamma.c, Welinder) ================
def _dpois_wrap(x_plus_1, lambda_, give_log):
    if not math.isfinite(lambda_):
        return _NEGINF if give_log else 0.0
    if x_plus_1 > 1:
        return _dpois_raw(x_plus_1 - 1, lambda_, give_log)
    if lambda_ > abs(x_plus_1 - 1) * _M_CUTOFF:
        v = -lambda_ - _lgammafn(x_plus_1)
        return v if give_log else math.exp(v)
    d = _dpois_raw(x_plus_1, lambda_, give_log)
    return (d + math.log(x_plus_1 / lambda_)) if give_log \
        else d * (x_plus_1 / lambda_)


def _pgamma_smallx(x, alph, lower_tail, log_p):
    sum_ = 0.0
    c = alph
    n = 0.0
    while True:
        n += 1
        c *= -x / n
        term = c / (alph + n)
        sum_ += term
        if abs(term) <= _DBL_EPSILON * abs(sum_):
            break
    if lower_tail:
        f1 = math.log1p(sum_) if log_p else 1 + sum_
        if alph > 1:
            f2 = _dpois_raw(alph, x, log_p)
            f2 = f2 + x if log_p else f2 * math.exp(x)
        elif log_p:
            f2 = alph * math.log(x) - _lgamma1p(alph)
        else:
            f2 = math.pow(x, alph) / math.exp(_lgamma1p(alph))
        return f1 + f2 if log_p else f1 * f2
    else:
        lf2 = alph * math.log(x) - _lgamma1p(alph)
        if log_p:
            return _R_Log1_Exp(math.log1p(sum_) + lf2)
        f1m1 = sum_
        f2m1 = math.expm1(lf2)
        return -(f1m1 + f2m1 + f1m1 * f2m1)


def _pd_upper_series(x, y, log_p):
    term = x / y
    sum_ = term
    while True:
        y += 1
        term *= x / y
        sum_ += term
        if not (term > sum_ * _DBL_EPSILON):
            break
    return math.log(sum_) if log_p else sum_


def _pd_lower_cf(y, d):
    sf = _PG_SCALEFACTOR
    max_it = 200000
    if y == 0:
        return 0.0
    f0 = y / d
    if abs(y - 1) < abs(d) * _DBL_EPSILON:
        return f0
    if f0 > 1.0:
        f0 = 1.0
    c2 = y
    c4 = d
    a1 = 0.0
    b1 = 1.0
    a2 = y
    b2 = d
    while b2 > sf:
        a1 /= sf
        b1 /= sf
        a2 /= sf
        b2 /= sf
    i = 0.0
    of = -1.0
    f = 0.0
    while i < max_it:
        i += 1
        c2 -= 1
        c3 = i * c2
        c4 += 2
        a1 = c4 * a2 + c3 * a1
        b1 = c4 * b2 + c3 * b1
        i += 1
        c2 -= 1
        c3 = i * c2
        c4 += 2
        a2 = c4 * a1 + c3 * a2
        b2 = c4 * b1 + c3 * b2
        if b2 > sf:
            a1 /= sf
            b1 /= sf
            a2 /= sf
            b2 /= sf
        if b2 != 0:
            f = a2 / b2
            if abs(f - of) <= _DBL_EPSILON * max(f0, abs(f)):
                return f
            of = f
    return f


def _pd_lower_series(lambda_, y):
    term = 1.0
    sum_ = 0.0
    while y >= 1 and term > sum_ * _DBL_EPSILON:
        term *= y / lambda_
        sum_ += term
        y -= 1
    if y != math.floor(y):
        f = _pd_lower_cf(y, lambda_ + 1 - y)
        sum_ += term * f
    return sum_


def _dpnorm(x, lower_tail, lp):
    if x < 0:
        x = -x
        lower_tail = not lower_tail
    if x > 10 and not lower_tail:
        term = 1 / x
        sum_ = term
        x2 = x * x
        i = 1.0
        while True:
            term *= -i / x2
            sum_ += term
            i += 2
            if not (abs(term) > _DBL_EPSILON * sum_):
                break
        return 1 / sum_
    d = dnorm5(x, 0.0, 1.0, False)
    return d / math.exp(lp)


_PPA_COEFS_A = (-1e99, 2 / 3., -4 / 135., 8 / 2835., 16 / 8505.,
                -8992 / 12629925., -334144 / 492567075., 698752 / 1477701225.)
_PPA_COEFS_B = (-1e99, 1 / 12., 1 / 288., -139 / 51840., -571 / 2488320.,
                163879 / 209018880., 5246819 / 75246796800.,
                -534703531 / 902961561600.)


def _ppois_asymp(x, lambda_, lower_tail, log_p):
    dfm = lambda_ - x
    pt_ = -_log1pmx(dfm / x)
    s2pt = math.sqrt(2 * x * pt_)
    if dfm < 0:
        s2pt = -s2pt
    res12 = 0.0
    res1_ig = res1_term = math.sqrt(x)
    res2_ig = res2_term = s2pt
    for i in range(1, 8):
        res12 += res1_ig * _PPA_COEFS_A[i]
        res12 += res2_ig * _PPA_COEFS_B[i]
        res1_term *= pt_ / i
        res2_term *= 2 * pt_ / (2 * i + 1)
        res1_ig = res1_ig / x + res1_term
        res2_ig = res2_ig / x + res2_term
    elfb = x
    elfb_term = 1.0
    for i in range(1, 8):
        elfb += elfb_term * _PPA_COEFS_B[i]
        elfb_term /= x
    if not lower_tail:
        elfb = -elfb
    f = res12 / elfb
    np_ = pnorm5(s2pt, 0.0, 1.0, not lower_tail, log_p)
    if log_p:
        n_d_over_p = _dpnorm(s2pt, not lower_tail, np_)
        return np_ + math.log1p(f * n_d_over_p)
    nd = dnorm5(s2pt, 0.0, 1.0, False)
    return np_ + f * nd


def pgamma_raw(x, alph, lower_tail, log_p):
    # R_P_bounds_01(x, 0, +Inf)
    if x <= 0:
        return _dt0(lower_tail, log_p)
    if x >= _INF:
        return _dt1(lower_tail, log_p)
    if x < 1:
        res = _pgamma_smallx(x, alph, lower_tail, log_p)
    elif x <= alph - 1 and x < 0.8 * (alph + 50):
        sum_ = _pd_upper_series(x, alph, log_p)
        d = _dpois_wrap(alph, x, log_p)
        if not lower_tail:
            res = _R_Log1_Exp(d + sum_) if log_p else 1 - d * sum_
        else:
            res = (sum_ + d) if log_p else sum_ * d
    elif alph - 1 < x and alph < 0.8 * (x + 50):
        d = _dpois_wrap(alph, x, log_p)
        if alph < 1:
            if x * _DBL_EPSILON > 1 - alph:
                sum_ = 0.0 if log_p else 1.0
            else:
                fcf = _pd_lower_cf(alph, x - (alph - 1)) * x / alph
                sum_ = math.log(fcf) if log_p else fcf
        else:
            sum_ = _pd_lower_series(x, alph - 1)
            sum_ = math.log1p(sum_) if log_p else 1 + sum_
        if not lower_tail:
            res = (sum_ + d) if log_p else sum_ * d
        else:
            res = _R_Log1_Exp(d + sum_) if log_p else 1 - d * sum_
    else:
        res = _ppois_asymp(alph - 1, x, not lower_tail, log_p)
    if (not log_p) and res < _DBL_MIN / _DBL_EPSILON:
        return math.exp(pgamma_raw(x, alph, lower_tail, True))
    return res


def pgamma(x, alph, scale, lower_tail=True, log_p=False):
    """R's ``pgamma(x, shape=alph, scale)`` (nmath/pgamma.c), bit-exact."""
    if math.isnan(x) or math.isnan(alph) or math.isnan(scale):
        return x + alph + scale
    if alph < 0.0 or scale <= 0.0:
        return _NAN
    x = x / scale
    if math.isnan(x):
        return x
    if alph == 0.0:
        return _dt0(lower_tail, log_p) if x <= 0 else _dt1(lower_tail, log_p)
    return pgamma_raw(x, alph, lower_tail, log_p)


# === dgamma — gamma density (nmath/dgamma.c) =================================
def dgamma(x, shape, scale, give_log=False):
    """R's ``dgamma(x, shape, scale, give_log)`` (nmath/dgamma.c), bit-exact."""
    if math.isnan(x) or math.isnan(shape) or math.isnan(scale):
        return x + shape + scale
    if shape < 0 or scale <= 0:
        return _NAN
    rd0 = _NEGINF if give_log else 0.0
    if x < 0:
        return rd0
    if shape == 0:
        return _INF if x == 0 else rd0
    if x == 0:
        if shape < 1:
            return _INF
        if shape > 1:
            return rd0
        return -math.log(scale) if give_log else 1 / scale
    if shape < 1:
        pr = _dpois_raw(shape, x / scale, give_log)
        if give_log:
            sx = shape / x
            return pr + (math.log(sx) if math.isfinite(sx)
                         else math.log(shape) - math.log(x))
        return pr * shape / x
    pr = _dpois_raw(shape - 1, x / scale, give_log)
    return (pr - math.log(scale)) if give_log else pr / scale


# === qgamma — gamma quantile (nmath/qgamma.c, AS 91 + Newton) ================
def _R_D_log(p, log_p):
    return p if log_p else math.log(p)


def _R_D_LExp(p, log_p):
    return _R_Log1_Exp(p) if log_p else math.log1p(-p)


def _R_DT_log(p, lower_tail, log_p):
    return _R_D_log(p, log_p) if lower_tail else _R_D_LExp(p, log_p)


def _R_DT_Clog(p, lower_tail, log_p):
    return _R_D_LExp(p, log_p) if lower_tail else _R_D_log(p, log_p)


def _R_DT_qIv(p, lower_tail, log_p):
    if log_p:
        return math.exp(p) if lower_tail else -math.expm1(p)
    return p if lower_tail else (0.5 - p + 0.5)


def _qchisq_appr(p, nu, g, lower_tail, log_p, tol):
    C7, C8, C9, C10 = 4.67, 6.66, 6.73, 13.32
    if math.isnan(p) or math.isnan(nu):
        return p + nu
    if (log_p and p > 0) or ((not log_p) and (p < 0 or p > 1)):
        return _NAN
    if nu <= 0:
        return _NAN
    alpha = 0.5 * nu
    c = alpha - 1
    p1 = _R_DT_log(p, lower_tail, log_p)
    if nu < (-1.24) * p1:
        lgam1pa = _lgamma1p(alpha) if alpha < 0.5 else (math.log(alpha) + g)
        ch = math.exp((lgam1pa + p1) / alpha + _M_LN2)
    elif nu > 0.32:
        x = qnorm5(p, 0, 1, lower_tail, log_p)
        p1 = 2. / (9 * nu)
        ch = nu * math.pow(x * math.sqrt(p1) + 1 - p1, 3)
        if ch > 2.2 * nu + 6:
            ch = -2 * (_R_DT_Clog(p, lower_tail, log_p)
                       - c * math.log(0.5 * ch) + g)
    else:
        ch = 0.4
        a = _R_DT_Clog(p, lower_tail, log_p) + g + c * _M_LN2
        while True:
            q = ch
            p1 = 1. / (1 + ch * (C7 + ch))
            p2 = ch * (C9 + ch * (C8 + ch))
            t = -0.5 + (C7 + 2 * ch) * p1 - (C9 + ch * (C10 + 3 * ch)) / p2
            ch -= (1 - math.exp(a + 0.5 * ch) * p2 * p1) / t
            if not (abs(q - ch) > tol * abs(ch)):
                break
    return ch


def qgamma(p, alpha, scale, lower_tail=True, log_p=False):
    """R's ``qgamma(p, shape=alpha, scale)`` (nmath/qgamma.c), bit-exact."""
    EPS1 = 1e-2
    EPS2 = 5e-7
    EPS_N = 1e-15
    MAXIT = 1000
    pMIN = 1e-100
    pMAX = 1 - 1e-14
    i420, i2520, i5040 = 1. / 420., 1. / 2520., 1. / 5040.
    if math.isnan(p) or math.isnan(alpha) or math.isnan(scale):
        return p + alpha + scale
    # R_Q_P01_boundaries(p, 0, +Inf)
    if log_p:
        if p > 0:
            return _NAN
        if p == 0:
            return _INF if lower_tail else 0.0
        if p == _NEGINF:
            return 0.0 if lower_tail else _INF
    else:
        if p < 0 or p > 1:
            return _NAN
        if p == 0:
            return 0.0 if lower_tail else _INF
        if p == 1:
            return _INF if lower_tail else 0.0
    if alpha < 0 or scale <= 0:
        return _NAN
    if alpha == 0:
        return 0.0
    max_it_Newton = 1
    if alpha < 1e-10:
        max_it_Newton = 7
    p_ = _R_DT_qIv(p, lower_tail, log_p)
    g = _lgammafn(alpha)
    ch = _qchisq_appr(p, 2 * alpha, g, lower_tail, log_p, EPS1)
    at_end = False
    if not math.isfinite(ch):
        max_it_Newton = 0
        at_end = True
    elif ch < EPS2:
        max_it_Newton = 20
        at_end = True
    elif p_ > pMAX or p_ < pMIN:
        max_it_Newton = 20
        at_end = True
    if not at_end:
        c = alpha - 1
        s6 = (120 + c * (346 + 127 * c)) * i5040
        ch0 = ch
        for i in range(1, MAXIT + 1):
            q = ch
            p1 = 0.5 * ch
            p2 = p_ - pgamma_raw(p1, alpha, True, False)
            if (not math.isfinite(p2)) or ch <= 0:
                ch = ch0
                max_it_Newton = 27
                break
            t = p2 * math.exp(alpha * _M_LN2 + g + p1 - c * math.log(ch))
            b = t / ch
            a = 0.5 * t - b * c
            s1 = (210 + a * (140 + a * (105 + a * (84 + a * (70 + 60 * a))))) * i420
            s2 = (420 + a * (735 + a * (966 + a * (1141 + 1278 * a)))) * i2520
            s3 = (210 + a * (462 + a * (707 + 932 * a))) * i2520
            s4 = (252 + a * (672 + 1182 * a)
                  + c * (294 + a * (889 + 1740 * a))) * i5040
            s5 = (84 + 2264 * a + c * (1175 + 606 * a)) * i2520
            ch += t * (1 + 0.5 * t * s1 - b * c
                       * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))))
            if abs(q - ch) < EPS2 * ch:
                break
            if abs(q - ch) > 0.1 * ch:
                ch = 0.9 * q if ch < q else 1.1 * q
    # END:
    x = 0.5 * scale * ch
    if max_it_Newton:
        if not log_p:
            p = math.log(p)
            log_p = True
        if x == 0:
            _1_p = 1. + 1e-7
            _1_m = 1. - 1e-7
            x = _DBL_MIN
            p_ = pgamma(x, alpha, scale, lower_tail, log_p)
            if ((lower_tail and p_ > p * _1_p)
                    or ((not lower_tail) and p_ < p * _1_m)):
                return 0.0
        else:
            p_ = pgamma(x, alpha, scale, lower_tail, log_p)
        if p_ == _NEGINF:
            return 0.0
        rd0 = _NEGINF if log_p else 0.0
        for i in range(1, max_it_Newton + 1):
            p1 = p_ - p
            if abs(p1) < abs(EPS_N * p):
                break
            g = dgamma(x, alpha, scale, log_p)
            if g == rd0:
                break
            t = (p1 * math.exp(p_ - g)) if log_p else (p1 / g)
            t = (x - t) if lower_tail else (x + t)
            p_ = pgamma(t, alpha, scale, lower_tail, log_p)
            if (abs(p_ - p) > abs(p1)
                    or (i > 1 and abs(p_ - p) == abs(p1))):
                break
            x = t
    return x


# === pbeta — incomplete beta (nmath/toms708.c, Morris ALGORITHM 708) =========
# Direct port of bratio() + all its sub-algorithms. Feeds pbeta -> pt/pf/
# pbinom/ppois and (via qbeta) the t/F/beta/binom quantiles.
_TOMS_EPS = 2.220446049250313e-16   # = 2 * d1mach(3) = DBL_EPSILON
_M_SQRT_PI = 1.772453850905516027298167483341


def _exparg(which):
    lnb = .69314718055995
    m = 1024 if which == 0 else (-1021 - 1)
    return m * lnb * .99999


def _esum(mu, x, give_log):
    if give_log:
        return x + mu
    if x > 0.:
        if mu > 0:
            return math.exp(mu) * math.exp(x)
        w = mu + x
        if w < 0.:
            return math.exp(mu) * math.exp(x)
    else:
        if mu < 0:
            return math.exp(mu) * math.exp(x)
        w = mu + x
        if w > 0.:
            return math.exp(mu) * math.exp(x)
    return math.exp(w)


def _rexpm1(x):
    p1 = 9.14041914819518e-10
    p2 = .0238082361044469
    q1 = -.499999999085958
    q2 = .107141568980644
    q3 = -.0119041179760821
    q4 = 5.95130811860248e-4
    if abs(x) <= 0.15:
        return x * (((p2 * x + p1) * x + 1.)
                    / ((((q4 * x + q3) * x + q2) * x + q1) * x + 1.))
    w = math.exp(x)
    if x > 0.:
        return w * (0.5 - 1. / w + 0.5)
    return w - 0.5 - 0.5


def _alnrel(a):
    if abs(a) > 0.375:
        return math.log(1. + a)
    p1 = -1.29418923021993
    p2 = .405303492862024
    p3 = -.0178874546012214
    q1 = -1.62752256355323
    q2 = .747811014037616
    q3 = -.0845104217945565
    t = a / (a + 2.)
    t2 = t * t
    w = ((((p3 * t2 + p2) * t2 + p1) * t2 + 1.)
         / (((q3 * t2 + q2) * t2 + q1) * t2 + 1.))
    return t * 2. * w


def _rlog1(x):
    a = .0566749439387324
    b = .0456512608815524
    p0 = .333333333333333
    p1 = -.224696413112536
    p2 = .00620886815375787
    q1 = -1.27408923933623
    q2 = .354508718369557
    if x < -0.39 or x > 0.57:
        w = x + 0.5 + 0.5
        return x - math.log(w)
    if x < -0.18:
        h = (x + .3) / .7
        w1 = a - h * .3
    elif x > 0.18:
        h = x * .75 - .25
        w1 = b + h / 3.
    else:
        h = x
        w1 = 0.
    r = h / (h + 2.)
    t = r * r
    w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1.)
    return t * 2. * (1. / (1. - r) - r * w) + w1


_ERF_A = (7.7105849500132e-5, -.00133733772997339, .0323076579225834,
          .0479137145607681, .128379167095513)
_ERF_B = (.00301048631703895, .0538971687740286, .375795757275549)
_ERF_P = (-1.36864857382717e-7, .564195517478974, 7.21175825088309,
          43.1622272220567, 152.98928504694, 339.320816734344,
          451.918953711873, 300.459261020162)
_ERF_Q = (1., 12.7827273196294, 77.0001529352295, 277.585444743988,
          638.980264465631, 931.35409485061, 790.950925327898,
          300.459260956983)
_ERF_R = (2.10144126479064, 26.2370141675169, 21.3688200555087,
          4.6580782871847, .282094791773523)
_ERF_S = (94.153775055546, 187.11481179959, 99.0191814623914, 18.0124575948747)
_ERF_C = .564189583547756


def _erf__(x):
    a, b, p, q, r, s = _ERF_A, _ERF_B, _ERF_P, _ERF_Q, _ERF_R, _ERF_S
    ax = abs(x)
    if ax <= 0.5:
        t = x * x
        top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.
        bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.
        return x * (top / bot)
    if ax <= 4.:
        top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4])
                * ax + p[5]) * ax + p[6]) * ax + p[7]
        bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4])
                * ax + q[5]) * ax + q[6]) * ax + q[7]
        R = 0.5 - math.exp(-x * x) * top / bot + 0.5
        return -R if x < 0 else R
    if ax >= 5.8:
        return 1. if x > 0 else -1.
    x2 = x * x
    t = 1. / x2
    top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4]
    bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.
    t = (_ERF_C - top / (x2 * bot)) / ax
    R = 0.5 - math.exp(-x2) * t + 0.5
    return -R if x < 0 else R


def _erfc1(ind, x):
    a, b, p, q, r, s = _ERF_A, _ERF_B, _ERF_P, _ERF_Q, _ERF_R, _ERF_S
    ax = abs(x)
    if ax <= 0.5:
        t = x * x
        top = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1.
        bot = ((b[0] * t + b[1]) * t + b[2]) * t + 1.
        ret = 0.5 - x * (top / bot) + 0.5
        if ind != 0:
            ret = math.exp(t) * ret
        return ret
    if ax <= 4.:
        top = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4])
                * ax + p[5]) * ax + p[6]) * ax + p[7]
        bot = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4])
                * ax + q[5]) * ax + q[6]) * ax + q[7]
        ret = top / bot
    else:
        if x <= -5.6:
            ret = 2.
            if ind != 0:
                ret = math.exp(x * x) * 2.
            return ret
        if ind == 0 and (x > 100. or x * x > -_exparg(1)):
            return 0.
        t = 1. / (x * x)
        top = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4]
        bot = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.
        ret = (_ERF_C - t * top / bot) / ax
    if ind != 0:
        if x < 0.:
            ret = math.exp(x * x) * 2. - ret
    else:
        w = x * x
        t = w
        e = w - t
        ret = (0.5 - e + 0.5) * math.exp(-t) * ret
        if x < 0.:
            ret = 2. - ret
    return ret


def _gam1(a):
    d = a - 0.5
    t = (d - 0.5) if d > 0. else a
    if t < 0.:
        r = (-.422784335098468, -.771330383816272, -.244757765222226,
             .118378989872749, 9.30357293360349e-4, -.0118290993445146,
             .00223047661158249, 2.66505979058923e-4, -1.32674909766242e-4)
        s1 = .273076135303957
        s2 = .0559398236957378
        top = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]) * t
                 + r[3]) * t + r[2]) * t + r[1]) * t + r[0]
        bot = (s2 * t + s1) * t + 1.
        w = top / bot
        return (t * w / a) if d > 0. else (a * (w + 0.5 + 0.5))
    elif t == 0:
        return 0.
    else:
        p = (.577215664901533, -.409078193005776, -.230975380857675,
             .0597275330452234, .0076696818164949, -.00514889771323592,
             5.89597428611429e-4)
        q = (1., .427569613095214, .158451672430138, .0261132021441447,
             .00423244297896961)
        top = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]) * t
               + p[1]) * t + p[0]
        bot = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1.
        w = top / bot
        return (t / a * (w - 0.5 - 0.5)) if d > 0. else (a * w)


def _gamln1(a):
    if a < 0.6:
        p0 = .577215664901533
        p1 = .844203922187225
        p2 = -.168860593646662
        p3 = -.780427615533591
        p4 = -.402055799310489
        p5 = -.0673562214325671
        p6 = -.00271935708322958
        q1 = 2.88743195473681
        q2 = 3.12755088914843
        q3 = 1.56875193295039
        q4 = .361951990101499
        q5 = .0325038868253937
        q6 = 6.67465618796164e-4
        w = (((((((p6 * a + p5) * a + p4) * a + p3) * a + p2) * a + p1) * a + p0)
             / ((((((q6 * a + q5) * a + q4) * a + q3) * a + q2) * a + q1) * a
                + 1.))
        return -a * w
    r0 = .422784335098467
    r1 = .848044614534529
    r2 = .565221050691933
    r3 = .156513060486551
    r4 = .017050248402265
    r5 = 4.97958207639485e-4
    s1 = 1.24313399877507
    s2 = .548042109832463
    s3 = .10155218743983
    s4 = .00713309612391
    s5 = 1.16165475989616e-4
    x = a - 0.5 - 0.5
    w = ((((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0)
         / (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1.))
    return x * w


_PSI_P1 = (.0089538502298197, 4.77762828042627, 142.441585084029,
           1186.45200713425, 3633.51846806499, 4138.10161269013,
           1305.60269827897)
_PSI_Q1 = (44.8452573429826, 520.752771467162, 2210.0079924783,
           3641.27349079381, 1908.310765963, 6.91091682714533e-6)
_PSI_P2 = (-2.12940445131011, -7.01677227766759, -4.48616543918019,
           -.648157123766197)
_PSI_Q2 = (32.2703493791143, 89.2920700481861, 54.6117738103215,
           7.77788548522962)


def _psi(x):
    piov4 = .785398163397448
    dx0 = 1.461632144968362341262659542325721325
    p1, q1, p2, q2 = _PSI_P1, _PSI_Q1, _PSI_P2, _PSI_Q2
    xmax1 = 2147483647.0  # INT_MAX
    d2 = 0.5 / (0.5 * _DBL_EPSILON)
    if xmax1 > d2:
        xmax1 = d2
    xsmall = 1e-9
    aug = 0.
    if x < 0.5:
        if abs(x) <= xsmall:
            if x == 0.:
                return 0.
            aug = -1. / x
        else:
            w = -x
            sgn = piov4
            if w <= 0.:
                w = -w
                sgn = -sgn
            if w >= xmax1:
                return 0.
            nq = int(w)
            w -= nq
            nq = int(w * 4.)
            w = (w - nq * 0.25) * 4.
            n = nq // 2
            if n + n != nq:
                w = 1. - w
            z = piov4 * w
            m = n // 2
            if m + m != n:
                sgn = -sgn
            n = (nq + 1) // 2
            m = n // 2
            m += m
            if m == n:
                if z == 0.:
                    return 0.
                aug = sgn * (math.cos(z) / math.sin(z) * 4.)
            else:
                aug = sgn * (math.sin(z) / math.cos(z) * 4.)
        x = 1. - x
    if x <= 3.:
        den = x
        upper = p1[0] * x
        for i in range(1, 6):
            den = (den + q1[i - 1]) * x
            upper = (upper + p1[i]) * x
        den = (upper + p1[6]) / (den + q1[5])
        xmx0 = x - dx0
        return den * xmx0 + aug
    if x < xmax1:
        w = 1. / (x * x)
        den = w
        upper = p2[0] * w
        for i in range(1, 4):
            den = (den + q2[i - 1]) * w
            upper = (upper + p2[i]) * w
        aug = upper / (den + q2[3]) - 0.5 / x + aug
    return aug + math.log(x)


_BCORR_C = (.0833333333333333, -.00277777777760991, 7.9365066682539e-4,
            -5.9520293135187e-4, 8.37308034031215e-4, -.00165322962780713)


def _bcorr(a0, b0):
    c0, c1, c2, c3, c4, c5 = _BCORR_C
    a = min(a0, b0)
    b = max(a0, b0)
    h = a / b
    c = h / (h + 1.)
    x = 1. / (h + 1.)
    x2 = x * x
    s3 = x + x2 + 1.
    s5 = x + x2 * s3 + 1.
    s7 = x + x2 * s5 + 1.
    s9 = x + x2 * s7 + 1.
    s11 = x + x2 * s9 + 1.
    t = 1. / b
    t *= t
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t
         + c1 * s3) * t + c0
    w *= c / b
    t = 1. / a
    t *= t
    return (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a + w


def _algdiv(a, b):
    c0, c1, c2, c3, c4, c5 = _BCORR_C
    if a > b:
        h = b / a
        c = 1. / (h + 1.)
        x = h / (h + 1.)
        d = a + (b - 0.5)
    else:
        h = a / b
        c = h / (h + 1.)
        x = 1. / (h + 1.)
        d = b + (a - 0.5)
    x2 = x * x
    s3 = x + x2 + 1.
    s5 = x + x2 * s3 + 1.
    s7 = x + x2 * s5 + 1.
    s9 = x + x2 * s7 + 1.
    s11 = x + x2 * s9 + 1.
    t = 1. / (b * b)
    w = ((((c5 * s11 * t + c4 * s9) * t + c3 * s7) * t + c2 * s5) * t
         + c1 * s3) * t + c0
    w *= c / b
    u = d * _alnrel(a / b)
    v = a * (math.log(b) - 1.)
    return (w - v - u) if u > v else (w - u - v)


def _gamln(a):
    c0, c1, c2, c3, c4, c5 = _BCORR_C
    d = .418938533204673
    if a <= 0.8:
        return _gamln1(a) - math.log(a)
    elif a <= 2.25:
        return _gamln1(a - 0.5 - 0.5)
    elif a < 10.:
        n = int(a - 1.25)
        t = a
        w = 1.
        for _i in range(1, n + 1):
            t += -1.
            w *= t
        return _gamln1(t - 1.) + math.log(w)
    t = 1. / (a * a)
    w = (((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0) / a
    return d + w + (a - 0.5) * (math.log(a) - 1.)


def _gsumln(a, b):
    x = a + b - 2.
    if x <= 0.25:
        return _gamln1(x + 1.)
    if x <= 1.25:
        return _gamln1(x) + _alnrel(x)
    return _gamln1(x - 1.) + math.log(x * (x + 1.))


def _betaln(a0, b0):
    a = min(a0, b0)
    b = max(a0, b0)
    if a < 8.:
        if a < 1.:
            if b < 8.:
                return _gamln(a) + (_gamln(b) - _gamln(a + b))
            return _gamln(a) + _algdiv(a, b)
        w = 0.
        skip_to_40 = False
        if a < 2.:
            if b <= 2.:
                return _gamln(a) + _gamln(b) - _gsumln(a, b)
            if b < 8.:
                w = 0.
                skip_to_40 = True
            else:
                return _gamln(a) + _algdiv(a, b)
        if not skip_to_40:
            if b <= 1e3:
                n = int(a - 1.)
                w = 1.
                for _i in range(1, n + 1):
                    a += -1.
                    h = a / b
                    w *= h / (h + 1.)
                w = math.log(w)
                if b >= 8.:
                    return w + _gamln(a) + _algdiv(a, b)
                # else fall to L40
            else:
                n = int(a - 1.)
                w = 1.
                for _i in range(1, n + 1):
                    a += -1.
                    w *= a / (a / b + 1.)
                return math.log(w) - n * math.log(b) + (_gamln(a)
                                                        + _algdiv(a, b))
        # L40: 1 < A <= B < 8 reduction of B
        n = int(b - 1.)
        z = 1.
        for _i in range(1, n + 1):
            b += -1.
            z *= b / (a + b)
        return w + math.log(z) + (_gamln(a) + (_gamln(b) - _gsumln(a, b)))
    e = .918938533204673
    w = _bcorr(a, b)
    h = a / b
    u = -(a - 0.5) * math.log(h / (h + 1.))
    v = b * _alnrel(h)
    if u > v:
        return math.log(b) * -0.5 + e + w - v - u
    return math.log(b) * -0.5 + e + w - u - v


def _fpser(a, b, x, eps, log_p):
    if log_p:
        ans = a * math.log(x)
    elif a > eps * 0.001:
        t = a * math.log(x)
        if t < _exparg(1):
            return 0.
        ans = math.exp(t)
    else:
        ans = 1.
    if log_p:
        ans += math.log(b) - math.log(a)
    else:
        ans *= b / a
    tol = eps / a
    an = a + 1.
    t = x
    s = t / an
    while True:
        an += 1.
        t = x * t
        c = t / an
        s += c
        if not (abs(c) > tol):
            break
    if log_p:
        ans += math.log1p(a * s)
    else:
        ans *= a * s + 1.
    return ans


def _apser(a, b, x, eps):
    g = .577215664901533
    bx = b * x
    t = x - bx
    if b * eps <= 0.02:
        c = math.log(x) + _psi(b) + g + t
    else:
        c = math.log(bx) + g + t
    tol = eps * 5. * abs(c)
    j = 1.
    s = 0.
    while True:
        j += 1.
        t *= x - bx / j
        aj = t / j
        s += aj
        if not (abs(aj) > tol):
            break
    return -a * (c + s)


def _bpser(a, b, x, eps, log_p):
    rd0 = _NEGINF if log_p else 0.0
    if x == 0.:
        return rd0
    a0 = min(a, b)
    if a0 >= 1.:
        z = a * math.log(x) - _betaln(a, b)
        ans = (z - math.log(a)) if log_p else (math.exp(z) / a)
    else:
        b0 = max(a, b)
        if b0 < 8.:
            if b0 <= 1.:
                if log_p:
                    ans = a * math.log(x)
                else:
                    ans = math.pow(x, a)
                    if ans == 0.:
                        return ans
                apb = a + b
                if apb > 1.:
                    u = a + b - 1.
                    z = (_gam1(u) + 1.) / apb
                else:
                    z = _gam1(apb) + 1.
                c = (_gam1(a) + 1.) * (_gam1(b) + 1.) / z
                if log_p:
                    ans += math.log(c * (b / apb))
                else:
                    ans *= c * (b / apb)
            else:
                u = _gamln1(a0)
                m = int(b0 - 1.)
                if m >= 1:
                    c = 1.
                    for _i in range(1, m + 1):
                        b0 += -1.
                        c *= b0 / (a0 + b0)
                    u += math.log(c)
                z = a * math.log(x) - u
                b0 += -1.
                apb = a0 + b0
                if apb > 1.:
                    u = a0 + b0 - 1.
                    t = (_gam1(u) + 1.) / apb
                else:
                    t = _gam1(apb) + 1.
                if log_p:
                    ans = z + math.log(a0 / a) + math.log1p(_gam1(b0)) - math.log(t)
                else:
                    ans = math.exp(z) * (a0 / a) * (_gam1(b0) + 1.) / t
        else:
            u = _gamln1(a0) + _algdiv(a0, b0)
            z = a * math.log(x) - u
            ans = (z + math.log(a0 / a)) if log_p else (a0 / a * math.exp(z))
    if ans == rd0 or ((not log_p) and a <= eps * 0.1):
        return ans
    tol = eps / a
    n = 0.
    sum_ = 0.
    c = 1.
    w = 0.
    while True:
        n += 1.
        c *= (0.5 - b / n + 0.5) * x
        w = c / (a + n)
        sum_ += w
        if not (n < 1e7 and abs(w) > tol):
            break
    if log_p:
        if a * sum_ > -1.:
            ans += math.log1p(a * sum_)
        else:
            ans = _NEGINF
    elif a * sum_ > -1.:
        ans *= (a * sum_ + 1.)
    else:
        ans = 0.
    return ans


def _bup(a, b, x, y, n, eps, give_log):
    apb = a + b
    ap1 = a + 1.
    if n > 1 and a >= 1. and apb >= ap1 * 1.1:
        mu = int(abs(_exparg(1)))
        k = int(_exparg(0))
        if mu > k:
            mu = k
        d = math.exp(-float(mu))
    else:
        mu = 0
        d = 1.
    ret = (_brcmp1(mu, a, b, x, y, True) - math.log(a)) if give_log \
        else (_brcmp1(mu, a, b, x, y, False) / a)
    if n == 1 or (give_log and ret == _NEGINF) or ((not give_log) and ret == 0.):
        return ret
    nm1 = n - 1
    w = d
    k = 0
    if b > 1.:
        if y > 1e-4:
            r = (b - 1.) * x / y - a
            if r >= 1.:
                k = int(r) if r < nm1 else nm1
        else:
            k = nm1
        for i in range(0, k):
            ll = float(i)
            d *= (apb + ll) / (ap1 + ll) * x
            w += d
    for i in range(k, nm1):
        ll = float(i)
        d *= (apb + ll) / (ap1 + ll) * x
        w += d
        if d <= eps * w:
            break
    if give_log:
        ret += math.log(w)
    else:
        ret *= w
    return ret


def _bfrac(a, b, x, y, lambda_, eps, log_p):
    if not math.isfinite(lambda_):
        return _NAN
    brc = _brcomp(a, b, x, y, log_p)
    if math.isnan(brc):
        return _NAN
    if (not log_p) and brc == 0.:
        return 0.
    c = lambda_ + 1.
    c0 = b / a
    c1 = 1. / a + 1.
    yp1 = y + 1.
    n = 0.
    p = 1.
    s = a + 1.
    an = 0.
    bn = 1.
    anp1 = 1.
    bnp1 = c / c1
    r = c1 / c
    r0 = 0.
    MAXIT = 1000
    while n < MAXIT:
        n += 1.
        w = n * x * (b - n)
        rescale = not math.isfinite(w)
        if rescale:
            w = n * x * math.ldexp(b - n, -20)
        t = n / a
        e = a / s
        alpha = p * (p + c0) * e * e * (w * x)
        e = (t + 1.) / (c1 + t + t)
        beta = w / s + (math.ldexp(n + e * (c + n * yp1), -20) if rescale
                        else (n + e * (c + n * yp1)))
        p = t + 1.
        s += 2.
        t = alpha * an + beta * anp1
        an = anp1
        anp1 = t
        t = alpha * bn + beta * bnp1
        bn = bnp1
        bnp1 = t
        r0 = r
        r = anp1 / bnp1
        if abs(r - r0) <= eps * r:
            break
        an /= bnp1
        bn /= bnp1
        anp1 = r
        bnp1 = 1.
    return (brc + math.log(r)) if log_p else (brc * r)


def _brcomp(a, b, x, y, log_p):
    rd0 = _NEGINF if log_p else 0.0
    if x == 0. or y == 0.:
        return rd0
    a0 = min(a, b)
    if a0 < 8.:
        if x <= .375:
            lnx = math.log(x)
            lny = _alnrel(-x)
        elif y > .375:
            lnx = math.log(x)
            lny = math.log(y)
        else:
            lnx = _alnrel(-y)
            lny = math.log(y)
        z = a * lnx + b * lny
        if a0 >= 1.:
            z -= _betaln(a, b)
            return z if log_p else math.exp(z)
        b0 = max(a, b)
        if b0 >= 8.:
            u = _gamln1(a0) + _algdiv(a0, b0)
            return (math.log(a0) + (z - u)) if log_p else (a0 * math.exp(z - u))
        if b0 <= 1.:
            e_z = z if log_p else math.exp(z)
            if (not log_p) and e_z == 0.:
                return 0.
            apb = a + b
            if apb > 1.:
                z = (_gam1(apb - 1.) + 1.) / apb
            else:
                z = _gam1(apb) + 1.
            c = (_gam1(a) + 1.) * (_gam1(b) + 1.) / z
            return (e_z + math.log(a0 * c) - math.log1p(a0 / b0)) if log_p \
                else (e_z * (a0 * c) / (a0 / b0 + 1.))
        u = _gamln1(a0)
        n = int(b0 - 1.)
        if n >= 1:
            c = 1.
            for _i in range(1, n + 1):
                b0 += -1.
                c *= b0 / (a0 + b0)
            u = math.log(c) + u
        z -= u
        b0 += -1.
        apb = a0 + b0
        if apb > 1.:
            u = a0 + b0 - 1.
            t = (_gam1(u) + 1.) / apb
        else:
            t = _gam1(apb) + 1.
        return (math.log(a0) + z + math.log1p(_gam1(b0)) - math.log(t)) if log_p \
            else (a0 * math.exp(z) * (_gam1(b0) + 1.) / t)
    else:
        const__ = .398942280401433
        apb = a + b
        lambda_ = (((a - apb * x) if a <= b else (apb * y - b))
                   if math.isfinite(apb) else (a * y - b * x))
        if a <= b:
            h = a / b
            x0 = h / (h + 1.)
            y0 = 1. / (h + 1.)
        else:
            h = b / a
            x0 = 1. / (h + 1.)
            y0 = h / (h + 1.)
        e = -lambda_ / a
        u = (e - math.log(x / x0)) if abs(e) > .6 else _rlog1(e)
        e = lambda_ / b
        v = _rlog1(e) if abs(e) <= .6 else (e - math.log(y / y0))
        z = -(a * u + b * v) if log_p else math.exp(-(a * u + b * v))
        return (-_M_LN_SQRT_2PI + .5 * math.log(b * x0) + z - _bcorr(a, b)) \
            if log_p else (const__ * math.sqrt(b * x0) * z * math.exp(-_bcorr(a, b)))


def _brcmp1(mu, a, b, x, y, give_log):
    a0 = min(a, b)
    if a0 < 8.:
        if x <= .375:
            lnx = math.log(x)
            lny = _alnrel(-x)
        elif y > .375:
            lnx = math.log(x)
            lny = math.log(y)
        else:
            lnx = _alnrel(-y)
            lny = math.log(y)
        z = a * lnx + b * lny
        if a0 >= 1.:
            z -= _betaln(a, b)
            return _esum(mu, z, give_log)
        b0 = max(a, b)
        if b0 >= 8.:
            u = _gamln1(a0) + _algdiv(a0, b0)
            return (math.log(a0) + _esum(mu, z - u, True)) if give_log \
                else (a0 * _esum(mu, z - u, False))
        elif b0 <= 1.:
            ans = _esum(mu, z, give_log)
            if ans == (_NEGINF if give_log else 0.):
                return ans
            apb = a + b
            if apb > 1.:
                z = (_gam1(apb - 1.) + 1.) / apb
            else:
                z = _gam1(apb) + 1.
            c = (math.log1p(_gam1(a)) + math.log1p(_gam1(b)) - math.log(z)) \
                if give_log else ((_gam1(a) + 1.) * (_gam1(b) + 1.) / z)
            return (ans + math.log(a0) + c - math.log1p(a0 / b0)) if give_log \
                else (ans * (a0 * c) / (a0 / b0 + 1.))
        u = _gamln1(a0)
        n = int(b0 - 1.)
        if n >= 1:
            c = 1.
            for _i in range(1, n + 1):
                b0 += -1.
                c *= b0 / (a0 + b0)
            u += math.log(c)
        z -= u
        b0 += -1.
        apb = a0 + b0
        if apb > 1.:
            t = (_gam1(apb - 1.) + 1.) / apb
        else:
            t = _gam1(apb) + 1.
        return (math.log(a0) + _esum(mu, z, True) + math.log1p(_gam1(b0))
                - math.log(t)) if give_log \
            else (a0 * _esum(mu, z, False) * (_gam1(b0) + 1.) / t)
    else:
        const__ = .398942280401433
        apb = a + b
        lambda_ = (((a - apb * x) if a <= b else (apb * y - b))
                   if math.isfinite(apb) else (a * y - b * x))
        if a > b:
            h = b / a
            x0 = 1. / (h + 1.)
            y0 = h / (h + 1.)
        else:
            h = a / b
            x0 = h / (h + 1.)
            y0 = 1. / (h + 1.)
        lx0 = -math.log1p(b / a)
        e = -lambda_ / a
        u = (e - math.log(x / x0)) if abs(e) > 0.6 else _rlog1(e)
        e = lambda_ / b
        v = (e - math.log(y / y0)) if abs(e) > 0.6 else _rlog1(e)
        z = _esum(mu, -(a * u + b * v), give_log)
        return (math.log(const__) + (math.log(b) + lx0) / 2. + z - _bcorr(a, b)) \
            if give_log \
            else (const__ * math.sqrt(b * x0) * z * math.exp(-_bcorr(a, b)))


def _grat_r(a, x, log_r, eps):
    if a * x == 0.:
        return math.exp(-log_r) if x <= a else 0.
    elif a == 0.5:
        if x < 0.25:
            p = _erf__(math.sqrt(x))
            return (0.5 - p + 0.5) * math.exp(-log_r)
        sx = math.sqrt(x)
        return _erfc1(1, sx) / sx * _M_SQRT_PI
    elif x < 1.1:
        an = 3.
        c = x
        sum_ = x / (a + 3.)
        tol = eps * 0.1 / (a + 1.)
        while True:
            an += 1.
            c *= -(x / an)
            t = c / (a + an)
            sum_ += t
            if not (abs(t) > tol):
                break
        j = a * x * ((sum_ / 6. - 0.5 / (a + 2.)) * x + 1. / (a + 1.))
        z = a * math.log(x)
        h = _gam1(a)
        g = h + 1.
        if (x >= 0.25 and (a < x / 2.59)) or (z > -0.13394):
            ll = _rexpm1(z)
            q = ((ll + 0.5 + 0.5) * j - ll) * g - h
            return 0. if q <= 0. else q * math.exp(-log_r)
        else:
            p = math.exp(z) * g * (0.5 - j + 0.5)
            return (0.5 - p + 0.5) * math.exp(-log_r)
    else:
        a2n_1 = 1.
        a2n = 1.
        b2n_1 = x
        b2n = x + (1. - a)
        c = 1.
        while True:
            a2n_1 = x * a2n + c * a2n_1
            b2n_1 = x * b2n + c * b2n_1
            am0 = a2n_1 / b2n_1
            c += 1.
            c_a = c - a
            a2n = a2n_1 + c_a * a2n
            b2n = b2n_1 + c_a * b2n
            an0 = a2n / b2n
            if not (abs(an0 - am0) >= eps * an0):
                break
        return an0


def _bgrat(a, b, x, y, w, eps, log_w):
    """Returns (w_new, ierr)."""
    n_terms = 30
    c = [0.0] * n_terms
    d = [0.0] * n_terms
    bm1 = b - 0.5 - 0.5
    nu = a + bm1 * 0.5
    lnx = math.log(x) if y > 0.375 else _alnrel(-y)
    z = -nu * lnx
    if b * z == 0.:
        return w, 1
    log_r = math.log(b) + math.log1p(_gam1(b)) + b * math.log(z) + nu * lnx
    log_u = log_r - (_algdiv(b, a) + b * math.log(nu))
    u = math.exp(log_u)
    if log_u == _NEGINF:
        return w, 2
    u_0 = (u == 0.)
    if log_w:
        ll = 0. if w == _NEGINF else math.exp(w - log_u)
    else:
        ll = 0. if w == 0. else math.exp(math.log(w) - log_u)
    q_r = _grat_r(b, z, log_r, eps)
    v = 0.25 / (nu * nu)
    t2 = lnx * 0.25 * lnx
    j = q_r
    sum_ = j
    t = 1.
    cn = 1.
    n2 = 0.
    ierr = 0
    for n in range(1, n_terms + 1):
        bp2n = b + n2
        j = (bp2n * (bp2n + 1.) * j + (z + bp2n + 1.) * t) * v
        n2 += 2.
        t *= t2
        cn /= n2 * (n2 + 1.)
        nm1 = n - 1
        c[nm1] = cn
        s = 0.
        if n > 1:
            coef = b - n
            for i in range(1, nm1 + 1):
                s += coef * c[i - 1] * d[nm1 - i]
                coef += b
        d[nm1] = bm1 * cn + s / n
        dj = d[nm1] * j
        sum_ += dj
        if sum_ <= 0.:
            return w, 3
        if abs(dj) <= eps * (sum_ + ll):
            ierr = 0
            break
        elif n == n_terms:
            ierr = 4
    if log_w:
        w = _logspace_add(w, log_u + math.log(sum_))
    else:
        w += (math.exp(log_u + math.log(sum_)) if u_0 else u * sum_)
    return w, ierr


def _basym(a, b, lambda_, eps, log_p):
    num_IT = 20
    e0 = 1.12837916709551
    e1 = .353553390593274
    ln_e0 = 0.120782237635245
    a0 = [0.0] * (num_IT + 1)
    b0 = [0.0] * (num_IT + 1)
    c = [0.0] * (num_IT + 1)
    d = [0.0] * (num_IT + 1)
    f = a * _rlog1(-lambda_ / a) + b * _rlog1(lambda_ / b)
    if log_p:
        t = -f
    else:
        t = math.exp(-f)
        if t == 0.:
            return 0.
    z0 = math.sqrt(f)
    z = z0 / e1 * 0.5
    z2 = f + f
    if a < b:
        h = a / b
        r0 = 1. / (h + 1.)
        r1 = (b - a) / b
        w0 = 1. / math.sqrt(a * (h + 1.))
    else:
        h = b / a
        r0 = 1. / (h + 1.)
        r1 = (b - a) / a
        w0 = 1. / math.sqrt(b * (h + 1.))
    a0[0] = r1 * .66666666666666663
    c[0] = a0[0] * -0.5
    d[0] = -c[0]
    j0 = 0.5 / e0 * _erfc1(1, z0)
    j1 = e1
    sum_ = j0 + d[0] * w0 * j1
    s = 1.
    h2 = h * h
    hn = 1.
    w = w0
    znm1 = z
    zn = z2
    for n in range(2, num_IT + 1, 2):
        hn *= h2
        a0[n - 1] = r0 * 2. * (h * hn + 1.) / (n + 2.)
        np1 = n + 1
        s += hn
        a0[np1 - 1] = r1 * 2. * s / (n + 3.)
        for i in range(n, np1 + 1):
            r = (i + 1.) * -0.5
            b0[0] = r * a0[0]
            for m in range(2, i + 1):
                bsum = 0.
                for jj in range(1, m):
                    mmj = m - jj
                    bsum += (jj * r - mmj) * a0[jj - 1] * b0[mmj - 1]
                b0[m - 1] = r * a0[m - 1] + bsum / m
            c[i - 1] = b0[i - 1] / (i + 1.)
            dsum = 0.
            for jj in range(1, i):
                dsum += d[i - jj - 1] * c[jj - 1]
            d[i - 1] = -(dsum + c[i - 1])
        j0 = e1 * znm1 + (n - 1.) * j0
        j1 = e1 * zn + n * j1
        znm1 = z2 * znm1
        zn = z2 * zn
        w *= w0
        t0 = d[n - 1] * w * j0
        w *= w0
        t1 = d[np1 - 1] * w * j1
        sum_ += t0 + t1
        if abs(t0) + abs(t1) <= eps * sum_:
            break
    if log_p:
        return ln_e0 + t - _bcorr(a, b) + math.log(sum_)
    u = math.exp(-_bcorr(a, b))
    return e0 * t * u * sum_


def _bratio(a, b, x, y, log_p):
    """R's ``bratio`` (toms708.c) -> (w, w1, ierr)."""
    rd0 = _NEGINF if log_p else 0.0
    rd1 = 0.0 if log_p else 1.0
    eps = _TOMS_EPS
    w = rd0
    w1 = rd0
    if math.isnan(x) or math.isnan(y) or math.isnan(a) or math.isnan(b):
        return w, w1, 9
    if a < 0. or b < 0.:
        return w, w1, 1
    if a == 0. and b == 0.:
        return w, w1, 2
    if x < 0. or x > 1.:
        return w, w1, 3
    if y < 0. or y > 1.:
        return w, w1, 4
    z = x + y - 0.5 - 0.5
    if abs(z) > eps * 3.:
        return w, w1, 5
    if x == 0.:
        return (w, w1, 6) if a == 0. else (rd0, rd1, 0)
    if y == 0.:
        return (w, w1, 7) if b == 0. else (rd1, rd0, 0)
    if a == 0.:
        return rd1, rd0, 0
    if b == 0.:
        return rd0, rd1, 0
    eps = max(eps, 1e-15)
    a_lt_b = (a < b)
    if (b if a_lt_b else a) < eps * .001:
        if log_p:
            if a_lt_b:
                w = math.log1p(-a / (a + b))
                w1 = math.log(a / (a + b))
            else:
                w = math.log(b / (a + b))
                w1 = math.log1p(-b / (a + b))
        else:
            w = b / (a + b)
            w1 = a / (a + b)
        return w, w1, 0

    ierr = 0
    do_swap = False

    def _end(wv, w1v):
        return (w1v, wv, ierr) if do_swap else (wv, w1v, ierr)

    def end_from_w(wv):
        if log_p:
            w1v = math.log1p(-wv)
            wv = math.log(wv)
        else:
            w1v = 0.5 - wv + 0.5
        return _end(wv, w1v)

    def end_from_w1(w1v):
        if log_p:
            wv = math.log1p(-w1v)
            w1v = math.log(w1v)
        else:
            wv = 0.5 - w1v + 0.5
        return _end(wv, w1v)

    def end_from_w1_log(w1v):
        if log_p:
            wv = _R_Log1_Exp(w1v)
        else:
            wv = -math.expm1(w1v)
            w1v = math.exp(w1v)
        return _end(wv, w1v)

    if min(a, b) <= 1.:
        do_swap = (x > 0.5)
        if do_swap:
            a0, x0, b0, y0 = b, y, a, x
        else:
            a0, x0, b0, y0 = a, x, b, y
        if b0 < min(eps, eps * a0):
            w = _fpser(a0, b0, x0, eps, log_p)
            w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
            return _end(w, w1)
        if a0 < min(eps, eps * b0) and b0 * x0 <= 1.:
            w1 = _apser(a0, b0, x0, eps)
            return end_from_w1(w1)
        did_bup = False
        go_bpser_w = go_bpser_w1 = do_L131 = False
        n = 20
        if max(a0, b0) > 1.:
            if b0 <= 1.:
                go_bpser_w = True
            elif x0 >= 0.29:
                go_bpser_w1 = True
            elif x0 < 0.1 and math.pow(x0 * b0, a0) <= 0.7:
                go_bpser_w = True
            elif b0 > 15.:
                w1 = 0.
                do_L131 = True
        else:
            if a0 >= min(0.2, b0):
                go_bpser_w = True
            elif math.pow(x0, a0) <= 0.9:
                go_bpser_w = True
            elif x0 >= 0.3:
                go_bpser_w1 = True
        if go_bpser_w:
            w = _bpser(a0, b0, x0, eps, log_p)
            w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
            return _end(w, w1)
        if go_bpser_w1:
            w1 = _bpser(b0, a0, y0, eps, log_p)
            w = _R_Log1_Exp(w1) if log_p else 0.5 - w1 + 0.5
            return _end(w, w1)
        if not do_L131:
            n = 20
            w1 = _bup(b0, a0, y0, x0, n, eps, False)
            did_bup = True
            b0 += n
        # L131:
        w1, ierr1 = _bgrat(b0, a0, y0, x0, w1, 15 * eps, False)
        if w1 == 0 or (0 < w1 < _DBL_MIN):
            if did_bup:
                w1 = _bup(b0 - n, a0, y0, x0, n, eps, True)
            else:
                w1 = _NEGINF
            w1, ierr1 = _bgrat(b0, a0, y0, x0, w1, 15 * eps, True)
            if ierr1:
                ierr = 10 + ierr1
            return end_from_w1_log(w1)
        if ierr1:
            ierr = 10 + ierr1
        return end_from_w1(w1)
    else:
        if math.isfinite(a + b):
            lambda_ = ((a + b) * y - b) if a > b else (a - (a + b) * x)
        else:
            lambda_ = a * y - b * x
        do_swap = (lambda_ < 0.)
        if do_swap:
            lambda_ = -lambda_
            a0, x0, b0, y0 = b, y, a, x
        else:
            a0, x0, b0, y0 = a, x, b, y
        go_bpser_w = go_bfrac = go_L140 = False
        if b0 < 40.:
            if b0 * x0 <= 0.7 or (log_p and lambda_ > 650.):
                go_bpser_w = True
            else:
                go_L140 = True
        elif a0 > b0:
            if b0 <= 100. or lambda_ > b0 * 0.03:
                go_bfrac = True
        elif a0 <= 100.:
            go_bfrac = True
        elif lambda_ > a0 * 0.03:
            go_bfrac = True
        if go_bpser_w:
            w = _bpser(a0, b0, x0, eps, log_p)
            w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
            return _end(w, w1)
        if go_bfrac:
            w = _bfrac(a0, b0, x0, y0, lambda_, eps * 15., log_p)
            w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
            return _end(w, w1)
        if go_L140:
            n = int(b0)
            b0 -= n
            if b0 == 0.:
                n -= 1
                b0 = 1.
            w = _bup(b0, a0, y0, x0, n, eps, False)
            if w < _DBL_MIN and log_p:
                b0 += n
                w = _bpser(a0, b0, x0, eps, log_p)
                w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
                return _end(w, w1)
            if x0 <= 0.7:
                w += _bpser(a0, b0, x0, eps, False)
                return end_from_w(w)
            if a0 <= 15.:
                n = 20
                w += _bup(a0, b0, x0, y0, n, eps, False)
                a0 += n
            w, ierr1 = _bgrat(a0, b0, x0, y0, w, 15 * eps, False)
            if ierr1:
                ierr = 10 + ierr1
            return end_from_w(w)
        # basym (L180)
        w = _basym(a0, b0, lambda_, eps * 100., log_p)
        w1 = _R_Log1_Exp(w) if log_p else 0.5 - w + 0.5
        return _end(w, w1)


def pbeta_raw(x, a, b, lower_tail, log_p):
    if x >= 1:
        return _dt1(lower_tail, log_p)
    if a == 0 or b == 0 or not math.isfinite(a) or not math.isfinite(b):
        if a == 0 and b == 0:
            return -_M_LN2 if log_p else 0.5
        if a == 0 or a / b == 0:
            return _dt1(lower_tail, log_p)
        if b == 0 or b / a == 0:
            return _dt0(lower_tail, log_p)
        return _dt0(lower_tail, log_p) if x < 0.5 else _dt1(lower_tail, log_p)
    if x <= 0:
        return _dt0(lower_tail, log_p)
    x1 = 0.5 - x + 0.5
    w, wc, _ierr = _bratio(a, b, x, x1, log_p)
    return w if lower_tail else wc


def pbeta(x, a, b, lower_tail=True, log_p=False):
    """R's ``pbeta(x, a, b)`` (nmath/pbeta.c -> toms708 bratio), bit-exact."""
    if math.isnan(x) or math.isnan(a) or math.isnan(b):
        return x + a + b
    if a < 0 or b < 0:
        return _NAN
    return pbeta_raw(x, a, b, lower_tail, log_p)


# === chains routed through pbeta / pgamma (pt/pf/ppois/pbinom; nmath) ========
def lbeta(a, b):
    """R's ``lbeta(a, b)`` (nmath/lbeta.c) — log Beta, bit-exact."""
    if math.isnan(a) or math.isnan(b):
        return a + b
    p = q = a
    if b < p:
        p = b
    if b > q:
        q = b
    if p < 0:
        return _NAN
    if p == 0:
        return _INF
    if not math.isfinite(q):
        return _NEGINF
    if p >= 10:
        corr = _lgammacor(p) + _lgammacor(q) - _lgammacor(p + q)
        return (math.log(q) * -0.5 + _M_LN_SQRT_2PI + corr
                + (p - 0.5) * math.log(p / (p + q)) + q * math.log1p(-p / (p + q)))
    elif q >= 10:
        corr = _lgammacor(q) - _lgammacor(p + q)
        return (_lgammafn(p) + corr + p - p * math.log(p + q)
                + (q - 0.5) * math.log1p(-p / (p + q)))
    if p < 1e-306:
        return _lgammafn(p) + (_lgammafn(q) - _lgammafn(p + q))
    return math.log(gammafn(p) * (gammafn(q) / gammafn(p + q)))


def pt(x, n, lower_tail=True, log_p=False):
    """R's ``pt(x, df=n)`` (nmath/pt.c) — routes through pbeta, bit-exact."""
    if math.isnan(x) or math.isnan(n):
        return x + n
    if n <= 0.0:
        return _NAN
    if not math.isfinite(x):
        return _dt0(lower_tail, log_p) if x < 0 else _dt1(lower_tail, log_p)
    if not math.isfinite(n):
        return pnorm5(x, 0.0, 1.0, lower_tail, log_p)
    nx = 1 + (x / n) * x
    if nx > 1e100:
        lval = (-0.5 * n * (2 * math.log(abs(x)) - math.log(n))
                - lbeta(0.5 * n, 0.5) - math.log(0.5 * n))
        val = lval if log_p else math.exp(lval)
    else:
        val = (pbeta(x * x / (n + x * x), 0.5, n / 2., False, log_p)
               if n > x * x else
               pbeta(1. / nx, n / 2., 0.5, True, log_p))
    if x <= 0.:
        lower_tail = not lower_tail
    if log_p:
        if lower_tail:
            return math.log1p(-0.5 * math.exp(val))
        return val - _M_LN2
    val /= 2.
    return (0.5 - val + 0.5) if lower_tail else val   # R_D_Cval


def pf(x, df1, df2, lower_tail=True, log_p=False):
    """R's ``pf(x, df1, df2)`` (nmath/pf.c) — routes through pbeta, bit-exact."""
    if math.isnan(x) or math.isnan(df1) or math.isnan(df2):
        return x + df2 + df1
    if df1 <= 0. or df2 <= 0.:
        return _NAN
    if x <= 0.:
        return _dt0(lower_tail, log_p)
    if x >= _INF:
        return _dt1(lower_tail, log_p)
    if df2 == _INF:
        if df1 == _INF:
            if x < 1.:
                return _dt0(lower_tail, log_p)
            if x == 1.:
                return -_M_LN2 if log_p else 0.5
            return _dt1(lower_tail, log_p)
        return pgamma(x * df1, df1 / 2.0, 2.0, lower_tail, log_p)
    if df1 == _INF:
        return pgamma(df2 / x, df2 / 2.0, 2.0, not lower_tail, log_p)
    if df1 * x > df2:
        x = pbeta(df2 / (df2 + df1 * x), df2 / 2., df1 / 2.,
                  not lower_tail, log_p)
    else:
        x = pbeta(df1 * x / (df2 + df1 * x), df1 / 2., df2 / 2.,
                  lower_tail, log_p)
    return x if not math.isnan(x) else _NAN


def ppois(x, lambda_, lower_tail=True, log_p=False):
    """R's ``ppois(x, lambda)`` (nmath/ppois.c) = pgamma(lambda, x+1, .., upper)."""
    if math.isnan(x) or math.isnan(lambda_):
        return x + lambda_
    if lambda_ < 0.:
        return _NAN
    if x < 0:
        return _dt0(lower_tail, log_p)
    if lambda_ == 0.:
        return _dt1(lower_tail, log_p)
    if not math.isfinite(x):
        return _dt1(lower_tail, log_p)
    x = math.floor(x + 1e-7)
    return pgamma(lambda_, x + 1, 1., not lower_tail, log_p)


def pbinom(x, n, p, lower_tail=True, log_p=False):
    """R's ``pbinom(x, n, p)`` (nmath/pbinom.c) = pbeta(p, x+1, n-x, upper)."""
    if math.isnan(x) or math.isnan(n) or math.isnan(p):
        return x + n + p
    if not math.isfinite(n) or not math.isfinite(p):
        return _NAN
    n = float(round(n))
    if n < 0 or p < 0 or p > 1:
        return _NAN
    if x < 0:
        return _dt0(lower_tail, log_p)
    x = math.floor(x + 1e-7)
    if n <= x:
        return _dt1(lower_tail, log_p)
    return pbeta(p, x + 1, n - x, not lower_tail, log_p)


# === qbeta — beta quantile (nmath/qbeta.c, AS 109 + Newton) ==================
_DBL_very_MIN = 2.2250738585072014e-308 / 4.
_DBL_log_v_MIN = _M_LN2 * (-1021 - 2)
_DBL_1__eps = float.fromhex("0x1.fffffffffffffp-1")


def _R_DT_CIv(p, lower_tail, log_p):
    if log_p:
        return (-math.expm1(p)) if lower_tail else math.exp(p)
    return (0.5 - p + 0.5) if lower_tail else p


def _R_pow_di3(x):
    return x * (x * x)   # R_pow_di(x, 3)


def _clog(x):
    """C ``log``: log(0) -> -Inf, log(<0) -> NaN (no Python exception)."""
    if x > 0:
        return math.log(x)
    return _NEGINF if x == 0 else _NAN


def _qbeta_raw(alpha, p, q, lower_tail, log_p):
    # public qbeta() always passes log_q_cut=-5, n_N=4 -> give_log_q=False
    log_q_cut = -5.0
    n_N = 4
    give_log_q = False
    use_log_x = give_log_q
    fpu = 3e-308
    acu_min = 1e-300
    p_lo = fpu
    p_hi = 1 - 2.22e-16
    const1, const2, const3, const4 = 2.30753, 0.27061, 0.99229, 0.04481
    DBL_MIN = _DBL_MIN

    def _q0():
        return (0.0, 1.0)

    def _q1():
        return (1.0, 0.0)

    # boundary cases
    if alpha == _dt0(lower_tail, log_p):
        return _q0()
    if alpha == _dt1(lower_tail, log_p):
        return _q1()
    if (log_p and alpha > 0) or ((not log_p) and (alpha < 0 or alpha > 1)):
        return _NAN, _NAN
    if p == 0 or q == 0 or not math.isfinite(p) or not math.isfinite(q):
        rdh = -_M_LN2 if log_p else 0.5
        if p == 0 and q == 0:
            if alpha < rdh:
                return _q0()
            if alpha > rdh:
                return _q1()
            return (0.5, 0.5)
        elif p == 0 or p / q == 0:
            return _q0()
        elif q == 0 or q / p == 0:
            return _q1()
        return (0.5, 0.5)

    p_ = _R_DT_qIv(alpha, lower_tail, log_p)
    logbeta = lbeta(p, q)
    swap_tail = (p_ > 0.5)
    log_eps_c = _M_LN2 * (1 - 53)

    y = -1.0
    u_n = 1.0
    add_N_step = True
    n_maybe_swaps = 0
    goto_return = False
    converged = False
    # values carried to L_return
    tx = 0.0
    a = la = pp = qq = u = xinbta = 0.0

    while True:  # maybe_swap
        if swap_tail:
            a = _R_DT_CIv(alpha, lower_tail, log_p)
            la = _R_DT_Clog(alpha, lower_tail, log_p)
            pp, qq = q, p
        else:
            a = p_
            la = _R_DT_log(alpha, lower_tail, log_p)
            pp, qq = p, q
        n_maybe_swaps += 1
        acu = max(acu_min, math.pow(10., -13. - 2.5 / (pp * pp) - 0.5 / (a * a)))
        u0 = (la + math.log(pp) + logbeta) / pp
        rp = pp * (1. - qq) / (pp + 1.)
        t = 0.2
        u0_maybe = (_M_LN2 * (-1021) < u0 < -0.01)
        u_n = 1.0
        skip_init = False
        if (u0_maybe and u0 < (t * log_eps_c - _clog(
                abs(pp * (1. - qq) * (2. - qq) / (2. * (pp + 2.))))) / 2.):
            rp = rp * math.exp(u0)
            u = (u0 - math.log1p(rp) / pp) if rp > -1. else u0
            tx = xinbta = math.exp(u)
            use_log_x = True
            skip_init = True

        if not skip_init:
            r = math.sqrt(-2 * la)
            y = r - (const1 + const2 * r) / (1. + (const3 + const4 * r) * r)
            if pp > 1 and qq > 1:
                r = (y * y - 3.) / 6.
                s = 1. / (pp + pp - 1.)
                t = 1. / (qq + qq - 1.)
                h = 2. / (s + t)
                w = y * math.sqrt(h + r) / h - (t - s) * (r + 5. / 6. - 2. / (3. * h))
                if w > 300:
                    t = w + w + math.log(qq) - math.log(pp)
                    u = (-math.log1p(math.exp(t))) if t <= 18 else (-t - math.exp(-t))
                    xinbta = math.exp(u)
                else:
                    xinbta = pp / (pp + qq * math.exp(w + w))
                    u = -math.log1p(qq / pp * math.exp(w + w))
            else:
                r = qq + qq
                t = 1. / (3. * math.sqrt(qq))
                t = r * _R_pow_di3(1. + t * (-t + y))
                s = 4. * pp + r - 2.
                if t == 0 or (t < 0. and s >= t):
                    l1ma = (_R_DT_log(alpha, lower_tail, log_p) if swap_tail
                            else _R_DT_Clog(alpha, lower_tail, log_p))
                    xx = (l1ma + math.log(qq) + logbeta) / qq
                    if xx <= 0.:
                        xinbta = -math.expm1(xx)
                        u = _R_Log1_Exp(xx)
                    else:
                        r_ = rp * math.exp(u0)
                        u = (u0 - math.log1p(r_) / pp) if r_ > -1. else u0
                        xinbta = math.exp(u)
                else:
                    t = s / t
                    if t <= 1.:
                        u = u0
                        xinbta = math.exp(u)
                    else:
                        xinbta = 1. - 2. / (t + 1.)
                        u = math.log1p(-2. / (t + 1.))

            if ((swap_tail and u >= -math.exp(log_q_cut))
                    or ((not swap_tail) and u >= -math.exp(4 * log_q_cut)
                        and pp / qq < 1000.)):
                swap_tail = not swap_tail
                if swap_tail:
                    a = _R_DT_CIv(alpha, lower_tail, log_p)
                    la = _R_DT_Clog(alpha, lower_tail, log_p)
                    pp, qq = q, p
                else:
                    a = p_
                    la = _R_DT_log(alpha, lower_tail, log_p)
                    pp, qq = p, q
                u = _R_Log1_Exp(u)
                xinbta = math.exp(u)

            if not use_log_x:
                use_log_x = (u < log_q_cut)
            bad_u = not math.isfinite(u)
            bad_init = bad_u or xinbta > p_hi
            tx = xinbta
            if bad_u or u < log_q_cut:
                w = pbeta_raw(_DBL_very_MIN, pp, qq, True, log_p)
                if w > (la if log_p else a):
                    if log_p or abs(w - a) < abs(0 - a):
                        tx = _DBL_very_MIN
                        u_n = _DBL_log_v_MIN
                    else:
                        tx = 0.
                        u_n = _NEGINF
                    use_log_x = bool(log_p)
                    add_N_step = False
                    goto_return = True
                else:
                    if u < _DBL_log_v_MIN:
                        u = _DBL_log_v_MIN
                        xinbta = _DBL_very_MIN
            if (not goto_return) and bad_init and not (use_log_x and tx > 0):
                if u == _NEGINF:
                    u = _M_LN2 * (-1021)
                    xinbta = DBL_MIN
                else:
                    xinbta = (0.5 if xinbta > 1.1
                              else (math.exp(u) if xinbta < p_lo else p_hi))
                    if bad_u:
                        u = math.log(xinbta)

        if goto_return:
            break

        # L_Newton
        r = 1 - pp
        t = 1 - qq
        wprev = 0.
        prev = 1.
        adj = 1.
        jump_swap = False
        if use_log_x:
            for i_pb in range(1000):
                y = pbeta_raw(xinbta, pp, qq, True, True)
                w = 0. if y == _NEGINF else \
                    (y - la) * math.exp(y - u + logbeta + r * u
                                        + t * _R_Log1_Exp(u))
                if not math.isfinite(w):
                    if n_maybe_swaps <= 1:
                        jump_swap = True
                        break
                    return _NAN, _NAN
                if i_pb >= n_N and w * wprev <= 0.:
                    prev = max(abs(adj), fpu)
                g = 1
                for _i_inn in range(1000):
                    adj = g * w
                    if abs(adj) < prev:
                        u_n = u - adj
                        if u_n <= 0.:
                            if prev <= acu or abs(w) <= acu:
                                converged = True
                            break
                    g /= 3
                if converged:
                    break
                D = min(abs(adj), abs(u_n - u))
                if D <= 4e-16 * abs(u_n + u):
                    converged = True
                    break
                u = u_n
                xinbta = math.exp(u)
                wprev = w
        else:
            for i_pb in range(1000):
                y = pbeta_raw(xinbta, pp, qq, True, log_p)
                w = ((y - la) * math.exp(y + logbeta + r * math.log(xinbta)
                                         + t * math.log1p(-xinbta)) if log_p
                     else (y - a) * math.exp(logbeta + r * math.log(xinbta)
                                             + t * math.log1p(-xinbta)))
                if not math.isfinite(w):
                    if n_maybe_swaps <= 2:
                        if (not log_p) and n_maybe_swaps == 2:
                            use_log_x = True
                        if (not log_p) or n_maybe_swaps <= 1:
                            jump_swap = True
                            break
                    return _NAN, _NAN
                if i_pb >= n_N and w * wprev <= 0.:
                    prev = max(abs(adj), fpu)
                g = 1
                for _i_inn in range(1000):
                    adj = g * w
                    if i_pb < n_N or abs(adj) < prev:
                        tx = xinbta - adj
                        if 0. <= tx <= 1.:
                            if prev <= acu or abs(w) <= acu:
                                converged = True
                                break
                            if tx != 0. and tx != 1:
                                break
                    g /= 3
                if converged:
                    break
                if abs(tx - xinbta) <= 4e-16 * (tx + xinbta):
                    converged = True
                    break
                xinbta = tx
                if tx == 0:
                    break
                wprev = w
        if jump_swap:
            continue
        # (R warns ME_PRECISION here if not converged; warning omitted)
        break

    # L_converged
    if not goto_return:
        log_ = log_p or use_log_x
        if (log_ and y == _NEGINF) or ((not log_) and y == 0):
            w = pbeta_raw(_DBL_very_MIN, pp, qq, True, log_)
            if log_ or abs(w - a) <= abs(y - a):
                tx = _DBL_very_MIN
                u_n = _DBL_log_v_MIN
            add_N_step = False
    # L_return
    r = 1 - pp
    t = 1 - qq
    if use_log_x:
        if add_N_step:
            if u_n != 1.:
                xinbta = math.exp(u_n)
            y = pbeta_raw(xinbta, pp, qq, True, log_p)
            w = ((y - la) * math.exp(y + logbeta + r * math.log(xinbta)
                                     + t * math.log1p(-xinbta)) if log_p
                 else (y - a) * math.exp(logbeta + r * math.log(xinbta)
                                         + t * math.log1p(-xinbta)))
            tx = (xinbta - w) if math.isfinite(w) else xinbta
        else:
            if swap_tail:
                return -math.expm1(u_n), math.exp(u_n)
            return math.exp(u_n), -math.expm1(u_n)
    if swap_tail:
        return 1 - tx, tx
    return tx, 1 - tx


def qbeta(alpha, p, q, lower_tail=True, log_p=False):
    """R's ``qbeta(alpha, p, q)`` (nmath/qbeta.c), bit-exact."""
    if math.isnan(p) or math.isnan(q) or math.isnan(alpha):
        return p + q + alpha
    if p < 0. or q < 0.:
        return _NAN
    return _qbeta_raw(alpha, p, q, lower_tail, log_p)[0]


# === dt / qt / qf (nmath dt.c / qt.c / qf.c) ================================
def dt(x, n, give_log=False):
    """R's ``dt(x, df=n)`` (nmath/dt.c) — t-density via bd0/stirlerr, bit-exact."""
    if math.isnan(x) or math.isnan(n):
        return x + n
    if n <= 0:
        return _NAN
    rd0 = _NEGINF if give_log else 0.0
    if not math.isfinite(x):
        return rd0
    if not math.isfinite(n):
        return dnorm5(x, 0., 1., give_log)
    t = -_bd0(n / 2., (n + 1) / 2.) + _stirlerr((n + 1) / 2.) - _stirlerr(n / 2.)
    x2n = x * x / n
    ax = 0.
    lrg_x2n = (x2n > 1. / _DBL_EPSILON)
    if lrg_x2n:
        ax = abs(x)
        l_x2n = math.log(ax) - math.log(n) / 2.
        u = n * l_x2n
    elif x2n > 0.2:
        l_x2n = math.log(1 + x2n) / 2.
        u = n * l_x2n
    else:
        l_x2n = math.log1p(x2n) / 2.
        u = -_bd0(n / 2., (n + x * x) / 2.) + x * x / 2.
    if give_log:
        return t - u - (_M_LN_SQRT_2PI + l_x2n)
    i_sqrt = (math.sqrt(n) / ax) if lrg_x2n else math.exp(-l_x2n)
    return math.exp(t - u) * _M_1_SQRT_2PI * i_sqrt


def tanpi(x):
    """R's portable ``tanpi(x) = tan(pi*x)`` (cospi.c fallback)."""
    if math.isnan(x):
        return x
    if not math.isfinite(x):
        return _NAN
    x = math.fmod(x, 1.)
    if x <= -0.5:
        x += 1
    elif x > 0.5:
        x -= 1
    return 0. if x == 0. else (_NAN if x == 0.5 else math.tan(math.pi * x))


_M_1_PI = 0.318309886183790671537767526745
_M_PI_2 = 1.570796326794896619231321691640


def qt(p, ndf, lower_tail=True, log_p=False):
    """R's ``qt(p, df=ndf)`` (nmath/qt.c) — routes via qnorm/pt/dt, bit-exact."""
    eps = 1.e-12
    if math.isnan(p) or math.isnan(ndf):
        return p + ndf
    # R_Q_P01_boundaries(p, -Inf, +Inf)
    if log_p:
        if p > 0:
            return _NAN
        if p == 0:
            return _INF if lower_tail else _NEGINF
        if p == _NEGINF:
            return _NEGINF if lower_tail else _INF
    else:
        if p < 0 or p > 1:
            return _NAN
        if p == 0:
            return _NEGINF if lower_tail else _INF
        if p == 1:
            return _INF if lower_tail else _NEGINF
    if ndf <= 0:
        return _NAN
    if ndf < 1:
        accu = 1e-13
        Eps = 1e-11
        pv = _R_DT_qIv(p, lower_tail, log_p)
        if pv > 1 - _DBL_EPSILON:
            return _INF
        pp = min(1 - _DBL_EPSILON, pv * (1 + Eps))
        ux = 1.
        while ux < 1.7976931348623157e308 and pt(ux, ndf, True, False) < pp:
            ux *= 2
        pp = pv * (1 - Eps)
        lx = -1.
        while lx > -1.7976931348623157e308 and pt(lx, ndf, True, False) > pp:
            lx *= 2
        it = 0
        nx = 0.5 * (lx + ux)
        while True:
            nx = 0.5 * (lx + ux)
            if pt(nx, ndf, True, False) > pv:
                ux = nx
            else:
                lx = nx
            it += 1
            if not ((ux - lx) / abs(nx) > accu and it < 1000):
                break
        return 0.5 * (lx + ux)
    if ndf > 1e20:
        return qnorm5(p, 0., 1., lower_tail, log_p)
    P = math.exp(p) if log_p else p     # R_D_qIv
    neg = ((not lower_tail) or P < 0.5) and (lower_tail or P > 0.5)
    is_neg_lower = (lower_tail == neg)
    if neg:
        P = 2 * ((P if lower_tail else -math.expm1(p)) if log_p
                 else (P if lower_tail else (0.5 - p + 0.5)))   # R_D_Lval
    else:
        P = 2 * ((-math.expm1(p) if lower_tail else P) if log_p
                 else ((0.5 - p + 0.5) if lower_tail else p))   # R_D_Cval
    if abs(ndf - 2) < eps:
        if P > _DBL_MIN:
            if 3 * P < _DBL_EPSILON:
                q = 1 / math.sqrt(P)
            elif P > 0.9:
                q = (1 - P) * math.sqrt(2 / (P * (2 - P)))
            else:
                q = math.sqrt(2 / (P * (2 - P)) - 2)
        else:
            if log_p:
                q = (math.exp(-p / 2) / _M_SQRT2) if is_neg_lower \
                    else 1 / math.sqrt(-math.expm1(p))
            else:
                q = _INF
    elif ndf < 1 + eps:
        if P == 1.:
            q = 0.
        elif P > 0:
            q = 1 / tanpi(P / 2.)
        else:
            if log_p:
                q = (_M_1_PI * math.exp(-p)) if is_neg_lower \
                    else -1. / (math.pi * math.expm1(p))
            else:
                q = _INF
    else:
        x = 0.
        log_P2 = 0.
        a = 1 / (ndf - 0.5)
        b = 48 / (a * a)
        c = ((20700 * a / b - 98) * a - 16) * a + 96.36
        d = ((94.5 / (b + c) - 3) / b + 1) * math.sqrt(a * _M_PI_2) * ndf
        P_ok1 = P > _DBL_MIN or not log_p
        P_ok = P_ok1
        if P_ok1:
            y = math.pow(d * P, 2.0 / ndf)
            P_ok = (y >= _DBL_EPSILON)
        if not P_ok:
            log_P2 = (_R_D_log(p, log_p) if is_neg_lower
                      else _R_D_LExp(p, log_p))
            x = (math.log(d) + _M_LN2 + log_P2) / ndf
            y = math.exp(2 * x)
        if (ndf < 2.1 and P > 0.5) or y > 0.05 + a:
            if P_ok:
                x = qnorm5(0.5 * P, 0., 1., True, False)
            else:
                x = qnorm5(log_P2, 0., 1., lower_tail, True)
            y = x * x
            if ndf < 5:
                c += 0.3 * (ndf - 4.5) * (x + 0.6)
            c = (((0.05 * d * x - 5) * x - 7) * x - 2) * x + b + c
            y = (((((0.4 * y + 6.3) * y + 36) * y + 94.5) / c
                  - y - 3) / b + 1) * x
            y = math.expm1(a * y * y)
            q = math.sqrt(ndf * y)
        elif (not P_ok) and x < -_M_LN2 * 53:
            q = math.sqrt(ndf) * math.exp(-x)
        else:
            y = ((1 / (((ndf + 6) / (ndf * y) - 0.089 * d - 0.822)
                       * (ndf + 2) * 3) + 0.5 / (ndf + 4))
                 * y - 1) * (ndf + 1) / (ndf + 2) + 1 / y
            q = math.sqrt(ndf * y)
        if P_ok1:
            M = abs(math.sqrt(1.7976931348623157e308 / 2.) - ndf)
            it = 0
            while it < 10:
                it += 1
                y = dt(q, ndf, False)
                if not (y > 0):
                    break
                x = (pt(q, ndf, False, False) - P / 2) / y
                if not math.isfinite(x) or not (abs(x) > 1e-14 * abs(q)):
                    break
                F = (q * (ndf + 1) / (2 * (q * q + ndf))) if abs(q) < M \
                    else ((ndf + 1) / (2 * (q + ndf / q)))
                del_q = x * (1. + x * F)
                if math.isfinite(del_q) and math.isfinite(q + del_q):
                    q += del_q
                elif math.isfinite(x) and math.isfinite(q + x):
                    q += x
                else:
                    break
    return -q if neg else q


def qf(p, df1, df2, lower_tail=True, log_p=False):
    """R's ``qf(p, df1, df2)`` (nmath/qf.c) — routes via qbeta/qchisq, bit-exact."""
    if math.isnan(p) or math.isnan(df1) or math.isnan(df2):
        return p + df1 + df2
    if df1 <= 0. or df2 <= 0.:
        return _NAN
    # R_Q_P01_boundaries(p, 0, +Inf)
    if log_p:
        if p > 0:
            return _NAN
        if p == 0:
            return _INF if lower_tail else 0.0
        if p == _NEGINF:
            return 0.0 if lower_tail else _INF
    else:
        if p < 0 or p > 1:
            return _NAN
        if p == 0:
            return 0.0 if lower_tail else _INF
        if p == 1:
            return _INF if lower_tail else 0.0
    if df1 <= df2 and df2 > 4e5:
        if not math.isfinite(df1):
            return 1.
        return qgamma(p, df1 / 2.0, 2.0, lower_tail, log_p) / df1
    elif df1 > 4e5:
        return df2 / qgamma(p, df2 / 2.0, 2.0, not lower_tail, log_p)
    p = (1. / qbeta(p, df2 / 2, df1 / 2, not lower_tail, log_p) - 1.) * (df2 / df1)
    return p if not math.isnan(p) else _NAN


# === public discrete/beta densities (dbinom/dpois/dbeta; nmath) =============
def _R_nonint(x):
    # R_nonint: |x - nearbyint(x)| > 1e-9*max(1,|x|); nearbyint == round-half-even
    return abs(x - float(round(x))) > 1e-9 * max(1.0, abs(x))


def dbinom(x, n, p, give_log=False):
    """R's ``dbinom(x, n, p)`` (nmath/dbinom.c), bit-exact."""
    if math.isnan(x) or math.isnan(n) or math.isnan(p):
        return x + n + p
    if p < 0 or p > 1 or (n < 0 or _R_nonint(n)):
        return _NAN
    rd0 = _NEGINF if give_log else 0.0
    if _R_nonint(x):
        return rd0
    if x < 0 or not math.isfinite(x):
        return rd0
    n = float(round(n))
    x = float(round(x))
    return _dbinom_raw(x, n, p, 1 - p, give_log)


def dpois(x, lambda_, give_log=False):
    """R's ``dpois(x, lambda)`` (nmath/dpois.c), bit-exact."""
    if math.isnan(x) or math.isnan(lambda_):
        return x + lambda_
    if lambda_ < 0:
        return _NAN
    rd0 = _NEGINF if give_log else 0.0
    if _R_nonint(x):
        return rd0
    if x < 0 or not math.isfinite(x):
        return rd0
    x = float(round(x))
    return _dpois_raw(x, lambda_, give_log)


def dbeta(x, a, b, give_log=False):
    """R's ``dbeta(x, a, b)`` (nmath/dbeta.c), bit-exact."""
    if math.isnan(x) or math.isnan(a) or math.isnan(b):
        return x + a + b
    if a < 0 or b < 0:
        return _NAN
    rd0 = _NEGINF if give_log else 0.0
    if x < 0 or x > 1:
        return rd0
    if a == 0 or b == 0 or not math.isfinite(a) or not math.isfinite(b):
        if a == 0 and b == 0:
            return _INF if (x == 0 or x == 1) else rd0
        if a == 0 or a / b == 0:
            return _INF if x == 0 else rd0
        if b == 0 or b / a == 0:
            return _INF if x == 1 else rd0
        return _INF if x == 0.5 else rd0
    if x == 0:
        if a > 1:
            return rd0
        if a < 1:
            return _INF
    if x == 1:
        if b > 1:
            return rd0
        if b < 1:
            return _INF
    if a <= 2 or b <= 2:
        lval = (a - 1) * math.log(x) + (b - 1) * math.log1p(-x) - lbeta(a, b)
    else:
        lval = math.log(a + b - 1) + _dbinom_raw(a - 1, a + b - 2, x, 1 - x, True)
    return lval if give_log else math.exp(lval)


# === qbinom / qpois — discrete quantiles (nmath qDiscrete_search.h) ==========
def _do_search(y, z, p, cdf, incr, lower_tail, log_p, y_max):
    # z is a 1-element mutable list [z_val]; returns root y, updates z.
    left = (z[0] >= p) if lower_tail else (z[0] < p)
    if left:
        while True:
            newz = -1.
            if y > 0:
                newz = cdf(y - incr, lower_tail, log_p)
            elif y < 0:
                y = 0
            if (y == 0 or math.isnan(newz)
                    or (newz < p if lower_tail else newz >= p)):
                return y
            y = max(0, y - incr)
            z[0] = newz
    else:
        while True:
            prevy = y
            newz = -1.
            y += incr
            if y_max is not None:
                if y < y_max:
                    newz = cdf(y, lower_tail, log_p)
                elif y > y_max:
                    y = y_max
            else:
                newz = cdf(y, lower_tail, log_p)
            if ((y_max is not None and y == y_max) or math.isnan(newz)
                    or (newz >= p if lower_tail else newz < p)):
                if incr <= 1:
                    z[0] = newz
                    return y
                return prevy
            z[0] = newz


def _q_discrete(p, lower_tail, log_p, mu, sigma, gamma, cdf, y_max):
    z = qnorm5(p, 0., 1., lower_tail, log_p)
    y = float(round(mu + sigma * (z + gamma * (z * z - 1) / 6)))
    if y_max is not None:
        if y > y_max:
            y = y_max
        elif y < 0:
            y = 0.
    elif y < 0:
        y = 0.
    zc = [cdf(y, lower_tail, log_p)]
    _pf_n_, _pf_L_, _yLarge_ = 8., 2., 4096.
    _incF_, _iShrink_, _relTol_, _xf_ = 1. / 64, 8., 1e-15, 4.
    if log_p:
        e = _pf_L_ * _DBL_EPSILON
        if lower_tail and p > -1.7976931348623157e308:
            p *= 1 + e
        else:
            p *= 1 - e
    else:
        e = _pf_n_ * _DBL_EPSILON
        if lower_tail:
            p *= 1 - e
        elif 1 - p > _xf_ * e:
            p *= 1 + e
    if y < _yLarge_:
        return _do_search(y, zc, p, cdf, 1, lower_tail, log_p, y_max)
    incr = math.floor(y * _incF_)
    while True:
        oldincr = incr
        y = _do_search(y, zc, p, cdf, incr, lower_tail, log_p, y_max)
        incr = max(1, math.floor(incr / _iShrink_))
        if not (oldincr > 1 and incr > y * _relTol_):
            break
    return y


def qpois(p, lambda_, lower_tail=True, log_p=False):
    """R's ``qpois(p, lambda)`` (nmath/qpois.c) — discrete search, bit-exact."""
    if math.isnan(p) or math.isnan(lambda_):
        return p + lambda_
    if not math.isfinite(lambda_):
        return _NAN
    if lambda_ < 0:
        return _NAN
    if (log_p and p > 0) or ((not log_p) and (p < 0 or p > 1)):
        return _NAN
    if lambda_ == 0:
        return 0.
    if p == _dt0(lower_tail, log_p):
        return 0.
    if p == _dt1(lower_tail, log_p):
        return _INF
    sigma = math.sqrt(lambda_)
    gamma = 1.0 / sigma

    def _cdf(y, lt, lg):
        return ppois(y, lambda_, lt, lg)
    return _q_discrete(p, lower_tail, log_p, lambda_, sigma, gamma, _cdf, None)


def qbinom(p, n, pr, lower_tail=True, log_p=False):
    """R's ``qbinom(p, n, pr)`` (nmath/qbinom.c) — discrete search, bit-exact."""
    if math.isnan(p) or math.isnan(n) or math.isnan(pr):
        return p + n + pr
    if not math.isfinite(n) or not math.isfinite(pr):
        return _NAN
    if not math.isfinite(p) and not log_p:
        return _NAN
    n = float(round(n))
    if pr < 0 or pr > 1 or n < 0:
        return _NAN
    # R_Q_P01_boundaries(p, 0, n)
    if log_p:
        if p > 0:
            return _NAN
        if p == 0:
            return n if lower_tail else 0.0
        if p == _NEGINF:
            return 0.0 if lower_tail else n
    else:
        if p < 0 or p > 1:
            return _NAN
        if p == 0:
            return 0.0 if lower_tail else n
        if p == 1:
            return n if lower_tail else 0.0
    if pr == 0. or n == 0:
        return 0.
    if pr == 1.:
        return n
    q = 1 - pr
    mu = n * pr
    sigma = math.sqrt(n * pr * q)
    gamma = (q - pr) / sigma

    def _cdf(y, lt, lg):
        return pbinom(y, n, pr, lt, lg)
    return _q_discrete(p, lower_tail, log_p, mu, sigma, gamma, _cdf, n)


# === dexp / pexp / qexp (nmath dexp.c / pexp.c / qexp.c) =====================
def dexp(x, scale, give_log=False):
    if math.isnan(x) or math.isnan(scale):
        return x + scale
    if scale <= 0.:
        return _NAN
    if x < 0.:
        return _NEGINF if give_log else 0.0
    return ((-x / scale) - math.log(scale)) if give_log \
        else math.exp(-x / scale) / scale


def pexp(x, scale, lower_tail=True, log_p=False):
    if math.isnan(x) or math.isnan(scale):
        return x + scale
    if scale < 0:
        return _NAN
    if x <= 0.:
        return _dt0(lower_tail, log_p)
    x = -(x / scale)
    if lower_tail:
        return _R_Log1_Exp(x) if log_p else -math.expm1(x)
    return x if log_p else math.exp(x)


def qexp(p, scale, lower_tail=True, log_p=False):
    if math.isnan(p) or math.isnan(scale):
        return p + scale
    if scale < 0:
        return _NAN
    if (log_p and p > 0) or ((not log_p) and (p < 0 or p > 1)):
        return _NAN
    if p == _dt0(lower_tail, log_p):
        return 0.
    return -scale * _R_DT_Clog(p, lower_tail, log_p)


# ============================================================================
# numpy-vectorized pure-Python fallbacks (bit-identical to the scalar kernels;
# used by _disp when the native Rust extension is absent). Same float-op order,
# masked per-element convergence. The TOMS-708 incomplete-beta core (pbeta ->
# pt/pf/pbinom) and the Newton quantiles (qgamma/qbeta) are NOT vectorized
# (deeply branched + per-element convergence) and keep the scalar loop.
# ============================================================================

def dgamma_vec(x, shape, scale, give_log=False):
    """Vectorised :func:`dgamma` over broadcast (x, shape, scale)."""
    x, shape, scale = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(shape, dtype=float),
        np.asarray(scale, dtype=float))
    rd0 = _NEGINF if give_log else 0.0
    out = np.full(x.shape, np.nan)
    ok = ~((shape < 0) | (scale <= 0))
    s0 = ok & (shape == 0)
    out[s0] = np.where(x[s0] == 0.0, _INF, rd0)
    ok = ok & ~s0
    out[ok & (x < 0)] = rd0
    ok = ok & (x >= 0)
    xz = ok & (x == 0.0)
    if xz.any():
        sh = shape[xz]
        out[xz] = np.where(sh < 1, _INF, np.where(
            sh > 1, rd0, (-np.log(scale[xz])) if give_log else 1.0 / scale[xz]))
    pos = ok & (x > 0)
    lt1 = pos & (shape < 1)
    ge1 = pos & (shape >= 1)
    if lt1.any():
        sh, xm, sc = shape[lt1], x[lt1], scale[lt1]
        pr = _dpois_raw(sh, xm / sc, give_log)
        if give_log:
            sx = sh / xm
            out[lt1] = pr + np.where(np.isfinite(sx),
                                     np.log(np.where(np.isfinite(sx), sx, 1.0)),
                                     np.log(sh) - np.log(xm))
        else:
            out[lt1] = pr * sh / xm
    if ge1.any():
        sh, xm, sc = shape[ge1], x[ge1], scale[ge1]
        pr = _dpois_raw(sh - 1.0, xm / sc, give_log)
        out[ge1] = (pr - np.log(sc)) if give_log else pr / sc
    return out


_PY_VEC["dgamma"] = dgamma_vec


def _R_Log1_Exp_vec(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > -_M_LN2, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def _logcf_vec(x, i, d, eps):
    """Vectorised :func:`_logcf` (x array; i, d, eps scalar). Masked CF."""
    x = np.asarray(x, dtype=float)
    shp = x.shape
    c1 = np.full(shp, 2.0 * d)
    c2 = np.full(shp, i + d)
    c4 = np.full(shp, (i + d) + d)
    a1 = np.full(shp, i + d)
    b1 = i * ((i + d) - i * x)
    b2 = d * d * x
    a2 = c4 * c2 - b2
    b2 = c4 * b1 - i * b2
    sf = _PG_SCALEFACTOR
    for _ in range(100000):
        m = np.abs(a2 * b1 - a1 * b2) > np.abs(eps * b1 * b2)
        if not m.any():
            break
        c3 = c2 * c2 * x
        c2 = np.where(m, c2 + d, c2)
        c4 = np.where(m, c4 + d, c4)
        a1 = np.where(m, c4 * a2 - c3 * a1, a1)
        b1 = np.where(m, c4 * b2 - c3 * b1, b1)
        c3 = c1 * c1 * x
        c1 = np.where(m, c1 + d, c1)
        c4 = np.where(m, c4 + d, c4)
        a2 = np.where(m, c4 * a1 - c3 * a2, a2)
        b2 = np.where(m, c4 * b1 - c3 * b2, b2)
        big = m & (np.abs(b2) > sf)
        sml = m & (np.abs(b2) < 1.0 / sf)
        if big.any():
            a1 = np.where(big, a1 / sf, a1)
            b1 = np.where(big, b1 / sf, b1)
            a2 = np.where(big, a2 / sf, a2)
            b2 = np.where(big, b2 / sf, b2)
        if sml.any():
            a1 = np.where(sml, a1 * sf, a1)
            b1 = np.where(sml, b1 * sf, b1)
            a2 = np.where(sml, a2 * sf, a2)
            b2 = np.where(sml, b2 * sf, b2)
    return a2 / b2


def _log1pmx_vec(x):
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape)
    far = (x > 1) | (x < -0.79149064)
    out[far] = np.log1p(x[far]) - x[far]
    near = ~far
    if near.any():
        xr = x[near]
        r = xr / (2 + xr)
        y = r * r
        res = np.empty(xr.shape)
        small = np.abs(xr) < 1e-2
        if small.any():
            two = 2.0
            ys, xs, rs = y[small], xr[small], r[small]
            res[small] = rs * ((((two / 9 * ys + two / 7) * ys + two / 5) * ys
                                + two / 3) * ys - xs)
        big = ~small
        if big.any():
            res[big] = r[big] * (2 * y[big] * _logcf_vec(y[big], 3, 2, 1e-14) - xr[big])
        out[near] = res
    return out


def _lgamma1p_vec(a):
    a = np.asarray(a, dtype=float)
    out = np.empty(a.shape)
    big = np.abs(a) >= 0.5
    out[big] = _lgammafn_arr(a[big] + 1)
    sm = ~big
    if sm.any():
        am = a[sm]
        eulers = 0.5772156649015328606065120900824024
        c = 0.2273736845824652515226821577978691e-12
        lgam = c * _logcf_vec(-am / 2, 42, 1, 1e-14)
        for i in range(39, -1, -1):
            lgam = _LGAMMA1P_COEFFS[i] - am * lgam
        out[sm] = (am * lgam - eulers) * am - _log1pmx_vec(am)
    return out


def _dpois_wrap_vec(x_plus_1, lam, give_log):
    x_plus_1, lam = np.broadcast_arrays(np.asarray(x_plus_1, float), np.asarray(lam, float))
    out = np.empty(x_plus_1.shape)
    notfin = ~np.isfinite(lam)
    out[notfin] = _NEGINF if give_log else 0.0
    m = ~notfin
    big_x = m & (x_plus_1 > 1)
    if big_x.any():
        out[big_x] = _dpois_raw(x_plus_1[big_x] - 1, lam[big_x], give_log)
    rest = m & ~big_x
    cut = rest & (lam > np.abs(x_plus_1 - 1) * _M_CUTOFF)
    if cut.any():
        v = -lam[cut] - _lgammafn_arr(x_plus_1[cut])
        out[cut] = v if give_log else np.exp(v)
    last = rest & ~cut
    if last.any():
        dd = _dpois_raw(x_plus_1[last], lam[last], give_log)
        xl = x_plus_1[last] / lam[last]
        out[last] = (dd + np.log(xl)) if give_log else dd * xl
    return out


def _pgamma_smallx_vec(x, alph, lower_tail, log_p):
    x, alph = np.broadcast_arrays(np.asarray(x, float), np.asarray(alph, float))
    sm = np.zeros(x.shape)
    c = np.array(alph, dtype=float)
    n = np.zeros(x.shape)
    active = np.ones(x.shape, dtype=bool)
    for _ in range(100000):
        if not active.any():
            break
        n = np.where(active, n + 1, n)
        c = np.where(active, c * (-x / n), c)
        term = c / (alph + n)
        sm = np.where(active, sm + term, sm)
        active = active & (np.abs(term) > _DBL_EPSILON * np.abs(sm))
    if lower_tail:
        f1 = np.log1p(sm) if log_p else 1 + sm
        f2 = np.empty(x.shape)
        a_gt1 = alph > 1
        if a_gt1.any():
            t = _dpois_raw(alph[a_gt1], x[a_gt1], log_p)
            f2[a_gt1] = (t + x[a_gt1]) if log_p else (t * np.exp(x[a_gt1]))
        rest = ~a_gt1
        if rest.any():
            if log_p:
                f2[rest] = alph[rest] * np.log(x[rest]) - _lgamma1p_vec(alph[rest])
            else:
                f2[rest] = np.power(x[rest], alph[rest]) / np.exp(_lgamma1p_vec(alph[rest]))
        return (f1 + f2) if log_p else (f1 * f2)
    lf2 = alph * np.log(x) - _lgamma1p_vec(alph)
    if log_p:
        return _R_Log1_Exp_vec(np.log1p(sm) + lf2)
    f1m1 = sm
    f2m1 = np.expm1(lf2)
    return -(f1m1 + f2m1 + f1m1 * f2m1)


def _pd_upper_series_vec(x, y, log_p):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    y = y.copy()
    term = x / y
    sm = term.copy()
    active = np.ones(x.shape, dtype=bool)
    for _ in range(100000):
        if not active.any():
            break
        y = np.where(active, y + 1, y)
        term = np.where(active, term * (x / y), term)
        sm = np.where(active, sm + term, sm)
        active = active & (term > sm * _DBL_EPSILON)
    return np.log(sm) if log_p else sm


def _pd_lower_cf_vec(y, d):
    y, d = np.broadcast_arrays(np.asarray(y, float), np.asarray(d, float))
    shp = y.shape
    sf = _PG_SCALEFACTOR
    out = np.empty(shp)
    zero = y == 0.0
    out[zero] = 0.0
    f0 = y / d
    early = (~zero) & (np.abs(y - 1) < np.abs(d) * _DBL_EPSILON)
    out[early] = f0[early]
    act = (~zero) & (~early)
    f0 = np.where(f0 > 1.0, 1.0, f0)
    c2 = y.copy()
    c4 = d.copy()
    a1 = np.zeros(shp)
    b1 = np.ones(shp)
    a2 = y.copy()
    b2 = d.copy()
    for _ in range(100000):
        resc = act & (b2 > sf)
        if not resc.any():
            break
        a1 = np.where(resc, a1 / sf, a1)
        b1 = np.where(resc, b1 / sf, b1)
        a2 = np.where(resc, a2 / sf, a2)
        b2 = np.where(resc, b2 / sf, b2)
    of = np.full(shp, -1.0)
    f = np.zeros(shp)
    converged = np.zeros(shp, dtype=bool)
    i = 0.0
    for _ in range(100001):
        run = act & ~converged
        if not run.any():
            break
        i += 1.0
        c2 = np.where(run, c2 - 1, c2)
        c3 = i * c2
        c4 = np.where(run, c4 + 2, c4)
        a1 = np.where(run, c4 * a2 + c3 * a1, a1)
        b1 = np.where(run, c4 * b2 + c3 * b1, b1)
        i += 1.0
        c2 = np.where(run, c2 - 1, c2)
        c3 = i * c2
        c4 = np.where(run, c4 + 2, c4)
        a2 = np.where(run, c4 * a1 + c3 * a2, a2)
        b2 = np.where(run, c4 * b1 + c3 * b2, b2)
        big = run & (b2 > sf)
        if big.any():
            a1 = np.where(big, a1 / sf, a1)
            b1 = np.where(big, b1 / sf, b1)
            a2 = np.where(big, a2 / sf, a2)
            b2 = np.where(big, b2 / sf, b2)
        nz = run & (b2 != 0.0)
        fnew = np.where(nz, a2 / np.where(b2 == 0.0, 1.0, b2), f)
        conv = nz & (np.abs(fnew - of) <= _DBL_EPSILON * np.maximum(f0, np.abs(fnew)))
        f = np.where(nz, fnew, f)
        of = np.where(nz, fnew, of)
        converged = converged | conv
    out[act] = f[act]
    return out


def _pd_lower_series_vec(lam, y):
    lam, y = np.broadcast_arrays(np.asarray(lam, float), np.asarray(y, float))
    y = y.copy()
    term = np.ones(y.shape)
    sm = np.zeros(y.shape)
    active = (y >= 1) & (term > sm * _DBL_EPSILON)
    for _ in range(100000):
        if not active.any():
            break
        term = np.where(active, term * (y / lam), term)
        sm = np.where(active, sm + term, sm)
        y = np.where(active, y - 1, y)
        active = (y >= 1) & (term > sm * _DBL_EPSILON)
    nf = y != np.floor(y)
    if nf.any():
        f = _pd_lower_cf_vec(y[nf], lam[nf] + 1 - y[nf])
        sm[nf] = sm[nf] + term[nf] * f
    return sm


def _dpnorm_vec(x, lower_tail, lp):
    x, lp = np.broadcast_arrays(np.asarray(x, float), np.asarray(lp, float))
    x = x.copy()
    lt = np.broadcast_to(lower_tail, x.shape).copy()
    neg = x < 0
    x = np.where(neg, -x, x)
    lt = np.where(neg, ~lt, lt)
    out = np.empty(x.shape)
    series = (x > 10) & (~lt)
    if series.any():
        xs = x[series]
        x2 = xs * xs
        term = 1 / xs
        sm = term.copy()
        i = 1.0
        active = np.ones(xs.shape, dtype=bool)
        for _ in range(100000):
            term = np.where(active, term * (-i / x2), term)
            sm = np.where(active, sm + term, sm)
            i += 2.0
            active = active & (np.abs(term) > _DBL_EPSILON * sm)
            if not active.any():
                break
        out[series] = 1 / sm
    rest = ~series
    if rest.any():
        dd = dnorm5_vec(x[rest], 0.0, 1.0, False)
        out[rest] = dd / np.exp(lp[rest])
    return out


def _ppois_asymp_vec(x, lam, lower_tail, log_p):
    x, lam = np.broadcast_arrays(np.asarray(x, float), np.asarray(lam, float))
    dfm = lam - x
    pt_ = -_log1pmx_vec(dfm / x)
    s2pt = np.sqrt(2 * x * pt_)
    s2pt = np.where(dfm < 0, -s2pt, s2pt)
    res12 = np.zeros(x.shape)
    res1_ig = np.sqrt(x)
    res1_term = res1_ig.copy()
    res2_ig = s2pt.copy()
    res2_term = s2pt.copy()
    for i in range(1, 8):
        res12 = res12 + res1_ig * _PPA_COEFS_A[i]
        res12 = res12 + res2_ig * _PPA_COEFS_B[i]
        res1_term = res1_term * (pt_ / i)
        res2_term = res2_term * (2 * pt_ / (2 * i + 1))
        res1_ig = res1_ig / x + res1_term
        res2_ig = res2_ig / x + res2_term
    elfb = x.copy()
    elfb_term = np.ones(x.shape)
    for i in range(1, 8):
        elfb = elfb + elfb_term * _PPA_COEFS_B[i]
        elfb_term = elfb_term / x
    if not lower_tail:
        elfb = -elfb
    f = res12 / elfb
    np_ = pnorm5_vec(s2pt, 0.0, 1.0, not lower_tail, log_p)
    if log_p:
        n_d_over_p = _dpnorm_vec(s2pt, not lower_tail, np_)
        return np_ + np.log1p(f * n_d_over_p)
    nd = dnorm5_vec(s2pt, 0.0, 1.0, False)
    return np_ + f * nd


def pgamma_raw_vec(x, alph, lower_tail, log_p):
    x, alph = np.broadcast_arrays(np.asarray(x, float), np.asarray(alph, float))
    out = np.empty(x.shape)
    le0 = x <= 0
    out[le0] = _dt0(lower_tail, log_p)
    infm = (~le0) & (x >= _INF)
    out[infm] = _dt1(lower_tail, log_p)
    rest = (~le0) & (~infm)
    b_small = rest & (x < 1)
    b_upper = rest & (~b_small) & (x <= alph - 1) & (x < 0.8 * (alph + 50))
    b_lower = rest & (~b_small) & (~b_upper) & (alph - 1 < x) & (alph < 0.8 * (x + 50))
    b_asymp = rest & (~b_small) & (~b_upper) & (~b_lower)
    if b_small.any():
        out[b_small] = _pgamma_smallx_vec(x[b_small], alph[b_small], lower_tail, log_p)
    if b_upper.any():
        xs, al = x[b_upper], alph[b_upper]
        sm = _pd_upper_series_vec(xs, al, log_p)
        dd = _dpois_wrap_vec(al, xs, log_p)
        if not lower_tail:
            out[b_upper] = _R_Log1_Exp_vec(dd + sm) if log_p else 1 - dd * sm
        else:
            out[b_upper] = (sm + dd) if log_p else sm * dd
    if b_lower.any():
        xs, al = x[b_lower], alph[b_lower]
        dd = _dpois_wrap_vec(al, xs, log_p)
        sm = np.empty(xs.shape)
        a_lt1 = al < 1
        if a_lt1.any():
            xa, aa = xs[a_lt1], al[a_lt1]
            sub = np.empty(xa.shape)
            cond = xa * _DBL_EPSILON > 1 - aa
            if cond.any():
                sub[cond] = 0.0 if log_p else 1.0
            nc = ~cond
            if nc.any():
                fcf = _pd_lower_cf_vec(aa[nc], xa[nc] - (aa[nc] - 1)) * xa[nc] / aa[nc]
                sub[nc] = np.log(fcf) if log_p else fcf
            sm[a_lt1] = sub
        a_ge1 = ~a_lt1
        if a_ge1.any():
            s = _pd_lower_series_vec(xs[a_ge1], al[a_ge1] - 1)
            sm[a_ge1] = np.log1p(s) if log_p else 1 + s
        if not lower_tail:
            out[b_lower] = (sm + dd) if log_p else sm * dd
        else:
            out[b_lower] = _R_Log1_Exp_vec(dd + sm) if log_p else 1 - dd * sm
    if b_asymp.any():
        out[b_asymp] = _ppois_asymp_vec(alph[b_asymp] - 1, x[b_asymp], not lower_tail, log_p)
    if not log_p:
        small_res = rest & (out < _DBL_MIN / _DBL_EPSILON)
        if small_res.any():
            out[small_res] = np.exp(
                pgamma_raw_vec(x[small_res], alph[small_res], lower_tail, True))
    return out


def pgamma_vec(x, alph, scale, lower_tail=True, log_p=False):
    x, alph, scale = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(alph, float), np.asarray(scale, float))
    out = np.empty(x.shape)
    nan = np.isnan(x) | np.isnan(alph) | np.isnan(scale)
    out[nan] = (x + alph + scale)[nan]
    bad = (~nan) & ((alph < 0) | (scale <= 0))
    out[bad] = _NAN
    ok = (~nan) & (~bad)
    if ok.any():
        xs = x[ok] / scale[ok]
        al = alph[ok]
        res = np.empty(xs.shape)
        xnan = np.isnan(xs)
        res[xnan] = xs[xnan]
        a0 = (~xnan) & (al == 0)
        if a0.any():
            res[a0] = np.where(xs[a0] <= 0, _dt0(lower_tail, log_p), _dt1(lower_tail, log_p))
        main = (~xnan) & (al != 0)
        if main.any():
            res[main] = pgamma_raw_vec(xs[main], al[main], lower_tail, log_p)
        out[ok] = res
    return out


def ppois_vec(x, lam, lower_tail=True, log_p=False):
    x, lam = np.broadcast_arrays(np.asarray(x, float), np.asarray(lam, float))
    out = np.empty(x.shape)
    nan = np.isnan(x) | np.isnan(lam)
    out[nan] = (x + lam)[nan]
    bad = (~nan) & (lam < 0)
    out[bad] = _NAN
    ok = (~nan) & (~bad)
    xneg = ok & (x < 0)
    out[xneg] = _dt0(lower_tail, log_p)
    lz = ok & (~xneg) & (lam == 0)
    out[lz] = _dt1(lower_tail, log_p)
    xinf = ok & (~xneg) & (lam != 0) & (~np.isfinite(x))
    out[xinf] = _dt1(lower_tail, log_p)
    main = ok & (~xneg) & (lam != 0) & np.isfinite(x)
    if main.any():
        xf = np.floor(x[main] + 1e-7)
        out[main] = pgamma_vec(lam[main], xf + 1, 1.0, not lower_tail, log_p)
    return out


_PY_VEC["pgamma"] = pgamma_vec
_PY_VEC["ppois"] = ppois_vec


def _chebyshev_eval_vec(x, a, n):
    x = np.asarray(x, dtype=float)
    twox = x * 2
    b2 = np.zeros(x.shape)
    b1 = np.zeros(x.shape)
    b0 = np.zeros(x.shape)
    for i in range(1, n + 1):
        b2 = b1
        b1 = b0
        b0 = twox * b1 - b2 + a[n - i]
    out = (b0 - b2) * 0.5
    return np.where((x < -1.1) | (x > 1.1), np.nan, out)


def _lgammacor_vec(x):
    x = np.asarray(x, dtype=float)
    out = np.where(x < 10, np.nan, 1.0 / (x * 12))
    small = (x >= 10) & (x < _LGC_XBIG)
    if small.any():
        tmp = 10 / x[small]
        out[small] = _chebyshev_eval_vec(tmp * tmp * 2 - 1, _ALGMCS, _NALGM) / x[small]
    return out


def gammafn_vec(x):
    """Vectorised :func:`gammafn`. x <= 0 routed to the scalar kernel (not hit by
    lbeta/dbeta, which only pass positive args)."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape)
    neg = ~(x > 0)
    if neg.any():
        out[neg] = np.array([gammafn(float(v)) for v in np.atleast_1d(x[neg])]).reshape(x[neg].shape)
    pos = x > 0
    if pos.any():
        xs = x[pos]
        res = np.empty(xs.shape)
        le10 = xs <= 10
        if le10.any():
            xl = xs[le10]
            n = np.trunc(xl).astype(np.int64)
            frac = xl - n
            n = n - 1
            r = _chebyshev_eval_vec(frac * 2 - 1, _GAMCS, _NGAM) + 0.9375
            npos = n[n > 0]
            maxn = int(npos.max()) if npos.size else 0
            for i in range(1, maxn + 1):
                r = np.where(n >= i, r * (frac + i), r)
            ng = n < 0
            if ng.any():
                negn = -n
                xsml = ng & (frac < _GAM_XSML)
                minn = int(negn[ng].max()) if ng.any() else 0
                for i in range(0, minn):
                    r = np.where(ng & (negn > i), r / (xl + i), r)
                r = np.where(xsml, _INF, r)
            res[le10] = r
        gt10 = ~le10
        if gt10.any():
            xg = xs[gt10]
            v = np.where(xg > _GAM_XMAX, _INF, 0.0)
            main = xg <= _GAM_XMAX
            if main.any():
                ym = xg[main]
                vv = np.empty(ym.shape)
                intfac = (ym <= 50) & (ym == np.trunc(ym))
                if intfac.any():
                    yi = np.trunc(ym[intfac]).astype(np.int64)
                    prod = np.ones(yi.shape)
                    maxk = int(yi.max()) if yi.size else 0
                    for k in range(2, maxk):
                        prod = np.where(yi > k, prod * k, prod)
                    vv[intfac] = prod
                els = ~intfac
                if els.any():
                    ye = ym[els]
                    half = (2 * ye) == np.trunc(2 * ye)
                    corr = np.where(half, _stirlerr(ye), _lgammacor_vec(ye))
                    vv[els] = np.exp((ye - 0.5) * np.log(ye) - ye + _M_LN_SQRT_2PI + corr)
                v[main] = vv
            res[gt10] = v
        out[pos] = res
    return out


def lbeta_vec(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.empty(a.shape)
    nan = np.isnan(a) | np.isnan(b)
    out[nan] = (a + b)[nan]
    ok = ~nan
    p = np.minimum(a, b)
    q = np.maximum(a, b)
    out[ok & (p < 0)] = np.nan
    out[ok & (p == 0)] = _INF
    out[ok & (p > 0) & (~np.isfinite(q))] = _NEGINF
    m = ok & (p > 0) & np.isfinite(q)
    if m.any():
        pp, qq = p[m], q[m]
        r = np.empty(pp.shape)
        b1 = pp >= 10
        if b1.any():
            pv, qv = pp[b1], qq[b1]
            corr = _lgammacor_vec(pv) + _lgammacor_vec(qv) - _lgammacor_vec(pv + qv)
            r[b1] = (np.log(qv) * -0.5 + _M_LN_SQRT_2PI + corr
                     + (pv - 0.5) * np.log(pv / (pv + qv)) + qv * np.log1p(-pv / (pv + qv)))
        b2 = (~b1) & (qq >= 10)
        if b2.any():
            pv, qv = pp[b2], qq[b2]
            corr = _lgammacor_vec(qv) - _lgammacor_vec(pv + qv)
            r[b2] = (_lgammafn_arr(pv) + corr + pv - pv * np.log(pv + qv)
                     + (qv - 0.5) * np.log1p(-pv / (pv + qv)))
        b3 = (~b1) & (~b2) & (pp < 1e-306)
        if b3.any():
            pv, qv = pp[b3], qq[b3]
            r[b3] = _lgammafn_arr(pv) + (_lgammafn_arr(qv) - _lgammafn_arr(pv + qv))
        b4 = (~b1) & (~b2) & (pp >= 1e-306)
        if b4.any():
            pv, qv = pp[b4], qq[b4]
            r[b4] = np.log(gammafn_vec(pv) * (gammafn_vec(qv) / gammafn_vec(pv + qv)))
        out[m] = r
    return out


def dbeta_vec(x, a, b, give_log=False):
    x, a, b = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(a, float), np.asarray(b, float))
    rd0 = _NEGINF if give_log else 0.0
    out = np.full(x.shape, np.nan)
    nan = np.isnan(x) | np.isnan(a) | np.isnan(b)
    out[nan] = (x + a + b)[nan]
    ok = (~nan) & (~((a < 0) | (b < 0)))
    oob = ok & ((x < 0) | (x > 1))
    out[oob] = rd0
    ok2 = ok & (~oob)
    edge = ok2 & ((a == 0) | (b == 0) | (~np.isfinite(a)) | (~np.isfinite(b)))
    if edge.any():
        xe, ae, be = x[edge], a[edge], b[edge]
        both0 = (ae == 0) & (be == 0)
        e_a = (~both0) & ((ae == 0) | (ae / be == 0))
        e_b = (~both0) & (~e_a) & ((be == 0) | (be / ae == 0))
        e_o = (~both0) & (~e_a) & (~e_b)
        ve = np.full(xe.shape, rd0)
        ve = np.where(both0, np.where((xe == 0) | (xe == 1), _INF, rd0), ve)
        ve = np.where(e_a, np.where(xe == 0, _INF, rd0), ve)
        ve = np.where(e_b, np.where(xe == 1, _INF, rd0), ve)
        ve = np.where(e_o, np.where(xe == 0.5, _INF, rd0), ve)
        out[edge] = ve
    main = ok2 & (~edge)
    if main.any():
        xm, am, bm = x[main], a[main], b[main]
        lval = np.empty(xm.shape)
        small = (am <= 2) | (bm <= 2)
        # x in {0,1} feeds log(0) into the formula but is overwritten below by
        # the boundary values — silence the (discarded) warnings.
        with np.errstate(divide="ignore", invalid="ignore"):
            if small.any():
                lval[small] = ((am[small] - 1) * np.log(xm[small])
                               + (bm[small] - 1) * np.log1p(-xm[small])
                               - lbeta_vec(am[small], bm[small]))
            big = ~small
            if big.any():
                lval[big] = (np.log(am[big] + bm[big] - 1)
                             + _dbinom_raw(am[big] - 1, am[big] + bm[big] - 2,
                                           xm[big], 1 - xm[big], True))
            val = lval if give_log else np.exp(lval)
        x0, x1 = xm == 0, xm == 1
        val = np.where(x0 & (am > 1), rd0, val)
        val = np.where(x0 & (am < 1), _INF, val)
        val = np.where(x1 & (bm > 1), rd0, val)
        val = np.where(x1 & (bm < 1), _INF, val)
        out[main] = val
    return out


_PY_VEC["dbeta"] = dbeta_vec
