"""Bit-exact ports of R's ``nlm()`` and ``optim(method="L-BFGS-B")``.

Two layers are mirrored for each optimizer, so the Python surface
reproduces R's *semantics*, not just the core algorithm:

* ``nlm`` — the R wrapper (``src/library/stats/R/nlm.R``: msg bit
  assembly from ``print.level``/``check.analyticals``) plus the C glue
  (``src/library/stats/src/optimize.c`` ``nlm``/``fcn``/``Cd1fcn``: the
  5-entry ring cache of function values keyed on bit-exact ``x``, the
  probe evaluation that detects analytic gradients/Hessians, the
  DBL_MAX mapping of non-finite objective values, ``method=1``/
  ``iexp=!iahflg``/``dlt=1.0``) → :func:`hea.R.uncmin.optif9`.
* ``optim`` — the R wrapper (``src/library/stats/R/optim.R``: control
  defaults, fnscale/parscale) plus the C driver (``src/library/stats/
  src/optim.c``: ``fminfn``/``fmingr`` scaling, nbd classification) →
  :func:`hea.R.lbfgsb.lbfgsb`. Only ``method="L-BFGS-B"`` is ported —
  it is the only ``optim`` method mgcv's ``gam.outer`` uses.

Objective conventions (R returns attributes; Python returns tuples):
``nlm``'s ``f(x)`` may return ``float`` — or ``(value, gradient)`` /
``(value, gradient, hessian)`` to supply analytic derivatives, the
equivalent of R's ``attr(ret, "gradient")``/``attr(ret, "hessian")``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .._dispatch import rs_fn
from .lbfgsb import lbfgsb as _lbfgsb_driver
from .uncmin import fdhess as _fdhess
from .uncmin import optif9 as _optif9

_rs_optif9 = rs_fn("optif9")
_rs_lbfgsb = rs_fn("lbfgsb_drive")
_rs_fdhess = rs_fn("uncmin_fdhess")

_DBL_MAX = float(np.finfo(np.float64).max)
_FT_SIZE = 5  # optimize.c: default size of the function-value table


def _fixparam(p, n=None, what="parameter"):
    """optimize.c ``fixparam`` (:594): coerce to double, reject NA/
    non-finite entries and length mismatches."""
    x = np.asarray(p, dtype=float).ravel().copy()
    if n is not None and x.size != n:
        raise ValueError("conflicting parameter lengths")
    if x.size <= 0:
        raise ValueError("invalid parameter length")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"missing value in {what}")
    return x


def _opterror(nerr):
    """optimize.c ``opterror`` (:637): fatal nlm input errors."""
    msgs = {
        -1: "non-positive number of parameters in nlm",
        -2: "nlm is inefficient for 1-d problems",
        -3: "invalid gradient tolerance in nlm",
        -4: "invalid iteration limit in nlm",
        -5: "minimization function has no good digits in nlm",
        -6: "no analytic gradient to check in nlm!",
        -7: "no analytic Hessian to check in nlm!",
        -21: "probable coding error in analytic gradient",
        -22: "probable coding error in analytic Hessian",
    }
    raise RuntimeError(
        msgs.get(
            nerr,
            f"*** unknown error message (msg = {nerr}) in nlm()"
            "\n*** should not happen!",
        )
    )


_OPTCODE_MSG = {
    1: "Relative gradient close to zero.\nCurrent iterate is probably solution.\n",
    2: "Successive iterates within tolerance.\nCurrent iterate is probably solution.\n",
    3: "Last global step failed to locate a point lower than x.\n"
    "Either x is an approximate local minimum of the function,\n"
    "the function is too non-linear for this algorithm,\n"
    "or steptol is too large.\n",
    4: "Iteration limit exceeded.  Algorithm failed.\n",
    5: "Maximum step size exceeded 5 consecutive times.\n"
    "Either the function is unbounded below,\n"
    "becomes asymptotic to a finite value\n"
    "from above in some direction,\n"
    "or stepmx is too small.\n",
}


class _FunctionInfo:
    """optimize.c ``function_info`` + ``FT_*`` (:394-496): the 5-entry
    ring cache of (x, f, gradient, hessian) evaluations, newest-first
    lookup with bit-exact x comparison."""

    def __init__(self, f, n, have_gradient, have_hessian):
        self.f = f
        self.n = n
        self.have_gradient = have_gradient
        self.have_hessian = have_hessian
        self.table = []  # list of (x, fval, grad, hess)
        self.last = -1

    def lookup(self, x):
        for i in range(_FT_SIZE):
            ind = (self.last - i) % _FT_SIZE
            if ind < len(self.table):
                ftx = self.table[ind][0]
                if all(float(x[j]) == float(ftx[j]) for j in range(self.n)):
                    return ind
        return -1

    def store(self, fval, x, grad, hess):
        ind = (self.last + 1) % _FT_SIZE
        entry = (
            np.array(x, dtype=float, copy=True),
            float(fval),
            None if grad is None else np.array(grad, dtype=float, copy=True),
            None if hess is None else np.array(hess, dtype=float, copy=True),
        )
        if ind < len(self.table):
            self.table[ind] = entry
        else:
            self.table.append(entry)
        self.last += 1

    def _parse(self, res):
        """Split an objective return into (value, gradient, hessian) —
        the Python equivalent of reading R's gradient/hessian
        attributes."""
        grad = hess = None
        if isinstance(res, (tuple, list)):
            val = res[0]
            if len(res) > 1:
                grad = res[1]
            if len(res) > 2:
                hess = res[2]
        else:
            val = res
        return float(val), grad, hess

    def fcn(self, x):
        """optimize.c ``fcn`` (:500): cached objective with R's
        non-finite value mapping."""
        ind = self.lookup(x)
        if ind >= 0:
            return self.table[ind][1]
        for j in range(self.n):
            if not np.isfinite(float(x[j])):
                raise ValueError("non-finite value supplied by 'nlm'")
        val, grad, hess = self._parse(self.f(x))
        if not np.isfinite(val):
            if val == float("-inf"):
                warnings.warn("-Inf replaced by maximally negative value")
                val = -_DBL_MAX
            else:
                what = "NA/NaN" if math.isnan(val) else "Inf"
                warnings.warn(f"{what} replaced by maximum positive value")
                val = _DBL_MAX
        self.store(
            val,
            x,
            grad if self.have_gradient else None,
            hess if self.have_hessian else None,
        )
        return val

    def d1fcn(self, x):
        """optimize.c ``Cd1fcn`` (:562): gradient from the cache."""
        ind = self.lookup(x)
        if ind < 0:
            self.fcn(x)
            ind = self.lookup(x)
            if ind < 0:
                raise RuntimeError(
                    "function value caching for optimization is seriously confused"
                )
        return np.array(self.table[ind][2], dtype=float, copy=True)

    def d2fcn(self, x, a):
        """optimize.c ``Cd2fcn`` (:577): Hessian lower triangle from
        the cache."""
        ind = self.lookup(x)
        if ind < 0:
            self.fcn(x)
            ind = self.lookup(x)
            if ind < 0:
                raise RuntimeError(
                    "function value caching for optimization is seriously confused"
                )
        hess = np.asarray(self.table[ind][3], dtype=float).reshape(self.n, self.n)
        for j in range(self.n):
            a[j:, j] = hess[j:, j]


def nlm(
    f,
    p,
    hessian=False,
    typsize=None,
    fscale=1.0,
    print_level=0,
    ndigit=12,
    gradtol=1e-6,
    stepmax=None,
    steptol=1e-6,
    iterlim=100,
    check_analyticals=True,
):
    """R's ``nlm()`` (nlm.R:19 + optimize.c ``nlm``, :699): Dennis-
    Schnabel UNCMIN minimization with the exact R semantics.

    ``f(x)`` returns a float, or ``(value, gradient)`` /
    ``(value, gradient, hessian)`` for analytic derivatives (R's
    ``attr(, "gradient")``/``attr(, "hessian")``). Returns a dict with
    ``minimum``, ``estimate``, ``gradient``, ``code``, ``iterations``
    (+ ``hessian`` when requested)."""
    print_level = int(print_level)
    if print_level < 0 or print_level > 2:
        raise ValueError("'print.level' must be in {0,1,2}")
    # msg is a bit pattern: 1 + (8, 0, 16)[print.level], +6 to skip the
    # analytic-derivative checks (nlm.R:29-30)
    msg = 1 + (8, 0, 16)[print_level]
    if not check_analyticals:
        msg += 2 + 4
    x = _fixparam(p)
    n = x.size
    if typsize is None:
        typsize = np.ones(n)
    typsiz = _fixparam(typsize, n)
    fscale = float(fscale)
    if math.isnan(fscale):
        raise ValueError("invalid NA value in parameter")
    if stepmax is None:
        stepmax = max(1000.0 * math.sqrt(float(np.sum((x / typsiz) ** 2))), 1000.0)
    omsg = msg
    want_hessian = bool(hessian)

    iagflg = 0
    iahflg = 0
    have_gradient = 0
    have_hessian = 0
    probe = f(x.copy())
    grad = hess = None
    if isinstance(probe, (tuple, list)):
        if len(probe) > 1:
            grad = probe[1]
        if len(probe) > 2:
            hess = probe[2]
    if grad is not None:
        garr = np.asarray(grad, dtype=float).ravel()
        if garr.size == n:
            iagflg = 1
            have_gradient = 1
            if hess is not None:
                harr = np.asarray(hess, dtype=float)
                if harr.size == n * n:
                    iahflg = 1
                    have_hessian = 1
                else:
                    warnings.warn(
                        "hessian supplied is of the wrong length or mode, so ignored"
                    )
        else:
            warnings.warn(
                "gradient supplied is of the wrong length or mode, so ignored"
            )
    if ((msg // 4) % 2) and not iahflg:  # skip check of analytic Hess.
        msg -= 4
    if ((msg // 2) % 2) and not iagflg:  # skip check of analytic grad.
        msg -= 2
    state = _FunctionInfo(f, n, have_gradient, have_hessian)

    method = 1  # line search
    iexp = 0 if iahflg else 1  # function calls are expensive
    dlt = 1.0
    if _rs_optif9 is not None:

        def _d2_flat(xv):
            a = np.zeros((n, n))
            state.d2fcn(xv, a)
            return a.ravel(order="F")

        xpls, fpls, gpls, code, itncnt, msg = _rs_optif9(
            x,
            state.fcn,
            state.d1fcn,
            _d2_flat if have_hessian else None,
            typsiz,
            fscale,
            method,
            iexp,
            msg,
            int(ndigit),
            int(iterlim),
            iagflg,
            iahflg,
            dlt,
            float(gradtol),
            float(stepmax),
            float(steptol),
        )
    else:
        xpls, fpls, gpls, code, itncnt, msg = _optif9(
            n,
            x,
            state.fcn,
            state.d1fcn,
            state.d2fcn,
            typsiz,
            fscale,
            method,
            iexp,
            msg,
            int(ndigit),
            int(iterlim),
            iagflg,
            iahflg,
            dlt,
            float(gradtol),
            float(stepmax),
            float(steptol),
        )
    if msg < 0:
        _opterror(msg)
    if code != 0 and (omsg & 8) == 0:
        print(_OPTCODE_MSG.get(code, ""))
    value = {
        "minimum": float(fpls),
        "estimate": np.asarray(xpls, dtype=float),
        "gradient": np.asarray(gpls, dtype=float),
        "code": int(code),
        "iterations": int(itncnt),
    }
    if want_hessian:
        if _rs_fdhess is not None:
            a = (
                np.asarray(
                    _rs_fdhess(
                        np.asarray(xpls, dtype=float),
                        float(fpls),
                        state.fcn,
                        int(ndigit),
                        typsiz,
                    )
                )
                .reshape(n, n, order="F")
                .copy()
            )
        else:
            a = np.zeros((n, n))
            _fdhess(n, xpls, float(fpls), state.fcn, a, n, int(ndigit), typsiz)
        for i in range(n):
            for j in range(i):
                a[i, j] = a[j, i]
        value["hessian"] = a
    return value


def _optim_fminfn(p, fn, fnscale, parscale, n):
    """stats optim.c ``fminfn`` (:65): scaled objective."""
    for i in range(n):
        if not np.isfinite(float(p[i])):
            raise ValueError("non-finite value supplied by optim")
    x = np.array([float(p[i]) * float(parscale[i]) for i in range(n)])
    val = float(fn(x))
    return val / fnscale


def _optim_fmingr(p, df, fn, gr, fnscale, parscale, ndeps, n, usebounds, lower, upper):
    """stats optim.c ``fmingr`` (:90): scaled analytic gradient, or the
    (possibly bound-clipped) central finite difference."""
    if gr is not None:
        for i in range(n):
            if not np.isfinite(float(p[i])):
                raise ValueError("non-finite value supplied by optim")
        x = np.array([float(p[i]) * float(parscale[i]) for i in range(n)])
        s = np.asarray(gr(x), dtype=float).ravel()
        if s.size != n:
            raise ValueError(f"gradient in optim evaluated to length {s.size} not {n}")
        for i in range(n):
            df[i] = float(s[i]) * float(parscale[i]) / fnscale
        return
    x = np.array([float(p[i]) * float(parscale[i]) for i in range(n)])
    if not usebounds:
        for i in range(n):
            eps = float(ndeps[i])
            x[i] = (float(p[i]) + eps) * float(parscale[i])
            val1 = float(fn(x.copy())) / fnscale
            x[i] = (float(p[i]) - eps) * float(parscale[i])
            val2 = float(fn(x.copy())) / fnscale
            df[i] = (val1 - val2) / (2 * eps)
            if not np.isfinite(float(df[i])):
                raise ValueError(f"non-finite finite-difference value [{i + 1}]")
            x[i] = float(p[i]) * float(parscale[i])
    else:
        for i in range(n):
            epsused = eps = float(ndeps[i])
            tmp = float(p[i]) + eps
            if tmp > float(upper[i]):
                tmp = float(upper[i])
                epsused = tmp - float(p[i])
            x[i] = tmp * float(parscale[i])
            val1 = float(fn(x.copy())) / fnscale
            tmp = float(p[i]) - eps
            if tmp < float(lower[i]):
                tmp = float(lower[i])
                eps = float(p[i]) - tmp
            x[i] = tmp * float(parscale[i])
            val2 = float(fn(x.copy())) / fnscale
            df[i] = (val1 - val2) / (epsused + eps)
            if not np.isfinite(float(df[i])):
                raise ValueError(f"non-finite finite-difference value [{i + 1}]")
            x[i] = float(p[i]) * float(parscale[i])


def optim(
    par,
    fn,
    gr=None,
    method="L-BFGS-B",
    lower=-math.inf,
    upper=math.inf,
    control=None,
    hessian=False,
):
    """R's ``optim()`` (optim.R:19 + stats optim.c:199), ported for
    ``method="L-BFGS-B"`` only — the method mgcv's ``gam.outer`` uses.

    Returns a dict with ``par``, ``value``, ``counts`` (function,
    gradient), ``convergence`` (0 ok, 1 maxit, 51 warning, 52 error)
    and ``message``."""
    if method != "L-BFGS-B":
        raise NotImplementedError(
            f"optim method {method!r} is not ported (only 'L-BFGS-B', "
            "the method used by mgcv's gam.outer)"
        )
    par = np.asarray(par, dtype=float).ravel().copy()
    npar = par.size
    # defaults (optim.R:35-42)
    con = {
        "trace": 0,
        "fnscale": 1.0,
        "parscale": np.ones(npar),
        "ndeps": np.full(npar, 1e-3),
        "maxit": 100,
        "abstol": -math.inf,
        "reltol": math.sqrt(float(np.finfo(np.float64).eps)),
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 2.0,
        "REPORT": 10,
        "warn.1d.NelderMead": True,
        "type": 1,
        "lmm": 5,
        "factr": 1e7,
        "pgtol": 0.0,
        "tmax": 10,
        "temp": 10.0,
    }
    control = dict(control or {})
    unknown = [k for k in control if k not in con]
    if unknown:
        warnings.warn("unknown names in control: " + ", ".join(unknown))
    con.update((k, v) for k, v in control.items() if k in con)
    if con["trace"] < 0:
        warnings.warn("read the documentation for 'trace' more carefully")
    if any(k in control for k in ("reltol", "abstol")):
        warnings.warn(
            "method L-BFGS-B uses 'factr' (and 'pgtol') "
            "instead of 'reltol' and 'abstol'"
        )
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (npar,)).astype(float)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (npar,)).astype(float)
    trace = int(con["trace"])
    fnscale = float(con["fnscale"])
    parscale = np.asarray(con["parscale"], dtype=float).ravel()
    if parscale.size != npar:
        raise ValueError("'parscale' is of the wrong length")
    ndeps = np.asarray(con["ndeps"], dtype=float).ravel()
    if gr is None and ndeps.size != npar:
        raise ValueError("'ndeps' is of the wrong length")
    maxit = int(con["maxit"])
    nREPORT = int(con["REPORT"])
    factr = float(con["factr"])
    pgtol = float(con["pgtol"])
    lmm = int(con["lmm"])
    dpar = np.array([float(par[i]) / float(parscale[i]) for i in range(npar)])
    lo = np.zeros(npar)
    up = np.zeros(npar)
    nbd = np.zeros(npar, dtype=np.int64)
    for i in range(npar):
        lo[i] = float(lower[i]) / float(parscale[i])
        up[i] = float(upper[i]) / float(parscale[i])
        if not np.isfinite(lo[i]):
            nbd[i] = 0 if not np.isfinite(up[i]) else 3
        else:
            nbd[i] = 1 if not np.isfinite(up[i]) else 2

    def fminfn(p):
        return _optim_fminfn(p, fn, fnscale, parscale, npar)

    def fmingr(p, df):
        _optim_fmingr(p, df, fn, gr, fnscale, parscale, ndeps, npar, True, lo, up)

    if _rs_lbfgsb is not None and trace == 0:

        def _gr_ret(p):
            df = np.zeros(npar)
            fmingr(p, df)
            return df

        dpar, val, fail, fncount, grcount, msg = _rs_lbfgsb(
            npar, lmm, dpar, lo, up, nbd, fminfn, _gr_ret, factr, pgtol, maxit
        )
    else:
        val, fail, fncount, grcount, msg = _lbfgsb_driver(
            npar,
            lmm,
            dpar,
            lo,
            up,
            nbd,
            fminfn,
            fmingr,
            factr,
            pgtol,
            maxit,
            trace,
            nREPORT,
        )
    out_par = np.array([float(dpar[i]) * float(parscale[i]) for i in range(npar)])
    res = {
        "par": out_par,
        "value": val * fnscale,
        "counts": {"function": fncount, "gradient": grcount},
        "convergence": int(fail),
        "message": msg,
    }
    if hessian:
        res["hessian"] = optimHess(out_par, fn, gr, control=con)
    return res


def optimHess(par, fn, gr=None, control=None):
    """R's ``optimHess`` (optim.R:80 + stats optim.c ``optimhess``,
    :408): central-difference Hessian of ``fn`` (through the gradient
    when analytic), with fnscale/parscale handling and
    symmetrization."""
    par = np.asarray(par, dtype=float).ravel()
    npar = par.size
    con = {"fnscale": 1.0, "parscale": np.ones(npar), "ndeps": np.full(npar, 1e-3)}
    con.update({k: v for k, v in dict(control or {}).items() if k in con})
    fnscale = float(con["fnscale"])
    parscale = np.asarray(con["parscale"], dtype=float).ravel()
    if parscale.size != npar:
        raise ValueError("'parscale' is of the wrong length")
    ndeps = np.asarray(con["ndeps"], dtype=float).ravel()
    if ndeps.size != npar:
        raise ValueError("'ndeps' is of the wrong length")
    ans = np.zeros((npar, npar))
    dpar = np.array([float(par[i]) / float(parscale[i]) for i in range(npar)])
    df1 = np.zeros(npar)
    df2 = np.zeros(npar)

    def fmingr(p, df):
        _optim_fmingr(p, df, fn, gr, fnscale, parscale, ndeps, npar, False, None, None)

    for i in range(npar):
        eps = float(ndeps[i]) / float(parscale[i])
        dpar[i] = float(dpar[i]) + eps
        fmingr(dpar, df1)
        dpar[i] = float(dpar[i]) - 2 * eps
        fmingr(dpar, df2)
        for j in range(npar):
            ans[j, i] = (
                fnscale
                * (float(df1[j]) - float(df2[j]))
                / (2 * eps * float(parscale[i]) * float(parscale[j]))
            )
        dpar[i] = float(dpar[i]) + eps
    for i in range(npar):
        for j in range(i):
            tmp = 0.5 * (float(ans[j, i]) + float(ans[i, j]))
            ans[j, i] = ans[i, j] = tmp
    return ans
