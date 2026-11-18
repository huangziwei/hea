"""Bit-exact port of R's ``src/appl/uncmin.c`` (R 4.6.0) — the Dennis +
Schnabel UNCMIN minimizer behind R's ``nlm()``.

The C file is itself an f2c translation of Schnabel/Koontz/Weiss's UNCMIN
Fortran, hand-edited by Saikat DebRoy; this module is a line-by-line
translation of that C into Python, keeping every accumulation order, skip
tolerance and control-flow quirk (including the ``goto L103/L105``
central-difference retry in ``optdrv`` and the ``itrmcd == 3`` reset in
``optdrv_end``). BLAS/LINPACK calls go through :mod:`hea.R._linpack`,
which reproduces the BLAS R actually links (Accelerate on the CRAN macOS
build — see that module's docstring for the exact-emulation boundary).

R's build compiles this C with clang, whose default ``-ffp-contract=on``
fuses ``a ± b*c`` within an expression into a single-rounding FMA on
arm64. Every such contraction on the path R's ``nlm`` can reach
(``method=1``; ``iexp=1`` unless an analytic Hessian is supplied) is
mirrored here via ``_rfma`` — verified against R's compiled ``optif9``
(the eval-by-eval trajectory probe that exposed ``x + λ*p`` as fused).
The trust-region drivers (``dogdrv``/``hookdrv``/``tregup``/``chlhsn``,
methods 2-3) are unreachable from R's ``nlm`` entry point and are kept
in plain reference order.

Callable interfaces (the C threads ``state`` through ``fcn_p`` pointers;
Python closures carry it):

* ``fcn(x) -> float`` — objective. ``x`` is the live numpy array; the
  finite-difference helpers temporarily perturb entries and restore them,
  exactly like the C mutates and restores ``x``/``xpls``.
* ``d1fcn(x) -> ndarray(n)`` — analytic gradient.
* ``d2fcn(x, a) -> None`` — analytic Hessian filled into the lower
  triangle of ``a`` (only reached when ``iahflg == 1``).

Vectors are 1-D float64 arrays; ``a`` is an ``(n, n)`` array indexed
``[row, col]`` like the C's ``a[i + j*nr]`` (``nr == n`` always — that is
how ``optif9`` is called from ``do_nlm``).

Entry point: :func:`optif9` (plus :func:`fdhess` for ``nlm(hessian=
TRUE)``); R-side semantics (function-value cache, msg bit handling,
DBL_MAX mapping) live in :mod:`hea.R.optimize`.
"""

from __future__ import annotations

import math

import numpy as np

from ._linpack import _ddot, _dnrm2, _dscal, _dtrsl
from ._shared import _rfma

_DBL_EPSILON = float(np.finfo(np.float64).eps)
_DBL_MAX = float(np.finfo(np.float64).max)


def _fmax2(a, b):
    return max(b, a)


def _fmin2(a, b):
    return min(b, a)


def fdhess(n, x, fval, fun, h, nfd, ndigit, typx):
    """uncmin.c ``fdhess`` (:50): forward-difference approximation to the
    upper triangle of the Hessian (Dennis & Schnabel A5.6.2), used by
    ``nlm(hessian=TRUE)``. Mutates ``x`` transiently; fills ``h``."""
    step = np.zeros(n)
    f = np.zeros(n)
    eta = 10.0 ** (-ndigit / 3.0)
    for i in range(n):
        step[i] = eta * _fmax2(float(x[i]), float(typx[i]))
        if float(typx[i]) < 0.0:
            step[i] = -step[i]
        tempi = float(x[i])
        x[i] = tempi + float(step[i])
        step[i] = float(x[i]) - tempi
        f[i] = fun(x)
        x[i] = tempi
    for i in range(n):
        tempi = float(x[i])
        x[i] = tempi + float(step[i]) * 2.0
        fii = fun(x)
        h[i, i] = ((fval - float(f[i])) + (fii - float(f[i]))) / (
            float(step[i]) * float(step[i])
        )
        x[i] = tempi + float(step[i])
        for j in range(i + 1, n):
            tempj = float(x[j])
            x[j] = tempj + float(step[j])
            fij = fun(x)
            h[i, j] = ((fval - float(f[i])) + (fij - float(f[j]))) / (
                float(step[i]) * float(step[j])
            )
            x[j] = tempj
        x[i] = tempi


def _mvmltl(n, a, x, y):
    """uncmin.c ``mvmltl`` (:132): y = L x, L lower triangular in a
    (clang-contracted accumulation)."""
    for i in range(n):
        s = 0.0
        for j in range(i + 1):
            s = _rfma(float(a[i, j]), float(x[j]), s)
        y[i] = s


def _mvmltu(n, a, x, y):
    """uncmin.c ``mvmltu`` (:160): y = L' x — an F77 ``ddot`` down each
    column from the diagonal (routed through the R-linked BLAS
    emulation)."""
    for i in range(n):
        y[i] = _ddot(n - i, a[:, i], x, ox=i, oy=i)


def _mvmlts(n, a, x, y):
    """uncmin.c ``mvmlts`` (:184): y = A x for symmetric A stored in the
    lower triangle (clang-contracted accumulation)."""
    for i in range(n):
        s = 0.0
        for j in range(i + 1):
            s = _rfma(float(a[i, j]), float(x[j]), s)
        for j in range(i + 1, n):
            s = _rfma(float(a[j, i]), float(x[j]), s)
        y[i] = s


def _lltslv(n, a, x, b):
    """uncmin.c ``lltslv`` (:215): solve LL'x = b via two dtrsl sweeps
    (jobs 0 then 10). ``x``/``b`` may be the same array."""
    if x is not b:
        x[:n] = b[:n]
    _dtrsl(a, n, x, 0)
    _dtrsl(a, n, x, 10)


def _choldc(n, a, diagmx, tol):
    """uncmin.c ``choldc`` (:242): perturbed Cholesky of a+D; returns
    addmax, mutates a (L in lower triangle + diagonal)."""
    addmax = 0.0
    aminl = math.sqrt(diagmx * tol)
    amnlsq = aminl * aminl
    for i in range(n):
        for j in range(i):
            s = 0.0
            for k in range(j):
                s = _rfma(float(a[i, k]), float(a[j, k]), s)
            a[i, j] = (float(a[i, j]) - s) / float(a[j, j])
        s = 0.0
        for k in range(i):
            s = _rfma(float(a[i, k]), float(a[i, k]), s)
        tmp1 = float(a[i, i]) - s
        if tmp1 >= amnlsq:
            a[i, i] = math.sqrt(tmp1)
        else:
            offmax = 0.0
            for j in range(i):
                tmp2 = abs(float(a[i, j]))
                offmax = max(offmax, tmp2)
            offmax = max(amnlsq, offmax)
            a[i, i] = math.sqrt(offmax)
            tmp2 = offmax - tmp1
            addmax = max(addmax, tmp2)
    return addmax


def _qraux1(n, r, i):
    """uncmin.c ``qraux1`` (:323): swap rows i, i+1 of r over columns
    i..n-1 (0-based)."""
    for j in range(i, n):
        r[i, j], r[i + 1, j] = float(r[i + 1, j]), float(r[i, j])


def _qraux2(n, r, i, a, b):
    """uncmin.c ``qraux2`` (:347): premultiply r by the Jacobi rotation
    j(i, i+1, a, b) over columns i..n-1. ``hypot`` goes through numpy
    (= the platform libm R's C calls; CPython's ``math.hypot`` is its
    own algorithm and differs in the last ulp). The rotation's two-
    product expressions are clang-contracted on the *first* product —
    verified against R's compiled optif9 trajectories."""
    den = float(np.hypot(a, b))
    c = a / den
    s = b / den
    for j in range(i, n):
        y = float(r[i, j])
        z = float(r[i + 1, j])
        r[i, j] = _rfma(c, y, -(s * z))
        r[i + 1, j] = _rfma(s, y, c * z)


def _qrupdt(n, a, u, v):
    """uncmin.c ``qrupdt`` (:382): rank-1 QR update — find (Q*)(R*) =
    R + u v'. Mutates a and u."""
    k = n - 1
    while k > 0 and float(u[k]) == 0.0:
        k -= 1
    ii = k
    while ii > 0:
        i = ii - 1
        if float(u[i]) == 0.0:
            _qraux1(n, a, i)
            u[i] = float(u[ii])
        else:
            _qraux2(n, a, i, float(u[i]), -float(u[ii]))
            u[i] = float(np.hypot(float(u[i]), float(u[ii])))
        ii = i
    for j in range(n):
        a[0, j] = _rfma(float(u[0]), float(v[j]), float(a[0, j]))
    for i in range(k):
        if float(a[i, i]) == 0.0:
            _qraux1(n, a, i)
        else:
            t1 = float(a[i, i])
            t2 = -float(a[i + 1, i])
            _qraux2(n, a, i, t1, t2)


def _tregup(
    n,
    x,
    f,
    g,
    a,
    fcn,
    sc,
    sx,
    nwtake,
    stepmx,
    steptl,
    dlt,
    iretcd,
    xplsp,
    fplsp,
    xpls,
    method,
    udiag,
):
    """uncmin.c ``tregup`` (:444): trust-region accept/update (methods
    2 & 3). Returns ``(dlt, iretcd, fplsp, fpls, mxtake)``; mutates
    ``xpls``/``xplsp``."""
    mxtake = False
    for i in range(n):
        xpls[i] = float(x[i]) + float(sc[i])
    fpls = fcn(xpls)
    dltf = fpls - f
    slp = _ddot(n, g, sc)
    if iretcd == 3 and (fpls >= fplsp or dltf > slp * 1e-4):
        iretcd = 0
        for i in range(n):
            xpls[i] = float(xplsp[i])
        fpls = fplsp
        dlt *= 0.5
    else:
        if dltf > slp * 1e-4:
            rln = 0.0
            for i in range(n):
                temp1 = abs(float(sc[i])) / _fmax2(
                    abs(float(xpls[i])), 1.0 / float(sx[i])
                )
                rln = max(rln, temp1)
            if rln < steptl:
                iretcd = 1
            else:
                iretcd = 2
                dltmp = -slp * dlt / ((dltf - slp) * 2.0)
                if dltmp < dlt * 0.1:
                    dlt *= 0.1
                else:
                    dlt = dltmp
        else:
            dltfp = 0.0
            if method == 2:
                for i in range(n):
                    temp1 = 0.0
                    for j in range(i, n):
                        temp1 += float(a[j, i]) * float(sc[j])
                    dltfp += temp1 * temp1
            else:
                for i in range(n):
                    dltfp += float(udiag[i]) * float(sc[i]) * float(sc[i])
                    temp1 = 0.0
                    for j in range(i + 1, n):
                        temp1 += float(a[i, j]) * float(sc[i]) * float(sc[j])
                    dltfp += temp1 * 2.0
            dltfp = slp + dltfp / 2.0
            if (
                iretcd != 2
                and abs(dltfp - dltf) <= abs(dltf) * 0.1
                and nwtake
                and dlt <= stepmx * 0.99
            ):
                iretcd = 3
                for i in range(n):
                    xplsp[i] = float(xpls[i])
                fplsp = fpls
                dlt = _fmin2(dlt * 2.0, stepmx)
            else:
                iretcd = 0
                if dlt > stepmx * 0.99:
                    mxtake = True
                if dltf >= dltfp * 0.1:
                    dlt *= 0.5
                elif dltf <= dltfp * 0.75:
                    dlt = _fmin2(dlt * 2.0, stepmx)
    return dlt, iretcd, fplsp, fpls, mxtake


def _lnsrch(n, x, f, g, p, xpls, fcn, stepmx, steptl, sx):
    """uncmin.c ``lnsrch`` (:614): backtracking line search (method 1).
    Mutates ``p`` (rescaled when longer than stepmx) and ``xpls``;
    returns ``(fpls, iretcd, mxtake)``. Quadratic first backtrack, cubic
    thereafter, with the BDR 2000 ``fpls >= DBL_MAX`` guard."""
    firstback = True
    pfpls = 0.0
    plmbda = 0.0
    temp1 = 0.0
    for i in range(n):
        temp1 = _rfma(float(sx[i]) * float(sx[i]) * float(p[i]), float(p[i]), temp1)
    sln = math.sqrt(temp1)
    if sln > stepmx:
        _dscal(n, stepmx / sln, p)
        sln = stepmx
    slp = _ddot(n, g, p)
    rln = 0.0
    for i in range(n):
        temp1 = abs(float(p[i])) / _fmax2(abs(float(x[i])), 1.0 / float(sx[i]))
        rln = max(rln, temp1)
    rmnlmb = steptl / rln
    lam = 1.0
    mxtake = False
    iretcd = 2
    fpls = 0.0
    while iretcd > 1:
        for i in range(n):
            xpls[i] = _rfma(lam, float(p[i]), float(x[i]))
        fpls = fcn(xpls)
        if fpls <= _rfma(slp * 1e-4, lam, f):
            iretcd = 0
            if lam == 1.0 and sln > stepmx * 0.99:
                mxtake = True
            return fpls, iretcd, mxtake
        if lam < rmnlmb:
            iretcd = 1
            return fpls, iretcd, mxtake
        if fpls >= _DBL_MAX:
            lam *= 0.1
            firstback = True
        else:
            if firstback:
                tlmbda = -lam * slp / ((fpls - f - slp) * 2.0)
                firstback = False
            else:
                t1 = _rfma(-lam, slp, fpls - f)
                t2 = _rfma(-plmbda, slp, pfpls - f)
                t3 = 1.0 / (lam - plmbda)
                a3 = 3.0 * t3 * (t1 / (lam * lam) - t2 / (plmbda * plmbda))
                b = t3 * (t2 * lam / (plmbda * plmbda) - t1 * plmbda / (lam * lam))
                disc = _rfma(b, b, -(a3 * slp))
                if disc > b * b:
                    tlmbda = (
                        -b + (-math.sqrt(disc) if a3 < 0 else math.sqrt(disc))
                    ) / a3
                else:
                    tlmbda = (
                        -b + (math.sqrt(disc) if a3 < 0 else -math.sqrt(disc))
                    ) / a3
                tlmbda = min(tlmbda, lam * 0.5)
            plmbda = lam
            pfpls = fpls
            if tlmbda < lam * 0.1:
                lam *= 0.1
            else:
                lam = tlmbda
    return fpls, iretcd, mxtake


def _dog_1step(
    n, g, a, p, sx, rnwtln, dlt, nwtake, fstdog, ssd, v, cln, eta, sc, stepmx
):
    """uncmin.c ``dog_1step`` (:742): one double-dogleg step (method 2).
    Returns ``(dlt, nwtake, fstdog, cln, eta)``; mutates ssd, v, sc."""
    nwtake = rnwtln <= dlt
    if nwtake:
        for i in range(n):
            sc[i] = float(p[i])
        dlt = rnwtln
        return dlt, nwtake, fstdog, cln, eta
    if fstdog:
        fstdog = False
        alpha = 0.0
        for i in range(n):
            alpha += float(g[i]) * float(g[i]) / (float(sx[i]) * float(sx[i]))
        bet = 0.0
        for i in range(n):
            tmp = 0.0
            for j in range(i, n):
                tmp += float(a[j, i]) * float(g[j]) / (float(sx[j]) * float(sx[j]))
            bet += tmp * tmp
        for i in range(n):
            ssd[i] = -(alpha / bet) * float(g[i]) / float(sx[i])
        cln = alpha * math.sqrt(alpha) / bet
        eta = 0.8 * alpha * alpha / (-bet * _ddot(n, g, p)) + 0.2
        for i in range(n):
            v[i] = eta * float(sx[i]) * float(p[i]) - float(ssd[i])
        if dlt == -1.0:
            dlt = _fmin2(cln, stepmx)
    if eta * rnwtln <= dlt:
        for i in range(n):
            sc[i] = dlt / rnwtln * float(p[i])
    elif cln >= dlt:
        for i in range(n):
            sc[i] = dlt / cln * float(ssd[i]) / float(sx[i])
    else:
        dot1 = _ddot(n, v, ssd)
        dot2 = _ddot(n, v, v)
        alam = (-dot1 + math.sqrt(dot1 * dot1 - dot2 * (cln * cln - dlt * dlt))) / dot2
        for i in range(n):
            sc[i] = (float(ssd[i]) + alam * float(v[i])) / float(sx[i])
    return dlt, nwtake, fstdog, cln, eta


def _dogdrv(n, x, f, g, a, p, xpls, fcn, sx, stepmx, steptl, dlt):
    """uncmin.c ``dogdrv`` (:840): double-dogleg driver (method 2).
    Returns ``(fpls, dlt, iretcd, mxtake)``; mutates ``xpls``."""
    ssd = np.zeros(n)
    v = np.zeros(n)
    xplsp = np.zeros(n)
    sc = np.zeros(n)
    fplsp = 0.0
    cln = 0.0
    eta = 0.0
    nwtake = False
    tmp = 0.0
    for i in range(n):
        tmp += float(sx[i]) * float(sx[i]) * float(p[i]) * float(p[i])
    rnwtln = math.sqrt(tmp)
    iretcd = 4
    fstdog = True
    fpls = 0.0
    mxtake = False
    while iretcd > 1:
        dlt, nwtake, fstdog, cln, eta = _dog_1step(
            n, g, a, p, sx, rnwtln, dlt, nwtake, fstdog, ssd, v, cln, eta, sc, stepmx
        )
        dlt, iretcd, fplsp, fpls, mxtake = _tregup(
            n,
            x,
            f,
            g,
            a,
            fcn,
            sc,
            sx,
            nwtake,
            stepmx,
            steptl,
            dlt,
            iretcd,
            xplsp,
            fplsp,
            xpls,
            2,
            ssd,
        )
    return fpls, dlt, iretcd, mxtake


def _hook_1step(
    n, g, a, udiag, p, sx, rnwtln, dlt, amu, dltp, phi, phip0, fstime, sc, wrk0, epsm
):
    """uncmin.c ``hook_1step`` (:908): one More-Hebdon step (method 3).
    Returns ``(dlt, amu, phi, phip0, fstime, nwtake)``; mutates a, sc,
    wrk0."""
    hi = 1.5
    alo = 0.75
    nwtake = rnwtln <= hi * dlt
    if nwtake:
        for i in range(n):
            sc[i] = float(p[i])
        dlt = _fmin2(dlt, rnwtln)
        amu = 0.0
        return dlt, amu, phi, phip0, fstime, nwtake
    if amu > 0.0:
        amu -= (phi + dltp) * (dltp - dlt + phi) / (dlt * phip0)
    phi = rnwtln - dlt
    if fstime:
        for i in range(n):
            wrk0[i] = float(sx[i]) * float(sx[i]) * float(p[i])
        _dtrsl(a, n, wrk0, 0)
        temp1 = _dnrm2(n, wrk0)
        phip0 = -(temp1 * temp1) / rnwtln
        fstime = False
    phip = phip0
    amulo = -phi / phip
    amuup = 0.0
    for i in range(n):
        amuup += float(g[i]) * float(g[i]) / (float(sx[i]) * float(sx[i]))
    amuup = math.sqrt(amuup) / dlt
    while True:
        if amu < amulo or amu > amuup:
            amu = _fmax2(math.sqrt(amulo * amuup), amuup * 0.001)
        for i in range(n):
            a[i, i] = float(udiag[i]) + amu * float(sx[i]) * float(sx[i])
            for j in range(i):
                a[i, j] = float(a[j, i])
        _choldc(n, a, 0.0, math.sqrt(epsm))
        for i in range(n):
            wrk0[i] = -float(g[i])
        _lltslv(n, a, sc, wrk0)
        stepln = 0.0
        for i in range(n):
            stepln += float(sx[i]) * float(sx[i]) * float(sc[i]) * float(sc[i])
        stepln = math.sqrt(stepln)
        phi = stepln - dlt
        for i in range(n):
            wrk0[i] = float(sx[i]) * float(sx[i]) * float(sc[i])
        _dtrsl(a, n, wrk0, 0)
        temp1 = _dnrm2(n, wrk0)
        phip = -(temp1 * temp1) / stepln
        if (alo * dlt <= stepln <= hi * dlt) or (amuup - amulo > 0.0):
            break
        temp1 = (amu - phi) / phip
        amulo = _fmax2(amulo, temp1)
        if phi < 0.0:
            amuup = _fmin2(amuup, amu)
        amu -= stepln * phi / (dlt * phip)
    return dlt, amu, phi, phip0, fstime, nwtake


def _hookdrv(
    n,
    x,
    f,
    g,
    a,
    udiag,
    p,
    xpls,
    fcn,
    sx,
    stepmx,
    steptl,
    dlt,
    amu,
    dltp,
    phi,
    phip0,
    epsm,
    itncnt,
):
    """uncmin.c ``hookdrv`` (:1047): More-Hebdon driver (method 3).
    Returns ``(fpls, dlt, iretcd, mxtake, amu, dltp, phi, phip0)``;
    mutates a, xpls."""
    sc = np.zeros(n)
    xplsp = np.zeros(n)
    wrk0 = np.zeros(n)
    fplsp = 0.0
    tmp = 0.0
    for i in range(n):
        tmp += float(sx[i]) * float(sx[i]) * float(p[i]) * float(p[i])
    rnwtln = math.sqrt(tmp)
    if itncnt == 1:
        amu = 0.0
        if dlt == -1.0:
            alpha = 0.0
            for i in range(n):
                alpha += float(g[i]) * float(g[i]) / (float(sx[i]) * float(sx[i]))
            bet = 0.0
            for i in range(n):
                tmp = 0.0
                for j in range(i, n):
                    tmp += float(a[j, i]) * float(g[j]) / (float(sx[j]) * float(sx[j]))
                bet += tmp * tmp
            dlt = alpha * math.sqrt(alpha) / bet
            dlt = min(dlt, stepmx)
    iretcd = 4
    fstime = True
    fpls = 0.0
    mxtake = False
    while iretcd > 1:
        dlt, amu, phi, phip0, fstime, nwtake = _hook_1step(
            n,
            g,
            a,
            udiag,
            p,
            sx,
            rnwtln,
            dlt,
            amu,
            dltp,
            phi,
            phip0,
            fstime,
            sc,
            wrk0,
            epsm,
        )
        dltp = dlt
        dlt, iretcd, fplsp, fpls, mxtake = _tregup(
            n,
            x,
            f,
            g,
            a,
            fcn,
            sc,
            sx,
            nwtake,
            stepmx,
            steptl,
            dlt,
            iretcd,
            xplsp,
            fplsp,
            xpls,
            3,
            udiag,
        )
    return fpls, dlt, iretcd, mxtake, amu, dltp, phi, phip0


def _secunf(n, x, g, a, udiag, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt):
    """uncmin.c ``secunf`` (:1147): unfactored BFGS update (method 3).
    Returns ``noupdt``; mutates a."""
    s = np.zeros(n)
    y = np.zeros(n)
    t = np.zeros(n)
    for i in range(n):
        a[i, i] = float(udiag[i])
        for j in range(i):
            a[i, j] = float(a[j, i])
    noupdt = itncnt == 1
    for i in range(n):
        s[i] = float(xpls[i]) - float(x[i])
        y[i] = float(gpls[i]) - float(g[i])
    den1 = _ddot(n, s, y)
    snorm2 = _dnrm2(n, s)
    ynrm2 = _dnrm2(n, y)
    if den1 < math.sqrt(epsm) * snorm2 * ynrm2:
        return noupdt
    _mvmlts(n, a, s, t)
    den2 = _ddot(n, s, t)
    if noupdt:
        gam = den1 / den2
        den2 *= gam
        for j in range(n):
            t[j] = float(t[j]) * gam
            for i in range(j, n):
                a[i, j] = float(a[i, j]) * gam
        noupdt = False
    skpupd = True
    for i in range(n):
        tol = rnf * _fmax2(abs(float(g[i])), abs(float(gpls[i])))
        if iagflg == 0:
            tol /= math.sqrt(rnf)
        if abs(float(y[i]) - float(t[i])) >= tol:
            skpupd = False
            break
    if skpupd:
        return noupdt
    for j in range(n):
        for i in range(j, n):
            a[i, j] = (
                float(a[i, j])
                + float(y[i]) * float(y[j]) / den1
                - float(t[i]) * float(t[j]) / den2
            )
    return noupdt


def _secfac(n, x, g, a, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt):
    """uncmin.c ``secfac`` (:1241): factored BFGS update of the Cholesky
    factor (methods 1 & 2). Returns ``noupdt``; mutates a."""
    s = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)
    w = np.zeros(n)
    noupdt = itncnt == 1
    for i in range(n):
        s[i] = float(xpls[i]) - float(x[i])
        y[i] = float(gpls[i]) - float(g[i])
    den1 = _ddot(n, s, y)
    snorm2 = _dnrm2(n, s)
    ynrm2 = _dnrm2(n, y)
    if den1 < math.sqrt(epsm) * snorm2 * ynrm2:
        return noupdt
    _mvmltu(n, a, s, u)
    den2 = _ddot(n, u, u)
    alp = math.sqrt(den1 / den2)
    if noupdt:
        for j in range(n):
            u[j] = alp * float(u[j])
            for i in range(j, n):
                a[i, j] = float(a[i, j]) * alp
        noupdt = False
        den2 = den1
        alp = 1.0
    _mvmltl(n, a, u, w)
    reltol = math.sqrt(rnf) if iagflg == 0 else rnf
    skpupd = True
    for i in range(n):
        skpupd = abs(float(y[i]) - float(w[i])) < reltol * _fmax2(
            abs(float(g[i])), abs(float(gpls[i]))
        )
        if not skpupd:
            break
    if skpupd:
        return noupdt
    for i in range(n):
        w[i] = _rfma(-alp, float(w[i]), float(y[i]))
    alp /= den1
    for i in range(n):
        u[i] = float(u[i]) * alp
    for i in range(1, n):
        for j in range(i):
            a[j, i] = float(a[i, j])
            a[i, j] = 0.0
    _qrupdt(n, a, u, w)
    for i in range(1, n):
        for j in range(i):
            a[i, j] = float(a[j, i])
    return noupdt


def _chlhsn(n, a, epsm, sx, udiag):
    """uncmin.c ``chlhsn`` (:1361): scaled, safely-positive-definite
    LL' of the model Hessian. Mutates a (L in lower triangle, Hessian
    in upper) and udiag."""
    for j in range(n):
        for i in range(j, n):
            a[i, j] = float(a[i, j]) / (float(sx[i]) * float(sx[j]))
    tol = math.sqrt(epsm)
    diagmx = float(a[0, 0])
    diagmn = float(a[0, 0])
    if n > 1:
        for i in range(1, n):
            tmp = float(a[i, i])
            diagmn = min(diagmn, tmp)
            diagmx = max(diagmx, tmp)
    posmax = _fmax2(diagmx, 0.0)
    if diagmn <= posmax * tol:
        amu = _rfma(tol, posmax - diagmn, -diagmn)
        if amu == 0.0:
            offmax = 0.0
            for i in range(1, n):
                for j in range(i):
                    tmp = abs(float(a[i, j]))
                    offmax = max(offmax, tmp)
            amu = 1.0 if offmax == 0.0 else offmax * (tol + 1.0)
        for i in range(n):
            a[i, i] = float(a[i, i]) + amu
        diagmx += amu
    for i in range(n):
        udiag[i] = float(a[i, i])
        for j in range(i):
            a[j, i] = float(a[i, j])
    addmax = _choldc(n, a, diagmx, tol)
    if addmax > 0.0:
        for i in range(n):
            a[i, i] = float(udiag[i])
            for j in range(i):
                a[i, j] = float(a[j, i])
        evmin = 0.0
        evmax = float(a[0, 0])
        for i in range(n):
            offrow = 0.0
            for j in range(i):
                offrow += abs(float(a[i, j]))
            for j in range(i + 1, n):
                offrow += abs(float(a[j, i]))
            tmp = float(a[i, i]) - offrow
            evmin = min(evmin, tmp)
            tmp = float(a[i, i]) + offrow
            evmax = max(evmax, tmp)
        sdd = _rfma(tol, evmax - evmin, -evmin)
        amu = _fmin2(sdd, addmax)
        for i in range(n):
            a[i, i] = float(a[i, i]) + amu
            udiag[i] = float(a[i, i])
        _choldc(n, a, 0.0, tol)
    for j in range(n):
        for i in range(j, n):
            a[i, j] = float(a[i, j]) * float(sx[i])
        for i in range(j):
            a[i, j] = float(a[i, j]) * float(sx[i]) * float(sx[j])
        udiag[j] = float(udiag[j]) * float(sx[j]) * float(sx[j])


def _hsnint(n, a, sx, method):
    """uncmin.c ``hsnint`` (:1539): initial Hessian for secant updates
    (diag(sx) as the factored L for methods 1-2, diag(sx²) for 3)."""
    for i in range(n):
        a[i, i] = float(sx[i]) * float(sx[i]) if method == 3 else float(sx[i])
        for j in range(i):
            a[i, j] = 0.0


def _fstofd(m, n, xpls, fcn_vec, fpls, a, sx, rnoise, icase):
    """uncmin.c ``fstofd`` (:1567): forward-difference derivative of the
    m-vector function ``fcn_vec`` at ``xpls`` into columns of ``a``.
    ``fpls`` is the m-vector value at ``xpls``; ``xpls`` perturbed and
    restored in place."""
    for j in range(n):
        stepsz = math.sqrt(rnoise) * _fmax2(abs(float(xpls[j])), 1.0 / float(sx[j]))
        xtmpj = float(xpls[j])
        xpls[j] = xtmpj + stepsz
        fhat = fcn_vec(xpls)
        xpls[j] = xtmpj
        for i in range(m):
            a[i, j] = (float(fhat[i]) - float(fpls[i])) / stepsz
    if icase == 3 and n > 1:
        for i in range(1, m):
            for j in range(i):
                a[i, j] = (float(a[i, j]) + float(a[j, i])) / 2.0


def _fstocd(n, x, fcn, sx, rnoise, g):
    """uncmin.c ``fstocd`` (:1648): central-difference gradient into
    ``g``; ``x`` perturbed and restored in place."""
    for i in range(n):
        xtempi = float(x[i])
        stepi = rnoise ** (1.0 / 3.0) * _fmax2(abs(xtempi), 1.0 / float(sx[i]))
        x[i] = xtempi + stepi
        fplus = fcn(x)
        x[i] = xtempi - stepi
        fminus = fcn(x)
        x[i] = xtempi
        g[i] = (fplus - fminus) / (stepi * 2.0)


def _sndofd(n, xpls, fcn, fpls, a, sx, rnoise):
    """uncmin.c ``sndofd`` (:1686): second-order forward-difference
    Hessian (no analytic gradient); fills the lower triangle of ``a``."""
    stepsz = np.zeros(n)
    anbr = np.zeros(n)
    for i in range(n):
        xtmpi = float(xpls[i])
        stepsz[i] = rnoise ** (1.0 / 3.0) * _fmax2(abs(xtmpi), 1.0 / float(sx[i]))
        xpls[i] = xtmpi + float(stepsz[i])
        anbr[i] = fcn(xpls)
        xpls[i] = xtmpi
    for i in range(n):
        xtmpi = float(xpls[i])
        xpls[i] = xtmpi + float(stepsz[i]) * 2.0
        fhat = fcn(xpls)
        a[i, i] = ((fpls - float(anbr[i])) + (fhat - float(anbr[i]))) / (
            float(stepsz[i]) * float(stepsz[i])
        )
        if i == 0:
            xpls[i] = xtmpi
            continue
        xpls[i] = xtmpi + float(stepsz[i])
        for j in range(i):
            xtmpj = float(xpls[j])
            xpls[j] = xtmpj + float(stepsz[j])
            fhat = fcn(xpls)
            a[i, j] = ((fpls - float(anbr[i])) + (fhat - float(anbr[j]))) / (
                float(stepsz[i]) * float(stepsz[j])
            )
            xpls[j] = xtmpj
        xpls[i] = xtmpi


def _grdchk(n, x, fcn, f, g, typsiz, sx, fscale, rnf, analtl, msg):
    """uncmin.c ``grdchk`` (:1760): compare analytic vs FD gradient;
    returns msg (−21 on probable coding error)."""
    wrk1 = np.zeros(n)
    fpls_vec = np.array([f])
    _fstofd(
        1,
        n,
        x,
        lambda xx: np.array([fcn(xx)]),
        fpls_vec,
        wrk1.reshape(1, n),
        sx,
        rnf,
        1,
    )
    for i in range(n):
        gs = _fmax2(abs(f), fscale) / _fmax2(abs(float(x[i])), float(typsiz[i]))
        if abs(float(g[i]) - float(wrk1[i])) > _fmax2(abs(float(g[i])), gs) * analtl:
            return -21
    return msg


def _heschk(n, x, fcn, d1fcn, d2fcn, f, g, a, typsiz, sx, rnf, analtl, iagflg, msg):
    """uncmin.c ``heschk`` (:1804): compare analytic vs FD Hessian;
    returns msg (−22 on probable coding error). Mutates a."""
    udiag = np.zeros(n)
    if iagflg:
        _fstofd(n, n, x, d1fcn, g, a, sx, rnf, 3)
    else:
        _sndofd(n, x, fcn, f, a, sx, rnf)
    for j in range(n):
        udiag[j] = float(a[j, j])
        for i in range(j + 1, n):
            a[j, i] = float(a[i, j])
    d2fcn(x, a)
    for j in range(n):
        hs = _fmax2(abs(float(g[j])), 1.0) / _fmax2(abs(float(x[j])), float(typsiz[j]))
        if (
            abs(float(a[j, j]) - float(udiag[j]))
            > _fmax2(abs(float(udiag[j])), hs) * analtl
        ):
            return -22
        for i in range(j + 1, n):
            temp1 = float(a[i, j])
            temp2 = abs(temp1 - float(a[j, i]))
            if temp2 > _fmax2(abs(temp1), hs) * analtl:
                return -22
    return msg


def _opt_stop(
    n,
    xpls,
    fpls,
    gpls,
    x,
    itncnt,
    icscmx,
    gradtl,
    steptl,
    sx,
    fscale,
    itnlim,
    iretcd,
    mxtake,
):
    """uncmin.c ``opt_stop`` (:1884): termination tests. Returns
    ``(itrmcd, icscmx)`` — 0 continue, 1 relgrad small, 2 relstep small,
    3 global step failed, 4 iteration limit, 5 stepmx hit 5×."""
    if iretcd == 1:
        return 3, icscmx
    d = _fmax2(abs(fpls), fscale)
    rgx = 0.0
    for i in range(n):
        relgrd = (
            abs(float(gpls[i])) * _fmax2(abs(float(xpls[i])), 1.0 / float(sx[i])) / d
        )
        rgx = max(rgx, relgrd)
    jtrmcd = 1
    if rgx > gradtl:
        if itncnt == 0:
            return 0, icscmx
        rsx = 0.0
        for i in range(n):
            relstp = abs(float(xpls[i]) - float(x[i])) / _fmax2(
                abs(float(xpls[i])), 1.0 / float(sx[i])
            )
            rsx = max(rsx, relstp)
        jtrmcd = 2
        if rsx > steptl:
            jtrmcd = 4
            if itncnt < itnlim:
                if not mxtake:
                    icscmx = 0
                    return 0, icscmx
                icscmx += 1
                if icscmx < 5:
                    return 0, icscmx
                jtrmcd = 5
    return jtrmcd, icscmx


def _optchk(
    n,
    x,
    typsiz,
    sx,
    fscale,
    gradtl,
    itnlim,
    ndigit,
    epsm,
    dlt,
    method,
    iexp,
    iagflg,
    iahflg,
    stepmx,
    msg,
):
    """uncmin.c ``optchk`` (:1973): validate/default the inputs. Mutates
    ``typsiz``/``sx``; returns the (possibly reset) scalars ``(fscale,
    itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, msg)``
    with msg < 0 on input error."""
    if method < 1 or method > 3:
        method = 1
    if iagflg != 1:
        iagflg = 0
    if iahflg != 1:
        iahflg = 0
    if iexp != 0:
        iexp = 1
    if (msg // 2) % 2 == 1 and iagflg == 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -6
    if (msg // 4) % 2 == 1 and iahflg == 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -7
    if n <= 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -1
    if n == 1 and msg % 2 == 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -2
    for i in range(n):
        if float(typsiz[i]) == 0.0:
            typsiz[i] = 1.0
        elif float(typsiz[i]) < 0.0:
            typsiz[i] = -float(typsiz[i])
        sx[i] = 1.0 / float(typsiz[i])
    if stepmx <= 0.0:
        stpsiz = 0.0
        for i in range(n):
            stpsiz = _rfma(
                float(x[i]) * float(x[i]) * float(sx[i]), float(sx[i]), stpsiz
            )
        stepmx = 1000.0 * _fmax2(math.sqrt(stpsiz), 1.0)
    if fscale == 0.0:
        fscale = 1.0
    elif fscale < 0.0:
        fscale = -fscale
    if gradtl < 0.0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -3
    if itnlim <= 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -4
    if ndigit == 0:
        return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, -5
    if ndigit < 0:
        ndigit = int(-math.log10(epsm))
    if dlt <= 0.0:
        dlt = -1.0
    elif dlt > stepmx:
        dlt = stepmx
    return fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, msg


def _prt_result(n, x, f, g, p, itncnt, iflg):
    """uncmin.c ``prt_result`` (:2078): iteration report (reached only
    when the msg print bits allow — never for ``nlm(print.level=0)``)."""
    print(f"iteration = {itncnt}")
    if iflg != 0:
        print("Step:")
        print(np.asarray(p[:n]))
    print("Parameter:")
    print(np.asarray(x[:n]))
    print("Function Value")
    print(f)
    print("Gradient:")
    print(np.asarray(g[:n]))
    print()


def optdrv(
    n,
    x,
    fcn,
    d1fcn,
    d2fcn,
    typsiz,
    fscale,
    method,
    iexp,
    msg,
    ndigit,
    itnlim,
    iagflg,
    iahflg,
    dlt,
    gradtl,
    stepmx,
    steptl,
    xpls,
    gpls,
):
    """uncmin.c ``optdrv`` (:2166): the optimization driver. Mutates
    ``x`` (working copy), fills ``xpls``/``gpls``; returns ``(fpls,
    itrmcd, itncnt, msg)`` with msg < 0 on input error, 0 otherwise."""
    a = np.zeros((n, n))
    udiag = np.zeros(n)
    g = np.zeros(n)
    p = np.zeros(n)
    sx = np.zeros(n)
    wrk1 = np.zeros(n)
    itncnt = 0
    epsm = _DBL_EPSILON
    (fscale, itnlim, ndigit, dlt, method, iexp, iagflg, iahflg, stepmx, msg) = _optchk(
        n,
        x,
        typsiz,
        sx,
        fscale,
        gradtl,
        itnlim,
        ndigit,
        epsm,
        dlt,
        method,
        iexp,
        iagflg,
        iahflg,
        stepmx,
        msg,
    )
    if msg < 0:
        return 0.0, 0, itncnt, msg
    rnf = _fmax2(10.0 ** (-float(ndigit)), epsm)
    analtl = _fmax2(0.1, math.sqrt(rnf))
    f = fcn(x)
    if not iagflg:
        _fstofd(
            1,
            n,
            x,
            lambda xx: np.array([fcn(xx)]),
            np.array([f]),
            g.reshape(1, n),
            sx,
            rnf,
            1,
        )
    else:
        g[:] = d1fcn(x)
        if (msg // 2) % 2 == 0:
            msg = _grdchk(n, x, fcn, f, g, typsiz, sx, fscale, rnf, analtl, msg)
            if msg < 0:
                return f, 0, itncnt, msg
    iretcd = -1
    itrmcd, icscmx = _opt_stop(
        n, x, f, g, wrk1, itncnt, 0, gradtl, steptl, sx, fscale, itnlim, iretcd, False
    )
    if itrmcd != 0:
        fpls = f
        for i in range(n):
            xpls[i] = float(x[i])
            gpls[i] = float(g[i])
        if (msg // 8) % 2 == 0:
            _prt_result(n, xpls, fpls, gpls, p, itncnt, 0)
        return fpls, itrmcd, itncnt, 0
    if iexp:
        _hsnint(n, a, sx, method)
    else:
        if not iahflg:
            if iagflg:
                _fstofd(n, n, x, d1fcn, g, a, sx, rnf, 3)
            else:
                _sndofd(n, x, fcn, f, a, sx, rnf)
        else:
            if (msg // 4) % 2 == 1:
                d2fcn(x, a)
            else:
                msg = _heschk(
                    n,
                    x,
                    fcn,
                    d1fcn,
                    d2fcn,
                    f,
                    g,
                    a,
                    typsiz,
                    sx,
                    rnf,
                    analtl,
                    iagflg,
                    msg,
                )
                if msg < 0:
                    return f, 0, itncnt, msg
    if (msg // 8) % 2 == 0:
        _prt_result(n, x, f, g, p, itncnt, 1)

    fpls = 0.0
    mxtake = False
    noupdt = False
    dltsav = dlpsav = phisav = amusav = phpsav = 0.0
    dltp = phi = phip0 = amu = 0.0
    while True:
        itncnt += 1
        if not (iexp and method != 3):
            _chlhsn(n, a, epsm, sx, udiag)
        while True:
            for i in range(n):
                wrk1[i] = -float(g[i])
            _lltslv(n, a, p, wrk1)
            if iagflg == 0 and method != 1:
                dltsav = dlt
                if method != 2:
                    amusav = amu
                    dlpsav = dltp
                    phisav = phi
                    phpsav = phip0
            if method == 1:
                fpls, iretcd, mxtake = _lnsrch(
                    n, x, f, g, p, xpls, fcn, stepmx, steptl, sx
                )
            elif method == 2:
                fpls, dlt, iretcd, mxtake = _dogdrv(
                    n, x, f, g, a, p, xpls, fcn, sx, stepmx, steptl, dlt
                )
            else:
                (fpls, dlt, iretcd, mxtake, amu, dltp, phi, phip0) = _hookdrv(
                    n,
                    x,
                    f,
                    g,
                    a,
                    udiag,
                    p,
                    xpls,
                    fcn,
                    sx,
                    stepmx,
                    steptl,
                    dlt,
                    amu,
                    dltp,
                    phi,
                    phip0,
                    epsm,
                    itncnt,
                )
            if iretcd == 1 and iagflg == 0:
                iagflg = -1
                _fstocd(n, x, fcn, sx, rnf, g)
                if method == 1:
                    continue  # goto L105
                dlt = dltsav
                if method == 2:
                    continue  # goto L105
                amu = amusav
                dltp = dlpsav
                phi = phisav
                phip0 = phpsav
                _chlhsn(n, a, epsm, sx, udiag)  # goto L103
                continue
            break
        for i in range(n):
            p[i] = float(xpls[i]) - float(x[i])
        if iagflg == -1:
            _fstocd(n, xpls, fcn, sx, rnf, gpls)
        elif iagflg == 0:
            _fstofd(
                1,
                n,
                xpls,
                lambda xx: np.array([fcn(xx)]),
                np.array([fpls]),
                gpls.reshape(1, n),
                sx,
                rnf,
                1,
            )
        else:
            gpls[:] = d1fcn(xpls)
        itrmcd, icscmx = _opt_stop(
            n,
            xpls,
            fpls,
            gpls,
            x,
            itncnt,
            icscmx,
            gradtl,
            steptl,
            sx,
            fscale,
            itnlim,
            iretcd,
            mxtake,
        )
        if itrmcd != 0:
            break
        if iexp:
            if method == 3:
                noupdt = _secunf(
                    n, x, g, a, udiag, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt
                )
            else:
                noupdt = _secfac(
                    n, x, g, a, xpls, gpls, epsm, itncnt, rnf, iagflg, noupdt
                )
        else:
            if not iahflg:
                if iagflg:
                    _fstofd(n, n, xpls, d1fcn, gpls, a, sx, rnf, 3)
                else:
                    _sndofd(n, xpls, fcn, fpls, a, sx, rnf)
            else:
                d2fcn(xpls, a)
        if (msg // 16) % 2 == 1:
            _prt_result(n, xpls, fpls, gpls, p, itncnt, 1)
        f = fpls
        for i in range(n):
            x[i] = float(xpls[i])
            g[i] = float(gpls[i])
    if itrmcd == 3:
        fpls = f
        for i in range(n):
            xpls[i] = float(x[i])
            gpls[i] = float(g[i])
    if (msg // 8) % 2 == 0:
        _prt_result(n, xpls, fpls, gpls, p, itncnt, 0)
    return fpls, itrmcd, itncnt, 0


def optif9(
    n,
    x,
    fcn,
    d1fcn,
    d2fcn,
    typsiz,
    fscale,
    method,
    iexp,
    msg,
    ndigit,
    itnlim,
    iagflg,
    iahflg,
    dlt,
    gradtl,
    stepmx,
    steptl,
):
    """uncmin.c ``optif9`` (:2561): the full-control entry point R's
    ``nlm`` uses. ``x``/``typsiz`` are mutated working copies. Returns
    ``(xpls, fpls, gpls, itrmcd, itncnt, msg)``."""
    xpls = np.zeros(n)
    gpls = np.zeros(n)
    fpls, itrmcd, itncnt, msg = optdrv(
        n,
        x,
        fcn,
        d1fcn,
        d2fcn,
        typsiz,
        fscale,
        method,
        iexp,
        msg,
        ndigit,
        itnlim,
        iagflg,
        iahflg,
        dlt,
        gradtl,
        stepmx,
        steptl,
        xpls,
        gpls,
    )
    return xpls, fpls, gpls, itrmcd, itncnt, msg
