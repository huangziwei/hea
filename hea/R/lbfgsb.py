"""Bit-exact port of R's L-BFGS-B — ``src/appl/lbfgsb.c`` (the f2c
translation of Zhu/Byrd/Lu-Chen/Nocedal's version 2.3 Fortran that R has
shipped since 2000) plus the ``lbfgsb()`` reverse-communication driver
from ``src/appl/optim.c`` that ``optim(method="L-BFGS-B")`` calls.

R's copy is NOT the L-BFGS-B 3.0 that scipy wraps: 2.3 and 3.0 differ in
the subspace minimization and in several guards, so scipy cannot
reproduce R's iterates. This is a line-by-line translation keeping every
accumulation order (BLAS level-1 through :mod:`hea.R._linpack`, LINPACK
``dtrsl``/``dpofa`` included), the task-string protocol, and the
Fortran-static state (here an explicit ``state`` dict threaded through
:func:`setulb`, reinitialized on ``task == "START"`` exactly like the C
statics).

R compiles this C with clang, whose default ``-ffp-contract=on`` fuses
``a ± b*c`` within one expression into a single-rounding FMA on arm64;
those contractions are mirrored via ``_rfma`` (fused on arm64, plain on
x86-64 — the per-arch policy matching R-as-built). BLAS calls resolve
to what R links (Accelerate on CRAN macOS — see
:mod:`hea.R._linpack`). Both mirrors are verified bit-exact against
R's own compiled ``lbfgsb`` driver (exported from ``libR.dylib``),
trajectory point by trajectory point.

Layout notes (f2c 1-based → Python 0-based):

* ``ws``/``wy`` are ``(n, m)`` arrays, ``sy``/``ss``/``wt`` are
  ``(m, m)``, ``wn``/``snd`` are ``(2m, 2m)``; ring pointers ``head``/
  ``pointr``/``itail`` stay 1-based as in the source (array accesses
  subtract 1).
* Index vectors (``indx``, ``indx2``, ``iorder``) store 1-based variable
  indices, as in the Fortran.
* ``wa`` is the shared ``8m`` scratch: cauchy's ``p``/``c``/``wbp``/``v``
  live at offsets ``0/2m/4m/6m``; ``cmprlb`` reads the ``c`` block and
  ``subsm`` reuses the front as ``wv`` — sharing that the algorithm
  relies on.

Entry point: :func:`lbfgsb` (the driver); the ``optim()``-level
fnscale/parscale semantics live in :mod:`hea.R.optimize`.
"""

from __future__ import annotations

import math

import numpy as np

from ._linpack import _daxpy, _dcopy, _ddot, _dpofa, _dscal, _dtrsl
from ._shared import _rfma

_DBL_EPSILON = float(np.finfo(np.float64).eps)


def _active(n, lo, u, nbd, x, iwhere, iprint, state):
    """lbfgsb.c ``active`` (:929): project x into the box, classify each
    variable in ``iwhere`` and set prjctd/cnstnd/boxed in ``state``."""
    nbdd = 0
    state["prjctd"] = False
    state["cnstnd"] = False
    state["boxed"] = True
    for i in range(n):
        if nbd[i] > 0:
            if nbd[i] <= 2 and float(x[i]) <= float(lo[i]):
                if float(x[i]) < float(lo[i]):
                    state["prjctd"] = True
                    x[i] = float(lo[i])
                nbdd += 1
            elif nbd[i] >= 2 and float(x[i]) >= float(u[i]):
                if float(x[i]) > float(u[i]):
                    state["prjctd"] = True
                    x[i] = float(u[i])
                nbdd += 1
    for i in range(n):
        if nbd[i] != 2:
            state["boxed"] = False
        if nbd[i] == 0:
            iwhere[i] = -1
        else:
            state["cnstnd"] = True
            if nbd[i] == 2 and float(u[i]) - float(lo[i]) <= 0.0:
                iwhere[i] = 3
            else:
                iwhere[i] = 0
    if iprint >= 0:
        if state["prjctd"]:
            print("The initial X is infeasible.  Restart with its projection.")
        if not state["cnstnd"]:
            print("This problem is unconstrained.")
    if iprint > 0:
        print(f"At X0, {nbdd} variables are exactly at the bounds")


def _bmv(m, sy, wt, col, v, p, ov=0, op=0):
    """lbfgsb.c ``bmv`` (:1025): p = M v for the 2col middle matrix of
    the compact L-BFGS formula (``v``/``p`` at offsets ``ov``/``op``).
    Returns ``info`` (nonzero when the dtrsl system is singular)."""
    if col == 0:
        return 0
    p[op + col] = float(v[ov + col])
    for i in range(2, col + 1):
        i2 = col + i
        s = 0.0
        for k in range(1, i):
            s += (
                float(sy[i - 1, k - 1]) * float(v[ov + k - 1]) / float(sy[k - 1, k - 1])
            )
        p[op + i2 - 1] = float(v[ov + i2 - 1]) + s
    info = _dtrsl(wt, col, p[op + col : op + 2 * col], 11)
    if info != 0:
        return info
    for i in range(1, col + 1):
        p[op + i - 1] = float(v[ov + i - 1]) / math.sqrt(float(sy[i - 1, i - 1]))
    info = _dtrsl(wt, col, p[op + col : op + 2 * col], 1)
    if info != 0:
        return info
    for i in range(1, col + 1):
        p[op + i - 1] = -float(p[op + i - 1]) / math.sqrt(float(sy[i - 1, i - 1]))
    for i in range(1, col + 1):
        s = 0.0
        for k in range(i + 1, col + 1):
            s += (
                float(sy[k - 1, i - 1])
                * float(p[op + col + k - 1])
                / float(sy[i - 1, i - 1])
            )
        p[op + i - 1] = float(p[op + i - 1]) + s
    return 0


def _hpsolb(n, t, iorder, iheap):
    """lbfgsb.c ``hpsolb`` (:2316): heapsort step — least element of
    ``t[:n]`` moved to ``t[n-1]``, remainder left as a heap (1-based
    heap arithmetic on 0-based storage)."""
    if iheap == 0:
        for k in range(2, n + 1):
            ddum = float(t[k - 1])
            indxin = int(iorder[k - 1])
            i = k
            while i > 1:
                j = i // 2
                if ddum < float(t[j - 1]):
                    t[i - 1] = float(t[j - 1])
                    iorder[i - 1] = int(iorder[j - 1])
                    i = j
                else:
                    break
            t[i - 1] = ddum
            iorder[i - 1] = indxin
    if n > 1:
        i = 1
        out = float(t[0])
        indxou = int(iorder[0])
        ddum = float(t[n - 1])
        indxin = int(iorder[n - 1])
        while True:
            j = i + i
            if j <= n - 1:
                if float(t[j]) < float(t[j - 1]):
                    j += 1
                if float(t[j - 1]) < ddum:
                    t[i - 1] = float(t[j - 1])
                    iorder[i - 1] = int(iorder[j - 1])
                    i = j
                    continue
            break
        t[i - 1] = ddum
        iorder[i - 1] = indxin
        t[n - 1] = out
        iorder[n - 1] = indxou


def _cauchy(
    n,
    x,
    lo,
    u,
    nbd,
    g,
    iorder,
    iwhere,
    t,
    d,
    xcp,
    m,
    wy,
    ws,
    sy,
    wt,
    theta,
    col,
    head,
    wa,
    iprint,
    sbgnrm,
    epsmch,
):
    """lbfgsb.c ``cauchy`` (:1154): generalized Cauchy point along the
    projected gradient path. Fills ``xcp`` (and iwhere/t/d/wa); returns
    ``(nint, info)``. ``wa`` blocks: p=0, c=2m, wbp=4m, v=6m."""
    op, oc, owbp, ov = 0, 2 * m, 4 * m, 6 * m
    if sbgnrm <= 0.0:
        if iprint >= 0:
            print("Subgnorm = 0.  GCP = X.")
        _dcopy(n, x, xcp)
        return 0, 0
    bnded = True
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = 2 * col
    f1 = 0.0
    if iprint >= 99:
        print("\n---------------- CAUCHY entered-------------------\n")
    for i in range(col2):
        wa[op + i] = 0.0
    tl = 0.0
    tu = 0.0
    for i in range(1, n + 1):
        neggi = -float(g[i - 1])
        if iwhere[i - 1] != 3 and iwhere[i - 1] != -1:
            if nbd[i - 1] <= 2:
                tl = float(x[i - 1]) - float(lo[i - 1])
            if nbd[i - 1] >= 2:
                tu = float(u[i - 1]) - float(x[i - 1])
            xlower = nbd[i - 1] <= 2 and tl <= 0.0
            xupper = nbd[i - 1] >= 2 and tu <= 0.0
            iwhere[i - 1] = 0
            if xlower:
                if neggi <= 0.0:
                    iwhere[i - 1] = 1
            elif xupper:
                if neggi >= 0.0:
                    iwhere[i - 1] = 2
            else:
                if abs(neggi) <= 0.0:
                    iwhere[i - 1] = -3
        pointr = head
        if iwhere[i - 1] != 0 and iwhere[i - 1] != -1:
            d[i - 1] = 0.0
        else:
            d[i - 1] = neggi
            f1 = _rfma(-neggi, neggi, f1)
            for j in range(1, col + 1):
                wa[op + j - 1] = _rfma(
                    float(wy[i - 1, pointr - 1]), neggi, float(wa[op + j - 1])
                )
                wa[op + col + j - 1] = _rfma(
                    float(ws[i - 1, pointr - 1]), neggi, float(wa[op + col + j - 1])
                )
                pointr = pointr % m + 1
            if nbd[i - 1] <= 2 and nbd[i - 1] != 0 and neggi < 0.0:
                nbreak += 1
                iorder[nbreak - 1] = i
                t[nbreak - 1] = tl / (-neggi)
                if nbreak == 1 or float(t[nbreak - 1]) < bkmin:
                    bkmin = float(t[nbreak - 1])
                    ibkmin = nbreak
            elif nbd[i - 1] >= 2 and neggi > 0.0:
                nbreak += 1
                iorder[nbreak - 1] = i
                t[nbreak - 1] = tu / neggi
                if nbreak == 1 or float(t[nbreak - 1]) < bkmin:
                    bkmin = float(t[nbreak - 1])
                    ibkmin = nbreak
            else:
                nfree -= 1
                iorder[nfree - 1] = i
                if abs(neggi) > 0.0:
                    bnded = False
    if theta != 1.0:
        _dscal(col, theta, wa, op + col)
    _dcopy(n, x, xcp)
    if nbreak == 0 and nfree == n + 1:
        if iprint > 100:
            print("Cauchy X = ", np.asarray(xcp[:n]))
        return 0, 0
    for j in range(col2):
        wa[oc + j] = 0.0
    f2 = -theta * f1
    f2_org = f2
    if col > 0:
        info = _bmv(m, sy, wt, col, wa, wa, ov=op, op=ov)
        if info != 0:
            return 0, info
        f2 -= _ddot(col2, wa, wa, ox=ov, oy=op)
    dtm = -f1 / f2
    tsum = 0.0
    nint = 1
    if iprint >= 99:
        print(f"There are {nbreak}  breakpoints")
    if nbreak != 0:
        nleft = nbreak
        iter_ = 1
        tj = 0.0
        while True:
            tj0 = tj
            if iter_ == 1:
                tj = bkmin
                ibp = int(iorder[ibkmin - 1])
            else:
                if iter_ == 2 and ibkmin != nbreak:
                    t[ibkmin - 1] = float(t[nbreak - 1])
                    iorder[ibkmin - 1] = int(iorder[nbreak - 1])
                _hpsolb(nleft, t, iorder, iter_ - 2)
                tj = float(t[nleft - 1])
                ibp = int(iorder[nleft - 1])
            dt = tj - tj0
            if dt != 0 and iprint >= 100:
                print(f"\nPiece    {nint} f1, f2 at start point {f1:.4e} {f2:.4e}")
                print(f"Distance to the next break point =  {dt:.4e}")
                print(f"Distance to the stationary point =  {dtm:.4e}")
            if dtm < dt:
                break
            tsum += dt
            nleft -= 1
            iter_ += 1
            dibp = float(d[ibp - 1])
            d[ibp - 1] = 0.0
            if dibp > 0.0:
                zibp = float(u[ibp - 1]) - float(x[ibp - 1])
                xcp[ibp - 1] = float(u[ibp - 1])
                iwhere[ibp - 1] = 2
            else:
                zibp = float(lo[ibp - 1]) - float(x[ibp - 1])
                xcp[ibp - 1] = float(lo[ibp - 1])
                iwhere[ibp - 1] = 1
            if iprint >= 100:
                print(f"Variable  {ibp}  is fixed.")
            if nleft == 0 and nbreak == n:
                dtm = dt
                if col > 0:
                    _daxpy(col2, dtm, wa, wa, ox=op, oy=oc)
                if iprint >= 100:
                    print("Cauchy X = ", np.asarray(xcp[:n]))
                if iprint >= 99:
                    print("\n---------------- exit CAUCHY----------------------\n")
                return nint, 0
            nint += 1
            dibp2 = dibp * dibp
            f1 = f1 + _rfma(-(theta * dibp), zibp, _rfma(dt, f2, dibp2))
            f2 = _rfma(-theta, dibp2, f2)
            if col > 0:
                _daxpy(col2, dt, wa, wa, ox=op, oy=oc)
                pointr = head
                for j in range(1, col + 1):
                    wa[owbp + j - 1] = float(wy[ibp - 1, pointr - 1])
                    wa[owbp + col + j - 1] = theta * float(ws[ibp - 1, pointr - 1])
                    pointr = pointr % m + 1
                info = _bmv(m, sy, wt, col, wa, wa, ov=owbp, op=ov)
                if info != 0:
                    return nint, info
                wmc = _ddot(col2, wa, wa, ox=oc, oy=ov)
                wmp = _ddot(col2, wa, wa, ox=op, oy=ov)
                wmw = _ddot(col2, wa, wa, ox=owbp, oy=ov)
                _daxpy(col2, -dibp, wa, wa, ox=owbp, oy=op)
                f1 = _rfma(dibp, wmc, f1)
                f2 = f2 + _rfma(2.0 * dibp, wmp, -(dibp2 * wmw))
            f2 = max(f2, epsmch * f2_org)
            if nleft > 0:
                dtm = -f1 / f2
                continue
            elif bnded:
                f1 = 0.0
                f2 = 0.0
                dtm = 0.0
            else:
                dtm = -f1 / f2
            break
    if iprint >= 99:
        print("\nGCP found in this segment")
        print(f"Piece    {nint} f1, f2 at start point {f1:.4e} {f2:.4e}")
        print(f"Distance to the stationary point =  {dtm:.4e}")
    dtm = max(0.0, dtm)
    tsum += dtm
    _daxpy(n, tsum, d, xcp)
    if col > 0:
        _daxpy(col2, dtm, wa, wa, ox=op, oy=oc)
    if iprint >= 100:
        print("Cauchy X = ", np.asarray(xcp[:n]))
    if iprint >= 99:
        print("\n---------------- exit CAUCHY----------------------\n")
    return nint, 0


def _cmprlb(
    n, m, x, g, ws, wy, sy, wt, z, r, wa, indx, theta, col, head, nfree, cnstnd
):
    """lbfgsb.c ``cmprlb`` (:1652): r = -Z'B(xcp-x) - Z'g using
    ``wa[2m:]`` = W'(xcp-x) from cauchy. Returns info (−8 on singular
    bmv system)."""
    if not cnstnd and col > 0:
        for i in range(n):
            r[i] = -float(g[i])
        return 0
    for i in range(1, nfree + 1):
        k = int(indx[i - 1])
        r[i - 1] = _rfma(-theta, float(z[k - 1]) - float(x[k - 1]), -float(g[k - 1]))
    info = _bmv(m, sy, wt, col, wa, wa, ov=2 * m, op=0)
    if info != 0:
        return -8
    pointr = head
    for j in range(1, col + 1):
        a1 = float(wa[j - 1])
        a2 = theta * float(wa[col + j - 1])
        for i in range(1, nfree + 1):
            k = int(indx[i - 1])
            r[i - 1] = float(r[i - 1]) + _rfma(
                float(wy[k - 1, pointr - 1]), a1, float(ws[k - 1, pointr - 1]) * a2
            )
        pointr = pointr % m + 1
    return 0


def _errclb(n, m, factr, lo, u, nbd, task):
    """lbfgsb.c ``errclb`` (:1745): validate inputs; returns
    ``(task, info, k)``."""
    info = 0
    k = 0
    if n <= 0:
        task = "ERROR: N .LE. 0"
    if m <= 0:
        task = "ERROR: M .LE. 0"
    if factr < 0.0:
        task = "ERROR: FACTR .LT. 0"
    for i in range(1, n + 1):
        if nbd[i - 1] < 0 or nbd[i - 1] > 3:
            task = "ERROR: INVALID NBD"
            info = -6
            k = i
        if nbd[i - 1] == 2 and float(lo[i - 1]) > float(u[i - 1]):
            task = "ERROR: NO FEASIBLE SOLUTION"
            info = -7
            k = i
    return task, info, k


def _formk(
    n,
    nsub,
    ind,
    nenter,
    ileave,
    indx2,
    iupdat,
    updatd,
    wn,
    wn1,
    m,
    ws,
    wy,
    sy,
    theta,
    col,
    head,
):
    """lbfgsb.c ``formk`` (:1800): form the LEL' factorization of the
    middle indefinite matrix K into ``wn`` (upper triangle), maintaining
    the inner-product table ``wn1``. Returns info (−1/−2 on Cholesky
    failure)."""
    if updatd:
        if iupdat > m:
            for jy in range(1, m):
                js = m + jy
                i2 = m - jy
                _dcopy(i2, wn1[jy : jy + i2, jy], wn1[jy - 1 : jy - 1 + i2, jy - 1])
                _dcopy(i2, wn1[js : js + i2, js], wn1[js - 1 : js - 1 + i2, js - 1])
                _dcopy(
                    m - 1,
                    wn1[m + 1 : m + 1 + (m - 1), jy],
                    wn1[m : m + (m - 1), jy - 1],
                )
        pbegin = 1
        pend = nsub
        dbegin = nsub + 1
        dend = n
        iy = col
        is_ = m + col
        ipntr = head + col - 1
        if ipntr > m:
            ipntr -= m
        jpntr = head
        for jy in range(1, col + 1):
            js = m + jy
            temp1 = 0.0
            temp2 = 0.0
            temp3 = 0.0
            for k in range(pbegin, pend + 1):
                k1 = int(ind[k - 1])
                temp1 = _rfma(
                    float(wy[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp1
                )
            for k in range(dbegin, dend + 1):
                k1 = int(ind[k - 1])
                temp2 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(ws[k1 - 1, jpntr - 1]), temp2
                )
                temp3 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp3
                )
            wn1[iy - 1, jy - 1] = temp1
            wn1[is_ - 1, js - 1] = temp2
            wn1[is_ - 1, jy - 1] = temp3
            jpntr = jpntr % m + 1
        jy = col
        jpntr = head + col - 1
        if jpntr > m:
            jpntr -= m
        ipntr = head
        for i in range(1, col + 1):
            is_ = m + i
            temp3 = 0.0
            for k in range(pbegin, pend + 1):
                k1 = int(ind[k - 1])
                temp3 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp3
                )
            ipntr = ipntr % m + 1
            wn1[is_ - 1, jy - 1] = temp3
        upcl = col - 1
    else:
        upcl = col
    ipntr = head
    for iy in range(1, upcl + 1):
        is_ = m + iy
        jpntr = head
        for jy in range(1, iy + 1):
            js = m + jy
            temp1 = 0.0
            temp2 = 0.0
            temp3 = 0.0
            temp4 = 0.0
            for k in range(1, nenter + 1):
                k1 = int(indx2[k - 1])
                temp1 = _rfma(
                    float(wy[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp1
                )
                temp2 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(ws[k1 - 1, jpntr - 1]), temp2
                )
            for k in range(ileave, n + 1):
                k1 = int(indx2[k - 1])
                temp3 = _rfma(
                    float(wy[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp3
                )
                temp4 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(ws[k1 - 1, jpntr - 1]), temp4
                )
            wn1[iy - 1, jy - 1] = float(wn1[iy - 1, jy - 1]) + temp1 - temp3
            wn1[is_ - 1, js - 1] = float(wn1[is_ - 1, js - 1]) - temp2 + temp4
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1
    ipntr = head
    for is_ in range(m + 1, m + upcl + 1):
        jpntr = head
        for jy in range(1, upcl + 1):
            temp1 = 0.0
            temp3 = 0.0
            for k in range(1, nenter + 1):
                k1 = int(indx2[k - 1])
                temp1 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp1
                )
            for k in range(ileave, n + 1):
                k1 = int(indx2[k - 1])
                temp3 = _rfma(
                    float(ws[k1 - 1, ipntr - 1]), float(wy[k1 - 1, jpntr - 1]), temp3
                )
            if is_ <= jy + m:
                wn1[is_ - 1, jy - 1] = float(wn1[is_ - 1, jy - 1]) + temp1 - temp3
            else:
                wn1[is_ - 1, jy - 1] = float(wn1[is_ - 1, jy - 1]) - temp1 + temp3
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1
    m2 = 2 * m
    for iy in range(1, col + 1):
        is_ = col + iy
        is1 = m + iy
        for jy in range(1, iy + 1):
            js = col + jy
            js1 = m + jy
            wn[jy - 1, iy - 1] = float(wn1[iy - 1, jy - 1]) / theta
            wn[js - 1, is_ - 1] = float(wn1[is1 - 1, js1 - 1]) * theta
        for jy in range(1, iy):
            wn[jy - 1, is_ - 1] = -float(wn1[is1 - 1, jy - 1])
        for jy in range(iy, col + 1):
            wn[jy - 1, is_ - 1] = float(wn1[is1 - 1, jy - 1])
        wn[iy - 1, iy - 1] = float(wn[iy - 1, iy - 1]) + float(sy[iy - 1, iy - 1])
    info = _dpofa(wn, col)
    if info != 0:
        return -1
    col2 = 2 * col
    for js in range(col + 1, col2 + 1):
        _dtrsl(wn, col, wn[0:col, js - 1], 11)
    for is_ in range(col + 1, col2 + 1):
        for js in range(is_, col2 + 1):
            wn[is_ - 1, js - 1] = float(wn[is_ - 1, js - 1]) + _ddot(
                col, wn[0:col, is_ - 1], wn[0:col, js - 1]
            )
    info = _dpofa(wn[col:m2, col:m2], col)
    if info != 0:
        return -2
    return 0


def _formt(m, wt, sy, ss, col, theta):
    """lbfgsb.c ``formt`` (:2143): T = theta*SS + L D^{-1} L' in the
    upper triangle of ``wt``, then Cholesky (J' upper). Returns info
    (−3 on failure)."""
    for j in range(1, col + 1):
        wt[0, j - 1] = theta * float(ss[0, j - 1])
    for i in range(2, col + 1):
        for j in range(i, col + 1):
            k1 = min(i, j) - 1
            ddum = 0.0
            for k in range(1, k1 + 1):
                ddum += (
                    float(sy[i - 1, k - 1])
                    * float(sy[j - 1, k - 1])
                    / float(sy[k - 1, k - 1])
                )
            wt[i - 1, j - 1] = _rfma(theta, float(ss[i - 1, j - 1]), ddum)
    info = _dpofa(wt, col)
    if info != 0:
        return -3
    return 0


def _freev(n, nfree, indx, indx2, iwhere, updatd, cnstnd, iprint, iter_):
    """lbfgsb.c ``freev`` (:2217): count entering/leaving variables and
    build the free/active index sets at the GCP. Returns
    ``(nfree, nenter, ileave, wrk)``."""
    nenter = 0
    ileave = n + 1
    if iter_ > 0 and cnstnd:
        for i in range(1, nfree + 1):
            k = int(indx[i - 1])
            if iwhere[k - 1] > 0:
                ileave -= 1
                indx2[ileave - 1] = k
                if iprint >= 100:
                    print(f"Variable {k} leaves the set of free variables")
        for i in range(nfree + 1, n + 1):
            k = int(indx[i - 1])
            if iwhere[k - 1] <= 0:
                nenter += 1
                indx2[nenter - 1] = k
                if iprint >= 100:
                    print(f"Variable {k} enters the set of free variables")
            if iprint >= 100:
                print(f"{n + 1 - ileave} variables leave; {nenter} variables enter")
    wrk = (ileave < n + 1) or (nenter > 0) or updatd
    nfree = 0
    iact = n + 1
    for i in range(1, n + 1):
        if iwhere[i - 1] <= 0:
            nfree += 1
            indx[nfree - 1] = i
        else:
            iact -= 1
            indx[iact - 1] = i
    if iprint >= 99:
        print(f"{nfree}  variables are free at GCP on iteration {iter_ + 1}")
    return nfree, nenter, ileave, wrk


def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """lbfgsb.c ``dcstep`` (:3237): MINPACK-2 safeguarded step update.
    Returns the updated ``(stx, fx, dx, sty, fy, dy, stp, brackt)``."""
    sgnd = dp * (dx / abs(dx))
    if fp > fx:
        theta = (fx - fp) * 3.0 / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        d1 = theta / s
        gamm = s * math.sqrt(_rfma(d1, d1, -(dx / s * (dp / s))))
        if stp < stx:
            gamm = -gamm
        p = gamm - dx + theta
        q = gamm - dx + gamm + dp
        r = p / q
        stpc = _rfma(r, stp - stx, stx)
        stpq = _rfma(dx / ((fx - fp) / (stp - stx) + dx) / 2.0, stp - stx, stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = (fx - fp) * 3.0 / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        d1 = theta / s
        gamm = s * math.sqrt(_rfma(d1, d1, -(dx / s * (dp / s))))
        if stp > stx:
            gamm = -gamm
        p = gamm - dp + theta
        q = gamm - dp + gamm + dx
        r = p / q
        stpc = _rfma(r, stx - stp, stp)
        stpq = _rfma(dp / (dp - dx), stx - stp, stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = (fx - fp) * 3.0 / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        t1 = theta / s
        d1 = _rfma(t1, t1, -(dx / s * (dp / s)))
        gamm = 0.0 if d1 < 0 else s * math.sqrt(d1)
        if stp > stx:
            gamm = -gamm
        p = gamm - dp + theta
        q = gamm + (dx - dp) + gamm
        r = p / q
        if r < 0.0 and gamm != 0.0:
            stpc = _rfma(r, stx - stp, stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = _rfma(dp / (dp - dx), stx - stp, stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            d1 = _rfma(sty - stp, 0.66, stp)
            if stp > stx:
                stpf = min(d1, stpf)
            else:
                stpf = max(d1, stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)
    else:
        if brackt:
            theta = (fp - fy) * 3.0 / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            d1 = theta / s
            gamm = s * math.sqrt(_rfma(d1, d1, -(dy / s * (dp / s))))
            if stp > sty:
                gamm = -gamm
            p = gamm - dp + theta
            q = gamm - dp + gamm + dy
            r = p / q
            stpc = _rfma(r, sty - stp, stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    stp = stpf
    return stx, fx, dx, sty, fy, dy, stp, brackt


def _dcsrch(f, g, stp, ftol, gtol, xtol, stpmin, stpmax, task, ls):
    """lbfgsb.c ``dcsrch`` (:2980): the More-Thuente line search
    (reverse communication). ``ls`` is the persistent-state dict (the C
    statics), reset when ``task == "START"``. Returns ``(stp, task)``."""
    if task.startswith("START"):
        if stp < stpmin:
            return stp, "ERROR: STP .LT. STPMIN"
        if stp > stpmax:
            return stp, "ERROR: STP .GT. STPMAX"
        if g >= 0.0:
            return stp, "ERROR: INITIAL G .GE. ZERO"
        if ftol < 0.0:
            return stp, "ERROR: FTOL .LT. ZERO"
        if gtol < 0.0:
            return stp, "ERROR: GTOL .LT. ZERO"
        if xtol < 0.0:
            return stp, "ERROR: XTOL .LT. ZERO"
        if stpmin < 0.0:
            return stp, "ERROR: STPMIN .LT. ZERO"
        if stpmax < stpmin:
            return stp, "ERROR: STPMAX .LT. STPMIN"
        ls.clear()
        ls["brackt"] = False
        ls["stage"] = 1
        ls["finit"] = f
        ls["ginit"] = g
        ls["gtest"] = ftol * g
        ls["width"] = stpmax - stpmin
        ls["width1"] = (stpmax - stpmin) / 0.5
        ls["stx"] = 0.0
        ls["fx"] = f
        ls["gx"] = g
        ls["sty"] = 0.0
        ls["fy"] = f
        ls["gy"] = g
        ls["stmin"] = 0.0
        ls["stmax"] = stp + stp * 4.0
        return stp, "FG"
    ftest = _rfma(stp, ls["gtest"], ls["finit"])
    if ls["stage"] == 1 and f <= ftest and g >= 0.0:
        ls["stage"] = 2
    if ls["brackt"] and (stp <= ls["stmin"] or stp >= ls["stmax"]):
        task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
    if ls["brackt"] and ls["stmax"] - ls["stmin"] <= xtol * ls["stmax"]:
        task = "WARNING: XTOL TEST SATISFIED"
    if stp == stpmax and f <= ftest and g <= ls["gtest"]:
        task = "WARNING: STP = STPMAX"
    if stp == stpmin and (f > ftest or g >= ls["gtest"]):
        task = "WARNING: STP = STPMIN"
    if f <= ftest and abs(g) <= gtol * (-ls["ginit"]):
        task = "CONVERGENCE"
    if task.startswith(("WARN", "CONV")):
        return stp, task
    if ls["stage"] == 1 and f <= ls["fx"] and f > ftest:
        fm = _rfma(-stp, ls["gtest"], f)
        fxm = _rfma(-ls["stx"], ls["gtest"], ls["fx"])
        fym = _rfma(-ls["sty"], ls["gtest"], ls["fy"])
        gm = g - ls["gtest"]
        gxm = ls["gx"] - ls["gtest"]
        gym = ls["gy"] - ls["gtest"]
        (ls["stx"], fxm, gxm, ls["sty"], fym, gym, stp, ls["brackt"]) = _dcstep(
            ls["stx"],
            fxm,
            gxm,
            ls["sty"],
            fym,
            gym,
            stp,
            fm,
            gm,
            ls["brackt"],
            ls["stmin"],
            ls["stmax"],
        )
        ls["fx"] = _rfma(ls["stx"], ls["gtest"], fxm)
        ls["fy"] = _rfma(ls["sty"], ls["gtest"], fym)
        ls["gx"] = gxm + ls["gtest"]
        ls["gy"] = gym + ls["gtest"]
    else:
        (
            ls["stx"],
            ls["fx"],
            ls["gx"],
            ls["sty"],
            ls["fy"],
            ls["gy"],
            stp,
            ls["brackt"],
        ) = _dcstep(
            ls["stx"],
            ls["fx"],
            ls["gx"],
            ls["sty"],
            ls["fy"],
            ls["gy"],
            stp,
            f,
            g,
            ls["brackt"],
            ls["stmin"],
            ls["stmax"],
        )
    if ls["brackt"]:
        if abs(ls["sty"] - ls["stx"]) >= ls["width1"] * 0.66:
            stp = ls["stx"] + (ls["sty"] - ls["stx"]) * 0.5
        ls["width1"] = ls["width"]
        ls["width"] = abs(ls["sty"] - ls["stx"])
    if ls["brackt"]:
        ls["stmin"] = min(ls["stx"], ls["sty"])
        ls["stmax"] = max(ls["stx"], ls["sty"])
    else:
        ls["stmin"] = _rfma(stp - ls["stx"], 1.1, stp)
        ls["stmax"] = _rfma(stp - ls["stx"], 4.0, stp)
    stp = max(stp, stpmin)
    stp = min(stp, stpmax)
    if (ls["brackt"] and (stp <= ls["stmin"] or stp >= ls["stmax"])) or (
        ls["brackt"] and ls["stmax"] - ls["stmin"] <= xtol * ls["stmax"]
    ):
        stp = ls["stx"]
    return stp, "FG"


def _lnsrlb(n, lo, u, nbd, x, f, g, d, r, t, z, task, st):
    """lbfgsb.c ``lnsrlb`` (:2425): the safeguarded line search driver.
    Scalars live in the mainlb state dict ``st`` (fold, gd, gdold, stp,
    dnorm, dtd, xstep, stpmx, ifun, iback, nfgv, info, csave); returns
    ``(f, task)``."""
    ftol = 0.001
    gtol = 0.9
    xtol = 0.1
    stpmin = 0.0
    if not task.startswith("FG_LN"):
        st["dtd"] = _ddot(n, d, d)
        st["dnorm"] = math.sqrt(st["dtd"])
        st["stpmx"] = 1e10
        if st["cnstnd"]:
            if st["iter"] == 0:
                st["stpmx"] = 1.0
            else:
                for i in range(1, n + 1):
                    a1 = float(d[i - 1])
                    if nbd[i - 1] != 0:
                        if a1 < 0.0 and nbd[i - 1] <= 2:
                            a2 = float(lo[i - 1]) - float(x[i - 1])
                            if a2 >= 0.0:
                                st["stpmx"] = 0.0
                            elif a1 * st["stpmx"] < a2:
                                st["stpmx"] = a2 / a1
                        elif a1 > 0.0 and nbd[i - 1] >= 2:
                            a2 = float(u[i - 1]) - float(x[i - 1])
                            if a2 <= 0.0:
                                st["stpmx"] = 0.0
                            elif a1 * st["stpmx"] > a2:
                                st["stpmx"] = a2 / a1
        if st["iter"] == 0 and not st["boxed"]:
            st["stp"] = min(1.0 / st["dnorm"], st["stpmx"])
        else:
            st["stp"] = 1.0
        _dcopy(n, x, t)
        _dcopy(n, g, r)
        st["fold"] = f
        st["ifun"] = 0
        st["iback"] = 0
        st["csave"] = "START"
    st["gd"] = _ddot(n, g, d)
    if st["ifun"] == 0:
        st["gdold"] = st["gd"]
        if st["gd"] >= 0.0:
            st["info"] = -4
            return f, task
    st["stp"], st["csave"] = _dcsrch(
        f,
        st["gd"],
        st["stp"],
        ftol,
        gtol,
        xtol,
        stpmin,
        st["stpmx"],
        st["csave"],
        st["dcsrch"],
    )
    st["xstep"] = st["stp"] * st["dnorm"]
    if not st["csave"].startswith("CONV") and not st["csave"].startswith("WARN"):
        task = "FG_LNSRCH"
        st["ifun"] += 1
        st["nfgv"] += 1
        st["iback"] = st["ifun"] - 1
        if st["stp"] == 1.0:
            _dcopy(n, z, x)
        else:
            for i in range(1, n + 1):
                x[i - 1] = _rfma(st["stp"], float(d[i - 1]), float(t[i - 1]))
    else:
        task = "NEW_X"
    return f, task


def _matupd(n, m, ws, wy, sy, ss, d, r, st):
    """lbfgsb.c ``matupd`` (:2562): store the new (s, y) pair in the
    ring buffers and refresh S'Y / S'S; updates itail/head/col/theta in
    ``st``."""
    if st["iupdat"] <= m:
        st["col"] = st["iupdat"]
        st["itail"] = (st["head"] + st["iupdat"] - 2) % m + 1
    else:
        st["itail"] = st["itail"] % m + 1
        st["head"] = st["head"] % m + 1
    _dcopy(n, d, ws[:, st["itail"] - 1])
    _dcopy(n, r, wy[:, st["itail"] - 1])
    st["theta"] = st["rr"] / st["dr"]
    if st["iupdat"] > m:
        for j in range(1, st["col"]):
            _dcopy(j, ss[1 : j + 1, j], ss[0:j, j - 1])
            i2 = st["col"] - j
            _dcopy(i2, sy[j : j + i2, j], sy[j - 1 : j - 1 + i2, j - 1])
    pointr = st["head"]
    col = st["col"]
    for j in range(1, col):
        sy[col - 1, j - 1] = _ddot(n, d, wy[:, pointr - 1])
        ss[j - 1, col - 1] = _ddot(n, ws[:, pointr - 1], d)
        pointr = pointr % m + 1
    if st["stp"] == 1.0:
        ss[col - 1, col - 1] = st["dtd"]
    else:
        ss[col - 1, col - 1] = st["stp"] * st["stp"] * st["dtd"]
    sy[col - 1, col - 1] = st["dr"]


def _projgr(n, lo, u, nbd, x, g):
    """lbfgsb.c ``projgr`` (:2663): infinity norm of the projected
    gradient."""
    sbgnrm = 0.0
    for i in range(n):
        gi = float(g[i])
        if nbd[i] != 0:
            if gi < 0.0:
                if nbd[i] >= 2:
                    d1 = float(x[i]) - float(u[i])
                    gi = max(gi, d1)
            else:
                if nbd[i] <= 2:
                    d1 = float(x[i]) - float(lo[i])
                    gi = min(gi, d1)
        sbgnrm = max(sbgnrm, abs(gi))
    return sbgnrm


def _subsm(n, m, nsub, ind, lo, u, nbd, x, d, ws, wy, theta, col, head, wv, wn, iprint):
    """lbfgsb.c ``subsm`` (:2708): direct subspace minimization over the
    free variables, backtracking into the box. Mutates ``x``/``d``;
    returns ``(iword, info)``."""
    if nsub <= 0:
        return 0, 0
    pointr = head
    for i in range(1, col + 1):
        temp1 = 0.0
        temp2 = 0.0
        for j in range(1, nsub + 1):
            k = int(ind[j - 1])
            temp1 = _rfma(float(wy[k - 1, pointr - 1]), float(d[j - 1]), temp1)
            temp2 = _rfma(float(ws[k - 1, pointr - 1]), float(d[j - 1]), temp2)
        wv[i - 1] = temp1
        wv[col + i - 1] = theta * temp2
        pointr = pointr % m + 1
    col2 = 2 * col
    info = _dtrsl(wn, col2, wv, 11)
    if info != 0:
        return 0, info
    for i in range(1, col + 1):
        wv[i - 1] = -float(wv[i - 1])
    info = _dtrsl(wn, col2, wv, 1)
    if info != 0:
        return 0, info
    pointr = head
    for jy in range(1, col + 1):
        js = col + jy
        for i in range(1, nsub + 1):
            k = int(ind[i - 1])
            d[i - 1] = float(d[i - 1]) + _rfma(
                float(ws[k - 1, pointr - 1]),
                float(wv[js - 1]),
                float(wy[k - 1, pointr - 1]) * float(wv[jy - 1]) / theta,
            )
        pointr = pointr % m + 1
    for i in range(1, nsub + 1):
        d[i - 1] = float(d[i - 1]) / theta
    alpha = 1.0
    temp1 = alpha
    ibd = 0
    for i in range(1, nsub + 1):
        k = int(ind[i - 1])
        dk = float(d[i - 1])
        if nbd[k - 1] != 0:
            if dk < 0.0 and nbd[k - 1] <= 2:
                temp2 = float(lo[k - 1]) - float(x[k - 1])
                if temp2 >= 0.0:
                    temp1 = 0.0
                elif dk * alpha < temp2:
                    temp1 = temp2 / dk
            elif dk > 0.0 and nbd[k - 1] >= 2:
                temp2 = float(u[k - 1]) - float(x[k - 1])
                if temp2 <= 0.0:
                    temp1 = 0.0
                elif dk * alpha > temp2:
                    temp1 = temp2 / dk
            if temp1 < alpha:
                alpha = temp1
                ibd = i
    if alpha < 1.0:
        dk = float(d[ibd - 1])
        k = int(ind[ibd - 1])
        if dk > 0.0:
            x[k - 1] = float(u[k - 1])
            d[ibd - 1] = 0.0
        elif dk < 0.0:
            x[k - 1] = float(lo[k - 1])
            d[ibd - 1] = 0.0
    for i in range(1, nsub + 1):
        x[int(ind[i - 1]) - 1] = _rfma(
            alpha, float(d[i - 1]), float(x[int(ind[i - 1]) - 1])
        )
    iword = 1 if alpha < 1.0 else 0
    return iword, 0


def _prn3lb(n, x, f, task, iprint, info, iter_, nfgv, nintol, nskip, nact, sbgnrm, k):
    """lbfgsb.c ``prn3lb`` (:3546): final report (iprint >= 0 only)."""
    if task.startswith("CONV"):
        if iprint >= 0:
            print(
                f"\niterations {iter_}\nfunction evaluations {nfgv}\n"
                f"segments explored during Cauchy searches {nintol}\n"
                f"BFGS updates skipped {nskip}\n"
                f"active bounds at final generalized Cauchy point "
                f"{nact}\n"
                f"norm of the final projected gradient {sbgnrm:g}\n"
                f"final function value {f:g}\n"
            )
        if iprint >= 100:
            print("X =", np.asarray(x[:n]))
        if iprint >= 1:
            print(f"F = {f:g}")
    if iprint >= 0:
        msgs = {
            -1: "Matrix in 1st Cholesky factorization in formk is not Pos. Def.",
            -2: "Matrix in 2st Cholesky factorization in formk is not Pos. Def.",
            -3: "Matrix in the Cholesky factorization in formt is not Pos. Def.",
            -4: "Derivative >= 0, backtracking line search impossible.",
            -5: "Warning:  more than 10 function and gradient "
            "evaluations\n   in the last line search",
            -6: f"Input nbd({k}) is invalid",
            -7: f"l({k}) > u({k}).  No feasible solution",
            -8: "The triangular system is singular.",
            -9: "Line search cannot locate an adequate point after 20 "
            "function\nand gradient evaluations",
        }
        if info in msgs:
            print(msgs[info])


def _mainlb(n, m, x, lo, u, nbd, f, g, factr, pgtol, task, iprint, st):
    """lbfgsb.c ``mainlb`` (:358): the main reverse-communication loop.
    ``st`` carries the C statics and the workspace arrays; returns
    ``(f, task)`` (``x``/``g`` mutated in place)."""
    ws = st["ws"]
    wy = st["wy"]
    sy = st["sy"]
    ss = st["ss"]
    wt = st["wt"]
    wn = st["wn"]
    snd = st["snd"]
    z = st["z"]
    r = st["r"]
    d = st["d"]
    t = st["t"]
    wa = st["wa"]
    indx = st["indx"]
    iwhere = st["iwhere"]
    indx2 = st["indx2"]
    k = 0
    jump = None
    if task.startswith("START"):
        st["epsmch"] = _DBL_EPSILON
        st["fold"] = 0.0
        st["dnorm"] = 0.0
        st["gd"] = 0.0
        st["sbgnrm"] = 0.0
        st["stp"] = 0.0
        st["xstep"] = 0.0
        st["stpmx"] = 0.0
        st["gdold"] = 0.0
        st["dtd"] = 0.0
        st["col"] = 0
        st["head"] = 1
        st["theta"] = 1.0
        st["iupdat"] = 0
        st["updatd"] = False
        st["iback"] = 0
        st["itail"] = 0
        st["ifun"] = 0
        st["iword"] = 0
        st["nact"] = 0
        st["ileave"] = 0
        st["nenter"] = 0
        st["iter"] = 0
        st["nfgv"] = 0
        st["nint"] = 0
        st["nintol"] = 0
        st["nskip"] = 0
        st["nfree"] = n
        st["tol"] = factr * st["epsmch"]
        st["info"] = 0
        st["dcsrch"] = {}
        st["csave"] = ""
        task, st["info"], k = _errclb(n, m, factr, lo, u, nbd, task)
        if task.startswith("ERROR"):
            _prn3lb(
                n,
                x,
                f,
                task,
                iprint,
                st["info"],
                st["iter"],
                st["nfgv"],
                st["nintol"],
                st["nskip"],
                st["nact"],
                st["sbgnrm"],
                k,
            )
            return f, task
        if iprint >= 0:
            print(f"N = {n}, M = {m} machine precision = {st['epsmch']:g}")
        _active(n, lo, u, nbd, x, iwhere, iprint, st)
        task = "FG_START"
        jump = "L1000"
    else:
        if task.startswith("FG_LN"):
            jump = "L666"
        elif task.startswith("NEW_X"):
            jump = "L777"
        elif task.startswith("FG_ST"):
            jump = "L111"
        elif task.startswith("STOP"):
            if task[6:9] == "CPU":
                _dcopy(n, t, x)
                _dcopy(n, r, g)
                f = st["fold"]
            jump = "L999"
        else:
            task = "FG_START"
            jump = "L1000"
    wrk = False
    while True:
        if jump == "L1000" or jump == "L999":
            break
        if jump == "L111":
            st["nfgv"] = 1
            st["sbgnrm"] = _projgr(n, lo, u, nbd, x, g)
            if iprint >= 1:
                print(
                    f"At iterate {st['iter']:5d}  f= {f:12.5g}  "
                    f"|proj g|= {st['sbgnrm']:12.5g}"
                )
            if st["sbgnrm"] <= pgtol:
                task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
                break
            jump = "L222"
            continue
        if jump == "L222":
            if iprint >= 99:
                print(f"Iteration {st['iter']}")
            st["iword"] = -1
            if not st["cnstnd"] and st["col"] > 0:
                _dcopy(n, x, z)
                wrk = st["updatd"]
                st["nint"] = 0
                jump = "L333"
                continue
            st["nint"], st["info"] = _cauchy(
                n,
                x,
                lo,
                u,
                nbd,
                g,
                indx2,
                iwhere,
                t,
                d,
                z,
                m,
                wy,
                ws,
                sy,
                wt,
                st["theta"],
                st["col"],
                st["head"],
                wa,
                iprint,
                st["sbgnrm"],
                st["epsmch"],
            )
            if st["info"] != 0:
                if iprint >= 1:
                    print(
                        "Singular triangular system detected;\n"
                        "   refresh the lbfgs memory and restart the "
                        "iteration."
                    )
                st["info"] = 0
                st["col"] = 0
                st["head"] = 1
                st["theta"] = 1.0
                st["iupdat"] = 0
                st["updatd"] = False
                jump = "L222"
                continue
            st["nintol"] += st["nint"]
            st["nfree"], st["nenter"], st["ileave"], wrk = _freev(
                n,
                st["nfree"],
                indx,
                indx2,
                iwhere,
                st["updatd"],
                st["cnstnd"],
                iprint,
                st["iter"],
            )
            st["nact"] = n - st["nfree"]
            jump = "L333"
            continue
        if jump == "L333":
            if st["nfree"] == 0 or st["col"] == 0:
                jump = "L555"
                continue
            if wrk:
                st["info"] = _formk(
                    n,
                    st["nfree"],
                    indx,
                    st["nenter"],
                    st["ileave"],
                    indx2,
                    st["iupdat"],
                    st["updatd"],
                    wn,
                    snd,
                    m,
                    ws,
                    wy,
                    sy,
                    st["theta"],
                    st["col"],
                    st["head"],
                )
            if st["info"] != 0:
                if iprint >= 0:
                    print(
                        "Nonpositive definiteness in Cholesky "
                        "factorization in formk;\n   refresh the "
                        "lbfgs memory and restart the iteration."
                    )
                st["info"] = 0
                st["col"] = 0
                st["head"] = 1
                st["theta"] = 1.0
                st["iupdat"] = 0
                st["updatd"] = False
                jump = "L222"
                continue
            st["info"] = _cmprlb(
                n,
                m,
                x,
                g,
                ws,
                wy,
                sy,
                wt,
                z,
                r,
                wa,
                indx,
                st["theta"],
                st["col"],
                st["head"],
                st["nfree"],
                st["cnstnd"],
            )
            if st["info"] == 0:
                st["iword"], st["info"] = _subsm(
                    n,
                    m,
                    st["nfree"],
                    indx,
                    lo,
                    u,
                    nbd,
                    z,
                    r,
                    ws,
                    wy,
                    st["theta"],
                    st["col"],
                    st["head"],
                    wa,
                    wn,
                    iprint,
                )
            if st["info"] != 0:
                if iprint >= 1:
                    print(
                        "Singular triangular system detected;\n"
                        "   refresh the lbfgs memory and restart the "
                        "iteration."
                    )
                st["info"] = 0
                st["col"] = 0
                st["head"] = 1
                st["theta"] = 1.0
                st["iupdat"] = 0
                st["updatd"] = False
                jump = "L222"
                continue
            jump = "L555"
            continue
        if jump == "L555":
            for i in range(1, n + 1):
                d[i - 1] = float(z[i - 1]) - float(x[i - 1])
            jump = "L666"
            continue
        if jump == "L666":
            f, task = _lnsrlb(n, lo, u, nbd, x, f, g, d, r, t, z, task, st)
            if st["info"] != 0 or st["iback"] >= 20:
                _dcopy(n, t, x)
                _dcopy(n, r, g)
                f = st["fold"]
                if st["col"] == 0:
                    if st["info"] == 0:
                        st["info"] = -9
                        st["nfgv"] -= 1
                        st["ifun"] -= 1
                        st["iback"] -= 1
                    task = "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH"
                    st["iter"] += 1
                    break
                else:
                    if iprint >= 1:
                        print(
                            "Bad direction in the line search;\n"
                            "   refresh the lbfgs memory and restart "
                            "the iteration."
                        )
                    if st["info"] == 0:
                        st["nfgv"] -= 1
                    st["info"] = 0
                    st["col"] = 0
                    st["head"] = 1
                    st["theta"] = 1.0
                    st["iupdat"] = 0
                    st["updatd"] = False
                    task = "RESTART_FROM_LNSRCH"
                    jump = "L222"
                    continue
            elif task.startswith("FG_LN"):
                break
            else:
                st["iter"] += 1
                st["sbgnrm"] = _projgr(n, lo, u, nbd, x, g)
                if iprint > 0 and st["iter"] % iprint == 0:
                    print(
                        f"At iterate {st['iter']:5d}  f = {f:12.5g}  "
                        f"|proj g|=  {st['sbgnrm']:12.5g}"
                    )
                break
        if jump == "L777":
            if st["sbgnrm"] <= pgtol:
                task = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
                break
            ddum = max(abs(st["fold"]), abs(f), 1.0)
            if st["fold"] - f <= st["tol"] * ddum:
                task = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH"
                if st["iback"] >= 10:
                    st["info"] = -5
                break
            for i in range(1, n + 1):
                r[i - 1] = float(g[i - 1]) - float(r[i - 1])
            st["rr"] = _ddot(n, r, r)
            if st["stp"] == 1.0:
                st["dr"] = st["gd"] - st["gdold"]
                ddum = -st["gdold"]
            else:
                st["dr"] = (st["gd"] - st["gdold"]) * st["stp"]
                _dscal(n, st["stp"], d)
                ddum = -st["gdold"] * st["stp"]
            if st["dr"] <= st["epsmch"] * ddum:
                st["nskip"] += 1
                st["updatd"] = False
                if iprint >= 1:
                    print(f"ys={st['dr']:10.3e}  -gs={ddum:10.3e}, BFGS update SKIPPED")
                jump = "L222"
                continue
            st["updatd"] = True
            st["iupdat"] += 1
            _matupd(n, m, ws, wy, sy, ss, d, r, st)
            st["info"] = _formt(m, wt, sy, ss, st["col"], st["theta"])
            if st["info"] != 0:
                if iprint >= 0:
                    print(
                        "Nonpositive definiteness in Cholesky "
                        "factorization in formt();\n   refresh the "
                        "lbfgs memory and restart the iteration."
                    )
                st["info"] = 0
                st["col"] = 0
                st["head"] = 1
                st["theta"] = 1.0
                st["iupdat"] = 0
                st["updatd"] = False
            jump = "L222"
            continue
        break
    st["isave13"] = st["nfgv"]
    _prn3lb(
        n,
        x,
        f,
        task,
        iprint,
        st["info"],
        st["iter"],
        st["nfgv"],
        st["nintol"],
        st["nskip"],
        st["nact"],
        st["sbgnrm"],
        k,
    )
    return f, task


def setulb(n, m, x, lo, u, nbd, f, g, factr, pgtol, task, iprint, state):
    """lbfgsb.c ``setulb`` (:133): allocate/partition the workspace on
    ``task == "START"`` and call :func:`_mainlb`. ``state`` plays the
    role of the C statics + the caller's ``wa``/``iwa``/``isave``;
    returns ``(f, task)``."""
    if task.startswith("START"):
        state.clear()
        state["ws"] = np.zeros((n, m))
        state["wy"] = np.zeros((n, m))
        state["sy"] = np.zeros((m, m))
        state["ss"] = np.zeros((m, m))
        state["wt"] = np.zeros((m, m))
        state["wn"] = np.zeros((2 * m, 2 * m))
        state["snd"] = np.zeros((2 * m, 2 * m))
        state["z"] = np.zeros(n)
        state["r"] = np.zeros(n)
        state["d"] = np.zeros(n)
        state["t"] = np.zeros(n)
        state["wa"] = np.zeros(8 * m)
        state["indx"] = np.zeros(n, dtype=np.int64)
        state["iwhere"] = np.zeros(n, dtype=np.int64)
        state["indx2"] = np.zeros(n, dtype=np.int64)
        state["isave13"] = 0
    return _mainlb(n, m, x, lo, u, nbd, f, g, factr, pgtol, task, iprint, state)


def lbfgsb(n, m, x, lo, u, nbd, fminfn, fmingr, factr, pgtol, maxit, trace, nREPORT):
    """R's ``lbfgsb()`` driver (src/appl/optim.c:642): the reverse-
    communication loop around :func:`setulb`. ``fminfn(x) -> float`` and
    ``fmingr(x, g) -> None`` are the (already fnscale/parscale-wrapped)
    objective and gradient. Mutates ``x``; returns ``(Fmin, fail,
    fncount, grcount, msg)``."""
    if n == 0:
        return fminfn(u), 0, 1, 0, "NOTHING TO DO"
    if nREPORT <= 0:
        raise ValueError('REPORT must be > 0 (method = "L-BFGS-B")')
    tr = {2: 0, 3: nREPORT, 4: 99, 5: 100, 6: 101}.get(trace, -1)
    fail = 0
    g = np.zeros(n)
    state = {}
    task = "START"
    f = 0.0
    iter_ = 0
    while True:
        f, task = setulb(n, m, x, lo, u, nbd, f, g, factr, pgtol, task, tr, state)
        if task.startswith("FG"):
            f = fminfn(x)
            if not np.isfinite(f):
                raise ValueError("L-BFGS-B needs finite values of 'fn'")
            fmingr(x, g)
        elif task.startswith("NEW_X"):
            iter_ += 1
            if trace == 1 and iter_ % nREPORT == 0:
                print(f"iter {iter_:4d} value {f:f}")
            if iter_ > maxit:
                fail = 1
                break
        elif task.startswith("WARN"):
            fail = 51
            break
        elif task.startswith("CONV"):
            break
        elif task.startswith("ERROR"):
            fail = 52
            break
        else:
            fail = 52
            break
    Fmin = f
    fncount = grcount = state.get("isave13", 0)
    if trace:
        print(f"final  value {Fmin:f} ")
        if iter_ < maxit and fail == 0:
            print("converged")
        else:
            print(f"stopped after {iter_} iterations")
    return Fmin, fail, fncount, grcount, task
