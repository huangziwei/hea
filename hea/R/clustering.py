"""hea.R.clustering — base-R ``stats`` hierarchical clustering.

Mechanical port of R's ``src/library/stats/R/hclust.R`` and its Fortran core
``src/library/stats/src/hclust.f`` (``HCLUST`` + ``HCASS2`` + ``IOFFST``). This
module owns the clustering *algorithms* and tree objects; it imports the
metric-space layer (``Dist``/``as_dist``/``as_matrix_dist``) one-way from
:mod:`hea.R.distance` (acyclic).

The Fortran is ported literally — **1-based indexing preserved** (arrays are
allocated length ``n+1`` and index ``0`` is unused) so the control flow, the
nearest-neighbour-chain scan order, and the tie-breaking match the source
exactly. That order is load-bearing for bit-exact ``$merge``/``$height``/
``$order`` against ``stats::hclust``, and it is inherently sequential — the Rust
end-goal kernel (plan step 10) must mirror it and must **not** be parallelized.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ._dispatch import rs_fn
from .distance import Dist, _pmatch, as_dist
from .distributions import _r_rng

__all__ = [
    "Hclust",
    "Kmeans",
    "hclust",
    "as_hclust",
    "cophenetic",
    "cutree",
    "fitted_kmeans",
    "kmeans",
    "print_hclust",
    "print_kmeans",
]

# order --> i.meth --> Fortran iOpt codes (1..8)
_METHODS = ("ward.D", "single", "complete", "average", "mcquitty",
            "median", "centroid", "ward.D2")

# Rust seam (plan build-order step 10): a serial ``hclust`` kernel mirroring
# ``_hclust_fortran``/``_hcass2`` 1:1. Inherently sequential (NN-chain
# agglomeration with data-dependent tie-breaking) — NEVER parallelize. ``None``
# until built ⇒ the pure-Python port below is the spec/oracle.
_rs_hclust = rs_fn("hclust")

# Rust seam (step 10): the Hartigan-Wong kernel (`_kmns`/`_optra`/`_qtran`).
# Sequential OPTRA/QTRAN transfer loops — NEVER parallelize.
_rs_kmns = rs_fn("kmns")


# --------------------------------------------------------------------------- #
# Fortran core (hclust.f), ported 1:1 with 1-based indexing
# --------------------------------------------------------------------------- #
def _hclust_fortran(n, diss, iopt, membr0):
    """Port of ``SUBROUTINE HCLUST``. ``diss`` is R's packed lower-triangle
    vector (0-based, length ``n*(n-1)/2``); ``iopt`` is the 1-based method code;
    ``membr0`` is the length-``n`` members vector. Returns ``(ia, ib, crit)`` as
    1-based lists (entries ``1..n-1`` used)."""
    inf = 1.0e300
    length = n * (n - 1) // 2

    # 1-based working arrays (index 0 unused), mirroring the Fortran declarations.
    d = np.empty(length + 1, dtype=float)
    d[1:length + 1] = diss
    ia = [0] * (n + 1)
    ib = [0] * (n + 1)
    crit = [0.0] * (n + 1)
    membr = [0.0] * (n + 1)
    for i in range(1, n + 1):
        membr[i] = float(membr0[i - 1])
    nn = [0] * (n + 1)
    disnn = [0.0] * (n + 1)
    flag = [False] * (n + 1)

    def ioffst(i, j):
        # map row i < col j of the symmetric matrix onto the packed vector
        return j + (i - 1) * n - (i * (i + 1)) // 2

    im = jj = jm = 0  # persistent locals (carry across iterations, as in Fortran)

    for i in range(1, n + 1):
        flag[i] = True
    ncl = n

    isward = (iopt == 1 or iopt == 8)
    if iopt == 8:  # Ward "D2": use squared distances
        for i in range(1, length + 1):
            d[i] = d[i] * d[i]

    # initial nearest-neighbour list (NN to the RIGHT of i)
    for i in range(1, n):
        dmin = inf
        for j in range(i + 1, n + 1):
            ind = ioffst(i, j)
            if dmin > d[ind]:
                dmin = d[ind]
                jm = j
        nn[i] = jm
        disnn[i] = dmin

    while True:  # 400 CONTINUE
        # least dissimilarity among the current NNs
        dmin = inf
        for i in range(1, n):
            if flag[i] and disnn[i] < dmin:
                dmin = disnn[i]
                im = i
                jm = nn[i]
        ncl -= 1

        i2 = min(im, jm)
        j2 = max(im, jm)
        ia[n - ncl] = i2
        ib[n - ncl] = j2
        if iopt == 8:
            dmin = math.sqrt(dmin)
        crit[n - ncl] = dmin
        flag[j2] = False

        # update dissimilarities from the new cluster
        dmin = inf
        for k in range(1, n + 1):
            if flag[k] and k != i2:
                ind1 = ioffst(i2, k) if i2 < k else ioffst(k, i2)
                ind2 = ioffst(j2, k) if j2 < k else ioffst(k, j2)
                d12 = d[ioffst(i2, j2)]

                if isward:
                    d[ind1] = ((membr[i2] + membr[k]) * d[ind1]
                               + (membr[j2] + membr[k]) * d[ind2]
                               - membr[k] * d12)
                    d[ind1] = d[ind1] / (membr[i2] + membr[j2] + membr[k])
                elif iopt == 2:  # single
                    d[ind1] = min(d[ind1], d[ind2])
                elif iopt == 3:  # complete
                    d[ind1] = max(d[ind1], d[ind2])
                elif iopt == 4:  # average (UPGMA)
                    d[ind1] = ((membr[i2] * d[ind1] + membr[j2] * d[ind2])
                               / (membr[i2] + membr[j2]))
                elif iopt == 5:  # mcquitty (WPGMA)
                    d[ind1] = (d[ind1] + d[ind2]) / 2
                elif iopt == 6:  # median (WPGMC)
                    d[ind1] = ((d[ind1] + d[ind2]) - d12 / 2) / 2
                elif iopt == 7:  # centroid (UPGMC)
                    d[ind1] = ((membr[i2] * d[ind1] + membr[j2] * d[ind2]
                                - membr[i2] * membr[j2] * d12
                                / (membr[i2] + membr[j2]))
                               / (membr[i2] + membr[j2]))

                if i2 < k:
                    if d[ind1] < dmin:
                        dmin = d[ind1]
                        jj = k
                else:  # i2 > k: keep correct NNs for non-monotone methods
                    if d[ind1] < disnn[k]:
                        disnn[k] = d[ind1]
                        nn[k] = i2
        membr[i2] = membr[i2] + membr[j2]
        disnn[i2] = dmin
        nn[i2] = jj

        # update the NN list where it pointed at the merged pair
        for i in range(1, n):
            if flag[i] and (nn[i] == i2 or nn[i] == j2):
                dmin = inf
                for j in range(i + 1, n + 1):
                    if flag[j]:
                        ind = ioffst(i, j)
                        if d[ind] < dmin:
                            dmin = d[ind]
                            jj = j
                nn[i] = jj
                disnn[i] = dmin

        if ncl > 1:
            continue
        break

    return ia, ib, crit


def _hcass2(n, ia, ib):
    """Port of ``SUBROUTINE HCASS2``: turn the agglomeration sequence into R's
    ``merge`` columns (``iia``/``iib``) and the leaf ``order``. ``ia``/``ib`` are
    1-based lists from :func:`_hclust_fortran`. Returns ``(iorder, iia, iib)``
    (all 1-based)."""
    iorder = [0] * (n + 1)
    iia = [0] * (n + 1)
    iib = [0] * (n + 1)
    for i in range(1, n + 1):
        iia[i] = ia[i]
        iib[i] = ib[i]
    for i in range(1, n - 1):  # 1..n-2
        k = min(ia[i], ib[i])  # smallest (+ve or -ve) seq. no.
        for j in range(i + 1, n):  # i+1..n-1
            if ia[j] == k:
                iia[j] = -i
            if ib[j] == k:
                iib[j] = -i
    for i in range(1, n):  # 1..n-1
        iia[i] = -iia[i]
        iib[i] = -iib[i]
    for i in range(1, n):
        if iia[i] > 0 and iib[i] < 0:
            k = iia[i]
            iia[i] = iib[i]
            iib[i] = k
        if iia[i] > 0 and iib[i] > 0:
            k1 = min(iia[i], iib[i])
            k2 = max(iia[i], iib[i])
            iia[i] = k1
            iib[i] = k2

    iorder[1] = iia[n - 1]
    iorder[2] = iib[n - 1]
    loc = 2
    for i in range(n - 2, 0, -1):  # N-2 .. 1
        for j in range(1, loc + 1):
            if iorder[j] == i:
                iorder[j] = iia[i]
                if j == loc:
                    loc += 1
                    iorder[loc] = iib[i]
                else:
                    loc += 1
                    for kk in range(loc, j + 1, -1):  # LOC .. J+2 step -1
                        iorder[kk] = iorder[kk - 1]
                    iorder[j + 1] = iib[i]
                break  # GOTO 171
    for i in range(1, n + 1):
        iorder[i] = -iorder[i]
    return iorder, iia, iib


def _cutree_c(merge, which):
    """Port of ``SEXP cutree`` (``src/library/stats/src/hclust-utils.c``):
    grouping vectors from cutting the tree into ``which[j]`` groups. ``merge`` is
    the ``(n-1, 2)`` R-convention matrix; ``which`` is the 1-D vector of group
    counts. Returns an ``n x len(which)`` int matrix (column ``j`` = the labels
    for ``which[j]`` clusters)."""
    merge = np.asarray(merge, dtype=np.int64).reshape(-1, 2)
    which = np.asarray(which, dtype=np.int64)
    n = merge.shape[0] + 1
    nw = which.size
    col1 = merge[:, 0]
    col2 = merge[:, 1]

    # 1-based working arrays (index 0 unused), as in the C "--" pointers.
    sing = [True] * (n + 1)   # is k-th obs still alone in a cluster?
    m_nr = [0] * (n + 1)      # last merge-step number containing k-th obs
    z = [0] * (n + 1)
    ans = np.zeros((n, nw), dtype=np.int64)

    first_col = -1
    for k in range(1, n):  # k-th merge
        m1 = int(col1[k - 1])
        m2 = int(col2[k - 1])
        if m1 < 0 and m2 < 0:  # merging atoms [-m1] and [-m2]
            m_nr[-m1] = k
            m_nr[-m2] = k
            sing[-m1] = False
            sing[-m2] = False
        elif m1 < 0 or m2 < 0:  # one atom, one cluster
            if m1 < 0:
                j = -m1
                m1 = m2
            else:
                j = -m2
            for ell in range(1, n + 1):
                if m_nr[ell] == m1:
                    m_nr[ell] = k
            m_nr[j] = k
            sing[j] = False
        else:  # both clusters
            for ell in range(1, n + 1):
                if m_nr[ell] == m1 or m_nr[ell] == m2:
                    m_nr[ell] = k

        # does this merge leave a requested number of groups (n - k)?
        found_j = False
        for j in range(nw):
            if which[j] == n - k:
                if not found_j:
                    found_j = True
                    for ell in range(1, n + 1):
                        z[ell] = 0
                    nclust = 0
                    first_col = j
                    for ell in range(1, n + 1):
                        if sing[ell]:
                            nclust += 1
                            ans[ell - 1, j] = nclust
                        else:
                            if z[m_nr[ell]] == 0:
                                nclust += 1
                                z[m_nr[ell]] = nclust
                            ans[ell - 1, j] = z[m_nr[ell]]
                else:  # duplicate group count: copy the already-built column
                    ans[:, j] = ans[:, first_col]

    for j in range(nw):  # trivial case which[] == n
        if which[j] == n:
            ans[:, j] = np.arange(1, n + 1)
    return ans


# --------------------------------------------------------------------------- #
# kmeans kernels — Lloyd/MacQueen (cluster_kmeans.c), Hartigan-Wong (kmns.f)
# --------------------------------------------------------------------------- #
def _kmeans_lloyd(x, centers, k, maxiter):
    """Port of ``kmeans_Lloyd`` (``cluster_kmeans.c``). ``x`` is ``(n, p)``,
    ``centers`` ``(k, p)``; returns ``(cl, cen, nc, wss, iter)`` with ``cl``
    1-based labels."""
    n, p = x.shape
    cen = np.array(centers, dtype=float)
    cl = np.full(n, -1, dtype=np.int64)
    nc = np.zeros(k, dtype=np.int64)
    broke = False
    iteration = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for iteration in range(maxiter):
            updated = False
            for i in range(n):
                best = np.inf
                inew = 0
                for j in range(k):
                    dd = 0.0
                    for c in range(p):
                        tmp = x[i, c] - cen[j, c]
                        dd += tmp * tmp
                    if dd < best:
                        best = dd
                        inew = j + 1
                if cl[i] != inew:
                    updated = True
                    cl[i] = inew
            if not updated:
                broke = True
                break
            cen[:] = 0.0
            nc[:] = 0
            for i in range(n):
                it = cl[i] - 1
                nc[it] += 1
                cen[it] += x[i]
            for j in range(k):
                cen[j] /= nc[j]
    c_iter = iteration if broke else maxiter
    wss = _kmeans_wss(x, cen, cl, k)
    return cl, cen, nc, wss, c_iter + 1


def _kmeans_macqueen(x, centers, k, maxiter):
    """Port of ``kmeans_MacQueen`` (``cluster_kmeans.c``): assign + centroid, then
    incremental running-mean updates."""
    n, p = x.shape
    cen = np.array(centers, dtype=float)
    cl = np.zeros(n, dtype=np.int64)
    nc = np.zeros(k, dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        # initial nearest-centre assignment
        for i in range(n):
            best = np.inf
            inew = 0
            for j in range(k):
                dd = 0.0
                for c in range(p):
                    tmp = x[i, c] - cen[j, c]
                    dd += tmp * tmp
                if dd < best:
                    best = dd
                    inew = j + 1
            if cl[i] != inew:
                cl[i] = inew
        # centroids
        cen[:] = 0.0
        nc[:] = 0
        for i in range(n):
            it = cl[i] - 1
            nc[it] += 1
            cen[it] += x[i]
        for j in range(k):
            cen[j] /= nc[j]
        # incremental refinement
        broke = False
        iteration = 0
        for iteration in range(maxiter):
            updated = False
            for i in range(n):
                best = np.inf
                inew = 0
                for j in range(k):
                    dd = 0.0
                    for c in range(p):
                        tmp = x[i, c] - cen[j, c]
                        dd += tmp * tmp
                    if dd < best:
                        best = dd
                        inew = j  # 0-based here, as in the C
                iold = cl[i] - 1
                if iold != inew:
                    updated = True
                    cl[i] = inew + 1
                    nc[iold] -= 1
                    nc[inew] += 1
                    for c in range(p):
                        cen[iold, c] += (cen[iold, c] - x[i, c]) / nc[iold]
                        cen[inew, c] += (x[i, c] - cen[inew, c]) / nc[inew]
            if not updated:
                broke = True
                break
    c_iter = iteration if broke else maxiter
    wss = _kmeans_wss(x, cen, cl, k)
    return cl, cen, nc, wss, c_iter + 1


def _kmeans_wss(x, cen, cl, k):
    """Per-cluster within sum-of-squares (the tail shared by Lloyd/MacQueen)."""
    n, p = x.shape
    wss = np.zeros(k)
    for i in range(n):
        it = cl[i] - 1
        for c in range(p):
            tmp = x[i, c] - cen[it, c]
            wss[it] += tmp * tmp
    return wss


def _kmns(x, centers, k, iter_max, trace=0):
    """Port of ``SUBROUTINE KMNS`` + ``OPTRA`` + ``QTRAN`` (``kmns.f``), the
    Hartigan-Wong algorithm. 1-based bookkeeping arrays mirror the Fortran. The
    nested ``_optra``/``_qtran`` close over the shared state and mutate it in
    place, exactly as the Fortran subroutines mutate their by-reference args.
    Returns a dict ``{cluster, centers, nc, wss, iter, ifault}``."""
    big = 1.0e30
    m, p = x.shape
    cen = np.array(centers, dtype=float)
    imaxqtr = min(2147483647, 50 * m)

    ifault = 3
    if k <= 1 or k >= m:
        return {"ifault": 3}
    ifault = 0

    # 1-based working arrays (index 0 unused).
    ic1 = [0] * (m + 1)
    ic2 = [0] * (m + 1)
    d = [0.0] * (m + 1)
    nc = [0] * (k + 1)
    ncp = [0] * (k + 1)
    an1 = [0.0] * (k + 1)
    an2 = [0.0] * (k + 1)
    live = [0] * (k + 1)
    itran = [0] * (k + 1)

    # For each point, its two closest centres IC1, IC2.
    for i in range(1, m + 1):
        ic1[i] = 1
        ic2[i] = 2
        dt = [0.0, 0.0]
        for il in (1, 2):
            dt[il - 1] = 0.0
            for j in range(1, p + 1):
                da = x[i - 1, j - 1] - cen[il - 1, j - 1]
                dt[il - 1] += da * da
        if dt[0] > dt[1]:
            ic1[i] = 2
            ic2[i] = 1
            dt[0], dt[1] = dt[1], dt[0]
        for ell in range(3, k + 1):
            db = 0.0
            skip = False
            for j in range(1, p + 1):
                dc = x[i - 1, j - 1] - cen[ell - 1, j - 1]
                db += dc * dc
                if db >= dt[1]:
                    skip = True
                    break
            if skip:
                continue
            if db >= dt[0]:
                dt[1] = db
                ic2[i] = ell
            else:
                dt[1] = dt[0]
                ic2[i] = ic1[i]
                dt[0] = db
                ic1[i] = ell

    # Update centres to the mean of their members; cluster sizes NC.
    for ell in range(1, k + 1):
        nc[ell] = 0
        for j in range(1, p + 1):
            cen[ell - 1, j - 1] = 0.0
    for i in range(1, m + 1):
        ell = ic1[i]
        nc[ell] += 1
        for j in range(1, p + 1):
            cen[ell - 1, j - 1] += x[i - 1, j - 1]
    for ell in range(1, k + 1):
        if nc[ell] == 0:
            return {"ifault": 1}
        aa = float(nc[ell])
        for j in range(1, p + 1):
            cen[ell - 1, j - 1] /= aa
        an2[ell] = aa / (aa + 1.0)
        an1[ell] = big
        if aa > 1.0:
            an1[ell] = aa / (aa - 1.0)
        itran[ell] = 1
        ncp[ell] = -1

    def _optra(indx):
        for ell in range(1, k + 1):
            if itran[ell] == 1:
                live[ell] = m + 1
        for i in range(1, m + 1):
            indx += 1
            l1 = ic1[i]
            l2 = ic2[i]
            ll = l2
            if nc[l1] != 1:
                if ncp[l1] != 0:
                    de = 0.0
                    for j in range(1, p + 1):
                        df = x[i - 1, j - 1] - cen[l1 - 1, j - 1]
                        de += df * df
                    d[i] = de * an1[l1]
                da = 0.0
                for j in range(1, p + 1):
                    db = x[i - 1, j - 1] - cen[l2 - 1, j - 1]
                    da += db * db
                r2 = da * an2[l2]
                for ell in range(1, k + 1):
                    if (i >= live[l1] and i >= live[ell]) or ell == l1 or ell == ll:
                        continue
                    rr = r2 / an2[ell]
                    dc = 0.0
                    skip = False
                    for j in range(1, p + 1):
                        dd = x[i - 1, j - 1] - cen[ell - 1, j - 1]
                        dc += dd * dd
                        if dc >= rr:
                            skip = True
                            break
                    if skip:
                        continue
                    r2 = dc * an2[ell]
                    l2 = ell
                if r2 >= d[i]:
                    ic2[i] = l2
                else:
                    indx = 0
                    live[l1] = m + i
                    live[l2] = m + i
                    ncp[l1] = i
                    ncp[l2] = i
                    al1 = float(nc[l1])
                    alw = al1 - 1.0
                    al2 = float(nc[l2])
                    alt = al2 + 1.0
                    for j in range(1, p + 1):
                        cen[l1 - 1, j - 1] = (cen[l1 - 1, j - 1] * al1
                                              - x[i - 1, j - 1]) / alw
                        cen[l2 - 1, j - 1] = (cen[l2 - 1, j - 1] * al2
                                              + x[i - 1, j - 1]) / alt
                    nc[l1] -= 1
                    nc[l2] += 1
                    an2[l1] = alw / al1
                    an1[l1] = big
                    if alw > 1.0:
                        an1[l1] = alw / (alw - 1.0)
                    an1[l2] = alt / al2
                    an2[l2] = alt / (alt + 1.0)
                    ic1[i] = l2
                    ic2[i] = l1
            if indx == m:
                return indx
        for ell in range(1, k + 1):
            itran[ell] = 0
            live[ell] = live[ell] - m
        return indx

    def _qtran(indx):
        nonlocal imaxqtr
        icoun = 0
        istep = 0
        while True:
            for i in range(1, m + 1):
                icoun += 1
                istep += 1
                if istep >= imaxqtr:
                    imaxqtr = -1
                    return indx
                l1 = ic1[i]
                l2 = ic2[i]
                if nc[l1] != 1:
                    if istep <= ncp[l1]:
                        da = 0.0
                        for j in range(1, p + 1):
                            db = x[i - 1, j - 1] - cen[l1 - 1, j - 1]
                            da += db * db
                        d[i] = da * an1[l1]
                    if istep < ncp[l1] or istep < ncp[l2]:
                        r2 = d[i] / an2[l2]
                        dd = 0.0
                        skip = False
                        for j in range(1, p + 1):
                            de = x[i - 1, j - 1] - cen[l2 - 1, j - 1]
                            dd += de * de
                            if dd >= r2:
                                skip = True
                                break
                        if not skip:
                            icoun = 0
                            indx = 0
                            itran[l1] = 1
                            itran[l2] = 1
                            ncp[l1] = istep + m
                            ncp[l2] = istep + m
                            al1 = float(nc[l1])
                            alw = al1 - 1.0
                            al2 = float(nc[l2])
                            alt = al2 + 1.0
                            for j in range(1, p + 1):
                                cen[l1 - 1, j - 1] = (cen[l1 - 1, j - 1] * al1
                                                      - x[i - 1, j - 1]) / alw
                                cen[l2 - 1, j - 1] = (cen[l2 - 1, j - 1] * al2
                                                      + x[i - 1, j - 1]) / alt
                            nc[l1] -= 1
                            nc[l2] += 1
                            an2[l1] = alw / al1
                            an1[l1] = big
                            if alw > 1.0:
                                an1[l1] = alw / (alw - 1.0)
                            an1[l2] = alt / al2
                            an2[l2] = alt / (alt + 1.0)
                            ic1[i] = l2
                            ic2[i] = l1
                if icoun == m:
                    return indx
            # GO TO 10: repeat the sweep

    indx = 0
    iter_returned = iter_max + 1  # set IFAULT=2 unless we break early
    ifault = 2
    for ij in range(1, iter_max + 1):
        indx = _optra(indx)
        if indx == m:
            iter_returned = ij
            ifault = 0
            break
        indx = _qtran(indx)
        if imaxqtr < 0:
            ifault = 4
            iter_returned = ij
            break
        if k == 2:
            iter_returned = ij
            ifault = 0
            break
        for ell in range(1, k + 1):
            ncp[ell] = 0

    # Within-cluster sum of squares (recomputes centres as the cluster means).
    wss = [0.0] * (k + 1)
    for ell in range(1, k + 1):
        for j in range(1, p + 1):
            cen[ell - 1, j - 1] = 0.0
    for i in range(1, m + 1):
        ii = ic1[i]
        for j in range(1, p + 1):
            cen[ii - 1, j - 1] += x[i - 1, j - 1]
    for j in range(1, p + 1):
        for ell in range(1, k + 1):
            cen[ell - 1, j - 1] /= float(nc[ell])
        for i in range(1, m + 1):
            ii = ic1[i]
            da = x[i - 1, j - 1] - cen[ii - 1, j - 1]
            wss[ii] += da * da

    return {
        "cluster": np.array(ic1[1:m + 1], dtype=np.int64),
        "centers": cen,
        "nc": np.array(nc[1:k + 1], dtype=np.int64),
        "wss": np.array(wss[1:k + 1], dtype=float),
        "iter": iter_returned,
        "ifault": ifault,
    }


# --------------------------------------------------------------------------- #
# the Hclust object
# --------------------------------------------------------------------------- #
class Hclust:
    """R's ``"hclust"`` object — the agglomeration history.

    Attributes mirror R 1:1 (and keep R's integer conventions so ``cutree`` /
    ``cophenetic`` / the dendrogram converters consume them unchanged):

    * ``merge`` — ``(n-1, 2)`` int array. Negative entries are singleton leaves
      (``-obs``); positive entries are earlier merge steps (1-based).
    * ``height`` — ``(n-1,)`` float, the agglomeration criterion per step.
    * ``order`` — ``(n,)`` int, the 1-based leaf order for plotting.
    * ``labels`` / ``method`` / ``call`` / ``dist_method``.
    """

    __slots__ = ("merge", "height", "order", "labels", "method", "call",
                 "dist_method")

    def __init__(self, merge, height, order, labels=None, method=None,
                 call=None, dist_method=None):
        self.merge = np.asarray(merge, dtype=np.int64).reshape(-1, 2)
        self.height = np.asarray(height, dtype=float)
        self.order = np.asarray(order, dtype=np.int64)
        self.labels = list(labels) if labels is not None else None
        self.method = method
        self.call = call
        self.dist_method = dist_method

    def __repr__(self):
        return print_hclust(self, _return=True)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def hclust(d, method="complete", members=None):
    """R ``stats::hclust(d, method, members)`` — agglomerative hierarchical
    clustering of a :class:`~hea.R.distance.Dist`.

    ``method`` is partial-matched against ``ward.D``/``single``/``complete``/
    ``average``/``mcquitty``/``median``/``centroid``/``ward.D2`` (the legacy
    ``"ward"`` maps to ``ward.D``). Returns an :class:`Hclust`.
    """
    if method == "ward":  # do not deprecate earlier than 2015!
        warnings.warn(
            'The "ward" method has been renamed to "ward.D"; note new "ward.D2"',
            stacklevel=2)
        method = "ward.D"
    i_meth = _pmatch(method, _METHODS)
    if i_meth is None:
        raise ValueError(f"invalid clustering method {method}")
    iopt = i_meth + 1  # Fortran 1-based code

    if not isinstance(d, Dist):
        n = getattr(d, "Size", None)
        if n is None:
            raise ValueError("invalid dissimilarities")
        labels = getattr(d, "Labels", None)
        dist_method = getattr(d, "method", None)
        data = np.asarray(d, dtype=float)
    else:
        n = d.Size
        labels = d.Labels
        dist_method = d.method
        data = d.data

    n = int(n)
    if n > 65536:
        raise ValueError("size cannot be NA nor exceed 65536")
    if n < 2:
        raise ValueError("must have n >= 2 objects to cluster")
    length = n * (n - 1) // 2
    if data.size != length:
        if data.size < length:
            raise ValueError("dissimilarities of improper length")
        warnings.warn("dissimilarities of improper length", stacklevel=2)

    if members is None:
        members = np.ones(n)
    else:
        members = np.asarray(members, dtype=float)
        if members.size != n:
            raise ValueError("invalid length of members")

    if _rs_hclust is not None:  # Rust accelerator (step 10); pure-Python is spec
        ia, ib, crit = _rs_hclust(n, np.ascontiguousarray(data), iopt, members)
        ia = [0, *list(ia)]
        ib = [0, *list(ib)]
        crit = [0.0, *list(crit)]
    else:
        ia, ib, crit = _hclust_fortran(n, data, iopt, members)

    iorder, iia, iib = _hcass2(n, ia, ib)

    merge = np.empty((n - 1, 2), dtype=np.int64)
    merge[:, 0] = iia[1:n]
    merge[:, 1] = iib[1:n]
    height = np.asarray(crit[1:n], dtype=float)
    order = np.asarray(iorder[1:n + 1], dtype=np.int64)

    return Hclust(merge, height, order, labels=labels, method=_METHODS[i_meth],
                  dist_method=dist_method)


def cutree(tree, k=None, h=None):
    """R ``stats::cutree(tree, k, h)`` — cut an :class:`Hclust` into groups.

    Specify either ``k`` (number of groups) or ``h`` (height). Each may be a
    scalar (→ a 1-D label vector) or a vector (→ an ``n x len`` matrix, one
    column per cut). Mirrors ``cutree.R``: the ``h`` → ``k`` map is
    ``n + 1 - which.max(c(height, Inf) > h)``.
    """
    merge = np.asarray(tree.merge, dtype=np.int64).reshape(-1, 2)
    n1 = merge.shape[0]
    if n1 < 1:
        raise ValueError("invalid 'tree' ('merge' component)")
    n = n1 + 1

    if k is None and h is None:
        raise ValueError("either 'k' or 'h' must be specified")

    if k is None:  # h |--> k
        height = np.asarray(tree.height, dtype=float)
        if np.any(np.diff(height) < 0):
            raise ValueError(
                "the 'height' component of 'tree' is not sorted (increasingly)")
        hvals = np.atleast_1d(np.asarray(h, dtype=float))
        heights_inf = np.concatenate([height, [np.inf]])
        which = np.empty(hvals.size, dtype=np.int64)
        for i, hv in enumerate(hvals):
            # which.max of the logical: index (1-based) of the first TRUE
            which[i] = n + 1 - (int(np.argmax(heights_inf > hv)) + 1)
    else:
        which = np.atleast_1d(np.asarray(k, dtype=np.int64))
        if which.min() < 1 or which.max() > n:
            raise ValueError(f"elements of 'k' must be between 1 and {n}")

    ans = _cutree_c(merge, which)
    if which.size == 1:
        return ans[:, 0]
    return ans


def cophenetic(x):
    """R ``stats::cophenetic(x)`` / ``cophenetic.default`` — cophenetic distances
    of an :class:`Hclust` (the height at which each pair of leaves is first
    joined), as a :class:`~hea.R.distance.Dist`.

    Mirrors ``cluster_hclust.R:220``: walk ``merge`` bottom-up, accumulating each
    cluster's leaf set, and write the join height into the cross-block of the two
    children; symmetrize (``out + tᵀ``) and coerce with ``as_dist``.
    (``cophenetic.dendrogram`` lands with the dendrogram subsystem, step 7.)
    """
    x = as_hclust(x)
    n = len(x.order)
    ilist = [None] * (n + 1)  # leaf sets per 1-based merge step (1..n-1)
    out = np.zeros((n, n), dtype=float)
    merge = x.merge
    height = x.height
    for i in range(1, n):  # merge step i
        a = int(merge[i - 1, 0])
        b = int(merge[i - 1, 1])
        ids1 = np.array([-a]) if a < 0 else ilist[a]
        ids2 = np.array([-b]) if b < 0 else ilist[b]
        ilist[i] = np.concatenate([ids1, ids2])
        rows = np.repeat(ids1, ids2.size)
        cols = np.tile(ids2, ids1.size)
        out[rows - 1, cols - 1] = height[i - 1]
    out = out + out.T
    d = as_dist(out)
    d.Labels = x.labels
    return d


def as_hclust(x, **kwargs):
    """R ``as.hclust(x)`` / ``as.hclust.default`` — identity for an
    :class:`Hclust`; otherwise an error (other coercions land with the
    dendrogram subsystem)."""
    if isinstance(x, Hclust):
        return x
    raise TypeError("argument 'x' cannot be coerced to class 'hclust'")


def print_hclust(x, _return=False):
    """R ``print(<hclust>)`` — the method / distance / object-count summary."""
    lines = []
    if x.method is not None:
        lines.append(f"Cluster method   : {x.method} ")
    if x.dist_method is not None:
        lines.append(f"Distance         : {x.dist_method} ")
    lines.append(f"Number of objects: {len(x.height) + 1} ")
    s = "\n".join(lines) + "\n"
    if _return:
        return s
    print(s, end="")
    return None


# --------------------------------------------------------------------------- #
# kmeans (public API)
# --------------------------------------------------------------------------- #
def _match_arg(arg, choices):
    """R ``match.arg`` for a single string: exact, else unique prefix."""
    if arg in choices:
        return arg
    hits = [c for c in choices if c.startswith(arg)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"'arg' should be one of {choices}")


def _unique_rows(x):
    """R ``unique(<matrix>)`` — distinct rows in first-appearance order."""
    seen = set()
    out = []
    for row in x:
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(row)
    return np.array(out)


def _has_duplicate_rows(x):
    """R ``any(duplicated(<matrix>))`` over rows."""
    seen = set()
    for row in np.atleast_2d(x):
        key = row.tobytes()
        if key in seen:
            return True
        seen.add(key)
    return False


class Kmeans:
    """R's ``"kmeans"`` object — the fitted partition.

    ``cluster`` is the 1-based assignment per point; ``centers`` the ``(k, p)``
    final centres; plus ``totss``/``withinss``/``tot_withinss``/``betweenss``/
    ``size``/``iter``/``ifault``, mirroring R.
    """

    __slots__ = ("cluster", "centers", "totss", "withinss", "tot_withinss",
                 "betweenss", "size", "iter", "ifault")

    def __init__(self, cluster, centers, totss, withinss, tot_withinss,
                 betweenss, size, iter, ifault=None):
        self.cluster = np.asarray(cluster, dtype=np.int64)
        self.centers = np.asarray(centers, dtype=float)
        self.totss = float(totss)
        self.withinss = np.asarray(withinss, dtype=float)
        self.tot_withinss = float(tot_withinss)
        self.betweenss = float(betweenss)
        self.size = np.asarray(size, dtype=np.int64)
        self.iter = int(iter)
        self.ifault = ifault

    def __repr__(self):
        return print_kmeans(self, _return=True)


def _do_one(nmeth, x, centers, k, iter_max, trace):
    """R ``do_one(nmeth)`` — dispatch to a kernel + the post-run warnings."""
    if nmeth == 1:  # Hartigan-Wong
        z = _kmns(x, centers, k, iter_max, 1 if trace else 0)
        ifault = z.get("ifault")
        if ifault == 1:
            raise ValueError(
                "empty cluster: try a better set of initial centers")
        if ifault == 3:
            raise ValueError(
                "number of cluster centres must lie between 1 and nrow(x)")
        if ifault == 4:
            warnings.warn(
                "Quick-TRANSfer stage steps exceeded maximum "
                f"(= {min(2147483647, 50 * x.shape[0])})", stacklevel=2)
        if z["iter"] > iter_max:
            warnings.warn(f"did not converge in {iter_max} iterations",
                          stacklevel=2)
        return z

    if nmeth == 2:  # Lloyd / Forgy
        cl, cen, nc, wss, it = _kmeans_lloyd(x, centers, k, iter_max)
    else:  # MacQueen
        cl, cen, nc, wss, it = _kmeans_macqueen(x, centers, k, iter_max)
    ifault = None
    if np.any(nc == 0):
        warnings.warn("empty cluster: try a better set of initial centers",
                      stacklevel=2)
    if it > iter_max:
        warnings.warn(f"did not converge in {iter_max} iterations", stacklevel=2)
        ifault = 2
    return {"cluster": cl, "centers": cen, "nc": nc, "wss": wss, "iter": it,
            "ifault": ifault}


def kmeans(x, centers, iter_max=10, nstart=1, algorithm="Hartigan-Wong",
           trace=False):
    """R ``stats::kmeans(x, centers, iter.max, nstart, algorithm)`` — k-means.

    ``centers`` is either the number of clusters ``k`` (initial centres drawn
    from ``x`` via :meth:`RMersenneTwister.sample_int`, R's RNG stream — set the
    seed with ``hea.R.set_seed`` for reproducibility) or an explicit ``(k, p)``
    centre matrix. ``algorithm`` ∈ ``Hartigan-Wong`` (default) / ``Lloyd`` /
    ``Forgy`` / ``MacQueen``. Returns a :class:`Kmeans`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    m, p = x.shape
    nmeth = {"Hartigan-Wong": 1, "Lloyd": 2, "Forgy": 2, "MacQueen": 3}[
        _match_arg(algorithm, ("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"))]

    cn = None
    mm = None
    if np.size(centers) == 1:
        k = int(np.asarray(centers).reshape(-1)[0])
        if nstart == 1:
            centers_mat = x[_r_rng().sample_int(m, k), :]
        if nstart >= 2 or _has_duplicate_rows(centers_mat):
            cn = _unique_rows(x)
            mm = cn.shape[0]
            if mm < k:
                raise ValueError(
                    "more cluster centers than distinct data points.")
            centers_mat = cn[_r_rng().sample_int(mm, k), :]
    else:
        centers_mat = np.atleast_2d(np.asarray(centers, dtype=float))
        if _has_duplicate_rows(centers_mat):
            raise ValueError("initial centers are not distinct")
        k = centers_mat.shape[0]
        if m < k:
            raise ValueError("more cluster centers than data points")

    k = int(k)
    if k == 1:
        nmeth = 3  # Hartigan-Wong (Fortran) needs k > 1
    iter_max = int(iter_max)
    if iter_max < 1:
        raise ValueError("'iter.max' must be positive")
    if centers_mat.shape[1] != p:
        raise ValueError("must have same number of columns in 'x' and 'centers'")

    z = _do_one(nmeth, x, centers_mat, k, iter_max, trace)
    best = float(z["wss"].sum())
    if nstart >= 2 and cn is not None:
        for _ in range(2, nstart + 1):
            centers_mat = cn[_r_rng().sample_int(mm, k), :]
            zz = _do_one(nmeth, x, centers_mat, k, iter_max, trace)
            tot = float(zz["wss"].sum())
            if tot < best:
                z = zz
                best = tot

    totss = float(((x - x.mean(axis=0)) ** 2).sum())
    return Kmeans(
        cluster=z["cluster"], centers=z["centers"], totss=totss,
        withinss=z["wss"], tot_withinss=best, betweenss=totss - best,
        size=z["nc"], iter=z["iter"], ifault=z.get("ifault"))


def fitted_kmeans(object, method="centers"):
    """R ``fitted(<kmeans>)`` — per-point fitted centres (``"centers"``) or the
    cluster labels (``"classes"``)."""
    method = _match_arg(method, ("centers", "classes"))
    if method == "centers":
        return object.centers[object.cluster - 1, :]
    return object.cluster


def print_kmeans(x, _return=False):
    """R ``print(<kmeans>)`` — the cluster-sizes / means / SS-ratio summary."""
    sizes = ", ".join(str(int(s)) for s in x.size)
    lines = [
        f"K-means clustering with {len(x.size)} clusters of sizes {sizes}",
        "",
        "Cluster means:",
        str(x.centers),
        "",
        "Clustering vector:",
        str(x.cluster),
        "",
        "Within cluster sum of squares by cluster:",
        str(x.withinss),
        f" (between_SS / total_SS = {100 * x.betweenss / x.totss:5.1f} %)",
    ]
    if x.ifault == 2:
        lines.append(
            "Warning: did *not* converge in specified number of iterations")
    s = "\n".join(lines) + "\n"
    if _return:
        return s
    print(s, end="")
    return None
