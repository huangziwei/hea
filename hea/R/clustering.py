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

from .._dispatch import rs_fn
from ._shared import _rfma
from .distance import Dist, _pmatch, as_dist, as_matrix_dist
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
    # dendrogram subsystem (cluster_dendrogram.R)
    "Dendrogram",
    "as_dendrogram",
    "cophenetic_dendrogram",
    "cut_dendrogram",
    "dendrapply",
    "is_leaf",
    "labels_dendrogram",
    "merge_dendrogram",
    "midcache_dendrogram",
    "nleaves",
    "nobs_dendrogram",
    "order_dendrogram",
    "print_dendrogram",
    "reorder",
    "reorder_dendrogram",
    "rev_dendrogram",
    "str_dendrogram",
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

# Rust seams for the remaining compiled-in-R kernels (R does these in C/Fortran,
# so the pure-Python ports are slow): the cutree grouping (`C_cutree`) and the
# Lloyd/MacQueen k-means (`cluster_kmeans.c`). Lloyd's assignment phase is
# rayon-parallel (independent per point); the rest stays sequential (0-ulp).
_rs_cutree = rs_fn("cutree")
_rs_lloyd = rs_fn("lloyd")
_rs_macqueen = rs_fn("macqueen")


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
                    # R's gfortran fuses the LW update to fmadd on arm64;
                    # ``_rfma`` mirrors it per-arch (plain a*b+c on x86).
                    d[ind1] = _rfma(membr[i2] + membr[k], d[ind1],
                                    (membr[j2] + membr[k]) * d[ind2])
                    d[ind1] = _rfma(-membr[k], d12, d[ind1])
                    d[ind1] = d[ind1] / (membr[i2] + membr[j2] + membr[k])
                elif iopt == 2:  # single
                    d[ind1] = min(d[ind1], d[ind2])
                elif iopt == 3:  # complete
                    d[ind1] = max(d[ind1], d[ind2])
                elif iopt == 4:  # average (UPGMA)
                    d[ind1] = (_rfma(membr[i2], d[ind1], membr[j2] * d[ind2])
                               / (membr[i2] + membr[j2]))
                elif iopt == 5:  # mcquitty (WPGMA)
                    d[ind1] = (d[ind1] + d[ind2]) / 2
                elif iopt == 6:  # median (WPGMC)
                    d[ind1] = ((d[ind1] + d[ind2]) - d12 / 2) / 2
                elif iopt == 7:  # centroid (UPGMC)
                    d[ind1] = ((_rfma(membr[i2], d[ind1], membr[j2] * d[ind2])
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
            # R fuses `wss += d*d` to fmadd on arm64; ``_rfma`` mirrors per-arch.
            wss[it] = _rfma(tmp, tmp, wss[it])
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
            wss[ii] = _rfma(da, da, wss[ii])

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

    merge = np.empty((n - 1, 2), dtype=np.int64)
    if _rs_hclust is not None:
        # Rust does the agglomeration AND hcass2 (the merge->order transform),
        # returning the final columns directly — no O(n^2) Python post-processing.
        # The pure-Python ``_hclust_fortran``/``_hcass2`` below stay the spec.
        iia, iib, height, order = _rs_hclust(
            n, np.ascontiguousarray(data), iopt, members)
        merge[:, 0] = iia
        merge[:, 1] = iib
        height = np.asarray(height, dtype=float)
        order = np.asarray(order, dtype=np.int64)
    else:
        ia, ib, crit = _hclust_fortran(n, data, iopt, members)
        iorder, iia, iib = _hcass2(n, ia, ib)
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

    if _rs_cutree is not None:  # Rust C_cutree port; pure-Python _cutree_c is spec
        ans = np.asarray(_rs_cutree(np.ascontiguousarray(merge), which))
    else:
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
    ``cophenetic.dendrogram`` (recursing over a :class:`Dendrogram`) is dispatched
    here when ``x`` is a dendrogram.
    """
    if isinstance(x, Dendrogram):
        return cophenetic_dendrogram(x)
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
    """R ``as.hclust(x)`` — identity for an :class:`Hclust`
    (``as.hclust.default``); a :class:`Dendrogram` is coerced back to an
    :class:`Hclust` via ``as.hclust.dendrogram``; otherwise an error."""
    if isinstance(x, Hclust):
        return x
    if isinstance(x, Dendrogram):
        return as_hclust_dendrogram(x, **kwargs)
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


def _kmns_rs(x, centers, k, iter_max):
    """Rust ``kmns`` (Hartigan-Wong) → the same dict shape as :func:`_kmns`.
    The pure-Python :func:`_kmns` stays the spec/oracle (A/B'd in test_kmeans)."""
    ifault, cluster, cen_flat, nc, wss, it = _rs_kmns(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(centers, dtype=float),
        int(k), int(iter_max))
    if ifault in (1, 3):
        return {"ifault": int(ifault)}
    return {
        "cluster": np.asarray(cluster, dtype=np.int64),
        "centers": np.asarray(cen_flat, dtype=float).reshape(k, x.shape[1]),
        "nc": np.asarray(nc, dtype=np.int64),
        "wss": np.asarray(wss, dtype=float),
        "iter": int(it),
        "ifault": int(ifault),
    }


def _kmeans_cd_rs(rs_kernel, x, centers, k, iter_max):
    """Rust Lloyd/MacQueen → ``(cl, cen, nc, wss, iter)`` like the pure-Python
    kernels (which stay the spec/oracle). ``cen`` is reshaped to ``(k, p)``."""
    cl, cen, nc, wss, it = rs_kernel(
        np.ascontiguousarray(x, dtype=float),
        np.ascontiguousarray(centers, dtype=float),
        int(k), int(iter_max))
    return (np.asarray(cl, dtype=np.int64),
            np.asarray(cen, dtype=float).reshape(k, x.shape[1]),
            np.asarray(nc, dtype=np.int64),
            np.asarray(wss, dtype=float),
            int(it))


def _do_one(nmeth, x, centers, k, iter_max, trace):
    """R ``do_one(nmeth)`` — dispatch to a kernel + the post-run warnings."""
    if nmeth == 1:  # Hartigan-Wong
        if _rs_kmns is not None:  # Rust accelerator; pure-Python _kmns is the spec
            z = _kmns_rs(x, centers, k, iter_max)
        else:
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
        if _rs_lloyd is not None:
            cl, cen, nc, wss, it = _kmeans_cd_rs(_rs_lloyd, x, centers, k, iter_max)
        else:
            cl, cen, nc, wss, it = _kmeans_lloyd(x, centers, k, iter_max)
    else:  # MacQueen
        if _rs_macqueen is not None:
            cl, cen, nc, wss, it = _kmeans_cd_rs(
                _rs_macqueen, x, centers, k, iter_max)
        else:
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


# --------------------------------------------------------------------------- #
# Dendrogram subsystem (port of cluster_dendrogram.R, non-graphics surface)
# --------------------------------------------------------------------------- #
class Dendrogram:
    """R's ``"dendrogram"`` object — a binary (or k-ary) tree carried as nested
    nodes with attributes, mirroring R's "list / integer + attributes" layout so
    the accessors/transforms compose.

    * a **node** has ``children`` (a list of :class:`Dendrogram`) and
      ``value = None``;
    * a **leaf** has ``children = None`` and ``value`` = its observation index
      (R's atomic-integer-with-attributes leaf), with ``attrs['leaf'] = True``.

    ``attrs`` holds R's node attributes verbatim (``members``, ``height``,
    ``midpoint``, ``label``, ``leaf``, ``x.member``, ``value`` …). ``len`` and
    ``[[`` follow R: ``len`` is the branch count (1 for a leaf) and ``d[k]`` is
    R's **1-based** ``[[.dendrogram`` (a tuple descends recursively)."""

    __slots__ = ("children", "value", "attrs")

    def __init__(self, children=None, value=None, attrs=None):
        self.children = children
        self.value = value
        self.attrs = {} if attrs is None else attrs

    def __len__(self):
        # R length(): #{branches} for a node, 1 for a leaf (a scalar integer).
        return 1 if self.children is None else len(self.children)

    def __int__(self):
        # R as.integer(<leaf>) — the observation index.
        return int(self.value)

    def __getitem__(self, key):
        # R `[[.dendrogram` (1-based); a sequence descends recursively.
        if isinstance(key, (tuple, list, np.ndarray)):
            node = self
            for k in key:
                node = node._child(int(k))
            return node
        return self._child(int(key))

    def _child(self, k):
        if self.children is None:
            raise IndexError("subscript out of bounds (leaf has no branches)")
        return self.children[k - 1]

    def __repr__(self):
        return print_dendrogram(self, _return=True)


def is_leaf(object):
    """R ``is.leaf(object)`` — ``attr(object, "leaf")`` is logical ``TRUE``."""
    return isinstance(object, Dendrogram) and object.attrs.get("leaf") is True


def _member_dend(x):
    """``.memberDend``: ``x.member %||% (members %||% 1)``."""
    v = x.attrs.get("x.member")
    if v is not None:
        return v
    v = x.attrs.get("members")
    return v if v is not None else 1


def _mid_dend(x):
    """``.midDend``: ``midpoint %||% 0``."""
    v = x.attrs.get("midpoint")
    return v if v is not None else 0


def _clone(d):
    """Deep copy of a dendrogram (structure + a fresh ``attrs`` dict at each
    node). R's copy-on-modify means each transform yields an independent tree;
    we mirror that by cloning where R would copy."""
    if d.children is None:
        return Dendrogram(children=None, value=d.value, attrs=dict(d.attrs))
    return Dendrogram(children=[_clone(c) for c in d.children],
                      value=d.value, attrs=dict(d.attrs))


def _unlist(d):
    """R ``unlist(<dendrogram>)`` — the leaf observation indices, in pre-order."""
    if d.children is None:
        return [d.value]
    out = []
    for ch in d.children:
        out.extend(_unlist(ch))
    return out


def _rapply_label(d):
    """``rapply(object, attr, which="label")`` — leaf labels, pre-order."""
    if d.children is None:
        return [d.attrs.get("label")]
    out = []
    for ch in d.children:
        out.extend(_rapply_label(ch))
    return out


def _fmt_digits(v, digits):
    """Approximate R ``format(v, digits=)`` for a single value (display only —
    the dendrogram parity gate is the numeric/structural data, not print text)."""
    if v is None:
        return "NA"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == 0:
        return "0"
    return np.format_float_positional(f, precision=int(digits),
                                      fractional=False, trim="-")


def nleaves(node):
    """R ``nleaves(node)`` — count the leaves of a dendrogram (iterative, the
    todo-stack traversal from ``cluster_dendrogram.R:85``)."""
    if is_leaf(node):
        return 1
    todo = None  # linked list of pending non-leaf nodes
    count = 0
    cur = list(node.children)
    while True:
        while cur:
            child = cur.pop(0)          # node[[1L]] ; node <- node[-1L]
            if is_leaf(child):
                count += 1
            else:
                todo = (child, todo)
        if todo is None:
            break
        node_, todo = todo
        cur = list(node_.children)
    return count


def _validity_hclust(x, merge=None, order=True):
    """Port of ``.validity.hclust`` (``cluster_hclust.R:121``) — returns ``True``
    or an error message string."""
    if merge is None:
        merge = x.merge
    merge = np.asarray(merge)
    if merge.ndim != 2 or merge.shape[1] != 2:
        return "invalid dendrogram"
    if np.any(merge.astype(np.int64) != merge):
        return "'merge' component in dendrogram must be integer"
    n1 = merge.shape[0]
    n = n1 + 1
    if len(x.height) != n1:
        return "'height' is of wrong length"
    if order and len(x.order) != n:
        return "'order' is of wrong length"
    # identical(sort(as.integer(merge)), c(-(n:1L), +seq_len(n-2L)))
    expected = np.concatenate([np.arange(-n, 0), np.arange(1, n - 1)])
    if np.array_equal(np.sort(merge.astype(np.int64).ravel()), expected):
        return True
    return "'merge' matrix has invalid contents"


def as_dendrogram(object, hang=-1, check=True, **kwargs):
    """R ``as.dendrogram(object)`` — coerce to a :class:`Dendrogram`.

    ``as.dendrogram.dendrogram`` is the identity; ``as.dendrogram.hclust``
    (``cluster_dendrogram.R:23``) is the main builder (``hang`` controls the
    height at which leaves hang below their merge)."""
    if isinstance(object, Dendrogram):
        return object
    if isinstance(object, Hclust):
        return _as_dendrogram_hclust(object, hang=hang, check=check)
    raise TypeError("no applicable method for 'as.dendrogram'")


def _as_dendrogram_hclust(object, hang=-1, check=True):
    """Port of ``as.dendrogram.hclust`` (``cluster_dendrogram.R:23``)."""
    nolabels = object.labels is None
    merge = object.merge
    if check:
        msg = _validity_hclust(object, merge, order=nolabels)
        if msg is not True:
            raise ValueError(msg)
    if nolabels:
        labels = np.arange(1, len(object.order) + 1)  # seq_along(order)
    else:
        labels = object.labels

    z = {}                                            # keyed by str(merge step)
    oHgt = object.height
    nMerge = len(oHgt)
    hMax = oHgt[nMerge - 1]
    k = 0
    for k in range(1, nMerge + 1):
        x0 = int(merge[k - 1, 0])
        x1 = int(merge[k - 1, 1])
        n0, n1 = x0 < 0, x1 < 0
        h0 = None
        if n0 or n1:
            h0 = 0 if hang < 0 else max(0, oHgt[k - 1] - hang * hMax)
        if n0 and n1:                                 # two leaves
            left = Dendrogram(value=-x0)
            right = Dendrogram(value=-x1)
            zk = Dendrogram(children=[left, right])
            zk.attrs["members"] = 2
            zk.attrs["midpoint"] = 0.5
            left.attrs["label"] = labels[-x0 - 1]
            right.attrs["label"] = labels[-x1 - 1]
            left.attrs["members"] = right.attrs["members"] = 1
            left.attrs["height"] = right.attrs["height"] = h0
            left.attrs["leaf"] = right.attrs["leaf"] = True
        elif n0 or n1:                                # one leaf, one node
            isL = n0                                  # leaf on the left?
            node = z[str(x1 if isL else x0)]          # z[[X[1 + isL]]]
            if isL:
                leaf = Dendrogram(value=-x0)
                zk = Dendrogram(children=[leaf, node])
            else:
                leaf = Dendrogram(value=-x1)
                zk = Dendrogram(children=[node, leaf])
            zk.attrs["members"] = node.attrs["members"] + 1
            zk.attrs["midpoint"] = (
                _member_dend(zk.children[0]) + node.attrs["midpoint"]) / 2
            leaf.attrs["members"] = 1                 # set AFTER midpoint (as in R)
            leaf.attrs["height"] = h0
            leaf.attrs["label"] = labels[leaf.value - 1]
            leaf.attrs["leaf"] = True
            del z[str(x1 if isL else x0)]
        else:                                         # two non-leaf nodes
            ln = z[str(x0)]
            rn = z[str(x1)]
            zk = Dendrogram(children=[ln, rn])
            zk.attrs["members"] = ln.attrs["members"] + rn.attrs["members"]
            zk.attrs["midpoint"] = (ln.attrs["members"]
                                    + ln.attrs["midpoint"]
                                    + rn.attrs["midpoint"]) / 2
            del z[str(x0)]
            del z[str(x1)]
        zk.attrs["height"] = oHgt[k - 1]
        z[str(k)] = zk
    return z[str(k)]


def as_hclust_dendrogram(x, **kwargs):
    """Port of ``as.hclust.dendrogram`` (``cluster_dendrogram.R:115``) — reverse
    a *binary* dendrogram into an :class:`Hclust` (``merge``/``height``/``order``).
    The pre-order traversal and the stable height sort reproduce R's merge order
    (``method``/``dist.method`` are ``NA``, lost in the round-trip)."""
    if not (x.children is not None and len(x.children) == 2):
        raise ValueError("as.hclust.dendrogram: need a list dendrogram of length 2")
    n = nleaves(x)
    if n != x.attrs.get("members"):
        raise ValueError("number of leaves != 'members' attribute")

    ord_ = np.zeros(n, dtype=np.int64)
    labsu = [None] * n
    n_h = n - 1
    height = np.zeros(n_h, dtype=float)
    myIdx = np.zeros((2, n_h), dtype=np.int64)        # NA -> 0 (root col unused)
    merge = np.zeros((2, n_h), dtype=np.int64)        # NA -> 0 (filled later)

    rem = list(x.children)                            # remaining children of node
    cur_height = x.attrs["height"]
    position = 0
    stack = None
    leafCount = 0
    nodeCount = 0
    myNodeIndex = 0
    while True:
        while len(rem):
            if position == 0:                         # first visit to this node
                nodeCount += 1
                myNodeIndex = nodeCount
                if nodeCount != 1:
                    myIdx[0, nodeCount - 1] = stack["position"]
                    myIdx[1, nodeCount - 1] = stack["myNodeIndex"]
                height[nodeCount - 1] = cur_height
            position += 1
            child = rem.pop(0)                        # x[[1L]] ; x <- x[-1L]
            if is_leaf(child):
                leafCount += 1
                labsu[leafCount - 1] = child.attrs.get("label")
                ord_[leafCount - 1] = int(child.value)
                merge[position - 1, myNodeIndex - 1] = -ord_[leafCount - 1]
            else:
                if len(child.children) != 2:
                    raise ValueError("as.hclust.dendrogram: non-binary node")
                stack = {"node": rem, "position": position,
                         "myNodeIndex": myNodeIndex, "stack": stack}
                rem = list(child.children)
                cur_height = child.attrs["height"]
                position = 0
        if stack is None:
            break
        position = stack["position"]
        rem = stack["node"]
        myNodeIndex = stack["myNodeIndex"]
        stack = stack["stack"]

    iOrd = np.argsort(ord_, kind="stable")            # sort.list(ord)
    if not np.array_equal(ord_[iOrd], np.arange(1, n + 1)):
        raise ValueError(
            f"dendrogram entries must be 1,2,..,{n} (in any order), "
            'to be coercible to "hclust"')
    # ii <- sort.list(height, decreasing=TRUE)[n.h:1L]  (stable; ties reversed)
    ii = np.argsort(-height, kind="stable")[::-1]
    if not (n_h == 0 or ii[n_h - 1] == 0):
        raise ValueError("internal: root is not the last (tallest) node")
    for kk in range(1, n_h):                          # k <- seq_len(n.h-1L)
        col = ii[kk - 1]
        pos = myIdx[0, col]
        node_idx = myIdx[1, col]
        merge[pos - 1, node_idx - 1] = kk             # merge[t(myIdx[,ii[k]])] <- +k

    final_merge = merge[:, ii].T                      # t(merge[,ii])
    final_height = height[ii]
    final_labels = [labsu[i] for i in iOrd]           # labsu[iOrd]
    return Hclust(merge=final_merge, height=final_height, order=ord_,
                  labels=final_labels, method=None, dist_method=None)


def nobs_dendrogram(object):
    """R ``nobs.dendrogram`` — the ``"members"`` attribute."""
    return object.attrs.get("members")


def order_dendrogram(x):
    """R ``order.dendrogram(x)`` — the leaf observation indices in plot order
    (``unlist`` of the leaves, pre-order)."""
    if not isinstance(x, Dendrogram):
        raise TypeError("'order.dendrogram' requires a dendrogram")
    if x.children is not None:
        return np.array(_unlist(x), dtype=np.int64)
    return np.array([x.value], dtype=np.int64)


def labels_dendrogram(object):
    """R ``labels.dendrogram`` — the leaf labels, pre-order."""
    if object.children is not None:
        return np.array(_rapply_label(object))
    return object.attrs.get("label")


def midcache_dendrogram(x, type="hclust", quiet=False):
    """R ``midcache.dendrogram`` (``cluster_dendrogram.R:232``) — recompute every
    node's ``"midpoint"`` (e.g. after ``reorder``/``rev``). Returns a fresh tree;
    for a binary node ``midpoint = (.memberDend(child1) + Σ .midDend(child)) / 2``,
    matching ``as.dendrogram.hclust``."""
    if not isinstance(x, Dendrogram):
        raise TypeError("'midcache.dendrogram' requires a dendrogram")

    def setmid(d):
        if is_leaf(d):                                # no "midpoint" for a leaf
            return Dendrogram(children=None, value=d.value, attrs=dict(d.attrs))
        new_children = [setmid(c) for c in d.children]
        k = len(new_children)
        if (not quiet) and type == "hclust" and k != 2:
            warnings.warn(
                "midcache() of non-binary dendrograms only partly implemented",
                stacklevel=2)
        midS = math.fsum(_mid_dend(c) for c in new_children)
        new = Dendrogram(children=new_children, value=d.value, attrs=dict(d.attrs))
        new.attrs["midpoint"] = (_member_dend(new_children[0]) + midS) / 2
        return new

    return setmid(x)


def rev_dendrogram(x):
    """R ``rev.dendrogram`` — reverse the order of branches recursively, then
    recompute midpoints (``cluster_dendrogram.R:755``)."""
    def _rev(d):
        if is_leaf(d):
            return Dendrogram(children=None, value=d.value, attrs=dict(d.attrs))
        k = len(d.children)
        new_children = [_rev(d.children[k - 1 - j]) for j in range(k)]
        return Dendrogram(children=new_children, value=d.value,
                          attrs=dict(d.attrs))
    return midcache_dendrogram(_rev(x))


def reorder_dendrogram(x, wts, agglo_FUN=np.sum):
    """R ``reorder.dendrogram`` (``cluster_dendrogram.R:710``) — give each leaf a
    weight ``wts[leaf]``, sort each node's branches by their aggregated weight
    (``agglo.FUN``, default ``sum``), then recompute midpoints."""
    if not isinstance(x, Dendrogram):
        raise TypeError("'reorder.dendrogram' requires a dendrogram")
    wts = np.asarray(wts, dtype=float)

    def oV(d):
        if is_leaf(d):
            new = Dendrogram(children=None, value=d.value, attrs=dict(d.attrs))
            new.attrs["value"] = wts[d.value - 1]     # wts[x[1L]]
            return new
        new_children = [oV(c) for c in d.children]
        vals = np.array([c.attrs["value"] for c in new_children])
        iOrd = np.argsort(vals, kind="stable")        # sort.list(vals)
        new_children = [new_children[i] for i in iOrd]
        new = Dendrogram(children=new_children, value=d.value,
                         attrs=dict(d.attrs))
        new.attrs["value"] = float(agglo_FUN(vals[iOrd]))
        return new

    return midcache_dendrogram(oV(x))


def reorder(x, *args, **kwargs):
    """R ``reorder(x, ...)`` generic — dispatches to :func:`reorder_dendrogram`
    for a :class:`Dendrogram`."""
    if isinstance(x, Dendrogram):
        return reorder_dendrogram(x, *args, **kwargs)
    raise TypeError("no applicable 'reorder' method")


def _add_ifleaf(i, add):
    """``add.ifleaf`` from ``merge.dendrogram`` — shift a leaf's observation index
    by ``add`` (R arithmetic keeps the leaf's other attributes)."""
    if is_leaf(i):
        return Dendrogram(children=None, value=i.value + add, attrs=dict(i.attrs))
    return i


def merge_dendrogram(x, y, *others, height=None, adjust="auto"):
    """R ``merge.dendrogram`` (``cluster_dendrogram.R:775``) — combine
    dendrograms under a new root. ``adjust="add.max"`` (the ``"auto"`` default
    when every component's leaves start at 1) shifts later components' leaf
    indices so they stay distinct; ``height`` defaults to ``1.1 * max`` child
    height."""
    if not (isinstance(x, Dendrogram) and isinstance(y, Dendrogram)):
        raise TypeError("merge: 'x' and 'y' must be dendrograms")
    adjust = _match_arg(adjust, ("auto", "add.max", "none"))
    if adjust == "auto":
        adjust = ("add.max" if (min(_unlist(x)) == 1 and min(_unlist(y)) == 1)
                  else "none")
    add = None
    if adjust == "add.max":
        add = max(_unlist(x))
        y = dendrapply(y, _add_ifleaf, add)
    xtr = list(others)
    for e in xtr:
        if not isinstance(e, Dendrogram):
            raise TypeError('extra argument is not of class "dendrogram"')
    r_children = [x, y]
    if xtr:
        if adjust == "add.max":
            add = max(add, max(_unlist(y)))
            for i in range(len(xtr)):
                if i > 0:
                    add = max(add, max(_unlist(xtr[i - 1])))
                xtr[i] = dendrapply(xtr[i], _add_ifleaf, add)
        r_children = r_children + xtr
    r = Dendrogram(children=r_children)
    r.attrs["members"] = sum(ch.attrs.get("members") for ch in r_children)
    h_max = max(ch.attrs.get("height") for ch in r_children)
    if height is None:
        height = 1.1 * h_max
    elif height < h_max:
        raise ValueError(
            f"'height' must be at least {h_max}, "
            "the maximal height of its components")
    r.attrs["height"] = height
    return midcache_dendrogram(r, quiet=True)


def dendrapply(X, FUN, *args, **kwargs):
    """R ``dendrapply(X, FUN, ...)`` (``cluster_dendrogram.R:825``) — apply ``FUN``
    to every node (the node first, then its children recursively replace the
    node's branches while keeping ``FUN(node)``'s attributes)."""
    if not isinstance(X, Dendrogram):
        raise TypeError("'X' is not a dendrogram")

    def napply(d):
        r = FUN(d, *args, **kwargs)
        if not is_leaf(d):
            new_children = [napply(c) for c in d.children]
            if isinstance(r, Dendrogram):
                r = Dendrogram(children=new_children, value=r.value,
                               attrs=dict(r.attrs))
            else:
                r = Dendrogram(children=new_children)
        return r

    return napply(X)


def cut_dendrogram(x, h):
    """R ``cut.dendrogram(x, h)`` (``cluster_dendrogram.R:644``) — cut at height
    ``h`` into ``{upper, lower}``: every subtree with ``height <= h`` becomes a
    leaf ``"Branch k"`` in ``upper`` and is collected (whole) into ``lower``."""
    lower = []
    counter = [1]

    def assign_nodes(subtree):
        if is_leaf(subtree):
            return subtree
        K = len(subtree.children)
        if K == 0:
            raise ValueError("non-leaf subtree of length 0")
        new_children = []
        new_mem = 0
        for k in range(K):
            sub = subtree.children[k]
            if sub.attrs.get("height") <= h:
                X = counter[0]
                at = dict(sub.attrs)                  # attributes(sub)
                at["leaf"] = True
                at.pop("class", None)
                at["x.member"] = at.get("members")    # before members <- 1
                at["members"] = 1
                new_mem += 1
                at["label"] = f"Branch {X}"
                new_children.append(Dendrogram(children=None, value=X, attrs=at))
                lower.append(_clone(sub))             # LOWER[[X]] <- sub
                counter[0] += 1
            else:
                child = assign_nodes(sub)
                new_children.append(child)
                new_mem += child.attrs.get("members")
        new = Dendrogram(children=new_children, value=subtree.value,
                         attrs=dict(subtree.attrs))
        new.attrs["x.member"] = new.attrs.get("members")
        new.attrs["members"] = new_mem
        return new

    return {"upper": assign_nodes(x), "lower": lower}


def print_dendrogram(x, digits=7, _return=False):
    """R ``print.dendrogram`` (``cluster_dendrogram.R:285``) — the concise
    one-line summary (display only; not byte-pinned to R)."""
    parts = ["'dendrogram' "]
    if is_leaf(x):
        parts.append("leaf '" + _fmt_digits(x.attrs.get("label"), digits) + "'")
    else:
        parts.append(f"with {len(x)} branches and "
                     f"{x.attrs.get('members')} members total")
    parts.append(", at height " + _fmt_digits(x.attrs.get("height"), digits) + " ")
    s = "".join(parts) + "\n"
    if _return:
        return s
    print(s, end="")
    return None


def str_dendrogram(object, max_level=None, digits_d=3, give_attr=False,
                   nest_lev=0, indent_str="", last_str="`", stem="--",
                   _return=False):
    """R ``str.dendrogram`` (``cluster_dendrogram.R:298``) — the nested-tree
    text rendering (display only; not byte-pinned to R)."""
    out = []

    def pasteLis(at, dropNam):
        items = [(k, v) for k, v in at.items() if k not in dropNam]
        return ", ".join(f"{k} = {_fmt_digits(v, digits_d)}" for k, v in items)

    todo = None
    while True:
        istr = (indent_str[:-1] + last_str) if indent_str.endswith(" ") \
            else indent_str
        out.append(istr + stem)
        at = object.attrs
        memb = at.get("members")
        hgt = at.get("height")
        if not is_leaf(object):
            le = len(object)
            extra = ""
            if give_attr:
                extra = pasteLis(at, ("class", "height", "members"))
                if extra:
                    extra = ", " + extra
            tail = (" .." if (max_level is not None and nest_lev == max_level)
                    else "")
            out.append(f"[dendrogram w/ {le} branches and {memb} members at h = "
                       f"{_fmt_digits(hgt, digits_d)}{extra}]{tail}\n")
            if max_level is None or nest_lev < max_level:
                nest_lev += 1
                todo = (object.children[le - 1], nest_lev,
                        indent_str + "  ", todo)
                indent_str = indent_str + " |"
                le -= 1
                while le > 0:
                    todo = (object.children[le - 1], nest_lev, indent_str, todo)
                    le -= 1
        else:                                         # leaf
            label = at.get("label")
            if isinstance(label, str):
                out.append('leaf "' + label + '" ')
            else:
                out.append("leaf " + _fmt_digits(object.value, digits_d) + " ")
            any_at = hgt != 0
            if any_at:
                out.append("(h=" + _fmt_digits(hgt, digits_d))
            if memb != 1:
                if any_at:
                    out.append(", memb= " + str(memb))
                else:
                    any_at = True
                    out.append("(memb= " + str(memb))
            tail = pasteLis(at, ("class", "height", "members", "leaf", "label"))
            if any_at or tail:
                out.append(("" if any_at else "(") + " " + tail + " )")
            out.append("\n")
        if todo is None:
            break
        object, nest_lev, indent_str, todo = todo
    s = "".join(out)
    if _return:
        return s
    print(s, end="")
    return None


def cophenetic_dendrogram(x):
    """R ``cophenetic.dendrogram`` (``cluster_hclust.R:238``) — cophenetic
    distances by recursing over a :class:`Dendrogram`: each split fills the
    between-children block with the node's height; leaves contribute a labelled
    ``0``. Returns a :class:`~hea.R.distance.Dist`."""
    if is_leaf(x):
        label = x.attrs.get("label")
        if label is None:
            raise ValueError("need dendrograms where all leaves have labels")
        d = as_dist(np.zeros((1, 1)))
        d.Labels = [label]
        return d
    children = [cophenetic_dendrogram(ch) for ch in x.children]
    lens = [c.Size for c in children]
    total = int(sum(lens))
    m = np.full((total, total), x.attrs.get("height"), dtype=float)
    hi = np.cumsum(lens)
    lo = np.concatenate([[0], hi[:-1]]).astype(int)
    for i, c in enumerate(children):
        m[lo[i]:hi[i], lo[i]:hi[i]] = as_matrix_dist(c)
    labels = []
    for c in children:
        if c.Labels is not None:
            labels.extend(c.Labels)
    d = as_dist(m)
    d.Labels = labels
    return d
