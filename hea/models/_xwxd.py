"""mgcv's discrete ``X'WX``: ``XWXd0`` (discrete.c:2056-2273) and ``XWXd1``
driving the per-block ``XWXijs`` (discrete.c:1672-2030), with ``tensorXj``,
``Ztb``, ``indReduce``, ``row_squash`` and ``up2lo``, as mgcv.so computes them
on arm64: the Python twin of the extension's ``xwxd.rs``, which bam's
``_xwxd0_mgcv`` and ``_xwxd1_mgcv`` call when the extension is absent.

A scalar ``acc += a*b*...*z`` keeps the leading products rounded and fuses the
last multiply into the add; in ``a*b + c*d`` the left product is the fused
one. Every multiply-add is fused, on any architecture, as the port's are
(``math.fma``; without it, before Python 3.13, the twin rounds twice). The
"simplest" diagonal block's contiguous reduction ``x += wb*Xj*Xi`` multiplies
eight terms at a time and adds them in order, fusing only the remainder. The
cross products are BLAS ``dgemm`` calls with the C's arguments
(:mod:`hea.models._mgcv_c`).

Layout (mgcv's): ``x`` is ``unlist(X)``, each marginal column-major and
``m[q] x p[q]``; ``k`` is the ``n x nk`` index matrix column-major and
0-based; ``ks`` is ``nx x 2`` column-major (start, end) into the columns of
``k``; ``ts`` (0-based) and ``dt`` give each term's first marginal and count;
``v`` is ``unlist(v)`` and ``qc`` the constraint codes. Results are
column-major and flat.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np

from .._rfma import _R_FMA, _rfma_vec
from ._mgcv_c import _inorder, dgemm, mat

_HAVE_FMA = hasattr(math, "fma")
_MASK64 = (1 << 64) - 1

if _HAVE_FMA:

    def _fma(a, b, c):
        try:
            return math.fma(a, b, c)
        except (OverflowError, ValueError):
            return a * b + c

else:

    def _fma(a, b, c):
        return a * b + c


if _R_FMA:
    _fma_vec = _rfma_vec
elif _HAVE_FMA:
    _fma_ufunc = np.frompyfunc(_fma, 3, 1)

    def _fma_vec(a, b, c):
        return np.asarray(_fma_ufunc(a, b, c), dtype=np.float64)

else:

    def _fma_vec(a, b, c):
        return a * b + c


def _round(v) -> int:
    """Rust's ``round() as usize`` on the integer-valued entries of ``v``."""
    return math.floor(float(v) + 0.5)


def _strict8_triple(a, b, c, length: int) -> float:
    """``x = 0; for kk: x += a[kk]*b[kk]*c[kk]`` as clang vectorizes it:
    eight rounded products ``(a*b)*c`` added in order, the ``len % 8`` tail
    fused."""
    nv = length & ~7
    x = _inorder((a[:nv] * b[:nv]) * c[:nv])
    for kk in range(nv, length):
        x = _fma(float(a[kk] * b[kk]), float(c[kk]), x)
    return x


def _sm_inihash() -> list[int]:
    """``SMinihash``: the 256 hash generator words."""
    ht = []
    h = 0x987564BACF987454
    for _ in range(256):
        for _ in range(31):
            h ^= h >> 7
            h = (h ^ (h << 11)) & _MASK64
            h ^= h >> 10
        ht.append(h)
    return ht


_HT = _sm_inihash()


class _Ctx(SimpleNamespace):
    def kcol(self, c: int) -> np.ndarray:
        return self.k[c * self.n : (c + 1) * self.n]

    def ks0(self, q: int) -> int:
        return int(self.ks[q])

    def ksn(self, q: int) -> int:
        return int(self.ks[q + self.nx] - self.ks[q])

    def wl(self) -> np.ndarray:
        return self.ws[self.n - 1 :]


def _tensor_xj(cx: _Ctx, xj, q0: int, dt: int, j: int, koff: int) -> None:
    """``tensorXj``: multiply ``xj`` elementwise by column ``j`` of the row
    tensor product of ``dt`` marginals starting at marginal ``q0``, using
    index columns ``ks[q0 + i] + koff``."""
    q = 1
    for i in range(dt):
        q *= int(cx.p[q0 + i])
    jp = j
    for i in range(dt):
        qi = q0 + i
        q //= int(cx.p[qi])
        lcol = jp // q
        jp %= q
        mi = int(cx.m[qi])
        col = cx.x[cx.off[qi] + mi * lcol : cx.off[qi] + mi * (lcol + 1)]
        kp = cx.kcol(cx.ks0(qi) + koff)
        xj[: cx.n] *= col[kp]


def _ind_reduce(ka, kb, w, ws, wl, tri, n, c, a, rc, cc, ra, trans) -> None:
    """``indReduce`` with ``buffer = 1``: accumulate the sparse weight matrix
    (``w[l]`` to ``(ka[l], kb[l])``, and with ``tri`` the super/sub weights
    to the off-diagonal positions) through mgcv's hash table, read it out in
    slot order, then ``C += W A`` (``trans`` false) or ``C += W' A``. ``C``
    is ``rc x cc``, ``A`` is ``ra x cc``, both column-major."""
    nil = -1
    sm = [nil] * n
    st_i: list[int] = []
    st_j: list[int] = []
    st_w: list[float] = []
    st_next: list[int] = []
    ka = ka.tolist()
    kb = kb.tolist()
    w = w.tolist()
    ws = ws.tolist()
    wl = wl.tolist()
    ntri = 3 if tri else 1
    for l in range(n):
        t = 0
        while t < ntri:
            if t == 0:
                i, j, wij = ka[l], kb[l], w[l]
            elif t == 1:
                i, j, wij = ka[l], kb[l + 1], ws[l]
            else:
                if l + 2 == n:
                    ntri = 1
                i, j, wij = ka[l + 1], kb[l], wl[l]
            h = 0x99EFB145DAA48450
            for byte in struct.pack("=ii", i, j):
                h = ((h * 7664345821815920749) & _MASK64) ^ _HT[byte]
            slot = h % n
            e = sm[slot]
            while e != nil:
                if st_i[e] == i and st_j[e] == j:
                    st_w[e] += wij
                    break
                e = st_next[e]
            if e == nil:
                st_i.append(i)
                st_j.append(j)
                st_w.append(wij)
                st_next.append(sm[slot])
                sm[slot] = len(st_i) - 1
            t += 1
    ko: list[int] = []
    lo: list[int] = []
    wo: list[float] = []
    for head in sm:
        e = head
        while e != nil:
            ko.append(st_i[e])
            lo.append(st_j[e])
            wo.append(st_w[e])
            e = st_next[e]
    # column by column in the C; the columns are independent, so all of
    # them at once, entry by entry
    C = mat(c, rc, cc)
    A = mat(a, ra, cc)
    if trans:
        for e in range(len(wo)):
            C[lo[e], :] = _fma_vec(A[ko[e], :], wo[e], C[lo[e], :])
    else:
        for e in range(len(wo)):
            C[ko[e], :] = _fma_vec(A[lo[e], :], wo[e], C[ko[e], :])


def _xwxijs(cx: _Ctx, xwx, base: int, nxwx: int, i: int, j: int, r: int, c: int):
    """``XWXijs``: sub-block ``(r, c)`` of the term block ``(i, j)`` of
    ``X'WX``, written at ``xwx[base:]`` with leading dimension ``nxwx``."""
    n = cx.n
    tri = cx.tri
    w = cx.w
    ts_i, ts_j = int(cx.ts[i]), int(cx.ts[j])
    dt_i, dt_j = int(cx.dt[i]), int(cx.dt[j])
    si = cx.ksn(ts_i)
    im = ts_i + dt_i - 1
    mim = int(cx.m[im])
    # row-tensor work columns, needed only by tensor terms
    tens = dt_i > 1 or dt_j > 1
    dxi = np.ones(n) if tens else None
    dxj = np.ones(n) if tens else None

    def at(row: int, col: int) -> int:
        return base + row + nxwx * col

    if dt_i == 1 and dt_j == 1 and int(cx.m[ts_i]) == n and int(cx.m[ts_j]) == n:
        # both stored matrices have n rows
        jm = ts_j
        mjm = int(cx.m[jm])
        pim = int(cx.p[im])
        pjm = int(cx.p[jm])
        sj = cx.ksn(ts_j)
        for ii in range(pim):
            for jj in range(pjm):
                xwx[at(ii, jj)] = 0.0
        for s in range(si):
            for t in range(sj):
                ki = cx.kcol(cx.ks0(im) + s)
                kj = cx.kcol(cx.ks0(jm) + t)
                for ii in range(pim):
                    xik = cx.x[cx.off[im] + mim * ii :][ki]
                    jj0 = ii if i == j else 0
                    for jj in range(jj0, pjm):
                        xjk = cx.x[cx.off[jm] + mjm * jj :][kj]
                        if tri:
                            ws, wl = cx.ws, cx.wl()
                            x = float(xik[0]) * _fma(
                                float(xjk[0]), float(w[0]), float(xjk[1] * ws[0])
                            )
                            s2 = _fma_vec(wl[: n - 1], xjk[: n - 1], w[1:n] * xjk[1:n])
                            s3 = _fma_vec(ws[1 : n - 1], xjk[2:n], s2[: n - 2])
                            for q, v in zip(xik[1 : n - 1].tolist(), s3.tolist()):
                                x = _fma(q, v, x)
                            x = _fma(float(xik[n - 1]), float(s2[n - 2]), x)
                        else:
                            x = 0.0
                            for a, b in zip((w[:n] * xik).tolist(), xjk.tolist()):
                                x = _fma(a, b, x)
                        if i == j:
                            v = xwx[at(ii, jj)] + x
                            xwx[at(ii, jj)] = v
                            xwx[at(jj, ii)] = v
                        else:
                            xwx[at(ii, jj)] += x
    elif not tri and i == j and si == 1:
        # simplest setup: accumulate the diagonal weights
        wb = np.zeros(mim)
        if dt_i > 1:
            _tensor_xj(cx, dxi, ts_i, dt_i - 1, r, 0)
            if r != c:
                _tensor_xj(cx, dxj, ts_i, dt_i - 1, c, 0)
        kf = cx.kcol(cx.ks0(im))
        if dt_i > 1:
            dxjr = dxj if r != c else dxi
            for b, a, ww in zip(kf.tolist(), (dxi * dxjr).tolist(), w[:n].tolist()):
                wb[b] = _fma(a, ww, float(wb[b]))
        else:
            np.add.at(wb, kf, w[:n])
        xt = cx.x[cx.off[im] :]
        pim = int(cx.p[im])

        def col(q: int) -> np.ndarray:
            return xt[mim * q : mim * (q + 1)]

        if r != c:
            for jj in range(pim):
                for ii in range(pim):
                    x = _strict8_triple(wb, col(jj), col(ii), mim)
                    xwx[at(c * pim + jj, r * pim + ii)] = x
                    xwx[at(r * pim + ii, c * pim + jj)] = x
        else:
            for ii in range(pim):
                for jj in range(ii, pim):
                    x = _strict8_triple(wb, col(jj), col(ii), mim)
                    xwx[at(r * pim + ii, c * pim + jj)] = x
                    xwx[at(r * pim + jj, c * pim + ii)] = x
                    xwx[base + (r * pim + ii) * nxwx + c * pim + jj] = x
                    xwx[base + (r * pim + jj) * nxwx + c * pim + ii] = x
    else:
        _general_block(cx, xwx, base, nxwx, i, j, r, c, dxi, dxj)


def _general_block(cx: _Ctx, xwx, base, nxwx, i, j, r, c, dxi, dxj) -> None:
    """The general branch of ``XWXijs`` (discrete.c:1796-2028)."""
    n = cx.n
    tri = cx.tri
    w = cx.w
    ts_i, ts_j = int(cx.ts[i]), int(cx.ts[j])
    dt_i, dt_j = int(cx.dt[i]), int(cx.dt[j])
    si = cx.ksn(ts_i)
    sj = cx.ksn(ts_j)
    im = ts_i + dt_i - 1
    jm = ts_j + dt_j - 1
    mim, mjm = int(cx.m[im]), int(cx.m[jm])
    pim, pjm = int(cx.p[im]), int(cx.p[jm])
    ddti, ddtj = dt_i - 1, dt_j - 1
    tensi, tensj = ddti > 0, ddtj > 0
    acc_w = n > mjm * mim
    if acc_w:
        rfac = not (
            pim * mim * mjm + pim * pjm * mjm > mim * mjm * pjm + pim * pjm * mim
        )
    else:
        ii = 2 if tensi else 1
        if tensj:
            ii += 1
        alpha = ii * 3 + 3 if tri else ii + 1
        a = alpha * si * sj * n
        rfac = not (a * pim + mjm * pim * pjm < a * pjm + mim * pim * pjm)
        if mim == n:
            rfac = False
        elif mjm == n:
            rfac = True
    wbar = np.zeros(mim * mjm) if acc_w else None
    cmat = np.zeros(mim * pjm) if rfac else None
    dmat = None if rfac else np.zeros(mjm * pim)
    ws_all = cx.ws
    wl_all = cx.wl() if tri else ws_all[:0]
    xi_all = cx.x[cx.off[im] : cx.off[im] + mim * pim]
    xj_all = cx.x[cx.off[jm] : cx.off[jm] + mjm * pjm]

    for s in range(si):
        for t in range(sj):
            if tensi:
                dxi[:] = 1.0
                _tensor_xj(cx, dxi, ts_i, ddti, r, s)
            if tensj:
                dxj[:] = 1.0
                _tensor_xj(cx, dxj, ts_j, ddtj, c, t)
            ki = cx.kcol(cx.ks0(im) + s)
            kj = cx.kcol(cx.ks0(jm) + t)
            if acc_w:
                _acc_wbar(
                    wbar, w, ws_all, wl_all, dxi, dxj, ki, kj, n, mim, tri, tensi, tensj
                )
            elif (rfac and pjm > 15) or (not rfac and pim > 15):
                # sparse accumulation of \bar W through indReduce
                wi, wsi, wli = _bar_w(w, ws_all, wl_all, dxi, dxj, n, tri, tensi, tensj)
                if rfac:
                    _ind_reduce(
                        ki, kj, wi, wsi, wli, tri, n, cmat, xj_all, mim, pjm, mjm, False
                    )
                else:
                    _ind_reduce(
                        ki, kj, wi, wsi, wli, tri, n, dmat, xi_all, mjm, pim, mim, True
                    )
            elif rfac:
                # direct accumulation of C = \bar W X_j
                _acc_direct(
                    mat(cmat, mim, pjm), mat(xj_all, mjm, pjm), w, ws_all, wl_all,
                    dxi, dxj, ki, kj, n, tri, tensi, tensj, False,
                )  # fmt: skip
            else:
                # direct accumulation of D = \bar W' X_i
                _acc_direct(
                    mat(dmat, mjm, pim), mat(xi_all, mim, pim), w, ws_all, wl_all,
                    dxi, dxj, ki, kj, n, tri, tensi, tensj, True,
                )  # fmt: skip
    if acc_w:
        if rfac:
            dgemm(
                b"N", b"N", mim, pjm, mjm, 1.0, wbar, mim, xj_all, mjm, 0.0, cmat, mim
            )
        else:
            dgemm(
                b"T", b"N", mjm, pim, mim, 1.0, wbar, mim, xi_all, mim, 0.0, dmat, mjm
            )
    out = xwx[base + r * pim + c * pjm * nxwx :]
    if rfac:
        dgemm(b"T", b"N", pim, pjm, mim, 1.0, xi_all, mim, cmat, mim, 0.0, out, nxwx)
    else:
        dgemm(b"T", b"N", pim, pjm, mjm, 1.0, dmat, mjm, xj_all, mjm, 0.0, out, nxwx)


def _acc_wbar(wbar, w, ws, wl, dxi, dxj, ki, kj, n, mim, tri, tensi, tensj) -> None:
    """``\\bar W`` (``mim x mjm``) accumulated observation by observation."""
    a3s = (ki + mim * kj).tolist()
    w = w[:n].tolist()
    if tri:
        a1s = (ki[: n - 1] + mim * kj[1:n]).tolist()
        a2s = (ki[1:n] + mim * kj[: n - 1]).tolist()
        sp_, lp_ = ws[: n - 1].tolist(), wl[: n - 1].tolist()
        if not tensi and not tensj:
            # the three adds of each observation in turn
            idx = np.empty(3 * (n - 1) + 1, dtype=np.int64)
            val = np.empty(3 * (n - 1) + 1)
            idx[0:-1:3], idx[1:-1:3], idx[2:-1:3] = a1s, a2s, a3s[: n - 1]
            val[0:-1:3], val[1:-1:3], val[2:-1:3] = sp_, lp_, w[: n - 1]
            idx[-1], val[-1] = a3s[n - 1], w[n - 1]
            np.add.at(wbar, idx, val)
            return
        di = dxi.tolist() if tensi else None
        dj = dxj.tolist() if tensj else None
        for q in range(n - 1):
            sp, wp, lp = sp_[q], w[q], lp_[q]
            a1, a2, a3 = a1s[q], a2s[q], a3s[q]
            if tensi and tensj:
                wbar[a1] = _fma(sp * di[q], dj[q + 1], wbar[a1])
                wbar[a2] = _fma(lp * di[q + 1], dj[q], wbar[a2])
                wbar[a3] = _fma(wp * di[q], dj[q], wbar[a3])
            elif tensi:
                wbar[a1] = _fma(sp, di[q], wbar[a1])
                wbar[a2] = _fma(lp, di[q + 1], wbar[a2])
                wbar[a3] = _fma(wp, di[q], wbar[a3])
            else:
                wbar[a1] = _fma(sp, dj[q + 1], wbar[a1])
                wbar[a2] = _fma(lp, dj[q], wbar[a2])
                wbar[a3] = _fma(wp, dj[q], wbar[a3])
        q = n - 1
        a3, wp = a3s[q], w[q]
        if tensi and tensj:
            wbar[a3] = _fma(wp * di[q], dj[q], wbar[a3])
        elif tensi:
            wbar[a3] = _fma(wp, di[q], wbar[a3])
        else:
            wbar[a3] = _fma(wp, dj[q], wbar[a3])
    elif not tensi and not tensj:
        np.add.at(wbar, a3s, w)
    else:
        di = dxi.tolist() if tensi else None
        dj = dxj.tolist() if tensj else None
        for q in range(n):
            a = a3s[q]
            if tensi and tensj:
                wbar[a] = _fma(w[q] * di[q], dj[q], wbar[a])
            elif tensi:
                wbar[a] = _fma(w[q], di[q], wbar[a])
            else:
                wbar[a] = _fma(w[q], dj[q], wbar[a])


def _bar_w(w, ws, wl, dxi, dxj, n, tri, tensi, tensj):
    """The weights ``indReduce`` accumulates: the diagonal and, with ``tri``,
    the super- and sub-diagonal, each times the row tensor factors."""
    wi = np.zeros(n)
    wsi = np.zeros(n if tri else 0)
    wli = np.zeros(n if tri else 0)
    if tri:
        m1 = n - 1
        if tensi and tensj:
            wi[:m1] = w[:m1] * dxi[:m1] * dxj[:m1]
            wsi[:m1] = ws[:m1] * dxi[:m1] * dxj[1:n]
            wli[:m1] = wl[:m1] * dxi[1:n] * dxj[:m1]
        elif tensi:
            wi[:m1] = w[:m1] * dxi[:m1]
            wsi[:m1] = ws[:m1] * dxi[:m1]
            wli[:m1] = wl[:m1] * dxi[1:n]
        elif tensj:
            wi[:m1] = w[:m1] * dxj[:m1]
            wsi[:m1] = ws[:m1] * dxj[1:n]
            wli[:m1] = wl[:m1] * dxj[:m1]
        else:
            wi[:m1] = w[:m1]
            wsi[:m1] = ws[:m1]
            wli[:m1] = wl[:m1]
        q = n - 1
        if tensi and tensj:
            wi[q] = w[q] * dxi[q] * dxj[q]
        elif tensi:
            wi[q] = w[q] * dxi[q]
        elif tensj:
            wi[q] = w[q] * dxj[q]
        else:
            wi[q] = w[q]
    elif tensi and tensj:
        wi[:] = w[:n] * dxi[:n] * dxj[:n]
    elif tensi:
        wi[:] = w[:n] * dxi[:n]
    elif tensj:
        wi[:] = w[:n] * dxj[:n]
    else:
        wi[:] = w[:n]
    return wi, wsi, wli


def _acc_direct(
    out, xm, w, ws, wl, dxi, dxj, ki, kj, n, tri, tensi, tensj, trans
) -> None:
    """``C = \\bar W X_j`` (``trans`` false; ``out`` is ``C``, ``xm`` is
    ``X_j``) or ``D = \\bar W' X_i`` accumulated observation by observation.
    The C runs the observations within each column of ``out``; the columns
    are independent, so they are taken all at once."""
    w = w[:n].tolist()
    ki = ki.tolist()
    kj = kj.tolist()
    di = dxi.tolist() if tensi else None
    dj = dxj.tolist() if tensj else None
    if trans:
        # D: rows of out are indexed by kj, rows of xm by ki
        if tri:
            ws, wl = ws.tolist(), wl.tolist()
            for l in range(n - 1):
                a, a1, b, b1 = ki[l], ki[l + 1], kj[l], kj[l + 1]
                xa, xa1 = xm[a, :], xm[a1, :]
                if tensi and tensj:
                    inner = _fma_vec(di[l + 1] * wl[l], xa1, (di[l] * w[l]) * xa)
                    out[b, :] = _fma_vec(dj[l], inner, out[b, :])
                    out[b1, :] = _fma_vec(ws[l] * di[l] * dj[l + 1], xa, out[b1, :])
                elif tensi:
                    s0 = _fma_vec(wl[l] * di[l + 1], xa1, (w[l] * di[l]) * xa)
                    out[b, :] += s0
                    out[b1, :] = _fma_vec(ws[l] * di[l], xa, out[b1, :])
                elif tensj:
                    inner = _fma_vec(wl[l], xa1, w[l] * xa)
                    out[b, :] = _fma_vec(dj[l], inner, out[b, :])
                    out[b1, :] = _fma_vec(ws[l] * dj[l + 1], xa, out[b1, :])
                else:
                    s0 = _fma_vec(wl[l], xa1, w[l] * xa)
                    out[b, :] += s0
                    out[b1, :] = _fma_vec(ws[l], xa, out[b1, :])
            l = n - 1
            a, b = ki[l], kj[l]
            if tensi and tensj:
                f = dj[l] * di[l] * w[l]
            elif tensi:
                f = w[l] * di[l]
            elif tensj:
                f = dj[l] * w[l]
            else:
                f = w[l]
            out[b, :] = _fma_vec(f, xm[a, :], out[b, :])
            return
        for l in range(n):
            a, b = ki[l], kj[l]
            if tensi and tensj:
                f = w[l] * di[l] * dj[l]
            elif tensi:
                f = w[l] * di[l]
            elif tensj:
                f = w[l] * dj[l]
            else:
                f = w[l]
            out[b, :] = _fma_vec(f, xm[a, :], out[b, :])
        return
    # C: rows of out are indexed by ki, rows of xm by kj
    if tri:
        ws, wl = ws.tolist(), wl.tolist()
        for l in range(n - 1):
            a, a1, b, b1 = ki[l], ki[l + 1], kj[l], kj[l + 1]
            xb, xb1 = xm[b, :], xm[b1, :]
            if tensi and tensj:
                inner = _fma_vec(ws[l] * dj[l + 1], xb1, (w[l] * dj[l]) * xb)
                out[a, :] = _fma_vec(di[l], inner, out[a, :])
                out[a1, :] = _fma_vec(wl[l] * di[l + 1] * dj[l], xb, out[a1, :])
            elif tensi:
                inner = _fma_vec(ws[l], xb1, w[l] * xb)
                out[a, :] = _fma_vec(di[l], inner, out[a, :])
                out[a1, :] = _fma_vec(wl[l] * di[l + 1], xb, out[a1, :])
            elif tensj:
                s0 = _fma_vec(ws[l] * dj[l + 1], xb1, (w[l] * dj[l]) * xb)
                out[a, :] += s0
                out[a1, :] = _fma_vec(wl[l] * dj[l], xb, out[a1, :])
            else:
                s0 = _fma_vec(ws[l], xb1, w[l] * xb)
                out[a, :] += s0
                out[a1, :] = _fma_vec(wl[l], xb, out[a1, :])
        l = n - 1
        a, b = ki[l], kj[l]
        if tensi and tensj:
            f = w[l] * di[l] * dj[l]
        elif tensi:
            f = di[l] * w[l]
        elif tensj:
            f = w[l] * dj[l]
        else:
            f = w[l]
        out[a, :] = _fma_vec(f, xm[b, :], out[a, :])
        return
    for l in range(n):
        a, b = ki[l], kj[l]
        if tensi and tensj:
            f = w[l] * di[l] * dj[l]
        elif tensi:
            f = w[l] * di[l]
        elif tensj:
            f = w[l] * dj[l]
        else:
            f = w[l]
        out[a, :] = _fma_vec(f, xm[b, :], out[a, :])


def _ztb(a, o1: int, o0: int, v, qc: int, di: int, p: int) -> None:
    """``Ztb``: ``b1 = Z' b0`` for the ``p`` elements of ``b0`` spaced
    ``di`` apart, written ``di`` apart from ``o1``; ``b0`` starts at ``o0``
    of the same array. Householder (``qc > 0``): ``(I - vv') b0`` with the
    first element dropped; ``qc < 0``: the Kronecker product of sum-to-zero
    contrasts."""
    if qc > 0:
        b0 = a[o0 : o0 + (p - 1) * di + 1 : di].tolist()
        vv = v[:p].tolist()
        x = 0.0
        for bl, vl in zip(b0, vv):
            x = _fma(bl, vl, x)
        for l in range(1, p):
            a[o1 + (l - 1) * di] = _fma(-vv[l], x, b0[l])
    elif qc < 0:
        w0 = a[o0 : o0 + (p - 1) * di + 1 : di].tolist()
        w1 = [0.0] * p
        mm = _round(v[0])
        pp = p
        for i in range(mm):
            pp //= _round(v[i + 1])
        k1 = p
        for kk in range(mm + 1):
            if kk < mm:
                mk = _round(v[kk + 1])
                mk1 = mk - 1
            else:
                mk = mk1 = pp
            p1m = k1 // mk
            q = 0
            for i in range(p1m):
                z = w0[i + (mk - 1) * p1m] if kk < mm else 0.0
                for jj in range(mk1):
                    w1[q] = w0[i + jj * p1m] - z
                    q += 1
            if kk < mm:
                k1 -= p1m
            w0, w1 = w1, w0
        for l in range(k1):
            a[o1 + l * di] = w0[l]


def _revsort(a: list[float], ib: list[int]) -> None:
    """R's ``revsort`` (sort.c): ``a`` into descending order by heapsort,
    ``ib`` carried alongside."""
    n = len(a)
    if n <= 1:
        return
    # 1-based, as R writes it
    lo = (n >> 1) + 1
    ir = n
    while True:
        if lo > 1:
            lo -= 1
            ra = a[lo - 1]
            ii = ib[lo - 1]
        else:
            ra = a[ir - 1]
            ii = ib[ir - 1]
            a[ir - 1] = a[0]
            ib[ir - 1] = ib[0]
            ir -= 1
            if ir == 1:
                a[0] = ra
                ib[0] = ii
                return
        i = lo
        j = lo << 1
        while j <= ir:
            if j < ir and a[j - 1] > a[j]:
                j += 1
            if ra > a[j - 1]:
                a[i - 1] = a[j - 1]
                ib[i - 1] = ib[j - 1]
                i = j
                j += i
            else:
                j = ir + 1
        a[i - 1] = ra
        ib[i - 1] = ii


def _layout(m, p, ts, dt, v, qc):
    """Each term's coefficient count ``pt``, its last marginal's ``pd``, the
    marginals' offsets into ``x`` and the terms' into ``v``."""
    nx, nt = len(m), len(ts)
    pt = [0] * nt
    pd = [0] * nt
    off = [0] * (nx + 1)
    voff = [0] * (nt + 1)
    q = 0
    for i in range(nt):
        dti = int(dt[i])
        for jj in range(dti):
            if jj == dti - 1:
                pd[i] = int(p[q])
            off[q + 1] = off[q] + int(p[q]) * int(m[q])
            pt[i] = int(p[q]) if jj == 0 else pt[i] * int(p[q])
            q += 1
        if qc[i] == 0:
            voff[i + 1] = voff[i]
        elif qc[i] > 0:
            voff[i + 1] = voff[i] + pt[i]
        else:
            voff[i + 1] = voff[i] + _round(v[voff[i]]) + 2
    return pt, pd, off, voff


def _constrained(i, pt, qc, v, voff) -> int:
    """Term ``i``'s coefficient count after its constraint."""
    if qc[i] == 0:
        return pt[i]
    if qc[i] > 0:
        return pt[i] - 1
    si = _round(v[voff[i]])
    return pt[i] - _round(v[voff[i] + si + 1])


def _ctx(x, w_in, k, ks, m, p, ts, dt, off, ar_weights) -> _Ctx:
    """The shared context, the AR1 tridiagonal weights formed as ``XWXd0``
    forms them."""
    n = w_in.size
    tri = ar_weights.size > 0
    if tri:
        sw = np.sqrt(w_in)
        arw = ar_weights
        ws = np.zeros(2 * n - 2)
        val = arw[1 : 2 * n - 1 : 2] * arw[2 : 2 * n : 2] * sw[1:n] * sw[: n - 1]
        ws[: n - 1] = val
        ws[n - 1 :] = val
        wd = sw.copy()
        a = _fma_vec(
            arw[1 : 2 * n - 1 : 2],
            arw[1 : 2 * n - 1 : 2],
            arw[0 : 2 * n - 2 : 2] * arw[0 : 2 * n - 2 : 2],
        )
        wd[: n - 1] *= a * sw[: n - 1]
        i = n - 1
        wd[i] *= sw[i] * arw[2 * i] * arw[2 * i]
        w, ws_ = wd, ws
    else:
        w, ws_ = w_in.copy(), np.zeros(0)
    return _Ctx(
        x=x, k=k, ks=ks, m=m, p=p, ts=ts, dt=dt, nx=len(m), n=n, off=off, w=w,
        ws=ws_, tri=tri,
    )  # fmt: skip


def _constrain(xwx, nxwx, pairs, pt, qc, dt, v, voff, tps_r, tpsu_r, tps_c, tpsu_c):
    """The constraints applied to the raw blocks of each ``(r, c)`` term
    pair, columns then rows (``Ztb``), and the blocks moved to their
    constrained positions."""
    for r, c in pairs:
        if dt[r] > 1 and qc[r] != 0:
            for jj in range(pt[c]):
                o0 = tpsu_r[r] + (tpsu_c[c] + jj) * nxwx
                o1 = tps_r[r] + (tpsu_c[c] + jj) * nxwx
                _ztb(xwx, o1, o0, v[voff[r] :], int(qc[r]), 1, pt[r])
            pa = pt[r] - 1
        else:
            pa = pt[r]
            if tpsu_r[r] != tps_r[r]:
                for jj in range(pt[c]):
                    o0 = tpsu_r[r] + (tpsu_c[c] + jj) * nxwx
                    o1 = tps_r[r] + (tpsu_c[c] + jj) * nxwx
                    xwx[o1 : o1 + pt[r]] = xwx[o0 : o0 + pt[r]].copy()
        if dt[c] > 1 and qc[c] != 0:
            for jj in range(pa):
                o0 = tps_r[r] + jj + tpsu_c[c] * nxwx
                o1 = tps_r[r] + jj + tps_c[c] * nxwx
                _ztb(xwx, o1, o0, v[voff[c] :], int(qc[c]), nxwx, pt[c])
        elif tpsu_c[c] != tps_c[c]:
            for jj in range(pa):
                o0 = tps_r[r] + jj + tpsu_c[c] * nxwx
                o1 = tps_r[r] + jj + tps_c[c] * nxwx
                src = xwx[o0 : o0 + (pt[c] - 1) * nxwx + 1 : nxwx].copy()
                xwx[o1 : o1 + (pt[c] - 1) * nxwx + 1 : nxwx] = src


def _check_index(k, ks, m, n: int, name: str) -> None:
    nx = len(m)
    for q in range(nx):
        c0, c1 = int(ks[q]), int(ks[q + nx])
        if c0 < 0 or c1 < c0 or c1 * n > k.size:
            raise ValueError(f"{name}: bad index columns")
        col = k[c0 * n : c1 * n]
        if col.size and (int(col.min()) < 0 or int(col.max()) >= int(m[q])):
            raise IndexError(f"{name}: an index exceeds its marginal's rows")


def _args(x, w, k, ks, m, p, ts, dt, v, qc, ar_weights):
    f = np.ascontiguousarray
    return (
        f(x, dtype=np.float64), f(w, dtype=np.float64), f(k, dtype=np.int32),
        f(ks, dtype=np.int32), f(m, dtype=np.int32), f(p, dtype=np.int32),
        f(ts, dtype=np.int32), f(dt, dtype=np.int32), f(v, dtype=np.float64),
        f(qc, dtype=np.int32), f(ar_weights, dtype=np.float64),
    )  # fmt: skip


def xwxd0(x, w, k, ks, m, p, ts, dt, v, qc, ar_weights) -> tuple[np.ndarray, int]:
    """mgcv's ``XWXd0``: the full ``X'WX`` (``ptot x ptot``, column-major,
    flat) and ``ptot``."""
    x, w, k, ks, m, p, ts, dt, v, qc, ar_weights = _args(
        x, w, k, ks, m, p, ts, dt, v, qc, ar_weights
    )
    n, nx, nt = w.size, m.size, ts.size
    if ks.size != 2 * nx or p.size != nx or ts.size != dt.size or ts.size != qc.size:
        raise ValueError("xwxd0: inconsistent dimensions")
    _check_index(k, ks, m, n, "xwxd0")
    with np.errstate(all="ignore"):
        pt, pd, off, voff = _layout(m, p, ts, dt, v, qc)
        tps = [0] * (nt + 1)
        tpsu = [0] * (nt + 1)
        for i in range(nt):
            tps[i + 1] = tps[i] + _constrained(i, pt, qc, v, voff)
            tpsu[i + 1] = tpsu[i] + pt[i]
        ptot, nxwx = tps[nt], tpsu[nt]
        cx = _ctx(x, w, k, ks, m, p, ts, dt, off, ar_weights)
        xwx = np.zeros(nxwx * nxwx)
        for rb in range(nt):
            for cb in range(rb, nt):
                si = cx.ksn(int(ts[rb]))
                rt = pt[rb] // pd[rb]
                ct = pt[cb] // pd[cb]
                base = tpsu[rb] + nxwx * tpsu[cb]
                if rb == cb and si == 1 and not cx.tri:
                    for r in range(rt):
                        for c in range(r, rt):
                            _xwxijs(cx, xwx, base, nxwx, rb, cb, r, c)
                else:
                    for r in range(rt):
                        for c in range(ct):
                            _xwxijs(cx, xwx, base, nxwx, rb, cb, r, c)
        pairs = [(r, c) for r in range(nt) for c in range(r, nt)]
        _constrain(xwx, nxwx, pairs, pt, qc, dt, v, voff, tps, tpsu, tps, tpsu)
        out = mat(xwx, nxwx, nxwx)[:ptot, :ptot].copy(order="F")
        iu = np.triu_indices(ptot, 1)
        out.T[iu] = out[iu]
    return out.ravel(order="F"), ptot


def xwxd1(
    x, w, k, ks, m, p, ts, dt, v, qc, ar_weights, rs, cs
) -> tuple[np.ndarray, int, int]:
    """mgcv's ``XWXd1`` (the ``lt``/``rt`` path of ``XWXd``): the blocks of
    ``X'WX`` for the row terms ``rs`` against the column terms ``cs``
    (0-based), an empty side meaning the other, as a ``ptot x ctot`` matrix
    (column-major, flat), with ``ptot`` and ``ctot``. Each block is
    constrained as ``XWXd0`` does. With one side given (``cs`` is then
    ``rs``) a single-index diagonal block forms only its upper sub-blocks
    and the result is mirrored; otherwise every sub-block is formed. mgcv
    squashes ``ptot`` columns out of the row padding, which leaves later
    columns in place when the column terms are wider; all ``ctot`` are
    squashed here."""
    x, w, k, ks, m, p, ts, dt, v, qc, ar_weights = _args(
        x, w, k, ks, m, p, ts, dt, v, qc, ar_weights
    )
    rs = [int(t) for t in rs]
    cs = [int(t) for t in cs]
    n, nx, nt = w.size, m.size, ts.size
    if ks.size != 2 * nx or p.size != nx or dt.size != nt or qc.size != nt:
        raise ValueError("xwxd1: inconsistent dimensions")
    if not rs and not cs:
        raise ValueError("xwxd1: give the row terms, the column terms or both")
    if any(t >= nt for t in rs + cs):
        raise IndexError("xwxd1: no such term")
    _check_index(k, ks, m, n, "xwxd1")
    if not rs:
        rs, symmetric = list(cs), True
    elif not cs:
        cs, symmetric = list(rs), True
    else:
        symmetric = False
    with np.errstate(all="ignore"):
        return _xwxd1(x, w, k, ks, m, p, ts, dt, v, qc, ar_weights, rs, cs, symmetric)


def _xwxd1(x, w, k, ks, m, p, ts, dt, v, qc, ar_weights, rs, cs, symmetric):
    nt = ts.size
    n = w.size
    pt, pd, off, voff = _layout(m, p, ts, dt, v, qc)
    # where each selected term starts, constrained and not, in the row set
    # and in the column set
    tpsr, tpsur = [0] * (nt + 1), [0] * (nt + 1)
    kk = ii = 0
    for r in rs:
        tpsr[r], tpsur[r] = kk, ii
        kk += _constrained(r, pt, qc, v, voff)
        ii += pt[r]
    ptot, nxwx = kk, ii
    tpsc, tpsuc = [0] * (nt + 1), [0] * (nt + 1)
    kk = ii = 0
    for c in cs:
        tpsc[c], tpsuc[c] = kk, ii
        kk += _constrained(c, pt, qc, v, voff)
        ii += pt[c]
    ctot, ncu = kk, ii
    cx = _ctx(x, w, k, ks, m, p, ts, dt, off, ar_weights)
    # the main blocks, their sub-block counts and rough costs; the sub-blocks
    # are formed most costly first (R's revsort), the order deciding which of
    # two overlapping mirrored writes survives
    bl_r, bl_c, sb, bl_cost = [], [], [0], []
    for ri, r in enumerate(rs):
        c0 = ri if symmetric else 0
        for c in cs[c0:]:
            bl_r.append(r)
            bl_c.append(c)
            si = cx.ksn(int(ts[r]))
            if symmetric and r == c and si == 1 and not cx.tri:
                i = pt[r] // pd[r]
                cnt = i * (i + 1) // 2
            else:
                cnt = (pt[r] // pd[r]) * (pt[c] // pd[c])
            sb.append(sb[-1] + cnt)
            # m[r] * m[c] in 32-bit arithmetic, as the C's int product
            mrc = (int(m[r]) * int(m[c]) + (1 << 31)) % (1 << 32) - (1 << 31)
            nn = min(n, mrc)
            bl_cost.append(min(float(nn) * pd[r], float(nn) * pd[c]))
    nsub = sb[-1]
    owner = [0] * nsub
    cost = [0.0] * nsub
    order = list(range(nsub))
    kb = 0
    for i in range(nsub):
        while i >= sb[kb + 1]:
            kb += 1
        owner[i] = kb
        cost[i] = bl_cost[kb]
    _revsort(cost, order)
    per_block: list[list[int]] = [[] for _ in bl_r]
    for kk in order:
        per_block[owner[kk]].append(kk)
    xwx = np.zeros(max(nxwx * max(ncu, nxwx), 1))
    for kb, subs in enumerate(per_block):
        rb, cb = bl_r[kb], bl_c[kb]
        ld = pt[rb]
        blk = np.zeros(max(pt[rb] * pt[cb], 1))
        rt = pt[rb] // pd[rb]
        ct = pt[cb] // pd[cb]
        for kk in subs:
            i = kk - sb[kb]
            if symmetric and sb[kb + 1] - sb[kb] < rt * ct:
                r = 0
                while i >= rt - r:
                    i -= rt - r
                    r += 1
                rc = (r, i + r)
            else:
                rc = (i // ct, i % ct)
            _xwxijs(cx, blk, 0, ld, rb, cb, rc[0], rc[1])
        base = tpsur[rb] + nxwx * tpsuc[cb]
        for jj in range(pt[cb]):
            xwx[base + jj * nxwx : base + jj * nxwx + ld] = blk[jj * ld : jj * ld + ld]
    pairs = []
    for ri, r in enumerate(rs):
        c0 = ri if symmetric else 0
        pairs.extend((r, c) for c in cs[c0:])
    _constrain(xwx, nxwx, pairs, pt, qc, dt, v, voff, tpsr, tpsur, tpsc, tpsuc)
    out = mat(xwx, nxwx, ctot)[:ptot, :].copy(order="F")
    if symmetric:
        iu = np.triu_indices(ptot, 1)
        out.T[iu] = out[iu]
    return out.ravel(order="F"), ptot, ctot
