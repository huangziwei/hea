"""Faithful port of R's ``src/library/stats/src/fexact.c`` — the Mehta-Patel
FEXACT network algorithm (ACM TOMS 643) that computes Fisher's exact test
p-value for an r×c contingency table.

The algorithm is deterministic (no RNG). All arithmetic is IEEE-754 double,
matching R's C bit-for-bit: the log-factorial table, the shortest/longest
path bounds, and the ``pre`` accumulation reproduce R's operation order, so
``fexact()`` is bit-exact to ``fisher.test(x)$p.value`` for r×c tables.

Because the p-value is accumulated in an order that depends on the hash-table
sizes ``ldkey``/``ldstp`` (which R derives from ``workspace`` and ``mult``),
the workspace accounting is reproduced exactly rather than replaced by dicts.

The public entry point is :func:`fexact`; the leaf routines keep the f2xact…
f11act names of the C for cross-reference.
"""

from __future__ import annotations

import math

from .._dispatch import rs_fn
from . import nmath as _nm

# Rust kernel — None when the extension is absent/disabled, in which case the
# pure-Python port below runs unchanged (bit-identical, just slower). The
# Python path is the reference oracle; tests/test_rs_parity.py pins rs == python.
_rs_fexact = rs_fn("fexact")

_INT_MAX = 2147483647
_TOL = 3.45254e-7  # sqrt of the smallest relative spacing
_AMISS = -12345.0  # returned when the probability is undefined
_LOG_2PI = 1.83787706640934548356065947281


class FexactError(RuntimeError):
    """Raised on the FEXACT workspace/stack-overflow conditions (R's
    ``FEXACT error`` family); the message suggests ``simulate_p_value=True``."""


def _f11act(arr, src, dst, i1, i2):
    """f11act — revise row totals: copy the ``src`` column to ``dst`` dropping
    the entry at 1-based position ``i1`` (``arr`` flat, columns pre-offset)."""
    for m in range(i1 - 1):
        arr[dst + m] = arr[src + m]
    for m in range(i1, i2 + 1):
        arr[dst + m - 1] = arr[src + m]


def _f8act(
    arr,
    src,
    dst,
    is_,
    i1,
    izero,
):
    """f8xact — reduce a vector that has a zero element: copy ``src`` to ``dst``
    inserting ``is_`` at its sorted position (both columns pre-offset, 1-based)."""
    i = 1
    while i < i1:
        arr[dst + i - 1] = arr[src + i - 1]
        i += 1
    while i <= izero - 1:
        if is_ >= arr[src + i]:  # irow[i+1]
            break
        arr[dst + i - 1] = arr[src + i]
        i += 1
    arr[dst + i - 1] = is_
    while True:
        i += 1
        if i > izero:
            return
        arr[dst + i - 1] = arr[src + i - 1]


def _f9xact(n, ntot, ir, fact):
    """f9xact — log of a multinomial coefficient ``log(ntot!) - sum log(ir!)``."""
    d = fact[ntot]
    for k in range(n):
        d -= fact[ir[k]]
    return d


def _f10act(nrow, irow, ncol, icol, val, fact):
    """f10act — shortest path length for special tables. ``irow``/``icol`` are
    0-based; returns ``(xmin, val)`` where ``xmin`` is True iff the shortest
    path was obtained (and ``val`` then updated)."""
    nd = [0] * nrow
    ne = [0] * ncol
    m = [0] * ncol

    is_ = icol[0] // nrow
    ix = icol[0] - nrow * is_
    ne[0] = is_
    m[0] = ix
    if ix != 0:
        nd[ix - 1] += 1
    for i in range(1, ncol):
        ix = icol[i] // nrow
        ne[i] = ix
        is_ += ix
        ix = icol[i] - nrow * ix
        m[i] = ix
        if ix != 0:
            nd[ix - 1] += 1
    for i in range(nrow - 3, -1, -1):
        nd[i] += nd[i + 1]

    ix = 0
    for i in range(nrow, 1, -1):
        ix += is_ + nd[nrow - i] - irow[i - 1]
        if ix < 0:
            return False, val
    for i in range(ncol):
        ix = ne[i]
        is_ = m[i]
        val += is_ * fact[ix + 1] + (nrow - is_) * fact[ix]
    return True, val


class _Fexact:
    """One FEXACT computation. Instance attributes carry the C ``static``
    state of f3xact (``nst``/``nitc``) and f5xact (``itp``) across their calls
    for the duration of a single :meth:`run`."""

    def __init__(self, nrow, ncol, table, expect, percnt, emin, workspace, mult):
        self.nrow = nrow
        self.ncol = ncol
        self.M = table  # M[i][j], 0-based; R's column-major matrix
        self.expect = expect
        self.percnt = percnt
        self.emin = emin
        self.workspace = workspace
        self.mult = mult
        # C statics
        self._f3_nst = 0
        self._f3_nitc = 0
        self._f5_itp = 0

    # -- f3xact static-carrying helpers are methods; leaf f6/f7 too ----------

    def run(self):
        nrow, ncol = self.nrow, self.ncol
        M = self.M
        ntot = 0
        for i in range(nrow):
            for j in range(ncol):
                if M[i][j] < 0:
                    raise FexactError("All elements of TABLE must be nonnegative.")
                ntot += M[i][j]
        if ntot == 0:
            return _AMISS

        nco = max(nrow, ncol)
        nro = min(nrow, ncol)
        k = nrow + ncol + 1
        kk = k * nco
        iwkmax = 2 * (self.workspace // 2)
        n2_stack = max(200, iwkmax // 1000)

        # Reproduce iwork()'s workspace accounting up to the hash tables so
        # that ldkey/ldstp match R exactly (the pre-accumulation order, hence
        # the last ulps of the p-value, depend on them).
        iwkpt = 0

        def _alloc(number, itype):
            nonlocal iwkpt
            if itype == 2 or itype == 3:
                iwkpt += number
            else:  # double: two int units per element
                iwkpt += number << 1

        _alloc(ntot + 1, 4)  # i1  fact
        _alloc(nco, 2)  # i2  ico
        _alloc(nco, 2)  # i3  iro
        _alloc(nco, 2)  # i3a kyy
        _alloc(nro, 2)  # i3b idif
        _alloc(nro, 2)  # i3c irn
        _alloc(max(k * 5 + (kk << 1), nco * 7 + 4 * n2_stack), 2)  # iiwk
        _alloc(max(nco + 1 + 2 * n2_stack, k), 4)  # irwk

        numb = 18 + 10 * self.mult
        ldkey = (iwkmax - iwkpt) // numb - 1
        if ldkey < 1:
            raise FexactError(
                "workspace too small for this table; increase 'workspace'"
            )
        if self.mult * ldkey > _INT_MAX:
            raise FexactError("integer overflow would happen in 'mult * ldkey'")
        ldstp = self.mult * ldkey
        self.ldkey = ldkey
        self.ldstp = ldstp
        self.n2_stack = n2_stack

        # Work arrays (1-based; index 0 is an unused sentinel).
        z = nco + 2
        self.iro = [0] * z
        self.ico = [0] * z
        self.kyy = [0] * z
        self.idif = [0] * z
        self.irn = [0] * z
        self.key = [-9999] * (2 * ldkey + 1)
        self.key2 = [-9999] * (2 * ldkey + 1)
        self.ipoin = [0] * (2 * ldkey + 1)
        self.LP = [0.0] * (2 * ldkey + 1)
        self.SP = [0.0] * (2 * ldkey + 1)
        self.tm = [0.0] * (2 * ldkey + 1)
        self.stp = [0.0] * (2 * ldstp + 1)
        self.ifrq = [0] * (6 * ldstp + 1)

        return self._f2xact(ntot, nco, nro)

    # ---------------------------------------------------------------- f2xact
    def _f2xact(self, ntot, nco_in, nro_in):
        nrow, ncol = self.nrow, self.ncol
        M = self.M
        iro, ico, kyy = self.iro, self.ico, self.kyy
        idif, irn = self.idif, self.irn
        key2, LP, SP, tm = self.key2, self.LP, self.SP, self.tm
        ldkey, ldstp = self.ldkey, self.ldstp
        tol = _TOL
        maybe_chisq = self.expect > 0.0
        expect, percnt, emin = self.expect, self.percnt, self.emin

        def TBL(i, j):  # 1-based, R column-major
            return M[i - 1][j - 1]

        nr_gt_nc = nrow > ncol
        nco = nrow if nr_gt_nc else ncol

        # Row marginals + total
        ntot = 0
        for i in range(1, nrow + 1):
            iro[i] = 0
            for j in range(1, ncol + 1):
                iro[i] += TBL(i, j)
            ntot += iro[i]
        # Column marginals
        for i in range(1, ncol + 1):
            ico[i] = 0
            for j in range(1, nrow + 1):
                ico[i] += TBL(j, i)

        iro[1 : nrow + 1] = sorted(iro[1 : nrow + 1])
        ico[1 : ncol + 1] = sorted(ico[1 : ncol + 1])

        if nr_gt_nc:
            nro = ncol
            for i in range(1, nco + 1):
                ii = iro[i]
                if i <= nro:
                    iro[i] = ico[i]
                ico[i] = ii
        else:
            nro = nrow

        # Hash-table multipliers
        kyy[1] = 1
        for i in range(1, nro):
            if iro[i] + 1 <= _INT_MAX // kyy[i]:
                kyy[i + 1] = kyy[i] * (iro[i] + 1)
            else:
                raise FexactError(
                    "the hash key would exceed the largest representable int; "
                    "consider using 'simulate_p_value=True'"
                )
        if iro[nro] + 1 > _INT_MAX // kyy[nro]:
            raise FexactError(
                "the hash key would exceed the largest representable int; "
                "consider using 'simulate_p_value=True'"
            )

        # Log factorials (R's exact recurrence, not lgamma)
        fact = [0.0] * (ntot + 1)
        if ntot >= 2:
            fact[2] = math.log(2.0)
        i = 3
        while i <= ntot:
            fact[i] = fact[i - 1] + math.log(float(i))
            j = i + 1
            if j <= ntot:
                fact[j] = fact[i] + fact[2] + fact[j // 2] - fact[j // 2 - 1]
            i += 2
        self.fact = fact

        # Observed path length
        obs = tol
        ntot = 0
        for j in range(1, nco + 1):
            dd = 0.0
            if nr_gt_nc:
                for i in range(1, nro + 1):
                    dd += fact[TBL(j, i)]
                    ntot += TBL(j, i)
            else:
                for i in range(1, nro + 1):
                    dd += fact[TBL(i, j)]
                    ntot += TBL(i, j)
            obs += fact[ico[j]] - dd

        dro = _f9xact(nro, ntot, iro[1:], fact)
        # (C also sets *prt = exp(obs - dro), the observed point probability;
        # R's Fexact returns only the p-value, so we skip it — as does the Rust.)
        pre = 0.0
        itop = 0

        # Buffer / stage pointers
        k = nco
        last = ldkey + 1
        jkey = ldkey + 1
        jstp = ldstp + 1
        jstp2 = ldstp * 3 + 1
        jstp3 = (ldstp << 2) + 1
        jstp4 = ldstp * 5 + 1
        ikkey = 0
        ikstp = 0
        ikstp2 = ldstp << 1
        ipo = 1
        self.ipoin[1] = 1
        self.stp[1] = 0.0
        self.ifrq[1] = 1
        self.ifrq[ikstp2 + 1] = -1

        stp, ifrq = self.stp, self.ifrq

        # Per-node state shared L150 -> L240 -> L300
        k1 = nro2 = nrb = 0
        ddf = drn = obs2 = obs3 = tmp = 0.0
        kval = 0
        itp = 0
        chisq = False
        kmax = kd = ks = 0
        n = 0
        ifreq = 0
        pastp = 0.0
        ipn = 0
        psh = True

        state = "Outer_Loop"
        while True:
            if state == "Outer_Loop":
                kb = nco - k + 1
                ks = 0
                n = ico[kb]
                kd = nro + 1
                kmax = nro
                for i in range(1, nro + 1):
                    idif[i] = 0
                # Generate first daughter
                while True:
                    kd -= 1
                    ntot = min(n, iro[kd])
                    idif[kd] = ntot
                    if idif[kmax] == 0:
                        kmax -= 1
                    n -= ntot
                    if not (n > 0 and kd != 1):
                        break
                if n != 0:
                    state = "L310"
                    continue
                k1 = k - 1
                n = ico[kb]
                ntot = 0
                for i in range(kb + 1, nco + 1):
                    ntot += ico[i]
                state = "L150"
                continue

            if state == "L150":
                kb = nco - k + 1
                for i in range(1, nro + 1):
                    irn[i] = iro[i] - idif[i]
                if k1 > 1:
                    irn[1 : nro + 1] = sorted(irn[1 : nro + 1])
                    nrb = 1
                    for i in range(1, nro + 1):
                        if irn[i] != 0:
                            nrb = i
                            break
                    else:
                        nrb = nro + 1
                else:
                    nrb = 1
                nro2 = nro - nrb + 1

                ddf = _f9xact(nro, n, idif[1:], fact)
                drn = _f9xact(nro2, ntot, irn[nrb:], fact) - dro + ddf

                if k1 > 1:
                    kval = irn[1]
                    for i in range(2, nro + 1):
                        kval += irn[i] * kyy[i]
                    ii_hash = kval % (ldkey << 1) + 1
                    found = False
                    for itp in range(ii_hash, (ldkey << 1) + 1):
                        ii = key2[itp]
                        if ii == kval:
                            found = True
                            break
                        elif ii < 0:
                            key2[itp] = kval
                            LP[itp] = 1.0
                            SP[itp] = 1.0
                            found = True
                            break
                    if not found:
                        for itp in range(1, ii_hash):
                            ii = key2[itp]
                            if ii == kval:
                                found = True
                                break
                            elif ii < 0:
                                key2[itp] = kval
                                LP[itp] = 1.0
                                found = True
                                break
                    if not found:
                        raise FexactError(
                            f"FEXACT error 6: LDKEY={ldkey} too small; increase "
                            "'workspace' (or use simulate_p_value=True)"
                        )
                state = "L240"
                continue

            if state == "L240":
                kb = nco - k + 1
                psh = True
                ipn = self.ipoin[ipo + ikkey]
                pastp = stp[ipn + ikstp]
                ifreq = ifrq[ipn + ikstp]
                if k1 > 1:
                    obs2 = obs - fact[ico[kb + 1]] - fact[ico[kb + 2]] - ddf
                    for i in range(3, k1 + 1):
                        obs2 -= fact[ico[kb + i]]
                    if LP[itp] > 0.0:
                        dspt = obs - obs2 - ddf
                        LP[itp] = self._f3xact(
                            nro2,
                            irn[nrb : nrb + nro2],
                            k1,
                            ico[kb + 1 : kb + 1 + k1],
                            ntot,
                            fact,
                            self.n2_stack,
                        )
                        LP[itp] = min(LP[itp], 0.0)
                        SP[itp] = self._f4xact(
                            nro2,
                            irn[nrb : nrb + nro2],
                            k1,
                            ico[kb + 1 : kb + 1 + k1],
                            dspt,
                            fact,
                            tol,
                        )
                        SP[itp] = min(SP[itp], 0.0)
                        if maybe_chisq and (irn[nrb] * ico[kb + 1]) > ntot * emin:
                            ncell = 0
                            for i in range(nro2):
                                for j in range(1, k1 + 1):
                                    if irn[nrb + i] * ico[kb + j] >= ntot * expect:
                                        ncell += 1
                            if ncell * 100 >= k1 * nro2 * percnt:
                                tmp = 0.0
                                for i in range(nro2):
                                    tmp += fact[irn[nrb + i]] - fact[irn[nrb + i] - 1]
                                tmp *= k1 - 1
                                for j in range(1, k1 + 1):
                                    tmp += (nro2 - 1) * (
                                        fact[ico[kb + j]] - fact[ico[kb + j] - 1]
                                    )
                                df = float((nro2 - 1) * (k1 - 1))
                                tmp += df * _LOG_2PI
                                tmp -= (nro2 * k1 - 1) * (fact[ntot] - fact[ntot - 1])
                                tm[itp] = (obs - dro) * -2.0 - tmp
                            else:
                                tm[itp] = -9876.0
                        else:
                            tm[itp] = -9876.0
                    obs3 = obs2 - LP[itp]
                    obs2 -= SP[itp]
                    if tm[itp] == -9876.0:
                        chisq = False
                    else:
                        chisq = True
                        tmp = tm[itp]
                else:
                    obs2 = obs - drn - dro
                    obs3 = obs2
                state = "L300"
                continue

            if state == "L300":
                kb = nco - k + 1
                if pastp <= obs3:
                    pre += float(ifreq) * math.exp(pastp + drn)
                elif pastp < obs2:
                    if chisq:
                        df = float((nro2 - 1) * (k1 - 1))
                        pv = _nm.pgamma(
                            max(0.0, tmp + (pastp + drn) * 2.0) / 2.0,
                            df / 2.0,
                            1.0,
                            False,
                            True,
                        )
                        pre += float(ifreq) * math.exp(pastp + drn + pv)
                    else:
                        itop = self._f5xact(
                            pastp + ddf,
                            kval,
                            ifreq,
                            itop,
                            jkey,
                            jstp,
                            jstp2,
                            jstp3,
                            jstp4,
                            psh,
                        )
                        psh = False
                ipn = ifrq[ipn + ikstp2]
                if ipn > 0:
                    pastp = stp[ipn + ikstp]
                    ifreq = ifrq[ipn + ikstp]
                    state = "L300"
                    continue
                ok_f7, kd, ks = self._f7xact(kmax, kd, ks)
                if ok_f7:
                    state = "L150"
                    continue
                state = "L310"
                continue

            if state == "L310":
                while True:
                    done, last, ipo = self._f6xact(nro, ikkey, ldkey, last)
                    if not done:
                        state = "Outer_Loop"
                        break
                    k -= 1
                    itop = 0
                    ikkey = jkey - 1
                    ikstp = jstp - 1
                    ikstp2 = jstp2 - 1
                    jkey = ldkey - jkey + 2
                    jstp = ldstp - jstp + 2
                    jstp2 = (ldstp << 1) + jstp
                    for i in range(1, (ldkey << 1) + 1):
                        key2[i] = -9999
                    if k < 2:
                        return pre
                continue

    # ---------------------------------------------------------------- f3xact
    def _f3xact(self, nrow, irow_s, ncol, icol_s, ntot, fact, ldst):
        """Longest path length ("LONGP"). ``irow_s``/``icol_s`` are 0-based
        marginal slices; returns the (negated) longest path."""
        # 1-based views
        irow = [0] + list(irow_s)
        icol = [0] + list(icol_s)

        if nrow <= 1:
            LP = 0.0
            if nrow > 0:
                for i in range(1, ncol + 1):
                    LP -= fact[icol[i]]
            return LP
        if ncol <= 1:
            LP = 0.0
            if ncol > 0:
                for i in range(1, nrow + 1):
                    LP -= fact[irow[i]]
            return LP
        if nrow * ncol == 4:
            n11 = (irow[1] + 1) * (icol[1] + 1) // (ntot + 2)
            n12 = irow[1] - n11
            return -(fact[n11] + fact[n12] + fact[icol[1] - n11] + fact[icol[2] - n12])

        # Test for optimal table
        val = 0.0
        if irow[nrow] <= irow[1] + ncol:
            xmin, val = _f10act(nrow, irow[1:], ncol, icol[1:], val, fact)
        else:
            xmin = False
        if (not xmin) and icol[ncol] <= icol[1] + nrow:
            xmin, val = _f10act(ncol, icol[1:], nrow, irow[1:], val, fact)
        if xmin:
            return -val

        # Dynamic-programming setup
        mx = max(nrow, ncol)
        ico = [0] * (mx + 2)
        iro = [0] * (mx + 2)
        it = [0] * (mx + 2)
        lb = [0] * (mx + 2)
        nr = [0] * (mx + 2)
        nt = [0] * (mx + 2)
        nu = [0] * (mx + 2)
        alen = [0.0] * (ncol + 2)
        itc = [0] * (2 * ldst + 1)
        ist = [-1] * (2 * ldst + 1)
        stv = [0.0] * (2 * ldst + 1)

        nn = ntot
        if nrow >= ncol:
            nro = nrow
            nco = ncol
            ico[1] = icol[1]
            nt[1] = nn - ico[1]
            for i in range(2, ncol + 1):
                ico[i] = icol[i]
                nt[i] = nt[i - 1] - ico[i]
            for i in range(1, nrow + 1):
                iro[i] = irow[i]
        else:
            nro = ncol
            nco = nrow
            ico[1] = irow[1]
            nt[1] = nn - ico[1]
            for i in range(2, nrow + 1):
                ico[i] = irow[i]
                nt[i] = nt[i - 1] - ico[i]
            for i in range(1, ncol + 1):
                iro[i] = icol[i]

        nc1s = nco - 1
        kyy = ico[nco] + 1
        irl = 1
        ks = 0
        k = ldst
        vmn = 1e100
        lev = 0
        nr1 = 0

        state = "LnewNode"
        while True:
            if state == "LnewNode":
                lev = 1
                nr1 = nro - 1
                nrt = iro[irl]
                nct = ico[1]
                lb[1] = (
                    int(
                        (float(nrt + 1) * (nct + 1)) / float(nn + nr1 * nc1s + 1) - _TOL
                    )
                    - 1
                )
                nu[1] = (
                    int((float(nrt + nc1s) * (nct + nr1)) / float(nn + nr1 + nc1s))
                    - lb[1]
                    + 1
                )
                nr[1] = nrt - lb[1]
                state = "LoopNode"
                continue

            if state == "LoopNode":
                nu[lev] -= 1
                if nu[lev] == 0:
                    if lev == 1:
                        state = "L200"
                        continue
                    lev -= 1
                    state = "LoopNode"
                    continue
                lb[lev] += 1
                nr[lev] -= 1
                while True:
                    alen[lev] = alen[lev - 1] + fact[lb[lev]]
                    if lev >= nc1s:
                        break
                    nn1 = nt[lev]
                    nrt = nr[lev]
                    lev += 1
                    nc1 = nco - lev
                    nct = ico[lev]
                    lb[lev] = int(
                        (float(nrt + 1) * (nct + 1)) / float(nn1 + nr1 * nc1 + 1) - _TOL
                    )
                    nu[lev] = int(
                        (float(nrt + nc1) * (nct + nr1)) / float(nn1 + nr1 + nc1)
                        - lb[lev]
                        + 1
                    )
                    nr[lev] = nrt - lb[lev]
                alen[nco] = alen[lev] + fact[nr[lev]]
                lb[nco] = nr[lev]
                v = val + alen[nco]
                if nro == 2:
                    v += fact[ico[1] - lb[1]] + fact[ico[2] - lb[2]]
                    for i in range(3, nco + 1):
                        v += fact[ico[i] - lb[i]]
                    vmn = min(vmn, v)
                    state = "LoopNode"
                    continue
                elif nro == 3 and nco == 2:
                    nn1 = nn - iro[irl] + 2
                    ic1 = ico[1] - lb[1]
                    ic2 = ico[2] - lb[2]
                    n11 = (iro[irl + 1] + 1) * (ic1 + 1) // nn1
                    n12 = iro[irl + 1] - n11
                    v += fact[n11] + fact[n12] + fact[ic1 - n11] + fact[ic2 - n12]
                    vmn = min(vmn, v)
                    state = "LoopNode"
                    continue
                else:
                    for i in range(1, nco + 1):
                        it[i] = max(ico[i] - lb[i], 0)
                    it[1 : nco + 1] = sorted(it[1 : nco + 1])
                    dky = float(kyy)
                    dkey = it[1] * dky + it[2]
                    for i in range(3, nco + 1):
                        dkey = it[i] + dkey * dky
                    if dkey > _INT_MAX:
                        raise FexactError(
                            "FEXACT[f3xact] hash key exceeds INT_MAX; "
                            "use simulate_p_value=True"
                        )
                    key = int(dkey)
                    ipn = key % ldst + 1
                    pushed = False
                    for itp in range(ipn, ldst + 1):
                        ii = ks + itp
                        if ist[ii] < 0:
                            ist[ii] = key
                            stv[ii] = v
                            self._f3_nst += 1
                            itc[self._f3_nst + ks] = itp
                            pushed = True
                            break
                        elif ist[ii] == key:
                            stv[ii] = min(stv[ii], v)
                            pushed = True
                            break
                    if not pushed:
                        for itp in range(1, ipn):
                            ii = ks + itp
                            if ist[ii] < 0:
                                ist[ii] = key
                                stv[ii] = v
                                self._f3_nst += 1
                                itc[self._f3_nst + ks] = itp
                                pushed = True
                                break
                            elif ist[ii] == key:
                                stv[ii] = min(stv[ii], v)
                                pushed = True
                                break
                    if not pushed:
                        raise FexactError(
                            "FEXACT error 30: stack length exceeded in f3xact; "
                            "increase 'workspace' (or use simulate_p_value=True)"
                        )
                    state = "LoopNode"
                    continue

            if state == "L200":
                if self._f3_nitc > 0:
                    itp = itc[self._f3_nitc + k] + k
                    self._f3_nitc -= 1
                    val = stv[itp]
                    key = ist[itp]
                    ist[itp] = -1
                    for i in range(nco, 1, -1):
                        ico[i] = key % kyy
                        key //= kyy
                    ico[1] = key
                    nt[1] = nn - ico[1]
                    for i in range(2, nco + 1):
                        nt[i] = nt[i - 1] - ico[i]
                    if iro[nro] <= iro[irl] + nco:
                        xmin, val = _f10act(nro, iro[irl:], nco, ico[1:], val, fact)
                    else:
                        xmin = False
                    if (not xmin) and ico[nco] <= ico[1] + nro:
                        xmin, val = _f10act(nco, ico[1:], nro, iro[irl:], val, fact)
                    if xmin:
                        vmn = min(vmn, val)
                        state = "L200"
                        continue
                    else:
                        state = "LnewNode"
                        continue
                elif nro > 2 and self._f3_nst > 0:
                    self._f3_nitc = self._f3_nst
                    self._f3_nst = 0
                    k = ks
                    ks = ldst - ks
                    nn -= iro[irl]
                    irl += 1
                    nro -= 1
                    state = "L200"
                    continue
                return -vmn

    # ---------------------------------------------------------------- f4xact
    def _f4xact(self, nrow, irow_s, ncol, icol_s, dspt, fact, tol):
        """Shortest path length ("SHORTP"). ``irow_s``/``icol_s`` are 0-based
        marginal slices; returns the (offset) shortest path."""
        irow = list(irow_s)
        icol = list(icol_s)

        if nrow == 1:
            SP = 0.0
            for i in range(ncol):
                SP -= fact[icol[i]]
            return SP
        if ncol == 1:
            SP = 0.0
            for i in range(nrow):
                SP -= fact[irow[i]]
            return SP
        if nrow * ncol == 4:
            if irow[1] <= icol[1]:
                return -(fact[irow[1]] + fact[icol[1]] + fact[icol[1] - irow[1]])
            else:
                return -(fact[icol[1]] + fact[irow[1]] + fact[irow[1] - icol[1]])

        NRP1 = nrow + ncol + 2
        IR = [0] * (nrow * NRP1)
        IC = [0] * (ncol * NRP1)
        nrstk = [0] * (NRP1 + 1)
        ncstk = [0] * (NRP1 + 1)
        lstk = [0] * (NRP1 + 1)
        mstk = [0] * (NRP1 + 1)
        nstk = [0] * (NRP1 + 1)
        ystk = [0.0] * (NRP1 + 1)

        # column istk (1-based) start offsets
        def ircol(istk):
            return (istk - 1) * nrow

        def iccol(istk):
            return (istk - 1) * ncol

        for i in range(1, nrow + 1):
            IR[ircol(1) + i - 1] = irow[nrow - i]
        for j in range(1, ncol + 1):
            IC[iccol(1) + j - 1] = icol[ncol - j]

        nro = nrow
        nco = ncol
        nrstk[1] = nro
        ncstk[1] = nco
        ystk[1] = 0.0
        y = 0.0
        istk = 1
        lvar = 1
        amx = 0.0
        SP = dspt
        m = n = 0

        state = "TOP"
        while True:
            if state == "TOP":
                ir1 = IR[ircol(istk) + 0]
                ic1 = IC[iccol(istk) + 0]
                if ir1 > ic1:
                    if nro >= nco:
                        m, n = nco - 1, 2
                    else:
                        m, n = nro, 1
                elif ir1 < ic1:
                    if nro <= nco:
                        m, n = nro - 1, 1
                    else:
                        m, n = nco, 2
                else:
                    if nro <= nco:
                        m, n = nro - 1, 1
                    else:
                        m, n = nco - 1, 2
                state = "L60"
                continue

            if state == "L60":
                if n == 1:
                    i, j = lvar, 1
                else:
                    i, j = 1, lvar
                irt = IR[ircol(istk) + i - 1]
                ict = IC[iccol(istk) + j - 1]
                y += fact[min(irt, ict)]
                if irt == ict:
                    nro -= 1
                    nco -= 1
                    _f11act(IR, ircol(istk), ircol(istk + 1), i, nro)
                    _f11act(IC, iccol(istk), iccol(istk + 1), j, nco)
                elif irt > ict:
                    nco -= 1
                    _f11act(IC, iccol(istk), iccol(istk + 1), j, nco)
                    _f8act(IR, ircol(istk), ircol(istk + 1), irt - ict, i, nro)
                else:
                    nro -= 1
                    _f11act(IR, ircol(istk), ircol(istk + 1), i, nro)
                    _f8act(IC, iccol(istk), iccol(istk + 1), ict - irt, j, nco)
                if nro == 1:
                    base = iccol(istk + 1)
                    for kk in range(1, nco + 1):
                        y += fact[IC[base + kk - 1]]
                    state = "L90"
                    continue
                if nco == 1:
                    base = ircol(istk + 1)
                    for kk in range(1, nro + 1):
                        y += fact[IR[base + kk - 1]]
                    state = "L90"
                    continue
                lstk[istk] = lvar
                mstk[istk] = m
                nstk[istk] = n
                istk += 1
                nrstk[istk] = nro
                ncstk[istk] = nco
                ystk[istk] = y
                lvar = 1
                state = "TOP"
                continue

            if state == "L90":
                if y > amx:
                    amx = y
                    if SP - amx <= tol:
                        return -dspt
                state = "L100"
                continue

            if state == "L100":
                istk -= 1
                if istk == 0:
                    SP -= amx
                    if SP - amx <= tol:
                        return -dspt
                    else:
                        return SP - dspt
                lvar = lstk[istk] + 1
                state = "L110"
                continue

            if state == "L110":
                go60 = False
                while True:
                    if lvar > mstk[istk]:
                        break
                    n = nstk[istk]
                    nro = nrstk[istk]
                    nco = ncstk[istk]
                    y = ystk[istk]
                    if n == 1:
                        if IR[ircol(istk) + lvar - 1] < IR[ircol(istk) + lvar - 2]:
                            go60 = True
                            break
                    elif (
                        n == 2
                        and IC[iccol(istk) + lvar - 1] < IC[iccol(istk) + lvar - 2]
                    ):
                        go60 = True
                        break
                    lvar += 1
                if go60:
                    state = "L60"
                else:
                    state = "L100"
                continue

    # ---------------------------------------------------------------- f5xact
    def _f5xact(self, pastp, kval, ifreq, itop, jkey, jstp, jstp2, jstp3, jstp4, psh):
        """Put a node on the stack ("PUT"): a per-key binary tree of past path
        lengths, merging entries within ``tol`` and accumulating frequencies."""
        ldkey, ldstp = self.ldkey, self.ldstp
        tol = _TOL
        key, ipoin, stp, ifrq = self.key, self.ipoin, self.stp, self.ifrq

        if psh:
            ird = kval % ldkey
            target = None
            itp = -1
            for itp in range(ird, ldkey):
                if key[jkey + itp] == kval:
                    target = "L40"
                    break
                if key[jkey + itp] < 0:
                    target = "L30"
                    break
            if target is None:
                for itp in range(ird):
                    if key[jkey + itp] == kval:
                        target = "L40"
                        break
                    if key[jkey + itp] < 0:
                        target = "L30"
                        break
            if target is None:
                raise FexactError(
                    f"FEXACT error 6 (f5xact): LDKEY={ldkey} too small (kval={kval}); "
                    "increase 'workspace'"
                )
            if target == "L30":
                key[jkey + itp] = kval
                itop += 1
                ipoin[jkey + itp] = itop
                if itop > ldstp:
                    raise FexactError(
                        f"FEXACT error 7 (f5xact): LDSTP={ldstp} too small; increase "
                        "'workspace' (or use simulate_p_value=True)"
                    )
                ifrq[jstp2 + itop - 1] = -1
                ifrq[jstp3 + itop - 1] = -1
                ifrq[jstp4 + itop - 1] = -1
                stp[jstp + itop - 1] = pastp
                ifrq[jstp + itop - 1] = ifreq
                self._f5_itp = itp
                return itop
            self._f5_itp = itp

        itp = self._f5_itp
        ipn = ipoin[jkey + itp]
        test1 = pastp - tol
        test2 = pastp + tol
        while True:
            s = stp[jstp + ipn - 1]
            if s < test1:
                ipn = ifrq[jstp4 + ipn - 1]
            elif s > test2:
                ipn = ifrq[jstp3 + ipn - 1]
            else:
                if _INT_MAX - ifrq[jstp + ipn - 1] < ifreq:
                    raise FexactError("integer overflow in exact computation")
                ifrq[jstp + ipn - 1] += ifreq
                return itop
            if not (ipn > 0):
                break

        itop += 1
        if itop > ldstp:
            raise FexactError(
                f"FEXACT error 7 (f5xact): LDSTP={ldstp} too small; increase "
                "'workspace' (or use simulate_p_value=True)"
            )
        ipn = ipoin[jkey + itp]
        itmp = ipn
        while True:
            s = stp[jstp + ipn - 1]
            if s < test1:
                itmp = ipn
                ipn = ifrq[jstp4 + ipn - 1]
                if ipn > 0:
                    continue
                ifrq[jstp4 + itmp - 1] = itop
                break
            elif s > test2:
                itmp = ipn
                ipn = ifrq[jstp3 + ipn - 1]
                if ipn > 0:
                    continue
                ifrq[jstp3 + itmp - 1] = itop
                break
            else:
                break
        ifrq[jstp2 + itop - 1] = ifrq[jstp2 + itmp - 1]
        ifrq[jstp2 + itmp - 1] = itop
        stp[jstp + itop - 1] = pastp
        ifrq[jstp + itop - 1] = ifreq
        ifrq[jstp4 + itop - 1] = -1
        ifrq[jstp3 + itop - 1] = -1
        return itop

    # ---------------------------------------------------------------- f6xact
    def _f6xact(self, nrow, ikkey, ldkey, last):
        """Pop a node off the stack ("GET"); decodes the row config into
        ``self.iro[1..nrow]``. Returns ``(no_more_nodes, last, ipn)``."""
        key, kyy, iro = self.key, self.kyy, self.iro
        while True:
            last += 1
            if last <= ldkey:
                if key[ikkey + last] < 0:
                    continue
                kval = key[ikkey + last]
                key[ikkey + last] = -9999
                for j in range(nrow - 1, 0, -1):
                    iro[1 + j] = kval // kyy[1 + j]
                    kval -= iro[1 + j] * kyy[1 + j]
                iro[1] = kval
                return False, last, last
            else:
                return True, 0, None

    # ---------------------------------------------------------------- f7xact
    def _f7xact(self, nrow, k, ks):
        """Generate the new nodes for given marginal totals. Mutates
        ``self.idif``; returns ``(generated, k, ks)``."""
        iro, idif = self.iro, self.idif
        if ks == 0:
            while True:
                ks += 1
                if idif[ks] != iro[ks]:
                    break
        if idif[k] > 0 and k > ks:
            idif[k] -= 1
            while True:
                k -= 1
                if iro[k] != 0:
                    break
            m = k
            while idif[m] >= iro[m]:
                m -= 1
            idif[m] += 1
            if m == ks and idif[m] == iro[m]:
                ks = k
            return True, k, ks

        while True:  # Loop
            kk = k + 1
            found_l70 = False
            while kk <= nrow:
                if idif[kk] > 0:
                    found_l70 = True
                    break
                kk += 1
            if not found_l70:
                return False, k, ks
            # L70
            mm = 1
            for i in range(1, k + 1):
                mm += idif[i]
                idif[i] = 0
            k = kk
            while True:
                k -= 1
                m = min(mm, iro[k])
                idif[k] = m
                mm -= m
                if not (mm > 0 and k != 1):
                    break
            if mm > 0:
                if kk != nrow:
                    k = kk
                    continue
                return False, k, ks
            idif[kk] -= 1
            ks = 0
            while True:
                ks += 1
                if ks > k:
                    return True, k, ks
                if not (idif[ks] >= iro[ks]):
                    break
            return True, k, ks


def fexact(
    nrow, ncol, table, expect=-1.0, percnt=100.0, emin=0.0, workspace=200000, mult=30
):
    """R's ``fexact()`` — Fisher's exact test p-value ("PRE") for the ``nrow``
    by ``ncol`` contingency ``table`` (a 0-based 2-D sequence indexed
    ``table[i][j]``). The defaults ``expect=-1, percnt=100, emin=0`` request
    the exact p-value (R's ``fisher.test`` non-hybrid path); ``expect>0``
    selects the hybrid asymptotic-χ² approximation."""
    if _rs_fexact is not None:
        flat = [int(v) for row in table for v in row]
        try:
            return _rs_fexact(
                int(nrow),
                int(ncol),
                flat,
                float(expect),
                float(percnt),
                float(emin),
                int(workspace),
                int(mult),
            )
        except RuntimeError as e:  # unify the error type
            raise FexactError(str(e)) from None
    inst = _Fexact(nrow, ncol, table, expect, percnt, emin, workspace, mult)
    return inst.run()
