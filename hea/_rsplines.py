"""R's ``splines::splineDesign`` and the C ``spline_basis`` beneath it.

Every B-spline design matrix mgcv builds (``ps``, ``bs``, ``cp`` via
``cSplineDes``, ``ad``) and R's ``bs()``/``ns()`` come from these, so the
evaluation follows ``splines.c`` statement by statement, vectorized over
``x``: ``set_cursor``, ``diff_table``, the de Boor recurrence of
``basis_funcs`` for values and ``evaluate`` for derivatives. On arm64 R's
``splines.so`` fuses ``saved + rdel[r]*term`` in ``basis_funcs`` and the
left product of ``a[k+1]*ldel + a[k]*rdel`` in ``evaluate``; ``_rfma_vec``
reproduces that per architecture. Divisions by a zero knot span stay
silent IEEE arithmetic, as in the C.

A leaf module, so :mod:`hea.formula` can use it without importing
:mod:`hea.R`.
"""

from __future__ import annotations

import numpy as np

from ._rfma import _rfma_vec


def _set_cursor(knots: np.ndarray, x: np.ndarray, order: int):
    """``set_cursor``: the index of the first knot greater than each x
    (the last knot equal to x when none is greater, -1 when every knot is
    below x), moved back to ``nknots - order`` with ``boundary`` set when x
    is that knot."""
    nk = knots.size
    above = np.searchsorted(knots, x, side="right")
    curs = np.where(above < nk, above, -1)
    at_end = (above == nk) & (nk > 0)
    if nk:
        curs = np.where(at_end & (knots[nk - 1] >= x), nk - 1, curs)
    last_legit = nk - order
    boundary = np.zeros(x.shape, dtype=bool)
    if 0 <= last_legit < nk:
        move = (curs > last_legit) & (x == knots[last_legit])
        boundary = move
        curs = np.where(move, last_legit, curs)
    return curs, boundary


def _diff_table(knots: np.ndarray, curs: np.ndarray, x: np.ndarray, ndiff: int):
    """``diff_table``: ``rdel[i] = knots[curs+i] - x`` and
    ``ldel[i] = x - knots[curs-(i+1)]`` for ``i < ndiff``."""
    hi = knots.size - 1
    rdel = [knots[np.clip(curs + i, 0, hi)] - x for i in range(ndiff)]
    ldel = [x - knots[np.clip(curs - (i + 1), 0, hi)] for i in range(ndiff)]
    return rdel, ldel


def _basis_funcs(knots, curs, x, ordm1: int) -> list[np.ndarray]:
    """``basis_funcs``: the ``ordm1 + 1`` non-zero B-splines at each x."""
    rdel, ldel = _diff_table(knots, curs, x, ordm1)
    b = [np.ones(x.shape)] + [np.zeros(x.shape) for _ in range(ordm1)]
    for j in range(1, ordm1 + 1):
        saved = np.zeros(x.shape)
        for r in range(j):
            den = rdel[r] + ldel[j - 1 - r]
            nz = den != 0
            term = b[r] / np.where(nz, den, 1.0)
            b_nz = _rfma_vec(rdel[r], term, saved)
            keep = np.zeros(x.shape, dtype=bool) if r != 0 else (rdel[r] == 0.0)
            b_z = np.where(keep, b[r], saved)
            b[r] = np.where(nz, b_nz, b_z)
            saved = np.where(nz, ldel[j - 1 - r] * term, 0.0)
        b[j] = saved
    return b


def _evaluate(knots, curs, boundary, x, a: list[np.ndarray], ordm1: int, nder: int):
    """``evaluate``: derivative ``nder`` of the spline with local
    coefficients ``a`` (overwritten), at each x."""
    outer = ordm1
    hi = knots.size - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(nder):
            for k in range(outer):
                span = (
                    knots[np.clip(curs + k, 0, hi)]
                    - knots[np.clip(curs - outer + k, 0, hi)]
                )
                a[k] = outer * (a[k + 1] - a[k]) / span
            outer -= 1
        rdel, ldel = _diff_table(knots, curs, x, outer)
        while outer:
            outer -= 1
            for k in range(outer + 1):
                lft = ldel[outer - k]
                rgt = rdel[k]
                a[k] = _rfma_vec(a[k + 1], lft, a[k] * rgt) / (rgt + lft)
    if nder == ordm1:
        return np.where(boundary, 0.0, a[0])
    return a[0]


def spline_basis(knots, order: int, x, derivs) -> tuple[np.ndarray, np.ndarray]:
    """C ``spline_basis(knots, order, xvals, derivs)``: the ``order``
    non-zero B-splines (or their derivatives ``derivs``, recycled along x)
    at each x, as an ``(len(x), order)`` array, and each row's offset
    ``curs - order`` (rows whose offset is outside ``0..len(knots)`` are
    NaN)."""
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    ders = np.asarray(derivs, dtype=int).ravel()
    nx, nd, nk = x.size, ders.size, knots.size
    ord_ = int(order)
    ordm1 = ord_ - 1
    out = np.full((nx, ord_), np.nan)
    if nx == 0:
        return out, np.zeros(0, dtype=int)
    curs, boundary = _set_cursor(knots, x, ord_)
    io = curs - ord_
    der = ders[np.arange(nx) % nd]
    valid = (io >= 0) & (io <= nk)
    bad = valid & (der >= ord_)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        if nd == 1:
            raise ValueError(
                f"derivs = {der[i]} >= ord = {ord_}, but should be in {{0,..,ord-1}}"
            )
        raise ValueError(
            f"derivs[{i + 1}] = {der[i]} >= ord = {ord_}, but should be in {{0,..,ord-1}}"
        )
    for d in np.unique(der[valid]):
        rows = valid & (der == d)
        xr, cr, br = x[rows], curs[rows], boundary[rows]
        if d == 0:
            cols = _basis_funcs(knots, cr, xr, ordm1)
        else:
            cols = []
            for ii in range(ord_):
                a = [np.full(xr.shape, 1.0 if j == ii else 0.0) for j in range(ord_)]
                cols.append(_evaluate(knots, cr, br, xr, a, ordm1, int(d)))
        out[rows] = np.column_stack(cols)
    return out, io


def spline_design(
    knots, x, ord: int = 4, derivs=0, outer_ok: bool = False
) -> np.ndarray:
    """R ``splines::splineDesign(knots, x, ord, derivs, outer.ok)`` (dense):
    the ``len(x) x (len(knots) - ord)`` B-spline design matrix. With
    ``outer_ok``, x outside the boundary knots gives zero rows and the knot
    vector is extended by repeating its end knots for the evaluation."""
    knots = np.asarray(knots, dtype=float).ravel()
    nk = knots.size
    if nk <= 0:
        raise ValueError("must have at least 'ord' knots")
    if np.any(np.diff(knots) < 0):
        knots = np.sort(knots, kind="stable")
    x = np.asarray(x, dtype=float).ravel()
    nx = x.size
    derivs = np.atleast_1d(np.asarray(derivs, dtype=int)).ravel()
    if derivs.size > nx:
        raise ValueError("length of 'derivs' is larger than length of 'x'")
    if derivs.size < 1:
        raise ValueError("empty 'derivs'")
    ord_ = int(ord)
    if ord_ > nk or ord_ < 1:
        raise ValueError("'ord' must be positive integer, at most the number of knots")
    if not outer_ok and nk < 2 * ord_ - 1:
        raise ValueError(f"need at least 2*ord -1 (={2 * ord_ - 1}) knots")
    degree = ord_ - 1
    need_outer = bool(np.any((x < knots[ord_ - 1]) | (knots[nk - degree - 1] < x)))
    x_out = False
    in_x = None
    if need_outer:
        if not outer_ok:
            raise ValueError(
                f"the 'x' data must be in the range {knots[ord_ - 1]:g} to "
                f"{knots[nk - degree - 1]:g} unless you set 'outer.ok = TRUE'"
            )
        in_x = (knots[0] <= x) & (x <= knots[nk - 1])
        x_out = not bool(np.all(in_x))
        if x_out:
            x = x[in_x]
        dkn = np.diff(knots)[::-1]
        pos = np.flatnonzero(dkn > 0)
        first = int(pos[0]) + 1 if pos.size else None
        n_right = max(0, ord_ - first) if first is not None else 0
        idx = np.concatenate(
            [np.zeros(degree, dtype=int), np.arange(nk), np.full(n_right, nk - 1)]
        )
        knots = knots[idx]
    temp, offsets = spline_basis(knots, ord_, x, derivs)
    ncoef = nk - ord_
    rows = np.flatnonzero(in_x) if (need_outer and x_out) else np.arange(nx)
    ii = np.repeat(rows, ord_)
    jj = (np.arange(ord_)[None, :] + offsets[:, None]).ravel()
    vals = temp.ravel()
    design = np.zeros((nx, ncoef))
    if need_outer:
        jj = jj - degree
        ok = (jj >= 0) & (jj < ncoef)
        design[ii[ok], jj[ok]] = vals[ok]
    else:
        design[ii, jj] = vals
    return design
