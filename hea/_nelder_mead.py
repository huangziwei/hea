"""Bounded-simplex Nelder-Mead — port of lme4's ``optimizer.cpp``.

lme4 ships its own Nelder-Mead implementation (derived from NLopt 2.2.4's
``nldrmd``) for the GLMM outer optimizer. Porting it directly — rather than
wrapping ``scipy.optimize.minimize(method="Nelder-Mead")`` — lets
:mod:`hea.lme` match lme4's iteration trajectory byte-for-byte when both
are run with ``optimizer="Nelder_Mead"``. scipy's Nelder-Mead uses a
different bounds-handling scheme and different default tolerances; even at
matched ``xtol`` settings the trajectories diverge after a few iterations.

The port preserves lme4's state-machine layout (stages
``restart → postreflect → {postexpand | postcontract}``), reflection
heuristic (``alpha=1, beta=0.5, gamm=2, delta=0.5``), and convergence
defaults from the R wrapper (``optimizer.R:27-33``: ``maxfun=10000``,
``FtolAbs=1e-5``, ``XtolRel=1e-7``, etc.). ``ftol``-style convergence is
defined in C++ but never invoked by the loop; we keep the parameter for
parity but it's effectively unused — only ``xtol``, ``maxeval``, and
``minf_max`` trigger termination.

References
----------
- ``/tmp/lme4/src/optimizer.cpp`` — C++ implementation (Bates/Mächler/Bolker).
- ``/tmp/lme4/src/optimizer.h`` — header with ``nm_status``/``nm_stage``
  enums and the heuristic constants.
- ``/tmp/lme4/R/optimizer.R`` — R wrapper exposing ``Nelder_Mead()``.
- NLopt's ``nldrmd.c`` — original algorithm by S. G. Johnson.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Callable, Optional

import numpy as np


# Heuristic "strategy" constants — optimizer.h:95.
ALPHA = 1.0   # reflection
BETA = 0.5    # contraction
GAMM = 2.0    # expansion
DELTA = 0.5   # shrink


class NMStatus(IntEnum):
    """Return code from :meth:`NelderMead.newf`.

    Mirrors lme4's ``nm_status`` enum (optimizer.h:89). ``active`` means
    "continue iterating"; any other value means the optimizer has stopped.
    """
    active = 0
    x0_not_feasible = 1   # nm_x0notfeasible — raised by ctor, never returned.
    no_feasible = 2       # nm_nofeasible    — raised by ctor, never returned.
    forced = 3            # nm_forced (set_force_stop=True)
    minf_max = 4          # objective dipped below ``minf_max``
    evals = 5             # hit ``maxeval``
    fcvg = 6              # ftol convergence (unused; preserved for parity)
    xcvg = 7              # xtol convergence


class _Stage(IntEnum):
    """Internal stage of the state machine — optimizer.h:92."""
    restart = 0
    postreflect = 1
    postexpand = 2
    postcontract = 3


def _close(a: float, b: float) -> bool:
    """Two values are within floating-point tolerance — optimizer.cpp:30."""
    return abs(a - b) <= 1e-13 * (abs(a) + abs(b))


def _relstop(vold: float, vnew: float, reltol: float, abstol: float) -> bool:
    """nl_stop's relative-stop predicate — optimizer.h:64-87.

    Returns ``True`` if ``vnew`` is within absolute/relative tolerance of
    ``vold``, or both are zero with ``reltol > 0``. Used by xtol checks
    in the ``restart`` stage.
    """
    if np.isinf(abs(vold)):
        return False
    return (
        abs(vnew - vold) < abstol
        or abs(vnew - vold) < reltol * (abs(vnew) + abs(vold)) * 0.5
        or (reltol > 0 and vnew == vold)
    )


class NelderMead:
    """Bounded-simplex Nelder-Mead — port of ``Nelder_Mead`` in optimizer.cpp.

    The caller drives the iteration via :meth:`xeval` (where to evaluate the
    objective next) and :meth:`newf` (feed the function value back). After
    :meth:`newf` returns a status other than :attr:`NMStatus.active`, the
    best point is at :meth:`xpos` with value :meth:`value`.

    Equivalent to lme4's R wrapper at optimizer.R:39-44::

        nm = NelderMead(lb, ub, xstep, x0)
        while True:
            f = objective(nm.xeval())
            status = nm.newf(f)
            if status != NMStatus.active:
                break

    Or use the convenience :meth:`minimize` for the common pattern.

    Parameters
    ----------
    lb, ub
        Element-wise lower/upper bounds. Use ``-np.inf``/``np.inf`` for
        unbounded coordinates. ``x0`` must be feasible.
    xstep
        Initial step sizes along each coordinate. The R wrapper at
        optimizer.R:5 defaults to ``rep(0.02, n)`` if not set; lme4's
        Stage 1 setup at lmer.R:2534-2540 uses ``0.2 * [0.1; min(βSD, 10)]``.
    x0
        Initial point. Must lie in ``[lb, ub]``.
    xtol_abs
        Per-coordinate absolute xtol. Defaults to ``|xstep| * 5e-4``
        matching the R wrapper at optimizer.R:6.
    xtol_rel, ftol_abs, ftol_rel
        Relative/absolute tolerances. ``ftol_*`` are stored but never
        consulted by the C++ implementation — included for API parity.
    maxeval
        Maximum function evaluations. Default 10000 (R wrapper default).
    minf_max
        Optimizer terminates when the function dips below this. Default
        ``-DBL_MAX``.
    """

    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        xstep: np.ndarray,
        x0: np.ndarray,
        *,
        xtol_abs: Optional[np.ndarray] = None,
        ftol_abs: float = 1e-5,
        ftol_rel: float = 1e-15,
        xtol_rel: float = 1e-7,
        maxeval: int = 10000,
        minf_max: float = -np.finfo(float).max,
    ):
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        xstep = np.asarray(xstep, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        n = x0.size
        if lb.size != n or ub.size != n or xstep.size != n:
            raise ValueError(
                f"lb/ub/xstep/x0 size mismatch: {lb.size}/{ub.size}/"
                f"{xstep.size}/{n}"
            )
        if np.any(x0 - lb < 0) or np.any(ub - x0 < 0):
            raise ValueError("initial x0 is not a feasible point")
        if np.any(xstep == 0):
            raise ValueError("xstep must be nonzero for every coordinate")
        if xtol_abs is None:
            xtol_abs = np.abs(xstep) * 5e-4
        xtol_abs = np.asarray(xtol_abs, dtype=float)
        if xtol_abs.size != n:
            raise ValueError(f"xtol_abs size {xtol_abs.size} != n={n}")

        # Build the initial simplex. Vertex 0 = x0; vertex j+1 = x0 + xstep[j]·e_j,
        # pinned into [lb, ub] via the constructor heuristics
        # (optimizer.cpp:71-91). For each coord:
        # 1. Add xstep.
        # 2. If outside ub: if there's "enough room" toward ub, clip to ub;
        #    else flip direction and use ``x0 - |xstep|``.
        # 3. Symmetric handling for lb violations.
        # 4. If after all that the vertex still coincides with x0, the
        #    simplex is degenerate — raise.
        pts = np.tile(x0[:, None], (1, n + 1))
        for i in range(n):
            j = i + 1
            pts[i, j] += xstep[i]
            if pts[i, j] > ub[i]:
                if ub[i] - x0[i] > abs(xstep[i]) * 0.1:
                    pts[i, j] = ub[i]
                else:
                    pts[i, j] = x0[i] - abs(xstep[i])
            if pts[i, j] < lb[i]:
                if x0[i] - lb[i] > abs(xstep[i]) * 0.1:
                    pts[i, j] = lb[i]
                else:
                    pts[i, j] = x0[i] + abs(xstep[i])
                    if pts[i, j] > ub[i]:
                        # go toward the farther of lb, ub
                        target = ub[i] if (ub[i] - x0[i] > x0[i] - lb[i]) else lb[i]
                        pts[i, j] = 0.5 * (target + x0[i])
            if _close(pts[i, j], x0[i]):
                raise ValueError("cannot generate feasible simplex")

        # Stored constraint bounds and step sizes.
        self.lb = lb
        self.ub = ub
        self.xstep = xstep
        self.n = n
        # Initial simplex.
        self.pts = pts
        # vals[i] = f(pts[:, i]). Will be filled in by the init phase.
        # Initialize to a sentinel so that "vals[i] uninitialised" is
        # distinguishable from a legitimate value.
        self.vals = np.full(n + 1, np.finfo(float).min, dtype=float)
        # Working buffers.
        self.c = np.zeros(n)             # centroid of n-1 simplex opposite high
        self.xcur = np.zeros(n)          # last candidate generated by reflectpt
        self.xeval_ = x0.copy()          # where to evaluate next
        # Best-seen state.
        self.x = x0.copy()
        self.minf = np.inf
        # State machine.
        self.stage = _Stage.restart
        self.init_pos = 0
        # Convergence params.
        self.xtol_abs = xtol_abs
        self.ftol_abs = ftol_abs
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.maxeval = maxeval
        self.minf_max = minf_max
        self.nevals = 0
        self.force_stop = False
        # Stage-local scratch (high/low values, indices, previous-iter f).
        self._f_old = 0.0
        self._fh = 0.0
        self._fl = 0.0
        self._ih = 0
        self._il = 0

    # ---- public interface -----------------------------------------------

    def xeval(self) -> np.ndarray:
        """Where to evaluate the objective next."""
        return self.xeval_

    def xpos(self) -> np.ndarray:
        """Best parameter vector found so far."""
        return self.x

    def value(self) -> float:
        """Best function value found so far."""
        return self.minf

    def set_force_stop(self, stop: bool) -> None:
        """Request early termination on next :meth:`newf`."""
        self.force_stop = stop

    def newf(self, f: float) -> NMStatus:
        """Install ``f = objective(xeval())`` and step the state machine.

        Port of ``Nelder_Mead::newf`` (optimizer.cpp:101-141). Returns
        :attr:`NMStatus.active` to keep iterating, or a terminal code.
        """
        self.nevals += 1
        if self.force_stop:
            return NMStatus.forced
        if f < self.minf:
            self.minf = f
            self.x = self.xeval_.copy()
            if self.minf < self.minf_max:
                return NMStatus.minf_max
        if self.maxeval > 0 and self.nevals > self.maxeval:
            return NMStatus.evals
        if self.init_pos <= self.n:
            return self._init(f)
        if self.stage == _Stage.restart:
            return self._restart(f)
        elif self.stage == _Stage.postreflect:
            return self._postreflect(f)
        elif self.stage == _Stage.postexpand:
            return self._postexpand(f)
        elif self.stage == _Stage.postcontract:
            return self._postcontract(f)
        return NMStatus.active

    def minimize(self, fn: Callable[[np.ndarray], float]) -> NMStatus:
        """Run the optimizer to a stopping condition, calling ``fn`` each step.

        Mirrors the R wrapper's loop at optimizer.R:39-44. Returns the
        terminal :class:`NMStatus`. After return, :meth:`xpos` and
        :meth:`value` hold the optimum.
        """
        while True:
            f = fn(self.xeval_)
            status = self.newf(f)
            if status != NMStatus.active:
                return status

    # ---- state-machine stages -------------------------------------------

    def _init(self, f: float) -> NMStatus:
        """Fill ``vals[init_pos]`` and queue the next simplex vertex.

        Port of ``Nelder_Mead::init`` (optimizer.cpp:150-156). Called for
        the first ``n+1`` evaluations to populate the initial simplex.
        """
        if self.init_pos > self.n:
            raise RuntimeError("init called after n+1 evaluations")
        self.vals[self.init_pos] = f
        self.init_pos += 1
        if self.init_pos > self.n:
            return self._restart(f)
        self.xeval_ = self.pts[:, self.init_pos].copy()
        return NMStatus.active

    def _restart(self, f: float) -> NMStatus:
        """Recompute high/low/centroid, check x-convergence, reflect.

        Port of ``Nelder_Mead::restart`` (optimizer.cpp:167-192).
        """
        # min/max function values across the simplex
        self._il = int(np.argmin(self.vals))
        self._fl = float(self.vals[self._il])
        self._ih = int(np.argmax(self.vals))
        self._fh = float(self.vals[self._ih])
        # centroid of n-1 simplex opposite the high vertex
        self.c = (self.pts.sum(axis=1) - self.pts[:, self._ih]) / self.n
        # x-convergence: max deviation from centroid in each coord small enough
        deviations = np.abs(self.pts - self.c[:, None]).max(axis=1)
        if self._x_conv(np.zeros(self.n), deviations):
            return NMStatus.xcvg
        # reflect the high vertex through the centroid (scale = alpha)
        if not self._reflectpt(self.xcur, self.c, ALPHA, self.pts[:, self._ih]):
            return NMStatus.xcvg
        self.xeval_ = self.xcur.copy()
        self.stage = _Stage.postreflect
        return NMStatus.active

    def _postreflect(self, f: float) -> NMStatus:
        """Decide what to do with the reflected point — port of
        ``Nelder_Mead::postreflect`` (optimizer.cpp:194-219).

        Three branches:

        * ``f < f_low`` → new best; try to expand further along the same
          direction (scale = gamm).
        * ``f < f_high`` → accept the reflected point; restart with the
          updated simplex.
        * else → contract. ``-beta`` (outside contraction) when the
          reflected point is at least as bad as the current high (the
          inside-contraction branch ``beta`` would only fire on a pathology
          where ``f < d_fh`` after we just established ``f >= d_fh`` —
          preserved for parity with lme4's expression).
        """
        if f < self._fl:
            # new best — set up expansion
            if not self._reflectpt(self.xeval_, self.c, GAMM, self.pts[:, self._ih]):
                return NMStatus.xcvg
            self.stage = _Stage.postexpand
            self._f_old = f
            return NMStatus.active
        if f < self._fh:
            # acceptable — replace high vertex and restart
            self.vals[self._ih] = f
            self.pts[:, self._ih] = self.xeval_
            return self._restart(f)
        # new worst point — contract
        scale = -BETA if self._fh <= f else BETA
        if not self._reflectpt(self.xcur, self.c, scale, self.pts[:, self._ih]):
            return NMStatus.xcvg
        self._f_old = f
        self.xeval_ = self.xcur.copy()
        self.stage = _Stage.postcontract
        return NMStatus.active

    def _postexpand(self, f: float) -> NMStatus:
        """Did expansion improve? Port of ``postexpand`` (optimizer.cpp:221-235)."""
        if f < self.vals[self._ih]:
            # expansion improved on the (already-replaced) high
            self.pts[:, self._ih] = self.xeval_
            self.vals[self._ih] = f
        else:
            # revert to the reflected point (xcur) with its function value (f_old)
            self.pts[:, self._ih] = self.xcur
            self.vals[self._ih] = self._f_old
        return self._restart(f)

    def _postcontract(self, f: float) -> NMStatus:
        """Did contraction improve? Port of ``postcontract`` (optimizer.cpp:237-256).

        If yes, accept and restart. Otherwise SHRINK the entire simplex
        toward the best vertex (``il``) — and re-initialize by evaluating
        every shrunk vertex (back to ``init`` phase).
        """
        if f < self._f_old and f < self._fh:
            self.pts[:, self._ih] = self.xeval_
            self.vals[self._ih] = f
            return self._restart(f)
        # shrink: each non-best vertex moves halfway to the best
        best = self.pts[:, self._il].copy()
        for i in range(self.n + 1):
            if i != self._il:
                target = np.empty(self.n)
                if not self._reflectpt(target, best, -DELTA, self.pts[:, i]):
                    return NMStatus.xcvg
                self.pts[:, i] = target
        # re-evaluate the shrunken simplex from scratch
        self.init_pos = 0
        self.xeval_ = self.pts[:, 0].copy()
        return NMStatus.active

    # ---- helpers --------------------------------------------------------

    def _reflectpt(self, xnew: np.ndarray, c: np.ndarray, scale: float,
                   xold: np.ndarray) -> bool:
        """``xnew = clip(c + scale·(c − xold), lb, ub)``.

        Port of ``Nelder_Mead::reflectpt`` (optimizer.cpp:269-289). Returns
        ``False`` if ``xnew`` numerically coincides with ``c`` *or* ``xold``
        in every coordinate — signal of a collapsed simplex, caller should
        terminate with ``nm_xcvg``.

        Mutates ``xnew`` in place to match the C++ pass-by-reference idiom.
        """
        np.copyto(xnew, c + scale * (c - xold))
        equalc = True
        equalold = True
        for i in range(self.n):
            newx = min(max(xnew[i], self.lb[i]), self.ub[i])
            equalc = equalc and _close(newx, c[i])
            equalold = equalold and _close(newx, xold[i])
            xnew[i] = newx
        return not (equalc or equalold)

    def _x_conv(self, x: np.ndarray, oldx: np.ndarray) -> bool:
        """All coordinates pass relstop — port of ``nl_stop::x`` (optimizer.cpp:299)."""
        for i in range(x.size):
            if not _relstop(oldx[i], x[i], self.xtol_rel, self.xtol_abs[i]):
                return False
        return True


__all__ = ["NelderMead", "NMStatus"]
