"""Shared low-level helpers for the hea.plot package."""

from __future__ import annotations

import numpy as np
import polars as pl

_R_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]


# The most recently used ``Axes`` for any hea primary plotter (``hist``,
# ``plot``, ``boxplot``, …). Used by overlay calls (``abline``, ``points``,
# ``lines``, ``segments``, ``qqline``, ``rug``, ``legend``) when their
# ``ax=`` is not given — R's ``abline(lmod)`` idiom without surfacing
# matplotlib's ``plt.gca()``. ``None`` until the first hea primary draws.
_LAST_AX = None


def _remember_ax(ax):
    """Record ``ax`` as the most recently used target. No-op on ``None``."""
    global _LAST_AX
    if ax is not None:
        _LAST_AX = ax


def resolve_ax(ax, *, figsize=None):
    """Return the target ``Axes`` for a single-panel base-graphics call.

    Cases, in order:

    1. ``ax`` was passed explicitly — record + return.
    2. A :func:`hea.plot.par` context is active and ``figsize`` is not
       set — pull the next cell from its grid. (``figsize`` only makes
       sense for a freshly-created figure, so it forces case 3 even
       inside ``par``.)
    3. Neither — create a fresh figure with ``plt.subplots(figsize=)``
       (R's "open a new device" default).

    In every case the returned ``Axes`` is stored as :data:`_LAST_AX` so
    later overlay calls (``abline``, ``points``, …) without an explicit
    ``ax=`` know where to draw.
    """
    if ax is not None:
        _remember_ax(ax)
        return ax
    # Local import keeps _util.py free of plt at module load (R's plotters
    # are sometimes used headless / with backend swaps in tests).
    from .par import _current_par

    p = _current_par()
    if p is not None and figsize is None:
        ax = p.next_cell()
    else:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=figsize)
    _remember_ax(ax)
    return ax


def resolve_overlay_ax(ax, fname: str):
    """Target axes for an overlay (``abline``, ``points``, …).

    Returns ``ax`` if given, otherwise the most recently drawn hea axes
    (set by :func:`resolve_ax`). Raises a clear error if neither — that
    case is "annotate without a plot" which has no sensible default.
    """
    if ax is not None:
        return ax
    if _LAST_AX is not None:
        return _LAST_AX
    raise ValueError(
        f"{fname}(): no ax= and no previous hea plot to attach to. "
        f"Call a primary plotter (plot, hist, …) first or pass ax=."
    )


def to_codes(x):
    """Polars Enum/Categorical → integer codes; numpy/list → unchanged."""
    if isinstance(x, pl.Series) and x.dtype in (pl.Enum, pl.Categorical):
        return x.to_physical().to_numpy()
    if isinstance(x, pl.Series):
        return x.to_numpy()
    return np.asarray(x)


def to_float(x):
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64).to_numpy()
    return np.asarray(x, dtype=float)


def r_lty(lty):
    """R lty (1 solid, 2 dashed, 3 dotted, 4 dotdash, 5 longdash, 6 twodash)
    or matplotlib string → matplotlib linestyle."""
    if lty is None:
        return "-"
    if isinstance(lty, str):
        return lty
    return {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (10, 3)),
            6: (0, (5, 1, 1, 1))}.get(int(lty), "-")


def draw_points(ax, x, y, *, pch=None, cex=None, col=None):
    """Per-marker scatter draw — no axis labels touched.

    Used by both ``scatter()`` (the primary plotter) and ``points()`` (the
    overlay). When ``pch`` is a vector of integer codes, splits into one
    ``scatter`` call per unique code so matplotlib gets a scalar marker."""
    xa = to_float(x)
    ya = to_float(y)
    pch_codes = to_codes(pch) if pch is not None else None
    col_codes = to_codes(col) if col is not None else None

    base = {"facecolor": "none", "edgecolor": "black"}
    if cex is not None:
        base["s"] = (cex * 6) ** 2  # rough R cex≈1 default

    if pch_codes is not None and getattr(pch_codes, "ndim", 0) > 0:
        for code in np.unique(pch_codes):
            mask = pch_codes == code
            marker = _R_MARKERS[int(code) % len(_R_MARKERS)]
            kw = dict(base)
            if col_codes is not None and getattr(col_codes, "ndim", 0) > 0:
                kw["c"] = col_codes[mask]
                kw.pop("edgecolor", None)
            ax.scatter(xa[mask], ya[mask], marker=marker, **kw)
        return

    kw = dict(base)
    if col_codes is not None:
        if getattr(col_codes, "ndim", 0) > 0:
            kw["c"] = col_codes
            kw.pop("edgecolor", None)
        else:
            kw["edgecolor"] = col_codes
    if pch_codes is not None:
        kw["marker"] = _R_MARKERS[int(pch_codes) % len(_R_MARKERS)] \
            if isinstance(pch_codes, (int, np.integer)) else "o"
    ax.scatter(xa, ya, **kw)
