"""Kernel density estimation — port of R's ``stats::density``.

``density(x)`` returns a small object holding the evaluation grid (``x``)
and density values (``y``) — same shape as R, where ``density()`` returns
a ``list`` with ``$x`` / ``$y`` / ``$bw`` / ``$n``. Call ``.plot(ax=)`` or
pass the object to :func:`hea.plot.lines` to draw it onto a histogram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from ._util import to_value_series


@dataclass
class _Density:
    """KDE result. Field names mirror R's ``density`` list elements."""

    x: np.ndarray
    y: np.ndarray
    bw: float
    n: int
    data_name: str = ""

    def __repr__(self) -> str:
        return (
            f"density({self.data_name or '<unnamed>'}): "
            f"n={self.n}, bw={self.bw:.4g}, "
            f"x in [{self.x[0]:.4g}, {self.x[-1]:.4g}]"
        )

    def plot(
        self,
        *,
        ax=None,
        xlab: str | None = None,
        ylab: str | None = None,
        main: str | None = None,
        col: str = "black",
        lty=None,
    ):
        """Draw this density curve. Mirrors R's ``plot.density``."""
        from ._util import r_lty, resolve_ax

        ax = resolve_ax(ax)
        ax.plot(self.x, self.y, color=col, linestyle=r_lty(lty))
        if xlab is None:
            xlab = f"N = {self.n}   Bandwidth = {self.bw:.4g}"
        if main is None:
            main = f"density.default(x = {self.data_name})" if self.data_name else ""
        if xlab:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)
        else:
            ax.set_ylabel("Density")
        if main is not None:
            ax.set_title(main)
        return ax


def density(
    x,
    *,
    bw="scott",
    n: int = 512,
    from_: float | None = None,
    to: float | None = None,
    cut: float = 3.0,
) -> _Density:
    """Gaussian kernel density estimate. Mirrors R's ``stats::density``.

    Parameters
    ----------
    x
        Numeric vector. Nulls / NaNs are dropped.
    bw
        Bandwidth. A positive float, or one of ``"scott"`` (default;
        same as R's ``"nrd0"`` for typical samples) or ``"silverman"``.
    n
        Number of grid points (R's default is 512).
    from_, to
        Evaluation range. Defaults to a span of ``cut`` bandwidths
        beyond the data range (matches R's ``cut=3`` default).
    cut
        Multiplier on the bandwidth used to extend the default range.

    Returns
    -------
    _Density
        With fields ``.x`` (grid), ``.y`` (density), ``.bw``, ``.n``,
        and ``.data_name``. Call ``.plot(ax=)`` to draw.
    """
    s = to_value_series(x, "density")
    name = s.name or ""
    vals = s.cast(pl.Float64).drop_nulls().to_numpy()
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        raise ValueError("density(): need at least 2 finite values.")

    kde = gaussian_kde(vals, bw_method=bw)
    bw_val = float(kde.factor * vals.std(ddof=1))
    if from_ is None:
        from_ = float(vals.min() - cut * bw_val)
    if to is None:
        to = float(vals.max() + cut * bw_val)
    grid = np.linspace(from_, to, n)
    return _Density(
        x=grid,
        y=kde(grid),
        bw=bw_val,
        n=int(vals.size),
        data_name=name,
    )
