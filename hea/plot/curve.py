"""Function plotter — port of R's ``graphics::curve``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._util import r_lty


def curve(
    f,
    from_: float | None = None,
    to: float | None = None,
    *,
    n: int = 101,
    add: bool = False,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    col: str = "black",
    lty=None,
    lwd: float | None = None,
    ax=None,
):
    """Evaluate and draw a function on ``[from_, to]``. Mirrors R's ``curve``.

    Parameters
    ----------
    f : callable
        Numeric function evaluated on ``n`` points. R captures a math
        expression via NSE (``curve(dnorm(x), -3, 3)``); in Python pass
        a callable: ``curve(lambda x: norm.pdf(x), -3, 3)``.
    from_, to
        Evaluation range. Required when ``add=False`` and ``ax`` is
        ``None``; when ``add=True`` (or ``ax`` has data already), the
        existing x-limits are used as defaults.
    n
        Number of evaluation points (R default is 101).
    add
        ``True`` overlays the curve on ``ax`` without redrawing labels —
        the typical R idiom for adding a theoretical density on top of
        a histogram.
    col, lty, lwd
        Line color, R-style line type, line width.
    ax
        Existing matplotlib ``Axes``. Created if ``None``.

    Notes
    -----
    ``from_`` (with a trailing underscore) is used because ``from`` is a
    Python keyword.
    """
    if ax is None:
        _, ax = plt.subplots()

    if from_ is None or to is None:
        x_lo, x_hi = ax.get_xlim()
        if from_ is None:
            from_ = float(x_lo)
        if to is None:
            to = float(x_hi)
        if from_ == to:
            raise ValueError(
                "curve(): from_ / to could not be inferred from ax — "
                "pass them explicitly."
            )

    xs = np.linspace(from_, to, n)
    ys = np.asarray(f(xs), dtype=float)

    line_kwargs = {"color": col, "linestyle": r_lty(lty)}
    if lwd is not None:
        line_kwargs["linewidth"] = float(lwd)
    ax.plot(xs, ys, **line_kwargs)

    if not add:
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)
        if main is not None:
            ax.set_title(main)
    return ax
