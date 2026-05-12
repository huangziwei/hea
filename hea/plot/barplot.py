"""Bar plot renderer — port of R's ``graphics::barplot``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def barplot(
    heights,
    *,
    names=None,
    horiz: bool = False,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    col=None,
    border="black",
    xlim=None,
    ylim=None,
    space: float = 0.2,
    ax=None,
):
    """Bar plot of ``heights``. Mirrors R's ``graphics::barplot``.

    Parameters
    ----------
    heights
        Bar heights. Accepts:

        * a numeric vector (list / np.ndarray / polars Series)
        * a 2-column polars ``DataFrame`` — the layout returned by
          :func:`hea.R.table`; column 0 is used for labels, column 1
          for heights. Lets you write ``barplot(table(x))`` directly.
    names
        Labels under each bar. Defaults to the first column of a
        ``table()`` frame, the Series name's levels for a categorical
        input, or sequential integers otherwise.
    horiz
        ``True`` to draw horizontal bars (R's ``horiz=TRUE``).
    xlab, ylab, main
        Axis labels / title.
    col, border
        Fill and edge colors.
    space
        Gap between bars as a fraction of bar width (matches R's default).
    ax
        Existing matplotlib ``Axes`` to draw into. A new figure is
        created when ``None``.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Accept the ``hea.R.table(x)`` layout: 2-col DataFrame (label, count).
    if isinstance(heights, pl.DataFrame) and heights.width == 2:
        if names is None:
            names = heights.to_series(0).cast(pl.Utf8).to_list()
        heights = heights.to_series(1).cast(pl.Float64).to_numpy()
    elif isinstance(heights, pl.Series):
        heights = heights.cast(pl.Float64).to_numpy()
    else:
        heights = np.asarray(heights, dtype=float)

    if names is None:
        names = [str(i + 1) for i in range(len(heights))]
    else:
        names = [str(n) for n in names]
    if len(names) != len(heights):
        raise ValueError(
            f"barplot(): names has {len(names)} entries but heights has {len(heights)}."
        )

    positions = np.arange(len(heights), dtype=float)
    width = 1.0 / (1.0 + space)
    bar_kwargs = {
        "color": col if col is not None else "lightgray",
        "edgecolor": border,
    }

    if horiz:
        ax.barh(positions, heights, height=width, **bar_kwargs)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
    else:
        ax.bar(positions, heights, width=width, **bar_kwargs)
        ax.set_xticks(positions)
        ax.set_xticklabels(names)

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax
