"""Histogram renderer — port of R's ``stats::hist``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _resolve_breaks(vals: np.ndarray, breaks) -> np.ndarray:
    """Return bin edges from R-style ``breaks`` argument.

    Accepts an integer (#bins), an explicit sequence of edges, or a
    method name: ``"sturges"`` (R default), ``"scott"``, or ``"fd"``
    (Freedman-Diaconis). Unknown strings fall back to numpy's named
    methods so ``"auto"``, ``"sqrt"``, ``"rice"`` etc. also work.
    """
    if isinstance(breaks, str):
        method = breaks.lower()
        if method == "sturges":
            # R's classical default: ceiling(log2(n) + 1)
            n = max(len(vals), 1)
            nb = int(np.ceil(np.log2(n) + 1))
            return np.histogram_bin_edges(vals, bins=nb)
        return np.histogram_bin_edges(vals, bins=method)
    if isinstance(breaks, (int, np.integer)):
        return np.histogram_bin_edges(vals, bins=int(breaks))
    return np.asarray(breaks, dtype=float)


def hist(
    x,
    breaks="sturges",
    *,
    freq: bool | None = None,
    probability: bool | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    col=None,
    border="black",
    xlim=None,
    ylim=None,
    ax=None,
):
    """Histogram of ``x``. Mirrors R's ``stats::hist``.

    Parameters
    ----------
    x : polars Series, numpy array, or list of numbers
        Data to bin. Nulls / NaNs are dropped (R's default).
    breaks
        Number of bins (int), explicit edges (sequence), or method name
        (``"sturges"`` (R default), ``"scott"``, ``"fd"``, or any name
        numpy's ``histogram_bin_edges`` accepts).
    freq, probability
        ``freq=True`` plots counts; ``freq=False`` plots density.
        ``probability`` is the negation. If neither is set, R's rule is
        applied: counts iff the bin widths are equal.
    xlab, ylab, main
        Axis labels and title. ``xlab`` defaults to a polars Series'
        ``.name``; ``main`` defaults to ``"Histogram of <name>"``.
        Pass ``main=""`` to suppress the default title (the call form
        used throughout Faraway's book).
    col, border
        Bar fill and edge colors.
    xlim, ylim
        Axis limits (any matplotlib-compatible tuple).
    ax
        Existing matplotlib ``Axes`` to draw into. A new figure is
        created when ``None``.
    """
    if ax is None:
        _, ax = plt.subplots()

    name = ""
    if isinstance(x, pl.Series):
        name = x.name or ""
        vals = x.cast(pl.Float64).drop_nulls().to_numpy()
    else:
        vals = np.asarray(x, dtype=float)
        vals = vals[~np.isnan(vals)]

    if freq is not None and probability is not None:
        raise ValueError("hist(): pass freq= or probability=, not both.")
    if probability is not None:
        freq = not probability

    edges = _resolve_breaks(vals, breaks)
    counts, _ = np.histogram(vals, bins=edges)
    widths = np.diff(edges)

    if freq is None:
        # R rule: counts when bin widths are equal, density otherwise.
        freq = bool(widths.size and np.allclose(widths, widths[0]))

    if freq:
        heights = counts.astype(float)
        ylab_default = "Frequency"
    else:
        total = float(counts.sum())
        heights = (counts / (total * widths)) if total > 0 else np.zeros_like(counts, dtype=float)
        ylab_default = "Density"

    bar_kwargs = {
        "edgecolor": border,
        "align": "edge",
        "color": col if col is not None else "lightgray",
    }
    ax.bar(edges[:-1], heights, width=widths, **bar_kwargs)

    if xlab is None:
        xlab = name
    if main is None:
        main = f"Histogram of {name}" if name else ""
    if ylab is None:
        ylab = ylab_default

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    ax.set_title(main)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax
