"""Binning helpers from ggplot2: ``cut_width``, ``cut_interval``, ``cut_number``.

Bin a continuous variable into a factor for discrete grouping/colouring::

    geom_boxplot(aes(group=cut_width("carat", 0.1)))            # inside aes()
    geom_boxplot(group=cut_width("carat", 0.1))                  # bare kwarg

All three accept either eager input (``Series`` / ``ndarray``) or a
column reference (bare column name string, like ``fct_*``, or a
``pl.col()`` expression for derived inputs like ``col("carat") * 1000``).
The column-reference form returns a callable that the build pipeline
resolves against the layer's data — same convention as
``aes(x=lambda d: d["wt"] * 2)``.

Algorithms mirror ``ggplot2::cut_width`` / ``cut_interval`` /
``cut_number`` (file ``R/bin.R`` in ggplot2). Each produces breaks and
delegates the actual binning to :func:`hea.R.cut`, so the resulting
labels follow R's ``"(a,b]"``-style formatting.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..R import cut


def _resolve(x, data: pl.DataFrame | None = None) -> np.ndarray:
    """Materialize ``x`` as a 1-d float numpy array.

    String / ``pl.Expr`` need a DataFrame context (the lazy form used
    inside ``aes()``); eager forms (Series / ndarray / list) resolve
    immediately.
    """
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, str):
        if data is None:
            raise ValueError(
                f"cut_*: column reference {x!r} needs a DataFrame "
                f"context — use this form inside aes(), or pass a "
                f"Series for eager use."
            )
        return data[x].to_numpy().astype(float)
    if isinstance(x, pl.Expr):
        if data is None:
            raise ValueError(
                "cut_*: polars expression needs a DataFrame context "
                "— use this form inside aes(), or pass a Series for "
                "eager use."
            )
        return data.select(x).to_series().to_numpy().astype(float)
    return np.asarray(x, dtype=float)


def _maybe_lazy(x, eager_fn):
    """Run ``eager_fn`` now if ``x`` is concrete; otherwise return a
    closure for the build pipeline to invoke against the layer data."""
    if isinstance(x, (str, pl.Expr)):
        def _lazy(data):
            return eager_fn(_resolve(x, data))
        return _lazy
    return eager_fn(_resolve(x))


def cut_width(x, width, *, center=None, boundary=None, closed="right"):
    """Bin numeric ``x`` into intervals of width ``width``.

    Mirrors ``ggplot2::cut_width``. Default boundary = ``width / 2``,
    so a call like ``cut_width(carat, 0.1)`` produces breaks at
    ``..., 0.05, 0.15, 0.25, ...`` (each bin centered at a tenth).

    Parameters
    ----------
    x : Series, ndarray, str, or pl.Expr
        Numeric data to bin. A string column name or polars expression
        defers binning until the build pipeline supplies a DataFrame.
    width : float
        Bin width.
    center : float, optional
        Center of one bin. Mutually exclusive with ``boundary``.
    boundary : float, optional
        Position of one breakpoint. Mutually exclusive with ``center``.
    closed : {"right", "left"}, default ``"right"``
        Side of each interval that is closed (the side that includes
        its endpoint).

    Returns
    -------
    pl.Series (Enum) | callable
        Series of bin labels, or a callable awaiting a DataFrame when
        ``x`` is a column reference.
    """
    if center is not None and boundary is not None:
        raise ValueError(
            "cut_width: only one of `center` and `boundary` may be specified"
        )

    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        b = boundary
        if b is None:
            b = (center - width / 2) if center is not None else width / 2
        x_min = float(finite.min())
        x_max = float(finite.max())
        # Shift origin so one breakpoint coincides with ``b`` modulo ``width``.
        shift = float(np.floor((x_min - b) / width))
        origin = b + shift * width
        # Pad one extra slot on the right so x_max always falls inside a bin
        # regardless of which side is closed (matches ggplot2's
        # ``seq(min_x, max(range) + width, width)``).
        n_bins = max(int(np.ceil((x_max - origin) / width)), 1)
        breaks = origin + np.arange(n_bins + 2) * width
        return cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _maybe_lazy(x, _eager)


def cut_interval(x, n=None, length=None, *, closed="right"):
    """Cut ``x`` into ``n`` intervals of equal length, or intervals of
    given ``length``. Exactly one of ``n`` and ``length`` must be set.

    Mirrors ``ggplot2::cut_interval``.
    """
    if (n is None) == (length is None):
        raise ValueError(
            "cut_interval: specify exactly one of `n` and `length`"
        )

    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        x_min = float(finite.min())
        x_max = float(finite.max())
        if length is not None:
            # ``fullseq``: round endpoints out to multiples of ``length``.
            start = float(np.floor(x_min / length) * length)
            end = float(np.ceil(x_max / length) * length)
            n_bins = max(int(round((end - start) / length)), 1)
            breaks = start + np.arange(n_bins + 1) * length
            if breaks[-1] < x_max:
                breaks = np.concatenate([breaks, [breaks[-1] + length]])
        else:
            breaks = np.linspace(x_min, x_max, n + 1)
        return cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _maybe_lazy(x, _eager)


def cut_number(x, n, *, closed="right"):
    """Cut ``x`` into ``n`` intervals containing equal counts.

    Quantile-based bin edges. Raises if there isn't enough variation
    in the data to produce ``n`` distinct breaks (ggplot2 raises the
    same way).
    """
    def _eager(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pl.Series([], dtype=pl.Utf8)
        probs = np.linspace(0.0, 1.0, n + 1)
        breaks = np.quantile(finite, probs)
        if np.any(np.diff(breaks) <= 0):
            raise ValueError(
                f"cut_number: insufficient data values to produce {n} bins"
            )
        return cut(arr, breaks, right=(closed == "right"), include_lowest=True)

    return _maybe_lazy(x, _eager)
