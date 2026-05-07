"""``stat_function()`` / ``geom_function()`` — plot ``y = fun(x)`` as a curve.

Unlike data-bound stats, this one synthesises its own ``(x, fun(x))`` rows
from the function and an x-range. The range comes from ``xlim`` if given, or
the plot's main data x column otherwise. ggplot2 treats ``stat_function`` and
``geom_function`` as aliases (different geom defaults aside) — we expose both.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .identity import StatIdentity


def _function_data_callable(fun, n, xlim, args):
    """Build a callable suitable for ``Layer.data`` that produces the
    synthetic ``{x, y}`` frame at build time.

    Resolving ``xlim`` from ``main`` (the plot's main data) is deferred until
    the layer runs so the function curve can pick up whatever x range the
    rest of the plot already implies. Override with explicit ``xlim=``.
    """
    def make(main):
        if xlim is not None:
            lo, hi = float(xlim[0]), float(xlim[1])
        elif isinstance(main, pl.DataFrame) and "x" in main.columns and len(main) > 0:
            x = main["x"].to_numpy()
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        else:
            lo, hi = 0.0, 1.0
        xs = np.linspace(lo, hi, int(n))
        ys = np.asarray(fun(xs, *args)) if args else np.asarray(fun(xs))
        return pl.DataFrame({"x": xs, "y": ys})
    return make


def stat_function(*, fun, n=101, xlim=None, args=(), geom="line",
                  mapping=None, **kwargs):
    """Plot ``y = fun(x)`` over ``xlim`` (or the plot's x range)."""
    from ..aes import Aes
    from ..geoms.path import GeomPath
    from ..geoms.point import GeomPoint
    from ..layer import Layer
    from ..positions import resolve_position

    if geom == "line":
        g = GeomPath(sort_by_x=True)
    elif geom == "point":
        g = GeomPoint()
    else:
        raise ValueError(f"stat_function: unknown geom {geom!r}; expected 'line' or 'point'")

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}

    layer = Layer(
        geom=g,
        stat=StatIdentity(),
        position=resolve_position("identity"),
        mapping=mapping if mapping is not None else Aes(x="x", y="y"),
        data=_function_data_callable(fun, n, xlim, args),
        aes_params=aes_params,
        # The synthesised ``{x, y}`` frame already carries the right columns;
        # don't merge in the plot's mapping, which might reference columns
        # that aren't in this layer's data.
        inherit_aes=False,
    )
    return layer


def geom_function(*, fun, n=101, xlim=None, args=(), mapping=None, **kwargs):
    """Alias for :func:`stat_function` with the line geom — matches ggplot2."""
    return stat_function(fun=fun, n=n, xlim=xlim, args=args, geom="line",
                         mapping=mapping, **kwargs)
