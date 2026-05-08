"""``geom_histogram()`` — :class:`GeomBar` + :class:`StatBin`.

``geom_freqpoly()`` lives here too since it shares the same stat — the
only difference is the geom (line through bin midpoints instead of bars).
"""

from __future__ import annotations

from .bar import GeomBar
from .path import GeomPath


def geom_histogram(mapping=None, data=None, *, stat="bin", bins=None, binwidth=None,
                   boundary=None, center=None, closed="right",
                   position="stack", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat
    from ..stats.bin import StatBin

    if stat == "bin":
        stat_obj = StatBin(bins=bins, binwidth=binwidth, boundary=boundary,
                           center=center, closed=closed)
    elif isinstance(stat, str):
        stat_obj = resolve_stat(stat)
    else:
        stat_obj = stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    geom_params = {k: v for k, v in kwargs.items() if k not in aes_params}

    return Layer(
        geom=GeomBar(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )


def geom_freqpoly(mapping=None, data=None, *, stat="bin", bins=None, binwidth=None,
                  boundary=None, center=None, closed="right", pad=True,
                  position="identity", **kwargs):
    """``geom_freqpoly()`` — frequency polygon.

    Same statistic as :func:`geom_histogram` (:class:`StatBin`) but
    drawn as a line connecting bin midpoints — useful for overlaying
    multiple distributions where stacked bars would be hard to read.
    Uses :class:`GeomPath`, so ``aes(colour=group)`` produces one line
    per group (sharing axes), and the legend key is a path glyph.

    ``pad=True`` (default) prepends/appends a zero-count bin on each
    side so the polygon returns to the baseline — matches ggplot2's
    ``geom_freqpoly`` default. Set ``pad=False`` to terminate at the
    outermost data bin instead.

    The ``stat=`` parameter accepts any registered stat name (resolved
    via :func:`resolve_stat`) or a stat instance — pass
    ``stat="identity"`` to plot pre-binned data.

    ``size`` (the layer's ``size`` aes) controls line width in mm; the
    geom converts to matplotlib pt internally. Defaults to ``GeomPath``'s
    ``size=0.5``.
    """
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat
    from ..stats.bin import StatBin

    if stat == "bin":
        stat_obj = StatBin(bins=bins, binwidth=binwidth, boundary=boundary,
                           center=center, closed=closed, pad=pad)
    elif isinstance(stat, str):
        stat_obj = resolve_stat(stat)
    else:
        stat_obj = stat

    # ``GeomPath`` has no ``fill`` — drop it from aes_params (matches
    # ggplot2: ``geom_freqpoly(fill=...)`` warns "Ignoring unknown
    # parameters: fill", we just skip silently).
    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}
    geom_params = {k: v for k, v in kwargs.items() if k not in aes_params}

    return Layer(
        geom=GeomPath(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
