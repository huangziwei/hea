"""``geom_histogram()`` — :class:`GeomBar` + :class:`StatBin`."""

from __future__ import annotations

from .bar import GeomBar


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
