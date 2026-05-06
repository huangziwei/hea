"""``geom_histogram()`` — :class:`GeomBar` + :class:`StatBin`."""

from __future__ import annotations

from .bar import GeomBar


def geom_histogram(mapping=None, data=None, *, bins=None, binwidth=None,
                   boundary=None, center=None, closed="right", **kwargs):
    from ..layer import Layer
    from ..positions.identity import PositionIdentity
    from ..stats.bin import StatBin

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomBar(),
        stat=StatBin(bins=bins, binwidth=binwidth, boundary=boundary,
                     center=center, closed=closed),
        position=PositionIdentity(),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
