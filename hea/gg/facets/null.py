"""``FacetNull`` — single-panel default. No real splitting."""

from __future__ import annotations

from .facet import Facet


class FacetNull(Facet):
    pass


def facet_null():
    return FacetNull()
