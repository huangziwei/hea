"""``class ggplot`` — the central plot object — and ``+``-dispatch.

A ``ggplot`` carries plot-wide data, the default mapping, the list of
layers, plus the (eventually) trained scales / facet / coord / theme /
labels. ``+`` is dispatched via :func:`functools.singledispatch` on the
right-hand side, so adding a new addable type later is a one-decorator
change rather than editing this file's ``__add__``.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field
from functools import singledispatch

import polars as pl

from .aes import Aes
from .coords.cartesian import CoordCartesian
from .coords.coord import Coord
from .facets.facet import Facet
from .facets.null import FacetNull
from .labels import Labels
from .layer import Layer
from .theme import Theme, theme_default


@dataclass
class ggplot:
    data: pl.DataFrame
    mapping: Aes = field(default_factory=Aes)
    layers: list = field(default_factory=list)
    coordinates: Coord = field(default_factory=CoordCartesian)
    facet: Facet = field(default_factory=FacetNull)
    theme: Theme = field(default_factory=theme_default)
    labels: dict = field(default_factory=dict)
    plot_env: dict = field(default_factory=dict, repr=False)

    def __init__(self, data: pl.DataFrame, mapping: Aes | None = None):
        # Captures globals + locals of the constructing frame so aes expressions
        # can resolve names the user had in scope (e.g. helper functions). Same
        # trick `hea.plot.dispatch.plot` uses; see `_frame_env` there.
        frame = inspect.currentframe().f_back
        env = {**frame.f_globals, **frame.f_locals} if frame is not None else {}

        self.data = data
        self.mapping = mapping if mapping is not None else Aes()
        self.layers = []
        self.coordinates = CoordCartesian()
        self.facet = FacetNull()
        self.theme = theme_default()
        self.labels = {}
        self.plot_env = env

    def __add__(self, other):
        return ggplot_add(other, self)

    def __radd__(self, other):
        # Supports rare `theme(...) + ggplot(...)` form.
        return ggplot_add(other, self)

    # ---- output ------------------------------------------------------

    def draw(self):
        from .build import build
        from .render import render
        return render(self, build(self))

    def show(self) -> None:
        import matplotlib.pyplot as plt
        self.draw()
        plt.show()

    def save(self, filename: str, *, width: float | None = None,
             height: float | None = None, dpi: int = 300, units: str = "in") -> None:
        fig = self.draw()
        if width is not None and height is not None:
            scale = {"in": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4}[units]
            fig.set_size_inches(width * scale, height * scale)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _repr_png_(self):
        import io

        fig = self.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        return buf.read()


def _copy_plot(plot: ggplot) -> ggplot:
    """Shallow copy with independent ``layers``/``labels`` so ``+`` is non-mutating."""
    out = copy.copy(plot)
    out.layers = list(plot.layers)
    out.labels = dict(plot.labels)
    return out


@singledispatch
def ggplot_add(thing, plot: ggplot) -> ggplot:
    raise TypeError(
        f"can't add {type(thing).__name__} to a ggplot — "
        f"only Layers, Coords, Facets, Themes, Labels (and lists thereof) are supported"
    )


@ggplot_add.register
def _(thing: Layer, plot):
    out = _copy_plot(plot)
    out.layers.append(thing)
    return out


@ggplot_add.register
def _(thing: Coord, plot):
    out = _copy_plot(plot)
    out.coordinates = thing
    return out


@ggplot_add.register
def _(thing: Facet, plot):
    out = _copy_plot(plot)
    out.facet = thing
    return out


@ggplot_add.register
def _(thing: Theme, plot):
    # Phase 1.8 will replace this with element-by-element merge.
    out = _copy_plot(plot)
    out.theme = thing
    return out


@ggplot_add.register
def _(thing: Labels, plot):
    out = _copy_plot(plot)
    out.labels.update(thing.labels)
    return out


@ggplot_add.register
def _(thing: list, plot):
    """Sugar: ``p + [geom_point(), geom_smooth()]`` — useful when layers are
    generated programmatically. ggplot2 supports this since 3.0."""
    for item in thing:
        plot = plot + item
    return plot
