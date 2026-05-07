"""Patchwork-style plot composition — `p1 + p2`, `p1 | p2`, `p1 / p2`,
``wrap_plots()``.

A :class:`PlotGrid` is a tree of plots laid out via
:class:`matplotlib.gridspec.GridSpec`. Operator semantics mirror R's
``library(patchwork)``:

* ``p1 + p2`` — auto-layout (square-ish grid, ``ceil(sqrt(n))`` columns).
  ``+`` is also the ggplot2 layer-add operator; we resolve by rhs type
  (ggplot/PlotGrid → composition, everything else → layer add).
* ``p1 | p2`` — explicit side-by-side (horizontal direction).
* ``p1 / p2`` — explicit stacked (vertical direction).
* Same-direction extension: ``p1 | p2 | p3`` produces a single 1×3 grid,
  not nested — operators flatten when consecutive operators agree.
  ``p1 + p2 + p3`` similarly flattens into one 4-cell auto-layout grid.
* Direction switch nests: ``(p1 | p2) / p3`` is a 2-row grid where
  row 0 is itself a 1×2 sub-grid containing ``p1`` and ``p2``.
* ``wrap_plots([p1, p2, p3, p4], nrow=2, ncol=2)`` for explicit grids.

Faceted ggplots compose correctly — :func:`render` accepts a
``subplotspec=`` argument and lays its facet sub-grid inside the parent
spec (see ``hea/ggplot/render.py``). ``PlotGrid + Theme`` (or other
non-plot rhs) raises — themes/layers/scales must be applied to
individual plots before composition, not to a composition wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Any


_DIRECTION_H = "h"
_DIRECTION_V = "v"
_DIRECTION_GRID = "grid"


@dataclass
class PlotGrid:
    children: list = field(default_factory=list)
    direction: str = _DIRECTION_H
    nrow: int | None = None
    ncol: int | None = None
    byrow: bool = True
    # Relative widths per column / heights per row. Length must match the
    # rendered ``ncol`` / ``nrow`` (matplotlib enforces). Either left None
    # uses equal sizing.
    widths: list | None = None
    heights: list | None = None

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __or__(self, other):
        from .core import ggplot
        if isinstance(other, ggplot):
            return _h_combine(self, other)
        if isinstance(other, PlotGrid):
            return _h_combine(self, other)
        return NotImplemented

    def __ror__(self, other):
        from .core import ggplot
        if isinstance(other, ggplot):
            return _h_combine(other, self)
        return NotImplemented

    def __truediv__(self, other):
        from .core import ggplot
        if isinstance(other, (ggplot, PlotGrid)):
            return _v_combine(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        from .core import ggplot
        if isinstance(other, ggplot):
            return _v_combine(other, self)
        return NotImplemented

    def __add__(self, other):
        from .core import ggplot
        if isinstance(other, (ggplot, PlotGrid)):
            return _grid_combine(self, other)
        if isinstance(other, PlotLayout):
            return self._with_layout(other)
        raise TypeError(
            f"can't add {type(other).__name__} to a PlotGrid — "
            "themes/layers/scales must be applied to individual plots "
            "before composing, not to the composition wrapper"
        )

    def _with_layout(self, layout: "PlotLayout") -> "PlotGrid":
        """Return a copy with layout fields overridden by ``layout`` (only
        non-None fields take effect)."""
        return PlotGrid(
            children=list(self.children),
            direction=self.direction,
            nrow=layout.nrow if layout.nrow is not None else self.nrow,
            ncol=layout.ncol if layout.ncol is not None else self.ncol,
            byrow=layout.byrow if layout.byrow is not None else self.byrow,
            widths=layout.widths if layout.widths is not None else self.widths,
            heights=layout.heights if layout.heights is not None else self.heights,
        )

    # ------------------------------------------------------------------
    # Layout introspection
    # ------------------------------------------------------------------

    def _dims(self) -> tuple[int, int]:
        n = len(self.children)
        if n == 0:
            return (1, 1)
        if self.direction == _DIRECTION_H:
            return (1, n)
        if self.direction == _DIRECTION_V:
            return (n, 1)
        # grid
        nrow, ncol = self.nrow, self.ncol
        if nrow is not None and ncol is not None:
            return (nrow, ncol)
        if nrow is None and ncol is None:
            ncol = int(ceil(sqrt(n)))
            nrow = int(ceil(n / ncol))
            return (nrow, ncol)
        if ncol is None:
            return (nrow, int(ceil(n / nrow)))
        return (int(ceil(n / ncol)), ncol)

    def _cell_for(self, idx: int) -> tuple[int, int]:
        nrow, ncol = self._dims()
        if self.direction == _DIRECTION_H:
            return (0, idx)
        if self.direction == _DIRECTION_V:
            return (idx, 0)
        if self.byrow:
            return (idx // ncol, idx % ncol)
        return (idx % nrow, idx // nrow)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, *, width=None, height=None, units="in", figsize=None):
        """Build the figure and render the whole tree into it."""
        import matplotlib.pyplot as plt

        from .core import _resize_figure

        nrow, ncol = self._dims()
        # Default figsize echoes the per-panel formula used by facet_wrap.
        default_figsize = (3.0 * ncol, 2.5 * nrow)
        fig = plt.figure(figsize=default_figsize)
        # Use SubFigures (not subgridspec) so each child plot's supxlabel/
        # supylabel scopes to its own region rather than the entire figure.
        # SubFigure has the same API surface as Figure for our needs.
        self._draw_into(fig)
        _resize_figure(fig, width=width, height=height, units=units,
                       figsize=figsize)
        return fig

    def _draw_into(self, parent) -> None:
        """Render this grid inside ``parent`` (a :class:`~matplotlib.figure.Figure`
        or :class:`~matplotlib.figure.SubFigure`). Each child gets its own
        SubFigure cell, isolating ``supxlabel``/``supylabel`` to that region.
        """
        nrow, ncol = self._dims()
        kw = {}
        if self.widths is not None:
            if len(self.widths) != ncol:
                raise ValueError(
                    f"PlotGrid: widths has length {len(self.widths)} "
                    f"but the grid has {ncol} columns"
                )
            kw["width_ratios"] = list(self.widths)
        if self.heights is not None:
            if len(self.heights) != nrow:
                raise ValueError(
                    f"PlotGrid: heights has length {len(self.heights)} "
                    f"but the grid has {nrow} rows"
                )
            kw["height_ratios"] = list(self.heights)
        subfigs = parent.subfigures(nrow, ncol, squeeze=False, **kw)
        for i, child in enumerate(self.children):
            r, c = self._cell_for(i)
            cell = subfigs[r, c]
            if isinstance(child, PlotGrid):
                child._draw_into(cell)
            else:
                child.draw(parent=cell)

    def show(self, *, width=None, height=None, units="in", figsize=None) -> None:
        import matplotlib.pyplot as plt
        self.draw(width=width, height=height, units=units, figsize=figsize)
        plt.show()

    def save(self, filename: str, *, width=None, height=None, dpi=300,
             units="in", figsize=None) -> None:
        from .core import _resize_figure
        fig = self.draw()
        _resize_figure(fig, width=width, height=height, units=units,
                       figsize=figsize)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _repr_png_(self):
        import io

        import matplotlib.pyplot as plt

        fig = self.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Combination helpers — flatten consecutive same-direction grids, nest on
# direction switch.
# ---------------------------------------------------------------------------


def _as_children(thing) -> tuple[list, str]:
    """Return ``(children_list, direction_or_None)`` for combining.

    ``direction_or_None`` is the operand's direction if it's a flat
    ``PlotGrid`` of the same kind we're about to combine into; ``None``
    means "treat as a single unit (nest if needed)"."""
    if isinstance(thing, PlotGrid):
        return list(thing.children), thing.direction
    return [thing], None


def _h_combine(a, b):
    a_children, a_dir = _as_children(a)
    b_children, b_dir = _as_children(b)

    out_children: list = []
    if a_dir == _DIRECTION_H:
        out_children.extend(a_children)
    else:
        out_children.append(a)
    if b_dir == _DIRECTION_H:
        out_children.extend(b_children)
    else:
        out_children.append(b)
    return PlotGrid(children=out_children, direction=_DIRECTION_H)


def _v_combine(a, b):
    a_children, a_dir = _as_children(a)
    b_children, b_dir = _as_children(b)

    out_children: list = []
    if a_dir == _DIRECTION_V:
        out_children.extend(a_children)
    else:
        out_children.append(a)
    if b_dir == _DIRECTION_V:
        out_children.extend(b_children)
    else:
        out_children.append(b)
    return PlotGrid(children=out_children, direction=_DIRECTION_V)


def _grid_combine(a, b):
    """``+`` composition — auto-layout (R patchwork semantics). Square-ish
    grid via ``ceil(sqrt(n))`` columns. Flattens consecutive ``+`` chains:
    ``p1 + p2 + p3`` produces one 3-element grid, not nested."""
    a_children, a_dir = _as_children(a)
    b_children, b_dir = _as_children(b)

    out_children: list = []
    if a_dir == _DIRECTION_GRID:
        out_children.extend(a_children)
    else:
        out_children.append(a)
    if b_dir == _DIRECTION_GRID:
        out_children.extend(b_children)
    else:
        out_children.append(b)
    return PlotGrid(children=out_children, direction=_DIRECTION_GRID)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def wrap_plots(plots: list, *, nrow: int | None = None, ncol: int | None = None,
               byrow: bool = True, widths: list | None = None,
               heights: list | None = None) -> PlotGrid:
    """Programmatic grid layout. ``nrow``/``ncol`` default to ``ceil(sqrt(n))``;
    pass either or both to constrain the shape. ``byrow=True`` (default)
    fills row-major; ``byrow=False`` fills column-major.

    ``widths`` / ``heights`` are length-``ncol`` / length-``nrow`` lists of
    relative cell sizes (e.g. ``widths=[1, 2]`` makes column 1 twice as wide
    as column 0).
    """
    return PlotGrid(
        children=list(plots),
        direction=_DIRECTION_GRID,
        nrow=nrow, ncol=ncol, byrow=byrow,
        widths=widths, heights=heights,
    )


# ---------------------------------------------------------------------------
# plot_layout — patchwork-style "+ plot_layout(widths=[1, 2])" config
# ---------------------------------------------------------------------------


@dataclass
class PlotLayout:
    """Layout overrides for a :class:`PlotGrid`. Added via ``+`` on a
    composition: ``(p1 + p2) + plot_layout(widths=[1, 2])``. Only non-None
    fields take effect; everything else inherits from the grid.
    """

    widths: list | None = None
    heights: list | None = None
    nrow: int | None = None
    ncol: int | None = None
    byrow: bool | None = None


def plot_layout(*, widths=None, heights=None, nrow=None, ncol=None,
                byrow=None) -> PlotLayout:
    """Patchwork-style layout config. Combine with a :class:`PlotGrid` via
    ``+``::

        (p1 + p2) + plot_layout(widths=[1, 2])
        wrap_plots([p1, p2, p3, p4]) + plot_layout(heights=[2, 1])
    """
    return PlotLayout(
        widths=widths, heights=heights, nrow=nrow, ncol=ncol, byrow=byrow,
    )
