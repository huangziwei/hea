"""Patchwork-style plot composition — `p1 | p2`, `p1 / p2`, ``wrap_plots()``.

A :class:`PlotGrid` is a tree of plots laid out via
:class:`matplotlib.gridspec.GridSpec`. Operator semantics:

* ``p1 | p2`` — side-by-side (horizontal direction).
* ``p1 / p2`` — stacked (vertical direction).
* Same-direction extension: ``p1 | p2 | p3`` produces a single 1×3 grid,
  not nested — operators flatten when consecutive operators agree.
* Direction switch nests: ``(p1 | p2) / p3`` is a 2-row grid where
  row 0 is itself a 1×2 sub-grid containing ``p1`` and ``p2``.
* ``wrap_plots([p1, p2, p3, p4], nrow=2, ncol=2)`` for explicit grids.

Faceted ggplots compose correctly — :func:`render` accepts a
``subplotspec=`` argument and lays its facet sub-grid inside the parent
spec (see ``hea/ggplot/render.py``). ``ggplot + PlotGrid`` (and the
reverse) raises with a "did you mean ``|`` or ``/``?" message.
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
        if isinstance(other, ggplot):
            raise TypeError(
                "can't `+` a ggplot into a PlotGrid — did you mean "
                "`|` (horizontal) or `/` (vertical)?"
            )
        return NotImplemented

    def __radd__(self, other):
        from .core import ggplot
        if isinstance(other, ggplot):
            raise TypeError(
                "can't `+` a PlotGrid into a ggplot — did you mean "
                "`|` (horizontal) or `/` (vertical)?"
            )
        return NotImplemented

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
        # A 1×1 top spec lets the recursive _draw_into use the same
        # subgridspec mechanism for the root and every nested PlotGrid.
        top_spec = fig.add_gridspec(1, 1)[0, 0]
        self._draw_into(fig, top_spec)
        # User-provided sizing overrides the default per-panel formula.
        _resize_figure(fig, width=width, height=height, units=units,
                       figsize=figsize)
        return fig

    def _draw_into(self, fig, parent_spec) -> None:
        nrow, ncol = self._dims()
        sub_gs = parent_spec.subgridspec(nrow, ncol)
        for i, child in enumerate(self.children):
            r, c = self._cell_for(i)
            cell_spec = sub_gs[r, c]
            if isinstance(child, PlotGrid):
                child._draw_into(fig, cell_spec)
            else:
                child.draw(subplotspec=cell_spec)

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


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def wrap_plots(plots: list, *, nrow: int | None = None, ncol: int | None = None,
               byrow: bool = True) -> PlotGrid:
    """Programmatic grid layout. ``nrow``/``ncol`` default to ``ceil(sqrt(n))``;
    pass either or both to constrain the shape. ``byrow=True`` (default)
    fills row-major; ``byrow=False`` fills column-major.
    """
    return PlotGrid(
        children=list(plots),
        direction=_DIRECTION_GRID,
        nrow=nrow, ncol=ncol, byrow=byrow,
    )
