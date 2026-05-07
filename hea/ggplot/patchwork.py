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
    # Top-level title/subtitle/caption + per-leaf tags via plot_annotation().
    annotation: "PlotAnnotation | None" = None

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
        if isinstance(other, PlotAnnotation):
            return self._with_annotation(other)
        # Patchwork's `+`-to-last-plot semantics: anything else
        # (Theme/Layer/Scale/Labels/Coord/Facet/...) gets applied to the
        # rightmost leaf plot. ``&`` (apply to all) is deferred polish.
        try:
            return self._apply_to_last_plot(other)
        except TypeError:
            raise TypeError(
                f"can't add {type(other).__name__} to a PlotGrid — "
                "PlotGrid only accepts other plots, plot_layout(), "
                "plot_annotation(), or anything addable to a ggplot"
            ) from None

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
            annotation=self.annotation,
        )

    def _with_annotation(self, annotation: "PlotAnnotation") -> "PlotGrid":
        """Return a copy with the annotation slot replaced. patchwork's
        last-`plot_annotation()`-wins semantics fall out of replacing rather
        than merging."""
        return PlotGrid(
            children=list(self.children),
            direction=self.direction,
            nrow=self.nrow, ncol=self.ncol, byrow=self.byrow,
            widths=self.widths, heights=self.heights,
            annotation=annotation,
        )

    def _apply_to_last_plot(self, thing) -> "PlotGrid":
        """Apply ``thing`` (a Theme/Layer/Scale/Labels/...) to the rightmost
        leaf plot in the tree — patchwork's `+`-on-grid behaviour."""
        from .core import ggplot

        if not self.children:
            raise TypeError("can't apply to last plot of an empty PlotGrid")
        last = self.children[-1]
        if isinstance(last, PlotGrid):
            new_last = last._apply_to_last_plot(thing)
        elif isinstance(last, ggplot):
            new_last = last + thing  # may itself raise — let it propagate
        else:
            raise TypeError(
                f"unexpected child type {type(last).__name__} in PlotGrid"
            )
        return PlotGrid(
            children=[*self.children[:-1], new_last],
            direction=self.direction,
            nrow=self.nrow, ncol=self.ncol, byrow=self.byrow,
            widths=self.widths, heights=self.heights,
            annotation=self.annotation,
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
            return _wrap_dims(n)
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
        """Build the figure and render the whole tree into it.

        Uses the gtable-style block engine: each ggplot child contributes a
        :class:`~hea.ggplot._block.PlotBlock`; nested :class:`PlotGrid`
        children become :class:`~hea.ggplot._block.SuperBlock`s. The
        outer super-gridspec takes max margins per side across siblings
        sharing a row or column so panels align by construction.
        """
        import matplotlib.pyplot as plt

        from ._block import compose_super_block, render_super_block
        from .core import _resolve_figsize

        sb = compose_super_block(self)
        target = _resolve_figsize(width=width, height=height, units=units,
                                   figsize=figsize)
        fig_w = target[0] if target is not None else sb.total_w_in
        fig_h = target[1] if target is not None else sb.total_h_in
        fig = plt.figure(figsize=(fig_w, fig_h))

        tag_iter = self._make_tag_iter()
        render_super_block(sb, fig, parent_subspec=None, tag_iter=tag_iter)
        return fig

    def _make_tag_iter(self):
        """If ``annotation.tag_levels`` is set, generate tags up front for
        every leaf and return an iterator. Otherwise return ``None``."""
        a = self.annotation
        if a is None or a.tag_levels is None:
            return None
        n = len(self.leaves())
        tags = _generate_tags(a.tag_levels, n)
        prefix = a.tag_prefix or ""
        suffix = a.tag_suffix or ""
        return iter(f"{prefix}{t}{suffix}" for t in tags)

    def leaves(self) -> list:
        """Return the depth-first list of leaf plots (reading order)."""
        out = []
        for c in self.children:
            if isinstance(c, PlotGrid):
                out.extend(c.leaves())
            else:
                out.append(c)
        return out

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


# ---------------------------------------------------------------------------
# plot_annotation — title/subtitle/caption + per-leaf tags
# ---------------------------------------------------------------------------


@dataclass
class PlotAnnotation:
    """Top-level annotation for a :class:`PlotGrid`.

    * ``title`` / ``subtitle`` / ``caption`` — figure-level text.
    * ``tag_levels`` — labels every leaf plot in reading order. Accepts:
      ``"a"``/``"A"``/``"1"`` for sequential letters/numbers,
      ``"i"``/``"I"`` for Roman numerals, or an explicit list.
    * ``tag_prefix``/``tag_suffix``/``tag_sep`` — surround/format the
      generated tag (e.g. ``tag_prefix="("``, ``tag_suffix=")"``).
    """

    title: str | None = None
    subtitle: str | None = None
    caption: str | None = None
    tag_levels: Any = None
    tag_prefix: str | None = None
    tag_suffix: str | None = None
    tag_sep: str | None = None


def plot_annotation(*, title=None, subtitle=None, caption=None,
                    tag_levels=None, tag_prefix=None, tag_suffix=None,
                    tag_sep=None) -> PlotAnnotation:
    """Patchwork-style figure annotation. Combine with a :class:`PlotGrid`
    via ``+``::

        (p1 | p2 | p3) + plot_annotation(title="Overview")
        (p1 + p2 + p3) + plot_annotation(tag_levels="A")  # → A, B, C
    """
    return PlotAnnotation(
        title=title, subtitle=subtitle, caption=caption,
        tag_levels=tag_levels, tag_prefix=tag_prefix, tag_suffix=tag_suffix,
        tag_sep=tag_sep,
    )


def _wrap_dims(n: int) -> tuple[int, int]:
    """Port of ggplot2's ``wrap_dims(n)`` default — which calls R's
    ``grDevices::n2mfrow`` and transposes so the result is ``(nrow, ncol)``.

    Special cases for small n preferred in R (more aesthetically balanced
    than blind ``ceil(sqrt)``):

    * n ≤ 3 → ``(1, n)`` (single row)
    * 4 ≤ n ≤ 6 → ``(2, ceil(n/2))``
    * 7 ≤ n ≤ 12 → ``(3, ceil(n/3))``
    * n > 12 → ``(ceil(sqrt(n)), ceil(sqrt(n)))``
    """
    if n <= 3:
        return (1, n)
    if n <= 6:
        return (2, (n + 1) // 2)
    if n <= 12:
        return (3, (n + 2) // 3)
    from math import ceil as _ceil, sqrt as _sqrt
    side = int(_ceil(_sqrt(n)))
    return (side, side)


def _attach_tag_to_axes(ax, tag: str) -> None:
    """Block-engine path: tag at upper-left of the panel axes, rendered as
    a Text artist outside the data area (above the panel's top edge)."""
    ax.text(
        0.0, 1.0, tag,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize="large", fontweight="bold",
    )


def _to_roman(n: int) -> str:
    if n <= 0:
        raise ValueError(f"can't render {n} as a Roman numeral")
    vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    out = []
    for v, s in zip(vals, syms):
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def _generate_tags(spec, n: int) -> list[str]:
    """Map a ``tag_levels`` spec onto ``n`` leaves."""
    if isinstance(spec, (list, tuple)):
        if len(spec) < n:
            raise ValueError(
                f"plot_annotation: tag_levels list has {len(spec)} entries "
                f"but the grid has {n} leaves"
            )
        return [str(s) for s in spec[:n]]
    if not isinstance(spec, str):
        raise TypeError(
            f"tag_levels must be a string or list, got {type(spec).__name__}"
        )
    if spec == "a":
        return [chr(ord("a") + i) for i in range(n)]
    if spec == "A":
        return [chr(ord("A") + i) for i in range(n)]
    if spec == "1":
        return [str(i + 1) for i in range(n)]
    if spec == "I":
        return [_to_roman(i + 1) for i in range(n)]
    if spec == "i":
        return [_to_roman(i + 1).lower() for i in range(n)]
    raise ValueError(
        f"unknown tag_levels {spec!r}; expected 'a'/'A'/'1'/'i'/'I' or a list"
    )
