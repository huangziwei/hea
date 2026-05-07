"""``coord_cartesian()`` and ``coord_fixed()`` — Cartesian coordinate systems.

* :func:`coord_cartesian` is the default identity coord. Optional ``xlim``/
  ``ylim`` *zoom* the view (vs. ``scale_*(limits=)`` / ``xlim``/``ylim`` which
  in ggplot2 *filter* — currently a deferred semantic gap, see Phase A plan).
* :func:`coord_fixed` enforces a fixed aspect ratio. ``ratio=1`` matches one
  unit of x to one unit of y.
"""

from __future__ import annotations

from dataclasses import dataclass

from .coord import Coord


@dataclass
class CoordCartesian(Coord):
    xlim: tuple | None = None
    ylim: tuple | None = None
    expand: bool = True

    def apply_to_axes(self, ax) -> None:
        """Override axis limits + aspect after scales have applied.

        Called from the renderer at the end so coord settings beat scale
        defaults (mirrors ggplot2: coord limits zoom rather than reshape data).
        """
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)


@dataclass
class CoordFixed(CoordCartesian):
    ratio: float = 1.0

    def apply_to_axes(self, ax) -> None:
        # ``adjustable``: when no user xlim/ylim, ``datalim`` lets matplotlib
        # stretch the data range to satisfy the aspect (single-plot case looks
        # natural). When the user pins limits via ``xlim=``/``ylim=``, switch
        # to ``box`` so the axes box resizes — otherwise matplotlib silently
        # ignores the limits to fulfil the aspect.
        adjustable = "box" if (self.xlim is not None or self.ylim is not None) else "datalim"
        ax.set_aspect(float(self.ratio), adjustable=adjustable)
        super().apply_to_axes(ax)


def coord_cartesian(*, xlim=None, ylim=None, expand=True):
    """Cartesian coords with optional zoom limits."""
    return CoordCartesian(xlim=xlim, ylim=ylim, expand=expand)


def coord_fixed(ratio=1.0, *, xlim=None, ylim=None, expand=True):
    """Cartesian coords with a fixed aspect ratio (``ratio = y unit / x unit``)."""
    return CoordFixed(ratio=ratio, xlim=xlim, ylim=ylim, expand=expand)
