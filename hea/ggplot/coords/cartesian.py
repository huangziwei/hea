"""``coord_cartesian()``, ``coord_fixed()``, ``coord_quickmap()`` —
Cartesian coordinate systems.

* :func:`coord_cartesian` is the default identity coord. Optional ``xlim``/
  ``ylim`` *zoom* the view (vs. ``scale_*(limits=)`` / ``xlim``/``ylim`` which
  in ggplot2 *filter* — currently a deferred semantic gap, see Phase A plan).
* :func:`coord_fixed` enforces a fixed aspect ratio. ``ratio=1`` matches one
  unit of x to one unit of y.
* :func:`coord_quickmap` approximates a Mercator-style map projection by
  setting the aspect ratio to ``1 / cos(mean_lat · π/180)`` — good enough
  for small/regional maps where the curvature of the earth doesn't matter.
"""

from __future__ import annotations

import math
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


@dataclass
class CoordQuickmap(CoordCartesian):
    """``coord_quickmap()`` — quick approximation of a Mercator projection.

    Sets the aspect ratio to ``1 / cos(mean_lat · π/180)`` so one degree
    of longitude near the mean latitude renders the same screen distance
    as one degree of latitude. Good for small/regional maps; for global
    or high-latitude maps you want a real projection (``coord_map`` in
    ggplot2 — not yet ported).

    Mirrors ggplot2's R/coord-quickmap.R::CoordQuickmap$aspect:

        lat   <- mean(y_range)
        ratio <- cos(lat · π/180)
        # ggplot2 returns Δy/Δx / ratio for its grid system.

    Matplotlib's ``set_aspect`` takes ``dy/dx``, so we feed it
    ``1/ratio`` directly — the panel size handles the Δy/Δx part.
    """

    def apply_to_axes(self, ax) -> None:
        super().apply_to_axes(ax)
        y_lo, y_hi = ax.get_ylim()
        mean_lat = 0.5 * (float(y_lo) + float(y_hi))
        cos_lat = math.cos(mean_lat * math.pi / 180.0)
        # Guard the polar edge case: ``cos(±90°) = 0`` would blow up the
        # aspect. Fall back to 1:1 (pure cartesian) — same effective
        # behaviour as ggplot2 at the limit.
        ratio = 1.0 / cos_lat if abs(cos_lat) > 1e-9 else 1.0
        adjustable = (
            "box" if (self.xlim is not None or self.ylim is not None) else "datalim"
        )
        ax.set_aspect(ratio, adjustable=adjustable)


def coord_cartesian(*, xlim=None, ylim=None, expand=True):
    """Cartesian coords with optional zoom limits."""
    return CoordCartesian(xlim=xlim, ylim=ylim, expand=expand)


def coord_fixed(ratio=1.0, *, xlim=None, ylim=None, expand=True):
    """Cartesian coords with a fixed aspect ratio (``ratio = y unit / x unit``)."""
    return CoordFixed(ratio=ratio, xlim=xlim, ylim=ylim, expand=expand)


def coord_quickmap(*, xlim=None, ylim=None, expand=True):
    """Quick Mercator-ish map projection — sets aspect from the mean latitude.

    Mirrors ggplot2's ``coord_quickmap()``. For small regional maps
    (``map_data("nz")``, US states, etc.) it gets shapes roughly right
    without the cost of a real projection.
    """
    return CoordQuickmap(xlim=xlim, ylim=ylim, expand=expand)
