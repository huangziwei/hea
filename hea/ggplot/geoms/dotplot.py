"""``geom_dotplot()`` — Wilkinson-style dot plot.

Bin the data along the x axis, then stack one circular dot per observation
inside each bin. Useful for small samples where individual points carry
information that a histogram bar would erase.

Implementation follows ggplot2's defaults: ``binwidth = (range/30)``,
``binaxis="x"``, ``stackdir="up"``. Dot diameter = ``binwidth``. Stacking
direction: each observation in a bin sits one diameter above the previous.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .geom import Geom


_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomDotplot(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "black",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x",)

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        if len(data) == 0:
            return
        x = data["x"].drop_nulls().to_numpy()
        if x.dtype.kind == "f":
            x = x[~np.isnan(x)]
        if len(x) == 0:
            return

        # Default binwidth: (range / 30). Override with ``binwidth`` aes_param.
        rng = float(x.max() - x.min())
        bw = float(data["binwidth"][0]) if "binwidth" in data.columns else (rng / 30)
        if bw <= 0:
            bw = max(rng / 30, 1.0)

        # Assign each x to its bin (centre at bin_lo + bw/2).
        bin_lo = x.min()
        bin_idx = ((x - bin_lo) / bw).astype(int)
        # Within each bin, stack: the k-th point in the bin sits at y = (k+0.5)*bw.
        order = np.argsort(bin_idx, kind="stable")
        sorted_bins = bin_idx[order]
        # Position within bin = cumulative count up to (and including) self.
        # Use np.unique on the sorted array to compute per-bin starting indices.
        positions = np.zeros(len(x))
        prev = -1
        counter = 0
        for i, b in enumerate(sorted_bins):
            if b != prev:
                counter = 0
                prev = b
            positions[order[i]] = counter + 0.5
            counter += 1

        x_centres = bin_lo + (bin_idx + 0.5) * bw
        y_centres = positions * bw  # stack upward by diameter

        colour = r_color(_first(data, "colour", "black"))
        fill = r_color(_first(data, "fill", "black"))
        alpha = float(_first(data, "alpha", 1.0))

        # Use scatter with marker='o' and size in pt². Convert binwidth (data
        # units) into a display-scale dot — ggplot2 does this geometrically;
        # we approximate by mapping bw/2 (radius) to pt via the axes transform.
        # Cheap proxy: fixed dot size equal to ``binwidth × 2.83 × 6`` pt.
        size_pt = (bw * _PT_PER_MM * 6) ** 2 if bw > 0 else 36
        ax.scatter(
            x_centres, y_centres,
            s=size_pt, c=fill, edgecolors=colour, alpha=alpha, marker="o",
        )


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    v = df[col][0]
    return default if v is None else v


def geom_dotplot(mapping=None, data=None, *, stat="identity", position="identity",
                 binwidth=None, **kwargs):
    """Wilkinson dot plot. Each observation becomes one dot, stacked
    vertically inside its x-bin. ``binwidth`` defaults to ``range/30``."""
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "alpha"}}
    if binwidth is not None:
        aes_params["binwidth"] = binwidth
    return Layer(
        geom=GeomDotplot(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
