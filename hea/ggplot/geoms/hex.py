"""``geom_hex()`` — hexagonal bin counts via :meth:`matplotlib.axes.Axes.hexbin`.

Unlike most geoms, hexbin both bins and renders. The geom side is thin —
matplotlib's hexbin handles the bin geometry, count aggregation, and the
colormap. We expose ``bins=`` (gridsize) and pass through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom


@dataclass
class GeomHex(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        if len(data) == 0:
            return
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        # ggplot2 default bins ≈ 30; matplotlib's gridsize is x-direction count.
        bins = int(data["bins"][0]) if "bins" in data.columns else 30
        alpha = float(data["alpha"][0]) if "alpha" in data.columns else 1.0
        ax.hexbin(x, y, gridsize=bins, alpha=alpha)


def geom_hex(mapping=None, data=None, *, stat="identity", position="identity",
             bins=30, **kwargs):
    """Hexagonal binning of ``(x, y)``. ``bins`` (default 30) controls
    grid resolution. Colour scaling uses matplotlib's default colormap;
    polish via :func:`scale_fill_*` waits on the binhex stat output."""
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"alpha"}}
    aes_params["bins"] = bins
    return Layer(
        geom=GeomHex(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
