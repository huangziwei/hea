"""``geom_bar()``, ``geom_col()`` — bar charts.

* ``geom_bar()`` — counts per discrete x via :class:`StatCount`.
* ``geom_col()`` — uses y as supplied (``stat_identity``).

``geom_histogram`` lives in ``histogram.py`` and reuses :class:`GeomBar`.

When the build pipeline produces ``ymin``/``ymax`` columns (which positions
like :class:`PositionStack` / :class:`PositionFill` add for stacking),
:meth:`GeomBar.draw_panel` reads them as ``bottom``/``top`` of each bar.
Otherwise bars draw from y=0 up to y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


@dataclass
class GeomBar(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": None,
        "fill": "grey35",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        from .._util import r_color

        x = data["x"].to_numpy()

        if "ymin" in data.columns and "ymax" in data.columns:
            ymin = data["ymin"].to_numpy()
            ymax = data["ymax"].to_numpy()
            height = ymax - ymin
            bottom = ymin
        else:
            height = data["y"].to_numpy()
            bottom = 0

        if "width" in data.columns:
            width = data["width"].to_numpy()
        else:
            width = 0.9

        # Per-row fill if the column is non-trivial (multiple distinct values
        # or anything other than the default), else scalar default. Lets
        # dodge/stack with explicit fills colour each bar individually.
        fill = _fill_value(data)
        edge = r_color(_edge_colour(data))
        alpha = float(_scalar(data, "alpha", default=1.0))

        # Under coord_flip the data has already been x↔y swapped by render:
        # ``data["x"]`` now holds the *values* (originally y) and
        # ``data["y"]`` (== ``height`` here) holds the *positions* (originally
        # x). Use ax.barh so bars extend along the visible x axis.
        if getattr(ax, "_hea_coord_flipped", False):
            ax.barh(height, width=x, height=width, left=bottom, color=fill,
                    edgecolor=edge, alpha=alpha, linewidth=0.5, align="center")
        else:
            ax.bar(x, height, width=width, bottom=bottom, color=fill,
                   edgecolor=edge, alpha=alpha, linewidth=0.5, align="center")


def _scalar(df, col, *, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _fill_value(df):
    from .._util import r_color

    if "fill" not in df.columns or len(df) == 0:
        return r_color("grey35")
    vals = df["fill"].to_list()
    if all(v is None for v in vals):
        return r_color("grey35")
    # Distinct fill values per row → return list; otherwise scalar.
    distinct = {v for v in vals if v is not None}
    if len(distinct) <= 1:
        return r_color(next(iter(distinct), "grey35"))
    return [r_color(v) if v is not None else r_color("grey35") for v in vals]


def _edge_colour(df):
    """Map ggplot2's ``colour = NA`` (no edge) to matplotlib's ``edgecolor='none'``."""
    if "colour" not in df.columns or len(df) == 0:
        return "none"
    val = df["colour"][0]
    if val is None:
        return "none"
    if isinstance(val, float) and np.isnan(val):
        return "none"
    return val


def geom_bar(mapping=None, data=None, *, stat="count", position="stack", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.count import StatCount

    if stat == "count":
        stat_obj = StatCount()
    elif hasattr(stat, "compute_layer"):
        stat_obj = stat
    else:
        raise ValueError(f"geom_bar: unknown stat {stat!r}")

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomBar(),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_col(mapping=None, data=None, *, position="identity", **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    return Layer(
        geom=GeomBar(),
        stat=StatIdentity(),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
