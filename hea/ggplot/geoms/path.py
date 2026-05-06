"""``geom_path()``, ``geom_line()``, ``geom_step()`` — connected-line geoms.

* ``geom_path()`` connects points in the order they appear in the data.
* ``geom_line()`` connects points sorted by x — what most people want.
* ``geom_step()`` draws a stair-step line (constant pieces between x's).

All three iterate over groups: each unique value of the ``group``
aesthetic gets its own connected line. Aesthetic constants (``colour``,
``size``, ``linetype``, ``alpha``) apply per-group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import Geom


@dataclass
class GeomPath(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("x", "y")

    sort_by_x: bool = False
    step_direction: str | None = None  # None | "hv" | "vh" | "mid"

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return

        if "group" in data.columns:
            for _, sub in data.group_by("group", maintain_order=True):
                self._draw_one(sub, ax, r_lty, r_color)
        else:
            self._draw_one(data, ax, r_lty, r_color)

    def _draw_one(self, sub, ax, r_lty, r_color) -> None:
        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()

        if self.sort_by_x:
            order = np.argsort(x)
            x, y = x[order], y[order]

        if self.step_direction is not None:
            x, y = _stairstep(x, y, self.step_direction)

        colour = r_color(_first(sub, "colour", "black"))
        size = float(_first(sub, "size", 0.5))
        linetype = _first(sub, "linetype", "solid")
        alpha = float(_first(sub, "alpha", 1.0))

        # ggplot2 size in mm → matplotlib linewidth in pt (1pt ≈ 0.353mm).
        ax.plot(x, y, color=colour, linewidth=size * 2.83,
                linestyle=r_lty(linetype), alpha=alpha)


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def _stairstep(x, y, direction: str):
    """Convert (x, y) into stair-step coordinates.

    * ``"hv"`` — horizontal then vertical (post step; matches ``geom_step()``).
    * ``"vh"`` — vertical then horizontal (pre step).
    * ``"mid"`` — step at the midpoint between consecutive x values.
    """
    n = len(x)
    if n < 2:
        return x, y

    if direction == "hv":
        new_x = np.empty(2 * n - 1)
        new_y = np.empty(2 * n - 1)
        new_x[0::2] = x
        new_x[1::2] = x[1:]
        new_y[0::2] = y
        new_y[1::2] = y[:-1]
        return new_x, new_y

    if direction == "vh":
        new_x = np.empty(2 * n - 1)
        new_y = np.empty(2 * n - 1)
        new_x[0::2] = x
        new_x[1::2] = x[:-1]
        new_y[0::2] = y
        new_y[1::2] = y[1:]
        return new_x, new_y

    if direction == "mid":
        mids = (x[:-1] + x[1:]) / 2
        new_x = np.empty(3 * n - 2)
        new_y = np.empty(3 * n - 2)
        new_x[0::3] = x
        new_x[1::3] = mids
        new_x[2::3] = mids
        new_y[0::3] = y
        new_y[1::3] = y[:-1]
        new_y[2::3] = y[1:]
        # last entry trims off
        return new_x[:-1], new_y[:-1]

    raise ValueError(f"step direction {direction!r} not in {{hv, vh, mid}}")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _layer(geom, mapping, data, kwargs):
    from ..layer import Layer
    from ..positions.identity import PositionIdentity
    from ..stats.identity import StatIdentity

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}
    return Layer(
        geom=geom,
        stat=StatIdentity(),
        position=PositionIdentity(),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_path(mapping=None, data=None, **kwargs):
    return _layer(GeomPath(), mapping, data, kwargs)


def geom_line(mapping=None, data=None, **kwargs):
    return _layer(GeomPath(sort_by_x=True), mapping, data, kwargs)


def geom_step(mapping=None, data=None, *, direction="hv", **kwargs):
    return _layer(GeomPath(sort_by_x=True, step_direction=direction),
                  mapping, data, kwargs)
