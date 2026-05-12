"""``coord_polar()`` — polar coordinate system, matplotlib-native.

Unlike ggplot2's `coord_polar` (which reprojects ``(x, y) → (cos(x)·y,
sin(x)·y)`` and renders on a Cartesian device, producing bent bars and
chord-gaps in paths), hea uses matplotlib's native ``projection="polar"``
axes. ``ax.bar(theta, height, width)`` draws proper wedges;
``ax.plot(theta, r)`` draws smooth radial paths; ``ax.fill_between(theta,
r1, r2)`` arc-bounds the fill. **No 2D reprojection.**

One 1D operation does happen: the x-aesthetic's trained range is
linearly rescaled to ``[0, 2π]`` so ordinal x (clarity, gear, …) fans
around the circle evenly the way ggplot2 does it. For data already in
radians (pycircstat2's case) the rescale is a no-op (factor = 1).

See ``.claude/plans/coord_polar.md`` for the full rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from .coord import Coord


@dataclass
class CoordPolar(Coord):
    """Polar coordinate system.

    Parameters
    ----------
    theta : {"x", "y"}
        Which aesthetic carries the angular variable. Default ``"x"``
        matches ggplot2 and pycircstat2. ``"y"`` is ggplot2's
        stacked-bar→pie idiom.
    start : float
        Offset of ``theta=0`` from matplotlib's default east-pointing
        zero, in **radians**. ``π/2`` puts zero at the top (north),
        ``π`` at the left (west), etc.
    direction : int
        ``1`` = clockwise (ggplot2 convention; compass-style).
        ``-1`` = counterclockwise (mathematical). The translation
        flips for matplotlib (whose ``set_theta_direction`` uses the
        opposite sign): we pass ``-direction`` in :meth:`apply_to_axes`.
    clip : {"on", "off"}
        Reserved for ggplot2 API parity; matplotlib polar always clips.
    """

    theta: str = "x"
    start: float = 0.0
    direction: int = 1
    clip: str = "on"
    is_linear: bool = False

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Identity. No 2D reprojection.

        The 1D theta-rescale lives in :meth:`rescale_theta` and runs
        at render time (needs the trained x-scale's range).
        """
        return data

    def rescale_theta(
        self, df: pl.DataFrame, x_range: tuple[float, float],
    ) -> pl.DataFrame:
        """Linearly map theta-aesthetic columns from ``x_range`` to ``[0, 2π]``.

        Matches ggplot2's coord_polar behaviour: the trained x-scale's
        domain spreads evenly around the circle, so 8 ordinal levels
        produce 8 wedges that span the full 2π.

        For data already in ``[0, 2π]`` (pycircstat2's case) the factor
        is 1.0 — no-op.

        ``x``/``xmin``/``xmax``/``xend`` get the full affine transform
        (``(v - lo) * factor``). ``width`` is a delta — multiplicative
        factor only, no offset.
        """
        lo, hi = x_range
        span = hi - lo
        if span <= 0:
            return df
        factor = (2 * math.pi) / span
        affine_cols = ("x", "xmin", "xmax", "xend")
        exprs = []
        for c in affine_cols:
            if c in df.columns and df[c].dtype.is_numeric():
                exprs.append(((pl.col(c) - lo) * factor).alias(c))
        if "width" in df.columns and df["width"].dtype.is_numeric():
            exprs.append((pl.col("width") * factor).alias("width"))
        return df.with_columns(exprs) if exprs else df

    def apply_to_axes(self, ax) -> None:
        """Configure the polar axes orientation. Called by render after
        the axes are created. matplotlib's ``set_theta_direction`` is
        inverted relative to ggplot2 (its ``1`` = CCW), so we negate.
        """
        ax.set_theta_offset(float(self.start))
        ax.set_theta_direction(-int(self.direction))


def coord_polar(
    theta: str = "x",
    *,
    start: float = 0.0,
    direction: int = 1,
    clip: str = "on",
) -> CoordPolar:
    """Polar coordinate system.

    See :class:`CoordPolar` for parameter details.

    Examples
    --------
    >>> # ggplot2's classic windrose / polar bar chart:
    >>> diamonds.ggplot().geom_bar(x="clarity", fill="clarity") + coord_polar()
    >>> # Compass-oriented, zero-at-top, clockwise (pycircstat2's default):
    >>> df.ggplot(aes(x="alpha", y="r")).geom_point() + coord_polar(start=π/2, direction=1)
    """
    return CoordPolar(theta=theta, start=start, direction=direction, clip=clip)
