"""``stat_ecdf()`` — empirical cumulative distribution function.

For each unique value of the input column, emit ``y = fraction of points ≤ x``.
The output is a step function from 0 to 1 (unscaled; users can rescale).

Default geom is :class:`GeomPath` with ``step_direction="hv"`` (right-
continuous, matches R's ``ecdf``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .stat import Stat


@dataclass
class StatEcdf(Stat):
    n: int | None = None  # if set, evaluate at this many evenly-spaced grid points

    def compute_group(self, data, params):
        if "x" not in data.columns or len(data) == 0:
            return pl.DataFrame()
        x = data["x"].drop_nulls().to_numpy()
        if x.dtype.kind == "f":
            x = x[~np.isnan(x)]
        if len(x) == 0:
            return pl.DataFrame()

        x_sorted = np.sort(x)
        n = len(x_sorted)
        if self.n is None:
            # Evaluate at the unique sorted x values: produces an honest
            # step plot with one point per data observation (right-continuous).
            xs = x_sorted
            ys = (np.arange(1, n + 1)) / n
        else:
            xs = np.linspace(x_sorted[0], x_sorted[-1], int(self.n))
            ys = np.searchsorted(x_sorted, xs, side="right") / n

        return pl.DataFrame({"x": xs, "y": ys})


def stat_ecdf(mapping=None, data=None, *, geom="step", n=None,
              position="identity", **kwargs):
    """Step plot of the empirical CDF. ``n=`` evaluates at evenly-spaced x."""
    from ..geoms.path import GeomPath
    from ..layer import Layer
    from ..positions import resolve_position

    if isinstance(geom, str):
        if geom == "step":
            geom_obj = GeomPath(step_direction="hv")
        elif geom in ("path", "line"):
            geom_obj = GeomPath()
        else:
            raise ValueError(
                f"stat_ecdf: unknown geom {geom!r}; expected 'step', 'path', or 'line'"
            )
    else:
        geom_obj = geom

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}

    return Layer(
        geom=geom_obj,
        stat=StatEcdf(n=n),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
