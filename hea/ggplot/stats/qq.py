"""``stat_qq`` and ``stat_qq_line`` — Q-Q plot machinery.

* :class:`StatQq` ranks the ``sample`` column (or ``y`` if ``sample`` is
  unmapped) and pairs each rank with the corresponding theoretical quantile
  from a target distribution. Default distribution is ``"norm"``.
* :class:`StatQqLine` returns a 2-row frame defining the reference line
  through the 25th and 75th sample/theoretical quantile pair (R's robust
  Q-Q line).

ggplot2's ``aes(sample = col)`` is supported. Distribution is named via
``distribution=`` (any ``scipy.stats`` continuous distribution); extra
parameters via ``dparams={"loc": ..., "scale": ...}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .stat import Stat


def _sample_column(data: pl.DataFrame) -> np.ndarray:
    """Return the sample vector from the canonical ``sample`` column, or
    ``y`` as fallback. Drops nulls and NaNs."""
    if "sample" in data.columns:
        s = data["sample"]
    elif "y" in data.columns:
        s = data["y"]
    else:
        return np.array([])
    arr = s.drop_nulls().to_numpy()
    if arr.dtype.kind == "f":
        arr = arr[~np.isnan(arr)]
    return arr


def _ppf(distribution: str, dparams: dict):
    from scipy import stats as scstats

    # Allow R-style aliases: "normal" → scipy "norm".
    name = {"normal": "norm", "exp": "expon"}.get(distribution, distribution)
    dist = getattr(scstats, name, None)
    if dist is None:
        raise ValueError(
            f"stat_qq: unknown distribution {distribution!r}; "
            "must be a scipy.stats continuous distribution name"
        )

    def ppf(p):
        return dist.ppf(p, **(dparams or {}))
    return ppf


@dataclass
class StatQq(Stat):
    distribution: str = "norm"
    dparams: dict = field(default_factory=dict)

    def compute_group(self, data, params):
        sample = _sample_column(data)
        n = len(sample)
        if n == 0:
            return pl.DataFrame()

        sample_sorted = np.sort(sample)
        # ggplot2 / R use ppoints: (1:n - a) / (n + 1 - 2a) with a = 0.5 for n > 10
        # else a = 3/8. We match that for parity.
        a = 0.5 if n > 10 else 3 / 8
        probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
        ppf = _ppf(self.distribution, self.dparams)
        theoretical = ppf(probs)

        return pl.DataFrame({
            "x": theoretical,
            "y": sample_sorted,
            "theoretical": theoretical,
            "sample": sample_sorted,
        })


@dataclass
class StatQqLine(Stat):
    distribution: str = "norm"
    dparams: dict = field(default_factory=dict)
    quantiles: tuple = (0.25, 0.75)

    def compute_group(self, data, params):
        sample = _sample_column(data)
        if len(sample) < 2:
            return pl.DataFrame()
        ppf = _ppf(self.distribution, self.dparams)
        q_lo, q_hi = self.quantiles
        x1, x2 = float(ppf(q_lo)), float(ppf(q_hi))
        y1, y2 = float(np.quantile(sample, q_lo)), float(np.quantile(sample, q_hi))
        # Extend the line over the full theoretical range so it visibly
        # crosses every plotted Q-Q point.
        n = len(sample)
        a = 0.5 if n > 10 else 3 / 8
        probs = (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)
        theoretical = ppf(probs)
        x_lo, x_hi = float(theoretical.min()), float(theoretical.max())
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
        intercept = y1 - slope * x1
        return pl.DataFrame({
            "x": [x_lo, x_hi],
            "y": [intercept + slope * x_lo, intercept + slope * x_hi],
        })


def stat_qq(mapping=None, data=None, *, geom="point", distribution="norm",
            dparams=None, position="identity", **kwargs):
    """Q-Q plot points (theoretical vs sample quantiles).

    Map the data column via ``aes(sample=col)`` (preferred) or ``aes(y=col)``.
    Default geom is ``"point"``.
    """
    from ..geoms.point import GeomPoint
    from ..layer import Layer
    from ..positions import resolve_position

    if isinstance(geom, str):
        if geom != "point":
            raise ValueError(f"stat_qq: unknown geom {geom!r}; expected 'point'")
        geom_obj = GeomPoint()
    else:
        geom_obj = geom

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "shape", "alpha"}}

    return Layer(
        geom=geom_obj,
        stat=StatQq(distribution=distribution, dparams=dparams or {}),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def stat_qq_line(mapping=None, data=None, *, geom="path", distribution="norm",
                 dparams=None, quantiles=(0.25, 0.75),
                 position="identity", **kwargs):
    """Reference line for a Q-Q plot — a robust line through the 25th and
    75th sample/theoretical quantile pair (R-style)."""
    from ..geoms.path import GeomPath
    from ..layer import Layer
    from ..positions import resolve_position

    if isinstance(geom, str):
        if geom not in ("path", "line"):
            raise ValueError(f"stat_qq_line: unknown geom {geom!r}; "
                             "expected 'path' or 'line'")
        geom_obj = GeomPath()
    else:
        geom_obj = geom

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "size", "linetype", "alpha"}}

    return Layer(
        geom=geom_obj,
        stat=StatQqLine(distribution=distribution, dparams=dparams or {},
                        quantiles=tuple(quantiles)),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )


def geom_qq(mapping=None, data=None, **kwargs):
    """Alias for :func:`stat_qq` with the point geom — matches ggplot2."""
    return stat_qq(mapping=mapping, data=data, **kwargs)


def geom_qq_line(mapping=None, data=None, **kwargs):
    """Alias for :func:`stat_qq_line` — matches ggplot2."""
    return stat_qq_line(mapping=mapping, data=data, **kwargs)
