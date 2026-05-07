"""Reference-line geoms — ``geom_hline``, ``geom_vline``, ``geom_abline``.

Unlike data-bound geoms, these don't inherit the plot's mapping or data —
they're constructed from constant intercepts/slopes. ggplot2's behaviour is
mirrored: each constructor builds its own one-row-per-line data frame and
the layer skips ``inherit_aes``. matplotlib's ``axhline``/``axvline``/
``axline`` already do the right thing on shared axes, so we just iterate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .geom import Geom


def _ensure_iterable(x, name: str):
    """Wrap a scalar in a list; pass an iterable through unchanged."""
    if x is None:
        raise TypeError(f"{name}= is required")
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
    return [x]


# Aesthetics that all three line geoms accept as constants on the layer.
_LINE_AES_PARAM_KEYS = frozenset({"colour", "color", "size", "linetype", "alpha"})


def _split_kwargs(kwargs):
    """Partition geom kwargs into aes-params (constants) vs geom-params."""
    aes_params = {k: v for k, v in kwargs.items() if k in _LINE_AES_PARAM_KEYS}
    geom_params = {k: v for k, v in kwargs.items() if k not in _LINE_AES_PARAM_KEYS}
    return aes_params, geom_params


@dataclass
class GeomHline(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("yintercept",)

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return
        yint = data["yintercept"].to_numpy()
        n = len(data)
        colour = data["colour"].to_list() if "colour" in data.columns else ["black"] * n
        size = data["size"].to_numpy() if "size" in data.columns else [0.5] * n
        linetype = data["linetype"].to_list() if "linetype" in data.columns else ["solid"] * n
        alpha = data["alpha"].to_numpy() if "alpha" in data.columns else [1.0] * n
        for i in range(n):
            ax.axhline(
                float(yint[i]),
                color=r_color(colour[i]),
                linewidth=float(size[i]) * 2.83,
                linestyle=r_lty(linetype[i]),
                alpha=float(alpha[i]),
            )


@dataclass
class GeomVline(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("xintercept",)

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return
        xint = data["xintercept"].to_numpy()
        n = len(data)
        colour = data["colour"].to_list() if "colour" in data.columns else ["black"] * n
        size = data["size"].to_numpy() if "size" in data.columns else [0.5] * n
        linetype = data["linetype"].to_list() if "linetype" in data.columns else ["solid"] * n
        alpha = data["alpha"].to_numpy() if "alpha" in data.columns else [1.0] * n
        for i in range(n):
            ax.axvline(
                float(xint[i]),
                color=r_color(colour[i]),
                linewidth=float(size[i]) * 2.83,
                linestyle=r_lty(linetype[i]),
                alpha=float(alpha[i]),
            )


@dataclass
class GeomAbline(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
    })
    required_aes: tuple = ("slope", "intercept")

    def draw_panel(self, data, ax) -> None:
        from ...plot._util import r_lty
        from .._util import r_color

        if len(data) == 0:
            return
        slope = data["slope"].to_numpy()
        intercept = data["intercept"].to_numpy()
        n = len(data)
        colour = data["colour"].to_list() if "colour" in data.columns else ["black"] * n
        size = data["size"].to_numpy() if "size" in data.columns else [0.5] * n
        linetype = data["linetype"].to_list() if "linetype" in data.columns else ["solid"] * n
        alpha = data["alpha"].to_numpy() if "alpha" in data.columns else [1.0] * n
        for i in range(n):
            # ``axline`` takes a point + slope; (0, intercept) is on the line.
            ax.axline(
                (0.0, float(intercept[i])),
                slope=float(slope[i]),
                color=r_color(colour[i]),
                linewidth=float(size[i]) * 2.83,
                linestyle=r_lty(linetype[i]),
                alpha=float(alpha[i]),
            )


def _build_layer(geom, data: pl.DataFrame, kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.identity import StatIdentity

    aes_params, geom_params = _split_kwargs(kwargs)
    return Layer(
        geom=geom,
        stat=StatIdentity(),
        position=resolve_position("identity"),
        mapping=None,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        # Reference-line geoms own their own data; never inherit from the plot.
        inherit_aes=False,
        show_legend=kwargs.pop("show_legend", True) if False else True,
        na_rm=False,
    )


def geom_hline(*, yintercept, **kwargs):
    """Horizontal line(s) at ``yintercept``. Pass a scalar or iterable."""
    yvals = _ensure_iterable(yintercept, "yintercept")
    data = pl.DataFrame({"yintercept": yvals})
    layer = _build_layer(GeomHline(), data, kwargs)
    # The layer needs a mapping so ``yintercept`` makes it through compute_aesthetics.
    from ..aes import Aes
    layer.mapping = Aes(yintercept="yintercept")
    return layer


def geom_vline(*, xintercept, **kwargs):
    """Vertical line(s) at ``xintercept``."""
    xvals = _ensure_iterable(xintercept, "xintercept")
    data = pl.DataFrame({"xintercept": xvals})
    layer = _build_layer(GeomVline(), data, kwargs)
    from ..aes import Aes
    layer.mapping = Aes(xintercept="xintercept")
    return layer


def geom_abline(*, slope=1.0, intercept=0.0, **kwargs):
    """Line(s) of form ``y = slope * x + intercept``.

    ``slope`` and ``intercept`` may both be scalars or iterables; iterables
    must have matching length (one line per pair).
    """
    s = _ensure_iterable(slope, "slope")
    b = _ensure_iterable(intercept, "intercept")
    if len(s) != len(b):
        if len(s) == 1:
            s = s * len(b)
        elif len(b) == 1:
            b = b * len(s)
        else:
            raise ValueError(
                f"slope and intercept must be the same length; got {len(s)} and {len(b)}"
            )
    data = pl.DataFrame({"slope": s, "intercept": b})
    layer = _build_layer(GeomAbline(), data, kwargs)
    from ..aes import Aes
    layer.mapping = Aes(slope="slope", intercept="intercept")
    return layer
