"""``geom_point()`` — scatter plot."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..aes import split_layer_kwargs
from .geom import Geom

_PT_PER_MM = 72.27 / 25.4


@dataclass
class GeomPoint(Geom):
    default_aes: dict = field(
        default_factory=lambda: {
            "colour": "black",
            "size": 1.5,
            "shape": "o",
            "alpha": 1.0,
            "fill": None,
            "stroke": 0.5,
        }
    )
    required_aes: tuple = ("x", "y")

    def draw_panel(self, data, ax) -> None:
        import numpy as np
        from matplotlib.colors import to_rgba

        from .._util import r_shape

        if len(data) == 0:
            return

        n = len(data)
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()

        colour = (
            data["colour"].to_list()
            if "colour" in data.columns
            else [self.default_aes["colour"]] * n
        )
        fill = (
            data["fill"].to_list() if "fill" in data.columns else colour
        )  # ggplot2: ``fill`` defaults to ``colour`` for fillable shapes.
        size_arr = (
            data["size"].to_numpy()
            if "size" in data.columns
            else np.full(n, self.default_aes["size"])
        )
        shape = (
            data["shape"].to_list()
            if "shape" in data.columns
            else [self.default_aes["shape"]] * n
        )
        alpha_arr = (
            data["alpha"].to_numpy()
            if "alpha" in data.columns
            else np.full(n, self.default_aes["alpha"])
        )

        s = (size_arr.astype(float) * _PT_PER_MM) ** 2

        translated = [r_shape(sh) for sh in shape]
        markers = [t[0] for t in translated]
        modes = [t[1] for t in translated]

        def _rgba(colours, alphas):
            """Bake per-row alpha into RGBA tuples. Avoids matplotlib's
            ``alpha=`` kwarg, which clobbers face=``"none"`` (transparent
            face becomes opaque black when alpha override hits a list of
            ``"none"`` strings)."""
            return [to_rgba(c, a) for c, a in zip(colours, alphas)]

        keys = list(dict.fromkeys(zip(markers, modes)))
        for marker, mode in keys:
            mask = np.array(
                [m == marker and md == mode for m, md in zip(markers, modes)]
            )
            sel_colour = [colour[i] for i, b in enumerate(mask) if b]
            sel_alpha = alpha_arr[mask]
            kw = {"s": s[mask], "marker": marker}
            if mode == "open":
                kw["facecolors"] = "none"
                kw["edgecolors"] = _rgba(sel_colour, sel_alpha)
            elif mode == "fillable":
                sel_fill = [fill[i] for i, b in enumerate(mask) if b]
                kw["facecolors"] = _rgba(sel_fill, sel_alpha)
                kw["edgecolors"] = _rgba(sel_colour, sel_alpha)
            else:
                kw["c"] = _rgba(sel_colour, sel_alpha)
            ax.scatter(x[mask], y[mask], **kw)


def geom_point(
    mapping=None,
    data=None,
    *,
    stat="identity",
    position="identity",
    na_rm=False,
    show_legend=True,
    inherit_aes=True,
    **kwargs,
):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=GeomPoint(),
        stat=resolve_stat(stat) if isinstance(stat, str) else stat,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
        inherit_aes=inherit_aes,
        show_legend=show_legend,
        na_rm=na_rm,
    )


def geom_jitter(
    mapping=None, data=None, *, width=None, height=None, seed=None, **kwargs
):
    """``geom_point(position=position_jitter(...))`` — shortcut matching ggplot2."""
    from ..positions.jitter import position_jitter

    return geom_point(
        mapping=mapping,
        data=data,
        position=position_jitter(width=width, height=height, seed=seed),
        **kwargs,
    )
