"""``geom_boxplot()`` — Tukey-style box-and-whisker plots.

Reads the per-group five-number summary produced by :class:`StatBoxplot`
and renders via matplotlib's :meth:`Axes.bxp` (which takes pre-computed
stats, sparing us from re-implementing the whisker/cap drawing).

Per-row fill colouring lands when 1.5 brings discrete colour scales; for
now boxes share the layer's scalar fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geom import Geom


@dataclass
class GeomBoxplot(Geom):
    default_aes: dict = field(default_factory=lambda: {
        "colour": "black",
        "fill": "white",
        "size": 0.5,
        "linetype": "solid",
        "alpha": 1.0,
        "shape": "o",
    })
    required_aes: tuple = ("x", "lower", "middle", "upper", "ymin", "ymax")

    outlier_size: float = 1.5
    outlier_colour: str | None = None
    outlier_alpha: float | None = None

    def draw_panel(self, data, ax) -> None:
        import polars as pl

        from .._util import r_color

        if len(data) == 0:
            return

        # ``ax.bxp`` only takes numeric ``positions``. When x is a discrete
        # column (``aes(x = species, …)``) we route the strings through
        # matplotlib's category unit — same path ``ax.bar``/``plot`` use
        # internally — so the box positions line up with the axis labels.
        x_col = data["x"]
        x_is_discrete = x_col.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)
        if x_is_discrete:
            string_values = [str(v) for v in x_col.to_list()]
            ax.xaxis.update_units(string_values)
            x_positions = [float(p) for p in ax.convert_xunits(string_values)]
        else:
            x_positions = [float(v) for v in x_col.to_list()]

        # One row per box (per (x, group) tuple).
        boxes = []
        positions = []
        widths = []
        for i, row in enumerate(data.iter_rows(named=True)):
            fliers = row.get("outliers") or []
            boxes.append({
                "med": float(row["middle"]),
                "q1": float(row["lower"]),
                "q3": float(row["upper"]),
                "whislo": float(row["ymin"]),
                "whishi": float(row["ymax"]),
                "fliers": list(fliers),
            })
            positions.append(x_positions[i])
            widths.append(float(row.get("width", 0.75)))

        fill = r_color(_first(data, "fill", "white"))
        edge = r_color(_first(data, "colour", "black"))
        alpha = float(_first(data, "alpha", 1.0))
        flier_color = r_color(self.outlier_colour or edge)
        flier_alpha = self.outlier_alpha if self.outlier_alpha is not None else alpha

        ax.bxp(
            boxes,
            positions=positions,
            widths=widths,
            patch_artist=True,
            boxprops={"facecolor": fill, "edgecolor": edge, "alpha": alpha},
            medianprops={"color": edge},
            whiskerprops={"color": edge},
            capprops={"color": edge},
            flierprops={
                "marker": "o",
                "markerfacecolor": flier_color,
                "markeredgecolor": flier_color,
                "markersize": self.outlier_size * 4,
                "alpha": flier_alpha,
            },
            manage_ticks=False,  # let scales handle ticks
        )


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_boxplot(mapping=None, data=None, *, position="dodge2",
                 outlier_size=1.5, outlier_colour=None, outlier_alpha=None,
                 coef=1.5, width=0.75, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats.boxplot import StatBoxplot

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}

    return Layer(
        geom=GeomBoxplot(outlier_size=outlier_size,
                         outlier_colour=outlier_colour,
                         outlier_alpha=outlier_alpha),
        stat=StatBoxplot(coef=coef, width=width),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
    )
