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


# ggplot2 sizes (incl. ``outlier.size``) are in mm; matplotlib's Line2D
# ``markersize`` is the marker DIAMETER in points. R/TeX convention:
# 72.27 pt/inch, 25.4 mm/inch.
_PT_PER_MM = 72.27 / 25.4


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
    key_glyph: str = "polygon"

    outlier_size: float = 1.5
    outlier_colour: str | None = None
    outlier_alpha: float | None = None

    def draw_panel(self, data, ax) -> None:
        import polars as pl

        from .._util import r_color

        if len(data) == 0:
            return

        # ``flipped_aes=True`` rows come from ``aes(x=…)`` (no y) and carry
        # x-prefixed stat columns; the cross-axis position is in ``y``.
        flipped = (
            "flipped_aes" in data.columns
            and bool(data["flipped_aes"].any())
        )
        if flipped:
            pos_col = "y"
            stat_cols = ("xmiddle", "xlower", "xupper", "xmin", "xmax")
        else:
            pos_col = "x"
            stat_cols = ("middle", "lower", "upper", "ymin", "ymax")
        med_c, q1_c, q3_c, lo_c, hi_c = stat_cols

        # ``ax.bxp`` only takes numeric ``positions``. When the cross-axis
        # column is discrete (``aes(x=species)``) we route the strings
        # through matplotlib's category unit so the box positions line up
        # with the axis labels. Register levels in the order R/ggplot2's
        # ``factor()`` would produce (sorted for plain strings, category
        # order for ``Categorical``/``Enum``); otherwise matplotlib uses
        # first-appearance order, which can leave the axis labelled
        # ``Adelie, Gentoo, Chinstrap`` instead of the expected sorted run.
        pos_series = data[pos_col]
        pos_is_discrete = pos_series.dtype in (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)
        pos_axis = ax.yaxis if flipped else ax.xaxis
        convert = ax.convert_yunits if flipped else ax.convert_xunits
        if pos_is_discrete:
            string_values = [str(v) for v in pos_series.to_list()]
            if pos_series.dtype in (pl.Categorical, pl.Enum):
                levels = [str(v) for v in pos_series.cat.get_categories().to_list()]
            else:
                levels = sorted(set(string_values))
            pos_axis.update_units(levels)
            positions = [float(p) for p in convert(string_values)]
        else:
            positions = [float(v) for v in pos_series.to_list()]

        # One row per box (per (cross-axis, group) tuple).
        boxes = []
        widths = []
        for row in data.iter_rows(named=True):
            fliers = row.get("outliers") or []
            boxes.append({
                "med": float(row[med_c]),
                "q1": float(row[q1_c]),
                "q3": float(row[q3_c]),
                "whislo": float(row[lo_c]),
                "whishi": float(row[hi_c]),
                "fliers": list(fliers),
            })
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
            orientation="horizontal" if flipped else "vertical",
            patch_artist=True,
            boxprops={"facecolor": fill, "edgecolor": edge, "alpha": alpha},
            medianprops={"color": edge},
            whiskerprops={"color": edge},
            capprops={"color": edge},
            flierprops={
                "marker": "o",
                "markerfacecolor": flier_color,
                "markeredgecolor": flier_color,
                "markersize": self.outlier_size * _PT_PER_MM,
                "alpha": flier_alpha,
            },
            manage_ticks=False,  # let scales handle ticks
        )


def _first(df, col, default):
    if col not in df.columns or len(df) == 0:
        return default
    val = df[col][0]
    return default if val is None else val


def geom_boxplot(mapping=None, data=None, *, stat="boxplot", position="dodge2",
                 outlier_size=1.5, outlier_colour=None, outlier_alpha=None,
                 coef=1.5, width=None, **kwargs):
    from ..layer import Layer
    from ..positions import resolve_position
    from ..stats import resolve_stat
    from ..stats.boxplot import StatBoxplot

    if stat == "boxplot":
        stat_obj = StatBoxplot(coef=coef, width=width)
    elif isinstance(stat, str):
        stat_obj = resolve_stat(stat)
    else:
        stat_obj = stat

    aes_params = {k: v for k, v in kwargs.items()
                  if k in {"colour", "color", "fill", "size", "linetype", "alpha"}}
    geom_params = {k: v for k, v in kwargs.items() if k not in aes_params}

    return Layer(
        geom=GeomBoxplot(outlier_size=outlier_size,
                         outlier_colour=outlier_colour,
                         outlier_alpha=outlier_alpha),
        stat=stat_obj,
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
