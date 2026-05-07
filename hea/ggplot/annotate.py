"""``annotate(geom, ...)`` — one-shot layer with constant aesthetics.

ggplot2's ``annotate`` produces a layer whose data is built from the
keyword arguments rather than from the plot's main data. Common cases:

::

    annotate("text", x=3, y=20, label="Outlier")
    annotate("rect", xmin=0, xmax=5, ymin=0, ymax=10, fill="red", alpha=0.2)
    annotate("segment", x=0, y=0, xend=10, yend=10)

Scalars are broadcast to a common length when an iterable is mixed in.
``inherit_aes`` is forced off (the layer doesn't pull from the plot
mapping) and ``broadcast_panels`` is on (the annotation appears on every
facet panel, matching ggplot2 behaviour).
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .aes import Aes
from .layer import Layer
from .positions import resolve_position
from .stats.identity import StatIdentity


_NAME_TO_GEOM_CLS: dict[str, str] = {
    "point": "geoms.point.GeomPoint",
    "text": "geoms.text.GeomText",
    "label": "geoms.text.GeomLabel",
    "rect": "geoms.rect.GeomRect",
    "tile": "geoms.rect.GeomTile",
    "raster": "geoms.rect.GeomRaster",
    "polygon": "geoms.polygon.GeomPolygon",
    "segment": "geoms.segment.GeomSegment",
    "curve": "geoms.segment.GeomCurve",
    "path": "geoms.path.GeomPath",
    "line": "geoms.path.GeomPath",
    "step": "geoms.path.GeomPath",
    "ribbon": "geoms.ribbon.GeomRibbon",
    "area": "geoms.ribbon.GeomArea",
    "hline": "geoms.refline.GeomHline",
    "vline": "geoms.refline.GeomVline",
    "abline": "geoms.refline.GeomAbline",
    "errorbar": "geoms.errorbar.GeomErrorbar",
    "errorbarh": "geoms.errorbar.GeomErrorbarh",
    "linerange": "geoms.errorbar.GeomLinerange",
    "pointrange": "geoms.errorbar.GeomPointrange",
    "crossbar": "geoms.errorbar.GeomCrossbar",
}


def _resolve_geom(geom):
    if hasattr(geom, "draw_panel"):
        return geom
    if not isinstance(geom, str):
        raise TypeError(
            f"annotate: geom must be a Geom or string name, got "
            f"{type(geom).__name__}"
        )
    if geom not in _NAME_TO_GEOM_CLS:
        raise ValueError(
            f"annotate: unknown geom {geom!r}; valid names: "
            f"{sorted(_NAME_TO_GEOM_CLS)}"
        )
    import importlib

    path = _NAME_TO_GEOM_CLS[geom]
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(f"hea.ggplot.{module_path}")
    return getattr(module, class_name)()


def _value_length(v):
    if isinstance(v, str) or not isinstance(v, Iterable):
        return 1
    return len(list(v))


def _broadcast(v, n):
    if isinstance(v, str) or not isinstance(v, Iterable):
        return [v] * n
    lst = list(v)
    if len(lst) == 1:
        return lst * n
    if len(lst) != n:
        raise ValueError(
            f"annotate: values have inconsistent lengths "
            f"(got {len(lst)} vs {n}; broadcast requires scalar or matching length)"
        )
    return lst


def annotate(geom, *, x=None, y=None, xmin=None, xmax=None, ymin=None, ymax=None,
             xend=None, yend=None, **kwargs):
    """One-row (or N-row) annotation layer with constant aesthetics."""
    geom_obj = _resolve_geom(geom)

    aes_values: dict = {}
    for key, value in (
        ("x", x), ("y", y),
        ("xmin", xmin), ("xmax", xmax),
        ("ymin", ymin), ("ymax", ymax),
        ("xend", xend), ("yend", yend),
    ):
        if value is not None:
            aes_values[key] = value
    for key, value in kwargs.items():
        if value is None:
            continue
        # Canonicalise British/American spelling for colour.
        canonical = "colour" if key == "color" else key
        aes_values[canonical] = value

    if not aes_values:
        raise ValueError(
            "annotate: at least one aesthetic value (x, y, xmin, …) is required"
        )

    n = max((_value_length(v) for v in aes_values.values()), default=1)
    df = pl.DataFrame({k: _broadcast(v, n) for k, v in aes_values.items()})

    mapping = Aes(**{k: k for k in aes_values})

    return Layer(
        geom=geom_obj,
        stat=StatIdentity(),
        position=resolve_position("identity"),
        mapping=mapping,
        data=df,
        aes_params={},
        inherit_aes=False,
        broadcast_panels=True,
    )


# ---------------------------------------------------------------------------
# annotation_custom — drop a matplotlib Artist at given data coordinates
# ---------------------------------------------------------------------------

from .geoms.geom import Geom  # noqa: E402


class _GeomCustomArtist(Geom):
    """Geom that paints a single matplotlib Artist (provided by the user)
    inside the ``[xmin, xmax] × [ymin, ymax]`` bounding box. Used by
    :func:`annotation_custom`.

    The artist is deep-copied per panel so faceted plots don't share one
    artist across panels (matplotlib won't render the same Artist on
    multiple axes).
    """

    def __init__(self, artist, xmin, xmax, ymin, ymax):
        super().__init__()
        self.artist = artist
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def draw_panel(self, data, ax) -> None:
        import copy as _copy

        artist = _copy.copy(self.artist)
        # If the artist has a settable bbox / extent, reposition it.
        if hasattr(artist, "set_extent"):
            artist.set_extent((self.xmin, self.xmax, self.ymin, self.ymax))
        elif hasattr(artist, "set_bounds"):
            artist.set_bounds(self.xmin, self.ymin,
                              self.xmax - self.xmin,
                              self.ymax - self.ymin)
        ax.add_artist(artist)


def annotation_custom(grob, *, xmin=None, xmax=None, ymin=None, ymax=None):
    """Place an arbitrary matplotlib :class:`~matplotlib.artist.Artist`
    (often an :class:`~matplotlib.image.AxesImage` or a
    :class:`~matplotlib.patches.Patch`) inside the
    ``[xmin, xmax] × [ymin, ymax]`` bounding box in data coordinates.

    Pass an iterable of ``-Inf``/``Inf`` (or ``None``) to span the panel.
    Currently None / Inf is mapped to "use the artist's intrinsic bounds";
    explicit numeric bounds are required for Image-style artists that
    accept ``set_extent``.
    """
    if xmin is None or xmax is None or ymin is None or ymax is None:
        raise ValueError(
            "annotation_custom: xmin/xmax/ymin/ymax must all be set "
            "(panel-spanning '-Inf'/Inf shorthand is not yet supported)"
        )
    geom = _GeomCustomArtist(grob, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # A 1-row layer with a non-null aes mapping keeps `_drop_na` from
    # blanking the row (the geom doesn't read the data; the artist + bounds
    # carry all the information).
    return Layer(
        geom=geom,
        stat=StatIdentity(),
        position=resolve_position("identity"),
        mapping=Aes(group=1),
        data=pl.DataFrame({"_marker": [1]}),
        aes_params={},
        inherit_aes=False,
        broadcast_panels=True,
    )
