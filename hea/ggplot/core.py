"""``class ggplot`` — the central plot object — and ``+``-dispatch.

A ``ggplot`` carries plot-wide data, the default mapping, the list of
layers, plus the (eventually) trained scales / facet / coord / theme /
labels. ``+`` is dispatched via :func:`functools.singledispatch` on the
right-hand side, so adding a new addable type later is a one-decorator
change rather than editing this file's ``__add__``.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from dataclasses import dataclass, field
from functools import singledispatch

import polars as pl

from .aes import Aes
from .coords.cartesian import CoordCartesian
from .coords.coord import Coord
from .facets.facet import Facet
from .facets.null import FacetNull
from .labels import Labels
from .layer import Layer
from .scales.list import ScalesList
from .scales.scale import Scale
from .theme import Theme, theme_default


class _PlotThemeHandle:
    """Returned by ``ggplot.theme``. Bridges the noun/verb collision
    between "the plot's current Theme" and the fluent ``theme(...)``
    factory: the handle delegates attribute access to the underlying
    :class:`Theme` (so ``plot.theme.get("legend.position")`` works) but
    is also callable (so ``plot.theme(aspect_ratio=1)`` returns a new
    plot with that theme merged in, mirroring ``plot + theme(...)``).

    Why a handle and not a plain field? ``theme`` is the only fluent
    install name (`_FLUENT_INSTALL_EXACT`) that also names a stored
    field on ``ggplot``. The dataclass-set instance attribute shadows
    the class-level fluent method — without this handle,
    ``plot.theme(...)`` would try to call the Theme instance and raise
    ``'Theme' object is not callable``. Other field names like
    ``coordinates`` / ``facet`` don't collide (no ``coordinates`` /
    ``facet`` factory in the install list).
    """

    __slots__ = ("_plot", "_theme")

    def __init__(self, plot, theme_obj):
        object.__setattr__(self, "_plot", plot)
        object.__setattr__(self, "_theme", theme_obj)

    def __call__(self, *args, **kwargs):
        from .theme import theme as theme_factory

        return self._plot + theme_factory(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._theme, name)

    def __add__(self, other):
        return self._theme + other

    def __radd__(self, other):
        return other + self._theme

    def __eq__(self, other):
        return self._theme == other

    def __bool__(self):
        return bool(self._theme)

    def __repr__(self):
        return repr(self._theme)


@dataclass
class ggplot:
    data: pl.DataFrame
    mapping: Aes = field(default_factory=Aes)
    layers: list = field(default_factory=list)
    scales: ScalesList = field(default_factory=ScalesList)
    coordinates: Coord = field(default_factory=CoordCartesian)
    facet: Facet = field(default_factory=FacetNull)
    _theme: Theme = field(default_factory=theme_default)
    labels: dict = field(default_factory=dict)
    plot_env: dict = field(default_factory=dict, repr=False)

    @property
    def theme(self):
        return _PlotThemeHandle(self, self._theme)

    @theme.setter
    def theme(self, value):
        if isinstance(value, _PlotThemeHandle):
            value = value._theme
        self._theme = value

    def __init__(
        self,
        data: pl.DataFrame,
        mapping: Aes | None = None,
        *,
        _env: dict | None = None,
        **aes_kwargs,
    ):
        """``ggplot(df, aes(x="col"))`` and ``ggplot(df, x="col")`` are
        equivalent — direct keyword args at the plot level are folded
        into the mapping. ``aes()`` is still supported for power users
        (composing/sharing mappings, layer-level overrides, after_stat).

        When both forms appear, kwargs override matching keys in
        ``mapping``::

            ggplot(df, aes(x="a", y="b"), color="c")  # x="a", y="b", color="c"
            ggplot(df, aes(x="a"), x="z")              # x="z"
        """
        if _env is not None:
            env = _env
        else:
            frame = inspect.currentframe().f_back
            env = {**frame.f_globals, **frame.f_locals} if frame is not None else {}

        if aes_kwargs:
            kwargs_aes = Aes()
            from .aes import _canon

            for k, v in aes_kwargs.items():
                kwargs_aes[_canon(k)] = v
            mapping = (mapping if mapping is not None else Aes()) + kwargs_aes

        self.data = data
        self.mapping = mapping if mapping is not None else Aes()
        self.layers = []
        self.scales = ScalesList()
        self.coordinates = CoordCartesian()
        self.facet = FacetNull()
        self._theme = theme_default()
        self.labels = {}
        self.plot_env = env

    def __add__(self, other):
        from .patchwork import GuideArea, PlotGrid, _grid_combine

        if isinstance(other, (ggplot, PlotGrid, GuideArea)):
            return _grid_combine(self, other)
        return ggplot_add(other, self)

    def __radd__(self, other):
        return ggplot_add(other, self)

    def __and__(self, other):
        """Patchwork's ``&`` — applied to a single plot, equivalent to ``+``.
        On a :class:`PlotGrid`, ``&`` broadcasts to every leaf plot.
        """
        return self + other

    def __or__(self, other):
        """Patchwork horizontal composition."""
        from .patchwork import GuideArea, PlotGrid, _h_combine

        if isinstance(other, (ggplot, PlotGrid, GuideArea)):
            return _h_combine(self, other)
        return NotImplemented

    def __ror__(self, other):
        from .patchwork import GuideArea, PlotGrid, _h_combine

        if isinstance(other, (ggplot, PlotGrid, GuideArea)):
            return _h_combine(other, self)
        return NotImplemented

    def __truediv__(self, other):
        """Patchwork vertical composition."""
        from .patchwork import GuideArea, PlotGrid, _v_combine

        if isinstance(other, (ggplot, PlotGrid, GuideArea)):
            return _v_combine(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        from .patchwork import GuideArea, PlotGrid, _v_combine

        if isinstance(other, (ggplot, PlotGrid, GuideArea)):
            return _v_combine(other, self)
        return NotImplemented

    def draw(
        self,
        ax=None,
        *,
        subplotspec=None,
        width=None,
        height=None,
        units="in",
        figsize=None,
    ):
        """Build the plot and render it to a matplotlib :class:`Figure`.

        ``ax``: optional existing axes to draw into (e.g. one cell from
        ``plt.subplot_mosaic``). When given, no new figure is created and
        ``ax.figure`` is returned (and sizing kwargs are ignored — the
        parent figure owns sizing).

        ``subplotspec``: a :class:`matplotlib.gridspec.SubplotSpec` to draw
        into — useful for manually integrating with a custom matplotlib
        gridspec.

        Sizing kwargs (interchangeable):

        * ``width=``/``height=`` with ``units="in"`` (default; also ``"cm"``
          or ``"mm"``).
        * ``figsize=(w, h)`` — matplotlib-style shorthand, always inches.

        ggplot2's grammar deliberately keeps size on the device, not the
        plot — see ``ggsave`` / ``options(repr.plot.width=...)``. We expose
        these kwargs as a Python convenience (a deliberate deviation).
        """
        from .build import build
        from .render import render

        if ax is None and subplotspec is None:
            import matplotlib.pyplot as plt

            from ._block import (
                default_figsize_for,
                measure_block,
                render_block,
            )

            bo = build(self)
            block = measure_block(self, bo)
            target = _resolve_figsize(
                width=width, height=height, units=units, figsize=figsize
            )
            fig_w, fig_h = target if target is not None else default_figsize_for(block)
            fig = plt.figure(figsize=(fig_w, fig_h))
            render_block(self, bo, block, fig=fig)
            return fig

        fig = render(self, build(self), ax=ax, subplotspec=subplotspec)
        return fig

    def show(self, *, width=None, height=None, units="in", figsize=None) -> None:
        import matplotlib.pyplot as plt

        self.draw(width=width, height=height, units=units, figsize=figsize)
        plt.show()

    def save(
        self,
        filename: str,
        *,
        width: float | None = None,
        height: float | None = None,
        dpi: int = 300,
        units: str = "in",
        figsize=None,
    ) -> None:
        fig = self.draw()
        _resize_figure(fig, width=width, height=height, units=units, figsize=figsize)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _repr_png_(self):
        import io

        import matplotlib.pyplot as plt

        fig = self.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


_UNIT_TO_INCHES = {"in": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4}


def _resolve_figsize(*, width, height, units, figsize) -> tuple[float, float] | None:
    """Resolve user-supplied size kwargs to ``(w_in, h_in)`` or ``None``."""
    if figsize is not None:
        if width is not None or height is not None:
            raise TypeError(
                "ggplot.draw/show/save: pass figsize=(w, h) OR width/height, not both"
            )
        if not (isinstance(figsize, (list, tuple)) and len(figsize) == 2):
            raise TypeError(
                f"figsize must be a (width, height) tuple/list; got {figsize!r}"
            )
        return (float(figsize[0]), float(figsize[1]))
    if width is None or height is None:
        return None
    if units not in _UNIT_TO_INCHES:
        raise ValueError(
            f"units must be one of {sorted(_UNIT_TO_INCHES)}; got {units!r}"
        )
    factor = _UNIT_TO_INCHES[units]
    return (float(width) * factor, float(height) * factor)


def _resize_figure(fig, *, width, height, units, figsize) -> None:
    """Resize ``fig`` to the requested width/height (in ``units`` or via
    matplotlib-style ``figsize=(w, h)``). No-op when nothing is requested.
    Re-runs ``tight_layout`` so the new size doesn't leave dead space.
    """
    if figsize is not None:
        if width is not None or height is not None:
            raise TypeError(
                "ggplot.draw/show/save: pass figsize=(w, h) OR width/height, not both"
            )
        if not (isinstance(figsize, (list, tuple)) and len(figsize) == 2):
            raise TypeError(
                f"figsize must be a (width, height) tuple/list; got {figsize!r}"
            )
        width, height = float(figsize[0]), float(figsize[1])
        units_in_inches = 1.0
    else:
        if width is None or height is None:
            return
        if units not in _UNIT_TO_INCHES:
            raise ValueError(
                f"units must be one of {sorted(_UNIT_TO_INCHES)}; got {units!r}"
            )
        units_in_inches = _UNIT_TO_INCHES[units]

    fig.set_size_inches(float(width) * units_in_inches, float(height) * units_in_inches)
    with contextlib.suppress(Exception):
        fig.tight_layout()


def _copy_plot(plot: ggplot) -> ggplot:
    """Shallow copy with independent ``layers``/``labels``/``scales`` so
    ``+`` is non-mutating."""
    out = copy.copy(plot)
    out.layers = list(plot.layers)
    out.labels = dict(plot.labels)
    out.scales = plot.scales.copy()
    return out


@singledispatch
def ggplot_add(thing, plot: ggplot) -> ggplot:
    raise TypeError(
        f"can't add {type(thing).__name__} to a ggplot — "
        f"only Layers, Coords, Facets, Themes, Labels (and lists thereof) are supported"
    )


@ggplot_add.register
def _(thing: Layer, plot):
    out = _copy_plot(plot)
    out.layers.append(thing)
    return out


@ggplot_add.register
def _(thing: Scale, plot):
    out = _copy_plot(plot)
    out.scales.add(thing)
    return out


@ggplot_add.register
def _(thing: Coord, plot):
    out = _copy_plot(plot)
    out.coordinates = thing
    return out


@ggplot_add.register
def _(thing: Facet, plot):
    out = _copy_plot(plot)
    out.facet = thing
    return out


@ggplot_add.register
def _(thing: Theme, plot):
    out = _copy_plot(plot)
    out.theme = out.theme + thing
    return out


@ggplot_add.register
def _(thing: Labels, plot):
    out = _copy_plot(plot)
    out.labels.update(thing.labels)
    return out


from .guides import Guides as _Guides


@ggplot_add.register
def _(thing: _Guides, plot):
    from .aes import _canon

    out = _copy_plot(plot)
    existing = getattr(out, "guide_overrides", {}) or {}
    new_overrides = {_canon(k): v for k, v in thing.overrides.items()}
    out.guide_overrides = {**existing, **new_overrides}
    return out


@ggplot_add.register
def _(thing: list, plot):
    """Sugar: ``p + [geom_point(), geom_smooth()]`` — useful when layers are
    generated programmatically. ggplot2 supports this since 3.0."""
    for item in thing:
        plot = plot + item
    return plot


_FLUENT_INSTALL_PREFIXES = (
    "geom_",
    "stat_",
    "scale_",
    "facet_",
    "coord_",
    "theme_",
)
_FLUENT_INSTALL_EXACT = frozenset(
    {
        "labs",
        "ggtitle",
        "xlab",
        "ylab",
        "xlim",
        "ylim",
        "lims",
        "annotate",
        "guides",
    }
)

_FLUENT_SKIP_PREFIXES = (
    "position_",  # kwargs to geoms, not addable on their own
    "element_",  # theme components, used inside theme(...) not added
    "after_",  # aes-modifiers (after_stat, after_scale)
)
_FLUENT_SKIP_EXACT = frozenset(
    {
        "aes",  # mapping arg, not addable
        "ggplot",  # the class itself (also not in __all__-style match anyway)
    }
)


def _should_install_fluent(name: str) -> bool:
    if name in _FLUENT_SKIP_EXACT:
        return False
    if any(name.startswith(p) for p in _FLUENT_SKIP_PREFIXES):
        return False
    if name in _FLUENT_INSTALL_EXACT:
        return True
    return any(name.startswith(p) for p in _FLUENT_INSTALL_PREFIXES)


def _install_fluent_methods(namespace: dict) -> None:
    """Install fluent methods on ``ggplot`` for every layer-addable name."""
    names = namespace.get("__all__") or [n for n in namespace if not n.startswith("_")]
    for name in names:
        if not _should_install_fluent(name):
            continue
        fn = namespace.get(name)
        if not callable(fn):
            continue

        def method(self, *args, _fn=fn, **kwargs):
            return self + _fn(*args, **kwargs)

        method.__name__ = name
        method.__qualname__ = f"ggplot.{name}"
        method.__doc__ = fn.__doc__
        setattr(ggplot, name, method)
