"""``class ggplot`` — the central plot object — and ``+``-dispatch.

A ``ggplot`` carries plot-wide data, the default mapping, the list of
layers, plus the (eventually) trained scales / facet / coord / theme /
labels. ``+`` is dispatched via :func:`functools.singledispatch` on the
right-hand side, so adding a new addable type later is a one-decorator
change rather than editing this file's ``__add__``.
"""

from __future__ import annotations

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


@dataclass
class ggplot:
    data: pl.DataFrame
    mapping: Aes = field(default_factory=Aes)
    layers: list = field(default_factory=list)
    scales: ScalesList = field(default_factory=ScalesList)
    coordinates: Coord = field(default_factory=CoordCartesian)
    facet: Facet = field(default_factory=FacetNull)
    theme: Theme = field(default_factory=theme_default)
    labels: dict = field(default_factory=dict)
    plot_env: dict = field(default_factory=dict, repr=False)

    def __init__(
        self,
        data: pl.DataFrame,
        mapping: Aes | None = None,
        *,
        _env: dict | None = None,
    ):
        # Captures globals + locals of the constructing frame so aes expressions
        # can resolve names the user had in scope (e.g. helper functions). Same
        # trick `hea.plot.dispatch.plot` uses; see `_frame_env` there.
        #
        # ``_env`` lets the caller pass a pre-captured env, used when ``ggplot``
        # is invoked through a wrapper (e.g. ``hea.DataFrame.ggplot``) — the
        # wrapper captures *its* caller's frame and passes it through, otherwise
        # ``f_back`` here would point at the wrapper, not the user.
        if _env is not None:
            env = _env
        else:
            frame = inspect.currentframe().f_back
            env = {**frame.f_globals, **frame.f_locals} if frame is not None else {}

        self.data = data
        self.mapping = mapping if mapping is not None else Aes()
        self.layers = []
        self.scales = ScalesList()
        self.coordinates = CoordCartesian()
        self.facet = FacetNull()
        self.theme = theme_default()
        self.labels = {}
        self.plot_env = env

    def __add__(self, other):
        # Patchwork: `ggplot + ggplot` and `ggplot + PlotGrid` compose into
        # an auto-layout grid (R `library(patchwork)` semantics). Layer/Theme/
        # Scale/etc. on the rhs falls through to the singledispatch table.
        from .patchwork import PlotGrid, _grid_combine
        if isinstance(other, (ggplot, PlotGrid)):
            return _grid_combine(self, other)
        return ggplot_add(other, self)

    def __radd__(self, other):
        # Supports rare `theme(...) + ggplot(...)` form.
        return ggplot_add(other, self)

    def __or__(self, other):
        """Patchwork horizontal composition."""
        from .patchwork import PlotGrid, _h_combine
        if isinstance(other, (ggplot, PlotGrid)):
            return _h_combine(self, other)
        return NotImplemented

    def __ror__(self, other):
        from .patchwork import PlotGrid, _h_combine
        if isinstance(other, (ggplot, PlotGrid)):
            return _h_combine(other, self)
        return NotImplemented

    def __truediv__(self, other):
        """Patchwork vertical composition."""
        from .patchwork import PlotGrid, _v_combine
        if isinstance(other, (ggplot, PlotGrid)):
            return _v_combine(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        from .patchwork import PlotGrid, _v_combine
        if isinstance(other, (ggplot, PlotGrid)):
            return _v_combine(other, self)
        return NotImplemented

    # ---- output ------------------------------------------------------

    def draw(self, ax=None, *, subplotspec=None,
             width=None, height=None, units="in", figsize=None):
        """Build the plot and render it to a matplotlib :class:`Figure`.

        ``ax``: optional existing axes to draw into (e.g. one cell from
        ``plt.subplot_mosaic``). When given, no new figure is created and
        ``ax.figure`` is returned (and ``width``/``height``/``figsize`` are
        ignored — sizing is the parent figure's responsibility).

        ``subplotspec``: a :class:`matplotlib.gridspec.SubplotSpec` to draw
        into. Used by patchwork composition (:class:`PlotGrid`) to host a
        ggplot — including faceted plots — inside one cell of a parent
        gridspec. Mutually exclusive with ``ax``.

        Sizing kwargs (interchangeable):

        * ``width=``/``height=`` with ``units="in"`` (default; also ``"cm"``
          or ``"mm"``).
        * ``figsize=(w, h)`` — matplotlib-style shorthand, always inches.

        ggplot2's grammar deliberately keeps size on the device, not the
        plot — see ``ggsave`` / ``options(repr.plot.width=...)``. We expose
        these kwargs as a Python convenience (also a Phase C deviation).
        """
        from .build import build
        from .render import render
        fig = render(self, build(self), ax=ax, subplotspec=subplotspec)
        if ax is None and subplotspec is None:
            _resize_figure(fig, width=width, height=height,
                           units=units, figsize=figsize)
        return fig

    def show(self, *, width=None, height=None, units="in", figsize=None) -> None:
        import matplotlib.pyplot as plt
        self.draw(width=width, height=height, units=units, figsize=figsize)
        plt.show()

    def save(self, filename: str, *, width: float | None = None,
             height: float | None = None, dpi: int = 300, units: str = "in",
             figsize=None) -> None:
        fig = self.draw()
        _resize_figure(fig, width=width, height=height,
                       units=units, figsize=figsize)
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _repr_png_(self):
        import io

        import matplotlib.pyplot as plt

        fig = self.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        # Close so Jupyter doesn't also auto-display the live figure
        # alongside the _repr_png_ bytes.
        plt.close(fig)
        buf.seek(0)
        return buf.read()


_UNIT_TO_INCHES = {"in": 1.0, "cm": 1 / 2.54, "mm": 1 / 25.4}


def _resize_figure(fig, *, width, height, units, figsize) -> None:
    """Resize ``fig`` to the requested width/height (in ``units`` or via
    matplotlib-style ``figsize=(w, h)``). No-op when nothing is requested.
    Re-runs ``tight_layout`` so the new size doesn't leave dead space.
    """
    if figsize is not None:
        if width is not None or height is not None:
            raise TypeError(
                "ggplot.draw/show/save: pass figsize=(w, h) OR width/height, "
                "not both"
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

    fig.set_size_inches(float(width) * units_in_inches,
                        float(height) * units_in_inches)
    try:
        fig.tight_layout()
    except Exception:
        # Some figure layouts (e.g. with a colorbar) emit a UserWarning and
        # may not converge — accept the new size without re-laying out rather
        # than failing the whole draw.
        pass


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
    # Theme merge semantics: complete themes (presets) replace, partial
    # ``theme(...)`` calls override field-by-field. Implementation in
    # ``Theme.__add__``.
    out.theme = out.theme + thing
    return out


@ggplot_add.register
def _(thing: Labels, plot):
    out = _copy_plot(plot)
    out.labels.update(thing.labels)
    return out


# ``guides(...)`` overrides per-aesthetic guide settings; rendering reads
# them via ``plot.guide_overrides``. Auto-build still covers the common
# case so this is currently a forward-compatible passthrough.
from .guides import Guides as _Guides  # noqa: E402

@ggplot_add.register
def _(thing: _Guides, plot):
    out = _copy_plot(plot)
    existing = getattr(out, "guide_overrides", {}) or {}
    out.guide_overrides = {**existing, **thing.overrides}
    return out


@ggplot_add.register
def _(thing: list, plot):
    """Sugar: ``p + [geom_point(), geom_smooth()]`` — useful when layers are
    generated programmatically. ggplot2 supports this since 3.0."""
    for item in thing:
        plot = plot + item
    return plot


# ---------------------------------------------------------------------------
# Fluent API auto-install (Phase B of `.claude/plans/method-based-ggplot-api.md`)
# ---------------------------------------------------------------------------

# Names whose `name(...)` produces a value that's `+`-able into a ggplot.
# Match by prefix:
_FLUENT_INSTALL_PREFIXES = (
    "geom_", "stat_", "scale_", "facet_", "coord_", "theme_",
)
# Match by exact name (top-level callables that aren't prefix-matched):
_FLUENT_INSTALL_EXACT = frozenset({
    "theme",
    "labs", "ggtitle", "xlab", "ylab", "xlim", "ylim", "lims", "annotate",
})

# Names that prefix-match but should NOT be installed:
_FLUENT_SKIP_PREFIXES = (
    "position_",  # kwargs to geoms, not addable on their own
    "element_",   # theme components, used inside theme(...) not added
    "after_",     # aes-modifiers (after_stat, after_scale)
)
# Exact names to skip even if they'd otherwise pattern-match:
_FLUENT_SKIP_EXACT = frozenset({
    "aes",        # mapping arg, not addable
    "ggplot",     # the class itself (also not in __all__-style match anyway)
})


def _should_install_fluent(name: str) -> bool:
    if name in _FLUENT_SKIP_EXACT:
        return False
    if any(name.startswith(p) for p in _FLUENT_SKIP_PREFIXES):
        return False
    if name in _FLUENT_INSTALL_EXACT:
        return True
    return any(name.startswith(p) for p in _FLUENT_INSTALL_PREFIXES)


def _install_fluent_methods(namespace: dict) -> None:
    """Install fluent methods on ``ggplot`` for every layer-addable name.

    Each matched name ``foo`` becomes a method ``ggplot.foo`` such that
    ``plot.foo(*a, **kw)`` is equivalent to ``plot + foo(*a, **kw)``.

    Called at the end of ``hea/ggplot/__init__.py`` once the package's
    namespace is fully populated. New geoms/scales/themes/etc. added to
    ``hea.ggplot.__all__`` automatically get fluent methods on the next
    package import — no per-name maintenance needed.

    Mirrors ``hea/dataframe.py:_install_series_subclass_overrides`` (Phase 4
    of the prerequisite plan ``dataframe-subclass-coverage.md``).
    """
    names = namespace.get("__all__") or [n for n in namespace if not n.startswith("_")]
    for name in names:
        if not _should_install_fluent(name):
            continue
        fn = namespace.get(name)
        if not callable(fn):
            continue

        # Bind ``fn`` as default arg so each closure captures by value
        # (avoids the late-binding pitfall in for-loop closures).
        def method(self, *args, _fn=fn, **kwargs):
            return self + _fn(*args, **kwargs)

        method.__name__ = name
        method.__qualname__ = f"ggplot.{name}"
        method.__doc__ = fn.__doc__
        setattr(ggplot, name, method)
