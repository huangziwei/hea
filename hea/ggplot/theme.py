"""Themes — control non-data plot appearance.

ggplot2's theme system is a tree of named *elements* (``axis.text``,
``panel.background``, etc.) where each element is one of:

* :class:`element_text` — text styling (font, size, colour, angle, …)
* :class:`element_line` — line styling (colour, size, linetype)
* :class:`element_rect` — rectangle styling (fill, colour, size, linetype)
* :class:`element_blank` — explicit "draw nothing"

Themes can be added (``+``); a *complete* theme (one of the
``theme_*()`` presets) replaces wholesale, while a partial ``theme(...)``
call merges field-by-field. This mirrors ggplot2's distinction between
``theme_bw()`` (complete) and ``theme(panel.background = ...)`` (partial).

The :func:`apply_theme` function in :mod:`render` translates the resolved
theme into matplotlib calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

@dataclass
class element_text:
    family: str | None = None
    face: str | None = None  # "plain" | "italic" | "bold" | "bold.italic"
    colour: str | None = None
    size: float | None = None
    hjust: float | None = None
    vjust: float | None = None
    angle: float | None = None
    lineheight: float | None = None


@dataclass
class element_line:
    colour: str | None = None
    size: float | None = None  # in mm; matplotlib pt = mm * 2.8454
    linetype: Any = None       # R lty int, name, or matplotlib dash tuple
    lineend: str | None = None


@dataclass
class element_rect:
    fill: str | None = None
    colour: str | None = None
    size: float | None = None
    linetype: Any = None


@dataclass
class element_blank:
    """Marker: 'render nothing for this element'. Distinct from ``None``,
    which means 'inherit / use matplotlib default'."""

    def __bool__(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def _merge_element(base, override):
    """Override's non-None fields win over base's."""
    if type(base) is not type(override):
        return override
    merged_kwargs = {}
    for f in fields(base):
        b = getattr(base, f.name)
        o = getattr(override, f.name)
        merged_kwargs[f.name] = o if o is not None else b
    return type(base)(**merged_kwargs)


@dataclass
class Theme:
    elements: dict = field(default_factory=dict)
    complete: bool = False

    def __add__(self, other: "Theme") -> "Theme":
        if not isinstance(other, Theme):
            return NotImplemented
        # A complete theme on the right replaces wholesale (matches
        # ggplot2's ``%+replace%`` for complete themes).
        if other.complete:
            return Theme(elements=dict(other.elements), complete=True)
        # Otherwise merge field-by-field.
        merged = dict(self.elements)
        for name, value in other.elements.items():
            existing = merged.get(name)
            if existing is None or isinstance(value, element_blank) \
                    or isinstance(existing, element_blank):
                merged[name] = value
            elif type(existing) is type(value):
                merged[name] = _merge_element(existing, value)
            else:
                merged[name] = value
        return Theme(elements=merged, complete=self.complete)

    def get(self, name, default=None):
        return self.elements.get(name, default)


def theme(*args, **kwargs) -> Theme:
    """Build a partial theme (overrides on top of the active theme).

    Pass kwargs with underscore-separated names that map to ggplot2's
    dot-separated element names, e.g. ``theme(panel_background=...)``
    sets ``"panel.background"``. To use the dotted form directly, pass
    a single dict: ``theme({"panel.background": ...})``.
    """
    elements: dict = {}
    if len(args) == 1 and isinstance(args[0], dict):
        elements.update(args[0])
    elif args:
        raise TypeError(
            "theme() takes optional kwargs and at most one positional dict"
        )
    for key, value in kwargs.items():
        canonical = key.replace("_", ".")
        elements[canonical] = value
    return Theme(elements=elements, complete=False)


# ---------------------------------------------------------------------------
# Theme presets — values mirror ggplot2/R/theme-defaults.R
# ---------------------------------------------------------------------------

def theme_gray() -> Theme:
    """ggplot2's default — gray panel, white gridlines, no axis spines."""
    return Theme(elements={
        "text": element_text(family="", size=11, colour="black"),
        "axis.text": element_text(size=8.8, colour="grey30"),
        "axis.title": element_text(size=11, colour="black"),
        "axis.title.y": element_text(angle=90),
        "axis.ticks": element_line(colour="grey20", size=0.25),
        "axis.line": element_blank(),
        "plot.title": element_text(size=14, colour="black", hjust=0),
        "plot.background": element_rect(fill="white"),
        "panel.background": element_rect(fill="#EBEBEB"),
        "panel.border": element_blank(),
        "panel.grid": element_line(colour="white"),
        "panel.grid.major": element_line(colour="white", size=0.5),
        "panel.grid.minor": element_line(colour="white", size=0.25),
        "strip.text": element_text(size=8.8, colour="grey10"),
        "strip.background": element_rect(fill="grey85"),
    }, complete=True)


def _preset_from(base_func, overrides) -> Theme:
    """Build a complete preset by merging ``overrides`` into ``base_func()``.

    Avoids running through ``Theme.__add__`` (whose "complete replaces"
    rule would discard the base's elements when both sides are complete)."""
    base = base_func()
    elements = dict(base.elements)
    elements.update(overrides)
    return Theme(elements=elements, complete=True)


def theme_bw() -> Theme:
    """White panel with a light gray grid and a black border."""
    return _preset_from(theme_gray, {
        "panel.background": element_rect(fill="white", colour="black"),
        "panel.grid.major": element_line(colour="grey92", size=0.5),
        "panel.grid.minor": element_line(colour="grey92", size=0.25),
        "panel.border": element_rect(fill=None, colour="grey20", size=0.5),
        "strip.background": element_rect(fill="grey85", colour="grey20"),
    })


def theme_minimal() -> Theme:
    """White panel, light gray gridlines, no border or axis spines."""
    return _preset_from(theme_bw, {
        "panel.background": element_blank(),
        "plot.background": element_blank(),
        "axis.line": element_blank(),
        "axis.ticks": element_blank(),
        "panel.border": element_blank(),
    })


def theme_classic() -> Theme:
    """White panel, no gridlines, black axis lines (statistics-textbook look)."""
    return _preset_from(theme_bw, {
        "panel.grid": element_blank(),
        "panel.grid.major": element_blank(),
        "panel.grid.minor": element_blank(),
        "panel.border": element_blank(),
        "axis.line": element_line(colour="black", size=0.5),
        "axis.ticks": element_line(colour="black", size=0.25),
    })


def theme_void() -> Theme:
    """Blanks out everything except data layers — useful for maps / collages."""
    return Theme(elements={
        "axis.text": element_blank(),
        "axis.title": element_blank(),
        "axis.ticks": element_blank(),
        "axis.line": element_blank(),
        "panel.background": element_blank(),
        "panel.grid": element_blank(),
        "panel.grid.major": element_blank(),
        "panel.grid.minor": element_blank(),
        "panel.border": element_blank(),
        "plot.background": element_blank(),
        "strip.text": element_blank(),
        "strip.background": element_blank(),
    }, complete=True)


def theme_dark() -> Theme:
    """Inverted gray theme — dark panel, light gridlines."""
    return _preset_from(theme_gray, {
        "panel.background": element_rect(fill="grey50"),
        "panel.grid.major": element_line(colour="grey42", size=0.5),
        "panel.grid.minor": element_line(colour="grey42", size=0.25),
    })


# Default applied at ``ggplot.__init__`` time.
def theme_default() -> Theme:
    return theme_gray()
