"""``Layer`` — geom + stat + position + mapping + data, plus aes/geom/stat
parameter dicts. Each ``geom_*()`` constructor returns one of these."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .aes import Aes
from .geoms.geom import Geom
from .positions.position import Position
from .stats.stat import Stat


@dataclass
class Layer:
    geom: Geom
    stat: Stat
    position: Position
    mapping: Aes | None = None
    data: pl.DataFrame | None = None
    aes_params: dict = field(default_factory=dict)
    geom_params: dict = field(default_factory=dict)
    stat_params: dict = field(default_factory=dict)
    inherit_aes: bool = True
    show_legend: Any = True
    na_rm: bool = False
    # Skip facet.map_data so the layer renders on every panel (used by
    # annotate() and reference geoms that don't have facet variable columns).
    broadcast_panels: bool = False

    def __post_init__(self):
        # Sweep up any aesthetic-named kwargs the factory left in
        # ``geom_params`` (e.g. ``geom_point(x="mpg")`` — the factory's
        # narrow aes filter doesn't list ``x``/``y`` because they're
        # usually plot-level). Build-time promotion later turns
        # string-valued ones that match a column into mappings, so
        # ``geom_point(x="mpg", y="disp")`` Just Works at the layer
        # level. Anything not in :data:`_ALL_AES_NAMES` stays as a
        # geom_param (e.g. ``geom_segment(arrow=...)``).
        #
        # ``show_legend`` / ``na_rm`` / ``inherit_aes`` are Layer-level
        # control flags. Most geom factories don't accept them
        # explicitly, so they land in ``**kwargs`` → ``geom_params``;
        # promote them to the corresponding Layer field here so users
        # can pass them uniformly to any geom.
        if not self.geom_params:
            return
        from .aes import _ALL_AES_NAMES
        moved = {}
        keep = {}
        for k, v in self.geom_params.items():
            if k == "show_legend":
                self.show_legend = v
            elif k == "na_rm":
                self.na_rm = v
            elif k == "inherit_aes":
                self.inherit_aes = v
            elif k in _ALL_AES_NAMES:
                moved[k] = v
            else:
                keep[k] = v
        self.geom_params = keep
        if moved:
            self.aes_params = {**self.aes_params, **moved}
