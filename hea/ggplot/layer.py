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
