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
    broadcast_panels: bool = False

    def __post_init__(self):
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
