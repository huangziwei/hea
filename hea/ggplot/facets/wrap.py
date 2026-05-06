"""``facet_wrap()`` — split panels by one or more variables, wrapped to a grid.

ggplot2's canonical ``facet_wrap(~ cyl, ncol = 2)``. In Python: pass column
name(s) as a string or list, optional ``ncol``/``nrow`` and ``scales``.

* ``scales = "fixed"`` (default) — all panels share x and y limits.
* ``scales = "free"`` — each panel has independent x and y.
* ``scales = "free_x"`` / ``"free_y"`` — only one axis is independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt

import polars as pl

from .facet import Facet


@dataclass
class FacetWrap(Facet):
    facets: list = field(default_factory=list)
    ncol: int | None = None
    nrow: int | None = None

    def facet_vars(self) -> list[str]:
        return list(self.facets)

    def grid_dims(self, n_panels: int) -> tuple[int, int]:
        if n_panels <= 0:
            return (1, 1)
        nrow, ncol = self.nrow, self.ncol
        if nrow is not None and ncol is not None:
            return (nrow, ncol)
        if ncol is None and nrow is None:
            ncol = int(ceil(sqrt(n_panels)))
            nrow = int(ceil(n_panels / ncol))
            return (nrow, ncol)
        if ncol is None:
            return (nrow, int(ceil(n_panels / nrow)))
        return (int(ceil(n_panels / ncol)), ncol)

    def compute_layout(self, data: pl.DataFrame) -> pl.DataFrame:
        if not self.facets:
            return pl.DataFrame({"PANEL": [1], "ROW": [1], "COL": [1]})

        # Unique combinations across facet variables, in row-major order.
        unique = (
            data.select(self.facets)
            .unique(maintain_order=True)
            .sort(self.facets)
        )
        n = len(unique)
        nrow, ncol = self.grid_dims(n)

        rows = (i // ncol + 1 for i in range(n))
        cols = (i % ncol + 1 for i in range(n))
        return unique.with_columns(
            PANEL=pl.Series(values=list(range(1, n + 1)), dtype=pl.Int64),
            ROW=pl.Series(values=list(rows), dtype=pl.Int64),
            COL=pl.Series(values=list(cols), dtype=pl.Int64),
        )

    def map_data(self, data: pl.DataFrame, layout: pl.DataFrame) -> pl.DataFrame:
        if "PANEL" in data.columns or len(layout) == 0:
            return data
        if not self.facets:
            return data.with_columns(PANEL=pl.lit(1, dtype=pl.Int64))

        keys = [c for c in self.facets if c in data.columns]
        if not keys:
            # Layer doesn't carry the facet vars — assign all to panel 1.
            return data.with_columns(PANEL=pl.lit(1, dtype=pl.Int64))
        lookup = layout.select(["PANEL", *keys])
        return data.join(lookup, on=keys, how="left")


def facet_wrap(facets, *, ncol=None, nrow=None, scales="fixed"):
    """Split into panels by ``facets``.

    Accepts ``"cyl"``, ``"~ cyl"``, ``"~ cyl + vs"``, or ``["cyl", "vs"]``.
    The R-style tilde is stripped if present.
    """
    if isinstance(facets, str):
        s = facets.strip()
        if s.startswith("~"):
            s = s[1:].strip()
        facet_list = [v.strip() for v in s.split("+") if v.strip()]
    elif isinstance(facets, (list, tuple)):
        facet_list = [str(v) for v in facets]
    else:
        facet_list = [str(facets)]
    if scales not in ("fixed", "free", "free_x", "free_y"):
        raise ValueError(
            f"scales must be one of 'fixed'/'free'/'free_x'/'free_y'; got {scales!r}"
        )
    return FacetWrap(facets=facet_list, ncol=ncol, nrow=nrow, scales=scales)
