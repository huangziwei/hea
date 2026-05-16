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
    as_table: bool = True

    def facet_vars(self) -> list[str]:
        return list(self.facets)

    def grid_dims(self, n_panels: int) -> tuple[int, int]:
        if n_panels <= 0:
            return (1, 1)
        nrow, ncol = self.nrow, self.ncol
        if nrow is not None and ncol is not None:
            return (nrow, ncol)
        if ncol is None and nrow is None:
            # Match ggplot2's ``wrap_dims`` default — calls R's
            # ``grDevices::n2mfrow`` and transposes. Special cases for
            # small n preferred in R: n=3 → 1 row × 3 cols (not 2×2).
            if n_panels <= 3:
                return (1, n_panels)
            if n_panels <= 6:
                return (2, (n_panels + 1) // 2)
            if n_panels <= 12:
                return (3, (n_panels + 2) // 3)
            side = int(ceil(sqrt(n_panels)))
            return (side, side)
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

        # ``as_table=True`` (ggplot2 default): smallest factor level at
        # top-left, panels flow top-to-bottom (the "table" reading order).
        # ``as_table=False`` flips rows so the smallest level lands at
        # bottom-left and panels flow bottom-to-top — lattice's default
        # for ``xyplot(... | f)``, matching the "graphics" convention
        # Bates uses in lmmwr Fig. 3.1.
        if self.as_table:
            row_list = [i // ncol + 1 for i in range(n)]
        else:
            row_list = [nrow - i // ncol for i in range(n)]
        col_list = [i % ncol + 1 for i in range(n)]
        # ``PANEL`` is the flat-axes index the renderer uses to drop each
        # panel into its grid cell (``flat_axes[PANEL - 1]``). Derive it
        # from (ROW, COL) so ``as_table=False`` actually relocates panels
        # — without this, the sequential 1..n numbering would silently
        # cancel the row flip and the plot would look unchanged.
        panel_list = [(r - 1) * ncol + c for r, c in zip(row_list, col_list)]
        return unique.with_columns(
            PANEL=pl.Series(values=panel_list, dtype=pl.Int64),
            ROW=pl.Series(values=row_list, dtype=pl.Int64),
            COL=pl.Series(values=col_list, dtype=pl.Int64),
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


def facet_wrap(facets, *, ncol=None, nrow=None, scales="fixed", as_table=True):
    """Split into panels by ``facets``.

    Accepts ``"cyl"``, ``"~ cyl"``, ``"~ cyl + vs"``, or ``["cyl", "vs"]``.
    The R-style tilde is stripped if present.

    ``as_table`` (default ``True``, ggplot2 convention): when ``False``,
    fills panels bottom-to-top instead of top-to-bottom — matches
    lattice's ``xyplot(... | f)`` default, useful for Bates-style
    "panels increase upward" displays.
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
    return FacetWrap(facets=facet_list, ncol=ncol, nrow=nrow,
                     scales=scales, as_table=as_table)
