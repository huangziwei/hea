"""``facet_grid()`` — split panels into a row × column matrix.

ggplot2's ``facet_grid(rows ~ cols)``. The matrix has one column per unique
value of the column-grouping variable(s) and one row per unique value of
the row-grouping variable(s). Empty (no-data) cells still get an axes.

Differs from :class:`FacetWrap` in two places:

* ``share_axes`` — ``"free_x"`` shares x within each *column*; ``"free_y"``
  shares y within each *row* (matplotlib ``sharex='col'`` / ``sharey='row'``).
* ``panel_labels`` — top strip on the first row only (column values),
  right strip on the last column only (row values).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .facet import Facet


@dataclass
class FacetGrid(Facet):
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    _nrow: int = field(default=1, init=False, repr=False)
    _ncol: int = field(default=1, init=False, repr=False)

    def facet_vars(self) -> list[str]:
        return list(self.rows) + list(self.cols)

    def grid_dims(self, n_panels: int) -> tuple[int, int]:
        return (self._nrow, self._ncol)

    def share_axes(self) -> tuple:
        sharex = {"fixed": True, "free_y": True, "free_x": "col", "free": "col"}[self.scales]
        sharey = {"fixed": True, "free_x": True, "free_y": "row", "free": "row"}[self.scales]
        return (sharex, sharey)

    def compute_layout(self, data: pl.DataFrame) -> pl.DataFrame:
        if not self.rows and not self.cols:
            self._nrow = 1
            self._ncol = 1
            return pl.DataFrame({"PANEL": [1], "ROW": [1], "COL": [1]})

        rows_u = self._unique_axis(data, self.rows, "ROW") if self.rows else None
        cols_u = self._unique_axis(data, self.cols, "COL") if self.cols else None

        self._nrow = len(rows_u) if rows_u is not None else 1
        self._ncol = len(cols_u) if cols_u is not None else 1

        if rows_u is not None and cols_u is not None:
            layout = rows_u.join(cols_u, how="cross")
        elif rows_u is not None:
            layout = rows_u.with_columns(COL=pl.lit(1, dtype=pl.Int64))
        else:
            layout = cols_u.with_columns(ROW=pl.lit(1, dtype=pl.Int64))

        return layout.with_columns(
            PANEL=((pl.col("ROW") - 1) * self._ncol + pl.col("COL")).cast(pl.Int64)
        )

    @staticmethod
    def _unique_axis(data: pl.DataFrame, vars_: list[str], idx_name: str) -> pl.DataFrame:
        return (
            data.select(vars_)
            .unique(maintain_order=True)
            .sort(vars_)
            .with_row_index(name="_idx")
            .with_columns(**{idx_name: (pl.col("_idx") + 1).cast(pl.Int64)})
            .drop("_idx")
        )

    def map_data(self, data: pl.DataFrame, layout: pl.DataFrame) -> pl.DataFrame:
        if "PANEL" in data.columns or len(layout) == 0:
            return data
        if not self.rows and not self.cols:
            return data.with_columns(PANEL=pl.lit(1, dtype=pl.Int64))
        keys = [c for c in self.facet_vars() if c in data.columns]
        if not keys:
            return data.with_columns(PANEL=pl.lit(1, dtype=pl.Int64))
        return data.join(layout.select(["PANEL", *keys]), on=keys, how="left")

    def panel_labels(self, panel_row: dict, layout: pl.DataFrame) -> dict:
        labels = {"top": "", "right": ""}
        if panel_row.get("ROW") == 1 and self.cols:
            labels["top"] = ", ".join(
                f"{panel_row[v]}" for v in self.cols if v in panel_row
            )
        if panel_row.get("COL") == self._ncol and self.rows:
            labels["right"] = ", ".join(
                f"{panel_row[v]}" for v in self.rows if v in panel_row
            )
        return labels


def facet_grid(formula=None, *, rows=None, cols=None, scales="fixed"):
    """Split into a row × column grid of panels.

    Two API forms accepted:

    * **Formula string** — ``facet_grid("rows ~ cols")``. R-style. Use
      ``"."`` on either side to omit (``". ~ year"`` for col-only;
      ``"country ~ ."`` for row-only). Multi-var with ``+``:
      ``"country + region ~ year"``.
    * **Keyword** — ``facet_grid(rows="country", cols=["year", "month"])``.
      Strings split on ``+`` like the formula form; lists are taken
      literally.

    ``scales`` controls axis sharing:

    * ``"fixed"`` (default) — all panels share x and y.
    * ``"free_x"`` — x scale varies between columns (panels in the same
      column share x); y is shared across all panels.
    * ``"free_y"`` — y scale varies between rows; x is shared across all.
    * ``"free"`` — both directions independent (per column for x, per row
      for y).
    """
    if formula is not None and (rows is not None or cols is not None):
        raise ValueError(
            "facet_grid: pass either a formula or rows=/cols=, not both"
        )

    if formula is not None:
        if not isinstance(formula, str):
            raise TypeError(
                "facet_grid: formula must be a string like 'rows ~ cols'; "
                f"got {type(formula).__name__}"
            )
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(
                f"facet_grid: formula must be 'rows ~ cols'; got {formula!r}"
            )
        row_list = _parse_side(parts[0])
        col_list = _parse_side(parts[1])
    else:
        row_list = _to_list(rows)
        col_list = _to_list(cols)

    if not row_list and not col_list:
        raise ValueError(
            "facet_grid: at least one of rows/cols must be non-empty "
            "(use facet_null() for a single panel)"
        )

    if scales not in ("fixed", "free", "free_x", "free_y"):
        raise ValueError(
            f"scales must be one of 'fixed'/'free'/'free_x'/'free_y'; got {scales!r}"
        )

    return FacetGrid(rows=row_list, cols=col_list, scales=scales)


def _parse_side(s: str) -> list[str]:
    s = s.strip()
    if s in ("", "."):
        return []
    return [v.strip() for v in s.split("+") if v.strip()]


def _to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_side(value)
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]
