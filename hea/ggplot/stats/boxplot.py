"""``stat_boxplot()`` — five-number summary + outlier detection per group.

Algorithm (mirrors ggplot2's ``StatBoxplot.compute_group``):

1. Compute quartiles ``Q0..Q4`` of y.
2. ``IQR = Q3 - Q1``; outliers are points beyond ``Q1 - coef·IQR`` or
   ``Q3 + coef·IQR`` (default ``coef = 1.5``).
3. Whiskers extend to the most extreme non-outlier point on each side.
4. Output one row per group with ``ymin`` (low whisker), ``lower`` (Q1),
   ``middle`` (median), ``upper`` (Q3), ``ymax`` (high whisker), plus a
   list-typed ``outliers`` column and ``width``.

Notches (``notchupper``, ``notchlower``) are deferred — uncommon in
practice and add ~30 lines of further bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .stat import Stat


@dataclass
class StatBoxplot(Stat):
    coef: float = 1.5
    width: float = 0.75

    def compute_panel(self, data, params):
        groupby_cols = ["x"]
        for aes in ("group", "fill", "colour"):
            if aes in data.columns and aes not in groupby_cols:
                groupby_cols.append(aes)

        rows = []
        for keys, sub in data.group_by(groupby_cols, maintain_order=True):
            row = self._row(sub, keys, groupby_cols)
            if row is not None:
                rows.append(row)
        if not rows:
            return pl.DataFrame()
        return pl.concat(rows)

    def _row(self, sub, keys, groupby_cols):
        y = sub["y"].to_numpy().astype(float)
        y = y[~np.isnan(y)]
        if len(y) == 0:
            return None

        q = np.quantile(y, [0.0, 0.25, 0.5, 0.75, 1.0])
        iqr = q[3] - q[1]
        lo_bound = q[1] - self.coef * iqr
        hi_bound = q[3] + self.coef * iqr

        outliers = y[(y < lo_bound) | (y > hi_bound)]
        non_out = y[(y >= lo_bound) & (y <= hi_bound)]
        ymin = float(non_out.min()) if len(non_out) else float(q[0])
        ymax = float(non_out.max()) if len(non_out) else float(q[4])

        cols: dict = {col: [keys[i]] for i, col in enumerate(groupby_cols)}
        cols.update({
            "ymin": [ymin],
            "lower": [float(q[1])],
            "middle": [float(q[2])],
            "upper": [float(q[3])],
            "ymax": [ymax],
            "width": [self.width],
        })
        df = pl.DataFrame(cols)
        # Polars list column needs explicit typing when the inner list is
        # numeric and might be empty.
        df = df.with_columns(
            outliers=pl.Series("outliers", [outliers.tolist()],
                               dtype=pl.List(pl.Float64)),
        )
        return df


def stat_boxplot(*, coef=1.5, width=0.75):
    return StatBoxplot(coef=coef, width=width)
