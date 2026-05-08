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


_DISCRETE_DTYPES = (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)


def _resolution(arr: np.ndarray) -> float:
    """Smallest non-zero gap between unique sorted values; mirrors
    ``ggplot2::resolution`` so the auto box width scales with the x
    spacing (categorical at integer positions → 1; bin midpoints at 0.1
    spacing → 0.1)."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 1.0
    u = np.unique(arr)
    if len(u) < 2:
        return 1.0
    return float(np.diff(u).min())


@dataclass
class StatBoxplot(Stat):
    coef: float = 1.5
    # ``None`` → auto: ``resolution(x) * 0.75`` per ggplot2's setup_params.
    # A float passes through as the literal box width (in x-axis units).
    width: float | None = None

    def compute_panel(self, data, params):
        if len(data) == 0:
            return pl.DataFrame()

        has_x = "x" in data.columns
        has_y = "y" in data.columns
        if not has_x and not has_y:
            return pl.DataFrame()

        # ggplot2's auto-orient: pick the *continuous* axis as the
        # distribution axis. The 5-number summary always runs on ``y``
        # internally, so when the distribution lives on x we rename
        # ``x``↔``y`` here and rename the output columns back below.
        #
        # Cases:
        #   * ``aes(x=...)`` alone           → flip (single-distribution; x at 0).
        #   * ``aes(y=...)`` alone           → no flip; pin x=0.
        #   * ``aes(x=cont, y=discrete)``    → flip (horizontal boxes).
        #   * ``aes(x=discrete, y=cont)``    → no flip (vertical boxes, default).
        #   * both continuous / both discrete → no flip (ggplot2's default).
        x_discrete = has_x and data["x"].dtype in _DISCRETE_DTYPES
        y_discrete = has_y and data["y"].dtype in _DISCRETE_DTYPES
        flipped = False
        if has_x and not has_y:
            flipped = True
            data = data.rename({"x": "y"}).with_columns(
                x=pl.lit(0.0).cast(pl.Float64),
            )
        elif has_y and not has_x:
            data = data.with_columns(x=pl.lit(0.0).cast(pl.Float64))
        elif y_discrete and not x_discrete:
            # Horizontal boxplot: y carries the categorical position,
            # x carries the distribution. Swap so the rest of compute_panel
            # (which assumes y == distribution) just works.
            flipped = True
            data = data.rename({"x": "_swap_y", "y": "_swap_x"}).rename(
                {"_swap_y": "y", "_swap_x": "x"}
            )

        x_is_discrete = data["x"].dtype in _DISCRETE_DTYPES

        # ggplot2 groups continuous-x boxplots by the layer's ``group``
        # aesthetic only — including ``x`` in groupby would over-split,
        # producing one box per unique x value instead of one per group
        # (the bug behind ``aes(group=cut_width(carat, 0.1))`` collapsing
        # to many tiny boxes regardless of bin width).
        groupby_cols: list[str] = []
        if x_is_discrete:
            groupby_cols.append("x")
        for aes in ("group", "fill", "colour"):
            if aes in data.columns and aes not in groupby_cols:
                groupby_cols.append(aes)

        # Capture source dtypes for the discrete grouping columns so
        # ``_row``'s ``pl.DataFrame(cols)`` doesn't downgrade an
        # ``Enum(['Fair','Good',...])`` x to plain Utf8 — which would
        # then sort alphabetically downstream and undo a deliberate
        # ``fct_reorder``. We cast back after concat below.
        preserve_dtypes = {col: data[col].dtype for col in groupby_cols
                           if col in data.columns}

        if not groupby_cols:
            row = self._row(data, keys=None, groupby_cols=(),
                            x_is_discrete=x_is_discrete)
            if row is None:
                return pl.DataFrame()
            out = row
        else:
            rows = []
            for keys, sub in data.group_by(groupby_cols, maintain_order=True):
                row = self._row(sub, keys=keys, groupby_cols=tuple(groupby_cols),
                                x_is_discrete=x_is_discrete)
                if row is not None:
                    rows.append(row)
            if not rows:
                return pl.DataFrame()
            out = pl.concat(rows)
            # Restore Enum / Categorical dtypes on the grouping columns —
            # ``pl.DataFrame(cols)`` in ``_row`` infers Utf8 from Python
            # strings, which would undo any user-supplied factor order
            # (``fct_reorder``, ``pl.Enum`` levels). Done here BEFORE the
            # flipped-rename below so the swap of ``x``↔``y`` carries the
            # restored dtype with it.
            for col, dtype in preserve_dtypes.items():
                if col in out.columns and out[col].dtype != dtype:
                    out = out.with_columns(
                        out[col].cast(dtype, strict=False).alias(col)
                    )

        # Auto width = ``resolution(box-centres) * 0.75``. ggplot2
        # computes resolution on the *raw* layer x, which for binned
        # continuous data (e.g. ``cut_width(carat, 0.1)``) returns the
        # underlying carat tick (~0.01) and produces line-thin boxes.
        # Using the box centres instead — i.e. the spacing between
        # adjacent boxes after groupby — gives boxes scaled to the bin
        # width, which is what users actually want.
        if self.width is not None:
            final_width = float(self.width)
        elif x_is_discrete:
            final_width = 0.75
        else:
            xs = out["x"].to_numpy().astype(float)
            final_width = _resolution(xs) * 0.75
        out = out.with_columns(
            width=pl.lit(final_width).cast(pl.Float64),
        )

        if flipped:
            # Mirror ggplot2's StatBoxplot: when the distribution axis is
            # x, the per-box stats become x-prefixed and ``y`` carries the
            # cross-axis position. Renaming (rather than passing a flag
            # through) lets the X scale auto-train on ``xmin``/``xmax``/…
            # via the existing ``_X_POSITIONAL_AES`` plumbing.
            out = out.rename({
                "x": "y",
                "ymin": "xmin",
                "lower": "xlower",
                "middle": "xmiddle",
                "upper": "xupper",
                "ymax": "xmax",
            })
            out = out.with_columns(flipped_aes=pl.lit(True))
        return out

    def _row(self, sub, *, keys, groupby_cols, x_is_discrete):
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

        # x position: discrete x → take the group key directly so the box
        # lines up with the categorical tick label; continuous x → mid of
        # the x range within the group (matches ggplot2's
        # ``mean(range(data$x))``).
        cols: dict = {}
        if x_is_discrete and "x" in groupby_cols:
            x_idx = groupby_cols.index("x")
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            cols["x"] = [keys_tuple[x_idx]]
        else:
            x_arr = sub["x"].to_numpy().astype(float)
            x_arr = x_arr[np.isfinite(x_arr)]
            if len(x_arr) == 0:
                return None
            cols["x"] = [float((x_arr.min() + x_arr.max()) / 2)]

        if keys is not None:
            keys_tuple = keys if isinstance(keys, tuple) else (keys,)
            for col, key_val in zip(groupby_cols, keys_tuple):
                if col == "x":
                    continue
                cols[col] = [key_val]
        cols.update({
            "ymin": [ymin],
            "lower": [float(q[1])],
            "middle": [float(q[2])],
            "upper": [float(q[3])],
            "ymax": [ymax],
        })
        df = pl.DataFrame(cols)
        # Polars list column needs explicit typing when the inner list is
        # numeric and might be empty.
        df = df.with_columns(
            outliers=pl.Series("outliers", [outliers.tolist()],
                               dtype=pl.List(pl.Float64)),
        )
        return df


def stat_boxplot(*, coef=1.5, width=None):
    return StatBoxplot(coef=coef, width=width)
