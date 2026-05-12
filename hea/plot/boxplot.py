"""Boxplot renderer — covers formula ``num ~ factor`` and the standalone
``boxplot(x)`` / ``boxplot([y1, y2, ...])`` forms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def boxplot(
    *args,
    names=None,
    horizontal: bool = False,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    ax=None,
):
    """Single- or multi-vector boxplot. Mirrors R's ``graphics::boxplot``.

    Accepts:

    * ``boxplot(x)`` — one vector → one box.
    * ``boxplot(x, y, z)`` — variadic vectors → side-by-side boxes.
    * ``boxplot([x, y, z], names=("a","b","c"))`` — a list of vectors.

    The formula form ``boxplot(y ~ g, data=df)`` is reached through
    :func:`hea.plot.plot`, which dispatches a ``num ~ factor`` RHS to
    :func:`boxplot_by`. This function is for the non-formula calls.
    """
    if not args:
        raise TypeError("boxplot(): pass at least one vector.")

    # ``boxplot([x, y, z])`` — flatten a single list-of-vectors arg.
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and args[0] \
            and not isinstance(args[0][0], (int, float, np.integer, np.floating)):
        groups = list(args[0])
    else:
        groups = list(args)

    arrays: list[np.ndarray] = []
    inferred_names: list[str] = []
    for i, g in enumerate(groups):
        if isinstance(g, pl.Series):
            arr = g.cast(pl.Float64).drop_nulls().to_numpy()
            inferred_names.append(g.name or str(i + 1))
        else:
            arr = np.asarray(g, dtype=float)
            arr = arr[~np.isnan(arr)]
            inferred_names.append(str(i + 1))
        arrays.append(arr)

    if names is None:
        names = inferred_names
    else:
        names = [str(n) for n in names]
        if len(names) != len(arrays):
            raise ValueError(
                f"boxplot(): names has {len(names)} entries but got "
                f"{len(arrays)} groups."
            )

    if ax is None:
        _, ax = plt.subplots()

    if horizontal:
        ax.boxplot(arrays, tick_labels=names, orientation="horizontal")
    else:
        ax.boxplot(arrays, tick_labels=names)

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)
    return ax


def boxplot_by(
    y,
    group,
    *,
    ax=None,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
):
    """Vertical boxplots of ``y`` grouped by ``group`` (a polars Enum/Categorical
    or any string-like Series). Level order is taken from ``group`` if it's an
    Enum (R-faithful), otherwise from sorted unique values."""
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(group, pl.Series):
        if group.dtype == pl.Enum:
            levels = group.cat.get_categories().to_list()
        elif group.dtype == pl.Categorical:
            levels = group.cat.get_categories().to_list()
        else:
            levels = sorted(group.drop_nulls().unique().to_list())
        g_arr = group.to_numpy()
    else:
        g_arr = np.asarray(group)
        levels = sorted(set(x for x in g_arr.tolist() if x is not None))

    if isinstance(y, pl.Series):
        y_arr = y.cast(pl.Float64).to_numpy()
    else:
        y_arr = np.asarray(y, dtype=float)

    groups = [y_arr[g_arr == lvl] for lvl in levels]

    ax.boxplot(groups, tick_labels=[str(lv) for lv in levels])

    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)
    if main is not None:
        ax.set_title(main)
    return ax
