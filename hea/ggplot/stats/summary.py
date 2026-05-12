"""``stat_summary()`` — group y by x and emit summary rows ``(y, ymin, ymax)``.

Four built-in summary helpers (looked up by name as ``fun_data="..."``):

* ``mean_se`` — sample mean ± ``mult`` × SE (default ``mult=1``).
* ``mean_cl_normal`` — mean ± t × SE / √n at confidence ``conf`` (default 0.95).
* ``mean_cl_boot`` — mean with bootstrap CI (B resamples, default 1000).
* ``median_hilow`` — median + symmetric quantile envelope at ``conf``.

Pass a callable for custom: ``f(y) → {"y": ..., "ymin": ..., "ymax": ...}``.

Default geom is ``"pointrange"``, mirroring ggplot2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..aes import split_layer_kwargs
from .stat import Stat


# ---------------------------------------------------------------------------
# Built-in summary helpers
# ---------------------------------------------------------------------------

def mean_se(x: np.ndarray, mult: float = 1.0) -> dict:
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return {"y": m, "ymin": m, "ymax": m}
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return {"y": m, "ymin": m - mult * se, "ymax": m + mult * se}


def mean_cl_normal(x: np.ndarray, conf: float = 0.95) -> dict:
    from scipy import stats as scstats

    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return {"y": m, "ymin": m, "ymax": m}
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    t = float(scstats.t.ppf(0.5 + conf / 2, n - 1))
    return {"y": m, "ymin": m - t * se, "ymax": m + t * se}


def mean_cl_boot(x: np.ndarray, conf: float = 0.95, B: int = 1000,
                 seed: int | None = None) -> dict:
    """Bootstrap CI for the mean. ``seed`` makes results reproducible (pass
    via ``fun_args={"seed": 42}``); without it, each run varies."""
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return {"y": m, "ymin": m, "ymax": m}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    lower = float(np.quantile(means, 0.5 - conf / 2))
    upper = float(np.quantile(means, 0.5 + conf / 2))
    return {"y": m, "ymin": lower, "ymax": upper}


def median_hilow(x: np.ndarray, conf: float = 0.95) -> dict:
    return {
        "y": float(np.median(x)),
        "ymin": float(np.quantile(x, 0.5 - conf / 2)),
        "ymax": float(np.quantile(x, 0.5 + conf / 2)),
    }


_NAMED_SUMMARIES = {
    "mean_se": mean_se,
    "mean_cl_normal": mean_cl_normal,
    "mean_cl_boot": mean_cl_boot,
    "median_hilow": median_hilow,
}


_NAMED_AGG = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
}


def _resolve_fun(f, *, default=None):
    if f is None:
        return default
    if callable(f):
        return f
    if isinstance(f, str):
        if f in _NAMED_AGG:
            return _NAMED_AGG[f]
        raise ValueError(
            f"unknown aggregation function name {f!r}; "
            f"valid: {sorted(_NAMED_AGG)}"
        )
    raise TypeError(f"expected callable or string, got {type(f).__name__}")


# ---------------------------------------------------------------------------
# StatSummary
# ---------------------------------------------------------------------------

_DISCRETE_DTYPES = (pl.Enum, pl.Categorical, pl.Utf8)


@dataclass
class StatSummary(Stat):
    fun_data: Callable | str | None = None
    fun: Callable | str | None = None
    fun_min: Callable | str | None = None
    fun_max: Callable | str | None = None
    fun_args: dict = field(default_factory=dict)
    # ``"x"`` (default ggplot2 behavior — group by x, summarize y),
    # ``"y"`` (transposed: group by y, summarize x), or ``"auto"`` — pick
    # ``"y"`` when y is discrete and x isn't, else ``"x"``.
    orientation: str = "auto"

    def _summary_fn(self) -> Callable[[np.ndarray], dict]:
        if self.fun_data is not None:
            if callable(self.fun_data):
                return self.fun_data
            if isinstance(self.fun_data, str):
                if self.fun_data in _NAMED_SUMMARIES:
                    return _NAMED_SUMMARIES[self.fun_data]
                raise ValueError(
                    f"unknown summary {self.fun_data!r}; "
                    f"valid: {sorted(_NAMED_SUMMARIES)}"
                )
            raise TypeError(
                f"fun_data must be callable or string, got "
                f"{type(self.fun_data).__name__}"
            )
        if self.fun is not None or self.fun_min is not None or self.fun_max is not None:
            f = _resolve_fun(self.fun, default=np.mean)
            fmin = _resolve_fun(self.fun_min, default=np.min)
            fmax = _resolve_fun(self.fun_max, default=np.max)

            def _componentwise(y):
                return {
                    "y": float(f(y)),
                    "ymin": float(fmin(y)),
                    "ymax": float(fmax(y)),
                }
            return _componentwise
        # ggplot2's default.
        return mean_se

    def _resolve_orientation(self, data) -> str:
        """Pick ``"x"`` or ``"y"`` for grouping. ``"auto"`` flips to ``"y"``
        when y is discrete (enum / categorical / utf8) and x isn't —
        matches ggplot2's auto-orientation."""
        if self.orientation in ("x", "y"):
            return self.orientation
        if self.orientation != "auto":
            raise ValueError(
                f"orientation= must be 'x', 'y', or 'auto'; got {self.orientation!r}."
            )
        y_disc = data.schema["y"] in _DISCRETE_DTYPES
        x_disc = data.schema["x"] in _DISCRETE_DTYPES
        return "y" if (y_disc and not x_disc) else "x"

    def compute_group(self, data, params):
        if "x" not in data.columns or "y" not in data.columns or len(data) == 0:
            return data

        orient = self._resolve_orientation(data)
        # ``orient`` names the axis we summarize *along*: orient="x" means
        # "x is the discrete axis; group by x and reduce y" (ggplot2's
        # historic default). orient="y" transposes — reduce x within each
        # y group and emit (y, x, xmin, xmax).
        if orient == "y":
            group_col, summary_col = "y", "x"
            out_centre, out_min, out_max = "x", "xmin", "xmax"
        else:
            group_col, summary_col = "x", "y"
            out_centre, out_min, out_max = "y", "ymin", "ymax"

        fn = self._summary_fn()
        rows: list[dict] = []
        for keys, sub in data.group_by(group_col, maintain_order=True):
            grp_val = keys[0] if isinstance(keys, tuple) else keys
            arr = sub[summary_col].drop_nulls().to_numpy()
            # Strip NaNs (drop_nulls only catches polars-null, not float-NaN).
            if arr.dtype.kind == "f":
                arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            summary = fn(arr, **self.fun_args)
            rows.append({
                group_col:  grp_val,
                out_centre: summary["y"],
                out_min:    summary["ymin"],
                out_max:    summary["ymax"],
            })

        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).sort(group_col)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_summary_geom(geom):
    """Map a ggplot2-style geom name to an instantiated :class:`Geom`.

    ggplot2's :func:`stat_summary` resolves ``geom=`` through the global
    geom registry, so any geom that consumes ``(x, y[, ymin, ymax])``
    works. hea has no central registry, so we enumerate the geoms whose
    required aesthetics are satisfied by ``StatSummary``'s output:

    ``y``-only geoms — ``"point"``, ``"line"``, ``"path"``, ``"step"``;
    range geoms (``ymin``/``ymax``) — ``"pointrange"``, ``"errorbar"``,
    ``"linerange"``, ``"crossbar"``, ``"ribbon"``, ``"area"``, ``"smooth"``;
    plus ``"bar"`` (uses ``y`` as the column height).

    Pass an already-built ``Geom`` instance to bypass the lookup.
    """
    from ..geoms.bar import GeomBar
    from ..geoms.errorbar import (
        GeomCrossbar, GeomErrorbar, GeomLinerange, GeomPointrange,
    )
    from ..geoms.path import GeomPath
    from ..geoms.point import GeomPoint
    from ..geoms.ribbon import GeomArea, GeomRibbon
    from ..geoms.smooth import GeomSmooth

    # Factory dict — values are zero-arg callables so we can instantiate
    # with constructor params where the geom needs them
    # (``geom_line`` = ``GeomPath(sort_by_x=True)``, etc.).
    geom_map = {
        "pointrange": GeomPointrange,
        "errorbar":   GeomErrorbar,
        "linerange":  GeomLinerange,
        "crossbar":   GeomCrossbar,
        "bar":        GeomBar,
        "point":      GeomPoint,
        "path":       GeomPath,
        "line":       lambda: GeomPath(sort_by_x=True),
        "step":       lambda: GeomPath(sort_by_x=True, step_direction="hv"),
        "ribbon":     GeomRibbon,
        "area":       GeomArea,
        "smooth":     GeomSmooth,
    }
    if isinstance(geom, str):
        if geom not in geom_map:
            raise ValueError(
                f"stat_summary: unknown geom {geom!r}; "
                f"valid: {sorted(geom_map)}"
            )
        return geom_map[geom]()
    if hasattr(geom, "draw_panel"):
        return geom
    raise TypeError(
        f"stat_summary: geom must be a Geom or string, got {type(geom).__name__}"
    )


def stat_summary(mapping=None, data=None, *, geom="pointrange",
                 fun_data=None, fun=None, fun_min=None, fun_max=None,
                 fun_args=None, orientation="auto", position="identity",
                 **kwargs):
    """Per-x summary of y. ``fun_data`` (preferred) returns ``{y, ymin, ymax}``;
    name a built-in (``"mean_se"``, ``"mean_cl_normal"``, ``"mean_cl_boot"``,
    ``"median_hilow"``) or pass a callable. Componentwise alternative:
    ``fun=`` for y, ``fun_min``/``fun_max`` for the range — strings name a
    numpy aggregation (``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"sum"``)
    or callables work too.

    ``geom`` defaults to ``"pointrange"`` (ggplot2 convention). Any of the
    following names resolve to the corresponding geom (matching the set
    accepted by ggplot2's :func:`stat_summary`):

    * ``y``-only consumers — ``"point"``, ``"line"``, ``"path"``, ``"step"``;
    * range consumers (need ``ymin``/``ymax``) — ``"pointrange"``,
      ``"errorbar"``, ``"linerange"``, ``"crossbar"``, ``"ribbon"``,
      ``"area"``, ``"smooth"``;
    * ``"bar"`` (uses ``y`` as the bar height).

    A ``Geom`` instance is also accepted directly.
    """
    from ..layer import Layer
    from ..positions import resolve_position

    geom_obj = _resolve_summary_geom(geom)

    aes_params, geom_params = split_layer_kwargs(kwargs)

    return Layer(
        geom=geom_obj,
        stat=StatSummary(
            fun_data=fun_data, fun=fun, fun_min=fun_min, fun_max=fun_max,
            fun_args=fun_args or {},
            orientation=orientation,
        ),
        position=resolve_position(position),
        mapping=mapping,
        data=data,
        aes_params=aes_params,
        geom_params=geom_params,
    )
