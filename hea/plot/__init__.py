"""Polars/matplotlib port of R's base-graphics plotting calls used in
Faraway's ``Linear Models with R``.

Spec
----
- ``plot()`` is a single dispatch entry that mirrors R's S3 ``plot.*``
  family. It returns a matplotlib ``Axes`` (or array of ``Axes`` for
  multi-panel diagnostic) so callers chain annotations explicitly via
  ``ax=``. Statefulness via ``plt.gca()`` is intentionally avoided.
- Multi-panel layouts: either pre-allocate with
  ``fig, axes = plt.subplots(...)`` and pass ``axes[i, j]`` to each
  call, or use the scoped ``with par(mfrow=(r, c)):`` context manager
  (R's ``par()`` idiom, but bounded by ``with`` so state can't leak).
- Categorical kwargs (``pch=``, ``col=``) accept polars Series/Enum
  directly; integer codes are derived internally via ``to_physical()``.
- Formula evaluation routes through ``hea.formula.parse``. Both LHS
  and RHS may be expressions (``residuals(m) ~ fitted(m)``,
  ``log(NOx) ~ E``, ``tail(r,n-1) ~ head(r,n-1)``). The evaluator pulls
  column names from ``data=`` and free variables (model objects, etc.)
  from the caller's frame plus a default math/model env.
- Math axis labels: pass LaTeX strings directly
  (``xlab=r"$\\hat{\\epsilon}_i$"``); an ``r_expr()`` translator for R's
  ``expression()`` mini-language ships in a later phase.

Dispatch table (Phase 1)
------------------------
``plot(formula_str, data=df)``  : route on RHS dtype
    num ~ num, with multi-RHS                          → scatter (one panel per RHS term)
    num ~ factor                                       → boxplot grouped by factor
    factor ~ num                                       → spineplot (TODO; deferred)
    factor ~ factor                                    → mosaic (TODO; deferred)
``plot(x, y)``                  : two numeric vectors  → scatter
``plot(vec)``                   : single vector        → vec vs index
``plot(lm_or_glm)``             : 4-panel diagnostic   → resid-fit / QQ / scale-loc / leverage

Phase 2+: annotations (``abline``/``points``/``lines``/``legend``/``segments``/
``qqline``), Faraway helpers (``qqnorm``/``halfnorm``/``termplot``), and the
long tail (``matplot``/``stripchart``/``interaction_plot``/``r_expr``).
"""

from .annotate import abline, legend, lines, points, qqline, rug, segments
from .barplot import barplot
from .boxplot import boxplot
from .curve import curve
from .density import density
from .dispatch import plot
from .helpers import halfnorm, interaction_plot, pairs, qqnorm, termplot
from .hist import hist
from .par import par
from .stubs import dev_off, gray, image, pdf, stripchart

__all__ = [
    "plot",
    "abline", "points", "lines", "legend", "segments", "qqline", "rug",
    "qqnorm", "halfnorm", "termplot", "pairs", "interaction_plot",
    "hist", "boxplot", "barplot", "density", "curve",
    "par",
    # No-op stubs for R base-graphics calls hea doesn't render yet.
    # Surfaced so translated scripts don't crash on the side-effect lines;
    # plotting calls between pdf()/dev.off() pairs still run.
    "pdf", "dev_off", "image", "stripchart", "gray",
]
