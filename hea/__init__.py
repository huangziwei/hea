"""hea — R/tidyverse-shaped statistical modeling on polars.

Top-level holds the three core data type classes plus :func:`data`,
the dataset loader — everything else lives in a sub-namespace.
``hea.DataFrame`` / ``hea.LazyFrame`` / ``hea.Series`` are *always* the
hea subclasses (they inherit from polars under the hood, but carry the
tidyverse verbs), so users never have to type ``pl.DataFrame`` to
construct a frame or run an ``isinstance`` check.

* ``hea.DataFrame`` / ``hea.LazyFrame`` / ``hea.Series`` — types you
                          write in ``isinstance`` checks, annotations,
                          and constructor calls
* ``hea.data(...)``     — R's :func:`data` (dataset loader): hit in
                          almost every example to pull a frame from
                          rdatasets/faraway/lme4/etc.
* ``hea.models``        — :func:`lm`, :func:`glm`, :func:`gam`, :func:`bam`, :func:`gmm`
* ``hea.tidy``          — tidyverse verbs (``desc``, ``case_when``,
                          ``fct_*``, …) plus the polars expression
                          builders (``col``, ``lit``, ``when``, …) used
                          inside a pipeline
* ``hea.dtypes``        — polars datatype names (``Int64``, ``String``, …)
* ``hea.io``            — readers / scanners / DataFrame factories
                          (``read_csv``, ``concat``, ``from_dict``, …)
* ``hea.family``        — GLM/GAM/LME exponential-family + link primitives
* ``hea.R``             — base-R muscle memory: hypothesis tests
                          (``t_test``, ``chisq_test``, …), model-comparison
                          generics (``anova``, ``AIC``, ``step``, …),
                          R utility functions (``factor``, ``cumsum``, …),
                          plus the small CRAN ``emmeans`` port
* ``hea.translate``     — R ↔ Python source-to-source translator
* ``hea.ggplot``        — port of ``ggplot2``
* ``hea.plot``          — port of base-R ``plot``/``boxplot``/``hist``/…
* ``hea.session_info``  — R-style ``sessionInfo()`` watermark

Polars's own sub-namespaces are re-exported as ``hea.selectors``,
``hea.exceptions``, ``hea.api``, ``hea.plugins``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars import api, exceptions, plugins, selectors

    from . import (
        R,
        dtypes,
        family,
        ggplot,
        io,
        models,
        plot,
        sparse,
        tidy,
        translate,
    )
    from .io import data, map_data
    from .session_info import session_info
    from .tidy import DataFrame, LazyFrame, Series

_SUBMODULES = frozenset(
    {
        "R",
        "dtypes",
        "family",
        "ggplot",
        "io",
        "models",
        "plot",
        "sparse",
        "tidy",
        "translate",
    }
)

_POLARS = frozenset({"api", "exceptions", "plugins", "selectors"})

_ATTRS = {
    "DataFrame": ".tidy",
    "LazyFrame": ".tidy",
    "Series": ".tidy",
    "data": ".io",
    "map_data": ".io",
    "session_info": ".session_info",
}

__all__ = sorted(_SUBMODULES | _POLARS | set(_ATTRS))  # noqa: PLE0605 - derived from the module registries, not a literal


def __getattr__(name: str):
    """Resolve a top-level name on first access — PEP 562."""
    from importlib import import_module

    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
    elif name in _POLARS:
        value = getattr(import_module("polars"), name)
    elif name in _ATTRS:
        value = getattr(import_module(_ATTRS[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
