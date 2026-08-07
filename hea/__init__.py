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

# Everything below is resolved on first attribute access (PEP 562), not at
# import time. The reason is a number: eager, ``import hea`` costs ~690 ms and
# pulls 1476 modules including ``matplotlib.pyplot``, and a *submodule* import
# cannot dodge its parent's cost -- so ``import hea.sparse`` paid all of it for
# numpy and ``scipy.sparse``. Lazily it is ~112 ms, which is the
# ``numpy + scipy.sparse`` floor to within 2 ms, and ``hea._rs`` alone is 12 ms
# against a bare interpreter's 11.
#
# That is what makes ``hea.sparse`` usable as a dependency by a package that
# wants a sparse Cholesky and nothing else -- the whole reason it is written
# with no hea-internal imports. It also speeds up every other consumer: a
# script that only needs ``hea.models`` no longer pays for ``ggplot``.
#
# Nothing about the public surface changes. Names resolve on first touch and
# are cached in the module dict, so the second access is a plain global.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Eager for type checkers and IDEs only -- never executed at runtime, so
    # completion and go-to-definition keep working without the import cost.
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

#: hea sub-modules, reachable as ``hea.tidy`` / ``hea.models`` / … after a bare
#: ``import hea``. ``sparse`` is here too: it is the CHOLMOD port, and it is the
#: one module that must stay cheap enough to depend on alone.
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

#: Polars sub-namespaces re-exported as ``hea.selectors`` etc. — the only
#: polars-flavored access points exposed at the top level.
_POLARS = frozenset({"api", "exceptions", "plugins", "selectors"})

#: Names that live in a sub-module but are hit often enough to belong at the
#: top level: the three core data types, and the loaders/watermark that appear
#: in nearly every notebook (``data('iris')``, ``session_info()``).
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
    # cache it, so this runs once per name and never again
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
