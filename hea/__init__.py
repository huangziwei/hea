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
* ``hea.models``        — :func:`lm`, :func:`glm`, :func:`gam`, :func:`bam`, :func:`lme`
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
                          generics (``anova``, ``AIC``, ``step``, …), and
                          R utility functions (``factor``, ``cumsum``, …)
* ``hea.emmeans``       — port of CRAN ``emmeans``
* ``hea.translate``     — R ↔ Python source-to-source translator
* ``hea.ggplot``        — port of ``ggplot2``
* ``hea.plot``          — port of base-R ``plot``/``boxplot``/``hist``/…
* ``hea.session_info``  — R-style ``sessionInfo()`` watermark

Polars's own sub-namespaces are re-exported as ``hea.selectors``,
``hea.exceptions``, ``hea.api``, ``hea.plugins``.
"""

# Polars sub-namespaces — useful as ``hea.selectors`` etc., the only
# polars-flavored access points we expose at the top level.
from polars import api, exceptions, plugins, selectors  # noqa: F401

# The three core data type classes. Always the hea subclasses (which
# inherit from polars and carry the tidyverse verbs) — never the raw
# ``pl.DataFrame`` / ``pl.LazyFrame`` / ``pl.Series``. Top-level so
# ``isinstance(x, hea.DataFrame)`` and ``hea.DataFrame({...})`` work
# without the ``hea.tidy.`` prefix.
from .tidy import DataFrame, LazyFrame, Series

# hea sub-modules — imported so ``hea.tidy`` / ``hea.models`` / … are
# attribute-accessible without ``import hea.X`` separately.
from . import (  # noqa: F401
    R,
    dtypes,
    emmeans,
    family,
    ggplot,
    io,
    models,
    plot,
    session_info,
    tidy,
    translate,
)

# ``data`` is exposed at the top level — used in nearly every example
# (``data('iris')``, ``data('penguins', package='palmerpenguins')``).
# This shadows the ``hea.data`` submodule attribute; the submodule is
# still importable via ``import hea.data`` / ``from hea.data import …``.
from .data import data  # noqa: F401
