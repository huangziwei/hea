"""ggplot extension function detection.

Used by ``r_to_py`` to decide whether ``ggplot(...) + foo()`` is a chain
extension (translate to method call) or just numeric / patchwork
composition (preserve as ``+``).

We detect chain extensions by **name shape**:

- Prefix ``geom_`` / ``scale_`` / ``coord_`` / ``facet_`` / ``theme_`` /
  ``stat_`` / ``position_`` / ``element_`` — these are open sets
  (ggplot2 + hea both keep adding). Prefix match is more robust than
  enumeration.
- Explicit set for the handful of non-prefixed extensions: ``labs``,
  ``xlab``, ``ylab``, ``ggtitle``, ``xlim``, ``ylim``, ``lims``,
  ``guides``, ``annotate``, ``coord_*`` (prefix), patchwork helpers.

This module is **purely declarative** — no AST handling lives here.
"""

from __future__ import annotations

_NON_PREFIXED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "labs",
        "xlab",
        "ylab",
        "ggtitle",
        "xlim",
        "ylim",
        "lims",
        "guides",
        "annotate",
        "annotation_custom",
        "plot_annotation",
        "plot_layout",
    }
)

_EXTENSION_PREFIXES: tuple[str, ...] = (
    "geom_",
    "scale_",
    "coord_",
    "facet_",
    "theme_",
    "stat_",
    "position_",
)


def is_chain_extension(func_name: str) -> bool:
    """``True`` iff ``ggplot(...) + func_name(...)`` is a chain extension.

    Used to decide whether to rewrite ``+`` as a method chain. Any
    ``+`` whose RHS isn't a known extension stays as a regular ``+``
    (arithmetic or patchwork composition).
    """
    if func_name in _NON_PREFIXED_EXTENSIONS:
        return True
    return any(func_name.startswith(p) for p in _EXTENSION_PREFIXES)


def is_theme_call(func_name: str) -> bool:
    return func_name == "theme"


def is_aes_call(func_name: str) -> bool:
    return func_name == "aes"
