"""Shared dataset registry — used by both translation directions.

The forward direction (R → hea) consults this to emit
``X = hea.data("X", package="pkg")`` preamble statements for bare-name
dataset references. The reverse direction (hea → R) consults this to
emit ``library(pkg)`` preamble calls.

Pulled into its own module so both directions can import without a
circular dependency.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def dataset_registry() -> dict[str, tuple[str, ...]]:
    """``{dataset_name: (pkg1, pkg2, ...)}`` from rdatasets.

    Cached for the life of the process. First call scans ~75 packages ×
    ~30 items each (≈600ms); subsequent calls are instant. Returns an
    empty dict if rdatasets isn't installed — translator still works,
    just without autoload inference.
    """
    try:
        import rdatasets
    except ImportError:
        return {}
    registry: dict[str, list[str]] = {}
    for pkg in rdatasets.packages():
        for item in rdatasets.items(pkg):
            name = item.removesuffix(".pkl")
            registry.setdefault(name, []).append(pkg)
    return {n: tuple(pkgs) for n, pkgs in registry.items()}


# Names that should never trigger an autoload lookup even if they match
# a rdatasets entry. Mix of Python builtins, hea / polars surface, and
# common short variable names that would cause noisy false positives.
DATASET_REF_EXCLUSIONS: frozenset[str] = frozenset({
    # Python builtins
    "True", "False", "None",
    "print", "len", "range", "list", "dict", "set", "tuple", "int",
    "float", "str", "bool", "type", "object",
    # hea / polars surface
    "hea", "pl", "col", "n", "desc", "selectors",
    "DataFrame", "LazyFrame", "Series", "Expr",
    # Common R-side names that map back via the FUNCTION_TABLE
    "case_when", "if_else", "coalesce", "data", "first", "last",
    # Single-letter / very-short names — common variable names that
    # happen to collide with datasets (e.g. ``x`` is a Sloan Digital
    # Sky Survey dataset; we don't want that triggering).
    "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "p", "q", "p1", "p2", "p3", "p4",
    "df", "dat", "obj", "out", "tmp", "res", "ans",
})


# R's default-loaded packages. ``library()`` for these is redundant; we
# omit them from autoload preambles even when a dataset resolves there.
R_DEFAULT_PACKAGES: frozenset[str] = frozenset({
    "base", "datasets", "graphics", "grDevices",
    "methods", "stats", "utils",
})


def resolve_dataset(
    name: str,
    *,
    loaded_packages: frozenset[str] = frozenset(),
) -> str | None:
    """Pick the right package for a bare-name dataset reference.

    Rules, in order:

    1. Excluded name → ``None``.
    2. Not in registry → ``None``.
    3. Unique in registry → the one package (skipping R defaults).
    4. Ambiguous in registry → return one of ``loaded_packages`` if it
       intersects; otherwise ``None`` (don't guess).

    ``loaded_packages`` is the set the user already declared via
    ``library(...)`` (reverse direction never has this; forward
    direction populates it).
    """
    if name in DATASET_REF_EXCLUSIONS:
        return None
    pkgs = dataset_registry().get(name)
    if not pkgs:
        return None
    if len(pkgs) == 1:
        pkg = pkgs[0]
        if pkg in R_DEFAULT_PACKAGES:
            return None
        return pkg
    # Ambiguous: prefer a package the user explicitly loaded.
    for p in pkgs:
        if p in loaded_packages and p not in R_DEFAULT_PACKAGES:
            return p
    return None
