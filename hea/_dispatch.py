"""Rust-extension dispatch, and the switch to the pure-Python copies.

The compiled ``hea._rs`` (Rust/PyO3, crate ``hea-rs`` in ``rust/``) runs hea's
numeric kernels — nmath d/p/q, the RNG, clustering/distance, linalg, the tp
smooth basis, mgcv's C routines — and carries the ports of R's reference
LAPACK routines. maturin is the build backend, so every install builds it and
it is the route hea runs. The pure-Python kernels are reference copies that
spell out what the extension computes; they run only when ``HEA_NO_RS`` is
set to a truthy value or the module is missing from a source checkout. Each
kernel's twin computes the same expressions, its BLAS calls going to the
routines the extension calls (the tests run the pairs side by side); the
LAPACK ports have none, and without the extension those calls go to SciPy's
LAPACK, so results that depend on them are not R's to the last bit.

This lives at the top level (not under ``hea/R/``) because the Rust extension is
package-wide: consumers include ``hea.R.*`` (nmath/rng/clustering/distance/
linalg) AND ``hea.formula`` / ``hea.family`` / ``hea.ggplot``.

Usage in a kernel module::

    from hea._dispatch import rs_fn
    _rs = rs_fn("pgamma")                 # None when unavailable/disabled
    ...
    if _rs is not None:
        return _rs(flat, ...).reshape(shape)
"""

from __future__ import annotations

import os
import sys

__all__ = ["ACCELERATE", "HAVE_RS", "rs", "rs_fn"]


def _load():
    if os.environ.get("HEA_NO_RS", "").lower() not in ("", "0", "false", "no"):
        return None
    try:
        from . import _rs  # hea._rs — the compiled Rust extension
    except Exception:  # noqa: BLE001
        return None
    return _rs


_RS = _load()
HAVE_RS = _RS is not None


def rs():
    """The compiled Rust extension module, or ``None`` if unavailable/disabled."""
    return _RS


def rs_fn(name: str):
    """A Rust kernel by ``name``, or ``None`` to signal "use the Python path"."""
    return getattr(_RS, name, None) if _RS is not None else None


def _accelerate() -> bool:
    info = rs_fn("build_info")
    if info is not None:
        return info()["backend"] == "accelerate"
    return sys.platform == "darwin"


# Whether the BLAS the ports call is Apple's Accelerate: as the extension was
# built when it is present; without it, on darwin, where R's BLAS is
# Accelerate (vecLib) and a default build of the extension links it. The
# Python copies call the routines the extension would.
ACCELERATE = _accelerate()
