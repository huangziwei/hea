"""Rust-extension dispatch with graceful pure-Python fallback.

The compiled ``hea._rs`` (Rust/PyO3, crate ``hea-rs`` in ``rust/``) accelerates
hea's hot numeric kernels — nmath d/p/q, the RNG, clustering/distance, linalg,
and the tp smooth basis. It is **optional**: on sdist / no-toolchain installs,
on an unsupported platform, or when ``HEA_NO_RS`` is set to a truthy value, the
pure-Python kernels run instead. Those kernels are the bit-exact reference (and
the oracle the Rust path is checked against, tests/test_rs_parity.py) — so the
fallback is never "wrong", only slower.

This lives at the top level (not under ``hea/R/``) because the Rust extension is
package-wide: consumers include ``hea.R.*`` (nmath/rng/clustering/distance/
linalg) AND ``hea.formula`` / ``hea.family`` / ``hea.ggplot``.

Usage in a kernel module::

    from hea._dispatch import rs_fn
    _rs = rs_fn("pgamma")                 # None when unavailable/disabled
    ...
    if _rs is not None:
        return _rs(flat, ...).reshape(shape)
    # else: pure-Python path
"""

from __future__ import annotations

import os

__all__ = ["HAVE_RS", "rs", "rs_fn"]


def _load():
    # HEA_NO_RS truthy → force the Python path (A/B parity, debugging).
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
