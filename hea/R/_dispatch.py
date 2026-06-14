"""Native-extension dispatch with graceful pure-Python fallback.

The compiled ``hea._native`` (Rust/PyO3, built from ``rust/``) accelerates the
nmath d/p/q kernels. It is **optional**: on sdist / no-toolchain installs, on an
unsupported platform, or when ``HEA_NO_NATIVE`` is set to a truthy value, the
pure-Python :mod:`hea.R.nmath` kernels run instead. Those kernels are the
bit-exact reference to R (tests/test_R.py) *and* the oracle the native path is
checked against (tests/test_native_parity.py) — so the fallback is never "wrong",
only slower. See plans/rust-port-implementation.md §1.3.

Usage in a kernel module::

    from ._dispatch import native_fn
    _nat = native_fn("pgamma")           # None when unavailable/disabled
    ...
    if _nat is not None:
        return _nat(flat, ...).reshape(shape)
    # else: pure-Python path
"""
from __future__ import annotations

import os

__all__ = ["HAVE_NATIVE", "native", "native_fn"]


def _load():
    # HEA_NO_NATIVE truthy → force the Python path (A/B parity, debugging).
    if os.environ.get("HEA_NO_NATIVE", "").lower() not in ("", "0", "false", "no"):
        return None
    try:
        from .. import _native  # hea._native — the compiled extension
    except Exception:
        return None
    return _native


_NATIVE = _load()
HAVE_NATIVE = _NATIVE is not None


def native():
    """The compiled extension module, or ``None`` if unavailable/disabled."""
    return _NATIVE


def native_fn(name: str):
    """A native kernel by ``name``, or ``None`` to signal "use the Python path"."""
    return getattr(_NATIVE, name, None) if _NATIVE is not None else None
