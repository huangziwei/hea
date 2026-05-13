"""R: ``sessionInfo()`` — Python+platform+package snapshot.

A self-printing object that mirrors R's ``sessionInfo()`` output. Used:

- Bare from a hea notebook (``hea.session_info()``) as a reproducibility
  watermark, in the spirit of jupyter-watermark but R-shaped.
- From translated R scripts that close with ``sessionInfo()``; the
  translator routes the R name to :func:`session_info` via the function
  registry.

Layout matches R's section order: Python/platform header, BLAS/LAPACK
(extracted from numpy's build config), locale + time zone, then two
package lists — a curated "attached" set (the core data-science stack
when present in ``sys.modules``) and the rest of ``sys.modules`` that
exposes a discoverable version, formatted in R-style indexed wrap.
"""

from __future__ import annotations

import datetime
import locale
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata


# Packages worth showing in the "attached" section when loaded. These
# are the libraries a hea user is likely to be using directly — the
# rest are deps that landed via transitive imports.
_CORE_PACKAGES: tuple[str, ...] = (
    "hea",
    "polars",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "pyarrow",
    "sklearn",
    "statsmodels",
    "torch",
    "tensorflow",
    "IPython",
    "jupyter",
    "pytest",
)


def _pkg_version(name: str) -> str | None:
    """Best-effort version lookup: metadata first, then ``__version__``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    mod = sys.modules.get(name)
    if mod is not None:
        v = getattr(mod, "__version__", None)
        if isinstance(v, str):
            return v
    return None


@dataclass
class SessionInfo:
    """R: ``sessionInfo`` return value.

    Captured fields are usable programmatically; ``__repr__`` formats
    them in R's section layout so notebooks / REPLs print cleanly.
    """

    python_version: str
    python_impl: str
    platform: str
    os_descriptive: str
    blas: str
    lapack: str
    locale: str
    timezone: str
    hea_version: str
    attached: dict[str, str] = field(default_factory=dict)
    loaded: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines: list[str] = [
            f"Python version {self.python_version} ({self.python_impl})",
            f"Platform: {self.platform}",
            f"Running under: {self.os_descriptive}",
            "",
            f"BLAS:   {self.blas}",
            f"LAPACK: {self.lapack}",
            "",
            f"locale: {self.locale}",
            "",
            f"time zone: {self.timezone}",
            "",
            f"hea version {self.hea_version}",
            "",
        ]
        if self.attached:
            lines.append("attached packages:")
            lines.extend(_fmt_pkg_list(self.attached))
            lines.append("")
        if self.loaded:
            lines.append("loaded via a namespace (and not attached):")
            lines.extend(_fmt_pkg_list(self.loaded))
        return "\n".join(lines).rstrip()


def session_info() -> SessionInfo:
    """R: ``sessionInfo()`` — return a printable session snapshot."""
    py_version = platform.python_version()
    py_impl = platform.python_implementation()
    uname = platform.uname()
    plat = f"{uname.machine}-{uname.system.lower()}{uname.release}"
    os_descriptive = _os_descriptive(uname)

    blas, lapack = _numpy_blas_lapack()

    try:
        parts = [p for p in locale.getlocale() if p]
        loc = ".".join(parts) if parts else "C"
    except Exception:
        loc = "C"

    tz = datetime.datetime.now().astimezone().tzname() or "unknown"

    hea_version = _pkg_version("hea") or "dev"

    attached: dict[str, str] = {}
    for name in _CORE_PACKAGES:
        v = _pkg_version(name)
        if v is not None:
            attached[name] = v

    loaded: dict[str, str] = {}
    seen = set(attached)
    for fullname in list(sys.modules):
        if "." in fullname or fullname.startswith("_"):
            continue
        if fullname in seen:
            continue
        v = _pkg_version(fullname)
        if v is not None:
            loaded[fullname] = v
            seen.add(fullname)

    return SessionInfo(
        python_version=py_version,
        python_impl=py_impl,
        platform=plat,
        os_descriptive=os_descriptive,
        blas=blas,
        lapack=lapack,
        locale=loc,
        timezone=tz,
        hea_version=hea_version,
        attached=attached,
        loaded=loaded,
    )


# ---- internals ------------------------------------------------------


def _os_descriptive(uname: platform.uname_result) -> str:
    """Friendly OS line — matches R's ``Running under:`` style."""
    sys_name = uname.system
    if sys_name == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else f"Darwin {uname.release}"
    if sys_name == "Linux":
        return f"Linux {uname.release}"
    if sys_name == "Windows":
        return f"Windows {uname.release}"
    return f"{sys_name} {uname.release}"


def _numpy_blas_lapack() -> tuple[str, str]:
    """BLAS/LAPACK provider names from numpy's build-time config."""
    try:
        import numpy as np

        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        blas = deps.get("blas", {}).get("name", "unknown")
        lapack = deps.get("lapack", {}).get("name", "unknown")
        return blas, lapack
    except Exception:
        return "unknown", "unknown"


def _fmt_pkg_list(pkgs: dict[str, str], *, width: int = 80) -> list[str]:
    """Format a name→version mapping as R's ``[N] name_ver  ...`` lines.

    Entries are sorted alphabetically, padded to the longest, and packed
    into ``width``-column lines prefixed by the 1-based index of the
    first entry on the line (R's convention).
    """
    entries = [f"{name}_{ver}" for name, ver in sorted(pkgs.items())]
    if not entries:
        return []
    pad = max(len(e) for e in entries)
    n = len(entries)
    idx_col = len(f"[{n}]")
    text_w = width - idx_col - 1
    per_line = max(1, (text_w + 1) // (pad + 1))
    out: list[str] = []
    for i in range(0, n, per_line):
        chunk = entries[i:i + per_line]
        padded = [e.ljust(pad) for e in chunk]
        idx = f"[{i + 1}]".rjust(idx_col)
        out.append(f"{idx} {' '.join(padded)}".rstrip())
    return out
