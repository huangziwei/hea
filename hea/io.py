"""I/O — readers, scanners, and dataset loaders.

* polars top-level *readers* (``read_csv``, ``read_parquet``, …) and
  *scanners* (``scan_csv``, ``scan_parquet``, …), re-exported and
  wrapped so the result is the hea subclass instead of bare polars.
  Plus a readr-friendly :func:`read_csv` shim that accepts the R
  kwargs (``na=``, ``skip=``, ``comment=``, ``col_names=``) common in
  translated R scripts.
* :func:`data` — fetch a named dataset. Pulls from the bundled
  ``rdatasets`` Python package when the ``(package, name)`` pair is
  covered there (``R``/``datasets``, ``MASS``, ``lme4``, ``nlme``);
  otherwise reads a CSV from this repo's ``datasets/`` tree
  (downloading on first access). R's factor type is restored from a
  JSON schema sidecar next to the corresponding CSV path.
* :func:`map_data` — fetch one of the polygon datasets bundled under
  ``datasets/maps/``, mirroring ``ggplot2::map_data()``.

DataFrame constructors (``from_dict`` / ``from_pandas`` / …) and
multi-frame combinators (``concat`` / ``align_frames`` / …) live in
:mod:`hea.tidy` — that's the frame namespace, period.

Usage:

>>> from hea.io import read_csv, scan_parquet
>>> from hea import data
>>> df = read_csv("flights.csv", na=["NA", "N/A"])
>>> iris = data("iris")
"""
from __future__ import annotations

import functools
import functools as _functools
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import polars as _pl
import polars as pl  # alias used by the dataset loaders below

from .formula import set_ordered_cols
from .tidy import DataFrame, _rewrap

# These polars I/O helpers return plain dict[str, DataType] (schema
# introspection only) — no wrapping needed.
from polars import (  # noqa: F401  re-exported as-is
    read_ipc_schema,
    read_parquet_metadata,
    read_parquet_schema,
)


def _wrap_factory(name: str):
    pl_func = getattr(_pl, name)

    @_functools.wraps(pl_func)
    def wrapper(*args, **kwargs):
        return _rewrap(pl_func(*args, **kwargs))

    return wrapper


# Eager I/O readers — return ``hea.DataFrame``.
_READERS = (
    "read_avro",
    "read_clipboard",
    "read_csv",
    "read_csv_batched",
    "read_database",
    "read_database_uri",
    "read_delta",
    "read_excel",
    "read_ipc",
    "read_ipc_stream",
    "read_json",
    "read_lines",
    "read_ndjson",
    "read_ods",
    "read_parquet",
)

# Lazy scanners — return ``hea.LazyFrame``.
_SCANNERS = (
    "scan_csv",
    "scan_delta",
    "scan_iceberg",
    "scan_ipc",
    "scan_lines",
    "scan_ndjson",
    "scan_parquet",
    "scan_pyarrow_dataset",
)

for _name in (*_READERS, *_SCANNERS):
    if hasattr(_pl, _name):
        globals()[_name] = _wrap_factory(_name)


# Override ``read_csv`` with a thin readr-kwarg shim. R-translated scripts
# use names like ``na=``, ``skip=``, ``comment=``, ``col_names=`` (readr);
# polars uses ``null_values=``, ``skip_rows=``, ``comment_prefix=``,
# ``has_header=`` / ``new_columns=``. The shim translates and dispatches.
_polars_read_csv = globals()["read_csv"]


def read_csv(source, *args, **kwargs):
    """readr-kwarg-friendly wrapper around polars ``read_csv``.

    Accepted readr aliases (translated to the polars equivalent):

    * ``na=`` → ``null_values=``
    * ``skip=`` → ``skip_rows=``
    * ``comment=`` → ``comment_prefix=``
    * ``col_names=False`` → ``has_header=False``
    * ``col_names=["a", "b", ...]`` → ``has_header=False`` + ``new_columns=...``

    Translator-stripped readr kwargs (so the .py never carries them):
    ``col_types=`` (use polars ``schema_overrides=`` for column-type
    hints), ``id=`` (multi-file id-column — port if needed).
    """
    if "na" in kwargs:
        kwargs["null_values"] = kwargs.pop("na")
    if "skip" in kwargs:
        kwargs["skip_rows"] = kwargs.pop("skip")
    if "comment" in kwargs:
        kwargs["comment_prefix"] = kwargs.pop("comment")
    if "col_names" in kwargs:
        col_names = kwargs.pop("col_names")
        if col_names is False:
            kwargs["has_header"] = False
        elif isinstance(col_names, (list, tuple)):
            kwargs["has_header"] = False
            kwargs["new_columns"] = list(col_names)
        # ``col_names=True`` is polars default — no-op.
    # readr accepts inline CSV content as the first arg (R detects this
    # heuristically — embedded newlines = literal). Polars's reader
    # treats every string as a path; wrap inline-string content in
    # StringIO so it gets parsed instead of being looked up on disk.
    if isinstance(source, str) and "\n" in source:
        import io as _io
        source = _io.StringIO(source)
    # readr also accepts a list of paths and concatenates the results
    # row-wise. Polars's reader takes a single path; emulate by reading
    # each and concatenating.
    if isinstance(source, (list, tuple)):
        frames = [_polars_read_csv(p, *args, **kwargs) for p in source]
        return _pl.concat(frames, how="vertical_relaxed")
    return _polars_read_csv(source, *args, **kwargs)


__all__ = [
    *_READERS, *_SCANNERS,
    "read_ipc_schema", "read_parquet_metadata", "read_parquet_schema",
    "data", "map_data",
]

del _name


# =============================================================================
# Dataset loaders (rdatasets + bundled CSV under ./datasets/)
# =============================================================================


# Our only label rewrite for rdatasets: ``"R"`` (hea's name for R's
# built-in ``datasets`` package — mirrors our ``datasets/R/`` folder, which
# avoids the ``datasets/datasets/`` path duplication). Every other package
# label is passed straight through to rdatasets.
_RDATASETS_PKG_ALIAS = {"R": "datasets"}


# (canonical_package, name) → frequency for datasets R classifies as ``ts``.
# rdatasets ships them as 2-col ``(time, value)`` frames because CSV can't
# carry an R class attribute — but on load we want them marked so the
# base-graphics plotters and ``summary()`` dispatch like R's ``ts`` would.
# The flag is set explicitly here (never inferred from column names), so
# a user-built ``DataFrame({"time": …, "value": …})`` won't accidentally
# fire the ts dispatch.
_KNOWN_TS_DATASETS: dict[tuple[str, str], float] = {
    ("datasets", "AirPassengers"):  12.0,   # monthly
    ("datasets", "BJsales"):         1.0,
    ("datasets", "BJsales.lead"):    1.0,
    ("datasets", "JohnsonJohnson"):  4.0,   # quarterly
    ("datasets", "LakeHuron"):       1.0,
    ("datasets", "Nile"):            1.0,
    ("datasets", "UKDriverDeaths"): 12.0,
    ("datasets", "UKgas"):           4.0,
    ("datasets", "USAccDeaths"):    12.0,
    ("datasets", "WWWusage"):        1.0,
    ("datasets", "airmiles"):        1.0,
    ("datasets", "austres"):         4.0,
    ("datasets", "co2"):            12.0,
    ("datasets", "discoveries"):     1.0,
    ("datasets", "fdeaths"):        12.0,
    ("datasets", "freeny.y"):        4.0,
    ("datasets", "ldeaths"):        12.0,
    ("datasets", "lh"):              1.0,
    ("datasets", "lynx"):            1.0,
    ("datasets", "mdeaths"):        12.0,
    ("datasets", "nhtemp"):          1.0,
    ("datasets", "nottem"):         12.0,
    ("datasets", "presidents"):      4.0,
    ("datasets", "sunspot.month"): 12.0,
    ("datasets", "sunspot.year"):   1.0,
    ("datasets", "sunspots"):       12.0,
    ("datasets", "treering"):        1.0,
    ("datasets", "uspop"):           0.1,
}


def _apply_ts_metadata(df: "DataFrame", package: str, name: str) -> "DataFrame":
    """Stamp ``_ts_meta`` on ``df`` if ``(package, name)`` is in the known
    R ts table. Otherwise return the frame unchanged.
    """
    canon_pkg = _RDATASETS_PKG_ALIAS.get(package, package)
    freq = _KNOWN_TS_DATASETS.get((canon_pkg, name))
    if freq is None:
        return df
    if list(df.columns) != ["time", "value"]:
        return df  # CSV shape unexpectedly diverged; skip silently.
    from .tidy.dataframe import TsMeta
    start = float(df["time"][0])
    end = float(df["time"][-1])
    df._ts_meta = TsMeta(start=start, end=end, frequency=freq)
    return df


def _find_bundled_dataset(package: str, name: str) -> Path | None:
    """Walk up from CWD looking for a bundled ``datasets/{package}/{name}.csv``.

    Returns the first match in CWD or any ancestor, or ``None`` if no
    bundled copy exists anywhere up the tree (e.g. when ``hea`` is
    installed as a package and the caller is outside the source repo).
    """
    rel = Path("datasets") / package / f"{name}.csv"
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        candidate = ancestor / rel
        if candidate.is_file():
            return candidate
    return None


def _find_schema(package: str, name: str) -> Path | None:
    """Walk up from CWD looking for ``datasets/{package}/{name}.schema.json``.

    Schema sidecars carry R factor info (levels + ordered flag) that CSV
    round-trip and ``rdatasets`` both erase. They are kept locally even when
    the data itself is sourced from ``rdatasets``.
    """
    rel = Path("datasets") / package / f"{name}.schema.json"
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        candidate = ancestor / rel
        if candidate.is_file():
            return candidate
    return None


def _normalize_rownames(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize the row-id column rdatasets injects.

    rdatasets always adds a ``rownames`` column; we rename it to
    ``rowname`` (singular — tibble convention, matches what
    ``export_data.R`` writes for bundled CSVs) and drop it entirely when
    the values are just sequential ``1..n``, which carries no information.
    """
    if "rownames" not in df.columns:
        return df
    rn = df["rownames"]
    if rn.dtype.is_integer() and rn.to_list() == list(range(1, df.height + 1)):
        return df.drop("rownames")
    return df.rename({"rownames": "rowname"})


def _try_load_rdatasets(package: str, name: str) -> pl.DataFrame | None:
    """Load ``(package, name)`` from the ``rdatasets`` package, or None if missing.

    Tries ``package`` (after the ``R`` → ``datasets`` alias) against the
    rdatasets package list, then against its item list. Returns None if
    either lookup fails — caller falls back to bundled CSV / download.
    The injected ``rownames`` column is normalized via
    ``_normalize_rownames``.
    """
    try:
        import rdatasets
    except ImportError:
        return None
    rd_pkg = _RDATASETS_PKG_ALIAS.get(package, package)
    if rd_pkg not in rdatasets.packages():
        return None
    items = {it.removesuffix(".pkl") for it in rdatasets.items(rd_pkg)}
    if name not in items:
        return None
    df = pl.from_pandas(rdatasets.data(rd_pkg, name))
    return _normalize_rownames(df)


# Accumulator for ordered-factor columns across data() calls within a session,
# mirroring tests/conftest.py. Polars has no per-column "ordered" flag, so
# ordered factors are tracked via a contextvar that hea.formula consults when
# building contrasts. This set lets multiple data() calls coexist without one
# clobbering another's ordered registrations.
_data_ordered_cols: set[str] = set()


def _apply_dataset_schema(df: pl.DataFrame, schema_path: Path | None) -> pl.DataFrame:
    """Apply the JSON schema sidecar at ``schema_path``, if present.

    Cast factor columns to ``pl.Enum`` and register ordered factors with the
    formula machinery. Without this, R's factor type erased by CSV round-trip
    (or stripped by ``rdatasets``) silently degrades ``s(...,bs='re')``,
    ``by=factor``, ``fs``, ``sz``, and ordered-contrast paths. Sidecar format
    mirrors tests/conftest.py: ``{"factors": {col: {"levels": [...], "ordered": bool}}}``.
    """
    if schema_path is None or not schema_path.is_file():
        return df
    try:
        sch = json.loads(schema_path.read_text())
    except (OSError, json.JSONDecodeError):
        return df
    factors = sch.get("factors") or {}
    if not factors:
        return df
    exprs = []
    new_ordered = set()
    for col, spec in factors.items():
        if col not in df.columns:
            continue
        levels = [str(v) for v in spec.get("levels", [])]
        if not levels:
            continue
        exprs.append(pl.col(col).cast(pl.Utf8).cast(pl.Enum(levels)))
        if spec.get("ordered"):
            new_ordered.add(col)
    if exprs:
        df = df.with_columns(exprs)
    if new_ordered:
        _data_ordered_cols.update(new_ordered)
        set_ordered_cols(frozenset(_data_ordered_cols))
    return df


@functools.lru_cache(maxsize=1)
def _rdatasets_index() -> dict[str, frozenset[str]]:
    """``{name: frozenset(packages)}`` for everything rdatasets carries.

    Cached — rdatasets's package and item lists are static for the
    process. The ``"datasets"`` package label is rewritten to ``"R"``
    to match the convention used by hea's bundled directory layout.
    """
    try:
        import rdatasets
    except ImportError:
        return {}
    index: dict[str, set[str]] = {}
    for rd_pkg in rdatasets.packages():
        display_pkg = "R" if rd_pkg == "datasets" else rd_pkg
        for it in rdatasets.items(rd_pkg):
            name = it.removesuffix(".pkl")
            index.setdefault(name, set()).add(display_pkg)
    return {name: frozenset(pkgs) for name, pkgs in index.items()}


def _bundled_index() -> dict[str, set[str]]:
    """``{name: {packages}}`` from local ``datasets/`` (CWD-walk).

    Picks up both ``.csv`` files (bundled data) and ``.schema.json``
    sidecars (covers entries where the schema is bundled but the CSV
    is downloaded on first access). Not cached — CWD changes between
    calls would invalidate the result, and a filesystem walk over
    ~10 small directories is sub-millisecond.
    """
    index: dict[str, set[str]] = {}
    seen_roots: set[Path] = set()
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        root = ancestor / "datasets"
        if not root.is_dir():
            continue
        try:
            real = root.resolve()
        except OSError:
            real = root
        if real in seen_roots:
            continue
        seen_roots.add(real)
        for pkg_dir in root.iterdir():
            if not pkg_dir.is_dir():
                continue
            pkg = pkg_dir.name
            for f in pkg_dir.iterdir():
                if f.suffix == ".csv":
                    name = f.stem
                elif f.name.endswith(".schema.json"):
                    name = f.name[: -len(".schema.json")]
                else:
                    continue
                index.setdefault(name, set()).add(pkg)
    return index


def _dataset_index() -> dict[str, list[str]]:
    """Merged ``{name: sorted(packages)}`` index across rdatasets and
    locally-bundled ``datasets/``. Drives ``data()``'s name-only
    resolution and clearer error messages on missing entries.
    """
    rd = _rdatasets_index()
    bundled = _bundled_index()
    merged: dict[str, set[str]] = {n: set(pkgs) for n, pkgs in rd.items()}
    for name, pkgs in bundled.items():
        merged.setdefault(name, set()).update(pkgs)
    return {n: sorted(pkgs) for n, pkgs in merged.items()}


def data(name: str, package: str | None = None, save_to: str = "./data",
         overwrite: bool = False) -> DataFrame:
    """Load a named dataset.

    Resolution
    ----------
    Pass ``name`` alone and hea searches the merged rdatasets +
    bundled-``datasets/`` index. If exactly one package carries that
    name, it's used. If multiple do, you get an error listing the
    candidates. If none does, you get an error saying so — without a
    doomed GitHub download.

    Pass ``package`` explicitly to override (or to pick between
    ambiguous names). Resolution order with ``package`` set:

    1. ``rdatasets`` — for any of its 75 packages (``MASS``, ``lme4``,
       ``nlme``, ``HistData``, ``ggplot2``, ``palmerpenguins``, …; see
       ``rdatasets.packages()``). The label ``"R"`` is aliased to
       rdatasets's ``"datasets"`` (R's built-in data). Offline,
       deterministic, ships with the package. The ``rownames`` column
       rdatasets injects is dropped.
    2. Bundled ``datasets/{package}/{name}.csv`` walked up from CWD —
       used for ``faraway``/``gamair``/``mgcv``/``rstanarm``/``synthetic``
       (not in rdatasets) and the few items rdatasets doesn't carry
       (e.g. ``lme4::ergoStool``).
    3. CSV download into ``save_to/{package}/{name}.csv`` — last resort,
       used when ``hea`` is installed outside the source repo and no
       bundled CSV exists. Pass ``overwrite=True`` to force a re-fetch.
       Now raises a clearer error than the bare 404 when the GitHub URL
       doesn't exist.

    A JSON schema sidecar (``datasets/{package}/{name}.schema.json``) is
    loaded next and used to restore R's factor type — columns listed
    under ``factors`` are cast to ``pl.Enum``, and ones with
    ``ordered: true`` are registered for poly contrasts. The sidecar is
    looked up via the same CWD-walk as the bundled CSV, so it applies
    even when the data itself came from rdatasets (which strips factor
    info on the way out of pandas).
    """
    if package is None:
        idx = _dataset_index()
        candidates = idx.get(name, [])
        if not candidates:
            raise ValueError(
                f"data(): {name!r} not found in rdatasets or any "
                "bundled datasets/ directory. Pass `package=` "
                "explicitly to attempt a GitHub download."
            )
        if len(candidates) > 1:
            quoted = ", ".join(repr(p) for p in candidates)
            raise ValueError(
                f"data(): {name!r} is ambiguous — found in {quoted}. "
                f"Pass `package=` to disambiguate, e.g. "
                f"data({name!r}, package={candidates[0]!r})."
            )
        package = candidates[0]
    else:
        # Explicit package: if the local index says the dataset isn't
        # there, raise immediately rather than launching a doomed
        # GitHub round-trip. We only short-circuit when ``name`` IS in
        # the index but under different packages — if ``name`` isn't
        # in the index at all, fall through to the download path
        # (covers fresh installs where ``datasets/`` is empty).
        idx = _dataset_index()
        if name in idx and package not in idx[name]:
            quoted = ", ".join(repr(p) for p in idx[name])
            raise ValueError(
                f"data(): {name!r} not in package {package!r}. "
                f"Available packages with this name: {quoted}."
            )

    df: pl.DataFrame | None = None

    if not overwrite:
        df = _try_load_rdatasets(package, name)
        if df is None:
            bundled = _find_bundled_dataset(package, name)
            if bundled is not None:
                df = pl.read_csv(bundled, null_values="NA")

    if df is None:
        datapath = os.path.join(save_to, package)
        csv_path = Path(datapath) / f"{name}.csv"
        if not csv_path.exists() or overwrite:
            # Snapshot which dirs don't exist yet so a failed download can
            # roll them back — otherwise a network error leaves an empty
            # data/<package>/ (and possibly data/) behind in the CWD.
            created_dirs = [Path(p) for p in (save_to, datapath) if not os.path.exists(p)]
            os.makedirs(datapath, exist_ok=True)
            print(f"Downloading {name} (from {package})...")
            base = f"https://raw.githubusercontent.com/huangziwei/hea/main/datasets/{package}/{name}"
            try:
                urllib.request.urlretrieve(f"{base}.csv", csv_path)
            except urllib.error.HTTPError as e:
                csv_path.unlink(missing_ok=True)
                for p in reversed(created_dirs):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                if e.code == 404:
                    suggestions = idx.get(name, [])
                    if suggestions:
                        quoted = ", ".join(repr(p) for p in suggestions)
                        raise ValueError(
                            f"data(): {name!r} not in package {package!r} "
                            f"(404 from GitHub). Available in: {quoted}."
                        ) from None
                    raise ValueError(
                        f"data(): {name!r} not in package {package!r} "
                        "(404 from GitHub) and not in any other known "
                        "package."
                    ) from None
                raise
            except Exception:
                csv_path.unlink(missing_ok=True)
                for p in reversed(created_dirs):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                raise
            try:
                urllib.request.urlretrieve(
                    f"{base}.schema.json", csv_path.with_suffix(".schema.json")
                )
            except urllib.error.HTTPError:
                pass
        df = pl.read_csv(csv_path, null_values="NA")

    df = _apply_dataset_schema(df, _find_schema(package, name))
    out = DataFrame._from_pydf(df._df)
    return _apply_ts_metadata(out, package, name)


# ---------------------------------------------------------------------------
# map_data — polygon datasets ported from R's ``maps`` package (CIA World
# Data Bank II), bundled as zstd-compressed parquet under
# ``datasets/maps/{name}.parquet``. Output schema mirrors
# ``ggplot2::map_data()``: ``long, lat, group, order, region, subregion``.
# ---------------------------------------------------------------------------

_MAP_DATA_NAMES = (
    "world", "world2", "usa", "state", "county",
    "nz", "france", "italy", "lakes",
)


def _find_bundled_map(name: str) -> Path | None:
    """Walk up from CWD looking for ``datasets/maps/{name}.parquet``."""
    rel = Path("datasets") / "maps" / f"{name}.parquet"
    cwd = Path.cwd()
    for ancestor in (cwd, *cwd.parents):
        candidate = ancestor / rel
        if candidate.is_file():
            return candidate
    return None


def map_data(name: str, *, save_to: str = "./data",
             overwrite: bool = False) -> DataFrame:
    """Load a bundled map polygon dataset, mirroring ``ggplot2::map_data()``.

    Returns a frame with columns ``long``, ``lat``, ``group``, ``order``,
    ``region``, ``subregion`` — ready for ``geom_polygon(aes(x=long,
    y=lat, group=group))``. Pair with ``coord_quickmap()`` for a
    rough-and-ready Mercator-ish projection.

    Available names: ``world``, ``world2``, ``usa``, ``state``,
    ``county``, ``nz``, ``france``, ``italy``, ``lakes``. Data was
    extracted from R's ``maps`` package (originally CIA World Data Bank
    II — public domain).

    Resolution mirrors :func:`data`: bundled ``datasets/maps/`` walked
    up from CWD first; on miss, a parquet download into
    ``save_to/maps/{name}.parquet`` (set ``overwrite=True`` to refetch).
    """
    if name not in _MAP_DATA_NAMES:
        raise ValueError(
            f"map_data(): unknown map {name!r}. "
            f"Available: {', '.join(_MAP_DATA_NAMES)}."
        )

    df: pl.DataFrame | None = None
    if not overwrite:
        bundled = _find_bundled_map(name)
        if bundled is not None:
            df = pl.read_parquet(bundled)

    if df is None:
        datapath = os.path.join(save_to, "maps")
        pq_path = Path(datapath) / f"{name}.parquet"
        if not pq_path.exists() or overwrite:
            created_dirs = [Path(p) for p in (save_to, datapath)
                            if not os.path.exists(p)]
            os.makedirs(datapath, exist_ok=True)
            print(f"Downloading map_data({name!r})...")
            url = (
                "https://raw.githubusercontent.com/huangziwei/hea/main/"
                f"datasets/maps/{name}.parquet"
            )
            try:
                urllib.request.urlretrieve(url, pq_path)
            except Exception:
                pq_path.unlink(missing_ok=True)
                for p in reversed(created_dirs):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                raise
        df = pl.read_parquet(pq_path)

    return DataFrame._from_pydf(df._df)
