"""Parity runner — run R + hea side-by-side, diff results, log gaps.

Public entry: :func:`parity`. Given an R script path, translates it to
hea Python, runs both, compares the resulting last-expression frames,
and appends any divergences to :mod:`hea.translate.gaps`'s registry.

Pieces:

- :func:`run_r` — subprocess ``R --vanilla`` with an inline driver that
  ``source()``s the user script, captures ``.Last.value``, and writes a
  CSV + schema JSON via ``readr::write_csv`` + ``jsonlite``.
- :func:`run_py` — subprocess ``python -m hea.translate._py_capture``
  (separate module — see that file).
- :func:`diff_frames` — load both sides, compare schema / row count /
  values (numeric with rel+abs tolerance) / factor levels.
- :func:`parity` — wires it all together.

Float tolerance defaults: ``rel_tol=1e-7, abs_tol=1e-9`` — tight enough
to catch real numerical divergences, loose enough to absorb the
double-to-string-to-double round-trip through CSV.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import gaps as _gaps
from .gaps import Gap
from .r_to_py import translate as _r_to_py


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunResult:
    """One side of the parity check — either R or Python.

    ``captured`` is set at the time of running, so the result stays
    interpretable after any temporary output directory has been cleaned
    up. Don't reach through ``out_csv``/``out_schema`` after the call —
    they may already be gone.
    """

    returncode: int
    stdout: str
    stderr: str
    out_csv: Optional[Path] = None
    out_schema: Optional[Path] = None
    captured: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.captured


@dataclass(slots=True)
class ParityResult:
    """Top-level outcome of :func:`parity`."""

    r_script: Path
    translated_py: str
    r_run: RunResult
    py_run: RunResult
    gaps: list[Gap] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.r_run.ok and self.py_run.ok and not self.gaps


# ---------------------------------------------------------------------------
# R driver (inline string template)
# ---------------------------------------------------------------------------


# The driver source()s the user script with print.eval=FALSE so we don't
# pollute stdout, then pulls the last expression's value off the source()
# return list. Serialization is via readr (CSV) + jsonlite (schema JSON).
_R_DRIVER_TEMPLATE = r"""
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(jsonlite)
})

.user_script <- "@@USER_SCRIPT@@"
.out_csv     <- "@@OUT_CSV@@"
.out_schema  <- "@@OUT_SCHEMA@@"

.handler <- function(e) {
  cat("R RUNTIME ERROR:\n", conditionMessage(e), "\n", sep = "", file = stderr())
  quit(status = 2)
}

.result <- tryCatch(
  source(.user_script, echo = FALSE, print.eval = FALSE)$value,
  error = .handler
)

# Capture factor levels and dtypes per column.
.capture_schema <- function(df) {
  factors <- list()
  dtypes  <- list()
  for (nm in names(df)) {
    x <- df[[nm]]
    dtypes[[nm]] <- class(x)[1]
    if (is.factor(x)) {
      factors[[nm]] <- list(levels = levels(x), ordered = is.ordered(x))
    }
  }
  list(dtypes = dtypes, factors = factors, shape = c(nrow(df), ncol(df)))
}

.write_capture <- function(df) {
  # CSV (NA values mark missing — R's default).
  readr::write_csv(df, .out_csv, na = "")
  schema <- .capture_schema(df)
  jsonlite::write_json(schema, .out_schema, auto_unbox = TRUE, null = "null")
}

if (inherits(.result, "data.frame")) {
  .write_capture(as.data.frame(.result))
} else if (is.atomic(.result)) {
  # Scalar / vector — wrap in a one-column df so the diff path stays uniform.
  .write_capture(data.frame(value = .result, stringsAsFactors = FALSE))
} else if (inherits(.result, "ggplot")) {
  # Plot output — for now, just capture the data slot if present.
  if (!is.null(.result$data) && inherits(.result$data, "data.frame")) {
    .write_capture(.result$data)
  } else {
    cat("R: plot result has no inspectable data slot\n", file = stderr())
    quit(status = 3)
  }
} else if (is.null(.result)) {
  # No last value — write empty.
  writeLines(character(0), .out_csv)
  jsonlite::write_json(
    list(dtypes = list(), factors = list(), shape = c(0L, 0L)),
    .out_schema, auto_unbox = TRUE, null = "null"
  )
} else {
  cat(
    "R: cannot serialize result of class ", paste(class(.result), collapse = "/"), "\n",
    sep = "", file = stderr()
  )
  quit(status = 3)
}
"""


def _build_r_driver(user_script: Path, out_csv: Path, out_schema: Path) -> str:
    """Render the R driver template with the user-supplied paths.

    Uses literal-marker substitution rather than ``str.format`` — the R
    driver contains many ``{`` / ``}`` (function bodies, ``tryCatch``)
    that would confuse ``format``-style placeholders.
    """
    def _r_escape(p: Path) -> str:
        # R double-quoted strings need ``\\`` and ``"`` escaped. Paths
        # under macOS / Linux won't contain either; defend regardless.
        return str(p).replace("\\", "\\\\").replace('"', '\\"')

    return (
        _R_DRIVER_TEMPLATE
        .replace("@@USER_SCRIPT@@", _r_escape(user_script))
        .replace("@@OUT_CSV@@", _r_escape(out_csv))
        .replace("@@OUT_SCHEMA@@", _r_escape(out_schema))
    )


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------


def run_r(script_path: Path, out_dir: Path, *, timeout: float = 60.0) -> RunResult:
    """Run ``script_path`` under R. Writes CSV + schema to ``out_dir``.

    Returns a :class:`RunResult` regardless of success — callers inspect
    ``ok``. R is invoked with ``--vanilla`` to skip user-side ``.Rprofile``
    and site-init, keeping the runner reproducible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "out_r.csv"
    out_schema = out_dir / "out_r.schema.json"

    driver = _build_r_driver(script_path, out_csv, out_schema)
    proc = subprocess.run(
        ["R", "--vanilla", "--slave", "-e", driver],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    captured = out_csv.exists() and out_schema.exists()
    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        out_csv=out_csv if captured else None,
        out_schema=out_schema if captured else None,
        captured=captured,
    )


def run_py(script_path: Path, out_dir: Path, *, timeout: float = 60.0) -> RunResult:
    """Run ``script_path`` under Python via the ``_py_capture`` driver."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "out_py.csv"
    out_schema = out_dir / "out_py.schema.json"

    proc = subprocess.run(
        [
            sys.executable, "-m", "hea.translate._py_capture",
            str(script_path), str(out_csv), str(out_schema),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    captured = out_csv.exists() and out_schema.exists()
    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        out_csv=out_csv if captured else None,
        out_schema=out_schema if captured else None,
        captured=captured,
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


# R dtype names (class()[1]) → broad polars/Python categories. We don't
# enforce exact dtype equality (Int64 vs Float64 round-trip noise is real);
# instead, we group both sides and compare the group.
_R_DTYPE_GROUP = {
    "integer":   "numeric",
    "numeric":   "numeric",
    "double":    "numeric",
    "complex":   "numeric",
    "character": "string",
    "factor":    "factor",
    "ordered":   "factor",
    "logical":   "bool",
    "Date":      "date",
    "POSIXct":   "datetime",
    "POSIXlt":   "datetime",
}


def _pl_dtype_group(dtype_str: str) -> str:
    s = dtype_str.lower()
    if any(k in s for k in ("int", "float", "decimal")):
        return "numeric"
    if "bool" in s:
        return "bool"
    if "date" in s and "datetime" not in s:
        return "date"
    if "datetime" in s or "timestamp" in s:
        return "datetime"
    if "enum" in s or "categorical" in s:
        return "factor"
    if "string" in s or "utf8" in s or "binary" in s:
        return "string"
    return s


def diff_frames(
    r_csv: Path,
    r_schema: Path,
    py_csv: Path,
    py_schema: Path,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
    source_label: str = "<unknown>",
) -> list[Gap]:
    """Diff the two CSV+schema captures. Returns a list of gap rows
    describing each divergence (empty list = perfect match).

    Numeric columns use a tolerant compare; everything else is exact.
    Factor levels are compared as ordered sets.
    """
    import polars as pl

    out: list[Gap] = []

    r_meta = json.loads(r_schema.read_text())
    py_meta = json.loads(py_schema.read_text())

    r_df = pl.read_csv(r_csv) if r_csv.stat().st_size > 0 else pl.DataFrame()
    py_df = pl.read_csv(py_csv) if py_csv.stat().st_size > 0 else pl.DataFrame()

    # --- Schema: columns ---
    if list(r_df.columns) != list(py_df.columns):
        out.append(Gap(
            kind="result_diff_schema",
            subject="columns",
            source=source_label,
            notes=f"R columns: {list(r_df.columns)}\nhea columns: {list(py_df.columns)}",
        ))
        return out  # can't compare further if columns disagree

    # --- Schema: dtype groups ---
    for col in r_df.columns:
        r_grp = _R_DTYPE_GROUP.get(r_meta["dtypes"].get(col, ""), "?")
        py_grp = _pl_dtype_group(py_meta["dtypes"].get(col, ""))
        if r_grp != py_grp:
            out.append(Gap(
                kind="result_diff_schema",
                subject=f"dtype:{col}",
                source=source_label,
                notes=f"R dtype: {r_meta['dtypes'].get(col)} → {r_grp}; "
                      f"hea dtype: {py_meta['dtypes'].get(col)} → {py_grp}",
            ))

    # --- Row count ---
    if r_df.height != py_df.height:
        out.append(Gap(
            kind="result_diff_row_count",
            subject="height",
            source=source_label,
            notes=f"R rows: {r_df.height}; hea rows: {py_df.height}",
        ))
        return out  # value diff requires aligned shapes

    # --- Factor levels ---
    r_factors = r_meta.get("factors", {})
    py_factors = py_meta.get("factors", {})
    for col in set(r_factors) | set(py_factors):
        r_lev = r_factors.get(col, {}).get("levels", [])
        py_lev = py_factors.get(col, {}).get("levels", [])
        if r_lev != py_lev:
            out.append(Gap(
                kind="result_diff_factor",
                subject=col,
                source=source_label,
                notes=f"R levels: {r_lev}\nhea levels: {py_lev}",
            ))

    # --- Values ---
    for col in r_df.columns:
        r_col = r_df[col]
        py_col = py_df[col]
        # Compatible dtypes (already grouped above; coerce for compare).
        r_grp = _R_DTYPE_GROUP.get(r_meta["dtypes"].get(col, ""), "?")
        if r_grp == "numeric":
            diff = _numeric_diff(r_col, py_col, rel_tol=rel_tol, abs_tol=abs_tol)
            if diff:
                out.append(Gap(
                    kind="result_diff_values",
                    subject=col,
                    source=source_label,
                    notes=diff,
                ))
        else:
            # Cast both to string for exact compare. CSV round-trip
            # normalizes most representational quirks.
            r_str = r_col.cast(pl.Utf8, strict=False)
            py_str = py_col.cast(pl.Utf8, strict=False)
            mismatches = (r_str != py_str).sum()
            # Treat both-null as match.
            both_null = r_str.is_null() & py_str.is_null()
            mismatches = mismatches - int(both_null.sum())
            if mismatches:
                out.append(Gap(
                    kind="result_diff_values",
                    subject=col,
                    source=source_label,
                    notes=f"{mismatches} non-numeric mismatches in {col!r}",
                ))

    return out


def _numeric_diff(r_col, py_col, *, rel_tol: float, abs_tol: float) -> str:
    """Return an empty string if values agree, else a human description
    of how many cells exceed the tolerance and the worst absolute diff."""
    import polars as pl

    # Cast both to Float64 for comparison.
    a = r_col.cast(pl.Float64, strict=False)
    b = py_col.cast(pl.Float64, strict=False)
    # Both null → match. One null → mismatch.
    a_null = a.is_null()
    b_null = b.is_null()
    one_null = (a_null & ~b_null) | (~a_null & b_null)
    n_one_null = int(one_null.sum())

    diff_series = (a - b).abs()
    threshold = (a.abs() * rel_tol).fill_null(0) + abs_tol
    violators = (diff_series > threshold).fill_null(False)
    n_violators = int(violators.sum())

    if n_violators == 0 and n_one_null == 0:
        return ""
    parts = []
    if n_violators:
        max_diff = float(diff_series.max() or 0.0)
        parts.append(f"{n_violators} values exceed tolerance (max abs diff {max_diff:.6g})")
    if n_one_null:
        parts.append(f"{n_one_null} cells null on one side only")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def parity(
    r_script_path: Path,
    *,
    rel_tol: float = 1e-7,
    abs_tol: float = 1e-9,
    log: bool = True,
) -> ParityResult:
    """Translate ``r_script_path`` → hea Python, run both, diff, log gaps.

    Side-effect: by default appends any divergences to the registry at
    ``tests/parity/gaps.jsonl``. Pass ``log=False`` to skip (useful for
    one-off diagnostic runs).
    """
    r_script_path = Path(r_script_path)
    r_source = r_script_path.read_text(encoding="utf-8")
    try:
        py_source = _r_to_py(r_source)
    except Exception as e:
        gap = Gap(
            kind="parse_error",
            subject=r_script_path.name,
            source=str(r_script_path),
            snippet=r_source[:200],
            notes=f"{type(e).__name__}: {e}",
        )
        if log:
            _gaps.log_gap(
                kind=gap.kind, subject=gap.subject, source=gap.source,
                snippet=gap.snippet, notes=gap.notes,
            )
        return ParityResult(
            r_script=r_script_path,
            translated_py="",
            r_run=RunResult(-1, "", str(e)),
            py_run=RunResult(-1, "", str(e)),
            gaps=[gap],
        )

    with tempfile.TemporaryDirectory(prefix="hea_parity_") as tmpdir:
        tmp = Path(tmpdir)
        py_path = tmp / (r_script_path.stem + ".py")
        py_path.write_text(py_source, encoding="utf-8")

        r_run = run_r(r_script_path, tmp)
        py_run = run_py(py_path, tmp)

        observed: list[Gap] = []
        if not r_run.ok:
            observed.append(Gap(
                kind="runtime_error_r",
                subject=r_script_path.name,
                source=str(r_script_path),
                notes=r_run.stderr[:1000],
            ))
        if not py_run.ok:
            observed.append(Gap(
                kind="runtime_error_py",
                subject=r_script_path.name,
                source=str(r_script_path),
                snippet=r_source[:200],
                translation=py_source[:200],
                notes=py_run.stderr[:1000],
            ))
        if r_run.ok and py_run.ok:
            observed.extend(diff_frames(
                r_run.out_csv, r_run.out_schema,
                py_run.out_csv, py_run.out_schema,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                source_label=str(r_script_path),
            ))

        if log:
            for g in observed:
                _gaps.log_gap(
                    kind=g.kind, subject=g.subject, source=g.source,
                    snippet=g.snippet, translation=g.translation, notes=g.notes,
                )

        return ParityResult(
            r_script=r_script_path,
            translated_py=py_source,
            r_run=r_run,
            py_run=py_run,
            gaps=observed,
        )
