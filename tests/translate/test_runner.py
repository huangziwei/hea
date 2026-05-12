"""Tests for hea.translate.runner — Phase 7 acceptance.

End-to-end parity tests run a small R script + its translated hea Python,
then assert the runner produces matching outputs (or a known classified
gap). Each test uses a self-contained data.frame literal so the harness
doesn't depend on any external dataset packages.

The diff-utility tests construct synthetic CSV+schema pairs to exercise
each gap category.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from hea.translate.gaps import Gap, _make_id, log_gap, read_gaps
from hea.translate.runner import (
    RunResult,
    diff_frames,
    parity,
    run_py,
    run_r,
)


# Skip everything if R isn't on PATH — the runner is fundamentally
# coupled to it, no point trying to mock around that.
_HAS_R = shutil.which("R") is not None
pytestmark = pytest.mark.skipif(not _HAS_R, reason="R not on PATH")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_r_script(content: str, tmp: Path, name: str = "script.R") -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def _write_capture_pair(tmp: Path, side: str, df: pl.DataFrame, factors: dict | None = None) -> tuple[Path, Path]:
    """Build a synthetic CSV + schema JSON, mimicking what run_r / run_py
    produce. Used for unit-testing :func:`diff_frames` directly."""
    csv = tmp / f"{side}.csv"
    schema_path = tmp / f"{side}.schema.json"
    df.write_csv(csv)
    # Use R-style dtype names for the R side, polars-style for the Py side.
    if side == "r":
        dtype_map = {col: _pl_to_r_dtype(d) for col, d in zip(df.columns, df.dtypes)}
    else:
        dtype_map = {col: str(d) for col, d in zip(df.columns, df.dtypes)}
    schema = {
        "dtypes": dtype_map,
        "factors": factors or {},
        "shape": list(df.shape),
    }
    schema_path.write_text(json.dumps(schema))
    return csv, schema_path


def _pl_to_r_dtype(dtype) -> str:
    """Approximate inverse of runner._R_DTYPE_GROUP for synthetic fixtures."""
    s = str(dtype).lower()
    if "int" in s:
        return "integer"
    if "float" in s:
        return "numeric"
    if "bool" in s:
        return "logical"
    if "string" in s or "utf8" in s:
        return "character"
    return s


# ---------------------------------------------------------------------------
# diff_frames unit tests — no R subprocess needed
# ---------------------------------------------------------------------------


class TestDiffFrames:
    """Validate the diff utility in isolation with synthetic fixtures."""

    def test_perfect_match(self, tmp_path):
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        r_csv, r_schema = _write_capture_pair(tmp_path, "r", df)
        py_csv, py_schema = _write_capture_pair(tmp_path, "py", df)
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert gaps == []

    def test_column_set_mismatch(self, tmp_path):
        r_df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        py_df = pl.DataFrame({"x": [1, 2]})
        r_csv, r_schema = _write_capture_pair(tmp_path, "r", r_df)
        py_csv, py_schema = _write_capture_pair(tmp_path, "py", py_df)
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert len(gaps) == 1
        assert gaps[0].kind == "result_diff_schema"
        assert gaps[0].subject == "columns"

    def test_row_count_mismatch(self, tmp_path):
        r_df = pl.DataFrame({"x": [1, 2, 3]})
        py_df = pl.DataFrame({"x": [1, 2]})
        r_csv, r_schema = _write_capture_pair(tmp_path, "r", r_df)
        py_csv, py_schema = _write_capture_pair(tmp_path, "py", py_df)
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert any(g.kind == "result_diff_row_count" for g in gaps)

    def test_numeric_within_tolerance(self, tmp_path):
        # Tiny floating-point noise within tolerance should NOT emit a gap.
        r_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        py_df = pl.DataFrame({"x": [1.0 + 1e-12, 2.0 - 1e-12, 3.0]})
        r_csv, r_schema = _write_capture_pair(tmp_path, "r", r_df)
        py_csv, py_schema = _write_capture_pair(tmp_path, "py", py_df)
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert gaps == []

    def test_numeric_outside_tolerance(self, tmp_path):
        r_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        py_df = pl.DataFrame({"x": [1.0, 2.5, 3.0]})  # 0.5 diff in row 1
        r_csv, r_schema = _write_capture_pair(tmp_path, "r", r_df)
        py_csv, py_schema = _write_capture_pair(tmp_path, "py", py_df)
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert any(g.kind == "result_diff_values" for g in gaps)

    def test_factor_levels_mismatch(self, tmp_path):
        df = pl.DataFrame({"x": ["a", "b", "c"]})
        r_csv, r_schema = _write_capture_pair(
            tmp_path, "r", df, factors={"x": {"levels": ["a", "b", "c"], "ordered": False}}
        )
        py_csv, py_schema = _write_capture_pair(
            tmp_path, "py", df, factors={"x": {"levels": ["a", "b"], "ordered": False}}
        )
        gaps = diff_frames(r_csv, r_schema, py_csv, py_schema)
        assert any(g.kind == "result_diff_factor" for g in gaps)


# ---------------------------------------------------------------------------
# Gap registry
# ---------------------------------------------------------------------------


class TestGapRegistry:
    def test_log_and_read(self, tmp_path):
        registry = tmp_path / "gaps.jsonl"
        g = log_gap(
            kind="unknown_function",
            subject="lubridate::ymd",
            source="example.R:42",
            registry=registry,
        )
        assert g.id  # auto-generated
        rows = read_gaps(registry)
        assert len(rows) == 1
        assert rows[0].subject == "lubridate::ymd"

    def test_dedup_on_repeat(self, tmp_path):
        registry = tmp_path / "gaps.jsonl"
        g1 = log_gap(kind="unknown_function", subject="foo", source="x.R", registry=registry)
        g2 = log_gap(kind="unknown_function", subject="foo", source="x.R", registry=registry)
        rows = read_gaps(registry)
        assert len(rows) == 1
        # The two writes resolve to the same id (stable hash on key fields).
        assert g1.id == g2.id

    def test_distinct_when_subject_differs(self, tmp_path):
        registry = tmp_path / "gaps.jsonl"
        log_gap(kind="unknown_function", subject="foo", source="x.R", registry=registry)
        log_gap(kind="unknown_function", subject="bar", source="x.R", registry=registry)
        assert len(read_gaps(registry)) == 2

    def test_unknown_kind_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            Gap(kind="not_a_real_kind", subject="x")


# ---------------------------------------------------------------------------
# End-to-end parity — subprocess R + Python
# ---------------------------------------------------------------------------


class TestParityE2E:
    def test_simple_pipeline_matches(self, tmp_path):
        # A small self-contained pipeline. ``data.frame()`` + ``filter`` +
        # ``mutate`` + ``arrange`` — every step is in the registry.
        script = _write_r_script(
            "library(dplyr)\n"
            "df <- data.frame(x = c(1, 2, 3, 4, 5), y = c(10, 20, 30, 40, 50))\n"
            "df |> filter(x > 2) |> mutate(z = x + y) |> arrange(desc(z))\n",
            tmp_path,
        )
        result = parity(script, log=False)
        assert result.ok, (
            f"parity failed: R stderr={result.r_run.stderr!r} "
            f"py stderr={result.py_run.stderr!r} "
            f"gaps={[g.kind for g in result.gaps]}"
        )
        assert result.gaps == []

    def test_group_by_summarize_matches(self, tmp_path):
        script = _write_r_script(
            "library(dplyr)\n"
            "df <- data.frame(\n"
            '  origin = c("A", "B", "A", "B"),\n'
            "  arr_delay = c(1.0, 2.0, 3.0, 4.0)\n"
            ")\n"
            "df |>\n"
            "  group_by(origin) |>\n"
            "  summarize(avg = mean(arr_delay, na.rm = TRUE))\n",
            tmp_path,
        )
        result = parity(script, log=False)
        assert result.ok, (
            f"R stderr={result.r_run.stderr!r} "
            f"py stderr={result.py_run.stderr!r} "
            f"gaps={[g.kind for g in result.gaps]}"
        )

    def test_translation_failure_logs_parse_error(self, tmp_path):
        # Non-parseable R triggers a parse_error gap. We use ``${{`` —
        # the dollar-followed-by-doublebrace is invalid R.
        script = _write_r_script("$invalid syntax here\n", tmp_path)
        result = parity(script, log=False)
        assert not result.ok
        # Parse failure either in the R parser or downstream — at minimum
        # a runtime_error_r or parse_error should land.
        assert any(g.kind in ("parse_error", "runtime_error_r") for g in result.gaps)

    def test_run_r_alone(self, tmp_path):
        script = _write_r_script(
            "data.frame(x = c(1L, 2L, 3L))\n",
            tmp_path,
        )
        result = run_r(script, tmp_path)
        assert result.ok
        df = pl.read_csv(result.out_csv)
        assert df.columns == ["x"]
        assert df["x"].to_list() == [1, 2, 3]

    def test_run_py_alone(self, tmp_path):
        py_script = tmp_path / "script.py"
        py_script.write_text(
            "import hea\n"
            "hea.DataFrame({'x': [1, 2, 3]})\n",
            encoding="utf-8",
        )
        result = run_py(py_script, tmp_path)
        assert result.ok
        df = pl.read_csv(result.out_csv)
        assert df.columns == ["x"]
        assert df["x"].to_list() == [1, 2, 3]

    def test_parity_writes_to_registry_when_log_true(self, tmp_path, monkeypatch):
        # Redirect the default registry path to a per-test tempfile.
        import hea.translate.gaps as gaps_mod
        registry = tmp_path / "gaps.jsonl"
        monkeypatch.setattr(gaps_mod, "_DEFAULT_REGISTRY", registry)

        # Trigger a failure: a Python runtime error from a translated
        # script that hits an unknown function on the Python side.
        # ``foo()`` in R isn't in the registry, so the translated
        # Python will be ``foo()`` — undefined → NameError.
        script = _write_r_script("foo()\n", tmp_path)
        result = parity(script, log=True)
        assert not result.ok
        rows = read_gaps(registry)
        assert len(rows) >= 1
