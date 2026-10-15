import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from hea.formula import set_ordered_cols

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def assert_fp_equiv(a, b):
    """Same-model assertion for two fits that share one code path."""
    np.testing.assert_allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-11
    )


FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = Path(__file__).parent.parent / "datasets"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def fixtures_by_kind(kind: str):
    m = load_manifest()
    return [
        e for e in m["entries"] if e.get("status") == "ok" and e.get("kind") == kind
    ]


_data_cache: dict[tuple[str, str], pl.DataFrame] = {}


def _pkg_subdir(pkg: str) -> str:
    return "R" if pkg == "datasets" else pkg


def _apply_schema(df: pl.DataFrame, pkg: str, name: str) -> pl.DataFrame:
    """Re-cast factor columns from the sidecar schema into pl.Enum."""
    from hea.io import _apply_dataset_schema

    schema_path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    return _apply_dataset_schema(df, schema_path)


def ordered_schema_cols(pkg: str, name: str) -> frozenset[str]:
    """Columns marked `ordered: true` in the dataset's schema sidecar."""
    path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    if not path.exists():
        return frozenset()
    sch = json.loads(path.read_text())
    return frozenset(
        col for col, spec in sch.get("factors", {}).items() if spec.get("ordered")
    )


_current_ordered_cols: "set[str]" = set()


def load_dataset(pkg: str, name: str) -> pl.DataFrame:
    """Test-side dataset loader. Delegates to ``hea.data`` (which
    routes to ``rdatasets`` when covered, bundled CSV otherwise) and caches
    the result so repeated fixture loads are cheap.
    """
    from hea import data as _data

    key = (pkg, name)
    if key not in _data_cache:
        df = _data(name, _pkg_subdir(pkg))
        if "rowname" in df.columns:
            df = df.drop("rowname")
        _data_cache[key] = df
    ordered = ordered_schema_cols(pkg, name)
    if ordered:
        _current_ordered_cols.update(ordered)
        set_ordered_cols(frozenset(_current_ordered_cols))
    return _data_cache[key].clone()


@pytest.fixture(autouse=True)
def _reset_ordered_cols():
    """Clear the ordered-cols contextvar and the accumulator before each test
    so cached-dataset fixtures from an earlier test don't bleed ordered labels
    into an unrelated one."""
    _current_ordered_cols.clear()
    set_ordered_cols(frozenset())
    yield
    _current_ordered_cols.clear()
    set_ordered_cols(frozenset())


def fixture_meta(fx_id: str) -> tuple[dict, dict]:
    fx = FIXTURE_ROOT / fx_id
    return (
        json.loads((fx / "meta.json").read_text()),
        json.loads((fx / "X_meta.json").read_text()),
    )


def fixture_X_ref(fx_id: str) -> pl.DataFrame:
    return pl.read_csv(FIXTURE_ROOT / fx_id / "X.csv", null_values="NA")


GLM_ORACLE_ROOT = FIXTURE_ROOT / "glm"


def load_glm_oracle(name: str) -> dict:
    """Load a stats::glm() oracle by id (e.g. 'poisson_log_quine')."""
    path = GLM_ORACLE_ROOT / name / "oracle.json"
    if not path.exists():
        raise FileNotFoundError(
            f"glm oracle {name!r} not found at {path}; "
            "regenerate via `Rscript tests/scripts/make_glm_oracles.R`"
        )
    return json.loads(path.read_text())


_NMATH_R_ORACLE = Path(__file__).parent / "scripts" / "nmath_r_oracle.R"


def have_rscript() -> bool:
    return shutil.which("Rscript") is not None


def r_scalar_values(exprs):
    """Evaluate each scalar R expression on THIS machine; return ``{expr: float}``."""
    body = "".join(f'cat(sprintf("%.17g\\n", as.double({e})))\n' for e in exprs)
    out = subprocess.run(
        ["Rscript", "-e", body],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return dict(zip(exprs, (float(x) for x in out.split())))


def run_rs_r_oracle(cases, workdir) -> dict:
    """Evaluate R's d/p/q for each case and return ``{name: np.ndarray}``."""
    workdir = str(workdir)
    os.makedirs(workdir, exist_ok=True)
    spec_lines = []
    for name, fn, arrays, flags in cases:
        argfiles = []
        for i, a in enumerate(arrays):
            fname = f"{name}__{i}.bin"
            np.ascontiguousarray(a, dtype="<f8").ravel().tofile(
                os.path.join(workdir, fname)
            )
            argfiles.append(fname)
        flagstr = ",".join("TRUE" if b else "FALSE" for b in flags)
        spec_lines.append(f"{name}|{fn}|{','.join(argfiles)}|{flagstr}")
    (Path(workdir) / "spec.txt").write_text("\n".join(spec_lines) + "\n")

    subprocess.run(
        ["Rscript", str(_NMATH_R_ORACLE), workdir],
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=300,
        capture_output=True,
        text=True,
    )

    out = {}
    for name, fn, arrays, flags in cases:
        out[name] = np.fromfile(os.path.join(workdir, f"{name}.out.bin"), dtype="<f8")
    return out


_RNG_R_ORACLE = Path(__file__).parent / "scripts" / "rng_r_oracle.R"


def run_rng_r_oracle(seed, cases, workdir) -> dict:
    """Evaluate R's `set.seed(seed); <rcall>` for each case; return ``{name:
    np.ndarray}``.
    """
    workdir = str(workdir)
    os.makedirs(workdir, exist_ok=True)
    spec_lines = [str(int(seed))]
    for case in cases:
        name, rcall, params = case[0], case[1], case[2]
        for i, a in enumerate(params):
            np.ascontiguousarray(a, dtype="<f8").ravel().tofile(
                os.path.join(workdir, f"{name}__{i}.bin")
            )
        spec_lines.append(f"{name}|{rcall}|{len(params)}")
    (Path(workdir) / "spec.txt").write_text("\n".join(spec_lines) + "\n")

    subprocess.run(
        ["Rscript", str(_RNG_R_ORACLE), workdir],
        stdin=subprocess.DEVNULL,
        check=True,
        timeout=300,
        capture_output=True,
        text=True,
    )

    return {
        case[0]: np.fromfile(os.path.join(workdir, f"{case[0]}.out.bin"), dtype="<f8")
        for case in cases
    }
