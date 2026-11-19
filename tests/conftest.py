import gzip
import json
import math
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


def assert_allclose_up_to_signs(
    got, want, *, rows=False, rtol=0.0, atol=0.0, err_msg=""
):
    """``got`` equals ``want`` with the signs of whole columns flipped (and of
    whole rows too, with ``rows``), to the tolerance: the comparison for
    values built from eigen- or singular vectors, whose signs follow the
    arithmetic. A vector is one column. Column signs alone are read from
    each column's inner product with ``want``; with ``rows`` they spread
    from the entries of ``want`` above ``atol``, so every entry must then fit
    one pattern of row and column signs."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{err_msg}: shape {got.shape} != {want.shape}"
    g = got.reshape(want.shape[0], -1)
    w = want.reshape(want.shape[0], -1)
    m, n = w.shape
    r = np.ones(m)
    if not rows:
        c = np.sign(np.sum(g * w, axis=0))
    else:
        r[:] = 0.0
        c = np.zeros(n)
        big = np.abs(w) > atol
        flip = np.sign(g) * np.sign(w)
        for start in range(m):
            if r[start]:
                continue
            r[start] = 1.0
            todo = [(True, start)]
            while todo:
                is_row, k = todo.pop()
                if is_row:
                    for j in np.flatnonzero(big[k] & (c == 0)):
                        c[j] = flip[k, j] * r[k] or 1.0
                        todo.append((False, j))
                else:
                    for i in np.flatnonzero(big[:, k] & (r == 0)):
                        r[i] = flip[i, k] * c[k] or 1.0
                        todo.append((True, i))
    c[c == 0] = 1.0
    np.testing.assert_allclose(
        g, r[:, None] * w * c[None, :], rtol=rtol, atol=atol, err_msg=err_msg
    )


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def load_hex_fixture(path: Path) -> dict:
    """A fixture file written by an R script's ``dumpl``: per line an array's
    name, its dimensions joined by commas and its values as hexadecimal
    floats (matrices column-major), R's ``NA`` as NaN; ``None`` where the
    line says "NULL", and the string after "CHR" where it holds one. A
    ``.gz`` file is read through gzip."""
    path = Path(path)
    text = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz"
        else path.read_text()
    )
    out: dict = {}
    for line in text.splitlines():
        parts = line.split()
        if parts[1] == "NULL":
            out[parts[0]] = None
            continue
        if parts[1] == "CHR":
            out[parts[0]] = parts[2]
            continue
        dims = [int(v) for v in parts[1].split(",")]
        vals = np.array(
            [math.nan if v == "NA" else float.fromhex(v) for v in parts[2:]],
            dtype=float,
        )
        out[parts[0]] = vals.reshape(dims, order="F") if len(dims) > 1 else vals
    return out


DATA_ROOT = Path(__file__).parent.parent / "datasets"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def _pinned_arithmetic() -> bool:
    import platform
    import sys

    import scipy

    from hea._dispatch import HAVE_RS
    from hea.sparse import build_info

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    if not HAVE_RS:
        # the reference-LAPACK ports (QR, Cholesky, the symmetric eigen
        # solvers) exist only in the extension
        return False
    deps = scipy.show_config(mode="dicts")["Build Dependencies"]
    return (
        deps["blas"]["name"].lower() == "accelerate"
        and build_info().get("backend") == "accelerate"
    )


# The R fixtures and exact-bit pins were written by R on darwin/arm64 with
# libRblas linked to Accelerate. Here hea issues the same BLAS calls to the
# same library and contracts a*b + c as that build does, so a result decided
# by last-bit rounding can equal R's only under this configuration: thin-plate
# Lanczos signs (which the null-space QR turns into a rotation of the whole
# wiggly basis) and the basis eigen() returns for a numerically zero
# eigenspace. Under other arithmetic those values move for R itself.
PINNED_ARITHMETIC = _pinned_arithmetic()


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
