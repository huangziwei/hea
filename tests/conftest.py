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

# Tests must never pop GUI windows or leak figures. Force the non-interactive Agg
# backend (force=True: hea may have imported pyplot already during the import
# above), and the autouse `_close_figures` fixture closes every figure after each
# test so the plot tests don't accumulate past matplotlib's 20-open warning.
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def assert_fp_equiv(a, b):
    """Same-model assertion for two fits that share one code path.

    These equivalences are verified as diff-exactly-0 in R, and the fits
    are bit-identical on run-to-run-stable BLAS builds (Linux OpenBLAS,
    arm64 Accelerate). On macOS x86_64, Accelerate's ddot/dgemv kernels
    pick their reduction order from the 32-byte phase of heap addresses,
    so ANY two fits — even the same call repeated — drift at the last ulp
    (measured ≤3e-14 relative). rtol=1e-11 still pins code-path
    equivalence: a genuine intake/model split sits orders of magnitude
    above it."""
    np.testing.assert_allclose(np.asarray(a, dtype=float),
                               np.asarray(b, dtype=float), rtol=1e-11)


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
    # datasets/R/ mirrors R's built-in `datasets` package.
    return "R" if pkg == "datasets" else pkg


def _apply_schema(df: pl.DataFrame, pkg: str, name: str) -> pl.DataFrame:
    """Re-cast factor columns from the sidecar schema into pl.Enum.

    Used by ``load_dataset`` (above) and by ``test_smooths_predict`` to
    re-attach factor types to ``predict_data.csv`` fixtures, which lose them
    on CSV round-trip just like the source datasets do.
    """
    from hea.io import _apply_dataset_schema
    schema_path = DATA_ROOT / _pkg_subdir(pkg) / f"{name}.schema.json"
    return _apply_dataset_schema(df, schema_path)


def ordered_schema_cols(pkg: str, name: str) -> frozenset[str]:
    """Columns marked `ordered: true` in the dataset's schema sidecar.

    The ``ordered`` flag is plumbed separately from level order — pl.Enum
    carries levels for both ordered and unordered factors, so this is what
    drives `hea.formula.with_ordered_cols(...)` for poly contrasts.
    """
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

    Drops the ``rowname`` column (R's row.names preserved on the bundled-CSV
    side, ``rownames`` injected on the rdatasets side). All R-side fixtures
    were generated without it, so ``y ~ .`` expansions and column lists
    would mismatch otherwise. User-facing ``hea.data`` keeps the
    column — that's the whole point of preserving meaningful row names
    like the Galápagos island IDs in ``faraway::gala``.
    """
    from hea import data as _data
    key = (pkg, name)
    if key not in _data_cache:
        df = _data(name, _pkg_subdir(pkg))
        if "rowname" in df.columns:
            df = df.drop("rowname")
        _data_cache[key] = df
    # `hea.data` already registers ordered-factor columns globally, but the
    # autouse `_reset_ordered_cols` fixture clears them per-test. Re-register
    # here so the contextvar accumulates across multiple loads inside one test.
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


# ---------------------------------------------------------------------------
# glm() oracle loader — reads the JSON dumped by tests/scripts/make_glm_oracles.R.
# ---------------------------------------------------------------------------
GLM_ORACLE_ROOT = FIXTURE_ROOT / "glm"


def load_glm_oracle(name: str) -> dict:
    """Load a stats::glm() oracle by id (e.g. 'poisson_log_quine').

    Returns the parsed JSON as a dict; numeric scalars are floats, vectors
    are plain Python lists (test code converts to numpy as needed).
    """
    path = GLM_ORACLE_ROOT / name / "oracle.json"
    if not path.exists():
        raise FileNotFoundError(
            f"glm oracle {name!r} not found at {path}; "
            "regenerate via `Rscript tests/scripts/make_glm_oracles.R`"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Live-R nmath oracle — drives the hea._rs (Rust) bit-exact parity gate.
# tests/scripts/nmath_r_oracle.R evaluates R's d/p/q on the SAME machine, so
# Rust and R share that platform's scalar libm and agree 0-ulp everywhere.
# ---------------------------------------------------------------------------
_NMATH_R_ORACLE = Path(__file__).parent / "scripts" / "nmath_r_oracle.R"


def have_rscript() -> bool:
    return shutil.which("Rscript") is not None


def r_scalar_values(exprs):
    """Evaluate each scalar R expression on THIS machine; return ``{expr: float}``.

    ``sprintf("%.17g")`` round-trips an IEEE double exactly, so callers can compare
    bit-for-bit. Drives the macOS-only live-R bit-exact checks in test_R.py (same
    libm rationale as :func:`run_rs_r_oracle` — hea and R share the platform's
    scalar libm only on macOS; on glibc the few-ulp floor means callers must
    compare with tolerance instead)."""
    body = "".join(f'cat(sprintf("%.17g\\n", as.double({e})))\n' for e in exprs)
    out = subprocess.run(
        ["Rscript", "-e", body], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
    ).stdout
    return dict(zip(exprs, (float(x) for x in out.split())))


def run_rs_r_oracle(cases, workdir) -> dict:
    """Evaluate R's d/p/q for each case and return ``{name: np.ndarray}``.

    ``cases``: iterable of ``(name, fn, arrays, flags)`` where ``fn`` is the
    hea/rs kernel name, ``arrays`` are the inputs in hea's argument order, and
    ``flags`` are the trailing booleans (``lower_tail``/``log_p`` or
    ``give_log``; empty for no-flag kernels). Inputs are written as raw little-
    endian f64 so R sees the exact same bits the Rust side receives.
    """
    workdir = str(workdir)
    os.makedirs(workdir, exist_ok=True)
    spec_lines = []
    for name, fn, arrays, flags in cases:
        argfiles = []
        for i, a in enumerate(arrays):
            fname = f"{name}__{i}.bin"
            np.ascontiguousarray(a, dtype="<f8").ravel().tofile(
                os.path.join(workdir, fname))
            argfiles.append(fname)
        flagstr = ",".join("TRUE" if b else "FALSE" for b in flags)
        spec_lines.append(f"{name}|{fn}|{','.join(argfiles)}|{flagstr}")
    (Path(workdir) / "spec.txt").write_text("\n".join(spec_lines) + "\n")

    subprocess.run(
        ["Rscript", str(_NMATH_R_ORACLE), workdir],
        stdin=subprocess.DEVNULL, check=True, timeout=300,
        capture_output=True, text=True,
    )

    out = {}
    for name, fn, arrays, flags in cases:
        out[name] = np.fromfile(
            os.path.join(workdir, f"{name}.out.bin"), dtype="<f8")
    return out


# ---------------------------------------------------------------------------
# Live-R RNG oracle — drives the 3-way (Rust / pure-Python / R) parity gate for
# hea.R.rng (tests/test_rs_rng_parity.py). tests/scripts/rng_r_oracle.R draws
# from R's default Mersenne-Twister stream on the SAME machine.
# ---------------------------------------------------------------------------
_RNG_R_ORACLE = Path(__file__).parent / "scripts" / "rng_r_oracle.R"


def run_rng_r_oracle(seed, cases, workdir) -> dict:
    """Evaluate R's `set.seed(seed); <rcall>` for each case; return ``{name:
    np.ndarray}``.

    ``cases``: iterable of ``(name, rcall, params)`` where ``rcall`` is an R
    expression producing a numeric vector and ``params`` is a list of f64 arrays
    exposed to it as ``p0``, ``p1``, ... (written as raw little-endian f64 so R
    sees the exact bits the Rust/Python sides use). The seed is shared, so the
    three implementations draw the identical MT stream.
    """
    workdir = str(workdir)
    os.makedirs(workdir, exist_ok=True)
    spec_lines = [str(int(seed))]
    for case in cases:
        name, rcall, params = case[0], case[1], case[2]
        for i, a in enumerate(params):
            np.ascontiguousarray(a, dtype="<f8").ravel().tofile(
                os.path.join(workdir, f"{name}__{i}.bin"))
        spec_lines.append(f"{name}|{rcall}|{len(params)}")
    (Path(workdir) / "spec.txt").write_text("\n".join(spec_lines) + "\n")

    subprocess.run(
        ["Rscript", str(_RNG_R_ORACLE), workdir],
        stdin=subprocess.DEVNULL, check=True, timeout=300,
        capture_output=True, text=True,
    )

    return {case[0]: np.fromfile(os.path.join(workdir, f"{case[0]}.out.bin"),
                                 dtype="<f8")
            for case in cases}
