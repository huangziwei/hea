"""``hea._libm``: the C library's math functions in numpy's calling forms.

With the compiled extension every float64 result must be the C library's own
value for that element (checked against the symbols called through ctypes),
on arrays of every size the extension handles, so a fit computes the same
numbers on every CPU. Without it the functions are numpy's. The call forms
follow numpy: scalars give numpy scalars, shapes are kept, integer powers
stay integral, and data numpy computes in another type goes to numpy.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np
import polars as pl
import pytest

from hea import _libm as libm
from hea._dispatch import HAVE_RS

_C = ctypes.CDLL(ctypes.util.find_library("m") or None)


def _c(name, nargs):
    f = getattr(_C, name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double] * nargs
    return f


_UNARY = {
    "exp": "exp",
    "log": "log",
    "log1p": "log1p",
    "expm1": "expm1",
    "log2": "log2",
    "log10": "log10",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "arctan": "atan",
    "arccos": "acos",
    "arcsin": "asin",
    "arcsinh": "asinh",
}
_BINARY = {"power": "pow", "arctan2": "atan2", "hypot": "hypot"}


def _inputs(n, seed):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [
            rng.normal(0.0, 4.0, n),
            rng.uniform(1e-8, 80.0, n),
            [0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 1e-310, 709.9, -745.2],
        ]
    )
    return x, rng.uniform(-3.0, 3.0, x.size)


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


needs_rs = pytest.mark.skipif(not HAVE_RS, reason="libm is reached through hea._rs")


@needs_rs
@pytest.mark.parametrize("n", [3, 5000, 40000])
@pytest.mark.parametrize("name", sorted(_UNARY))
def test_unary_is_the_c_library(name, n):
    x, _ = _inputs(n, 1)
    c = _c(_UNARY[name], 1)
    want = np.array([c(v) for v in x.tolist()])
    np.testing.assert_array_equal(_bits(getattr(libm, name)(x)), _bits(want))


@needs_rs
@pytest.mark.parametrize("n", [3, 5000, 40000])
@pytest.mark.parametrize("name", sorted(_BINARY))
def test_binary_is_the_c_library(name, n):
    x, y = _inputs(n, 2)
    if name == "power":
        x = np.abs(x)
    c = _c(_BINARY[name], 2)
    want = np.array([c(a, b) for a, b in zip(x.tolist(), y.tolist())])
    np.testing.assert_array_equal(_bits(getattr(libm, name)(x, y)), _bits(want))


@needs_rs
def test_power_with_one_exponent_or_base_is_the_c_library():
    x, y = _inputs(20000, 3)
    x = np.abs(x)
    c = _c("pow", 2)
    for e in (1.7, -2.5, 3.0, 1.0, 0.0):
        want = np.array([c(v, e) for v in x.tolist()])
        np.testing.assert_array_equal(_bits(libm.power(x, e)), _bits(want))
        np.testing.assert_array_equal(_bits(libm.pow_op(x, e)), _bits(want))
    want = np.array([c(2.5, v) for v in y.tolist()])
    np.testing.assert_array_equal(_bits(libm.pow_op(2.5, y)), _bits(want))


def test_power_keeps_numpys_exact_forms():
    # numpy computes a single exponent of 0.5, 2 or -1 as sqrt, square and
    # reciprocal, not pow; and keeps integer powers integral
    x = np.array([-0.0, -np.inf, 5.043676514210691, 1.4475147912467394, 3.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_array_equal(libm.power(x, 0.5), np.sqrt(x))
        np.testing.assert_array_equal(libm.power(x, 2.0), x * x)
        np.testing.assert_array_equal(libm.power(x, -1.0), 1.0 / x)
        np.testing.assert_array_equal(libm.pow_op(x, 2), x * x)
        np.testing.assert_array_equal(libm.pow_op(x, -1), 1.0 / x)
    k = np.arange(6)
    for got in (libm.power(k, 3), libm.pow_op(k, 3), libm.pow_op(k, k)):
        assert got.dtype.kind == "i"
    np.testing.assert_array_equal(libm.pow_op(k, 3), k**3)
    assert libm.pow_op(3, 4) == 81 and type(libm.pow_op(3, 4)) is int
    assert type(libm.pow_op(2.0, 0.5)) is float


def test_numpy_call_forms():
    x = np.linspace(0.5, 4.0, 12).reshape(3, 4)
    out = libm.log(x)
    assert out.shape == (3, 4) and out.dtype == np.float64
    np.testing.assert_allclose(libm.log(x.T), np.log(x.T), rtol=1e-15)
    assert type(libm.exp(1.5)) is np.float64
    assert type(libm.exp(np.array(1.5))) is np.float64
    assert type(libm.exp(2)) is np.float64
    assert isinstance(libm.exp([0.0, 1.0]), np.ndarray)
    np.testing.assert_allclose(libm.power(x, x[:1]), np.power(x, x[:1]), rtol=1e-15)
    assert libm.exp(np.float32(1.0)).dtype == np.float32
    assert libm.exp(np.array([1 + 1j])).dtype == np.complex128
    assert isinstance(libm.log(pl.Series("a", [1.0, 2.0])), pl.Series)
    m = np.ma.array([1.0, 2.0], mask=[False, True])
    assert isinstance(libm.log(m), np.ma.MaskedArray)


def test_hea_reaches_numpy_math_only_through_libm():
    # every numpy ufunc numpy dispatches by CPU (exp, log, power, ...) is
    # called through hea._libm, so no module computes them with numpy's
    # CPU-specific kernels
    import re
    from pathlib import Path

    import hea

    root = Path(hea.__file__).parent
    names = sorted(
        set(_UNARY) | set(_BINARY) | {"exp2", "cbrt", "arcsin", "arccosh", "arctanh"}
    )
    pat = re.compile(r"\b(?:np|numpy)\.(" + "|".join(names) + r")\b(?!\w)")
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == "_libm.py" or rel.startswith("translate/"):
            continue
        for k, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if pat.search(code):
                found.append(f"{rel}:{k}: {line.strip()}")
    assert not found, "numpy math outside hea._libm:\n" + "\n".join(found)
