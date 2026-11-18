"""The C library's math functions, as R's compiled code calls them.

R evaluates ``exp``, ``log``, ``^`` and the rest of its elementwise math with
the platform libm, one element at a time, on every machine. numpy's float64
ufuncs choose an implementation by the CPU they run on: on x86-64 its AVX2
and AVX-512 kernels for ``exp``, ``log``, ``power`` and others are not libm's
and round differently in the last bit, so the same fit would differ between
machines and from R. The functions here take numpy's calling forms and
compute float64 values with libm through ``hea._rs``; anything else (complex
or object input, other float widths, integer powers, array-likes with their
own ufunc handling such as polars Series or masked arrays) and every call
without the extension goes to numpy unchanged.

``libm_lgamma`` is ``lgamma`` (``math.lgamma`` is CPython's own approximation,
which differs from libm in the last bits), through the reentrant
``lgamma_r``, which returns the same value without writing the global
``signgam``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import operator

import numpy as np

from ._dispatch import rs_fn

_libm = ctypes.CDLL(ctypes.util.find_library("m") or None)
_lgamma_r = _libm.lgamma_r
_lgamma_r.restype = ctypes.c_double
_lgamma_r.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int)]


def libm_lgamma(x: float) -> float:
    """libm ``lgamma(x)``."""
    sign = ctypes.c_int(0)
    return _lgamma_r(float(x), ctypes.byref(sign))


_F64 = np.dtype(np.float64)
_SCALARS = (float, int, np.floating, np.integer, np.bool_, bool)


def _as_f64(x):
    """``x`` as a C-contiguous float64 array when numpy would compute it in
    float64 (float64 data, or integers of 32 bits and wider), else None:
    complex, object, narrower float or integer, boolean data, and objects
    other than ndarrays, lists, tuples and scalars go to numpy, whose own
    type rules and ``__array_ufunc__`` handling decide the result."""
    if type(x) is np.ndarray:
        a = x
    elif isinstance(x, (list, tuple, *_SCALARS)):
        a = np.asarray(x)
    else:
        return None
    dt = a.dtype
    if dt == _F64 or (dt.kind in "iu" and dt.itemsize >= 4):
        return np.asarray(a, dtype=np.float64, order="C")
    return None


def _shaped(out, a):
    return out[0] if a.ndim == 0 else out.reshape(a.shape)


def _unary(name: str, np_fn):
    vec, one = rs_fn(f"libm_{name}"), rs_fn(f"libm_{name}1")
    if vec is None or one is None:
        return np_fn

    def fn(x):
        tx = type(x)
        if tx is float or tx is np.float64:
            return np.float64(one(x))
        a = _as_f64(x)
        if a is None:
            return np_fn(x)
        return _shaped(vec(a), a)

    fn.__name__ = fn.__qualname__ = np_fn.__name__
    fn.__doc__ = (
        f"libm ``{name}`` elementwise, in the form of ``numpy.{np_fn.__name__}``."
    )
    return fn


def _pair(vec, x, y):
    """``vec`` over the broadcast float64 arrays of ``x`` and ``y``, or None
    when either is not data numpy would compute in float64."""
    a, b = _as_f64(x), _as_f64(y)
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        a, b = np.broadcast_arrays(a, b)
        a, b = np.asarray(a, order="C"), np.asarray(b, order="C")
    return _shaped(vec(a, b), a)


def _binary(name: str, np_fn):
    vec, one = rs_fn(f"libm_{name}"), rs_fn(f"libm_{name}1")
    if vec is None or one is None:
        return np_fn

    def fn(x, y):
        if (type(x) is float or type(x) is np.float64) and (
            type(y) is float or type(y) is np.float64
        ):
            return np.float64(one(x, y))
        out = _pair(vec, x, y)
        return np_fn(x, y) if out is None else out

    fn.__name__ = fn.__qualname__ = np_fn.__name__
    fn.__doc__ = (
        f"libm ``{name}`` elementwise, in the form of ``numpy.{np_fn.__name__}``."
    )
    return fn


exp = _unary("exp", np.exp)
log = _unary("log", np.log)
log1p = _unary("log1p", np.log1p)
expm1 = _unary("expm1", np.expm1)
log2 = _unary("log2", np.log2)
log10 = _unary("log10", np.log10)
sin = _unary("sin", np.sin)
cos = _unary("cos", np.cos)
tan = _unary("tan", np.tan)
sinh = _unary("sinh", np.sinh)
cosh = _unary("cosh", np.cosh)
tanh = _unary("tanh", np.tanh)
arctan = _unary("atan", np.arctan)
arccos = _unary("acos", np.arccos)
arcsin = _unary("asin", np.arcsin)
arcsinh = _unary("asinh", np.arcsinh)
arctan2 = _binary("atan2", np.arctan2)
hypot = _binary("hypot", np.hypot)

_pow_vec, _pow_one = rs_fn("libm_pow"), rs_fn("libm_pow1")
_pow_ys, _pow_xs = rs_fn("libm_pow_ys"), rs_fn("libm_pow_xs")

# numpy's ``power`` computes a single (scalar or broadcast) exponent of these
# values with an exact elementwise operation instead of ``pow``.
_POW_EXACT = {0.5: np.sqrt, 2.0: np.square, -1.0: np.reciprocal}


def _scalar_value(y):
    """The value of an exponent numpy treats as a single scalar - a real
    scalar, a 0-d array or an array broadcast from one value (every stride
    0) - or None."""
    if isinstance(y, _SCALARS):
        return y
    if (
        type(y) is np.ndarray
        and y.size
        and not any(y.strides)
        and y.dtype.kind in "fiub"
    ):
        return y.flat[0]
    return None


def _int_valued(v) -> bool:
    """Integer or boolean data, which numpy's ``power`` keeps integral."""
    if isinstance(v, (bool, int, np.integer, np.bool_)):
        return True
    return type(v) is np.ndarray and v.dtype.kind in "biu"


def _power(x, y):
    """``numpy.power`` with libm ``pow`` for each element of float64 data,
    after numpy's exact forms for a single exponent of 0.5, 2 or -1
    (``sqrt``, ``square``, ``reciprocal``); integer powers stay numpy's."""
    if _int_valued(x) and _int_valued(y):
        return np.power(x, y)
    e = _scalar_value(y)
    if e is not None and e in _POW_EXACT:
        a = _as_f64(x)
        if a is not None:
            return _POW_EXACT[float(e)](a)
    if (type(x) is float or type(x) is np.float64) and (
        type(y) is float or type(y) is np.float64
    ):
        return np.float64(_pow_one(x, y))
    if np.ndim(y) == 0 and e is not None:
        a = _as_f64(x)
        if a is not None and _as_f64(y) is not None:
            return _shaped(_pow_ys(a, float(e)), a)
    elif np.ndim(x) == 0 and _as_f64(x) is not None:
        b = _as_f64(y)
        if b is not None:
            return _shaped(_pow_xs(float(_as_f64(x)), b), b)
    out = _pair(_pow_vec, x, y)
    return np.power(x, y) if out is None else out


def _pow_op(x, y):
    """``x ** y``: Python's or numpy's own operator unless an operand is an
    ndarray of non-integer data, then ``_power``."""
    if type(x) is not np.ndarray and type(y) is not np.ndarray:
        return x**y
    if _int_valued(x) and _int_valued(y):
        return x**y
    return _power(x, y)


if None in (_pow_vec, _pow_one, _pow_ys, _pow_xs):
    power = np.power
    pow_op = operator.pow
else:
    power = _power
    power.__name__ = power.__qualname__ = "power"
    pow_op = _pow_op
