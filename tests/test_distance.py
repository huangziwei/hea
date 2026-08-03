"""Bit-exact parity for ``hea.R.distance`` against R's ``stats``.

Two layers, mirroring the project idiom:

* **Committed pins** (always run): values captured from ``stats::dist`` /
  ``as.dist`` / ``as.matrix`` on fixed inputs, asserted bit-for-bit. Strict
  (0-ulp) on macOS where hea and R share Apple libm; a few-ulp tolerance off
  macOS (the glibc/numpy libm floor — same rationale as ``test_rs_parity``).
* **Live-R differential** (skipped without ``Rscript``): random matrices, all
  six metrics, compared to ``stats::dist`` on this machine.

The R snippet that produced the pins is in :func:`_r_dist` / the docstrings.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import have_rscript

from hea.R.distance import (
    Dist,
    as_dist,
    as_matrix_dist,
    cmdscale,
    dist,
    labels_dist,
    mahalanobis,
)

# minkowski is the only metric using ``pow``, and it is 0-ulp to R only when hea
# reproduces R's ``R_pow`` exactly: that needs the shared platform scalar libm
# (macOS: Apple libm) for the general path AND the ported integer special-cases
# (``x*x*x``/``x*x*x*x`` for ``|x|<=11``; per R's arithmetic.c). Off macOS
# the libm differs, so ``pow`` drifts a few ulp.
_STRICT = sys.platform == "darwin"
_RTOL = 1e-14


def _assert_eq(got, exp):
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    if _STRICT:
        gb = got.view(np.int64)
        eb = exp.view(np.int64)
        # NaN bit-patterns: treat any-NaN == any-NaN
        nan_g, nan_e = np.isnan(got), np.isnan(exp)
        assert np.array_equal(nan_g, nan_e), (got, exp)
        mask = ~nan_g
        assert np.array_equal(gb[mask], eb[mask]), (got, exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=_RTOL, equal_nan=True)


def _assert_tol(got, exp):
    """Tolerance compare for BLAS/LAPACK-bound results (mahalanobis): R and numpy
    use different BLAS builds, so ``solve``/matmul drift ~1 ulp even on one
    machine — not 0-ulp like the pure-arithmetic ``dist`` kernels."""
    np.testing.assert_allclose(
        np.asarray(got, float), np.asarray(exp, float), rtol=1e-12, equal_nan=True
    )


# --------------------------------------------------------------------------- #
# fixed inputs + R-captured pins
# --------------------------------------------------------------------------- #
# X1 <- matrix(c(1,2,3, 4,6,8, 1,0,0, 2,2,2), nrow=4, byrow=TRUE)
_X1 = np.array([[1, 2, 3], [4, 6, 8], [1, 0, 0], [2, 2, 2]], dtype=float)
# X2 <- matrix(c(1,2,NA, 4,6,8, 1,0,0, 2,NA,2), nrow=4, byrow=TRUE)
_X2 = np.array([[1, 2, np.nan], [4, 6, 8], [1, 0, 0], [2, np.nan, 2]], dtype=float)
# X3 <- matrix(c(1,0,1,0, 1,1,0,0, 0,0,1,1), nrow=3, byrow=TRUE)
_X3 = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)

# as.vector(dist(X1, method=...)) — packed lower triangle, column-major.
_PINS = {
    "euclidean": (
        _X1,
        "euclidean",
        2,
        [
            7.0710678118654755,
            3.6055512754639891,
            1.4142135623730951,
            10.440306508910551,
            7.4833147735478827,
            3,
        ],
    ),
    "maximum": (_X1, "maximum", 2, [5, 3, 1, 8, 6, 2]),
    "manhattan": (_X1, "manhattan", 2, [12, 5, 2, 17, 12, 5]),
    "canberra": (
        _X1,
        "canberra",
        2,
        [
            1.5545454545454547,
            2,
            0.53333333333333333,
            2.6000000000000001,
            1.4333333333333331,
            2.333333333333333,
        ],
    ),
    "binary": (
        _X1,
        "binary",
        2,
        [0, 0.66666666666666663, 0, 0.66666666666666663, 0, 0.66666666666666663],
    ),
    "minkowski_p3": (
        _X1,
        "minkowski",
        3,
        [
            5.9999999999999991,
            3.2710663101885897,
            1.2599210498948732,
            9.1057484912345714,
            6.6038544977892526,
            2.5712815906582351,
        ],
    ),
    "minkowski_p1.5": (
        _X1,
        "minkowski",
        1.5,
        [
            8.4071244231919309,
            4.0081889926881784,
            1.5874010519681994,
            12.182384301187385,
            8.6692457442272186,
            3.5387186276812526,
        ],
    ),
    "NA_euclidean": (
        _X2,
        "euclidean",
        2,
        [
            6.1237243569579451,
            2.4494897427831779,
            1.7320508075688772,
            10.440306508910551,
            7.745966692414834,
            2.7386127875258306,
        ],
    ),
    "NA_manhattan": (_X2, "manhattan", 2, [10.5, 3, 3, 17, 12, 4.5]),
    "NA_canberra": (
        _X2,
        "canberra",
        2,
        [1.6500000000000001, 1.5, 1, 2.6000000000000001, 1.4000000000000001, 2],
    ),
    "NA_binary": (_X2, "binary", 2, [0, 0.5, 0, 0.66666666666666663, 0, 0.5]),
    "NA_maximum": (_X2, "maximum", 2, [4, 2, 1, 8, 6, 2]),
    "binary3": (_X3, "binary", 2, [0.66666666666666663, 0.66666666666666663, 1]),
}


@pytest.mark.parametrize("name", list(_PINS))
def test_dist_pins(name):
    x, method, p, expected = _PINS[name]
    _assert_eq(dist(x, method=method, p=p), expected)


def test_as_dist_pin():
    # as.dist(matrix(c(0,2,3, 2,0,4, 3,4,0), 3, byrow=TRUE)) -> c(2,3,4)
    m = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]], dtype=float)
    _assert_eq(as_dist(m), [2, 3, 4])


def test_as_matrix_dist_pin():
    # as.matrix(dist(X1)) flattened column-major.
    expected = [
        0,
        7.0710678118654755,
        3.6055512754639891,
        1.4142135623730951,
        7.0710678118654755,
        0,
        10.440306508910551,
        7.4833147735478827,
        3.6055512754639891,
        10.440306508910551,
        0,
        3,
        1.4142135623730951,
        7.4833147735478827,
        3,
        0,
    ]
    _assert_eq(as_matrix_dist(dist(_X1, "euclidean")).ravel(order="F"), expected)


# --------------------------------------------------------------------------- #
# mahalanobis
# --------------------------------------------------------------------------- #
# x <- matrix(c(2,1, 3,4, 5,2, 1,1), nrow=4, byrow=TRUE)
# center <- c(2.5, 2); S <- matrix(c(2,0.5,0.5,1), 2, 2)
_MX = np.array([[2, 1], [3, 4], [5, 2], [1, 1]], dtype=float)
_MCENTER = np.array([2.5, 2.0])
_MCOV = np.array([[2, 0.5], [0.5, 1]], dtype=float)


def test_mahalanobis_pin():
    _assert_tol(
        mahalanobis(_MX, _MCENTER, _MCOV),
        [1, 4.1428571428571423, 3.5714285714285712, 1.5714285714285712],
    )


def test_mahalanobis_inverted_pin():
    # inverted=TRUE: cov is already the precision matrix.
    _assert_tol(
        mahalanobis(_MX, _MCENTER, np.linalg.inv(_MCOV), inverted=True),
        [1, 4.1428571428571423, 3.5714285714285712, 1.5714285714285712],
    )


def test_mahalanobis_center_false_pin():
    # center=FALSE -> R isFALSE(center): skip centering.
    _assert_tol(
        mahalanobis(_MX, False, _MCOV),
        [
            2.2857142857142856,
            16.571428571428569,
            13.142857142857142,
            1.1428571428571428,
        ],
    )


def test_mahalanobis_vector_input():
    # is.vector(x): treated as a single row.
    _assert_tol(
        mahalanobis(np.array([4.0, 3.0]), _MCENTER, _MCOV), [1.5714285714285712]
    )


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
def test_mahalanobis_vs_live_R():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10, 3))
    center = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    n, k = x.shape
    xs = ",".join(float(v).hex() for v in x.flatten(order="F"))
    cs = ",".join(float(v).hex() for v in center)
    ss = ",".join(float(v).hex() for v in cov.flatten(order="F"))
    rexpr = (
        f"x<-matrix(c({xs}),{n},{k});ctr<-c({cs});S<-matrix(c({ss}),{k},{k});"
        f'cat(sprintf("%.17g",as.double(mahalanobis(x,ctr,S))),sep="\\n")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    expected = np.array([float(s) for s in out.split()])
    _assert_tol(mahalanobis(x, center, cov), expected)


# --------------------------------------------------------------------------- #
# cmdscale (classical MDS) — points compared up to per-column sign
# --------------------------------------------------------------------------- #
def _assert_points_up_to_sign(got, exp, atol=1e-8):
    got = np.asarray(got, float)
    exp = np.asarray(exp, float)
    assert got.shape == exp.shape, (got.shape, exp.shape)
    for j in range(exp.shape[1]):
        ok = np.allclose(got[:, j], exp[:, j], atol=atol) or np.allclose(
            got[:, j], -exp[:, j], atol=atol
        )
        assert ok, f"column {j}: {got[:, j]} vs +/-{exp[:, j]}"


# 6-point config the cmdscale oracle is built from (two well-separated clusters).
_XC = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=float)
# cmdscale(dist(X), k=2) on the 6-point config.
_CMD_PTS = np.array(
    [
        -4.0069384267237682,
        -3.2998316455372239,
        -3.2998316455372234,
        3.0641293851417086,
        3.7712361663282552,
        3.7712361663282552,
        0,
        0.7071067811865458,
        -0.70710678118655224,
        -1.1591903465957709e-14,
        0.70710678118654446,
        -0.70710678118654113,
    ]
).reshape(6, 2, order="F")
_CMD_EIG = np.array(
    [
        75.666666666666686,
        1.9999999999999907,
        4.2632564145606011e-14,
        7.9936057773011271e-15,
        1.1899071437799712e-15,
        5.4161430964655891e-16,
    ]
)


def test_cmdscale_points_pin():
    _assert_points_up_to_sign(cmdscale(dist(_XC), k=2), _CMD_PTS)


def test_cmdscale_eig_and_gof():
    res = cmdscale(dist(_XC), k=2, eig=True)
    np.testing.assert_allclose(res["eig"], _CMD_EIG, atol=1e-8)
    np.testing.assert_allclose(res["GOF"], [0.99999999999999922] * 2, atol=1e-12)
    _assert_points_up_to_sign(res["points"], _CMD_PTS)


def test_cmdscale_from_matrix():
    m = as_matrix_dist(dist(_XC))
    _assert_points_up_to_sign(cmdscale(m, k=2), _CMD_PTS)


def test_cmdscale_add_constant():
    res = cmdscale(dist(_XC), k=2, add=True)
    _assert_points_up_to_sign(res["points"], _CMD_PTS, atol=1e-6)
    # _XC is perfectly Euclidean, so the additive constant is theoretically 0.
    # ``ac`` is ``max(Re(eigvals(Z)))`` (1:1 with R) of a matrix whose largest
    # eigenvalue is 0, so the computed value is pure ``dgeev`` noise and varies
    # by BLAS build: R/hea on Accelerate ~1e-15, OpenBLAS (Linux CI) ~1e-7. The
    # 1e-6 bound matches the sibling points tolerance and still separates "~0"
    # from a genuine (O(1)+) additive constant.
    assert abs(res["ac"]) < 1e-6


def test_cmdscale_k_out_of_range():
    with pytest.raises(ValueError, match=r"'k' must be in"):
        cmdscale(dist(_XC), k=6)  # n-1 == 5
    with pytest.raises(ValueError, match=r"'k' must be in"):
        cmdscale(dist(_XC), k=0)


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
def test_cmdscale_vs_live_R():
    rng = np.random.default_rng(2024)
    x = rng.standard_normal((10, 4))
    d = dist(x)
    res = cmdscale(d, k=3, eig=True)
    elems = ",".join(float(v).hex() for v in d.data)
    rexpr = (
        f"d<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,"
        f'method="euclidean",class="dist");r<-cmdscale(d,k=3,eig=TRUE);'
        'cat(sprintf("%.17g",as.vector(r$points)),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",r$eig),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    psec, esec = out.split("\n##\n")
    pts = np.array([float(v) for v in psec.split()]).reshape(d.Size, 3, order="F")
    eig = np.array([float(v) for v in esec.split()])
    _assert_points_up_to_sign(res["points"], pts, atol=1e-8)
    np.testing.assert_allclose(res["eig"], eig, atol=1e-8)


# --------------------------------------------------------------------------- #
# round-trips, object behaviour, errors
# --------------------------------------------------------------------------- #
def test_dist_object_is_vector_like():
    d = dist(_X1, "euclidean")
    assert isinstance(d, Dist)
    assert len(d) == 6
    assert d.Size == 4
    assert d.method == "euclidean"
    assert d.p is None
    assert np.asarray(d).shape == (6,)
    assert list(d) == list(d.data)


def test_minkowski_p_stored():
    d = dist(_X1, "minkowski", p=3)
    assert d.p == 3.0


def test_as_matrix_dist_roundtrip_as_dist():
    d = dist(_X1, "manhattan")
    m = as_matrix_dist(d)
    # as.dist of the expanded matrix returns the original packed vector.
    _assert_eq(as_dist(m), d.data)


def test_as_dist_passthrough_sets_diag_upper():
    d = dist(_X1, "euclidean")
    d2 = as_dist(d, diag=True, upper=True)
    assert d2.Diag is True and d2.Upper is True
    _assert_eq(d2, d.data)


def test_labels_dist_default_none():
    assert labels_dist(dist(_X1, "euclidean")) is None


def test_method_partial_match_and_alias():
    # "euclidian" misspelling alias, and prefix matching ("man" -> manhattan).
    _assert_eq(dist(_X1, "euclidian"), dist(_X1, "euclidean").data)
    _assert_eq(dist(_X1, "man"), dist(_X1, "manhattan").data)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="invalid distance method"):
        dist(_X1, "nope")


def test_invalid_minkowski_p_raises():
    with pytest.raises(ValueError, match="invalid p"):
        dist(_X1, "minkowski", p=0)
    with pytest.raises(ValueError, match="invalid p"):
        dist(_X1, "minkowski", p=-1)


def test_single_row_empty_dist():
    d = dist(np.array([[1.0, 2.0, 3.0]]), "euclidean")
    assert len(d) == 0
    assert d.Size == 1


# --------------------------------------------------------------------------- #
# live-R differential (random inputs, all metrics)
# --------------------------------------------------------------------------- #
def _r_dist(mat, method, p=2):
    """``as.vector(stats::dist(mat, method, p))`` via Rscript on this machine.

    The matrix is embedded as a ``matrix(c(...))`` literal using Python
    ``float.hex()`` per double (a C99 hex-float literal — bit-exact, unlike a
    decimal ``repr`` which this R build's ``strtod`` rounds by up to 1 ulp; that
    lossy transfer, not the kernel, was the arm64 ``*_vs_live_R`` failure). NaN
    -> R ``NA``; R prints the packed vector with ``%.17g`` (exact f64 round-trip).
    """
    n, k = mat.shape
    flat = mat.flatten(order="F")
    elems = ",".join("NA" if np.isnan(v) else float(v).hex() for v in flat)
    rexpr = (
        f"x<-matrix(c({elems}),{n},{k});"
        f'cat(sprintf("%.17g",as.double(as.vector('
        f'dist(x,method="{method}",p={p})))),sep="\\n")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return np.array([float(s) for s in out.split()]) if out.strip() else np.empty(0)


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize(
    "method", ["euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"]
)
@pytest.mark.parametrize("with_na", [False, True])
def test_dist_vs_live_R(method, with_na):
    rng = np.random.default_rng(20260621)
    x = rng.standard_normal((9, 5))
    # a couple of zeros so 'binary' has signal, and a sign mix for canberra
    x[0, 0] = 0.0
    x[3, 2] = 0.0
    if with_na:
        x[1, 1] = np.nan
        x[7, 4] = np.nan
    p = 1.7 if method == "minkowski" else 2
    _assert_eq(dist(x, method=method, p=p), _r_dist(x, method, p))


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("p", [1.5, 2.0, 3.0, 4.0, 5.0])
def test_minkowski_pow_vs_live_R(p):
    """``R_pow`` parity across both regimes. For integer ``p in {3,4}`` R uses the
    naive products ``x*x*x``/``x*x*x*x`` (up to 1 ulp off libm ``pow``) when
    ``|dev|<=11``, and falls back to ``pow`` for larger deviations; ``p in {2}``
    is ``x*x`` and non-integer/``p>=5`` is plain ``pow``. The deviations are
    non-integer so the products genuinely differ from ``pow``; large rows force
    the fallback. Caught nothing before because the only integer-p pin used
    integer-valued inputs (``x*x*x == pow`` exactly there)."""
    rng = np.random.default_rng(0xB16B00B5)
    x = np.vstack(
        [
            rng.standard_normal((6, 5)),  # |dev| <= 11  -> naive products
            rng.standard_normal((3, 5)) * 25.0,  # |dev| >  11  -> libm pow fallback
        ]
    )
    _assert_eq(dist(x, method="minkowski", p=p), _r_dist(x, "minkowski", p))


# --------------------------------------------------------------------------- #
# Rust ``cdist`` kernel: A/B 0-ulp vs the pure-Python ``_cdist`` oracle.
# The pure-Python kernel is the spec (pinned to R above); the Rust kernel must
# mirror it bit-for-bit. We call BOTH kernels directly (not via the import-time
# ``_rs_cdist`` seam) so the A/B runs in one process regardless of ``HEA_NO_RS``.
# --------------------------------------------------------------------------- #
_rs_mod = pytest.importorskip("hea._rs")
_HAS_CDIST = hasattr(_rs_mod, "cdist")


def _ab_inputs():
    rng = np.random.default_rng(424242)
    x = rng.standard_normal((40, 7))
    x[3, 2] = np.nan
    x[10, 0] = np.nan
    x[25, 5] = np.nan
    x[0, 0] = 0.0  # zeros for binary/canberra signal
    b = (rng.standard_normal((30, 6)) > 0.3).astype(float)
    b[2, 1] = np.nan
    return x, b


@pytest.mark.skipif(not _HAS_CDIST, reason="hea._rs.cdist not built")
@pytest.mark.parametrize(
    "mi,method,p",
    [
        (0, "euclidean", 2.0),
        (1, "maximum", 2.0),
        (2, "manhattan", 2.0),
        (3, "canberra", 2.0),
        (5, "minkowski", 1.7),
        (5, "minkowski", 3.0),
        (5, "minkowski", 4.0),
    ],
)
def test_rs_cdist_matches_python(mi, method, p):
    from hea.R.distance import _cdist

    x, _ = _ab_inputs()
    arr = np.ascontiguousarray(x, dtype=float)
    py = _cdist(arr, mi, p)
    rs = np.asarray(_rs_mod.cdist(arr, mi, float(p)))
    _assert_eq(rs, py)


@pytest.mark.skipif(not _HAS_CDIST, reason="hea._rs.cdist not built")
def test_rs_cdist_binary_matches_python():
    from hea.R.distance import _cdist

    _, b = _ab_inputs()
    arr = np.ascontiguousarray(b, dtype=float)
    py = _cdist(arr, 4, 2.0)
    rs = np.asarray(_rs_mod.cdist(arr, 4, 2.0))
    _assert_eq(rs, py)


@pytest.mark.skipif(not _HAS_CDIST, reason="hea._rs.cdist not built")
def test_rs_cdist_n1_empty():
    # single row -> no pairs -> empty vector (parity at the degenerate edge).
    arr = np.ascontiguousarray(np.array([[1.0, 2.0, 3.0]]))
    rs = np.asarray(_rs_mod.cdist(arr, 0, 2.0))
    assert rs.size == 0
