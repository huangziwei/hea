"""Bit-exact parity for ``hea.R.clustering`` against R's ``stats``.

Committed pins (always run) plus a live-R differential (skipped without
``Rscript``). ``merge``/``order`` are integer encodings (exact on every
platform); ``height`` is strict 0-ulp on macOS (shared Apple libm) and a few-ulp
tolerance off macOS — same rationale as ``test_distance`` / ``test_rs_parity``.
On arm64 macOS R's gfortran fuses the Lance-Williams update to ``fmadd`` (hea
mirrors it per-arch via ``_rfma``), so fma-affected committed heights use a
small arm64 pin branch (``_HEIGHT_ARM64``); see that table.
"""

from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import pytest
from conftest import have_rscript

from hea.R.clustering import (
    Hclust,
    as_hclust,
    cophenetic,
    cutree,
    hclust,
    print_hclust,
)
from hea.R.distance import dist

_STRICT = sys.platform == "darwin"
_ARM64 = platform.machine().lower() in ("arm64", "aarch64")
_RTOL = 1e-13
_METHODS = [
    "ward.D",
    "single",
    "complete",
    "average",
    "mcquitty",
    "median",
    "centroid",
    "ward.D2",
]

# X <- matrix(c(0,0, 1,0, 0,1, 5,5, 6,5, 5,6), nrow=6, byrow=TRUE); d <- dist(X)
_X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=float)

# merge is column-major (R as.vector(h$merge)); height; order.
_PINS = {
    "ward.D": (
        [-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
        [1, 1, 1.2761423749153966, 1.2761423749153966, 19.077909746461739],
        [3, 1, 2, 6, 4, 5],
    ),
    "single": (
        [-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
        [1, 1, 1, 1, 6.4031242374328485],
        [3, 1, 2, 6, 4, 5],
    ),
    "complete": (
        [-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
        [1, 1, 1.4142135623730951, 1.4142135623730951, 7.810249675906654],
        [3, 1, 2, 6, 4, 5],
    ),
    "average": (
        [-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
        [1, 1, 1.2071067811865475, 1.2071067811865475, 7.1180173737923766],
        [3, 1, 2, 6, 4, 5],
    ),
    "mcquitty": (
        [-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
        [1, 1, 1.2071067811865475, 1.2071067811865475, 7.1194336759327044],
        [3, 1, 2, 6, 4, 5],
    ),
    "median": (
        [-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
        [1, 0.95710678118654746, 1, 0.95710678118654746, 6.3908802853394313],
        [3, 1, 2, 6, 4, 5],
    ),
    "centroid": (
        [-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
        [1, 0.95710678118654746, 1, 0.95710678118654746, 6.3593032488205781],
        [3, 1, 2, 6, 4, 5],
    ),
    "ward.D2": (
        [-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
        [1, 1, 1.2909944487358058, 1.2909944487358058, 12.247448713915889],
        [3, 1, 2, 6, 4, 5],
    ),
}


# R's Fortran hclust (`hclust.f`) fuses the Lance-Williams update
# `(membr*d + membr*d) ...` to a single `fmadd` on arm64 (gfortran's default
# contraction), where x86 keeps two roundings. So the committed (x86-captured)
# heights differ by <= a few ulp on arm64 for the multiply-add methods. hea
# mirrors R per-arch via ``_rfma`` (see hea/R/_shared.py); these overrides keep
# ``test_hclust_pins`` 0-ulp on arm64 too. The bit-exact arch-correct guarantee
# is ``test_hclust_vs_live_R`` (live R on whatever machine runs); only the
# always-run committed pin needs the branch. Only ward.D2 differs for ``_X``.
_HEIGHT_ARM64 = {
    "ward.D2": [1.0, 1.0, 1.2909944487358058, 1.2909944487358058, 12.24744871391589],
}


def _assert_height(got, exp):
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    if _STRICT:
        assert np.array_equal(got.view(np.int64), exp.view(np.int64)), (got, exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=_RTOL)


@pytest.mark.parametrize("method", _METHODS)
def test_hclust_pins(method):
    merge_exp, height_exp, order_exp = _PINS[method]
    if _ARM64 and method in _HEIGHT_ARM64:
        height_exp = _HEIGHT_ARM64[method]
    h = hclust(dist(_X), method=method)
    assert h.merge.ravel(order="F").tolist() == merge_exp
    assert h.order.tolist() == order_exp
    _assert_height(h.height, height_exp)
    assert h.method == method
    assert h.dist_method == "euclidean"


def test_ward_alias_warns_and_maps():
    with pytest.warns(UserWarning, match="renamed"):
        h = hclust(dist(_X), method="ward")
    assert h.method == "ward.D"


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="invalid clustering method"):
        hclust(dist(_X), method="nope")


def test_n_too_small_raises():
    with pytest.raises(ValueError, match="n >= 2"):
        hclust(dist(np.array([[1.0, 2.0]])))


def test_as_hclust_identity_and_error():
    h = hclust(dist(_X))
    assert as_hclust(h) is h
    with pytest.raises(TypeError, match="cannot be coerced"):
        as_hclust(object())


def test_print_hclust_summary():
    h = hclust(dist(_X), method="average")
    s = print_hclust(h, _return=True)
    assert "Cluster method   : average" in s
    assert "Distance         : euclidean" in s
    assert "Number of objects: 6" in s


def test_invalid_members_length():
    with pytest.raises(ValueError, match="invalid length of members"):
        hclust(dist(_X), members=np.ones(3))


def test_hclust_object_shapes():
    h = hclust(dist(_X))
    assert isinstance(h, Hclust)
    assert h.merge.shape == (5, 2)
    assert h.height.shape == (5,)
    assert h.order.shape == (6,)


# --------------------------------------------------------------------------- #
# cutree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "k,expected",
    [
        (2, [1, 1, 1, 2, 2, 2]),
        (3, [1, 1, 1, 2, 2, 3]),
        (1, [1, 1, 1, 1, 1, 1]),
        (6, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_cutree_by_k(k, expected):
    h = hclust(dist(_X), method="complete")
    assert cutree(h, k=k).tolist() == expected


@pytest.mark.parametrize(
    "hcut,expected",
    [
        (2, [1, 1, 1, 2, 2, 2]),
        (5, [1, 1, 1, 2, 2, 2]),
    ],
)
def test_cutree_by_h(hcut, expected):
    h = hclust(dist(_X), method="complete")
    assert cutree(h, h=hcut).tolist() == expected


def test_cutree_matrix_multi_k():
    h = hclust(dist(_X), method="complete")
    got = cutree(h, k=[2, 3, 4])
    assert got.shape == (6, 3)
    assert got.ravel(order="F").tolist() == [
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        3,
        1,
        1,
        2,
        3,
        3,
        4,
    ]


def test_cutree_matrix_multi_h():
    h = hclust(dist(_X), method="complete")
    got = cutree(h, h=[1.5, 3, 8])
    assert got.ravel(order="F").tolist() == [
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


def test_cutree_requires_k_or_h():
    with pytest.raises(ValueError, match="either 'k' or 'h'"):
        cutree(hclust(dist(_X)))


def test_cutree_k_out_of_range():
    with pytest.raises(ValueError, match="between 1 and"):
        cutree(hclust(dist(_X)), k=99)


def test_compose_dist_hclust_cutree():
    # compose gate: the full dist -> hclust -> cutree pipeline.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((20, 3))
    h = hclust(dist(x), method="average")
    labels = cutree(h, k=4)
    assert set(labels.tolist()) == {1, 2, 3, 4}


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("method", ["complete", "average", "ward.D2"])
def test_cutree_vs_live_R(method):
    rng = np.random.default_rng(42)
    x = rng.standard_normal((16, 4))
    d = dist(x)
    h = hclust(d, method=method)
    # build the same hclust in R, then cut by a range of k and h.
    elems = ",".join(float(v).hex() for v in d.data)
    rexpr = (
        f"d<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,"
        f'method="euclidean",class="dist");h<-hclust(d,method="{method}");'
        'cat(as.integer(cutree(h,k=c(2,5,8))),sep=" ");cat("\\n##\\n");'
        'cat(as.integer(cutree(h,h=stats::median(h$height))),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    ksec, hsec = out.split("\n##\n")
    kmat = np.array([int(v) for v in ksec.split()]).reshape(d.Size, 3, order="F")
    hvec = [int(v) for v in hsec.split()]
    assert np.array_equal(cutree(h, k=[2, 5, 8]), kmat)
    assert cutree(h, h=float(np.median(h.height))).tolist() == hvec


# --------------------------------------------------------------------------- #
# cophenetic
# --------------------------------------------------------------------------- #
_COPH_PINS = {
    "complete": [
        1,
        1.4142135623730951,
        7.810249675906654,
        7.810249675906654,
        7.810249675906654,
        1.4142135623730951,
        7.810249675906654,
        7.810249675906654,
        7.810249675906654,
        7.810249675906654,
        7.810249675906654,
        7.810249675906654,
        1,
        1.4142135623730951,
        1.4142135623730951,
    ],
    "single": [
        1,
        1,
        6.4031242374328485,
        6.4031242374328485,
        6.4031242374328485,
        1,
        6.4031242374328485,
        6.4031242374328485,
        6.4031242374328485,
        6.4031242374328485,
        6.4031242374328485,
        6.4031242374328485,
        1,
        1,
        1,
    ],
}


@pytest.mark.parametrize("method", list(_COPH_PINS))
def test_cophenetic_pins(method):
    h = hclust(dist(_X), method=method)
    _assert_height(cophenetic(h), _COPH_PINS[method])


def test_compose_dist_hclust_cophenetic():
    # compose gate: dist -> hclust -> cophenetic, round-tripped through R.
    rng = np.random.default_rng(11)
    x = rng.standard_normal((12, 3))
    h = hclust(dist(x), method="average")
    coph = cophenetic(h)
    assert coph.Size == 12
    assert len(coph) == 12 * 11 // 2


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("method", ["complete", "average", "centroid", "ward.D"])
def test_cophenetic_vs_live_R(method):
    rng = np.random.default_rng(123)
    x = rng.standard_normal((13, 4))
    d = dist(x)
    h = hclust(d, method=method)
    elems = ",".join(float(v).hex() for v in d.data)
    rexpr = (
        f"d<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,"
        f'method="euclidean",class="dist");h<-hclust(d,method="{method}");'
        'cat(sprintf("%.17g",as.vector(cophenetic(h))),sep="\\n")'
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
    _assert_height(cophenetic(h), expected)


# --------------------------------------------------------------------------- #
# live-R differential
# --------------------------------------------------------------------------- #
def _r_hclust(packed, n, method, members=None):
    """``stats::hclust`` on this machine; returns ``(merge, height, order)``.

    The packed lower-triangle vector is rebuilt into a ``"dist"`` object in R so
    the exact same dissimilarities are clustered.
    """
    elems = ",".join(float(v).hex() for v in packed)
    mem = (
        f"members=c({','.join(float(v).hex() for v in members)})"
        if members is not None
        else ""
    )
    rexpr = (
        f"d<-structure(c({elems}),Size={n}L,Diag=FALSE,Upper=FALSE,"
        f'method="euclidean",class="dist");'
        f'h<-hclust(d,method="{method}"{("," + mem) if mem else ""});'
        'cat(as.integer(h$merge),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",h$height),sep=" ");cat("\\n##\\n");'
        'cat(as.integer(h$order),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    msec, hsec, osec = out.split("\n##\n")
    merge = np.array([int(v) for v in msec.split()]).reshape(n - 1, 2, order="F")
    height = np.array([float(v) for v in hsec.split()])
    order = [int(v) for v in osec.split()]
    return merge, height, order


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("method", _METHODS)
def test_hclust_vs_live_R(method):
    rng = np.random.default_rng(31)
    x = rng.standard_normal((14, 4))
    d = dist(x)
    h = hclust(d, method=method)
    merge, height, order = _r_hclust(d.data, d.Size, method)
    assert np.array_equal(h.merge, merge)
    assert h.order.tolist() == order
    _assert_height(h.height, height)


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
def test_hclust_members_vs_live_R():
    rng = np.random.default_rng(99)
    x = rng.standard_normal((10, 3))
    d = dist(x)
    members = np.array([1.0, 2, 1, 3, 1, 2, 1, 1, 2, 1])
    h = hclust(d, method="centroid", members=members)
    merge, height, order = _r_hclust(d.data, d.Size, "centroid", members=members)
    assert np.array_equal(h.merge, merge)
    assert h.order.tolist() == order
    _assert_height(h.height, height)


# --------------------------------------------------------------------------- #
# Rust ``hclust`` kernel: A/B 0-ulp vs the pure-Python ``_hclust_fortran`` +
# ``_hcass2``. The Rust kernel does the agglomeration AND hcass2 and returns the
# final ``(merge_a, merge_b, height, order)``; the Python reference runs both
# stages. merge/order are integer-exact; height is 0-ulp (macOS) / tol (off).
# --------------------------------------------------------------------------- #
_rs_mod = pytest.importorskip("hea._rs")
_HAS_RS_HCLUST = hasattr(_rs_mod, "hclust")


def _py_hclust_cols(n, data, iopt, members):
    """Pure-Python reference: ``_hclust_fortran`` + ``_hcass2`` -> the same
    ``(merge_a, merge_b, height, order)`` columns the Rust kernel returns."""
    from hea.R.clustering import _hcass2, _hclust_fortran

    ia, ib, crit = _hclust_fortran(n, data, iopt, members)
    iorder, iia, iib = _hcass2(n, ia, ib)
    return (
        list(iia[1:n]),
        list(iib[1:n]),
        np.asarray(crit[1:n], dtype=float),
        list(iorder[1 : n + 1]),
    )


@pytest.mark.skipif(not _HAS_RS_HCLUST, reason="hea._rs.hclust not built")
@pytest.mark.parametrize("method", _METHODS)
def test_rs_hclust_matches_python(method):
    rng = np.random.default_rng(314)
    x = rng.standard_normal((22, 5))
    d = dist(x)
    data = np.ascontiguousarray(d.data, dtype=float)
    n = d.Size
    iopt = _METHODS.index(method) + 1
    members = np.ones(n)
    p_a, p_b, p_h, p_o = _py_hclust_cols(n, data, iopt, members)
    r_a, r_b, r_h, r_o = _rs_mod.hclust(n, data, iopt, members)
    assert list(r_a) == p_a
    assert list(r_b) == p_b
    assert list(r_o) == p_o
    _assert_height(np.asarray(r_h), p_h)


@pytest.mark.skipif(not hasattr(_rs_mod, "cutree"), reason="hea._rs.cutree not built")
def test_rs_cutree_matches_python():
    # A/B: Rust cutree (C_cutree port) vs pure-Python _cutree_c. Integer-exact.
    from hea.R.clustering import _cutree_c

    rng = np.random.default_rng(11)
    x = rng.standard_normal((60, 4))
    h = hclust(dist(x), method="average")
    merge = np.ascontiguousarray(h.merge, dtype=np.int64)
    which = np.array([2, 3, 5, 7, 12, 30, 60], dtype=np.int64)
    py = _cutree_c(merge, which)
    rs = np.asarray(_rs_mod.cutree(merge, which))
    assert np.array_equal(rs, py)


@pytest.mark.skipif(not _HAS_RS_HCLUST, reason="hea._rs.hclust not built")
def test_rs_hclust_members_matches_python():
    rng = np.random.default_rng(2718)
    x = rng.standard_normal((12, 3))
    d = dist(x)
    n = d.Size
    data = np.ascontiguousarray(d.data, dtype=float)
    members = np.array([1.0, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2])
    for method in ("ward.D", "centroid", "ward.D2"):
        iopt = _METHODS.index(method) + 1
        p_a, p_b, p_h, p_o = _py_hclust_cols(n, data, iopt, members)
        r_a, r_b, r_h, r_o = _rs_mod.hclust(n, data, iopt, members)
        assert list(r_a) == p_a
        assert list(r_b) == p_b
        assert list(r_o) == p_o
        _assert_height(np.asarray(r_h), p_h)
