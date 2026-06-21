"""Bit-exact parity for ``hea.R.clustering`` against R's ``stats``.

Committed pins (always run) plus a live-R differential (skipped without
``Rscript``). ``merge``/``order`` are integer encodings (exact on every
platform); ``height`` is strict 0-ulp on macOS (shared Apple libm) and a few-ulp
tolerance off macOS — same rationale as ``test_distance`` / ``test_rs_parity``.
"""
from __future__ import annotations

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
_RTOL = 1e-13
_METHODS = ["ward.D", "single", "complete", "average", "mcquitty",
            "median", "centroid", "ward.D2"]

# X <- matrix(c(0,0, 1,0, 0,1, 5,5, 6,5, 5,6), nrow=6, byrow=TRUE); d <- dist(X)
_X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=float)

# merge is column-major (R as.vector(h$merge)); height; order.
_PINS = {
    "ward.D": ([-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
               [1, 1, 1.2761423749153966, 1.2761423749153966, 19.077909746461739],
               [3, 1, 2, 6, 4, 5]),
    "single": ([-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
               [1, 1, 1, 1, 6.4031242374328485],
               [3, 1, 2, 6, 4, 5]),
    "complete": ([-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
                 [1, 1, 1.4142135623730951, 1.4142135623730951, 7.810249675906654],
                 [3, 1, 2, 6, 4, 5]),
    "average": ([-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
                [1, 1, 1.2071067811865475, 1.2071067811865475, 7.1180173737923766],
                [3, 1, 2, 6, 4, 5]),
    "mcquitty": ([-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
                 [1, 1, 1.2071067811865475, 1.2071067811865475, 7.1194336759327044],
                 [3, 1, 2, 6, 4, 5]),
    "median": ([-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
               [1, 0.95710678118654746, 1, 0.95710678118654746, 6.3908802853394313],
               [3, 1, 2, 6, 4, 5]),
    "centroid": ([-1, -3, -4, -6, 2, -2, 1, -5, 3, 4],
                 [1, 0.95710678118654746, 1, 0.95710678118654746, 6.3593032488205781],
                 [3, 1, 2, 6, 4, 5]),
    "ward.D2": ([-1, -4, -3, -6, 3, -2, -5, 1, 2, 4],
                [1, 1, 1.2909944487358058, 1.2909944487358058, 12.247448713915889],
                [3, 1, 2, 6, 4, 5]),
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
@pytest.mark.parametrize("k,expected", [
    (2, [1, 1, 1, 2, 2, 2]),
    (3, [1, 1, 1, 2, 2, 3]),
    (1, [1, 1, 1, 1, 1, 1]),
    (6, [1, 2, 3, 4, 5, 6]),
])
def test_cutree_by_k(k, expected):
    h = hclust(dist(_X), method="complete")
    assert cutree(h, k=k).tolist() == expected


@pytest.mark.parametrize("hcut,expected", [
    (2, [1, 1, 1, 2, 2, 2]),
    (5, [1, 1, 1, 2, 2, 2]),
])
def test_cutree_by_h(hcut, expected):
    h = hclust(dist(_X), method="complete")
    assert cutree(h, h=hcut).tolist() == expected


def test_cutree_matrix_multi_k():
    h = hclust(dist(_X), method="complete")
    got = cutree(h, k=[2, 3, 4])
    assert got.shape == (6, 3)
    assert got.ravel(order="F").tolist() == [
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 3, 1, 1, 2, 3, 3, 4]


def test_cutree_matrix_multi_h():
    h = hclust(dist(_X), method="complete")
    got = cutree(h, h=[1.5, 3, 8])
    assert got.ravel(order="F").tolist() == [
        1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1]


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
    elems = ",".join(repr(float(v)) for v in d.data)
    rexpr = (
        f'd<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,'
        f'method="euclidean",class="dist");h<-hclust(d,method="{method}");'
        'cat(as.integer(cutree(h,k=c(2,5,8))),sep=" ");cat("\\n##\\n");'
        'cat(as.integer(cutree(h,h=stats::median(h$height))),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
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
    "complete": [1, 1.4142135623730951, 7.810249675906654, 7.810249675906654,
                 7.810249675906654, 1.4142135623730951, 7.810249675906654,
                 7.810249675906654, 7.810249675906654, 7.810249675906654,
                 7.810249675906654, 7.810249675906654, 1, 1.4142135623730951,
                 1.4142135623730951],
    "single": [1, 1, 6.4031242374328485, 6.4031242374328485, 6.4031242374328485,
               1, 6.4031242374328485, 6.4031242374328485, 6.4031242374328485,
               6.4031242374328485, 6.4031242374328485, 6.4031242374328485, 1, 1, 1],
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
    elems = ",".join(repr(float(v)) for v in d.data)
    rexpr = (
        f'd<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,'
        f'method="euclidean",class="dist");h<-hclust(d,method="{method}");'
        'cat(sprintf("%.17g",as.vector(cophenetic(h))),sep="\\n")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
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
    elems = ",".join(repr(float(v)) for v in packed)
    mem = (f"members=c({','.join(repr(float(v)) for v in members)})"
           if members is not None else "")
    rexpr = (
        f'd<-structure(c({elems}),Size={n}L,Diag=FALSE,Upper=FALSE,'
        f'method="euclidean",class="dist");'
        f'h<-hclust(d,method="{method}"{("," + mem) if mem else ""});'
        'cat(as.integer(h$merge),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",h$height),sep=" ");cat("\\n##\\n");'
        'cat(as.integer(h$order),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
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
