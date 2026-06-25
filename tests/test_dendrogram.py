"""Bit-exact parity for the ``hea.R.clustering`` dendrogram subsystem against R's
``stats`` (port of ``cluster_dendrogram.R``, non-graphics surface).

Committed pins (always run) plus live-R differentials (skipped without
``Rscript``). The dendrogram is a tree of integers + float attributes:
``members``/``order``/``merge`` are integer encodings (exact on every platform);
``height``/``midpoint``/cophenetic values are strict 0-ulp on macOS (shared Apple
libm) and a tolerance off macOS — same rationale as ``test_clustering``.

``as.hclust(as.dendrogram(h))`` is **not** an identity on ``merge``: the reverse
coercion re-derives the merge order from a stable height sort, so equal-height
merges can swap. The oracle is therefore R's *own* round-trip, not the original
``hclust`` — verified to agree bit-for-bit below.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import have_rscript

from hea.R.clustering import (
    Dendrogram,
    as_dendrogram,
    as_hclust,
    cophenetic,
    cut_dendrogram,
    dendrapply,
    hclust,
    is_leaf,
    labels_dendrogram,
    merge_dendrogram,
    nleaves,
    nobs_dendrogram,
    order_dendrogram,
    print_dendrogram,
    reorder,
    rev_dendrogram,
    str_dendrogram,
)
from hea.R.clustering import _unlist
from hea.R.distance import dist

_STRICT = sys.platform == "darwin"
_RTOL = 1e-13

# Same canonical config as test_clustering.py (two well-separated triangles).
_X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5], [5, 6]], dtype=float)


def _assert_close(got, exp):
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    if _STRICT:
        assert np.array_equal(got.view(np.int64), exp.view(np.int64)), (got, exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=_RTOL)


def _dd_X():
    return as_dendrogram(hclust(dist(_X), method="complete"))


def _preorder_nodes(d, acc):
    """(members, midpoint, height) for each internal node, pre-order."""
    if d.children is None:
        return
    acc.append((d.attrs["members"], d.attrs["midpoint"], d.attrs["height"]))
    for c in d.children:
        _preorder_nodes(c, acc)


# --------------------------------------------------------------------------- #
# as.dendrogram: structure, members, midpoint, height
# --------------------------------------------------------------------------- #
def test_as_dendrogram_root_attrs():
    dd = _dd_X()
    assert isinstance(dd, Dendrogram)
    assert not is_leaf(dd)
    assert nobs_dendrogram(dd) == 6
    assert nleaves(dd) == 6
    assert len(dd) == 2  # binary root
    assert dd.attrs["midpoint"] == 2.25
    _assert_close([dd.attrs["height"]], [7.810249675906654])


def test_as_dendrogram_preorder_members_and_midpoints():
    dd = _dd_X()
    acc = []
    _preorder_nodes(dd, acc)
    members = [a[0] for a in acc]
    midpoints = [a[1] for a in acc]
    heights = [a[2] for a in acc]
    # pinned from R as.dendrogram(hclust(dist(X),"complete"))
    assert members == [6, 3, 2, 3, 2]
    assert midpoints == [2.25, 0.75, 0.5, 0.75, 0.5]
    _assert_close(heights, [7.810249675906654, 1.4142135623730951, 1.0,
                            1.4142135623730951, 1.0])


def test_order_and_labels_dendrogram():
    dd = _dd_X()
    assert order_dendrogram(dd).tolist() == [3, 1, 2, 6, 4, 5]
    # no labels on the hclust -> leaves are labelled 1..n
    assert labels_dendrogram(dd).tolist() == [3, 1, 2, 6, 4, 5]


def test_labels_dendrogram_string_labels():
    h = hclust(dist(_X), method="complete")
    h.labels = ["a", "b", "c", "d", "e", "f"]
    dd = as_dendrogram(h)
    # leaf order [3,1,2,6,4,5] -> labels c,a,b,f,d,e
    assert labels_dendrogram(dd).tolist() == ["c", "a", "b", "f", "d", "e"]


def test_leaf_is_leaf_and_int():
    dd = _dd_X()
    leaf = dd[(1, 1)]  # R dd[[1]][[1]] -> the leaf labelled 3
    assert is_leaf(leaf)
    assert int(leaf) == 3
    assert leaf.attrs["members"] == 1
    assert leaf.attrs["height"] == 0


# --------------------------------------------------------------------------- #
# as.hclust.dendrogram round-trip (vs R's own round-trip, not the original)
# --------------------------------------------------------------------------- #
def test_as_hclust_dendrogram_roundtrip_pins():
    dd = _dd_X()
    h2 = as_hclust(dd)
    assert h2.merge.ravel(order="F").tolist() == [
        -4, -1, -6, -3, 4, -5, -2, 1, 2, 3]
    assert h2.order.tolist() == [3, 1, 2, 6, 4, 5]
    _assert_close(h2.height, [1.0, 1.0, 1.4142135623730951,
                              1.4142135623730951, 7.810249675906654])
    assert h2.method is None  # NA in R


# --------------------------------------------------------------------------- #
# cut / reorder / rev / merge / cophenetic
# --------------------------------------------------------------------------- #
def test_cut_dendrogram_pins():
    dd = _dd_X()
    ct = cut_dendrogram(dd, 2.0)
    assert ct["upper"].attrs["members"] == 2
    assert order_dendrogram(ct["upper"]).tolist() == [1, 2]
    assert len(ct["lower"]) == 2
    assert [low.attrs["members"] for low in ct["lower"]] == [3, 3]
    # each lower branch is a valid sub-dendrogram
    assert nleaves(ct["lower"][0]) == 3
    # the cut nodes became "Branch k" leaves in upper
    assert [is_leaf(c) for c in ct["upper"].children] == [True, True]
    assert ct["upper"].children[0].attrs["label"] == "Branch 1"


def test_reorder_dendrogram_pins():
    dd = _dd_X()
    rr = reorder(dd, np.array([6.0, 5, 4, 3, 2, 1]))
    assert order_dendrogram(rr).tolist() == [6, 5, 4, 3, 2, 1]


def test_rev_dendrogram_pins():
    dd = _dd_X()
    rv = rev_dendrogram(dd)
    # reverse of [3,1,2,6,4,5]
    assert order_dendrogram(rv).tolist() == [5, 4, 6, 2, 1, 3]
    # rev is an involution on order
    assert order_dendrogram(rev_dendrogram(rv)).tolist() == [3, 1, 2, 6, 4, 5]


def test_merge_dendrogram_add_max_pins():
    # two 3-leaf trees with leaves 1..3 each -> add.max shifts the second by 3.
    xa = np.array([[0.0, 0], [0, 1], [1, 1]])
    xb = np.array([[0.0, 0], [5, 5], [6, 5]])
    da = as_dendrogram(hclust(dist(xa)))
    db = as_dendrogram(hclust(dist(xb)))
    m = merge_dendrogram(da, db)
    assert m.attrs["members"] == 6
    # leaves in tree order; db's 1..3 were shifted by max(da)=3 -> 4..6
    assert _unlist(m) == [3, 1, 2, 4, 5, 6]
    assert sorted(_unlist(m)) == [1, 2, 3, 4, 5, 6]
    # height defaults to 1.1 * max child height
    hmax = max(da.attrs["height"], db.attrs["height"])
    _assert_close([m.attrs["height"]], [1.1 * hmax])


def test_merge_dendrogram_height_too_small_raises():
    da = as_dendrogram(hclust(dist(_X[:3])))
    db = as_dendrogram(hclust(dist(_X[3:])))
    with pytest.raises(ValueError, match="must be at least"):
        merge_dendrogram(da, db, height=0.0)


def test_cophenetic_dendrogram_pins():
    h = hclust(dist(_X), method="complete")
    h.labels = ["a", "b", "c", "d", "e", "f"]
    dd = as_dendrogram(h)
    cd = cophenetic(dd)
    # leaf order is [c,a,b,f,d,e]; distances pinned from R cophenetic(dd)
    exp = [1.4142135623730951, 1.4142135623730951, 7.810249675906654,
           7.810249675906654, 7.810249675906654, 1.0, 7.810249675906654,
           7.810249675906654, 7.810249675906654, 7.810249675906654,
           7.810249675906654, 7.810249675906654, 1.4142135623730951,
           1.4142135623730951, 1.0]
    _assert_close(np.asarray(cd), exp)
    assert cd.Labels == ["c", "a", "b", "f", "d", "e"]


def test_cophenetic_dendrogram_needs_labels():
    # a leaf with no label -> error (cannot recover object names)
    leaf = Dendrogram(value=1, attrs={"leaf": True, "members": 1, "height": 0})
    root = Dendrogram(children=[leaf, leaf], attrs={"members": 2, "height": 1})
    with pytest.raises(ValueError, match="all leaves have labels"):
        cophenetic(root)


# --------------------------------------------------------------------------- #
# dendrapply / getitem / print / str / errors
# --------------------------------------------------------------------------- #
def test_dendrapply_visits_every_node():
    dd = _dd_X()
    seen = []

    def tag(node):
        seen.append(is_leaf(node))
        return node

    out = dendrapply(dd, tag)
    assert isinstance(out, Dendrogram)
    # 6 leaves + 5 internal nodes = 11 visits
    assert len(seen) == 11
    assert sum(seen) == 6  # six leaves


def test_dendrapply_can_transform_attrs():
    dd = _dd_X()

    def bump(node):
        if not is_leaf(node):
            node = Dendrogram(children=node.children, value=node.value,
                              attrs=dict(node.attrs))
            node.attrs["height"] = node.attrs["height"] + 100.0
        return node

    out = dendrapply(dd, bump)
    assert out.attrs["height"] == pytest.approx(107.810249675906654)
    # structure (order) preserved
    assert order_dendrogram(out).tolist() == [3, 1, 2, 6, 4, 5]


def test_getitem_recursive_1based():
    dd = _dd_X()
    # dd[[2]] is the right branch (members 3), dd[[2]][[2]] the {4,5} node
    assert dd[2].attrs["members"] == 3
    assert order_dendrogram(dd[(2, 2)]).tolist() == [4, 5]


def test_print_and_str_dendrogram_text():
    dd = _dd_X()
    s = print_dendrogram(dd, _return=True)
    assert "'dendrogram'" in s
    assert "6 members total" in s
    st = str_dendrogram(dd, _return=True)
    assert "dendrogram w/ 2 branches and 6 members" in st
    assert st.count("leaf") == 6


def test_as_dendrogram_identity():
    dd = _dd_X()
    assert as_dendrogram(dd) is dd


def test_order_dendrogram_type_error():
    with pytest.raises(TypeError, match="requires a dendrogram"):
        order_dendrogram(object())


# --------------------------------------------------------------------------- #
# live-R differential
# --------------------------------------------------------------------------- #
def _r_dend_diff(d, method):
    """Build the same hclust->dendrogram in R; return its accessors."""
    elems = ",".join(float(v).hex() for v in d.data)
    rexpr = (
        f'd<-structure(c({elems}),Size={d.Size}L,Diag=FALSE,Upper=FALSE,'
        f'method="euclidean",class="dist");h<-hclust(d,method="{method}");'
        'dd<-as.dendrogram(h);'
        'cat(order.dendrogram(dd),sep=" ");cat("\\n##\\n");'
        'h2<-as.hclust(dd);cat(as.vector(h2$merge),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",h2$height),sep=" ");cat("\\n##\\n");'
        'cat(h2$order,sep=" ");cat("\\n##\\n");'
        'ct<-cut(dd,h=stats::median(h$height));'
        'cat(attr(ct$upper,"members"),length(ct$lower),sep=" ");cat("\\n##\\n");'
        'cat(sapply(ct$lower,function(z)attr(z,"members")),sep=" ");cat("\\n##\\n");'
        'cat(order.dendrogram(reorder(dd,as.double(1:attr(dd,"members")))),sep=" ");'
        'cat("\\n##\\n");'
        'cat(sprintf("%.17g",as.vector(cophenetic(dd))),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
    ).stdout
    return out.split("\n##\n")


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("method", ["complete", "average", "ward.D2", "single"])
def test_dendrogram_vs_live_R(method):
    rng = np.random.default_rng(2026)
    x = rng.standard_normal((14, 3))
    d = dist(x)
    h = hclust(d, method=method)
    dd = as_dendrogram(h)
    (r_ord, r_merge, r_height, r_rtord, r_cut, r_lowmem,
     r_reord, r_coph) = _r_dend_diff(d, method)

    # order.dendrogram
    assert order_dendrogram(dd).tolist() == [int(v) for v in r_ord.split()]

    # as.hclust round-trip (merge col-major / height / order)
    h2 = as_hclust(dd)
    assert h2.merge.ravel(order="F").tolist() == [int(v) for v in r_merge.split()]
    _assert_close(h2.height, [float(v) for v in r_height.split()])
    assert h2.order.tolist() == [int(v) for v in r_rtord.split()]

    # cut at the median height
    ct = cut_dendrogram(dd, float(np.median(h.height)))
    up_mem, n_low = (int(v) for v in r_cut.split())
    assert ct["upper"].attrs["members"] == up_mem
    assert len(ct["lower"]) == n_low
    assert [low.attrs["members"] for low in ct["lower"]] == \
        [int(v) for v in r_lowmem.split()]

    # reorder by leaf index weights
    wts = np.arange(1, h.merge.shape[0] + 2, dtype=float)
    assert order_dendrogram(reorder(dd, wts)).tolist() == \
        [int(v) for v in r_reord.split()]

    # cophenetic.dendrogram
    _assert_close(np.asarray(cophenetic(dd)), [float(v) for v in r_coph.split()])


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
def test_merge_dendrogram_vs_live_R():
    rng = np.random.default_rng(7)
    xa = rng.standard_normal((5, 2))
    xb = rng.standard_normal((4, 2)) + 10
    da = as_dendrogram(hclust(dist(xa)))
    db = as_dendrogram(hclust(dist(xb)))
    m = merge_dendrogram(da, db)

    ea = ",".join(float(v).hex() for v in dist(xa).data)
    eb = ",".join(float(v).hex() for v in dist(xb).data)
    rexpr = (
        f'da<-as.dendrogram(hclust(structure(c({ea}),Size=5L,Diag=FALSE,'
        'Upper=FALSE,method="euclidean",class="dist")));'
        f'db<-as.dendrogram(hclust(structure(c({eb}),Size=4L,Diag=FALSE,'
        'Upper=FALSE,method="euclidean",class="dist")));'
        'm<-merge(da,db);'
        'cat(attr(m,"members"),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",attr(m,"height")));cat("\\n##\\n");'
        'cat(order.dendrogram(m),sep=" ");cat("\\n##\\n");cat(unlist(m),sep=" ")'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr], stdin=subprocess.DEVNULL, check=True,
        capture_output=True, text=True, timeout=120,
    ).stdout
    mem, hgt, ordr, unl = out.split("\n##\n")
    assert m.attrs["members"] == int(mem)
    _assert_close([m.attrs["height"]], [float(hgt)])
    assert order_dendrogram(m).tolist() == [int(v) for v in ordr.split()]
    assert _unlist(m) == [int(v) for v in unl.split()]
