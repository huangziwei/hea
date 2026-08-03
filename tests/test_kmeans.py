"""Bit-exact parity for ``hea.R.clustering.kmeans`` against ``stats::kmeans``.

The algorithm itself (Hartigan-Wong / Lloyd / MacQueen) is bit-exact: with
explicit initial centres the C/Fortran kernels are ported 1:1, so ``cluster`` /
``centers`` / ``withinss`` / ``iter`` are 0-ulp on macOS (Apple libm), tolerance
off macOS.

The ``nstart`` *selection* criterion is ``sum(withinss)``, which R accumulates
in **long double** while numpy sums in double — so when two starts tie to within
~1 ulp, R and hea may pick different (equal-cost) starts. The ``nstart`` tests
therefore check that the RNG draws match R's ``sample.int`` stream and that the
achieved ``tot.withinss`` matches (tolerance), not that a near-tied partition is
reproduced. Same long-double-vs-double reason makes ``totss`` tolerance-bound.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import have_rscript

from hea.R import set_seed
from hea.R.clustering import Kmeans, fitted_kmeans, kmeans, print_kmeans

_STRICT = sys.platform == "darwin"
_ALGOS = ["Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"]


def _bits_equal(got, exp):
    got = np.asarray(got, float)
    exp = np.asarray(exp, float)
    return np.array_equal(got.view(np.int64), exp.view(np.int64))


def _assert_centers(got, exp):
    if _STRICT:
        assert _bits_equal(got, exp), (got, exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=1e-13)


# Three well-separated blobs; explicit centres make every run deterministic.
def _blobs():
    rng = np.random.default_rng(2026)
    return np.vstack(
        [
            rng.normal(0, 0.5, (8, 2)),
            rng.normal(5, 0.5, (8, 2)),
            rng.normal([0, 5], 0.5, (8, 2)),
        ]
    )


_CENTERS = np.array([[0.2, 0.1], [4.8, 5.1], [0.1, 4.9]])


def _r_kmeans_explicit(x, centers, algo):
    n, p = x.shape
    k = centers.shape[0]
    xs = ",".join(float(v).hex() for v in x.flatten(order="F"))
    cs = ",".join(float(v).hex() for v in centers.flatten(order="F"))
    rexpr = (
        f"x<-matrix(c({xs}),{n},{p});C<-matrix(c({cs}),{k},{p});"
        f'z<-kmeans(x,C,algorithm="{algo}");'
        'cat(z$cluster,sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",as.vector(z$centers)),sep=" ");cat("\\n##\\n");'
        'cat(sprintf("%.17g",z$withinss),sep=" ");cat("\\n##\\n");cat(z$iter)'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    cl, ce, ws, it = out.split("\n##\n")
    return (
        np.array([int(v) for v in cl.split()]),
        np.array([float(v) for v in ce.split()]).reshape(k, p, order="F"),
        np.array([float(v) for v in ws.split()]),
        int(it),
    )


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("algo", _ALGOS)
def test_kmeans_explicit_centers_vs_R(algo):
    x = _blobs()
    z = kmeans(x, _CENTERS, algorithm=algo)
    rcl, rce, rws, rit = _r_kmeans_explicit(x, _CENTERS, algo)
    assert np.array_equal(z.cluster, rcl)
    _assert_centers(z.centers, rce)
    if _STRICT:
        assert _bits_equal(z.withinss, rws)
    else:
        np.testing.assert_allclose(z.withinss, rws, rtol=1e-13)
    assert z.iter == rit


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize("algo", _ALGOS)
def test_kmeans_explicit_larger_vs_R(algo):
    rng = np.random.default_rng(55)
    x = np.vstack([rng.normal(c, 0.8, (12, 4)) for c in (-4, 0, 4, 8)])
    centers = np.array([x[0], x[15], x[28], x[40]])
    z = kmeans(x, centers, algorithm=algo, iter_max=50)
    rcl, rce, rws, rit = _r_kmeans_explicit(x, centers, algo)
    assert np.array_equal(z.cluster, rcl)
    _assert_centers(z.centers, rce)
    assert z.iter == rit


def test_totss_is_sum_of_within_plus_between():
    x = _blobs()
    z = kmeans(x, _CENTERS, algorithm="Lloyd")
    # identity that must hold regardless of long-double sum drift
    np.testing.assert_allclose(z.tot_withinss + z.betweenss, z.totss, rtol=1e-12)
    np.testing.assert_allclose(z.withinss.sum(), z.tot_withinss, rtol=1e-12)


# --------------------------------------------------------------------------- #
# nstart RNG seam: draws + achieved cost match R
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
@pytest.mark.parametrize(
    "algo,seed,k,nstart",
    [
        ("Hartigan-Wong", 42, 3, 5),
        ("Lloyd", 1, 4, 3),
        ("MacQueen", 99, 3, 4),
    ],
)
def test_kmeans_nstart_cost_vs_R(algo, seed, k, nstart):
    rng = np.random.default_rng(7)
    x = np.vstack(
        [
            rng.normal(0, 1, (15, 3)),
            rng.normal(4, 1, (15, 3)),
            rng.normal(-3, 1, (15, 3)),
        ]
    )
    n, p = x.shape
    set_seed(seed)
    z = kmeans(x, k, nstart=nstart, algorithm=algo)
    xs = ",".join(float(v).hex() for v in x.flatten(order="F"))
    rexpr = (
        f"set.seed({seed});x<-matrix(c({xs}),{n},{p});"
        f'z<-kmeans(x,{k},nstart={nstart},algorithm="{algo}");'
        'cat(sprintf("%.17g",z$tot.withinss))'
    )
    out = subprocess.run(
        ["Rscript", "-e", rexpr],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    # achieved within-cluster cost matches R (long-double vs double sum tolerance)
    np.testing.assert_allclose(z.tot_withinss, float(out), rtol=1e-12)


@pytest.mark.skipif(not have_rscript(), reason="Rscript not on PATH (install R)")
def test_kmeans_nstart_draws_match_R():
    # the RNG seam: hea's sample_int must reproduce R's sample.int(mm, k) stream.
    from hea.R.distributions import _r_rng

    set_seed(99)
    draws = np.concatenate([_r_rng().sample_int(45, 3) for _ in range(4)]) + 1
    out = subprocess.run(
        [
            "Rscript",
            "-e",
            "set.seed(99);cat(as.vector(replicate(4, sample.int(45,3))),sep=' ')",
        ],
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    r_draws = np.array([int(v) for v in out.split()])
    assert np.array_equal(draws, r_draws)


# --------------------------------------------------------------------------- #
# object behaviour, k==1, fitted/print, errors
# --------------------------------------------------------------------------- #
def test_kmeans_k1_uses_macqueen():
    # k == 1 forces nmeth=3 (HW Fortran needs k>1); one cluster = all points.
    x = _blobs()
    z = kmeans(x, np.array([[2.0, 2.0]]))
    assert isinstance(z, Kmeans)
    assert set(z.cluster.tolist()) == {1}
    assert z.size.tolist() == [len(x)]


def test_fitted_kmeans_centers_and_classes():
    x = _blobs()
    z = kmeans(x, _CENTERS, algorithm="Lloyd")
    fc = fitted_kmeans(z, "centers")
    assert fc.shape == x.shape
    np.testing.assert_array_equal(fc, z.centers[z.cluster - 1, :])
    assert np.array_equal(fitted_kmeans(z, "classes"), z.cluster)


def test_print_kmeans_summary():
    x = _blobs()
    z = kmeans(x, _CENTERS, algorithm="Lloyd")
    s = print_kmeans(z, _return=True)
    assert "K-means clustering with 3 clusters" in s
    assert "Cluster means:" in s
    assert "between_SS / total_SS" in s


def test_kmeans_duplicate_explicit_centers_raises():
    x = _blobs()
    with pytest.raises(ValueError, match="not distinct"):
        kmeans(x, np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]))


def test_kmeans_more_centers_than_points_raises():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="more cluster centers than data points"):
        kmeans(x, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))


def test_kmeans_bad_itermax_raises():
    x = _blobs()
    with pytest.raises(ValueError, match="must be positive"):
        kmeans(x, _CENTERS, iter_max=0)


def test_kmeans_column_mismatch_raises():
    x = _blobs()
    with pytest.raises(ValueError, match="same number of columns"):
        kmeans(x, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))


# --------------------------------------------------------------------------- #
# Rust ``kmns`` kernel: A/B 0-ulp vs the pure-Python ``_kmns`` (Hartigan-Wong).
# Both kernels are called directly (not via the ``_do_one`` seam), so the A/B
# runs in one process. ``kmns`` is pure IEEE arithmetic (no transcendentals), so
# rs == python is bit-for-bit on every platform.
# --------------------------------------------------------------------------- #
_rs_mod = pytest.importorskip("hea._rs")
_HAS_RS_KMNS = hasattr(_rs_mod, "kmns")


def _hw_fixture():
    rng = np.random.default_rng(2026)
    x = np.vstack(
        [
            rng.normal(0, 0.5, (15, 4)),
            rng.normal(5, 0.5, (15, 4)),
            rng.normal([0, 5, 0, 5], 0.5, (15, 4)),
            rng.normal(-4, 0.6, (15, 4)),
        ]
    )
    centers = np.array([x[0], x[16], x[31], x[46]])
    return x, centers, 4


@pytest.mark.skipif(not _HAS_RS_KMNS, reason="hea._rs.kmns not built")
@pytest.mark.parametrize("iter_max", [10, 50])
def test_rs_kmns_matches_python(iter_max):
    from hea.R.clustering import _kmns

    x, centers, k = _hw_fixture()
    py = _kmns(x, centers, k, iter_max)
    ifault, cluster, cen_flat, nc, wss, it = _rs_mod.kmns(
        np.ascontiguousarray(x), np.ascontiguousarray(centers), k, iter_max
    )
    assert int(ifault) == py["ifault"]
    assert np.array_equal(np.asarray(cluster), py["cluster"])
    assert np.array_equal(np.asarray(nc), py["nc"])
    assert int(it) == py["iter"]
    cen = np.asarray(cen_flat, dtype=float).reshape(k, x.shape[1])
    assert _bits_equal(cen, py["centers"])  # 0-ulp (pure arithmetic)
    assert _bits_equal(np.asarray(wss), py["wss"])


@pytest.mark.skipif(not _HAS_RS_KMNS, reason="hea._rs.kmns not built")
def test_rs_kmns_ifault3_k_out_of_range():
    # k <= 1 or k >= m -> ifault 3, empty arrays (matches _kmns early return).
    x = np.ascontiguousarray(_blobs())
    centers = np.ascontiguousarray(x[:1])
    ifault, cluster, cen_flat, nc, wss, it = _rs_mod.kmns(x, centers, 1, 10)
    assert int(ifault) == 3
    assert np.asarray(cluster).size == 0


@pytest.mark.skipif(not hasattr(_rs_mod, "lloyd"), reason="hea._rs.lloyd not built")
@pytest.mark.parametrize(
    "algo,rs_name,py_name",
    [
        ("Lloyd", "lloyd", "_kmeans_lloyd"),
        ("MacQueen", "macqueen", "_kmeans_macqueen"),
    ],
)
def test_rs_lloyd_macqueen_matches_python(algo, rs_name, py_name):
    # A/B: Rust Lloyd/MacQueen vs the pure-Python kernels. Pure IEEE arithmetic
    # (no transcendentals) -> 0-ulp on every platform.
    import hea.R.clustering as C

    x, centers, k = _hw_fixture()
    py_kernel = getattr(C, py_name)
    cl, cen, nc, wss, it = py_kernel(x, centers, k, 50)
    r_cl, r_cen, r_nc, r_wss, r_it = getattr(_rs_mod, rs_name)(
        np.ascontiguousarray(x), np.ascontiguousarray(centers), k, 50
    )
    assert np.array_equal(np.asarray(r_cl), cl)
    assert np.array_equal(np.asarray(r_nc), nc)
    assert int(r_it) == it
    assert _bits_equal(np.asarray(r_cen).reshape(k, x.shape[1]), cen)
    assert _bits_equal(np.asarray(r_wss), wss)
