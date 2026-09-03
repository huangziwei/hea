"""Sparse symbolic parity — hea's Rust port vs SuiteSparse 7.6.0.

The port targets SuiteSparse **7.6.0** (AMD 3.3.1), the version R's ``Matrix``
ships and therefore the one ``lme4`` factorizes with. The references below were
produced by compiling upstream ``AMD/Source/{amd_2,amd_postorder,
amd_post_tree}.c`` at that tag and driving them exactly as
``CHOLMOD/Cholesky/cholmod_amd.c`` does — build ``C = A+A'`` (pattern, no
diagonal, ``mode = -2`` elbow room) and call ``amd_2`` directly, with
``Control = NULL`` so AMD's own defaults apply, which is also what CHOLMOD
passes (``prune_dense = 10.0``, ``aggressive = TRUE``).

Do **not** re-pin these from ``scikit-sparse``'s ``F.perm``. That is a
*different quantity*: ``cholmod_analyze`` composes the fill-reducing ordering
with a weighted etree postorder before storing it (``cholmod_analyze.c:832-845``,
``Lperm[k] = Lperm[Post[k]]``). The two agree only when the postorder happens to
be the identity — true for banded/tridiagonal/arrow and for every ``gmm`` model
tested, false for 5 of the 17 matrices in the original sweep. AMD itself is
byte-identical between 7.6.0 and 7.12.2, so the divergence is the postorder, not
a version gap.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

import hea.sparse
from hea import _rs


@dataclass(frozen=True)
class _Ref:
    n: int
    head: list[int]
    sha: str
    lnz: float
    ndiv: float
    nms_ldl: float
    nms_lu: float
    ndense: float
    dmax: float
    ncmpa: float


REF: dict[str, _Ref] = {
    "banded-50-2": _Ref(
        n=50,
        head=[49, 48, 47, 46, 45, 44, 43, 42],
        sha="5b5995cd8af354d2",
        lnz=97,
        ndiv=97,
        nms_ldl=145,
        nms_lu=193,
        ndense=0,
        dmax=3,
        ncmpa=0,
    ),
    "banded-200-3": _Ref(
        n=200,
        head=[199, 198, 197, 196, 195, 194, 193, 192],
        sha="fa55878c26c425f1",
        lnz=595,
        ndiv=595,
        nms_ldl=1190,
        nms_lu=1785,
        ndense=0,
        dmax=5,
        ncmpa=0,
    ),
    "random-60": _Ref(
        n=60,
        head=[9, 18, 0, 44, 35, 10, 11, 7],
        sha="39de3234cc133dc7",
        lnz=764,
        ndiv=764,
        nms_ldl=7025,
        nms_lu=13286,
        ndense=0,
        dmax=30,
        ncmpa=0,
    ),
    "random-300": _Ref(
        n=300,
        head=[18, 12, 120, 15, 137, 292, 165, 40],
        sha="78ac4bae623fdd76",
        lnz=23241,
        ndiv=23241,
        nms_ldl=1446009,
        nms_lu=2868777,
        ndense=0,
        dmax=199,
        ncmpa=0,
    ),
    "random-400": _Ref(
        n=400,
        head=[26, 56, 91, 97, 24, 190, 146, 90],
        sha="9907959c7de319e8",
        lnz=36546,
        ndiv=36546,
        nms_ldl=2869581,
        nms_lu=5702616,
        ndense=0,
        dmax=251,
        ncmpa=1,
    ),
    "block-diagonal": _Ref(
        n=55,
        head=[0, 5, 4, 1, 2, 6, 3, 8],
        sha="2366db6163d89381",
        lnz=235,
        ndiv=235,
        nms_ldl=947,
        nms_lu=1659,
        ndense=0,
        dmax=14,
        ncmpa=0,
    ),
    "arrow-300": _Ref(
        n=300,
        head=[1, 2, 3, 4, 5, 6, 7, 8],
        sha="0f35f3f8609dfdc2",
        lnz=299,
        ndiv=299,
        nms_ldl=299,
        nms_lu=299,
        ndense=1,
        dmax=2,
        ncmpa=0,
    ),
    "tridiagonal-200": _Ref(
        n=200,
        head=[199, 198, 197, 196, 195, 194, 193, 192],
        sha="5385aa30ea848768",
        lnz=199,
        ndiv=199,
        nms_ldl=199,
        nms_lu=199,
        ndense=0,
        dmax=2,
        ncmpa=0,
    ),
    "diagonal-32": _Ref(
        n=32,
        head=[0, 1, 2, 3, 4, 5, 6, 7],
        sha="bcc9bcfc670935c6",
        lnz=0,
        ndiv=0,
        nms_ldl=0,
        nms_lu=0,
        ndense=0,
        dmax=1,
        ncmpa=0,
    ),
    "kron-duplicate-rows-120": _Ref(
        n=120,
        head=[39, 38, 37, 36, 43, 42, 41, 40],
        sha="c07a57c208523bd2",
        lnz=6488,
        ndiv=6488,
        nms_ldl=228416,
        nms_lu=450344,
        ndense=0,
        dmax=104,
        ncmpa=0,
    ),
}


def corpus() -> list[tuple[str, sp.csc_array]]:
    """The matrices the references above were measured on. Every construction
    is seeded, so the corpus is reproducible; changing it invalidates ``REF``."""
    rng = np.random.default_rng(0)
    out: list[tuple[str, sp.csc_array]] = []

    for n, bw in [(50, 2), (200, 3)]:
        d = [np.full(n - k, 1.0 / (k + 1)) for k in range(bw + 1)]
        A = sp.diags_array(d, offsets=list(range(bw + 1)))
        A = (A + A.T).tocsc() + sp.eye_array(n) * (2 * bw + 2)
        out.append((f"banded-{n}-{bw}", sp.csc_array(A)))

    for n, dens in [(60, 0.08), (300, 0.03), (400, 0.02)]:
        A = sp.random_array((n, n), density=dens, rng=rng)
        A = (A + A.T).tocsc()
        A = (A + sp.eye_array(n) * (abs(A).sum(axis=0).max() + 1.0)).tocsc()
        out.append((f"random-{n}", sp.csc_array(A)))

    blocks = [sp.random_array((k, k), density=0.3, rng=rng) for k in (7, 13, 5, 21, 9)]
    blocks = [(b + b.T + sp.eye_array(b.shape[0]) * 10).tocsc() for b in blocks]
    out.append(("block-diagonal", sp.csc_array(sp.block_diag(blocks).tocsc())))

    n = 300
    A = sp.lil_array((n, n))
    A.setdiag(4.0)
    A[0, :] = 1.0
    A[:, 0] = 1.0
    A[0, 0] = float(n + 4)
    out.append(("arrow-300", sp.csc_array(A.tocsc())))

    n = 200
    A = sp.diags_array(
        [np.full(n, 4.0), np.full(n - 1, -1.0), np.full(n - 1, -1.0)],
        offsets=[0, 1, -1],
    )
    out.append(("tridiagonal-200", sp.csc_array(A.tocsc())))

    out.append(("diagonal-32", sp.csc_array(sp.eye_array(32).tocsc())))

    n = 120
    base = rng.integers(0, 2, size=(n // 4, n // 4))
    base = np.kron(base + base.T, np.ones((4, 4), dtype=int))
    A = (base > 0).astype(float)
    np.fill_diagonal(A, float(n))
    out.append(("kron-duplicate-rows-120", sp.csc_array(sp.csc_matrix(A))))

    return out


def amd_order(M, stype: int = 1, **kw):
    M = sp.csc_array(M)
    perm, info = _rs.amd_order(
        M.shape[0], M.indptr.astype(np.int64), M.indices.astype(np.int64), stype, **kw
    )
    return np.asarray(perm), info


CORPUS = corpus()


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("use_long", [False, True])
def test_amd_matches_suitesparse_760(name, M, use_long):
    """Permutation and every ``Info`` statistic, bit-exact to C AMD 3.3.1."""
    ref = REF[name]
    perm, info = amd_order(M, use_long=use_long)

    assert perm.shape == (ref.n,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(ref.n))
    np.testing.assert_array_equal(perm[:8], ref.head)
    assert hashlib.sha256(perm.tobytes()).hexdigest()[:16] == ref.sha

    assert info["AMD_LNZ"] == ref.lnz
    assert info["AMD_NDIV"] == ref.ndiv
    assert info["AMD_NMULTSUBS_LDL"] == ref.nms_ldl
    assert info["AMD_NMULTSUBS_LU"] == ref.nms_lu
    assert info["AMD_NDENSE"] == ref.ndense
    assert info["AMD_DMAX"] == ref.dmax
    assert info["AMD_NCMPA"] == ref.ncmpa
    # the two derived quantities cholmod_amd.c:177-180 hands back to Common
    assert info["lnz"] == ref.n + ref.lnz
    assert info["fl"] == ref.ndiv + 2.0 * ref.nms_ldl + ref.n


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_amd_stype_sign_is_immaterial_for_a_symmetric_pattern(name, M):
    """``stype`` picks which triangle is the stored half. Both readings of a
    fully-stored symmetric matrix build the same ``C = A+A'``, and CHOLMOD's
    construction leaves each column of ``C`` sorted either way, so the ordering
    cannot depend on the sign."""
    np.testing.assert_array_equal(amd_order(M, stype=1)[0], amd_order(M, stype=-1)[0])


def test_amd_garbage_collection_actually_fires_somewhere():
    """AMD compresses ``Iw`` in place when it runs out of elbow room
    (``amd_2.c:870-938``) — a branch worth ~70 lines that no other case here
    reaches. Guard against the corpus silently losing its only cover."""
    assert any(REF[name].ncmpa > 0 for name, _ in CORPUS)


def test_amd_dense_row_detection_actually_fires_somewhere():
    """Likewise for the ``deg > dense`` branch (``amd_2.c:660-677``), which
    parks a row as a non-principal variable with no parent and orders it last."""
    assert any(REF[name].ndense > 0 for name, _ in CORPUS)


def test_amd_empty_and_singleton():
    assert amd_order(sp.csc_array((0, 0)))[0].shape == (0,)
    np.testing.assert_array_equal(amd_order(sp.csc_array(sp.eye_array(1)))[0], [0])


def test_amd_rejects_bad_input():
    """The port indexes its workspaces without a per-access bounds check, so
    this validation is what stands between a malformed pattern and an
    out-of-bounds write. It runs on the arrays as they arrive from Python,
    which is the only place they are untrusted."""
    M = sp.csc_array(sp.eye_array(5).tocsc())
    indptr = M.indptr.astype(np.int64)
    indices = M.indices.astype(np.int64)

    with pytest.raises(ValueError, match="stype must be nonzero"):
        amd_order(M, stype=0)
    with pytest.raises(ValueError, match="indptr has length"):
        _rs.amd_order(6, indptr, indices, 1)

    bad = indices.copy()
    bad[0] = 99
    with pytest.raises(ValueError, match="out of range"):
        _rs.amd_order(5, indptr, bad, 1)
    bad[0] = -1
    with pytest.raises(ValueError, match="out of range"):
        _rs.amd_order(5, indptr, bad, 1)

    bad = indptr.copy()
    bad[2] = 0
    with pytest.raises(ValueError, match="non-decreasing"):
        _rs.amd_order(5, bad, indices, 1)
    bad = indptr.copy()
    bad[0] = -1
    with pytest.raises(ValueError, match="non-decreasing"):
        _rs.amd_order(5, bad, indices, 1)
    bad = indptr.copy()
    bad[5] = 99
    with pytest.raises(ValueError, match="out of range for 5 row indices"):
        _rs.amd_order(5, bad, indices, 1)


def test_amd_ignores_the_unstored_triangle():
    """CHOLMOD's ``cholmod_copy`` *copies* one triangle into both halves rather
    than adding the two — entries in the ignored half are dropped, not folded
    in (``t_cholmod_copy_worker.c:78-105``). So perturbing only the lower half
    must not move a ``stype=+1`` ordering."""
    n = 40
    rng = np.random.default_rng(7)
    A = sp.random_array((n, n), density=0.06, rng=rng)
    A = sp.csc_array((sp.triu(A, 1) + sp.eye_array(n) * 5.0).tocsc())
    base = amd_order(A, stype=1)[0]

    extra = sp.csc_array(sp.tril(sp.random_array((n, n), density=0.1, rng=rng), -1))
    perturbed = sp.csc_array((A + extra).tocsc())
    np.testing.assert_array_equal(amd_order(perturbed, stype=1)[0], base)


def test_amd_is_deterministic():
    _, M = CORPUS[3]
    first = amd_order(M)[0]
    for _ in range(3):
        np.testing.assert_array_equal(amd_order(M)[0], first)


@dataclass(frozen=True)
class _RefAnalyze:
    n: int
    head: list[int]
    perm_sha: str
    colcount_sha: str
    lnz: float
    fl: float
    anz: float
    default_ordering: int


REF_ANALYZE: dict[str, _RefAnalyze] = {
    "banded-50-2": _RefAnalyze(
        n=50,
        head=[49, 48, 47, 46, 45, 44, 43, 42],
        perm_sha="5b5995cd8af354d2",
        colcount_sha="865156499aecbf3d",
        lnz=147.0,
        fl=437.0,
        anz=147.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "banded-200-3": _RefAnalyze(
        n=200,
        head=[199, 198, 197, 196, 195, 194, 193, 192],
        perm_sha="fa55878c26c425f1",
        colcount_sha="4c586cf3222672ae",
        lnz=794.0,
        fl=3166.0,
        anz=794.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "random-60": _RefAnalyze(
        n=60,
        head=[11, 41, 46, 9, 18, 0, 44, 35],
        perm_sha="567be1e4d0ed6b6e",
        colcount_sha="c0b682d588e71da7",
        lnz=824.0,
        fl=14874.0,
        anz=336.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "random-300": _RefAnalyze(
        n=300,
        head=[195, 283, 67, 80, 237, 258, 101, 253],
        perm_sha="e6107fb72caac2e5",
        colcount_sha="92aea86e3f8a50b6",
        lnz=23541.0,
        fl=2915559.0,
        anz=2951.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "random-400": _RefAnalyze(
        n=400,
        head=[56, 91, 295, 245, 287, 332, 383, 97],
        perm_sha="ca152a6905c228eb",
        colcount_sha="fc3a5493740ce535",
        lnz=36946.0,
        fl=5776108.0,
        anz=3544.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "block-diagonal": _RefAnalyze(
        n=55,
        head=[5, 4, 0, 1, 2, 6, 3, 18],
        perm_sha="ff67a2c26e8233af",
        colcount_sha="6f6a0e55b43fff0e",
        lnz=289.0,
        fl=2157.0,
        anz=234.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "arrow-300": _RefAnalyze(
        n=300,
        head=[1, 2, 3, 4, 5, 6, 7, 8],
        perm_sha="0f35f3f8609dfdc2",
        colcount_sha="5a05756d08f2e75e",
        lnz=599.0,
        fl=1197.0,
        anz=599.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "tridiagonal-200": _RefAnalyze(
        n=200,
        head=[199, 198, 197, 196, 195, 194, 193, 192],
        perm_sha="5385aa30ea848768",
        colcount_sha="643b7fb9515a8a78",
        lnz=399.0,
        fl=797.0,
        anz=399.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "diagonal-32": _RefAnalyze(
        n=32,
        head=[0, 1, 2, 3, 4, 5, 6, 7],
        perm_sha="bcc9bcfc670935c6",
        colcount_sha="6ba64591dc5d5fa6",
        lnz=32.0,
        fl=32.0,
        anz=32.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "kron-duplicate-rows-120": _RefAnalyze(
        n=120,
        head=[52, 53, 54, 55, 84, 85, 86, 87],
        perm_sha="dca8001383725e4e",
        colcount_sha="5ee6d99e18ede87a",
        lnz=6544.0,
        fl=451344.0,
        anz=5350.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "laplacian3d-23": _RefAnalyze(
        n=12167,
        head=[8750, 5416, 8562, 7968, 7990, 7460, 5852, 6382],
        perm_sha="a2311e8879e91519",
        colcount_sha="397ac515c1b90c0b",
        lnz=1523945.0,
        fl=716240133.0,
        anz=47081.0,
        default_ordering=2,  # 2 = AMD, 3 = METIS
    ),
    "laplacian3d-24": _RefAnalyze(
        n=13824,
        head=[4932, 3205, 6759, 6757, 7357, 7359, 6230, 6782],
        perm_sha="8f5d6ea77ecb78fc",
        colcount_sha="c7bfdc80bb3737c4",
        lnz=1874559.0,
        fl=969172461.0,
        anz=53568.0,
        default_ordering=3,  # 2 = AMD, 3 = METIS
    ),
}


def laplacian3d(m: int) -> sp.csc_array:
    """7-point Laplacian on an ``m**3`` grid — the shape where nested
    dissection beats minimum degree, and the only one in this file that makes
    CHOLMOD's default strategy go past AMD."""
    n = m * m * m
    idx = np.arange(n).reshape(m, m, m)
    r: list[np.ndarray] = []
    c: list[np.ndarray] = []
    for ax in range(3):
        a = np.take(idx, range(m - 1), axis=ax).ravel()
        b = np.take(idx, range(1, m), axis=ax).ravel()
        r += [a, b]
        c += [b, a]
    rr = np.concatenate(r + [np.arange(n)])
    cc = np.concatenate(c + [np.arange(n)])
    v = np.concatenate([-np.ones(len(rr) - n), np.full(n, 7.0)])
    return sp.csc_array(sp.coo_array((v, (rr, cc)), shape=(n, n)).tocsc())


ANALYZE_CORPUS = CORPUS + [
    ("laplacian3d-23", laplacian3d(23)),
    ("laplacian3d-24", laplacian3d(24)),
]


def analyze(M, stype: int = 1, **kw):
    """``cholmod_analyze``, **pinned to AMD** unless told otherwise."""
    kw.setdefault("ordering", "amd")
    M = sp.csc_array(M)
    tri = sp.triu(M) if stype > 0 else sp.tril(M)
    T = sp.csc_array(tri.tocsc())
    return _rs.analyze(
        T.shape[0], T.indptr.astype(np.int64), T.indices.astype(np.int64), stype, **kw
    )


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_analyze_matches_cholmod(name, M, stype):
    """``Perm`` and the per-column nnz of ``L``, bit-exact to CHOLMOD."""
    ref = REF_ANALYZE[name]
    got = analyze(M, stype=stype)
    perm = np.asarray(got["perm"])
    colcount = np.asarray(got["colcount"])

    assert perm.shape == (ref.n,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(ref.n))
    np.testing.assert_array_equal(perm[: len(ref.head)], ref.head)
    assert hashlib.sha256(perm.tobytes()).hexdigest()[:16] == ref.perm_sha
    assert hashlib.sha256(colcount.tobytes()).hexdigest()[:16] == ref.colcount_sha

    assert got["lnz"] == ref.lnz
    assert got["fl"] == ref.fl
    assert got["anz"] == ref.anz


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_lnz_and_fl_are_the_column_count_moments(name, M):
    """``cholmod_rowcolcounts.c:517-524`` — ``lnz`` is the sum of the column
    counts and ``fl`` the sum of their squares. Both are read back by
    ``cholmod_analyze``: ``fl/lnz`` is what picks supernodal over simplicial."""
    got = analyze(M)
    cc = np.asarray(got["colcount"], dtype=np.float64)
    assert got["lnz"] == cc.sum()
    assert got["fl"] == (cc * cc).sum()


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_leaves_a_postordered_tree(name, M):
    """After the composition every node's parent is above it and each column of
    ``L`` fits below the diagonal. A left-to-right numeric sweep depends on
    both, so a composition that permuted ``Lparent`` inconsistently with
    ``Lperm`` would show up here."""
    got = analyze(M)
    parent = np.asarray(got["parent"])
    cc = np.asarray(got["colcount"])
    n = len(parent)
    j = np.arange(n)
    assert np.all((parent == -1) | (parent > j)), f"{name}: parent points down"
    assert np.all(parent < n)
    assert np.all(cc >= 1) and np.all(cc <= n - j)


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_natural_ordering_is_the_weighted_postorder(name, M):
    """With no fill-reducing ordering the composition has nothing to compose
    with, so ``Lperm`` is the weighted postorder itself — which is why upstream
    relabels the result ``CHOLMOD_POSTORDERED`` (``cholmod_analyze.c:875-878``)
    rather than leaving it ``CHOLMOD_NATURAL``."""
    got = analyze(M, ordering="natural")
    np.testing.assert_array_equal(got["perm"], got["post"])
    assert got["ordering"] == "postordered"
    assert got["metis_would_be_tried"] is False


def test_analyze_composition_actually_moves_the_amd_ordering():
    """Guard against the weighted postorder degenerating into a no-op and this
    file testing nothing. It *is* the identity on banded/tridiagonal/arrow
    shapes — that is why AMD-level values must not be pinned from ``F.perm``
    (see the module docstring) — so require a corpus matrix where it is not."""
    moved = [
        name
        for name, M in ANALYZE_CORPUS
        if not np.array_equal(analyze(M)["perm"], amd_order(M)[0])
    ]
    assert moved, "the weighted postorder was the identity on every matrix"


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_reports_when_cholmod_would_try_metis(name, M):
    """The one place this port's *candidate set* can diverge from a CHOLMOD
    built with the Partition module, reported rather than hidden.
    (``cholmod_analyze.c:767-781``). When it breaks, both stop at AMD and the
    """
    ref = REF_ANALYZE[name]
    got = analyze(M, ordering="best")
    assert got["metis_would_be_tried"] == (ref.default_ordering != 2)

    fl, lnz, anz = got["amd_fl"], got["amd_lnz"], got["amd_anz"]
    assert got["metis_would_be_tried"] == (not (fl < 500 * lnz or lnz < 5 * anz))


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_best_selects_but_does_not_invent(name, M):
    """``ordering="best"`` returns one of the orderings it tried, unchanged.
    The trial loop is CHOLMOD's (``cholmod_analyze.c:554-782``): run each
    """
    best = analyze(M, ordering="best")
    amd = analyze(M, ordering="amd")
    same_as = (
        amd if best["ordering"] == "amd" else analyze(M, ordering=best["ordering"])
    )

    np.testing.assert_array_equal(best["perm"], same_as["perm"])
    np.testing.assert_array_equal(best["colcount"], same_as["colcount"])
    assert best["lnz"] == same_as["lnz"]
    assert best["lnz"] <= amd["lnz"]
    if not best["metis_would_be_tried"]:
        assert best["ordering"] == "amd"


def test_analyze_empty_and_singleton():
    for arr in ("perm", "colcount", "parent", "post"):
        assert np.asarray(analyze(sp.csc_array((0, 0)))[arr]).shape == (0,)
    got = analyze(sp.csc_array(sp.eye_array(1)))
    np.testing.assert_array_equal(got["perm"], [0])
    np.testing.assert_array_equal(got["colcount"], [1])
    np.testing.assert_array_equal(got["parent"], [-1])


def test_analyze_rejects_bad_input():
    """Same precondition as ``amd_order``'s, on the same grounds — plus the
    ``stype == 0`` case, which is a real ``cholmod_analyze`` mode (``LL' =
    AA'``) that this port does not implement and must not silently mis-answer."""
    M = sp.csc_array(sp.eye_array(5).tocsc())
    indptr = M.indptr.astype(np.int64)
    indices = M.indices.astype(np.int64)

    with pytest.raises(ValueError, match="stype must be nonzero"):
        _rs.analyze(5, indptr, indices, 0)
    with pytest.raises(ValueError, match="ordering must be"):
        _rs.analyze(5, indptr, indices, 1, "nesdis")
    with pytest.raises(ValueError, match="indptr has length"):
        _rs.analyze(6, indptr, indices, 1)

    bad = indices.copy()
    bad[0] = 99
    with pytest.raises(ValueError, match="out of range"):
        _rs.analyze(5, indptr, bad, 1)
    bad = indptr.copy()
    bad[2] = 0
    with pytest.raises(ValueError, match="non-decreasing"):
        _rs.analyze(5, bad, indices, 1)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class _RefFactor:
    n: int
    nnz: int
    nzmax: int
    minor: int
    lp_sha: str
    lnz_sha: str
    li_sha: str
    lx_sha: tuple[str, ...]
    rowfacfl: float


REF_FACTOR: dict[tuple[str, int], _RefFactor] = {
    ("banded-50-2", 1): _RefFactor(
        n=50,
        nnz=147,
        nzmax=147,
        minor=50,
        lp_sha="6af2a582b1936279",
        lnz_sha="865156499aecbf3d",
        li_sha="9e66f884da8d629b",
        lx_sha=("2290d18d163f796f",),
        rowfacfl=387.0,
    ),
    ("banded-50-2", -1): _RefFactor(
        n=50,
        nnz=147,
        nzmax=147,
        minor=50,
        lp_sha="6af2a582b1936279",
        lnz_sha="865156499aecbf3d",
        li_sha="9e66f884da8d629b",
        lx_sha=("2290d18d163f796f",),
        rowfacfl=387.0,
    ),
    ("banded-200-3", 1): _RefFactor(
        n=200,
        nnz=794,
        nzmax=794,
        minor=200,
        lp_sha="4a3be5d58a61601b",
        lnz_sha="4c586cf3222672ae",
        li_sha="15bc059d03669e8f",
        lx_sha=("c4e945cc5cfab9d2",),
        rowfacfl=2966.0,
    ),
    ("banded-200-3", -1): _RefFactor(
        n=200,
        nnz=794,
        nzmax=794,
        minor=200,
        lp_sha="4a3be5d58a61601b",
        lnz_sha="4c586cf3222672ae",
        li_sha="15bc059d03669e8f",
        lx_sha=("c4e945cc5cfab9d2",),
        rowfacfl=2966.0,
    ),
    ("random-60", 1): _RefFactor(
        n=60,
        nnz=824,
        nzmax=824,
        minor=60,
        lp_sha="e2cff2be76a52029",
        lnz_sha="c0b682d588e71da7",
        li_sha="c696af543d1d7903",
        lx_sha=("01b818b855cfbec3", "10456c9c429bb506"),
        rowfacfl=14814.0,
    ),
    ("random-60", -1): _RefFactor(
        n=60,
        nnz=824,
        nzmax=824,
        minor=60,
        lp_sha="e2cff2be76a52029",
        lnz_sha="c0b682d588e71da7",
        li_sha="c696af543d1d7903",
        lx_sha=("ddc544a9daeeeac2", "83250f17baa11ac6"),
        rowfacfl=14814.0,
    ),
    ("random-300", 1): _RefFactor(
        n=300,
        nnz=23541,
        nzmax=23541,
        minor=300,
        lp_sha="e0afbd5dc9892e75",
        lnz_sha="92aea86e3f8a50b6",
        li_sha="8a144f151496a722",
        lx_sha=("0b58f70634e6a7d4", "cff69ae82fe159a9"),
        rowfacfl=2915259.0,
    ),
    ("random-300", -1): _RefFactor(
        n=300,
        nnz=23541,
        nzmax=23541,
        minor=300,
        lp_sha="e0afbd5dc9892e75",
        lnz_sha="92aea86e3f8a50b6",
        li_sha="8a144f151496a722",
        lx_sha=("b5204e0254dfeca7", "8d0bc85bc51711d5"),
        rowfacfl=2915259.0,
    ),
    ("random-400", 1): _RefFactor(
        n=400,
        nnz=36946,
        nzmax=36946,
        minor=400,
        lp_sha="790e67d16de86eb3",
        lnz_sha="fc3a5493740ce535",
        li_sha="7f2dd9766114ce7d",
        lx_sha=("e35399ba32a60278", "1b2f985f23c69701"),
        rowfacfl=5775708.0,
    ),
    ("random-400", -1): _RefFactor(
        n=400,
        nnz=36946,
        nzmax=36946,
        minor=400,
        lp_sha="790e67d16de86eb3",
        lnz_sha="fc3a5493740ce535",
        li_sha="7f2dd9766114ce7d",
        lx_sha=("4afffbb544167b62", "47790aafb2b78577"),
        rowfacfl=5775708.0,
    ),
    ("block-diagonal", 1): _RefFactor(
        n=55,
        nnz=289,
        nzmax=289,
        minor=55,
        lp_sha="88458d28d71490ee",
        lnz_sha="6f6a0e55b43fff0e",
        li_sha="853fc1e3e66190fe",
        lx_sha=("436f8b0493d15d69", "df25f6990bd546d0"),
        rowfacfl=2102.0,
    ),
    ("block-diagonal", -1): _RefFactor(
        n=55,
        nnz=289,
        nzmax=289,
        minor=55,
        lp_sha="88458d28d71490ee",
        lnz_sha="6f6a0e55b43fff0e",
        li_sha="853fc1e3e66190fe",
        lx_sha=("8633debf1f0da1da", "b56168b324f19401"),
        rowfacfl=2102.0,
    ),
    ("arrow-300", 1): _RefFactor(
        n=300,
        nnz=599,
        nzmax=599,
        minor=300,
        lp_sha="e0e20bc1372bebcc",
        lnz_sha="5a05756d08f2e75e",
        li_sha="ec4d5a84f794f211",
        lx_sha=("dd8f2c6fc038d9d8",),
        rowfacfl=897.0,
    ),
    ("arrow-300", -1): _RefFactor(
        n=300,
        nnz=599,
        nzmax=599,
        minor=300,
        lp_sha="e0e20bc1372bebcc",
        lnz_sha="5a05756d08f2e75e",
        li_sha="ec4d5a84f794f211",
        lx_sha=("dd8f2c6fc038d9d8",),
        rowfacfl=897.0,
    ),
    ("tridiagonal-200", 1): _RefFactor(
        n=200,
        nnz=399,
        nzmax=399,
        minor=200,
        lp_sha="6903593832f56f1f",
        lnz_sha="643b7fb9515a8a78",
        li_sha="ab288bfd44b154df",
        lx_sha=("1620eef639965fce",),
        rowfacfl=597.0,
    ),
    ("tridiagonal-200", -1): _RefFactor(
        n=200,
        nnz=399,
        nzmax=399,
        minor=200,
        lp_sha="6903593832f56f1f",
        lnz_sha="643b7fb9515a8a78",
        li_sha="ab288bfd44b154df",
        lx_sha=("1620eef639965fce",),
        rowfacfl=597.0,
    ),
    ("diagonal-32", 1): _RefFactor(
        n=32,
        nnz=32,
        nzmax=32,
        minor=32,
        lp_sha="4e8adfac993e338c",
        lnz_sha="6ba64591dc5d5fa6",
        li_sha="bcc9bcfc670935c6",
        lx_sha=("acfc7c36fce590b1",),
        rowfacfl=0.0,
    ),
    ("diagonal-32", -1): _RefFactor(
        n=32,
        nnz=32,
        nzmax=32,
        minor=32,
        lp_sha="4e8adfac993e338c",
        lnz_sha="6ba64591dc5d5fa6",
        li_sha="bcc9bcfc670935c6",
        lx_sha=("acfc7c36fce590b1",),
        rowfacfl=0.0,
    ),
    ("kron-duplicate-rows-120", 1): _RefFactor(
        n=120,
        nnz=6544,
        nzmax=6544,
        minor=120,
        lp_sha="1ea65d7e3ce4c0a0",
        lnz_sha="5ee6d99e18ede87a",
        li_sha="0177729cb00e2d22",
        lx_sha=("bfaf6a07daf5abd7", "31d6fa7999a18f48"),
        rowfacfl=451224.0,
    ),
    ("kron-duplicate-rows-120", -1): _RefFactor(
        n=120,
        nnz=6544,
        nzmax=6544,
        minor=120,
        lp_sha="1ea65d7e3ce4c0a0",
        lnz_sha="5ee6d99e18ede87a",
        li_sha="0177729cb00e2d22",
        lx_sha=("622ea1aa286fe3a4", "962d63b27a458182"),
        rowfacfl=451224.0,
    ),
}


def factorize(M, stype: int = 1, **kw):
    M = sp.csc_array(M)
    tri = sp.triu(M) if stype > 0 else sp.tril(M)
    T = sp.csc_array(tri.tocsc())
    T.sort_indices()
    return _rs.factorize(
        T.shape[0],
        T.indptr.astype(np.int64),
        T.indices.astype(np.int64),
        T.data.astype(np.float64),
        stype,
        **kw,
    )


def live(got) -> tuple[np.ndarray, np.ndarray]:
    """The entries ``L`` actually holds."""
    lp, lnz = np.asarray(got["Lp"]), np.asarray(got["Lnz"])
    idx = np.concatenate([np.arange(lp[j], lp[j] + lnz[j]) for j in range(len(lnz))])
    return np.asarray(got["Li"])[idx], np.asarray(got["Lx"])[idx]


def dense_ldl(got, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``L`` (unit diagonal) and ``D``, densely, from the unpacked columns."""
    lp, lnz, li, lx = (np.asarray(got[k]) for k in ("Lp", "Lnz", "Li", "Lx"))
    L = np.zeros((n, n))
    D = np.zeros(n)
    for j in range(n):
        p = lp[j]
        D[j] = lx[p]
        L[j, j] = lx[p] if got["is_ll"] else 1.0
        L[li[p + 1 : p + lnz[j]], j] = lx[p + 1 : p + lnz[j]]
    return L, D


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_factorize_matches_cholmod(name, M, stype):
    """Every entry of ``L``, bit-exact to ``cholmod_l_factorize``."""
    ref = REF_FACTOR[(name, stype)]
    got = factorize(M, stype=stype)

    assert got["minor"] == ref.minor
    assert got["nzmax"] == ref.nzmax
    assert int(np.asarray(got["Lnz"]).sum()) == ref.nnz
    assert got["rowfacfl"] == ref.rowfacfl
    assert not got["is_ll"], "cholmod_defaults leaves final_ll false"
    assert _sha(np.asarray(got["Lp"])) == ref.lp_sha
    assert _sha(np.asarray(got["Lnz"])) == ref.lnz_sha
    li, lx = live(got)
    assert _sha(li) == ref.li_sha
    assert _sha(lx) in ref.lx_sha, "L's values are not bit-exact to CHOLMOD's"


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_factorize_reconstructs_the_matrix(name, M, stype):
    """``L D L' == P A P'`` — the pins above say we match CHOLMOD, this says
    what we both compute is the factorization."""
    n = M.shape[0]
    got = factorize(M, stype=stype)
    L, D = dense_ldl(got, n)
    p = np.asarray(got["perm"])
    pap = np.asarray(M.todense())[np.ix_(p, p)]
    resid = np.abs(L @ np.diag(D) @ L.T - pap).max()
    assert resid < 1e-9 * max(1.0, np.abs(pap).max())


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_factorize_ll_is_the_ldl_with_d_folded_in(name, M):
    """``final_ll`` factorizes to ``LL'`` directly rather than converting, so it
    is a separate code path in the worker and gets its own check."""
    n = M.shape[0]
    ldl = factorize(M, final_ll=False)
    ll = factorize(M, final_ll=True)
    assert ll["is_ll"] and not ldl["is_ll"]
    np.testing.assert_array_equal(np.asarray(ll["Lnz"]), np.asarray(ldl["Lnz"]))
    np.testing.assert_array_equal(live(ll)[0], live(ldl)[0])

    Lu, D = dense_ldl(ldl, n)
    Ll, _ = dense_ldl(ll, n)
    assert (D > 0).all(), "the corpus is positive definite"
    np.testing.assert_allclose(Ll, Lu * np.sqrt(D)[None, :], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_factorize_columns_are_exactly_colcount_long(name, M):
    """``cholmod_factorize`` zeroes ``Common->grow2`` for the first
    factorization of a symbolic factor (``cholmod_factorize.c:388-392``), so
    every column comes out sized to its exact ``ColCount`` and ``L`` is packed
    and monotonic despite ``final_asis`` leaving it "unpacked"."""
    got = factorize(M)
    lp, lnz = np.asarray(got["Lp"]), np.asarray(got["Lnz"])
    analysis = analyze(M)
    np.testing.assert_array_equal(lnz, np.asarray(analysis["colcount"]))
    np.testing.assert_array_equal(np.diff(lp), lnz)
    assert got["is_monotonic"]
    assert got["nzmax"] == int(lnz.sum())


def test_factorize_beta_shifts_the_diagonal():
    """``beta`` factorizes ``beta*I + A`` without touching the pattern."""
    M = dict(CORPUS)["random-60"]
    plain = factorize(M)
    shifted = factorize(M, beta=2.5)
    np.testing.assert_array_equal(np.asarray(plain["Lnz"]), np.asarray(shifted["Lnz"]))
    np.testing.assert_array_equal(live(plain)[0], live(shifted)[0])
    _, d0 = dense_ldl(plain, M.shape[0])
    _, d1 = dense_ldl(shifted, M.shape[0])
    assert (d1 > d0).all()

    n = M.shape[0]
    L, D = dense_ldl(shifted, n)
    p = np.asarray(shifted["perm"])
    pap = np.asarray((M + sp.eye_array(n) * 2.5).todense())[np.ix_(p, p)]
    assert np.abs(L @ np.diag(D) @ L.T - pap).max() < 1e-9 * np.abs(pap).max()


def test_factorize_reports_where_it_stopped_being_positive_definite():
    """Not positive definite is reported through ``L->minor``, not an error —
    ``rowfac`` sets it and carries on (``t_cholmod_rowfac_worker.c:430-434``)."""
    M = sp.csc_array(np.array([[1.0, 2.0], [2.0, 1.0]]))
    ldl = factorize(M, ordering="natural")
    assert ldl["minor"] == 2
    assert np.asarray(ldl["Lx"])[np.asarray(ldl["Lp"])[1]] == -3.0
    ll = factorize(M, ordering="natural", final_ll=True)
    assert ll["minor"] == 1

    S = sp.csc_array(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert factorize(S, ordering="natural")["minor"] == 1


def test_factorize_empty_and_singleton():
    got = factorize(sp.csc_array((0, 0)))
    assert np.asarray(got["Lp"]).tolist() == [0]
    assert np.asarray(got["Lnz"]).size == 0
    assert got["minor"] == 0

    got = factorize(sp.csc_array(np.array([[4.0]])))
    np.testing.assert_array_equal(np.asarray(got["Lnz"]), [1])
    assert np.asarray(got["Lx"])[0] == 4.0
    assert factorize(sp.csc_array(np.array([[4.0]])), final_ll=True)["Lx"][0] == 2.0


def test_a_square_stype_zero_input_factorizes_aat():
    ip = np.array([0, 1, 2], dtype=np.int64)
    ii = np.array([0, 1], dtype=np.int64)
    ax = np.array([1.0, 1.0])
    f = _rs.factorize(2, ip, ii, ax, 0)
    assert f["Lnz"].tolist() == [1, 1]
    assert f["Li"][: f["Lnz"].sum()].tolist() == [0, 1]
    assert f["Lx"][: f["Lnz"].sum()].tolist() == [1.0, 1.0]
    assert f["minor"] == 2


def test_factorize_rejects_bad_input():
    ip = np.array([0, 1, 2], dtype=np.int64)
    ii = np.array([0, 1], dtype=np.int64)
    ax = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="ordering must be"):
        _rs.factorize(2, ip, ii, ax, 1, ordering="nesdis")
    with pytest.raises(ValueError, match="indptr"):
        _rs.factorize(2, np.array([0, 2, 1], dtype=np.int64), ii, ax, 1)
    with pytest.raises(ValueError, match="row index"):
        _rs.factorize(2, ip, np.array([0, 9], dtype=np.int64), ax, 1)


# ---------------------------------------------------------------------------
# cholmod_solve
# ---------------------------------------------------------------------------
#
# The pins below are ``cholmod_l_solve`` on the factor the section above pins,
# driven identically. Two shas per case rather than one, because they fail for
# different reasons:
#
# * ``by_rank`` concatenates ``X`` for ``nrhs = 1..5``, which is what selects
#   among the four separately-unrolled kernels (and, at 5, the blocking loop
#   that runs a 4-block then a 1-block). Upstream's ``ltsolve`` rank-4 kernel is
#   **not** the other three with a wider inner loop: its three-column branch is
#   commented out (``t_cholmod_ltsolve_template.c:656``), so it walks a
#   three-column chain as 2+1 and sums different products. That difference is
#   invisible at ``nrhs <= 3`` and invisible in any residual norm.
# * ``by_system`` concatenates ``X`` over all nine ``sys`` values at
#   ``nrhs = 4``, which is what selects among the six kernels
#   ``simplicial_solver`` dispatches to plus the two permutation applies.
#
# Both are pinned for ``LDL'`` and ``LL'`` separately: the two forms divide by
# ``d`` at different points, and ``rowfac`` fails an ``LL'`` on any non-positive
# pivot but an ``LDL'`` only on a zero one, so they do not agree entry for entry
# and are not meant to.


@dataclass(frozen=True)
class _RefSolve:
    by_rank: tuple[str, ...]
    by_system: tuple[str, ...]
    half_log_det: float
    nnz: int


REF_SOLVE: dict[tuple[str, int, bool], _RefSolve] = {
    ("banded-50-2", 1, False): _RefSolve(
        by_rank=("b3e42e61a4b58a2f", "1ec5f608e32e8c0d"),
        by_system=("b12c8ed8326542a8", "90c7ac074e3f2750"),
        half_log_det=51.85502920029276,
        nnz=147,
    ),
    ("banded-50-2", 1, True): _RefSolve(
        by_rank=("5c34d46355996fe9", "194b0de66702943d"),
        by_system=("50b906aabb225065", "066fd6707c138f30"),
        half_log_det=51.85502920029276,
        nnz=147,
    ),
    ("banded-50-2", -1, False): _RefSolve(
        by_rank=("b3e42e61a4b58a2f", "1ec5f608e32e8c0d"),
        by_system=("b12c8ed8326542a8", "90c7ac074e3f2750"),
        half_log_det=51.85502920029276,
        nnz=147,
    ),
    ("banded-50-2", -1, True): _RefSolve(
        by_rank=("5c34d46355996fe9", "194b0de66702943d"),
        by_system=("50b906aabb225065", "066fd6707c138f30"),
        half_log_det=51.85502920029276,
        nnz=147,
    ),
    ("banded-200-3", 1, False): _RefSolve(
        by_rank=("ca0c07e824e8eb70", "380d6f80a84e29be"),
        by_system=("5b884584ecc8a82f", "1e039947b0e75bd8"),
        half_log_det=229.8662405766237,
        nnz=794,
    ),
    ("banded-200-3", 1, True): _RefSolve(
        by_rank=("85ead7c33ccc3a97", "50b4a2bae7906f3e"),
        by_system=("5f565f2d4721ff12", "96a937f2b205265b"),
        half_log_det=229.8662405766237,
        nnz=794,
    ),
    ("banded-200-3", -1, False): _RefSolve(
        by_rank=("ca0c07e824e8eb70", "380d6f80a84e29be"),
        by_system=("5b884584ecc8a82f", "1e039947b0e75bd8"),
        half_log_det=229.8662405766237,
        nnz=794,
    ),
    ("banded-200-3", -1, True): _RefSolve(
        by_rank=("85ead7c33ccc3a97", "50b4a2bae7906f3e"),
        by_system=("5f565f2d4721ff12", "96a937f2b205265b"),
        half_log_det=229.8662405766237,
        nnz=794,
    ),
    ("random-60", 1, False): _RefSolve(
        by_rank=("26588f2c9c573dcd", "6548c362149f13eb"),
        by_system=("1e35b9a9335587fc", "677c2c80acfb8c4f"),
        half_log_det=67.55521807419473,
        nnz=824,
    ),
    ("random-60", 1, True): _RefSolve(
        by_rank=("22d55cc107432fbf", "bd61e0fea966b084"),
        by_system=("0d67d5f0e35bb004", "8c29d33d2c294da6"),
        half_log_det=67.55521807419473,
        nnz=824,
    ),
    ("random-60", -1, False): _RefSolve(
        by_rank=("6fdb3470d7d370f3", "5077631cec201b50"),
        by_system=("bdd473fcff5706f3", "7c464b5ca86ffcc7"),
        half_log_det=67.55521807419473,
        nnz=824,
    ),
    ("random-60", -1, True): _RefSolve(
        by_rank=("c9b26437fe723ead", "aa0731b4e22dd41d"),
        by_system=("fdeb25a2c7a61787", "69c01c3ec08d6875"),
        half_log_det=67.55521807419473,
        nnz=824,
    ),
    ("random-300", 1, False): _RefSolve(
        by_rank=("bb929bb03cde74fe", "c79400aca798f1c8"),
        by_system=("7ea0550c5a98fa0b", "fd347a0184475e8a"),
        half_log_det=426.52057262126283,
        nnz=23541,
    ),
    ("random-300", 1, True): _RefSolve(
        by_rank=("239086e986b5834f", "b909bd2c7dd0eb40"),
        by_system=("01c43ac0530c59f9", "264455985744e9c9"),
        half_log_det=426.52057262126283,
        nnz=23541,
    ),
    ("random-300", -1, False): _RefSolve(
        by_rank=("03c8daff36c296a2", "eebc2d3c3cd71f25"),
        by_system=("8e1e0e80e1ef592b", "f29205bffab687dd"),
        half_log_det=426.52057262126283,
        nnz=23541,
    ),
    ("random-300", -1, True): _RefSolve(
        by_rank=("7982117bc5c55ac9", "e0f2ebedbda5db0b"),
        by_system=("34e0f0d664e086d9", "ae526e3e14a9d07d"),
        half_log_det=426.52057262126283,
        nnz=23541,
    ),
    ("random-400", 1, False): _RefSolve(
        by_rank=("f01d1adeec10afcc", "9fd27508597258d2"),
        by_system=("3e614a2f7370f7fd", "ce2cab1c6094b781"),
        half_log_det=580.4342322486121,
        nnz=36946,
    ),
    ("random-400", 1, True): _RefSolve(
        by_rank=("23e2559e666a54e1", "ac2a779336fe2a55"),
        by_system=("ce35b933b93b91e6", "dca9ade25983896b"),
        half_log_det=580.4342322486121,
        nnz=36946,
    ),
    ("random-400", -1, False): _RefSolve(
        by_rank=("c96879fe5eb7db2e", "35d1eba8c08cddcd"),
        by_system=("73e380e7400145ff", "c70447c85a3c3fec"),
        half_log_det=580.4342322486121,
        nnz=36946,
    ),
    ("random-400", -1, True): _RefSolve(
        by_rank=("a98df719e8d9e304", "37495f1e8fbf4e4d"),
        by_system=("cc139784cd41d4ef", "e45d8811560947ae"),
        half_log_det=580.4342322486121,
        nnz=36946,
    ),
    ("block-diagonal", 1, False): _RefSolve(
        by_rank=("748381de6744fc04", "47f54ee5c0aca1a9"),
        by_system=("060f7da7d9767534", "4352ad6e21b8cd31"),
        half_log_det=64.01329336034428,
        nnz=289,
    ),
    ("block-diagonal", 1, True): _RefSolve(
        by_rank=("11ac8d501814975d", "6fe2875ba909f266"),
        by_system=("ec0f86e06c52f452", "30356d0afadc6dc0"),
        half_log_det=64.01329336034426,
        nnz=289,
    ),
    ("block-diagonal", -1, False): _RefSolve(
        by_rank=("57512f8719a70943", "8ecb6df2faea2741"),
        by_system=("c516b9674899d53c", "c9a3f5234a3eafd1"),
        half_log_det=64.01329336034428,
        nnz=289,
    ),
    ("block-diagonal", -1, True): _RefSolve(
        by_rank=("fea0d53d9b9e8cb0", "b99b0dcdf64dc80f"),
        by_system=("cf303bcbfea670e0", "694050f7bade9f3b"),
        half_log_det=64.01329336034426,
        nnz=289,
    ),
    ("arrow-300", 1, False): _RefSolve(
        by_rank=("9844aa5fc62245fb",),
        by_system=("56ffdf70317835f5",),
        half_log_det=209.96841354299235,
        nnz=599,
    ),
    ("arrow-300", 1, True): _RefSolve(
        by_rank=("05754071abd6a611",),
        by_system=("d3b8a83ee655fda8",),
        half_log_det=209.96841354299235,
        nnz=599,
    ),
    ("arrow-300", -1, False): _RefSolve(
        by_rank=("9844aa5fc62245fb",),
        by_system=("56ffdf70317835f5",),
        half_log_det=209.96841354299235,
        nnz=599,
    ),
    ("arrow-300", -1, True): _RefSolve(
        by_rank=("05754071abd6a611",),
        by_system=("d3b8a83ee655fda8",),
        half_log_det=209.96841354299235,
        nnz=599,
    ),
    ("tridiagonal-200", 1, False): _RefSolve(
        by_rank=("4390891dfcb5dc58", "44284127e2a3a790"),
        by_system=("8194fd9d58ed5020", "2b06a748db79b96d"),
        half_log_det=131.73304197849657,
        nnz=399,
    ),
    ("tridiagonal-200", 1, True): _RefSolve(
        by_rank=("c4b726272b19f5fc", "70e71a88419baa25"),
        by_system=("6ea1b4729ce9a12f", "349074074779abcc"),
        half_log_det=131.73304197849657,
        nnz=399,
    ),
    ("tridiagonal-200", -1, False): _RefSolve(
        by_rank=("4390891dfcb5dc58", "44284127e2a3a790"),
        by_system=("8194fd9d58ed5020", "2b06a748db79b96d"),
        half_log_det=131.73304197849657,
        nnz=399,
    ),
    ("tridiagonal-200", -1, True): _RefSolve(
        by_rank=("c4b726272b19f5fc", "70e71a88419baa25"),
        by_system=("6ea1b4729ce9a12f", "349074074779abcc"),
        half_log_det=131.73304197849657,
        nnz=399,
    ),
    ("diagonal-32", 1, False): _RefSolve(
        by_rank=("ac73e4c83918f2b8",),
        by_system=("5b4e20ba6bcbe892",),
        half_log_det=0.0,
        nnz=32,
    ),
    ("diagonal-32", 1, True): _RefSolve(
        by_rank=("ac73e4c83918f2b8",),
        by_system=("5b4e20ba6bcbe892",),
        half_log_det=0.0,
        nnz=32,
    ),
    ("diagonal-32", -1, False): _RefSolve(
        by_rank=("ac73e4c83918f2b8",),
        by_system=("5b4e20ba6bcbe892",),
        half_log_det=0.0,
        nnz=32,
    ),
    ("diagonal-32", -1, True): _RefSolve(
        by_rank=("ac73e4c83918f2b8",),
        by_system=("5b4e20ba6bcbe892",),
        half_log_det=0.0,
        nnz=32,
    ),
    ("kron-duplicate-rows-120", 1, False): _RefSolve(
        by_rank=("c6451d89bd3efdc6", "5f7ac47859a43105"),
        by_system=("cff539f1ebdf7b72", "67109746401bd5ad"),
        half_log_det=287.1094728591422,
        nnz=6544,
    ),
    ("kron-duplicate-rows-120", 1, True): _RefSolve(
        by_rank=("158a443ef230cd17", "f770c94e3c098768"),
        by_system=("8f4713f4501eba86", "321c6022a1ae017b"),
        half_log_det=287.1094728591422,
        nnz=6544,
    ),
    ("kron-duplicate-rows-120", -1, False): _RefSolve(
        by_rank=("af2051f7272af7eb", "656154fd66221728"),
        by_system=("bc1fd6a62fa36f5b", "7f3f449fd4c5ce1b"),
        half_log_det=287.1094728591422,
        nnz=6544,
    ),
    ("kron-duplicate-rows-120", -1, True): _RefSolve(
        by_rank=("a91221d70d16e6cc", "e6be38d7e8892f16"),
        by_system=("2a4a9e9c01f83e8c", "04e65b9aa034c9f7"),
        half_log_det=287.1094728591422,
        nnz=6544,
    ),
}


SYSTEMS = ("A", "LDLt", "LD", "DLt", "L", "Lt", "D", "P", "Pt")


def _shaf(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()[
        :16
    ]


def solve_rhs(n: int, nrhs: int) -> np.ndarray:
    """The right-hand sides the pins were taken on. Seeded, like the corpus."""
    return np.random.default_rng(20260805).standard_normal((n, nrhs))


def cho(M, stype: int = 1, use_ll: bool = False, **kw):
    """A factor of the stored triangle, through the public facade, **pinned to
    the simplicial path** unless told otherwise.
    """
    kw.setdefault("supernodal", "simplicial")
    tri = sp.csc_array((sp.triu(M) if stype > 0 else sp.tril(M)).tocsc())
    tri.sort_indices()
    return hea.sparse.Factor(tri, lower=stype < 0, use_ll=use_ll, **kw)


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
@pytest.mark.parametrize("use_ll", [False, True])
def test_solve_matches_cholmod(name, M, stype, use_ll):
    """Every entry of ``X``, bit-exact to ``cholmod_l_solve``."""
    ref = REF_SOLVE[(name, stype, use_ll)]
    n = M.shape[0]
    F = cho(M, stype, use_ll)
    assert F.nnz == ref.nnz
    assert F.half_log_det() == ref.half_log_det

    parts = [
        np.ravel(np.atleast_2d(F.solve(solve_rhs(n, k))).reshape(n, k), order="F")
        for k in (1, 2, 3, 4, 5)
    ]
    assert _shaf(np.concatenate(parts) if n else np.zeros(0)) in ref.by_rank

    parts = [
        np.ravel(F.solve(solve_rhs(n, 4), s).reshape(n, 4), order="F") for s in SYSTEMS
    ]
    assert _shaf(np.concatenate(parts) if n else np.zeros(0)) in ref.by_system


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_supernodal_and_simplicial_agree(name, M, stype):
    """The two factorizations are the same factorization.
    (``cholmod_analyze.c:887-891``) — and it is worth 4x on the matrices that
    """
    n = M.shape[0]
    if n == 0:
        pytest.skip("nothing to factorize")
    sup = cho(M, stype, use_ll=True, supernodal="supernodal")
    sim = cho(M, stype, use_ll=True, supernodal="simplicial")
    assert sup.is_super and not sim.is_super

    np.testing.assert_array_equal(sup.P, sim.P)
    assert sup.nnz == sim.nnz
    Ls, Li = sup.L, sim.L
    np.testing.assert_array_equal(Ls.indptr, Li.indptr)
    np.testing.assert_array_equal(Ls.indices, Li.indices)
    scale = max(np.abs(Li.data).max(), 1.0)
    assert np.abs(Ls.data - Li.data).max() <= 1e-12 * scale
    assert sup.half_log_det() == pytest.approx(sim.half_log_det(), rel=1e-12)

    B = solve_rhs(n, 3)
    assert np.abs(sup.solve(B) - sim.solve(B)).max() <= 1e-9 * scale


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_solve_inverts_the_matrix(name, M, stype):
    """The pins say we match CHOLMOD; this says what we both compute solves
    the system."""
    n = M.shape[0]
    if n == 0:
        pytest.skip("no system to solve")
    F = cho(M, stype, use_ll=True)
    B = solve_rhs(n, 3)
    X = F.solve(B)
    assert np.abs(M @ X - B).max() < 1e-9


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_solve_composes_out_of_its_pieces(name, M):
    """``A \\ b`` built from ``P``, ``L``, ``D``, ``Lt``, ``Pt`` is the one-shot
    solve, entry for entry — which is what makes the ``sys`` dispatch checkable
    without a second oracle."""
    n = M.shape[0]
    if n == 0:
        pytest.skip("no system to solve")
    F = cho(M, use_ll=False)
    b = solve_rhs(n, 2)
    want = F.solve(b)
    t = F.solve(F.solve(F.solve(F.solve(b, "P"), "L"), "D"), "Lt")
    np.testing.assert_array_equal(F.solve(t, "Pt"), want)


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_solve_permutation_systems_are_inverses(name, M):
    """``system="P"`` and ``"Pt"`` undo each other. scikit-sparse 0.5.0 rejects
    both, which is why hea had to fancy-index the right-hand side instead."""
    n = M.shape[0]
    if n == 0:
        pytest.skip("no system to solve")
    F = cho(M)
    b = solve_rhs(n, 2)
    np.testing.assert_array_equal(F.solve(F.solve(b, "P"), "Pt"), b)
    p = F.P
    np.testing.assert_array_equal(F.solve(b, "P"), b[p])


@pytest.mark.parametrize("name,M", CORPUS, ids=[n for n, _ in CORPUS])
def test_solve_factor_reconstructs_the_permuted_matrix(name, M):
    """``L @ L.T == A[p][:, p]``, which is the contract every consumer that
    solves against ``L`` directly depends on."""
    n = M.shape[0]
    if n == 0:
        pytest.skip("no factor")
    F = cho(M, use_ll=True)
    p = F.P
    L = F.L
    assert np.abs((L @ L.T).toarray() - M.toarray()[p][:, p]).max() < 1e-9


def test_solve_reuses_the_analysis_for_new_values():
    """``factorize`` on the same pattern must equal a factorization from
    scratch, bit for bit — that reuse is the whole reason gmm holds a factor."""
    _, M = CORPUS[1]
    n = M.shape[0]
    tri = sp.csc_array(sp.triu(M).tocsc())
    tri.sort_indices()
    F = hea.sparse.Factor(tri)
    tri2 = tri.copy()
    tri2.data = tri2.data * 1.5
    F.factorize(tri2)
    fresh = hea.sparse.Factor(tri2)
    b = solve_rhs(n, 4)
    np.testing.assert_array_equal(F.solve(b), fresh.solve(b))
    np.testing.assert_array_equal(F.L.toarray(), fresh.L.toarray())
    assert F.half_log_det() == fresh.half_log_det()


def test_solve_accepts_the_full_matrix_or_its_triangle():
    """``stype`` says which half is stored; the other is ignored rather than
    folded in, so passing the full symmetric matrix is the same problem."""
    _, M = CORPUS[2]
    b = solve_rhs(M.shape[0], 3)
    for stype in (1, -1):
        full = hea.sparse.Factor(M, lower=stype < 0, use_ll=False).solve(b)
        np.testing.assert_array_equal(cho(M, stype).solve(b), full)
    assert not np.array_equal(cho(M, 1).solve(b), cho(M, -1).solve(b))


def test_solve_rejects_bad_input():
    _, M = CORPUS[0]
    F = cho(M)
    n = M.shape[0]
    with pytest.raises(ValueError, match="system must be one of"):
        F.solve(np.ones(n), "Q")
    with pytest.raises(ValueError, match="rows"):
        F.solve(np.ones(n + 1))
    with pytest.raises(ValueError, match="1- or 2-D"):
        F.solve(np.ones((n, 1, 1)))
    with pytest.raises(ValueError, match="square"):
        hea.sparse.Factor(sp.csc_array(np.ones((3, 4))))


def test_solve_not_positive_definite_is_an_error():
    """``rowfac`` fails an ``LL'`` on any non-positive pivot but an ``LDL'``
    only on a zero one, so the two forms disagree about what "not positive
    definite" means, and this port disagrees the same way."""
    indefinite = sp.csc_array(np.array([[1.0, 2.0], [2.0, 1.0]]))
    singular = sp.csc_array(np.array([[0.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(hea.sparse.CholmodError, match="not positive definite"):
        hea.sparse.cho_factor(indefinite)
    hea.sparse.Factor(indefinite, use_ll=False)  # LDL' tolerates it, as CHOLMOD does
    for use_ll in (True, False):
        with pytest.raises(hea.sparse.CholmodError):
            hea.sparse.Factor(singular, use_ll=use_ll)


def test_cho_solve_one_shot_matches_the_factor():
    _, M = CORPUS[3]
    b = solve_rhs(M.shape[0], 1)
    np.testing.assert_array_equal(
        hea.sparse.cho_solve(M, b), hea.sparse.cho_factor(M).solve(b)
    )


def test_refactorize_accepts_a_pattern_that_shrank():
    """``factorize`` takes ``A``'s pattern, not just its values."""
    full = sp.csc_array(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
    F = hea.sparse.cho_factor(full)

    thin = sp.csc_array(np.diag([4.0, 3.0, 2.0]))
    assert thin.nnz < full.nnz
    F.factorize(thin)

    b = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        F.solve(b), np.linalg.solve(thin.toarray(), b), atol=1e-12
    )
    np.testing.assert_allclose(
        F.half_log_det(), 0.5 * np.linalg.slogdet(thin.toarray())[1], rtol=1e-12
    )
    F.factorize(full)
    np.testing.assert_allclose(
        F.solve(b), np.linalg.solve(full.toarray(), b), atol=1e-12
    )


def test_refactorize_grows_the_pattern_on_the_simplicial_path():
    """A *wider* ``A`` is factorized, not refused — because ``rowfac`` derives
    each row's pattern from ``A`` and the etree as it goes and grows ``L`` when
    it has to. That is upstream's behaviour and it is load-bearing.
    """
    ztz = np.array([[9.0, 3, 3], [3, 3, 0], [3, 0, 3]])
    lam = np.array([[1.0, 0, 0], [0.3, 1, 0], [0.2, 0.4, 1]])
    narrow = sp.csc_array(ztz + np.eye(3))
    wide = sp.csc_array(lam @ ztz @ lam.T + np.eye(3))
    assert wide.nnz > narrow.nnz

    F = hea.sparse.cho_factor(narrow)
    assert not F.is_super
    F.factorize(wide)

    b = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        F.solve(b), np.linalg.solve(wide.toarray(), b), atol=1e-12
    )
    np.testing.assert_allclose(
        F.half_log_det(), 0.5 * np.linalg.slogdet(wide.toarray())[1], rtol=1e-12
    )
    F.factorize(narrow)
    np.testing.assert_allclose(
        F.solve(b), np.linalg.solve(narrow.toarray(), b), atol=1e-12
    )


def test_refactorize_reanalyzes_when_the_supernodes_cannot_hold_the_pattern():
    """The supernodal path grows too — by re-analyzing, not by refusing."""
    ztz = np.array([[9.0, 3, 3], [3, 3, 0], [3, 0, 3]])
    lam = np.array([[1.0, 0, 0], [0.3, 1, 0], [0.2, 0.4, 1]])
    narrow = sp.csc_array(ztz + np.eye(3))
    wide = sp.csc_array(lam @ ztz @ lam.T + np.eye(3))
    b = np.array([1.0, 2.0, 3.0])

    for mode in ("simplicial", "supernodal"):
        F = hea.sparse.Factor(narrow, supernodal=mode)
        assert F.is_super == (mode == "supernodal")
        F.factorize(wide)
        assert F.is_super == (mode == "supernodal")  # and it stays on its path
        np.testing.assert_allclose(
            F.solve(b), np.linalg.solve(wide.toarray(), b), atol=1e-12
        )
        np.testing.assert_allclose(
            F.half_log_det(), 0.5 * np.linalg.slogdet(wide.toarray())[1], rtol=1e-12
        )
        p = F.P
        ld = F.L.toarray()
        np.testing.assert_allclose(ld @ ld.T, wide.toarray()[np.ix_(p, p)], atol=1e-12)
        F.factorize(narrow)
        np.testing.assert_allclose(
            F.solve(b), np.linalg.solve(narrow.toarray(), b), atol=1e-12
        )

    with pytest.raises(ValueError, match="expected n = 3"):
        hea.sparse.cho_factor(narrow).factorize(sp.csc_array(np.eye(4)))
