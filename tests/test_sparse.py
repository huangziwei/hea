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


# AMD 3.3.1 (SuiteSparse 7.6.0). ``head`` is Perm[:8] and ``sha`` the first 16
# hex digits of sha256 over the full int64 permutation; the rest is AMD's
# ``Info`` array under its upstream names.
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
    # the one case that forces a garbage collection (ncmpa = 1)
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
    # one dense row/column — the only case that trips AMD's `dense` threshold
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
    # every row has degree 0 — the `deg == 0` init branch, exclusively
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
    # many identical row patterns — hammers supervariable detection
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
    """Permutation and every ``Info`` statistic, bit-exact to C AMD 3.3.1.

    ``use_long`` selects which C build to reproduce. AMD's `hash` is `UInt`,
    which follows `Int`, so the two builds can only diverge where `hash`
    overflows; on this corpus they must agree.
    """
    ref = REF[name]
    perm, info = amd_order(M, use_long=use_long)

    assert perm.shape == (ref.n,)
    # a permutation, not just the right values
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

    # a non-monotone indptr is what would otherwise send the column loop past
    # the end of `indices`
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


# ---------------------------------------------------------------------------
# cholmod_analyze
# ---------------------------------------------------------------------------
#
# From here on ``scikit-sparse``'s ``F.perm`` *is* the right comparison: it is
# exactly what ``analyze`` returns, because the weighted postorder has now been
# composed in. The pins below were taken from ``cholmod_l_analyze`` driven the
# way the port drives it (``supernodal = CHOLMOD_SIMPLICIAL``, ``nmethods = 1``
# with ``method[0].ordering = CHOLMOD_AMD``) and each one was cross-checked
# against ``sksparse.cho_factor(A, order="amd")`` before being written down.
#
# The six upstream files this stage ports are byte-identical between v7.6.0 and
# v7.12.2 apart from ``cholmod_analyze.c`` saving and restoring
# ``Common->try_catch`` rather than clearing it, which is error-reporting state.
# So a system CHOLMOD at either tag is a valid oracle here.


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
    M = sp.csc_array(M)
    tri = sp.triu(M) if stype > 0 else sp.tril(M)
    T = sp.csc_array(tri.tocsc())
    return _rs.analyze(
        T.shape[0], T.indptr.astype(np.int64), T.indices.astype(np.int64), stype, **kw
    )


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
@pytest.mark.parametrize("stype", [1, -1])
def test_analyze_matches_cholmod(name, M, stype):
    """``Perm`` and the per-column nnz of ``L``, bit-exact to CHOLMOD.

    ``stype`` picks which triangle is passed, not which matrix it is, so both
    readings have to give the same analysis.
    """
    ref = REF_ANALYZE[name]
    got = analyze(M, stype=stype)
    perm = np.asarray(got["perm"])
    colcount = np.asarray(got["colcount"])

    assert perm.shape == (ref.n,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(ref.n))
    np.testing.assert_array_equal(perm[: len(ref.head)], ref.head)
    assert hashlib.sha256(perm.tobytes()).hexdigest()[:16] == ref.perm_sha
    assert hashlib.sha256(colcount.tobytes()).hexdigest()[:16] == ref.colcount_sha

    # Common->lnz, ->fl and ->anz as cholmod_rowcolcounts leaves them: the
    # exact counts, not AMD's upper bounds
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
    got = analyze(M, ordering="natural", default_strategy=False)
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
    """The one place this port can diverge from a CHOLMOD built with the
    Partition module, reported rather than hidden.

    With ``Common->nmethods == 0`` upstream runs AMD and then breaks out of the
    method loop if ``fl < 500*lnz`` or ``lnz < 5*anz`` on AMD's own estimates
    (``cholmod_analyze.c:767-781``). When it breaks, METIS is never called and
    this port's answer is CHOLMOD's by construction. When it does not, METIS is
    tried and may win. ``laplacian3d-24`` is the case where it does: CHOLMOD
    selects METIS there and gets nnz(L) 1.87M against AMD's 2.30M, while
    ``laplacian3d-23``, one grid step smaller, still breaks out.
    """
    ref = REF_ANALYZE[name]
    got = analyze(M)
    assert got["ordering"] == "amd"
    assert got["metis_would_be_tried"] == (ref.default_ordering != 2)

    fl, lnz, anz = got["amd_fl"], got["amd_lnz"], got["amd_anz"]
    assert got["metis_would_be_tried"] == (not (fl < 500 * lnz or lnz < 5 * anz))


@pytest.mark.parametrize("name,M", ANALYZE_CORPUS, ids=[n for n, _ in ANALYZE_CORPUS])
def test_analyze_agrees_with_scikit_sparse(name, M):
    """The same claim as ``test_analyze_matches_cholmod``, against a live
    CHOLMOD rather than pinned literals — so a stale pin cannot pass both.
    ``order="amd"`` pins the method, because the default strategy tries METIS
    on ``laplacian3d-24`` and this port has no METIS."""
    cholmod = pytest.importorskip("sksparse.cholmod")
    got = analyze(M)
    F = cholmod.cho_factor(sp.csc_array(M), order="amd")
    np.testing.assert_array_equal(got["perm"], np.asarray(F.perm))
    np.testing.assert_array_equal(got["colcount"], np.diff(F.L.tocsc().indptr))


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
        _rs.analyze(5, indptr, indices, 1, "metis")
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
