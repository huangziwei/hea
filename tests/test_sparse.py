"""AMD ordering parity — hea's Rust port vs SuiteSparse 7.6.0.

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
    M = sp.csc_array(sp.eye_array(5).tocsc())
    with pytest.raises(ValueError, match="stype must be nonzero"):
        amd_order(M, stype=0)
    with pytest.raises(ValueError, match="indptr has length"):
        _rs.amd_order(6, M.indptr.astype(np.int64), M.indices.astype(np.int64), 1)
    bad = M.indices.astype(np.int64).copy()
    bad[0] = 99
    with pytest.raises(ValueError, match="out of range"):
        _rs.amd_order(5, M.indptr.astype(np.int64), bad, 1)


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
