"""mgcv's ``XWXd`` with term selection (``lt``, ``rt``: the C ``XWXd1``)
against mgcv itself, on a discrete design with parametric, single-smooth,
constrained tensor and by-factor terms, through the extension's ports of
``XWXd0`` and ``XWXd1`` and their Python twins (:mod:`hea.models._xwxd`) on
mgcv's layout of hea's design. The fixtures in ``tests/fixtures/xwxd1`` are
written by ``tests/scripts/make_xwxd1_fixtures.R``. Random layouts that reach
every branch of ``XWXijs`` (the weight table, ``indReduce`` either way round,
the direct products, tensor and AR1 weights, the summation convention and each
constraint) put the twins against the ports.

Under the pinned arithmetic every block is mgcv's to the bit, the
sub-blocks formed in mgcv's order (R's ``revsort`` on their costs), which
decides which of two mirrored writes survives. When a row term carries a
tensor constraint and the selected columns outnumber the rows, mgcv's
``XWXd1`` squashes only ``ptot`` columns out of the row padding and returns
the later ones misplaced; hea squashes them all, so there the first ``ptot``
columns are mgcv's and every column is the matching block of mgcv's whole
``X'WX``.
"""

from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC, load_hex_fixture

import hea
from hea._dispatch import HAVE_RS, rs_fn
from hea.models import _xwxd

_XWXD1 = FIXTURE_ROOT / "xwxd1"
_bam = importlib.import_module("hea.models.bam")

# the extension's ports, and the Python twins
IMPLS = [
    pytest.param(
        "rs", marks=pytest.mark.skipif(not HAVE_RS, reason="the extension is absent")
    ),
    "py",
]

# off the pinned arithmetic the BLAS rounds the cross products differently:
# each block to this fraction of its largest entry
_OFF = 5e-12


def _cases() -> list[tuple[str, list[int] | None, list[int] | None, np.ndarray]]:
    out = []
    for line in (_XWXD1 / "cases.txt").read_text().splitlines():
        name, dims, *vals = line.split()
        want = np.array([float.fromhex(v) for v in vals]).reshape(
            [int(x) for x in dims.split(",")], order="F"
        )
        _, lt, _, rt = name.split("_")
        lt = None if lt == "none" else [int(v) - 1 for v in lt.split("-")]
        rt = None if rt == "none" else [int(v) - 1 for v in rt.split("-")]
        out.append((name, lt, rt, want))
    return out


@pytest.fixture(scope="module")
def design():
    d = load_hex_fixture(_XWXD1 / "data.txt")
    df = pl.DataFrame(
        {
            "x": d["x"],
            "z": d["z"],
            "fac": pl.Series(np.array(["a", "b", "c"])[d["fac"].astype(int) - 1]).cast(
                pl.Enum(["a", "b", "c"])
            ),
            "y": d["y"],
        }
    )
    m = hea.models.bam("y ~ fac + s(z) + te(x, z) + s(x, by = fac)", df, discrete=True)
    return m._discrete_design, d["w"], d["width"].astype(int)


def _blocks(lt, rt, width):
    start = np.concatenate([[0], np.cumsum(width)])
    rows = np.concatenate([np.arange(start[i], start[i + 1]) for i in lt])
    cols = np.concatenate([np.arange(start[i], start[i + 1]) for i in rt])
    return rows, cols


def _same(got, want, err_msg):
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(got, want, err_msg=err_msg)
    else:
        scale = float(np.max(np.abs(want)))
        np.testing.assert_allclose(
            got, want, rtol=0, atol=_OFF * scale, err_msg=err_msg
        )


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_xwxd_term_selection_matches_mgcv(design, case, impl, monkeypatch):
    """Each selection is mgcv's ``XWXd(..., lt, rt)``; without ``lt`` mgcv
    returns the whole ``X'WX``, as its dense branch tests ``is.null(lt)``
    twice."""
    if impl == "py":
        monkeypatch.setattr(_bam, "_xwxd0_rs", None)
        monkeypatch.setattr(_bam, "_xwxd1_rs", None)
    des, w, width = design
    name, lt, rt, want = case
    full = _cases()[0][3]
    if lt is None:
        got = _bam._xwxd0_mgcv(des, w, None)
        _same(got, want, name)
        return
    got = _bam._xwxd1_mgcv(des, w, None, lt, rt)
    assert got.shape == want.shape
    rows, cols = _blocks(lt, rt if rt is not None else lt, width)
    misplaced = rows.size < cols.size and any(
        int(des.terms[i].qc) != 0 and len(des.terms[i].Xd_list) > 1 for i in lt
    )
    if not misplaced:
        _same(got, want, name)
        return
    _same(got[:, : rows.size], want[:, : rows.size], name)
    ref = full[np.ix_(rows, cols)]
    tol = 1e-14 if PINNED_ARITHMETIC else _OFF
    np.testing.assert_allclose(
        got, ref, rtol=0, atol=tol * float(np.max(np.abs(ref))), err_msg=name
    )
    assert not np.allclose(want[:, rows.size :], ref[:, rows.size :])


def _layout(rng, n, terms):
    """A random discrete layout in mgcv's arrays. ``terms`` lists each term's
    marginals as ``(rows, columns)``, its constraint code (``qc > 0`` a
    Householder vector, ``qc < 0`` sum-to-zero contrasts on every margin)
    and its number of index columns; a marginal with ``n`` rows gets a
    permutation for its index."""
    xs, kcols, ks0, ks1, m, p, ts, dt, v, qc = ([] for _ in range(10))
    for margs, q, si in terms:
        ts.append(len(m))
        dt.append(len(margs))
        for mq, pq in margs:
            xs.append(rng.standard_normal(mq * pq))
            ks0.append(len(kcols))
            for _ in range(si):
                kcols.append(rng.integers(0, mq, n) if mq < n else rng.permutation(n))
            ks1.append(len(kcols))
            m.append(mq)
            p.append(pq)
        qc.append(q)
        pt = int(np.prod([pq for _, pq in margs]))
        if q > 0:
            h = rng.standard_normal(pt)
            v.extend(h / np.linalg.norm(h) * np.sqrt(2.0))
        elif q < 0:
            dims = [pq for _, pq in margs]
            v.extend([len(dims), *dims, pt - int(np.prod([d - 1 for d in dims]))])
    i32 = np.int32
    return (
        np.concatenate(xs),
        np.concatenate(kcols).astype(i32),
        np.array(ks0 + ks1, dtype=i32),
        np.array(m, dtype=i32),
        np.array(p, dtype=i32),
        np.array(ts, dtype=i32),
        np.array(dt, dtype=i32),
        np.array(v, dtype=float),
        np.array(qc, dtype=i32),
    )


# (n, terms, AR1 weights, the (lt, rt) selections for XWXd1); between them
# they reach every branch of XWXijs both ways round
_LAYOUTS = [
    (120, [([(120, 3)], 0, 1), ([(6, 4)], 0, 1), ([(5, 3), (4, 3)], -1, 1)], False,
     [([0, 1], []), ([], [2]), ([2], [0, 1])]),
    (120, [([(120, 3)], 0, 1), ([(6, 4)], 1, 1), ([(5, 3), (4, 3)], 1, 1)], True,
     [([0, 1], []), ([2], [1]), ([1, 2], [0, 2])]),
    (150, [([(40, 18)], 0, 1), ([(50, 17)], 0, 1), ([(9, 3), (8, 3)], 0, 1)], False,
     [([0, 1], [1, 2]), ([2], [0])]),
    (150, [([(40, 18)], 0, 1), ([(50, 17)], 0, 1), ([(9, 3), (8, 3)], 0, 1)], True,
     [([0, 1], [1, 2]), ([2], [0]), ([1], [])]),
    (150, [([(30, 5)], 0, 2), ([(35, 6), (6, 3)], -1, 2), ([(7, 3)], 0, 2)], False,
     [([0], [1, 2]), ([], [1])]),
    (150, [([(30, 5)], 0, 2), ([(35, 6), (6, 3)], -1, 2), ([(7, 3)], 0, 2)], True,
     [([0], [1, 2]), ([], [1]), ([2], [1])]),
    (100, [([(4, 3), (5, 2)], 0, 1), ([(6, 2), (3, 3)], 0, 1)], True,
     [([0], [1]), ([1], [0]), ([], [0, 1])]),
    (100, [([(30, 20), (5, 2)], 0, 1), ([(40, 19), (3, 3)], 1, 1)], True,
     [([0], [1]), ([1], [0])]),
    (100, [([(30, 20), (5, 2)], 0, 1), ([(40, 19), (3, 3)], 1, 1)], False,
     [([0], [1]), ([1], [0]), ([], [0, 1])]),
    (90, [([(90, 4)], 0, 1), ([(90, 3)], 0, 1)], True, [([0], [1]), ([], [0, 1])]),
    (200, [([(60, 16)], 0, 1), ([(12, 4), (12, 5)], 0, 1)], True,
     [([0], [1]), ([1], [0])]),
    (200, [([(60, 16)], 0, 1), ([(12, 4), (12, 5)], 0, 1)], False,
     [([0], [1]), ([1], [0])]),
    (150, [([(50, 18)], 0, 1), ([(40, 20)], 0, 1)], False, [([0], [1]), ([1], [0])]),
    (150, [([(50, 18)], 0, 1), ([(40, 20)], 0, 1)], True, [([0], [1]), ([1], [0])]),
    (150, [([(5, 3), (30, 18)], 0, 1), ([(40, 20)], 0, 1), ([(4, 2), (35, 17)], 1, 1)],
     False, [([0, 1, 2], []), ([2], [0]), ([1], [2])]),
    (150, [([(5, 3), (30, 18)], 0, 1), ([(40, 20)], 0, 1), ([(4, 2), (35, 17)], 1, 1)],
     True, [([0, 1, 2], []), ([2], [0]), ([1], [2])]),
    (150, [([(10, 3), (30, 6)], 0, 1), ([(8, 3), (40, 7)], -1, 1)], False,
     [([0], [1]), ([1], [0])]),
    (150, [([(10, 3), (30, 6)], 0, 1), ([(8, 3), (40, 7)], -1, 1)], True,
     [([0], [1]), ([1], [0])]),
    (150, [([(10, 3), (30, 18)], 0, 1), ([(8, 3), (40, 17)], 0, 1)], False,
     [([0], [1]), ([1], [0])]),
    (150, [([(10, 3), (30, 18)], 0, 1), ([(8, 3), (40, 17)], 0, 1)], True,
     [([0], [1]), ([1], [0])]),
]  # fmt: skip


def _agree(got, want, err_msg):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, err_msg
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(got, want, err_msg=err_msg)
    else:
        scale = float(np.max(np.abs(want)))
        np.testing.assert_allclose(
            got, want, rtol=0, atol=1e-13 * scale, err_msg=err_msg
        )


@pytest.mark.skipif(not HAVE_RS, reason="the extension is absent")
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("case", range(len(_LAYOUTS)))
def test_xwxd_twins_match_ports(case, seed):
    """``XWXd0`` and ``XWXd1`` (each selection) by the Python twins are the
    extension's to the bit under the pinned arithmetic."""
    n, terms, ar, sels = _LAYOUTS[case]
    rng = np.random.default_rng(100 * seed + case)
    lay = _layout(rng, n, terms)
    w = rng.uniform(0.1, 2.0, n)
    arw = rng.uniform(0.2, 1.5, 2 * n - 1) if ar else np.zeros(0)
    want, pw = rs_fn("xwxd0")(*lay[:1], w, *lay[1:], arw)
    got, pg = _xwxd.xwxd0(*lay[:1], w, *lay[1:], arw)
    assert pg == pw
    _agree(got, want, "xwxd0")
    for lt, rt in sels:
        want = rs_fn("xwxd1")(*lay[:1], w, *lay[1:], arw, lt, rt)
        got = _xwxd.xwxd1(*lay[:1], w, *lay[1:], arw, lt, rt)
        assert tuple(got[1:]) == tuple(want[1:]), (lt, rt)
        _agree(got[0], want[0], f"xwxd1 {lt} {rt}")


@pytest.mark.parametrize("impl", IMPLS)
def test_xwxd_rejects_bad_layouts(impl):
    """Inconsistent dimensions, out-of-range indices and unknown terms are
    refused, as the ports refuse them."""
    xwxd0 = rs_fn("xwxd0") if impl == "rs" else _xwxd.xwxd0
    xwxd1 = rs_fn("xwxd1") if impl == "rs" else _xwxd.xwxd1
    rng = np.random.default_rng(9)
    x, k, ks, m, p, ts, dt, v, qc = _layout(
        rng, 30, [([(6, 3)], 0, 1), ([(5, 2)], 0, 1)]
    )
    w = np.ones(30)
    arw = np.zeros(0)
    with pytest.raises(ValueError, match="inconsistent"):
        xwxd0(x, w, k, ks[:3], m, p, ts, dt, v, qc, arw)
    with pytest.raises(ValueError, match="index columns"):
        xwxd0(x, w, k, np.array([0, 1, 1, 9], dtype=np.int32), m, p, ts, dt, v, qc, arw)
    bad = k.copy()
    bad[0] = 6
    with pytest.raises(IndexError, match="exceeds"):
        xwxd0(x, w, bad, ks, m, p, ts, dt, v, qc, arw)
    with pytest.raises(ValueError, match="row terms"):
        xwxd1(x, w, k, ks, m, p, ts, dt, v, qc, arw, [], [])
    with pytest.raises(IndexError, match="no such term"):
        xwxd1(x, w, k, ks, m, p, ts, dt, v, qc, arw, [2], [])
