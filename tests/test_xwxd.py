"""mgcv's ``XWXd`` with term selection (``lt``, ``rt``: the C ``XWXd1``)
against mgcv itself, on a discrete design with parametric, single-smooth,
constrained tensor and by-factor terms, through the Rust ports of ``XWXd0``
and ``XWXd1`` on mgcv's layout of hea's design. The fixtures in
``tests/fixtures/xwxd1`` are written by ``tests/scripts/make_xwxd1_fixtures.R``.

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
from hea._dispatch import HAVE_RS

_XWXD1 = FIXTURE_ROOT / "xwxd1"
_bam = importlib.import_module("hea.models.bam")

pytestmark = pytest.mark.skipif(not HAVE_RS, reason="the XWXd ports are compiled")

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


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_xwxd_term_selection_matches_mgcv(design, case):
    """Each selection is mgcv's ``XWXd(..., lt, rt)``; without ``lt`` mgcv
    returns the whole ``X'WX``, as its dense branch tests ``is.null(lt)``
    twice."""
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
