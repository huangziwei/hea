"""mgcv's ``gdi1`` and ``pls_fit1`` (gdi.c) against mgcv itself:
``_rs.gdi1_c`` and ``_rs.pls_fit1_c`` on the arguments gam.fit3 passed to
``.C(C_gdi1, ...)`` and ``.C(C_pls_fit1, ...)`` in mgcv's own fits, compared
with every array those calls changed. The fixtures in ``tests/fixtures/gdi1``
and ``tests/fixtures/pls_fit1`` are written by
``tests/scripts/make_gdi1_fixtures.R``: the first, middle and last call of
each fit, over REML, ML and GCV/UBRE, Fisher scoring and full
Newton (with negative weights, for REML and for ML), a fixed penalty from
``min.sp``, a smoothing parameter fixed through ``lsp0``, a flat large-lambda
direction, ranks 8 to 28 and 30 to 300 observations.

Under the arithmetic the fixtures were written with (``PINNED_ARITHMETIC``)
every value must be mgcv's to the bit. Elsewhere the BLAS, the C library's
``log`` and the fused multiply-adds round differently: each array is then
compared to a fraction of its largest entry, as the small entries of ``K``,
``w1`` and ``b1`` are sums that cancel: ``1e-10`` for gdi1's derivatives,
``1e-12`` for pls_fit1's fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC, load_hex_fixture

from hea._dispatch import HAVE_RS, rs_fn

_GDI1 = FIXTURE_ROOT / "gdi1"
_CASES = sorted(p.name for p in _GDI1.glob("*.txt.gz"))
_PLS = FIXTURE_ROOT / "pls_fit1"
_PLS_CASES = sorted(p.name for p in _PLS.glob("*.txt.gz"))

pytestmark = pytest.mark.skipif(not HAVE_RS, reason="the gdi1 port is compiled")


def _flat(v) -> list[float]:
    return np.ravel(np.asarray(v, dtype=float), order="F").tolist()


def _call(name: str, root=_GDI1, fn="gdi1_c"):
    fx = load_hex_fixture(root / name)
    args = {k[2:]: _flat(v) for k, v in fx.items() if k.startswith("a_")}
    want = {k[2:]: np.ravel(v, order="F") for k, v in fx.items() if k.startswith("o_")}
    got = rs_fn(fn)(args)
    return args, want, {k: np.asarray(v, dtype=float) for k, v in got.items()}


def _check(args, want, got, off_pin=1e-10):
    for key, w in want.items():
        g = got[key][: w.size]
        if PINNED_ARITHMETIC:
            np.testing.assert_array_equal(g, w, err_msg=key)
        else:
            scale = float(np.max(np.abs(w))) if w.size else 0.0
            np.testing.assert_allclose(g, w, rtol=0, atol=off_pin * scale, err_msg=key)
    for key, g in got.items():
        if key in want:
            continue
        a = np.asarray(args[key], dtype=float)
        if PINNED_ARITHMETIC:
            np.testing.assert_array_equal(g, a, err_msg=key)
        else:
            # left as it came in by mgcv's call, recomputed here
            scale = float(np.max(np.abs(a))) if a.size else 0.0
            np.testing.assert_allclose(
                g[: a.size], a, rtol=0, atol=off_pin * scale, err_msg=key
            )


def test_fixtures_cover_every_branch():
    """The recorded calls reach REML, ML and GCV/UBRE, both Fisher and
    Newton weights, negative weights under REML and under ML, the fixed
    penalty, and ranks below the 12 at which clang starts vectorizing
    ``beta' S beta``."""
    seen = set()
    for name in _CASES:
        fx = load_hex_fixture(_GDI1 / name)
        reml = int(fx["a_REML"][0])
        fisher = int(fx["a_fisher"][0])
        seen.add(("REML", reml))
        seen.add(("fisher", fisher))
        if np.any(fx["a_w"] < 0):
            seen.add(("neg_w", reml))
        if int(fx["a_fixed.penalty"][0]):
            seen.add("fixed penalty")
        if int(fx["a_p"][0]) < 12:
            seen.add("rank below 12")
        if int(fx["a_n"][0]) < 34:
            seen.add("n below 34")
    want = {
        ("REML", 1),
        ("REML", -1),
        ("REML", 0),
        ("fisher", 0),
        ("fisher", 1),
        ("neg_w", 1),
        ("neg_w", -1),
        "fixed penalty",
        "rank below 12",
        "n below 34",
    }
    assert want <= seen, want - seen


@pytest.mark.parametrize("name", _CASES)
def test_gdi1_matches_mgcv(name):
    """Every array ``gdi1`` changed is mgcv's (``K`` in ``X``, the
    coefficients and their derivatives, the deviance, Pearson and ``tr(A)``
    derivatives, the REML or ML penalty in ``rank.tol`` with ``beta' S beta``
    in ``conv.tol``, ``rV``, the rank, and ``E`` and ``rS`` as it leaves
    them); every array it left alone comes back unchanged."""
    args, want, got = _call(name)
    assert int(got["rank.est"][0]) == int(want.get("rank.est", args["rank.est"])[0])
    _check(args, want, got)


@pytest.mark.parametrize("name", _PLS_CASES)
def test_pls_fit1_matches_mgcv(name):
    """The PIRLS steps of the same fits: the coefficients (in ``y``), the
    linear predictor, the penalty and ``E`` with its unidentifiable columns
    dropped are mgcv's; what the call left alone comes back unchanged."""
    _check(*_call(name, _PLS, "pls_fit1_c"), off_pin=1e-12)
