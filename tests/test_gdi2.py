"""mgcv's C ``gdi2`` (gdi.c:1953) against mgcv itself, through the Rust port
``hea._rs.gdi2_c``: each recorded ``.C(C_gdi2, ...)`` call of gam.fit4 is
replayed and every array it changed compared. The fixtures in
``tests/fixtures/gdi2`` are written by ``tests/scripts/make_gdi2_fixtures.R``
from fits of nb (REML and ML, estimated and fixed theta), tw, scat (negative
weights), betar, ocat and ziP, and nb with zero prior weights (gdi2's
``type = 1``).

Under ``PINNED_ARITHMETIC`` every array must be mgcv's to the bit; elsewhere
to ``1e-10`` of its largest entry.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, load_hex_fixture
from test_gdi import _call, _check

from hea._dispatch import HAVE_RS

_GDI2 = FIXTURE_ROOT / "gdi2"
_CASES = sorted(p.name for p in _GDI2.glob("*.txt.gz"))

pytestmark = pytest.mark.skipif(not HAVE_RS, reason="the gdi2 port is compiled")


@pytest.mark.parametrize("name", _CASES)
def test_gdi2_matches_mgcv(name):
    """Coefficients and their derivatives, the deviance's, penalty's and
    log determinant's derivatives, ``rV``, ``K`` (in ``X``), the weights'
    derivatives and the rank are mgcv's; arrays mgcv's call left alone are
    left alone."""
    args, want, got = _call(name, root=_GDI2, fn="gdi2_c")
    _check(args, want, got)


def test_gdi2_fixtures_cover_every_branch():
    """The recorded calls reach REML and ML, one and two likelihood
    parameters (gam.fit4 passes a fixed theta too, gam.fit4.r:260),
    negative weights, and ``type = 1``."""
    seen = set()
    for name in _CASES:
        fx = load_hex_fixture(_GDI2 / name)
        seen.add("ml" if fx["a_ldet"][0] < 0 else "reml")
        seen.add(f"theta{int(fx['a_n.theta'][0])}")
        seen.add("type1" if fx["a_type"][0] == 1 else "type0")
        if np.any(np.asarray(fx["a_w"]) < 0):
            seen.add("negative w")
    assert seen == {"ml", "reml", "theta1", "theta2", "type0", "type1", "negative w"}
