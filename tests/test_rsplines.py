"""B-splines as R evaluates them: ``splines::splineDesign``, ``bs()`` and
``ns()``, and mgcv's B-spline smooth classes (``ps``, ``cp``, ``ad``)
through ``smoothCon`` and ``PredictMat``, against R's own outputs in
``tests/fixtures/splines`` (written by
``tests/scripts/make_spline_fixtures.R``).

The inputs are rebuilt from R's generator (:mod:`hea.R.rng`) and plain
arithmetic, so they hold R's bits. Under the arithmetic the fixtures were
written with (``PINNED_ARITHMETIC``: arm64, where R's ``splines.so`` fuses
two statements of the de Boor recurrence) every output must be R's to the
bit; elsewhere to a tolerance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC

from hea._rsplines import spline_design
from hea.formula import _bs_basis, _ns_basis, expand, materialize_smooths, parse
from hea.R.rng import RGenerator

FIX = FIXTURE_ROOT / "splines"


def _read(name: str, shape) -> np.ndarray:
    return np.fromfile(FIX / name, dtype="<f8").reshape(shape, order="F")


def _check(got: np.ndarray, name: str) -> None:
    ref = _read(name, got.shape)
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(got, ref)
    else:
        scale = max(1.0, float(np.abs(ref).max()))
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * scale)


def _design_inputs():
    g = RGenerator(21)
    knots = np.sort(-2 + 5 * g.uniform(size=12))
    x = knots[3] + (knots[8] - knots[3]) * g.uniform(size=60)
    x2 = -3 + 7 * g.uniform(size=40)
    k4 = np.concatenate([np.zeros(4), np.sort(g.uniform(size=5)), np.ones(4)])
    x4 = np.concatenate([g.uniform(size=30), [0.0, 1.0, k4[5]]])
    return knots, x, x2, k4, x4


def test_spline_design_matches_r():
    knots, x, x2, k4, x4 = _design_inputs()
    _check(spline_design(knots, x, 4), "sd_values.bin")
    _check(spline_design(knots, x, 4, np.tile(np.arange(4), 15)), "sd_derivs.bin")
    _check(spline_design(knots, x2, 3, outer_ok=True), "sd_outer.bin")
    _check(spline_design(k4, x4, 4), "sd_repeated.bin")
    _check(spline_design(k4, x4, 4, 2), "sd_repeated_d2.bin")


def test_spline_design_rejects_x_outside_without_outer_ok():
    knots, _x, x2, _k4, _x4 = _design_inputs()
    with pytest.raises(ValueError, match="outer.ok"):
        spline_design(knots, x2, 3)


def test_bs_ns_match_r_with_extrapolation():
    x = -2 + 7 * RGenerator(22).uniform(size=120)
    kn = np.array([-0.5, 0.75, 1.875, 3.25])
    bnd = (-1.0, 4.0)
    assert np.any(x < bnd[0]) and np.any(x > bnd[1])
    _check(_bs_basis(x, 3, bnd, kn, None, False), "bs3.bin")
    _check(_bs_basis(x, 2, bnd, kn, None, True), "bs2_int.bin")
    _check(_ns_basis(x, bnd, kn, None, False), "ns.bin")
    _check(_ns_basis(x, bnd, kn, None, True), "ns_int.bin")


@pytest.mark.parametrize(
    ("name", "spec"),
    [
        ("ps", "s(x, bs='ps')"),
        ("cp", "s(x, bs='cp')"),
        ("ad", "s(x, bs='ad', k=20)"),
        ("ad2", "s(x, z, bs='ad', k=8)"),
    ],
)
def test_bspline_smooths_match_mgcv(name: str, spec: str):
    g = RGenerator(23)
    x = 3 * g.uniform(size=150)
    z = -1 + 2 * g.uniform(size=150)
    data = pl.DataFrame({"x": x, "z": z, "y": x})
    nd = pl.DataFrame(
        {
            "x": [-0.375, 0.0, 1.25, 2.9375, 3.25, 3.625, 0.75],
            "z": [0.125, -1.25, 0.5, 0.875, 1.375, -0.25, 0.0],
        }
    )
    (blk,) = materialize_smooths(expand(parse("y ~ " + spec)), data)[0]
    _check(blk.X, f"sc_{name}_X.bin")
    for j, S in enumerate(blk.S, start=1):
        _check(S, f"sc_{name}_S{j}.bin")
    _check(np.asarray(blk.spec.predict_mat(nd), dtype=float), f"sc_{name}_P.bin")


@pytest.mark.parametrize(
    ("name", "spec", "n_pen"),
    [
        ("b32", "s(x, bs='bs')", 1),
        ("b5", "s(x, bs='bs', k=5)", 1),
        ("b321", "s(x, bs='bs', m=c(3,2,1))", 2),
        ("b42", "s(x, bs='bs', m=c(4,2))", 1),
        ("b21", "s(x, bs='bs', m=c(2,1))", 1),
        ("b33", "s(x, bs='bs', m=c(3,3))", 1),
        ("b11", "s(x, bs='bs', m=c(1,1))", 1),
        ("b52", "s(x, bs='bs', m=c(5,2), k=12)", 1),
    ],
)
def test_bs_smooth_penalties_match_mgcv(name: str, spec: str, n_pen: int):
    # smooth.construct.bs.smooth.spec's penalties: W1 = t(P) %*% H %*% P, the
    # band of W factored by mgcv's bandchol (LAPACK dpbtrf), crossprod.
    from hea.formula import _bs_raw

    data = pl.DataFrame({"x": RGenerator(9).uniform(size=80)})
    call = expand(parse("y ~ " + spec)).smooths[0]
    _X, S_list, _kn, _m0 = _bs_raw(call, data, ["x"])
    assert len(S_list) == n_pen
    for j, S in enumerate(S_list, 1):
        _check(S, f"bspen_{name}_S{j}.bin")


def test_bs_smooth_with_one_interval_fails_as_mgcv_does():
    # k = m[1] + 1 leaves one knot interval; mgcv's band assembly then hands
    # bandchol a square matrix it reads as a full one, and stops.
    from hea.formula import _bs_raw

    data = pl.DataFrame({"x": RGenerator(9).uniform(size=80)})
    call = expand(parse("y ~ s(x, bs='bs', k=4)")).smooths[0]
    with pytest.raises(ValueError, match="not positive definite"):
        _bs_raw(call, data, ["x"])


@pytest.mark.parametrize("name", ["u", "r", "z", "e"])
@pytest.mark.parametrize("df", [4, 5, 7, 12])
def test_bs_ns_df_knots_match_r(name: str, df: int):
    # The interior knots bs()/ns() choose for df: quantile type 7 of the
    # points inside the boundary knots, ties at a boundary shoved inside.
    import warnings

    from hea.formula import _r_df_knots

    x = np.fromfile(FIX / f"dfk_{name}_x.bin")
    bnd = (float(x.min()), float(x.max()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got_bs = _r_df_knots(x, df - 3, bnd, ns=False)
        got_ns = _r_df_knots(x, df - 1, bnd, ns=True)
    _check(got_bs, f"dfk_{name}_bs{df}.bin")
    _check(got_ns, f"dfk_{name}_ns{df}.bin")
