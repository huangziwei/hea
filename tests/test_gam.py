"""End-to-end tests for ``hea.gam``.

Sections (top → bottom):

1. **mgcv-oracle parity** — each test pins the printed numerical
   outputs of ``mgcv::gam(..., method=...)`` on a fixed dataset so
   the hea port can be validated against the canonical R/mgcv
   results. Coverage spans tp / cr / ps basis types; REML, ML, and
   GCV.Cp criteria; parametric + smooth combinations; tensor-product
   (te) smooths; by=factor multi-block smooths; random-effect
   (bs='re') smooths; family/link plumbing (Gaussian, Gamma, IG,
   Tweedie/tw, Binomial); LHS expressions; offset(...) handling;
   plot_smooth dispatching; select=TRUE null-space penalties;
   gam.check. Tolerances are set per-quantity — ρ=log(sp) typically
   pins to 4 decimals for tp and ps; edf, σ², and the criterion
   typically agree to 4–5 decimals. Smooth basis coefficients
   themselves are not pinned (identifiable only up to mgcv's
   reparametrization), but per-smooth edf totals are.

2. **vis()** — port of mgcv's ``vis.gam``. Correctness invariant is
   ``vis(view) == predict(grid)``: the method just calls predict on a
   regular grid over two view variables, with all other variables
   held at their typical (median / modal) value. The predict
   end-to-end vs mgcv comparison lives in ``test_smooths.py``; here
   we check grid construction, dtype handling, SE pipeline,
   ``too_far`` masking, and factor-axis support.

3. **get_difference() — port of itsadug::get_difference** —
   numerical-equivalence tests against R's ``itsadug``. Per case
   under ``tests/fixtures/itsadug_plot_diff/``, re-fit the same
   model in hea, replay the same ``(comp, cond, f, sim_ci,
   rm_ranef)`` arguments, and compare per-row ``difference`` and
   ``CI`` against R's output. The deterministic ``se_fit`` lands at
   high precision; the empirical ``crit`` matches to Monte-Carlo SE
   (Python and R don't share an RNG).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest
from conftest import assert_fp_equiv as _assert_fp_equiv
from conftest import load_dataset

from hea.family import Gamma, Poisson, Tweedie, tw
from hea.models import gam, glm

matplotlib.use("Agg")  # headless — must be set before pyplot import below.
import matplotlib.pyplot as plt


def _allclose(actual, expected, *, atol, name=""):
    np.testing.assert_allclose(
        actual, expected, atol=atol, err_msg=f"{name}: {actual} vs {expected}"
    )


def _assert_param(m, col, est, *, atol=5e-3):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=atol, err_msg=f"param[{col}]")


def test_mcycle_tp_REML():
    """gam(accel ~ s(times), data=mcycle, method="REML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")

    assert m.n == 133
    _allclose(m.sp[0], 7.758035879e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.624691, atol=5e-4, name="edf_total")
    _allclose(m.sigma_squared, 506.3529, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 616.1420, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.7831484, atol=5e-4, name="r2adj")
    _allclose(m.loglike, -597.8345, atol=5e-3, name="loglike")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 8.624691, atol=5e-4, name="edf[s(times)]")


def test_mcycle_tp_GCV():
    """gam(accel ~ s(times), data=mcycle, method="GCV.Cp")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="GCV.Cp")

    assert m.n == 133
    _allclose(m.sp[0], 6.195886e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.693314, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 506.0017, atol=5e-3, name="sigma2")
    _allclose(m.GCV_score, 545.7792, atol=5e-3, name="GCV")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)


def test_gamSim_eg1_four_smooths_REML():
    """gam(y ~ s(x0)+s(x1)+s(x2)+s(x3), data=gamSim(eg=1), method="REML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)", d, method="REML")

    assert m.n == 400
    _allclose(m.edf_total, 15.88548, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.897969, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 861.1296, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.7156242, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x0)"], 3.020970, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.843246, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.019844, atol=5e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 1.001421, atol=5e-2, name="edf[s(x3)]")


def test_gamSim_eg1_tensor_REML():
    """gam(y ~ s(x0) + te(x1, x2), data=gamSim(eg=1), method="REML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + te(x1, x2)", d, method="REML")

    assert m.n == 400
    _allclose(m.edf_total, 17.55095, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 4.386049, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 881.5002, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.6800164, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x0)"], 3.097122, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["te(x1,x2)"], 13.45382, atol=5e-3, name="edf[te]")

    assert m.sp.shape == (3,)  # 1 for s(x0) + 2 for te
    _allclose(m.sp[0], 1.492971, atol=5e-3, name="sp[s(x0)]")
    _allclose(m.sp[1], 33.05461, atol=1e-1, name="sp[te-1]")
    _allclose(m.sp[2], 0.0882241, atol=5e-3, name="sp[te-2]")


def test_byfactor_smooth_REML():
    """gam(y ~ g + s(x, by=g), data=<synth>, method="REML")"""
    d = load_dataset("synthetic", "seed_synth_gam_by_factor")
    if d.schema["g"] != pl.Enum(["A", "B", "C"]):
        d = d.with_columns(pl.col("g").cast(pl.Enum(["A", "B", "C"])))
    m = gam("y ~ g + s(x, by=g)", d, method="REML")

    assert m.n == 300
    assert m.sp.shape == (3,)
    _allclose(m.edf_total, 21.36070, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.04265686, atol=5e-4, name="sigma2")
    _allclose(m.REML_criterion / 2, -9.890208, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.9164980, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 0.02332958, atol=5e-3)
    _assert_param(m, "gB", -0.06749164, atol=5e-3)
    _assert_param(m, "gC", 0.63793878, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x):gA"], 6.953522, atol=5e-3, name="edf[s(x):gA]")
    _allclose(m.edf_by_smooth["s(x):gB"], 6.745235, atol=5e-3, name="edf[s(x):gB]")
    _allclose(m.edf_by_smooth["s(x):gC"], 4.661939, atol=5e-3, name="edf[s(x):gC]")


def test_mcycle_ps_REML():
    """gam(accel ~ s(times, bs="ps"), data=mcycle, method="REML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times, bs='ps')", d, method="REML")

    assert m.n == 133
    _allclose(m.sp[0], 0.09454488, atol=5e-3, name="sp")
    _allclose(m.edf_total, 8.801932, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 727.3234, atol=5e-2, name="sigma2")
    _allclose(m.REML_criterion / 2, 637.9549, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.6885152, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 7.801932, atol=5e-3, name="edf[s(times)]")


def test_gamSim_eg1_overlap_gamSide_REML():
    """gam(y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2), method='REML')"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2)", d, method="REML")

    assert m.n == 400
    assert m.bhat.shape[1] == 42, (
        f"gam.side drop failed: p={m.bhat.shape[1]} (expected 42)"
    )

    assert m.sp.shape == (4,)
    _allclose(m.sp[1], 7.998938e-03, atol=5e-4, name="sp[s(x2)]")

    _allclose(m.sigma_squared, 4.149471, atol=5e-2, name="sigma2")
    _allclose(m.r_squared_adjusted, 0.697276, atol=5e-3, name="r2adj")
    _allclose(m.REML_criterion / 2, 866.7819, atol=5e-1, name="REML/2")
    _assert_param(m, "(Intercept)", 7.642771, atol=5e-3)
    _assert_param(m, "x0", 0.394401, atol=5e-3)

    _allclose(m.edf_total, 13.836828, atol=5e-1, name="edf_total")
    _allclose(m.edf_by_smooth["s(x1)"], 2.790683, atol=2e-1, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.044964, atol=5e-2, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["te(x1,x2)"], 1.001181, atol=5e-1, name="edf[te]")


def test_gam_side_repeated_and_partial_overlap_matches_mgcv():
    """gam.side beyond strict nesting (mgcv.r:565-720, with.pen=TRUE):
    X1 collects EVERY earlier smooth sharing ANY variable — repeated
    same-variable smooths and partial tensor overlaps included — with
    each design augmented by a scaled √total-penalty block
    (augment.smX) so fixDependence mostly sees null-space dependencies;
    dropped-column penalties get their rank recomputed (R's dqrdc2 QR)
    and rank-0 penalties are deleted. mgcv 1.9-4 pins; set.seed(21)
    replicated via hea.R.rng. Working-infinity sp directions carry the
    usual optimizer endpoint scatter (REML agrees to ~5e-8), so those
    entries are asserted as >1e4 rather than pinned.
    """
    import warnings as _warnings

    from hea.R.rng import RGenerator

    g = RGenerator(21)
    n = 200
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    w = g.uniform(0, 1, n)
    y = (
        np.sin(2 * np.pi * x)
        + np.cos(np.pi * z)
        + 0.5 * z * w
        + g.normal(0, 1, n) * 0.3
    )
    df = pl.DataFrame({"x": x, "z": z, "w": w, "y": y})

    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        m1 = gam("y ~ s(x) + s(x, bs='cr', k=8)", df, method="REML")
    assert any("repeated 1-d" in str(r.message) for r in rec)
    assert m1.p == 16
    np.testing.assert_allclose(m1.sp[1], 13.2410824, rtol=1e-4)
    assert m1.sp[0] > 1e2  # ~1-edf smooth at working infinity
    np.testing.assert_allclose(m1.REML_criterion / 2, 224.7937630575, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m1.edf_total, 6.26987813, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[:3],
        [-0.9281402580, 1.0988953071, -0.7586534549],
        rtol=0,
        atol=1e-5,
    )

    m2 = gam("y ~ te(x, z, k=c(4,4)) + te(z, w, k=c(4,4))", df, method="REML")
    assert m2.p == 30
    np.testing.assert_allclose(
        m2.sp[:3], [0.02893250712, 88.64266152, 0.567903102], rtol=1e-4
    )
    assert m2.sp[3] > 1e4
    np.testing.assert_allclose(m2.REML_criterion / 2, 71.2556613189, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m2.edf_total, 13.83053789, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(m2.fitted_values)[:3],
        [-0.0788588401, 1.0766835318, 0.0468759735],
        rtol=0,
        atol=1e-6,
    )

    m3 = gam("y ~ s(x) + s(z) + te(x, z, k=c(5,5))", df, method="REML")
    assert m3.p == 41
    np.testing.assert_allclose(m3.sp[:2], [0.0125571115, 0.1251747408], rtol=1e-4)
    np.testing.assert_allclose(m3.REML_criterion / 2, 72.8001350418, rtol=0, atol=1e-5)
    m4 = gam("y ~ s(x) + s(z) + s(x, z, k=20)", df, method="REML")
    assert m4.p == 36
    np.testing.assert_allclose(m4.sp[:2], [0.01251179556, 0.123522571], rtol=1e-4)
    np.testing.assert_allclose(m4.REML_criterion / 2, 72.5576488704, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m4.edf_total, 12.31004124, rtol=1e-5)


# ---------------------------------------------------------------------------
# 8) nlme::Machines — re smooths (Wood 2017 §6.5 example)
# ---------------------------------------------------------------------------


def test_machines_re_smooths_REML():
    """gam(score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re'),
       data=Machines, method='REML') and the by=Machine variant.
    Two random-effect formulations from Wood 2017 §6.5. Exercises:
    AIC uses df = sum(edf2)+1 (Wood 2017 §6.11.3) with edf2 including
    """
    d = load_dataset("nlme", "Machines")

    b1 = gam(
        "score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re')",
        data=d,
        method="REML",
    )
    assert b1.n == 54
    assert b1.p == 27  # full mgcv design, no gam.side surgery on re smooths
    # mgcv reference (mgcv 1.9-4, REML, gam.vcomp). hea's outer Newton now
    # uses mgcv's exact stopping rule (gam.fit3.r:1646-1658,
    # ``max(|grad|) ≤ score.scale·conv.tol·5`` AND
    # ``|Δscore| ≤ score.scale·conv.tol``) at the same default ``conv.tol=1e-6``.
    # That puts hea and mgcv inside the same stopping band; residual drift
    # is the natural noise of where each implementation lands within the
    # band (≈ a few×1e-3 on the most leveraged CI, the small-std re-by-factor
    # smooth s(Worker):MachineA).
    _allclose(b1.edf_total, 17.76461, atol=5e-5, name="b1.edf")
    _allclose(b1.edf1_total, 17.99523, atol=5e-5, name="b1.edf1")
    _allclose(b1.edf2_total, 17.85995, atol=5e-5, name="b1.edf2")
    assert b1.edf_total < b1.edf2_total <= b1.edf1_total
    _allclose(b1.sigma_squared, 0.92463, atol=5e-5, name="b1.sigma2")
    _allclose(b1.loglike, -63.73532, atol=5e-4, name="b1.loglike")
    _allclose(b1.AIC, 165.19055, atol=5e-4, name="b1.AIC")
    _assert_param(b1, "(Intercept)", 52.3556, atol=5e-3)
    assert b1.edf_by_smooth["s(Worker)"] > 3.0
    assert b1.edf_by_smooth["s(Machine,Worker)"] > 8.0

    vc = b1.vcomp
    assert vc.shape == (3, 4)
    assert vc["name"].to_list() == ["s(Worker)", "s(Machine,Worker)", "scale"]
    expected = {
        "s(Worker)": (4.78106, 2.24987, 10.15997),
        "s(Machine,Worker)": (3.72952, 2.38281, 5.83737),
        "scale": (0.96158, 0.76325, 1.21143),
    }
    for nm, (sd, lo, hi) in expected.items():
        row = vc.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"vcomp {nm}.std")
        _allclose(row["lower"], lo, atol=5e-4, name=f"vcomp {nm}.lo")
        _allclose(row["upper"], hi, atol=5e-4, name=f"vcomp {nm}.hi")

    b2 = gam(
        "score ~ Machine + s(Worker, bs='re') + s(Worker, bs='re', by=Machine)",
        data=d,
        method="REML",
    )
    assert b2.n == 54
    assert b2.sp.shape == (4,)
    _allclose(b2.edf_total, 17.64453, atol=5e-5, name="b2.edf")
    _allclose(b2.edf2_total, 17.98557, atol=5e-5, name="b2.edf2")
    _allclose(b2.sigma_squared, 0.92463, atol=5e-5, name="b2.sigma2")
    _allclose(b2.loglike, -63.82464, atol=5e-4, name="b2.loglike")
    _allclose(b2.AIC, 165.62043, atol=5e-4, name="b2.AIC")

    vc2 = b2.vcomp
    assert vc2.shape == (5, 4)
    assert vc2["name"].to_list() == [
        "s(Worker)",
        "s(Worker):MachineA",
        "s(Worker):MachineB",
        "s(Worker):MachineC",
        "scale",
    ]
    expected_b2 = {
        "s(Worker)": (3.78595, 1.79873, 7.96861),
        "s(Worker):MachineA": (1.94032, 0.25319, 14.86973),
        "s(Worker):MachineB": (5.87402, 2.98833, 11.54628),
        "s(Worker):MachineC": (2.84547, 0.82993, 9.75584),
        "scale": (0.96158, 0.76325, 1.21143),
    }
    for nm, (sd, lo, hi) in expected_b2.items():
        row = vc2.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"b2 vcomp {nm}.std")
        _allclose(row["lower"], lo, atol=5e-4, name=f"b2 vcomp {nm}.lo")
        if nm.startswith("s(Worker):Machine"):
            np.testing.assert_allclose(
                row["upper"],
                hi,
                rtol=2e-3,
                err_msg=f"b2 vcomp {nm}.hi: {row['upper']} vs {hi}",
            )
        else:
            _allclose(row["upper"], hi, atol=5e-4, name=f"b2 vcomp {nm}.hi")


def test_data_helper_applies_schema_sidecar():
    """`hea.data()` must restore R's factor type via the JSON schema sidecar."""
    from hea import data

    d = data("Machines", "nlme")
    assert isinstance(d.schema["Worker"], pl.Enum), (
        f"Worker should be pl.Enum, got {d.schema['Worker']}"
    )
    assert isinstance(d.schema["Machine"], pl.Enum), (
        f"Machine should be pl.Enum, got {d.schema['Machine']}"
    )


def test_factor_helper():
    """`hea.R.factor()` is the polars equivalent of R's factor() — the
    user-side fix for wild-data Int64-stored factor columns.
    """
    import rdatasets

    from hea.formula import _ORDERED_COLS_CV, set_ordered_cols
    from hea.R import factor

    df = pl.from_pandas(rdatasets.data("nlme", "Machines")).drop("rownames")
    assert df.schema["Worker"] == pl.Int64  # the wild-data scenario

    out = factor(df["Worker"])
    assert isinstance(out.dtype, pl.Enum)
    assert out.dtype.categories.to_list() == ["1", "2", "3", "4", "5", "6"]
    assert out.name == "Worker"  # preserved → with_columns replaces

    out2 = factor(df["Worker"], levels=["6", "2", "4", "1", "3", "5"])
    assert out2.dtype.categories.to_list() == ["6", "2", "4", "1", "3", "5"]

    set_ordered_cols(frozenset())  # clean slate
    df_fixed = df.with_columns(factor(df["Worker"]))
    m = gam("score ~ Machine + s(Worker, bs='re')", data=df_fixed, method="REML")
    assert m.edf_total > 4.0, f"factor() didn't fix the re basis: edf={m.edf_total}"

    set_ordered_cols(frozenset())
    factor(df["Worker"], ordered=True)
    assert "Worker" in _ORDERED_COLS_CV.get()
    factor(df["Worker"], ordered=False)
    assert "Worker" in _ORDERED_COLS_CV.get(), "ordered=False shouldn't unregister"

    test = pl.Series("test", [0, 1, 1, 0, 1])
    out_l = factor(test, labels={0: "negative", 1: "positive"})
    assert out_l.dtype.categories.to_list() == ["negative", "positive"]
    assert out_l.to_list() == [
        "negative",
        "positive",
        "positive",
        "negative",
        "positive",
    ]
    assert out_l.name == "test"

    out_rev = factor(test, labels={1: "positive", 0: "negative"})
    assert out_rev.dtype.categories.to_list() == ["positive", "negative"]

    with pytest.raises(pl.exceptions.InvalidOperationError):
        factor(pl.Series([0, 1, 2]), labels={0: "a", 1: "b"})

    with pytest.raises(ValueError, match="not both"):
        factor(test, levels=[0, 1], labels={0: "a", 1: "b"})

    with pytest.raises(TypeError, match="not a dict"):
        factor(test, levels={0: "negative", 1: "positive"})


def test_factor_deferred_in_mutate_and_select():
    """`hea.R.factor("col")` returns a placeholder so the tidyverse-style
    ``df.mutate(species=hea.R.factor("species"))`` works — the eager Series
    form would force ``df.with_columns(hea.R.factor(df["species"]))``,
    repeating the frame name. ``mutate`` / ``select`` peek at the frame
    to auto-detect Enum levels at call time.
    """
    import hea
    from hea.R import factor

    df = pl.DataFrame({"g": ["b", "a", "b", "a"], "x": [1.0, 2.0, 3.0, 4.0]})

    out = hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor("g"))
    assert isinstance(out.schema["g"], pl.Enum)
    assert out.schema["g"].categories.to_list() == ["a", "b"]

    out2 = hea.tidy.DataFrame._from_pydf(df._df).mutate(
        g=factor("g", levels=["b", "a"])
    )
    assert out2.schema["g"].categories.to_list() == ["b", "a"]

    out3 = hea.tidy.DataFrame._from_pydf(df._df).mutate(
        g=factor("g", labels={"a": "Alpha", "b": "Bravo"})
    )
    assert out3["g"].to_list() == ["Bravo", "Alpha", "Bravo", "Alpha"]

    out4 = hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor(pl.col("g")))
    assert isinstance(out4.schema["g"], pl.Enum)

    out5 = hea.tidy.DataFrame._from_pydf(df._df).select("x", grp=factor("g"))
    assert out5.columns == ["x", "grp"]
    assert isinstance(out5.schema["grp"], pl.Enum)

    df_typo = pl.DataFrame({"g": ["a", "b", "x", "a"]})
    out6 = hea.tidy.DataFrame._from_pydf(df_typo._df).mutate(
        g=factor("g", levels=["a", "b"])
    )
    assert out6["g"].to_list() == ["a", "b", None, "a"]
    with pytest.raises(pl.exceptions.InvalidOperationError):
        hea.tidy.DataFrame._from_pydf(df_typo._df).mutate(
            g=factor("g", levels=["a", "b"], strict=True)
        )

    with pytest.raises(ValueError, match="auto-detect levels"):
        hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor("missing"))


def test_sp_passthrough_matches_optimized():
    """Calling gam(..., sp=m_opt.sp) must give the same fit as the optimized one."""
    d = load_dataset("MASS", "mcycle")
    m_opt = gam("accel ~ s(times)", d, method="REML")
    m_fix = gam("accel ~ s(times)", d, method="REML", sp=m_opt.sp)

    np.testing.assert_allclose(m_fix.sigma_squared, m_opt.sigma_squared, atol=1e-6)
    np.testing.assert_allclose(m_fix.edf_total, m_opt.edf_total, atol=1e-6)
    np.testing.assert_allclose(m_fix.fitted, m_opt.fitted, atol=1e-6)


def test_predict_inSample_matches_fitted():
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    np.testing.assert_array_equal(m.predict()["fit"].to_numpy(), m.fitted)


def test_pirls_init_canonical_inverse_gaussian():
    """IG canonical fit on Wald-distributed data must converge."""
    from hea.family import inverse_gaussian
    from hea.R.rng import RGenerator

    gen = RGenerator(0)
    n = 200
    x = gen.uniform(0.0, 1.0, n)
    mu = 1.5 + 0.5 * np.sin(2 * np.pi * x)  # ∈ [1.0, 2.0], strictly positive
    y = inverse_gaussian().rd(gen, mu, np.ones(n), 1.0)
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", df, family=inverse_gaussian(), method="REML")
    assert m.n == n
    assert np.all(np.isfinite(m._beta))
    assert np.all(np.isfinite(m.fitted))
    assert np.all(m.fitted > 0)
    assert np.all(m.linear_predictors > 0)
    # Unknown-scale family ⇒ log φ enters the outer
    # vector and `m._log_phi_hat` is finite. ``m.scale = m.sigma_squared``
    # is the post-fit scale estimate (mgcv's ``m$sig2 = scale.est``) —
    # Pearson with the Fletcher (2012) correction, mgcv's default
    # estimator (gam.fit3.r:596-603). For canonical IG the correction is
    # non-trivial: s̄ = 3·mean((y−μ̂)/μ̂) ≠ 0, so Fletcher ≠ Pearson here.
    # The optimizer's converged scale ``exp(log φ̂)`` (mgcv's
    # ``reml.scale``) lives on ``m._log_phi_hat`` — for REML the two
    # coincide at the optimum (FOC); for ML they don't.
    assert m._log_phi_hat is not None
    assert np.isfinite(m._log_phi_hat)
    np.testing.assert_allclose(m.scale, m._fletcher_scale, atol=0.0)
    assert np.isfinite(m._pearson_scale)
    assert m.sigma_squared > 0
    np.testing.assert_allclose(m.sigma_squared, 0.821638681219, rtol=5e-6)
    np.testing.assert_allclose(m._pearson_scale, 0.781145231711, rtol=5e-6)
    intercept = m.bhat["(Intercept)"][0]
    assert 0.40 < intercept < 0.80


def test_trees_gamma_log_smoke():
    """trees + Gamma(log), method='REML': pin family-agnostic post-fit values
    against mgcv (those that don't depend on sp), and hea's own
    sp-dependent values as a regression guard."""
    from hea.family import Gamma

    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"), method="REML")

    assert m.family.name == "Gamma"
    assert m.family.link.name == "log"
    assert m.family.scale_known is False

    assert np.all(m.fitted_values > 0)
    np.testing.assert_allclose(
        m.fitted_values,
        np.exp(m.linear_predictors),
        atol=1e-12,
    )
    assert m.fitted is m.fitted_values or np.array_equal(m.fitted, m.fitted_values)

    assert m.n == 31
    np.testing.assert_allclose(m.df_null, 30.0, atol=0.0)

    np.testing.assert_allclose(m.r_squared_adjusted, 0.9744391060, atol=5e-5)
    np.testing.assert_allclose(m.deviance_explained, 0.9782902227, atol=5e-5)
    np.testing.assert_allclose(m.null_deviance, 8.3172012147, atol=5e-7)
    np.testing.assert_allclose(m.deviance, 0.1805645860, atol=5e-4)
    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756440543, atol=5e-3)

    np.testing.assert_allclose(m.sp[1], 0.2112713142, rtol=2e-3)
    np.testing.assert_allclose(m.edf_total, 4.738161, atol=5e-2)
    np.testing.assert_allclose(m.edf2_total, 5.270166, atol=5e-2)
    np.testing.assert_allclose(m.scale, 0.0068696749, atol=5e-5)
    np.testing.assert_allclose(m.sigma_squared, m.scale, atol=0.0)
    np.testing.assert_allclose(m.logLik, -65.9017771491, atol=2e-2)
    np.testing.assert_allclose(m.AIC, 144.3438870069, atol=1e-1)

    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    np.testing.assert_allclose(
        m.BIC, -2.0 * m.logLik + np.log(m.n) * m.npar, atol=1e-10
    )

    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756425861, atol=5e-5)

    np.testing.assert_allclose(
        m.fitted_values[:5],
        [10.62414379, 10.36186212, 10.41212209, 16.42891707, 19.68356227],
        atol=5e-3,
    )

    pearson = m.residuals_of("pearson")
    working = m.residuals_of("working")
    np.testing.assert_allclose(pearson, working, atol=1e-12)
    response = m.residuals_of("response")
    np.testing.assert_allclose(response, m._y_arr - m.fitted_values, atol=0.0)
    np.testing.assert_array_equal(m.residuals, m.residuals_of("deviance"))


def test_gaussian_residual_identities_and_aic_self_consistency():
    """For Gaussian-identity all four residual types collapse to (y-μ).
    Independent of mgcv pins, so this catches future regressions in
    residuals_of without depending on a fixture."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    y = m._y_arr
    mu = m.fitted_values
    target = y - mu
    np.testing.assert_allclose(m.linear_predictors, mu, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("response"), target, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("deviance"), target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("pearson"), target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("working"), target, atol=1e-12)
    np.testing.assert_array_equal(m.residuals, m.residuals_of("deviance"))
    np.testing.assert_allclose(
        np.sum(m.residuals_of("deviance") ** 2), m.deviance, atol=1e-9
    )
    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    with pytest.raises(ValueError):
        m.residuals_of("partial")


def test_reml_finite_for_trees_gamma_log():
    """Sanity: for the converged Gamma(log) fit, `_reml` returns a
    finite value at the hea-current sp. φ̂ is a joint outer variable, so
    this checks the formula is wired up correctly."""
    from hea.family import Gamma
    from hea.models import gam

    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"), method="REML")
    log_phi = float(np.log(m.scale))
    v = m._reml(m._rho_hat, log_phi)
    assert np.isfinite(v)


def test_kcheck_mcycle_matches_mgcv():
    """k.check on `accel ~ s(times)` (1D smooth, REML)."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    ktab = m._k_check(seed=0, n_rep=2000)
    assert ktab[""].to_list() == ["s(times)"]
    np.testing.assert_allclose(ktab["k'"].to_list(), [9.0], atol=0)
    np.testing.assert_allclose(ktab["edf"].to_list(), [8.62469100], atol=5e-5)
    np.testing.assert_allclose(ktab["k-index"].to_list(), [1.14736165], atol=5e-5)
    np.testing.assert_allclose(ktab["p-value"].to_list(), [0.951], atol=1e-12)


def test_kcheck_handles_no_smooths_returns_none():
    """k.check is undefined when there are no smooth blocks. Mirrors
    mgcv: `k.check` returns NULL → `gam.check` skips the table."""
    d = load_dataset("R", "trees")
    m = gam("Volume ~ Height + Girth", d, method="REML")
    assert m._k_check() is None


def test_check_prints_convergence_block(capsys):
    """`gam.check()` runs end-to-end and emits the mgcv-style header."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    m.check(seed=0, k_rep=200, plots=False)
    out = capsys.readouterr().out
    assert "Method: REML" in out
    assert "Optimizer: outer newton" in out
    assert "iteration" in out
    assert "Gradient range" in out
    assert "score " in out and "scale " in out
    assert "Hessian" in out and "eigenvalue range" in out
    assert "Model rank = " in out
    assert "Basis dimension (k) checking" in out
    assert "s(times)" in out


def test_check_no_smooth_path(capsys):
    """When the model has no smooths, the convergence block reports
    `Model required no smoothing parameter selection` (mgcv text) and
    the k-check table is omitted."""
    d = load_dataset("R", "trees")
    m = gam("Volume ~ Height + Girth", d, method="REML")
    m.check(plots=False)
    out = capsys.readouterr().out
    assert "Model required no smoothing parameter selection" in out
    assert "Basis dimension" not in out


def test_lhs_power_brain_matches_mgcv():
    """Wood §7.2: `gam(medFPQ^.25 ~ s(Y, X, k=100), data=brain)`."""
    d = load_dataset("gamair", "brain").filter(pl.col("medFPQ") > 5e-5)
    m = gam("medFPQ^.25 ~ s(Y, X, k=100)", d)
    assert m.n == 1565
    assert m.y.name == "medFPQ^0.25"
    _allclose(m.edf_total, 65.1763, atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.039541, atol=5e-6, name="sigma2")
    _allclose(m.GCV_score, 0.041259, atol=5e-6, name="GCV")


def test_lhs_log_matches_manual_transform():
    """`log(y) ~ ...` should be identical to pre-computing log(y) in
    polars and fitting `log_y ~ ...` on the same RHS."""
    d = load_dataset("R", "trees")
    m_lhs = gam("log(Volume) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns(pl.col("Volume").log().alias("log_v"))
    m_pre = gam("log_v ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    np.testing.assert_allclose(m_lhs.sp, m_pre.sp, atol=0)
    np.testing.assert_allclose(m_lhs._beta, m_pre._beta, atol=1e-12)
    assert m_lhs.y.name == "log(Volume)"


def test_lhs_I_div_matches_manual_transform():
    """`I(y/100) ~ ...` is just an unwrap; should equal pre-computing
    y/100. Also verifies the deparsed label survives I()."""
    d = load_dataset("R", "trees")
    m_lhs = gam("I(Volume / 100) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns((pl.col("Volume") / 100.0).alias("v100"))
    m_pre = gam("v100 ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    assert "Volume" in m_lhs.y.name and "100" in m_lhs.y.name


def test_lhs_unsupported_function_raises():
    """An unsupported function on the LHS should error with a helpful
    message naming the allowed transforms."""
    d = load_dataset("R", "trees")
    with pytest.raises(NotImplementedError, match="not supported"):
        gam("foo(Volume) ~ s(Height)", d, method="REML")


def test_lhs_cbind_raises():
    """cbind() responses are binomial-only (C10) — non-binomial families
    error clearly (mgcv dies with an obscure subscript error there)."""
    d = load_dataset("R", "trees")
    with pytest.raises(ValueError, match="family=Binomial"):
        gam("cbind(Volume, Height) ~ s(Girth)", d, method="REML")


def test_lhs_unknown_column_raises():
    """Reference to a non-existent column inside an LHS expression."""
    d = load_dataset("R", "trees")
    with pytest.raises(KeyError, match="nope"):
        gam("log(nope) ~ s(Height)", d, method="REML")


@pytest.mark.filterwarnings("ignore:Fitting terminated with step failure:UserWarning")
def test_lhs_na_omit_drops_lhs_referenced_columns():
    """If the LHS expression touches a column that has NAs, those rows
    must be dropped before evaluating the response — otherwise polars
    would surface NaN through the transform."""
    d = pl.DataFrame(
        {
            "a": [1.0, 4.0, None, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    m = gam("sqrt(a) ~ s(x, k=4)", d, method="REML")
    assert m.n == 9
    np.testing.assert_allclose(
        np.asarray(m.y.to_list()),
        np.sqrt([1, 4, 16, 25, 36, 49, 64, 81, 100]),
        atol=1e-12,
    )


def test_check_outer_info_is_populated_after_fit():
    """`_outer_info` should be filled with grad/hess/score/iter after
    a smooth fit, and remain None for the no-smooth path."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    info = m._outer_info
    assert info is not None
    assert info["iter"] >= 1
    g = info["grad"]
    H = info["hess"]
    assert g.size >= len(m.sp)
    assert H.shape == (g.size, g.size)
    assert np.isfinite(info["score"])
    ev = np.linalg.eigvalsh(0.5 * (info["hess"] + info["hess"].T))
    assert ev.min() > 0

    d2 = load_dataset("R", "trees")
    m2 = gam("Volume ~ Height + Girth", d2, method="REML")
    assert m2._outer_info is None


def test_gam_offset_in_formula_matches_glm():
    """No smooths → gam == glm. Offset(...) inside the formula must
    propagate identically through both."""
    from hea.family import Quasi

    d = load_dataset("MASS", "quine")  # count data
    d = d.with_columns(
        off=pl.lit(0.3) * pl.col("Days").cast(pl.Float64).clip(lower_bound=1).log()
    )
    formula = "Days ~ offset(off) + Sex + Age"
    fam = Quasi(link="log", variance="mu")
    b_glm = glm(formula, family=fam, data=d)
    b_gam = gam(formula, family=fam, data=d, method="REML")
    np.testing.assert_allclose(
        b_gam._beta,
        b_glm._bhat_arr,
        atol=1e-10,
    )
    np.testing.assert_allclose(b_gam.deviance, b_glm.deviance, atol=1e-10)
    np.testing.assert_allclose(
        b_gam.fitted_values,
        b_glm.fitted_values,
        atol=1e-10,
    )


def test_gam_offset_kwarg_equivalent_to_formula_offset():
    """offset(off) in formula should give the same fit as offset=off kwarg."""
    rng = np.random.default_rng(0)
    n = 100
    d = pl.DataFrame(
        {
            "y": rng.poisson(3.0, n).astype(float),
            "x": rng.standard_normal(n),
            "off_col": rng.uniform(0.0, 1.0, n),
        }
    )
    from hea.family import Poisson

    a = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    b = gam(
        "y ~ x", family=Poisson(), data=d, method="REML", offset=d["off_col"].to_numpy()
    )
    np.testing.assert_allclose(a._beta, b._beta, atol=1e-10)
    np.testing.assert_allclose(a.deviance, b.deviance, atol=1e-10)


def test_gam_gamma_kwarg_matches_mgcv_on_trees():
    """``gamma=`` (mgcv's smoothing-strength multiplier) — Wood §4.6 cites
    ``gamma=1.4`` as a reasonable default for over-fit protection.
    """
    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees")

    m_gcv_1 = gam(
        "Volume ~ s(Height) + s(Girth)",
        family=Gamma(link="log"),
        data=trees,
        method="GCV.Cp",
        gamma=1.0,
    )
    np.testing.assert_allclose(m_gcv_1.GCV_score, 0.008082356, atol=1e-6)
    np.testing.assert_allclose(m_gcv_1.sp[1], 0.342711, atol=1e-4)

    m_gcv_14 = gam(
        "Volume ~ s(Height) + s(Girth)",
        family=Gamma(link="log"),
        data=trees,
        method="GCV.Cp",
        gamma=1.4,
    )
    np.testing.assert_allclose(m_gcv_14.GCV_score, 0.009228008, atol=1e-6)
    np.testing.assert_allclose(m_gcv_14.sp[1], 0.524542, atol=1e-4)
    assert m_gcv_14.sp[1] > m_gcv_1.sp[1]

    m_reml_1 = gam(
        "Volume ~ s(Height) + s(Girth)",
        family=Gamma(link="log"),
        data=trees,
        method="REML",
        gamma=1.0,
    )
    np.testing.assert_allclose(m_reml_1.REML_criterion / 2, 78.00469, atol=1e-3)

    m_reml_14 = gam(
        "Volume ~ s(Height) + s(Girth)",
        family=Gamma(link="log"),
        data=trees,
        method="REML",
        gamma=1.4,
    )
    np.testing.assert_allclose(m_reml_14.REML_criterion / 2, 59.35457, atol=1e-3)


def test_plot_smooth_dispatches_2d_to_contour():
    """``plot_smooth`` should auto-render contour for 2D smooths
    (Wood 2017 Fig. 4.14 — bold/dashed/dotted contours + data scatter)."""
    import matplotlib

    matplotlib.use("Agg")
    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees")
    ct5 = gam("Volume ~ s(Height, Girth, k=25)", family=Gamma(link="log"), data=trees)
    fig = ct5.plot_smooth(too_far=0.1)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "s(Height,Girth," in ax.get_title()
    assert ax.get_xlabel() == "Height"
    assert ax.get_ylabel() == "Girth"

    m = gam(
        "Volume ~ s(Height) + s(Height, Girth, k=20)",
        family=Gamma(link="log"),
        data=trees,
    )
    fig2 = m.plot_smooth(too_far=0.1)
    assert len(fig2.axes) == 2
    assert fig2.axes[0].get_title() == ""  # 1D panel
    assert "s(Height," in fig2.axes[0].get_ylabel()
    assert "s(Height,Girth," in fig2.axes[1].get_title()  # 2D panel


def test_plot_smooth_all_terms_factor_termplot():
    """``plot_smooth(all_terms=True)`` should add a parametric panel for
    the factor — Wood 2017 Fig. 4.15."""
    import matplotlib

    matplotlib.use("Agg")
    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees").with_columns(
        Hclass=((pl.col("Height") / 10).floor() - 5)
        .cast(pl.Int64)
        .replace_strict(
            [1, 2, 3],
            ["small", "medium", "large"],
            return_dtype=pl.Enum(["small", "medium", "large"]),
        ),
    )
    ct7 = gam("Volume ~ Hclass + s(Girth)", family=Gamma(link="log"), data=trees)
    fig = ct7.plot_smooth(all_terms=True)
    assert len(fig.axes) == 2

    assert fig.axes[0].get_xlabel() == "Girth"
    assert "s(Girth," in fig.axes[0].get_ylabel()

    assert fig.axes[1].get_xlabel() == "Hclass"
    assert fig.axes[1].get_ylabel() == "Partial for Hclass"
    xticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert xticks == ["small", "medium", "large"]

    fig2 = ct7.plot_smooth()
    assert len(fig2.axes) == 1


def test_plot_smooth_select_by_name_and_list():
    """``select=`` accepts a smooth label, a list of labels, or a list of
    ints; ordering follows the list."""
    import matplotlib

    matplotlib.use("Agg")
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")

    fig = m.plot_smooth(select="s(x2)")
    assert len(fig.axes) == 1
    assert "s(x2," in fig.axes[0].get_ylabel()

    fig = m.plot_smooth(select=["s(x3)", "s(x1)"])
    assert len(fig.axes) == 2
    assert "s(x3," in fig.axes[0].get_ylabel()
    assert "s(x1," in fig.axes[1].get_ylabel()

    fig = m.plot_smooth(select=[0, "s(x3)"])
    assert len(fig.axes) == 2
    assert "s(x1," in fig.axes[0].get_ylabel()
    assert "s(x3," in fig.axes[1].get_ylabel()

    with pytest.raises(ValueError, match="doesn't match"):
        m.plot_smooth(select="s(missing)")
    with pytest.raises(IndexError, match="out of range"):
        m.plot_smooth(select=99)


def test_plot_smooth_scheme_persp_for_2d():
    """``scheme=1`` renders a 2D smooth as a 3D persp wireframe; the panel's
    axes must be a 3D Axes3D and carry the smooth label as zlabel."""
    import matplotlib

    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d import Axes3D

    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height, Girth, k=20)", family=Gamma(link="log"), data=trees)

    fig = m.plot_smooth(scheme=1)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], Axes3D)
    assert "s(Height,Girth," in fig.axes[0].get_zlabel()

    fig = m.plot_smooth(scheme=0)
    assert not isinstance(fig.axes[0], Axes3D)
    assert "s(Height,Girth," in fig.axes[0].get_title()


def test_plot_smooth_scheme_per_panel_list():
    """``scheme=[...]`` aligns to selected panels — 1D smooths ignore it,
    2D panels get persp where requested. Mirrors Wood 2017 Fig. 7.9."""
    import matplotlib

    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d import Axes3D

    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees")
    m = gam(
        "Volume ~ s(Height) + s(Height, Girth, k=20)",
        family=Gamma(link="log"),
        data=trees,
    )

    fig = m.plot_smooth(scheme=[0, 1])
    assert len(fig.axes) == 2
    assert not isinstance(fig.axes[0], Axes3D)
    assert isinstance(fig.axes[1], Axes3D)

    with pytest.raises(ValueError, match="scheme list must have length 2"):
        m.plot_smooth(scheme=[0, 1, 0])


def test_plot_smooth_ax_3d_required_for_persp():
    """Passing ``ax=`` for a 2D scheme=1 panel demands a 3D Axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hea.family import Gamma

    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height, Girth, k=20)", family=Gamma(link="log"), data=trees)

    fig, ax2d = plt.subplots()
    with pytest.raises(TypeError, match="3D Axes"):
        m.plot_smooth(scheme=1, ax=ax2d)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    out = m.plot_smooth(scheme=1, ax=ax3d)
    assert out is ax3d


def _plot_curve(m, **kw):
    """The (x, y) polyline ``plot_smooth`` actually draws for a 1D panel."""
    fig, ax = plt.subplots()
    m.plot_smooth(ax=ax, **kw)
    curve = next(ln for ln in ax.lines if len(ln.get_xdata()) > 2)
    xs = np.asarray(curve.get_xdata(), dtype=float)
    ys = np.asarray(curve.get_ydata(), dtype=float)
    plt.close(fig)
    return xs, ys


def test_plot_smooth_1d_evaluates_on_a_grid_not_the_stored_design():
    """1D panels must rebuild the basis on an ``n_grid_1d`` grid spanning the
    covariate range (mgcv ``plot.mgcv.smooth`` plots.r:929-956), never index
    ``block.X`` by data row.
    """
    from hea.models.bam import bam

    rng = np.random.default_rng(11)
    n = 2000
    x = np.round(rng.uniform(0, 1, n), 2)  # 101 unique values -> real compression
    d = pl.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)})

    m = bam("y ~ s(x, k=10)", d, discrete=True)
    assert np.shape(m._blocks[0].X)[0] != d.height  # the compressed mini-frame

    xs, ys = _plot_curve(m, select=0, rug=False)
    assert len(xs) == 100  # mgcv's n
    np.testing.assert_allclose(xs[[0, -1]], [x.min(), x.max()], rtol=1e-12)
    idx = [0, 24, 49, 74, 99]
    np.testing.assert_allclose(
        xs[idx],
        [0.0, 0.24242424242424243, 0.49494949494949497, 0.74747474747474751, 1.0],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ys[idx],
        [
            -0.083890896283563457,
            0.99266179501224905,
            0.00050279336478448655,
            -1.0007728914105209,
            0.0084398854201928053,
        ],
        rtol=0,
        atol=1e-6,
    )

    assert len(_plot_curve(m, select=0, rug=False, n_grid_1d=37)[0]) == 37
    m_dense = bam("y ~ s(x, k=10)", d, discrete=False, chunk_size=200)
    assert np.shape(m_dense._blocks[0].X)[0] != d.height
    assert len(_plot_curve(m_dense, select=0, rug=False)[0]) == 100


def test_plot_smooth_1d_band_is_1p96_se():
    """mgcv widens the 1D band by ``-qnorm((1-clev)/2)`` with ``clev=.95``
    (plots.r:952/1417) — 1.959964, not 2."""
    from hea.models.gam import _SE_MULT_1D

    np.testing.assert_allclose(_SE_MULT_1D, 1.9599639845400534, rtol=1e-14)

    rng = np.random.default_rng(11)
    n = 600
    x = np.round(rng.uniform(0, 1, n), 2)
    d = pl.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)})
    m = gam("y ~ s(x, k=10)", d)

    fig, ax = plt.subplots()
    m.plot_smooth(select=0, ax=ax, rug=False)
    xs, ys = _plot_curve(m, select=0, rug=False)
    band = ax.collections[0].get_paths()[0].vertices
    lo = band[1 : len(xs) + 1, 1]
    hi = band[len(xs) + 2 : 2 * len(xs) + 2, 1][::-1]
    plt.close(fig)

    B = m._block_predict_mat(m._blocks[0], pl.DataFrame({"x": xs}), apply_by=False)
    a, b = m._block_col_ranges[0]
    se = np.sqrt(((B @ m.Vp[a:b, a:b]) * B).sum(axis=1))
    np.testing.assert_allclose(hi - ys, _SE_MULT_1D * se, rtol=1e-12)
    np.testing.assert_allclose(ys - lo, _SE_MULT_1D * se, rtol=1e-12)


def test_plot_smooth_1d_by_panel_rug_and_resids():
    """For a ``by=`` smooth mgcv rugs the *whole* model-frame column
    (``raw <- data[x$term][[1]]``, plots.r:930 — no level filter) and skips
    partial residuals unless ``by.resids=TRUE`` (plots.r:1097)."""
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(0, 1, n)
    g = rng.choice(["a", "b"], n)
    y = np.where(g == "a", np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)) + rng.normal(
        0, 0.3, n
    )
    d = pl.DataFrame({"x": x, "g": g, "y": y}).with_columns(
        pl.col("g").cast(pl.Categorical)
    )
    m = gam("y ~ g + s(x, by=g, k=8)", d)

    fig, ax = plt.subplots()
    m.plot_smooth(select=0, ax=ax, rug=True, partial_residuals=True)
    rug = next(ln for ln in ax.lines if ln.get_marker() == "|")
    assert len(rug.get_xdata()) == n  # not just the 'a' rows
    assert not [c for c in ax.collections if c.get_offsets().shape[0] > 10]
    plt.close(fig)

    fig, ax = plt.subplots()
    m.plot_smooth(select=0, ax=ax, rug=False, partial_residuals=True, by_resids=True)
    pts = [c for c in ax.collections if c.get_offsets().shape[0] > 10]
    assert len(pts) == 1 and pts[0].get_offsets().shape[0] == n
    plt.close(fig)


def test_gam_gamma_validation():
    d = load_dataset("R", "iris")
    with pytest.raises(ValueError, match="gamma"):
        gam("Sepal.Length ~ s(Petal.Length)", d, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        gam("Sepal.Length ~ s(Petal.Length)", d, gamma=-0.5)


def test_gam_predict_reevaluates_offset_on_newdata():
    """predict.gam re-evaluates formula offset(...) atoms on newdata."""
    rng = np.random.default_rng(0)
    n = 80
    d = pl.DataFrame(
        {
            "y": rng.poisson(4.0, n).astype(float),
            "x": rng.standard_normal(n),
            "off_col": rng.uniform(0.5, 1.5, n),
        }
    )
    from hea.family import Poisson

    m = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    new = d.with_columns((pl.col("off_col") + 2.0).alias("off_col"))
    eta_orig = m.predict(type="link")["fit"].to_numpy()
    eta_new = m.predict(new, type="link")["fit"].to_numpy()
    np.testing.assert_allclose(eta_new - eta_orig, 2.0, atol=1e-10)


def test_select_true_doubles_n_sp():
    """select=TRUE adds one null-space penalty per smooth → n_sp doubles."""
    d = load_dataset("synthetic", "seed_synth_basic")
    m_off = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")
    m_on = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)
    assert len(m_off.sp) == 3
    assert len(m_on.sp) == 6


def test_select_true_three_smooth_REML():
    """gam(y ~ s(x1)+s(x2)+s(x3), data=seed_synth_basic, method="REML", select=TRUE)"""
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)

    _allclose(m.edf_total, 2.912088577, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.8940008109, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 277.0814067, atol=5e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 1.091137918, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x1)"], 0.9739738079, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 0.9379440321, atol=5e-3, name="edf[s(x2)]")
    assert m.edf_by_smooth["s(x3)"] < 1e-2, (
        f"s(x3) should be selected out, got edf={m.edf_by_smooth['s(x3)']}"
    )


def test_select_true_single_smooth_sp_passthrough():
    """Single-smooth select=TRUE: gam(..., sp=m_free.sp) reproduces the
    free-optimization fit at the same sp — the augmented sp vector (now
    length 2) is correctly threaded through the slot machinery, and the
    sp= path's profile-out log φ̂ matches the optimizer's converged log φ̂
    to optimizer-tolerance precision.
    """
    d = load_dataset("MASS", "mcycle")
    m_free = gam("accel ~ s(times)", d, method="REML", select=True)
    assert len(m_free.sp) == 2
    m_fix = gam("accel ~ s(times)", d, method="REML", select=True, sp=m_free.sp)
    np.testing.assert_allclose(m_fix.fitted, m_free.fitted, atol=1e-10)
    np.testing.assert_allclose(m_fix.edf_total, m_free.edf_total, atol=1e-10)
    np.testing.assert_allclose(m_fix.sigma_squared, m_free.sigma_squared, rtol=1e-5)
    np.testing.assert_allclose(m_fix.REML_criterion, m_free.REML_criterion, rtol=1e-7)


def test_select_true_at_mgcv_sp_matches_mgcv():
    """At a fixed sp vector, hea's select=TRUE fit must reproduce mgcv's
    post-fit numbers — checks the null-space penalty math directly,
    bypassing optimizer convergence differences.
    """
    d = load_dataset("mgcv", "gamSim_eg1")
    sp_mgcv = np.array(
        [
            2.521010255,
            423334.7801,  # s(x0): wig, null
            1.843214985,
            1.820731653,  # s(x1): wig, null
            0.00569866453,
            47639.04804,  # s(x2): wig, null
            84968.55542,
            131.2834178,  # s(x3): wig, null (essentially zeroed)
        ]
    )
    m = gam(
        "y ~ s(x0) + s(x1) + s(x2) + s(x3)", d, method="REML", select=True, sp=sp_mgcv
    )

    _allclose(m.edf_total, 14.45446565, atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.933035582, atol=1e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 868.3979813, atol=1e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 7.833279497, atol=1e-3)
    _allclose(m.edf_by_smooth["s(x0)"], 2.418051213, atol=1e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.839713272, atol=1e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 7.448219388, atol=1e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 0.7484817774, atol=1e-3, name="edf[s(x3)]")


def test_select_true_binomial_summary_matches_mgcv():
    """hea's summary() must dispatch on ``family.scale_known``: known-scale
    families use the Wald z-test for parametric coefficients and the Wood
    (2013) reTest with Davies' weighted-χ² CDF for smooth significance,
    not t/F. Pinned to mgcv on wesdr at mgcv's converged sp.
    """
    from scipy.stats import norm

    from hea.family import Binomial

    d = load_dataset("gamair", "wesdr")
    sp_mgcv = np.array(
        [
            0.0164113465035,
            4.59199813892,  # s(dur): wig, null
            1793.09515417,
            0.953183305109,  # s(gly): wig, null
            0.0458306723482,
            5.7780644155,  # s(bmi): wig, null
        ]
    )
    m = gam(
        "ret ~ s(dur,k=5) + s(gly,k=5) + s(bmi,k=5)",
        d,
        family=Binomial(),
        method="REML",
        select=True,
        sp=sp_mgcv,
    )

    assert m.family.scale_known is True

    _allclose(m.edf_total, 7.430392736, atol=1e-3, name="edf_total")
    _allclose(m.REML_criterion / 2, 389.4888704, atol=1e-3, name="REML/2")

    j = list(m.bhat.columns).index("(Intercept)")
    est = float(m._beta[j])
    se = float(m._se[j])
    z = est / se
    p_z = 2.0 * norm.sf(abs(z))
    _allclose(est, -0.4150103, atol=1e-3, name="intercept")
    _allclose(se, 0.0887844, atol=1e-3, name="intercept SE")
    _allclose(z, -4.674361, atol=5e-3, name="z")
    _allclose(p_z, 2.948704e-06, atol=5e-7, name="Pr(>|z|)")

    targets = [
        ("s(dur)", 2.982517, 4, 15.58609, 0.0007177005),
        ("s(gly)", 0.989778, 4, 91.07272, 0.0),
        ("s(bmi)", 2.458097, 4, 13.64956, 0.0008958199),
    ]
    for m_idx, (label, edf_t, refdf_t, chisq_t, pv_t) in enumerate(targets):
        a, bcol = m._block_col_ranges[m_idx]
        beta_b = m._beta[a:bcol]
        Vp_b = m.Vp[a:bcol, a:bcol]
        stat, pval, ref_df = m._re_test(m_idx, beta_b, Vp_b)
        _allclose(float(m.edf[a:bcol].sum()), edf_t, atol=1e-3, name=f"edf[{label}]")
        assert int(ref_df) == refdf_t, f"Ref.df[{label}]: {ref_df} vs {refdf_t}"
        _allclose(stat, chisq_t, atol=5e-3, name=f"Chi.sq[{label}]")
        if pv_t > 0:
            _allclose(pval, pv_t, atol=1e-5, name=f"p-value[{label}]")
        else:
            assert pval < 1e-15, f"p-value[{label}] not vanishing: {pval}"


# ---------------------------------------------------------------------------
# method="ML" — Laplace marginal likelihood (does not profile out fixed
# effects, so scores are comparable across different fixed-effect structures
# in `anova(m1, m2)` LRTs). Differs from REML by a `Mp·log(2π·φ)` constant
# in the score formula (gam.fit3.r:616, remlInd=0).
# ---------------------------------------------------------------------------


def test_mcycle_tp_ML():
    """gam(accel ~ s(times), data=mcycle, method="ML")"""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="ML")

    assert m.n == 133
    _allclose(m.sp[0], 7.742109e-04, atol=5e-5, name="sp")
    _allclose(m.edf_total, 9.625375, atol=5e-4, name="edf_total")
    _allclose(m.sigma_squared, 506.3487, atol=5e-3, name="sigma2")
    _allclose(m.ML_criterion / 2, 622.2919, atol=5e-3, name="ML/2")
    _allclose(m.r_squared_adjusted, 0.7831502, atol=5e-4, name="r2adj")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 8.625375, atol=5e-4, name="edf[s(times)]")


def test_gamSim_eg1_four_smooths_ML():
    """gam(y ~ s(x0)+s(x1)+s(x2)+s(x3), data=gamSim(eg=1), method="ML")"""
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)", d, method="ML")

    assert m.n == 400
    _allclose(m.edf_total, 15.44156, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.897865, atol=5e-3, name="sigma2")
    _allclose(m.ML_criterion / 2, 860.3114, atol=5e-3, name="ML/2")
    _allclose(m.r_squared_adjusted, 0.7156318, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)", 7.833279, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x0)"], 2.816760, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.620159, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.004539, atol=5e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 1.000098, atol=5e-2, name="edf[s(x3)]")


def test_wesdr_binomial_ML():
    """gam(ret ~ s(dur)+s(gly)+s(bmi), data=wesdr, family=binomial, method="ML")
    gdi.c:1532-1680). mgcv's pins:
    """
    from hea.family import Binomial

    d = load_dataset("gamair", "wesdr")
    m_ml = gam("ret ~ s(dur) + s(gly) + s(bmi)", d, family=Binomial(), method="ML")

    _allclose(m_ml.ML_criterion / 2, 384.0036, atol=5e-3, name="ML/2")
    _allclose(m_ml.edf_total, 8.416686, atol=5e-3, name="edf_total")
    _assert_param(m_ml, "(Intercept)", -0.4176841, atol=5e-3)
    _allclose(m_ml.sp[0], 0.07866319, atol=5e-3, name="sp[s(dur)]")
    _allclose(m_ml.sp[2], 0.2152721, atol=5e-3, name="sp[s(bmi)]")


def test_method_validation():
    """gam() rejects bogus method strings before doing any work, but accepts
    mgcv's full criterion set. ``UBRE`` is an internal criterion, not a user
    ``method=`` value (mgcv.r:1915), so it still raises; ``NCV``/``QNCV`` are
    valid in mgcv but not yet ported, so they raise NotImplementedError."""
    d = load_dataset("MASS", "mcycle")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="UBRE")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="bogus")
    with pytest.raises(NotImplementedError, match="NCV"):
        gam("accel ~ s(times)", d, method="NCV")
    for m in ("GACV.Cp", "P-REML", "P-ML"):
        assert gam("accel ~ s(times)", d, method=m).method in (m, "REML", "ML")


def _simulate_compound_poisson_gamma(rng, n, p_true=1.5, phi_true=1.0):
    """Compound Poisson-Gamma sample: N_i ~ Poisson(λ_i), N_i Gamma jumps."""
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.exp(0.5 + 1.5 * np.sin(2.0 * np.pi * x))
    lam = mu_true ** (2.0 - p_true) / (phi_true * (2.0 - p_true))
    N = rng.poisson(lam)
    shape = (2.0 - p_true) / (p_true - 1.0)
    scale = phi_true * (p_true - 1.0) * mu_true ** (p_true - 1.0)
    y = np.zeros(n)
    for i in range(n):
        if N[i] > 0:
            y[i] = rng.gamma(shape * N[i], scale[i])
    return x, y, mu_true


def test_gam_fit_with_tweedie_fixed_p():
    rng = np.random.default_rng(42)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=200)
    df = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x, k=8)", df, family=Tweedie(p=1.5), method="REML")
    assert np.isfinite(m.REML_criterion)
    assert 1.0 < m.edf_total < 8.0
    assert 0.5 < float(np.exp(m._log_phi_hat)) < 2.0


def test_gam_fit_with_tw_recovers_p_near_truth():
    """tw() with default initialisation should converge near the true p."""
    rng = np.random.default_rng(123)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=400, p_true=1.5)
    df = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x, k=10)", df, family=tw(), method="REML")
    info = m._tw_info
    assert info is not None
    assert 1.30 < info["p_hat"] < 1.70


def test_gam_tw_mack_with_expression_in_s_mgcv_oracle():
    """User's literal R formula: ``s(I(b.depth^.5))`` and ``offset(...)``."""
    mack = load_dataset("gamair", "mack")
    keep_cols = [
        "egg.count",
        "lon",
        "lat",
        "b.depth",
        "c.dist",
        "salinity",
        "temp.surf",
        "temp.20m",
        "net.area",
    ]
    mack = mack.drop_nulls(subset=keep_cols)
    mack = mack.with_columns(log_net_area=pl.col("net.area").log())

    m = gam(
        "egg.count ~ s(lon, lat, k=100) + s(I(b.depth^0.5)) + s(c.dist) "
        "+ s(salinity) + s(temp.surf) + s(temp.20m) + offset(log_net_area)",
        mack,
        family=tw(),
        method="REML",
        select=True,
    )
    info = m._tw_info
    assert info is not None
    np.testing.assert_allclose(info["p_hat"], 1.33307185396394, atol=1e-3)
    np.testing.assert_allclose(m.REML_criterion / 2, 927.776776447335, atol=5e-3)
    assert "s(I(b.depth^0.5))" in m.edf_by_smooth
    np.testing.assert_allclose(
        m.edf_by_smooth["s(I(b.depth^0.5))"],
        2.37609109,
        atol=5e-2,
    )
    np.testing.assert_allclose(m.edf_total, 47.4833915, atol=5e-2)


def test_gam_tw_mack_mgcv_oracle():
    """Pin tw() joint outer-Newton output against mgcv 1.9-4 on gamair::mack."""
    mack = load_dataset("gamair", "mack")
    keep_cols = [
        "egg.count",
        "lon",
        "lat",
        "b.depth",
        "c.dist",
        "salinity",
        "temp.surf",
        "temp.20m",
        "net.area",
    ]
    mack = mack.drop_nulls(subset=keep_cols)
    mack = mack.with_columns(log_net_area=pl.col("net.area").log())

    m = gam(
        "egg.count ~ s(lon, lat, k=20) + s(temp.surf)",
        mack,
        family=tw(),
        method="REML",
        offset=mack["log_net_area"].to_numpy().tolist(),
    )
    info = m._tw_info
    assert info is not None
    np.testing.assert_allclose(info["p_hat"], 1.39920632555438, atol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2, 945.744274311548, atol=1e-4)
    np.testing.assert_allclose(np.exp(info["log_phi_hat"]), 4.00764107362287, rtol=5e-4)
    np.testing.assert_allclose(m.edf_total, 17.9986147698585, atol=5e-2)
    np.testing.assert_allclose(m.sp[0], 0.161829581092981, rtol=5e-3)


def test_gam_fit_tw_score_no_worse_than_fixed_p():
    """Joint outer Newton over (ρ, log φ, θ_tw) only accepts steps that
    improve the criterion, so tw()'s REML score should be ≤ Tweedie(1.5)'s."""
    rng = np.random.default_rng(7)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=300, p_true=1.4)
    df = pl.DataFrame({"y": y, "x": x})
    m_fixed = gam("y ~ s(x, k=8)", df, family=Tweedie(p=1.5), method="REML")
    m_tw = gam("y ~ s(x, k=8)", df, family=tw(), method="REML")
    assert m_tw.REML_criterion <= m_fixed.REML_criterion + 1e-6


def test_tw_gcv_method_coerced_to_reml():
    # mgcv silently coerces extended families onto (RE)ML — gam.fit4 has
    # no GCV/UBRE path (mgcv.r:1892). hea mirrors the coercion.
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x, k=6)", df, family=tw(), method="GCV.Cp")
    assert m.method == "REML"
    assert np.isfinite(m.REML_criterion)


def test_tw_rejects_fixed_sp():
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="incompatible"):
        gam("y ~ s(x, k=6)", df, family=tw(), method="REML", sp=np.array([0.1]))


@pytest.fixture(scope="module")
def trees_te():
    """trees with a 2D tensor smooth — the canonical vis.gam example."""
    data = load_dataset("mgcv", "trees").rename(
        {"Volume": "vol", "Girth": "g", "Height": "h"}
    )
    m = gam("vol ~ te(g, h)", data=data, method="REML")
    return m, data


@pytest.fixture(scope="module")
def factor_model():
    """A model with one numeric and one factor RHS variable."""
    rng = np.random.RandomState(0)
    df = pl.DataFrame(
        {
            "y": rng.randn(120),
            "x": rng.rand(120),
            "g": (["a", "b", "c"] * 40),
        }
    )
    m = gam("y ~ s(x) + g", data=df)
    return m, df


def test_vis_matches_predict_on_same_grid(trees_te):
    """vis(view, n_grid) must equal predict(grid) — no extra computation."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=20, type="link")

    G, H = np.meshgrid(v.m1, v.m2, indexing="ij")
    new = pl.DataFrame({"g": G.ravel(), "h": H.ravel()}).with_columns(
        pl.col("g").cast(data["g"].dtype),
        pl.col("h").cast(data["h"].dtype),
    )
    fit_pred = m.predict(new, type="link")["fit"].to_numpy().reshape(20, 20)
    assert np.allclose(v.fit, fit_pred, atol=1e-12, rtol=0)


def test_vis_se_matches_predict_se(trees_te):
    """SE on the grid must match predict(se_fit=True) on the same grid."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=15, type="link", se=True)

    G, H = np.meshgrid(v.m1, v.m2, indexing="ij")
    new = pl.DataFrame({"g": G.ravel(), "h": H.ravel()}).with_columns(
        pl.col("g").cast(data["g"].dtype),
        pl.col("h").cast(data["h"].dtype),
    )
    pred = m.predict(new, type="link", se_fit=True)
    fit_pred = pred["fit"].to_numpy()
    se_pred = pred["se.fit"].to_numpy()
    assert np.allclose(v.fit, fit_pred.reshape(15, 15), atol=1e-12)
    assert np.allclose(v.se, se_pred.reshape(15, 15), atol=1e-12)


def test_vis_response_scale_matches_link_via_inverse(trees_te):
    """type='response' = linkinv(η̂); SE scaled by |dμ/dη| (delta method)."""
    m, _ = trees_te
    v_link = m.vis(view=("g", "h"), n_grid=10, type="link", se=True)
    v_resp = m.vis(view=("g", "h"), n_grid=10, type="response", se=True)
    assert np.allclose(v_link.fit, v_resp.fit)
    assert np.allclose(v_link.se, v_resp.se)


def test_auto_pick_view(trees_te):
    """No `view`: pick the first two RHS vars with variation."""
    m, _ = trees_te
    v = m.vis()
    assert v.view == ("g", "h")
    assert v.fit.shape == (30, 30)


def test_grid_endpoints(trees_te):
    """Numeric grids span [min(x), max(x)] of the fit data."""
    m, data = trees_te
    v = m.vis(view=("g", "h"), n_grid=8)
    assert v.m1[0] == float(data["g"].min())
    assert v.m1[-1] == float(data["g"].max())
    assert v.m2[0] == float(data["h"].min())
    assert v.m2[-1] == float(data["h"].max())


def test_too_far_masks_distant_points(trees_te):
    """``too_far > 0`` replaces distant grid cells with NaN."""
    m, _ = trees_te
    v0 = m.vis(view=("g", "h"), n_grid=20, too_far=0.0)
    v1 = m.vis(view=("g", "h"), n_grid=20, too_far=0.1)
    assert np.all(np.isfinite(v0.fit))
    assert np.any(np.isnan(v1.fit))
    keep = ~np.isnan(v1.fit)
    assert np.allclose(v0.fit[keep], v1.fit[keep])


def test_cond_overrides_typical_value():
    """`cond={var: val}` shifts the held-fixed value, changing the surface."""
    rng = np.random.RandomState(1)
    df = pl.DataFrame(
        {
            "y": rng.randn(80),
            "x1": rng.rand(80),
            "x2": rng.rand(80),
            "x3": rng.rand(80),
        }
    )
    m = gam("y ~ s(x1) + s(x2) + s(x3)", data=df, method="REML")
    v_default = m.vis(view=("x1", "x2"), n_grid=8)
    v_override = m.vis(view=("x1", "x2"), n_grid=8, cond={"x3": 0.9})
    diff = v_default.fit - v_override.fit
    assert np.std(diff) < 1e-10
    assert abs(np.mean(diff)) > 1e-6  # but the offset is non-zero


def test_factor_view_axis(factor_model):
    """Factor view: m2 contains the level names; surface is well-defined."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10)
    assert v.fit.shape == (10, 10)
    assert set(np.unique(v.m2)) == {"a", "b", "c"}
    assert np.all(np.isfinite(v.fit))


def test_factor_view_too_far_returns_no_mask(factor_model):
    """too_far is undefined when an axis is a factor; mgcv would crash, we
    quietly return all-False."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10, too_far=0.5)
    assert np.all(np.isfinite(v.fit))


def test_invalid_view():
    """View must be 2 names from the formula's RHS variables."""
    rng = np.random.RandomState(0)
    df = pl.DataFrame({"y": rng.standard_normal(10), "x": np.arange(10.0)})
    m = gam("y ~ s(x)", data=df, method="REML")
    with pytest.raises(ValueError):
        m.vis()
    with pytest.raises(ValueError):
        m.vis(view=("x",))
    with pytest.raises(ValueError):
        m.vis(view=("x", "nope"))


def test_invalid_type(trees_te):
    m, _ = trees_te
    with pytest.raises(ValueError):
        m.vis(view=("g", "h"), type="bogus")


def test_vis_result_repr(trees_te):
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=5)
    s = repr(v)
    assert "VisResult" in s and "view=('g', 'h')" in s


def test_plot_contour_smoke(trees_te):
    """``.plot(kind='contour')`` returns an Axes without raising."""
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=10, se=True)
    ax = v.plot(kind="contour")
    assert ax is not None
    plt.close("all")


def test_plot_persp_smoke(trees_te):
    """``.plot(kind='persp')`` with se_mult draws ± envelopes."""
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=10, se=True)
    ax = v.plot(kind="persp", se_mult=2.0)
    assert ax is not None
    plt.close("all")


def test_plot_factor_axis_ticks(factor_model):
    """Factor axis on a contour plot: ticks rendered as level names."""
    m, _ = factor_model
    v = m.vis(view=("x", "g"), n_grid=10)
    ax = v.plot(kind="contour")
    yticks = [t.get_text() for t in ax.get_yticklabels()]
    assert {t for t in yticks if t} <= {"a", "b", "c"}
    plt.close("all")


def test_invalid_plot_kind(trees_te):
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=5)
    with pytest.raises(ValueError):
        v.plot(kind="surface")  # only contour/persp supported


_ITSADUG_ROOT = Path(__file__).parent / "fixtures" / "itsadug_plot_diff"
_ITSADUG_MODEL_DIR = _ITSADUG_ROOT / "_model"
_ITSADUG_CASE_DIRS = (
    sorted(
        p for p in _ITSADUG_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    if _ITSADUG_ROOT.exists()
    else []
)
_ITSADUG_CASE_IDS = [p.name for p in _ITSADUG_CASE_DIRS]


def _itsadug_load_data() -> pl.DataFrame:
    """Load the synthetic dataset and re-attach factor levels — CSV
    round-trip drops R's factor type, but hea is happy with either pl.Utf8
    or pl.Enum at materialize time. We use pl.Enum with the explicit level
    order R wrote (A,B,C / Y,Z) for parity with mgcv's contrasts.
    """
    df = pl.read_csv(_ITSADUG_MODEL_DIR / "data.csv", null_values="NA")
    df = df.with_columns(
        [
            df["group"].cast(pl.Enum(["A", "B", "C"])),
            df["cohort"].cast(pl.Enum(["Y", "Z"])),
        ]
    )
    return df


@pytest.fixture(scope="module")
def itsadug_fitted_model():
    data = _itsadug_load_data()
    m = gam("y ~ group + cohort + s(x, by=group)", data=data, method="REML")
    return m, data


def _itsadug_parse_args(path: Path) -> dict:
    """args.json round-trip: itsadug's R script wraps each value as a list,
    so a length-1 vector lands as ``["A"]`` not ``"A"``. We don't unwrap —
    hea's get_difference accepts list values uniformly.
    """
    raw = json.loads(path.read_text())
    comp = {k: tuple(v) for k, v in raw["comp"].items()}
    cond = {}
    for k, v in raw["cond"].items():
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            cond[k] = np.asarray(v, dtype=float)
        else:
            cond[k] = list(v) if isinstance(v, list) else [v]
    return {
        "case_id": raw["case_id"],
        "comp": comp,
        "cond": cond,
        "f": float(raw["f"]),
        "sim_ci": bool(raw["sim.ci"]),
        "rm_ranef": raw["rm.ranef"],
        "n_grid": int(raw["n_grid"]),
        "has_sim_ci_col": bool(raw["has_sim_ci_col"]),
        "sim_seed": raw.get("sim_seed"),
    }


@pytest.mark.parametrize("case_id", _ITSADUG_CASE_IDS)
def test_get_difference_matches_itsadug(itsadug_fitted_model, case_id: str):
    m, _ = itsadug_fitted_model
    case_dir = _ITSADUG_ROOT / case_id
    args = _itsadug_parse_args(case_dir / "args.json")
    ref = pl.read_csv(case_dir / "diff_table.csv", null_values="NA")

    rm_ranef = args["rm_ranef"]
    if not isinstance(rm_ranef, bool):
        rm_ranef = list(rm_ranef) if isinstance(rm_ranef, list) else rm_ranef

    res = m.get_difference(
        comp=args["comp"],
        cond=args["cond"],
        f=args["f"],
        sim_ci=args["sim_ci"],
        rm_ranef=rm_ranef,
        rng=args["sim_seed"],
        n_sim=10_000,
    )

    assert res.difference.shape[0] == args["n_grid"], (
        f"{case_id}: got {res.difference.shape[0]} grid rows, want {args['n_grid']}"
    )

    np.testing.assert_allclose(
        res.difference,
        ref["difference"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: difference diverges from itsadug",
    )

    np.testing.assert_allclose(
        res.ci,
        ref["CI"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: CI diverges from itsadug",
    )

    if args["sim_ci"]:
        assert args["has_sim_ci_col"], (
            "fixture mislabel: sim.ci=TRUE but no sim.CI column"
        )
        assert res.sim_ci is not None and res.crit is not None

        ref_se_fit = pl.read_csv(case_dir / "se_fit.csv")["se_fit"].to_numpy()
        ours_se_fit = res.sim_ci / res.crit
        np.testing.assert_allclose(
            ours_se_fit,
            ref_se_fit,
            rtol=1e-3,
            atol=2e-4,
            err_msg=f"{case_id}: simultaneous se_fit diverges",
        )

        ref_crit = float(pl.read_csv(case_dir / "crit.csv")["crit"][0])
        np.testing.assert_allclose(
            res.crit,
            ref_crit,
            rtol=8e-3,
            err_msg=f"{case_id}: simultaneous crit diverges (ours={res.crit}, R={ref_crit})",
        )


def test_cohort_y_matches_basic(itsadug_fitted_model):
    """Sanity: with the model ``y ~ group + cohort + s(x, by=group)`` and
    no group:cohort interaction, the (group=A) − (group=B) difference is
    identical regardless of the cohort the comparison is held at — both
    p1 and p2 carry the same cohort column, so it cancels. This case
    exists to exercise the cond-string-coerce path; the numerics should
    coincide with case_basic to machine precision.
    """
    m, _ = itsadug_fitted_model
    args_b = _itsadug_parse_args(_ITSADUG_ROOT / "case_basic" / "args.json")
    args_y = _itsadug_parse_args(_ITSADUG_ROOT / "case_cohortY" / "args.json")
    res_b = m.get_difference(
        comp=args_b["comp"],
        cond=args_b["cond"],
        f=args_b["f"],
        sim_ci=False,
        rm_ranef=True,
    )
    res_y = m.get_difference(
        comp=args_y["comp"],
        cond=args_y["cond"],
        f=args_y["f"],
        sim_ci=False,
        rm_ranef=True,
    )
    np.testing.assert_allclose(
        res_y.difference, res_b.difference, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(res_y.ci, res_b.ci, rtol=1e-12, atol=1e-12)


def _borderline_gaussian(seed: int, amp: float, n: int = 130) -> pl.DataFrame:
    from hea.R.rng import RGenerator

    g = RGenerator(seed)
    x = g.uniform(0.0, 1.0, n)
    y = amp * np.sin(4 * np.pi * x) + g.normal(0.0, 1.0, n)
    return pl.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize(
    "seed, amp, expected",
    [
        (16, 0.45, (2.67146916085, 3.32082935917, 0.924820270116, 0.431982202344)),
        (2, 0.50, (5.09662376462, 6.17398455164, 4.31714021834, 0.000494598563479)),
    ],
)
def test_teststat_mixture_pvalue_gaussian_matches_mgcv(seed, amp, expected):
    """Wood (2013) testStat: fractional-rank mixture reference distribution
    (psum.chisq) + d/d1 averaging — not the pf(F, rank, res.df) fallback."""
    m = gam("y ~ s(x)", _borderline_gaussian(seed, amp), method="REML")
    label, edf, ref_df, stat_col, p_val = m._smooth_significance_rows()[0]
    assert label == "s(x)"
    np.testing.assert_allclose(
        [edf, ref_df, stat_col, p_val],
        expected,
        rtol=5e-4,
        err_msg="s(x) row vs mgcv s.table",
    )


def test_teststat_mixture_pvalue_poisson_matches_mgcv():
    """Known-scale branch: chi-squared mixture via psum.chisq, and the
    statistic built on the √W-weighted design (mgcv tests against object$R,
    the QR factor of √W·X — unweighted X is only its legacy fallback)."""
    from hea.family import Poisson
    from hea.R.rng import RGenerator

    g = RGenerator(16)
    n = 160
    x = g.uniform(0.0, 1.0, n)
    y = np.asarray(g.poisson(np.exp(0.30 * np.sin(4 * np.pi * x))), dtype=float)
    d = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", d, family=Poisson(), method="REML")
    _label, edf, ref_df, stat_col, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose(
        [edf, ref_df, stat_col, p_val],
        (3.61350424706, 4.48011605323, 7.88459555761, 0.128871644663),
        rtol=5e-4,
        err_msg="s(x) row vs mgcv s.table (poisson)",
    )


def test_rank_deficient_design_detected_and_warned():
    """Exactly collinear parametric columns: mgcv detects rank 11/12 and
    *drops* a coefficient (pls_fit1, gdi.c:1740-1775) — zero-filled coef,
    SE 0, t NaN — and hea now does the same (the internal fit runs on the
    reduced design; reporting re-inflates)."""
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.uniform(0, 1, n)
    x2 = x1.copy()
    z = rng.uniform(0, 1, n)
    y = 1 + 2 * x1 + np.sin(2 * np.pi * z) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("y ~ x1 + x2 + s(z)", df, method="REML")
    assert m.rank == 11
    assert m.p == 11  # internal (reduced) parameter count
    assert m._p_orig == 12  # original — what check()/summary report
    b1 = float(np.asarray(m.bhat["x1"])[0])
    b2 = float(np.asarray(m.bhat["x2"])[0])
    assert (b1 == 0.0) != (b2 == 0.0)
    assert abs(b1 + b2 - 2.0) < 0.25


def test_rank_deficient_drop_matches_mgcv_exactly():
    from hea.R.rng import RGenerator

    g = RGenerator(3)
    n = 80
    x1 = g.uniform(0.0, 1.0, n)
    z = g.uniform(0.0, 1.0, n)
    y = 1.96 * x1 + np.sin(2 * np.pi * z) + g.normal(0.0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x1.copy(), "z": z, "y": y})
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("y ~ x1 + x2 + s(z)", df, method="REML")
    kept = float(np.asarray(m.bhat["x2"])[0]) or float(np.asarray(m.bhat["x1"])[0])
    np.testing.assert_allclose(kept, 1.981807588, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.REML_criterion / 2, 29.91747549, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.09205759387, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.sp[0], 0.01349564141, rtol=1e-5)
    np.testing.assert_allclose(
        m.predict(df.head(1))["fit"][0],
        -0.619726771,
        rtol=0,
        atol=1e-6,
    )
    tvals = m.t_values
    t1 = float(np.asarray(tvals["x1"])[0])
    t2 = float(np.asarray(tvals["x2"])[0])
    assert np.isnan(t1) != np.isnan(t2)


def test_full_rank_fit_reports_p_and_does_not_warn():
    import warnings

    d = load_dataset("MASS", "mcycle")
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        m = gam("accel ~ s(times)", d, method="REML")
    assert m.rank == m.p
    assert not any("rank deficient" in str(w.message) for w in wlist)


def _id_linked_data() -> pl.DataFrame:
    """Two covariates on *different ranges* — the acid test for id basis
    sharing (pooled knots over [0, 3] differ from each smooth's own).
    R-native (set.seed(13), 0-ulp: runif/rnorm)."""
    from hea.R.rng import RGenerator

    g = RGenerator(13)
    n = 250
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 3.0, n)
    y = np.sin(2 * np.pi * x0) + np.sin(2 * np.pi * x1 / 3) + g.normal(0.0, 0.35, n)
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y})


@pytest.mark.parametrize(
    "formula, exp_sp, exp_edf, exp_reml",
    [
        (
            "y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)",
            2.85748958973,
            (5.22667532821, 7.68279449191),
            130.595923509,
        ),
        (
            "y ~ s(x0, id=1) + s(x1, id=1)",  # tp (default basis)
            0.00111215693922,
            (4.0700018545, 8.67029420876),
            135.136095842,
        ),
    ],
)
def test_id_links_smoothing_parameters_matches_mgcv(
    formula,
    exp_sp,
    exp_edf,
    exp_reml,
):
    """mgcv id= semantics: ONE working λ shared across the linked smooths
    (L-matrix), bases built from POOLED covariate values (idLinksBases),
    penalties rescaled and constrained against the pooled construction —
    sp, per-smooth edf, and the REML score all pin to mgcv."""
    m = gam(formula, _id_linked_data(), method="REML")
    assert len(m.sp) == 1  # working sp (mgcv's m$sp)
    assert len(m._slots) == 2  # two penalties share it
    np.testing.assert_allclose(np.exp(m._rho_hat), [m.sp[0]] * 2, rtol=1e-12)
    np.testing.assert_allclose(m.sp[0], exp_sp, rtol=1e-4)
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()),
        exp_edf,
        rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, exp_reml, rtol=1e-6)
    pred = m.predict(_id_linked_data().head(40))["fit"].to_numpy()
    np.testing.assert_allclose(pred, m.fitted[:40], rtol=1e-10)


def test_id_by_factor_single_lambda_matches_mgcv():
    """``s(x2, by=fac, id=1)``: all by-level blocks share one λ — the
    canonical id idiom (mgcv gam.models docs; fixture mgcv_0080's formula).
    mgcv reference: sp=(0.01235565525, 0.0215017895), full.sp repeats the
    first across the three level blocks; -REML=199.87702188,
    scale=0.1657339158."""
    from hea.R.rng import RGenerator

    gen = RGenerator(5)
    n = 300
    x2 = gen.uniform(0, 1, n)
    x0 = gen.uniform(0, 1, n)
    fac = gen.mt.sample_int(3, n, replace=True) + 1  # R sample.int(3,…) → {1,2,3}
    fl = np.array([0.0, 1.0, 2.0])[fac - 1]
    amp = np.where(fac == 1, 1.0, np.where(fac == 2, 1.5, 0.5))
    y = (
        fl
        + amp * np.sin(2 * np.pi * x2)
        + np.cos(2 * np.pi * x0)
        + gen.normal(0, 0.4, n)
    )
    df = pl.DataFrame(
        {
            "x2": x2,
            "x0": x0,
            "fac": [f"f{i}" for i in fac],
            "y": y,
        }
    ).with_columns(pl.col("fac").cast(pl.Enum(["f1", "f2", "f3"])))
    m = gam("y ~ fac + s(x2, by=fac, id=1) + s(x0)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(
        m.sp,
        [0.01235565525, 0.0215017895],
        rtol=1e-4,
    )
    np.testing.assert_allclose(  # full.sp expansion
        np.exp(m._rho_hat),
        [m.sp[0], m.sp[0], m.sp[0], m.sp[1]],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()),
        [6.1706570, 5.7539111, 5.8854072, 6.9305014],
        rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 199.87702188, rtol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.1657339158, rtol=1e-5)


def test_id_tensor_smooths_match_mgcv():
    """id across te() smooths links pairwise (1st penalty ↔ 1st, 2nd ↔
    2nd) with pooled marginal bases. (R-native seed: 21 no longer
    saturates the 2nd penalty, so reseeded 21→10 to keep the λ→∞ tail
    coverage — same intent as the original.) mgcv reference:
    sp[0]=0.1704650207, sp[1] on the flat λ→∞ tail (mgcv 3.59e6 — only
    its order of magnitude is determined); -REML=97.72779351; per-smooth
    edf 8.6701264/8.6638061; scale 0.1111029015."""
    from hea.R.rng import RGenerator

    gen = RGenerator(10)
    n = 220
    x0, x1 = gen.uniform(0, 1, n), gen.uniform(0, 1, n)
    z, u = gen.uniform(0, 1, n), gen.uniform(0, 1, n)
    y = (
        np.sin(2 * np.pi * x0) * np.cos(np.pi * z)
        + 0.8 * np.sin(2 * np.pi * x1) * np.cos(np.pi * u)
        + gen.normal(0, 0.3, n)
    )
    df = pl.DataFrame({"x0": x0, "x1": x1, "z": z, "u": u, "y": y})
    m = gam("y ~ te(x0, z, id=1) + te(x1, u, id=1)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(m.sp[0], 0.1704650207, rtol=1e-4)
    assert m.sp[1] > 1e5  # flat saturation tail
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()),
        [8.6701264, 8.6638061],
        rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 97.72779351, rtol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.1111029015, rtol=1e-5)


def test_id_fixed_sp_takes_working_length():
    """``sp=`` supplies the *working* parameters (mgcv semantics): one
    value drives both linked penalties. mgcv reference at sp=2.0:
    sum(edf)=14.594997."""
    d = _id_linked_data()
    m = gam("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)", d, sp=[2.0])
    np.testing.assert_allclose(m.edf_total, 14.3473868864, rtol=1e-5)
    with pytest.raises(ValueError, match="length 1"):
        gam("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)", d, sp=[2.0, 3.0])


def test_id_singleton_is_noop():
    """An id used by a single smooth links nothing — same model as no id."""
    d = _id_linked_data()
    m1 = gam("y ~ s(x0, id=9) + s(x1)", d, method="REML")
    m0 = gam("y ~ s(x0) + s(x1)", d, method="REML")
    assert len(m1.sp) == 2
    np.testing.assert_allclose(m1.sp, m0.sp, rtol=1e-10)
    np.testing.assert_allclose(
        m1.REML_criterion,
        m0.REML_criterion,
        rtol=1e-12,
    )


def test_bam_links_id_like_gam():
    """bam has gam's working-θ L-matrix layer: id= shares ONE working λ across
    the linked smooths. Full mgcv-bam parity (sp/edf/criterion/fitted) lives in
    test_bam.py; this confirms bam links the same structure gam does on the
    shared fixture."""
    from hea.models.bam import bam

    d = _id_linked_data()
    m = bam("y ~ s(x0, id=1) + s(x1, id=1)", d, method="REML")
    assert len(m.sp) == 1 and len(m._slots) == 2
    np.testing.assert_allclose(np.exp(m._rho_hat), [m.sp[0]] * 2, rtol=1e-12)


def test_sz_id_kwarg_still_allowed():
    """bs='sz' legitimately consumes id= (within-term penalty merging) —
    the guard must not catch it."""
    rng = np.random.default_rng(2)
    n = 150
    x = rng.uniform(0, 1, n)
    g = rng.choice(["a", "b", "c"], n)
    y = np.sin(2 * np.pi * x) + (g == "b") * 0.5 + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x": x, "g": g, "y": y}).with_columns(
        pl.col("g").cast(pl.Enum(["a", "b", "c"]))
    )
    m = gam("y ~ s(x) + s(g, x, bs='sz', id=1)", df, method="REML")
    assert len(m.sp) == 2  # one for s(x), one merged sz penalty


def test_retest_with_sibling_random_effects_matches_mgcv():
    """reTest for one bs='re' term treats the *other* re terms as fully
    random (recov's re-branch, mgcv.r:3640-3713) — not as fixed."""
    from hea.R.rng import RGenerator

    gen = RGenerator(7)  # R-native (set.seed(7), 0-ulp)
    n = 250
    x = gen.uniform(0.0, 1.0, n)
    g1 = gen.mt.sample_int(8, n, replace=True)  # = R sample.int(8, …) - 1
    g2 = gen.mt.sample_int(6, n, replace=True)
    b1 = gen.normal(0.0, 0.5, 8)
    b2 = gen.normal(0.0, 0.09, 6)
    y = np.sin(2 * np.pi * x) + b1[g1] + b2[g2] + gen.normal(0.0, 0.5, n)
    df = pl.DataFrame(
        {
            "x": x,
            "g1": [f"a{i}" for i in g1],
            "g2": [f"b{i}" for i in g2],
            "y": y,
        }
    ).with_columns(
        pl.col("g1").cast(pl.Enum([f"a{i}" for i in range(8)])),
        pl.col("g2").cast(pl.Enum([f"b{i}" for i in range(6)])),
    )
    m = gam("y ~ s(x) + s(g1, bs='re') + s(g2, bs='re')", df, method="REML")
    rows = {r[0]: r[1:] for r in m._smooth_significance_rows()}
    np.testing.assert_allclose(
        rows["s(g1)"],
        (6.72214656471, 7.0, 24.6471297862, 0.0),
        rtol=5e-4,
        atol=1e-12,
        err_msg="s(g1) vs mgcv",
    )
    np.testing.assert_allclose(
        rows["s(g2)"],
        (3.51278498412, 5.0, 3.49421820734, 0.00267046065254),
        rtol=5e-4,
        err_msg="s(g2) vs mgcv",
    )


def _noncanonical_pirls_data():
    from hea.R.rng import RGenerator

    gen = RGenerator(101)
    n = 400
    x = gen.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * x**2
    mu = np.exp(0.3 + f)
    y_gamma = gen.gamma(3.0, scale=mu / 3.0, size=n)
    y_glog = mu + gen.normal(0, 1.0, n)
    return pl.DataFrame({"x": x, "yg": y_gamma, "yn": y_glog})


def test_pirls_noncanonical_gamma_log_matches_mgcv():
    # gam(yg ~ s(x), Gamma(log), REML) — non-canonical link, so the inner
    # loop takes full-Newton steps (gam.fit3.r:118). mgcv 1.9-4 reference.
    df = _noncanonical_pirls_data()
    m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    np.testing.assert_allclose(
        m.REML_criterion / 2, 499.0247951500, rtol=0, atol=1e-6, err_msg="REML"
    )
    np.testing.assert_allclose(
        float(np.sum(m.edf)), 7.7000813600, rtol=0, atol=5e-4, err_msg="edf"
    )
    np.testing.assert_allclose(
        m.sigma_squared, 0.3237704458, rtol=0, atol=5e-6, err_msg="sig2"
    )


def test_pirls_gaussian_log_negative_newton_weights_match_mgcv():
    # gaussian(link="log") with y ≤ 0 in 42 rows: needs mgcv's fix.family
    # starting values (mustart = pmax(y, .01·sd(y)), gam.fit3.r:2550), and
    # at convergence several rows carry *negative* Newton weights — mgcv
    # keeps the signed weights in the REML score (gam.fit3.r:505-515). A
    # Fisher-fallback score is off by ~0.06 on this criterion.
    df = _noncanonical_pirls_data()
    from hea.family import Gaussian

    m = gam("yn ~ s(x)", df, family=Gaussian(link="log"), method="REML")
    np.testing.assert_allclose(
        m.REML_criterion / 2, 587.3725200800, rtol=0, atol=1e-6, err_msg="REML"
    )
    np.testing.assert_allclose(m.sp[0], 0.1261684077, rtol=1e-3, err_msg="sp")
    np.testing.assert_allclose(
        float(np.sum(m.edf)), 7.4639360300, rtol=0, atol=1e-3, err_msg="edf"
    )
    np.testing.assert_allclose(
        m.sigma_squared, 1.0350749500, rtol=0, atol=5e-6, err_msg="sig2"
    )
    b0 = float(np.asarray(m.coef)[0])
    np.testing.assert_allclose(b0, 0.5001308373, rtol=0, atol=1e-5, err_msg="intercept")


def test_pirls_clean_fits_emit_no_warnings():
    import warnings as w

    df = _noncanonical_pirls_data()
    with w.catch_warnings():
        w.simplefilter("error")
        m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    assert float(np.sum(m.edf)) > 1.0


def test_null_deviance_offset_and_no_intercept_match_mgcv():
    # mgcv: gam.fit3 always runs with intercept=TRUE (mgcv.r:1667) so the
    # base null deviance is dev(weighted-mean) for every formula; for
    # intercept+offset models estimate.gam refits glm(y ~ offset(off))
    # (mgcv.r:2072-2075). df.null = n-1 always. mgcv 1.9-4 references.
    from hea.R.rng import RGenerator

    gen = RGenerator(33)  # R-native (set.seed(33), 0-ulp)
    n = 200
    x = gen.uniform(0.0, 1.0, n)
    expo = gen.uniform(0.5, 2.0, n)
    mu = expo * np.exp(0.4 + np.sin(2 * np.pi * x))
    y = np.asarray(gen.poisson(mu), dtype=float)
    df = pl.DataFrame({"x": x, "expo": expo, "y": y})
    from hea.family import Poisson

    m = gam("y ~ s(x) - 1 + offset(log(expo))", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(m.null_deviance, 493.6575994189, rtol=0, atol=1e-7)
    assert m.df_null == n - 1

    m2 = gam("y ~ s(x) + offset(log(expo))", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(m2.null_deviance, 408.7157165875, rtol=0, atol=1e-7)

    # scaled.pearson = pearson/√φ̂ (mgcv.r:3457); φ=1 for Poisson so the
    # no-intercept fit pins R's residuals(m, "scaled.pearson") directly.
    r = m.residuals_of("scaled.pearson")
    np.testing.assert_allclose(
        r[:3],
        [-0.4081571746, -0.0352311928, -0.3567882629],
        rtol=0,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="scaled.pearson"):
        m.residuals_of("nonsense")


# ---------------------------------------------------------------------------
# 1.8 tp/ds/sos max.knots seeded subsampling + R RNG port. mgcv takes a
# seeded random subsample of 2000 knots when a smooth has > 2000 unique
# covariate locations (smooth.r:1286/3031/3239, temp.seed(1)); matching it
# requires R's RNG bit-for-bit (set.seed scrambling, MT19937, R_unif_index
# rejection sampling, do_sample pool walk).
# ---------------------------------------------------------------------------


def test_r_rng_port_is_bit_exact():
    from hea.formula import _RUnif

    r = _RUnif(1)
    np.testing.assert_array_equal(
        [r.unif_rand() for _ in range(5)],
        [
            0.26550866314209998,
            0.37212389963679016,
            0.57285336335189641,
            0.90820778999477625,
            0.2016819310374558,
        ],
    )
    np.testing.assert_array_equal(
        _RUnif(1).sample_int(4000, 8) + 1,
        [1017, 3908, 679, 2177, 930, 1533, 471, 2347],
    )
    np.testing.assert_array_equal(
        _RUnif(42).sample_int(10, 10) + 1,
        [1, 5, 10, 8, 2, 4, 6, 9, 7, 3],
    )
    s = _RUnif(1).sample_int(3000, 2000) + 1
    assert s.sum() == 2979991
    assert s[:5].tolist() == [1017, 679, 2177, 930, 1533]
    assert s[-3:].tolist() == [2694, 2568, 1897]
    from hea.models.bam import RMersenneTwister as _bam_alias
    from hea.R import RMersenneTwister

    assert type(_RUnif(1)) is RMersenneTwister is _bam_alias
    r_scalar = _RUnif(1)
    np.testing.assert_array_equal(
        RMersenneTwister(1).unif_rand(5), [r_scalar.unif_rand() for _ in range(5)]
    )
    np.testing.assert_array_equal(
        RMersenneTwister(1).sample_int(5, 4, replace=True) + 1, [1, 4, 1, 2]
    )
    assert RMersenneTwister(3).permute(["a", "b", "c"]).tolist() == ["a", "b", "c"]
    assert RMersenneTwister(4).permute(["a", "b", "c"]).tolist() == list("cab")


def test_tp_max_knots_subsample_matches_mgcv():
    from hea.R.rng import RGenerator

    gen = RGenerator(2024)
    n = 4000
    x = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x) + gen.normal(0, 0.4, n)
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x, k=20)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 2026.8098237600, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.007596398516, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 16.9507672900, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sigma_squared, 0.1583687217, rtol=0, atol=1e-8)


def _tp_ds_2d_data():
    from hea.R.rng import RGenerator

    gen = RGenerator(77)
    n = 3000
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = (
        np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)
        + (x1 - 0.5) ** 2
        + gen.normal(0, 0.3, n)
    )
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_tp_2d_subsample_matches_mgcv():
    df = _tp_ds_2d_data()
    m = gam("y ~ s(x1, x2, k=40)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 706.9094382400, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1278327768, rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 38.2325721600, rtol=0, atol=1e-4)


def test_ds_subsample_and_reported_sp_match_mgcv():
    df = _tp_ds_2d_data()
    m3 = gam("y ~ s(x1, x2, bs='ds', k=40)", df, method="REML")
    np.testing.assert_allclose(m3.REML_criterion / 2, 703.3490103000, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m3.sp[0], 0.03473596538, rtol=1e-5)

    m15 = gam("y ~ s(x1, x2, bs='ds', k=40)", df.head(1500), method="REML")
    np.testing.assert_allclose(
        m15.REML_criterion / 2, 348.9165255200, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(m15.sp[0], 0.02406765059, rtol=1e-6)


def test_initial_spg_seed_matches_mgcv():
    df = _noncanonical_pirls_data()
    m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    np.testing.assert_allclose(m._initial_sp_rho(), [0.7701263106], rtol=0, atol=1e-8)
    y = df["yg"].to_numpy()
    mu0 = np.full(len(y), y.mean())
    ns = float(np.sum(m.family.dev_resids(y, mu0, np.ones(len(y))))) / len(y)
    np.testing.assert_allclose(np.log(ns / 10.0), -2.7357189373, rtol=0, atol=1e-8)

    df2 = _tp_ds_2d_data()
    m2 = gam("y ~ s(x1, x2, k=40) + s(x1, k=10)", df2, method="REML")
    np.testing.assert_allclose(
        m2._initial_sp_rho(),
        [1.3431758542, 2.5092598794],
        rtol=0,
        atol=1e-8,
    )

    d = load_dataset("MASS", "mcycle")
    m3 = gam("accel ~ s(times)", d, method="REML")
    np.testing.assert_allclose(m3._initial_sp_rho(), [-1.0127433870], rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(m3.edf)) - 1.0, 8.624691, rtol=0, atol=2e-5)


def _tw_24_data():
    from hea.R.rng import RGenerator

    gen = RGenerator(55)
    n = 500
    x = gen.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x))
    y = gen.gamma(2.0, scale=mu / 2.0, size=n)
    y[gen.uniform(0, 1, n) < 0.08] = 0.0
    return pl.DataFrame({"x": x, "y": y})


def test_tw_reml_scale_and_vc_match_mgcv():
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 818.3711103000, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.8716551837, rtol=0, atol=1e-6)
    assert m.sigma_squared == np.exp(m._log_phi_hat)
    np.testing.assert_allclose(m.sp[0], 0.08512769779, rtol=1e-4)
    # Vc/edf2 with the family-θ column of db.drho. Vc2's Cholesky seed uses
    # the Fisher penalized Hessian, like gam.fit3.post.proc's R
    # (gam.fit4.r:798 Fisher-type weights).
    np.testing.assert_allclose(np.diag(m.Vc)[0], 0.001504373687, rtol=1e-6)
    np.testing.assert_allclose(m.edf2_total, 7.45509931, rtol=0, atol=1e-6)


def test_tw_db_dtheta_column_matches_finite_differences():
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="REML")
    fit = m._fit_given_rho(m._rho_hat)
    analytic = m._db_dtheta_fam(fit)[:, 0]
    th0 = float(m.family.get_theta()[0])
    delta = 1e-5
    m.family.set_theta(np.array([th0 + delta]))
    b_plus = m._fit_given_rho(m._rho_hat).beta
    m.family.set_theta(np.array([th0 - delta]))
    b_minus = m._fit_given_rho(m._rho_hat).beta
    m.family.set_theta(np.array([th0]))
    fd = (b_plus - b_minus) / (2 * delta)
    np.testing.assert_allclose(analytic, fd, rtol=5e-3, atol=1e-6)


def test_tw_ml_projected_gradient_matches_mgcv():
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 816.3142306600, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.8698129495, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.09324923391, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2.5 gam.control(edge.correct=): post-convergence walk of Hessian-flat
# smoothing parameters + the k=2 Vc recomputation with the weaker 1e-7 Vr
# prior (gam.fit3.r:1670-1716, post.proc K loop).
# ---------------------------------------------------------------------------


def test_edge_correct_vc_matches_mgcv():
    from hea.R.rng import RGenerator

    gen = RGenerator(99)
    n = 300
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + gen.normal(0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m1 = gam("y ~ s(x1) + s(x2)", df, method="REML", control={"edge_correct": True})
    np.testing.assert_allclose(
        np.diag(m1.Vc)[[1, 9, 10]],
        [0.0323371108, 0.101825207, 0.00661200289],
        rtol=5e-4,
    )
    np.testing.assert_allclose(m1.edf2_total, 10.72965196, rtol=2e-5)

    gen = RGenerator(14)  # R-native seed 7 over-saturates s(x2)
    n = 200  # (edge-correct Vc then numerically delicate);
    x1 = gen.uniform(0, 1, n)  # 14 gives the original's mild-saturation regime
    x2 = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + gen.normal(0, 0.25, n)
    df2 = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m2 = gam("y ~ s(x1) + s(x2)", df2, method="REML", control={"edge_correct": True})
    np.testing.assert_allclose(
        np.diag(m2.Vc)[[1, 10, 11]],
        [0.0310210091, 0.00438953063, 0.00894119165],
        rtol=5e-4,
    )
    m0 = gam("y ~ s(x1) + s(x2)", df2, method="REML")
    np.testing.assert_allclose(
        np.diag(m0.Vc)[[1, 10, 11]],
        [0.0309144583, 0.00332593994, 0.00649458005],
        rtol=5e-4,
    )

    with pytest.raises(ValueError, match="edge_correct"):
        gam("y ~ s(x1)", df2, method="REML", control={"edge_correct": -1.0})


def test_edge_correct_general_family_matches_mgcv():
    # gam.control(edge.correct=) on the gam.fit5 rail: newton's
    # post-convergence walk (gam.fit3.r:1669-1713, dispatching to
    # gam.fit5 at deriv 0) + the post.proc K=2 loop (gam.fit4.r:
    # 1650-1663) — reported Vc/V.sp from the edge-corrected deriv-2
    # refit at lsp1, edf2 from the un-shifted k=1 quantities with the
    # extra repara pair (gam.fit4.r:1691-1694), and sp.vcov's hess1
    # branch with mgcv's DIAGONAL regularizer (mgcv.r:4227-4229).
    # mgcv 1.9-4: gam(list(y~s(x)+s(w), ~s(z)), gaulss(), REML,
    # control=gam.control(edge.correct=TRUE)) — s(w) has linear truth,
    # its sp sits at working infinity (~2.1e5) and the walk moves it
    # back exactly 6 unit log steps (lsp 12.2703 → lsp1 6.2703).
    #
    # Tolerances are RECEIPT-derived, not felt. Along the flat
    # direction newton's stop criterion (|grad| < score_scale·1e-6 =
    # 2.2e-4) meets curvature H₁₁ = 1.08e-4, so the flat endpoint may
    # legitimately sit O(1) log-sp units from the stationary point —
    # each platform's arithmetic stops it somewhere else (darwin
    # landed 4e-4 from mgcv's, a linux CI ~2e-3). Measured
    # sensitivities at ±0.02 endpoint budget: V1[0,1] ~1e-3 rel,
    # V1[1,1] ~2e-2 rel, lsp1[1] one-for-one. So: well-determined
    # entries pin at 1e-4; flat-COUPLED cross terms get 4e-3; the
    # flat direction's OWN entries are asserted structurally (the
    # walk's exact 6.0 steps at 1e-8; the 35× sp-variance collapse
    # 903 → 25 that IS the edge correction) instead of to digits.
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(
        ["y ~ s(x) + s(w)", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        control={"edge_correct": True},
    )
    np.testing.assert_allclose(m.sp[[0, 2]], [0.1441987021, 0.2282998535], rtol=1e-4)
    assert m.sp[1] > 1e5  # working infinity
    np.testing.assert_allclose(m.REML_criterion / 2, 218.1333837242, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.log(m.sp) - m._lsp1_ec, [0.0, 6.0, 0.0], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(m._lsp1_ec)[[0, 2]], [-1.936563055, -1.477095367], rtol=1e-4
    )
    assert abs(m._lsp1_ec[1] - 6.270342521) < 0.1
    np.testing.assert_allclose(
        np.diag(m.Vc)[[0, 1, 2, 9, 19]],
        [0.0008154561995, 0.06446281033, 0.4595239323, 0.1837733338, 0.002468328152],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.diag(m.Vc)[11], 0.003206806604, rtol=5e-3
    )  # flat smooth's entry
    # edf2 keeps the fitted-model k=1 pieces — but differs from the
    # plain fit through mgcv's repara pair on Vc1 (gam.fit4.r:1691).
    np.testing.assert_allclose(m.edf2_total, 16.1095862277, rtol=1e-5)
    np.testing.assert_allclose(m.edf_total, 13.9708235675, rtol=1e-6)
    # sp.vcov: edge branch = solve(hess1 + diag·reg); plain branch =
    # solve(hess + reg) elementwise (mgcv.r:4227-4231).
    V1 = m.sp_vcov()
    np.testing.assert_allclose(
        [V1[0, 0], V1[0, 2], V1[2, 2]],
        [0.3653756883, 0.009384734478, 0.8330026212],
        rtol=1e-4,
    )
    np.testing.assert_allclose(V1[0, 1], -0.004164655708, rtol=4e-3)
    V0 = m.sp_vcov(edge_correct=False)
    np.testing.assert_allclose(
        [V0[0, 0], V0[2, 2]], [0.365438595, 0.8330972547], rtol=1e-3
    )
    np.testing.assert_allclose(V0[0, 1], -0.3386173704, rtol=5e-3)
    assert 20.0 < V1[1, 1] < 32.0
    assert 800.0 < V0[1, 1] < 1000.0
    Vsp = np.asarray(m._V_sp)
    np.testing.assert_allclose(
        [Vsp[0, 0], Vsp[2, 2]], [0.3655093291, 0.8337004788], rtol=1e-4
    )
    assert 20.0 < Vsp[1, 1] < 33.0

    m0 = gam(["y ~ s(x) + s(w)", "~ s(z)"], df, family=gaulss(), method="REML")
    assert m0._lsp1_ec is None and m0._hess1_ec is None
    np.testing.assert_allclose(
        np.diag(m0.Vc)[[0, 1, 2, 9, 19]],
        [0.0008127134141, 0.06429968272, 0.4544038077, 0.1834664554, 0.002468231747],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.diag(m0.Vc)[11], 1.672278335e-05, rtol=5e-3, atol=1e-7
    )
    np.testing.assert_allclose(m0.edf2_total, 14.7125330455, rtol=1e-5)
    np.testing.assert_allclose(
        m0.sp_vcov(), m0.sp_vcov(edge_correct=False), rtol=0, atol=0
    )

    m2 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        control={"edge_correct": True},
    )
    np.testing.assert_allclose(m2.REML_criterion / 2, 216.8833933834, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m2._lsp1_ec, np.log(m2.sp), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.diag(m2.Vc)[[0, 1, 2, 9, 11, 19]],
        [
            0.003195481437,
            0.009735812132,
            0.06430404532,
            1.044650124,
            0.002468292745,
            1.610887396,
        ],
        rtol=1e-4,
    )
    np.testing.assert_allclose(m2.edf2_total, 16.1091522577, rtol=1e-5)
    V1 = m2.sp_vcov()
    np.testing.assert_allclose(
        [V1[0, 0], V1[0, 1], V1[1, 1]],
        [0.3653182258, 0.009403836416, 0.8324621223],
        rtol=1e-5,
    )
    V0 = m2.sp_vcov(edge_correct=False)
    np.testing.assert_allclose(
        [V0[0, 0], V0[0, 1], V0[1, 1]],
        [0.3653114662, 0.009099642886, 0.832446719],
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# 2.2 gam.reparam / get_stableS: log|Sλ|+ and its ρ-derivatives via mgcv's
# similarity-transform reparameterization (gdi.c:550-792) — immune to
# λ-ratio "machine zero leakage" between penalty components.
# ---------------------------------------------------------------------------


def test_get_stable_s_matches_mgcv_oracle():
    from hea.models.gam import _gam_reparam
    from hea.R.rng import RGenerator

    gen = RGenerator(404)
    R1 = gen.normal(0, 1, 18).reshape((3, 6), order="F").round(6)
    gen.normal(0, 1, 18)  # R2 drawn but unused here (keeps stream)
    gen2 = RGenerator(405)
    R3 = gen2.normal(0, 1, 24).reshape((4, 6), order="F").round(6)

    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 0.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 6.0727466274, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        rp["det1"], [2.6513413098, 3.3486586902], rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        rp["det2"],
        [[0.2270958080, -0.2270958080], [-0.2270958080, 0.2270958080]],
        rtol=0,
        atol=1e-9,
    )
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 20.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 85.0190848310, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        rp["det1"], [2.0000000039, 3.9999999961], rtol=0, atol=1e-8
    )
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 40.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 165.0190848271, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rp["det1"], [2.0, 4.0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        rp["E"].T @ rp["E"], rp["S"], rtol=0, atol=1e-12 * np.abs(rp["S"]).max()
    )
    S1, S3 = R1.T @ R1, R3.T @ R3
    lam = np.exp(np.array([0.0, 40.0]))
    w = np.linalg.eigvalsh(lam[0] * S1 + lam[1] * S3)
    top = np.clip(np.sort(w)[::-1][:6], 1e-300, None)
    det_naive = float(np.sum(np.log(top)))
    assert abs(det_naive - 165.0190848271) > 1.0


def test_extreme_fixed_sp_tensor_criterion_matches_mgcv():
    from hea.R.rng import RGenerator

    gen = RGenerator(77)
    n = 3000
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = (
        np.sin(2 * np.pi * x1) * np.cos(np.pi * x2)
        + (x1 - 0.5) ** 2
        + gen.normal(0, 0.3, n)
    )
    df = pl.DataFrame({"x1": x1[:400], "x2": x2[:400], "y": y[:400]})

    m = gam("y ~ te(x1, x2, k=5)", df, method="REML", sp=np.array([1e-8, 1e8]))
    np.testing.assert_allclose(m.REML_criterion / 2, 164.9817197500, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.0963046569, rtol=0, atol=1e-9)

    m2 = gam("y ~ te(x1, x2, k=5)", df, method="REML")
    np.testing.assert_allclose(m2.REML_criterion / 2, 118.9165697700, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m2.sp, [0.124003427, 121.697916], rtol=1e-4)


def test_pls_qr_negative_weight_correction_is_exact():
    rng = np.random.default_rng(8)
    n, p = 60, 5
    X = rng.normal(size=(n, p))
    z = rng.normal(size=n)
    w = rng.uniform(0.2, 2.0, n)
    w[:6] = -rng.uniform(0.01, 0.05, 6)  # mildly negative rows: A stays PD
    E = np.diag(rng.uniform(0.5, 2.0, p))  # S = E'E

    from hea.models.gam import _pls_qr

    beta, R, log_det, ok = _pls_qr(X, {"g": None, "o": None}, w, z, E)
    assert ok
    A = X.T @ (w[:, None] * X) + E.T @ E
    np.testing.assert_allclose(R.T @ R, A, rtol=0, atol=1e-10)
    np.testing.assert_allclose(A @ beta, X.T @ (w * z), rtol=0, atol=1e-10)
    np.testing.assert_allclose(log_det, np.linalg.slogdet(A)[1], rtol=0, atol=1e-10)
    assert np.all(np.diag(R) > 0)  # unique Cholesky normalization

    w_bad = w.copy()
    w_bad[:20] = -5.0
    A_bad = X.T @ (w_bad[:, None] * X) + E.T @ E
    assert np.linalg.eigvalsh(A_bad).min() < 0
    *_, ok_bad = _pls_qr(X, {"g": None, "o": None}, w_bad, z, E)
    assert not ok_bad


def test_single_sp_matches_mgcv():
    # mgcv:::single.sp(X, S, target) — the target-edf single-penalty sp
    # utility (mgcv.r:4504). Reference values from mgcv on a fixed cubic
    # design; hea matches to machine precision.
    from hea.models.gam import _single_sp

    x = np.arange(10) / 10.0
    X = np.column_stack([np.ones(10), x, x**2, x**3])
    S = np.diag([0.0, 1.0, 4.0, 9.0])
    ref = {0.3: 0.227845055132977, 0.5: 0.0186045610868318, 0.7: 0.0010350029849568}
    for target, r in ref.items():
        assert _single_sp(X, S, target=target) == pytest.approx(r, rel=1e-10)
    Xd = np.column_stack([np.ones(4), np.ones(4), np.arange(4.0)])
    assert _single_sp(Xd, np.eye(3)) == -1.0


def test_vcov_and_sandwich_match_mgcv():
    # mgcv vcov.gam / gam.sandwich (mgcv.r:4396 / 4374) — R set.seed(2);
    # x<-runif(200); y<-rpois(200, exp(0.6*sin(2*pi*x))), reproduced
    # bit-for-bit by hea.R.rng (edf 5.1526182 matched). hea's vcov()/sandwich
    # accessors match mgcv to ~1e-6.
    from hea.family import Poisson, Tweedie, nb
    from hea.R.rng import RGenerator

    g = RGenerator(2)
    n = 200
    x = g.uniform(0, 1, n)
    mu = np.exp(0.6 * np.sin(2 * np.pi * x))
    y = g.poisson(mu).astype(float)
    m = gam("y ~ s(x)", pl.DataFrame({"y": y, "x": x}), family=Poisson(), method="REML")
    np.testing.assert_allclose(
        np.diag(m.vcov())[:3], [0.005480767622, 0.1999983528, 0.9787285692], rtol=1e-6
    )
    np.testing.assert_allclose(
        np.diag(m.vcov(freq=True))[:3],
        [0.005435521101, 0.127677218, 0.3855910047],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.diag(m.vcov(sandwich=True))[:3],
        [0.005526626483, 0.1927452693, 1.006391415],
        rtol=1e-6,
    )
    np.testing.assert_allclose(m.vcov(), m.Vp)
    np.testing.assert_allclose(m.vcov(unconditional=True), m.Vc)
    np.testing.assert_allclose(m.vcov(dispersion=float(m.scale)), m.vcov())
    mtw = gam(
        "y ~ s(x)",
        pl.DataFrame({"y": y + 0.1, "x": x}),
        family=Tweedie(p=1.5),
        method="REML",
    )
    assert np.all(np.isfinite(np.diag(mtw.vcov(sandwich=True))))
    # Extended families (nb/scat) use the dDeta-based meat (mgcv.r:4384) —
    # scat matches mgcv's gam.sandwich to ~1e-6 (see the scat parity below);
    # here just confirm nb's accessor produces a finite symmetric matrix.
    mnb = gam("y ~ s(x)", pl.DataFrame({"y": y, "x": x}), family=nb(), method="REML")
    Vs = mnb.vcov(sandwich=True)
    assert np.all(np.isfinite(Vs))
    np.testing.assert_allclose(Vs, Vs.T, atol=1e-12)


def test_sandwich_extended_matches_mgcv():
    # mgcv gam.sandwich extended-family branch (mgcv.r:4384) — R set.seed(4);
    # x<-runif(200); y<-2*sin(2*pi*x)+rt(200,4)*0.5; scat() fit, reproduced
    # by hea.R.rng (edf 8.0668295 matched).
    from hea.family import Scat
    from hea.R.rng import RGenerator

    g = RGenerator(4)
    n = 200
    x = g.uniform(0, 1, n)
    y = 2 * np.sin(2 * np.pi * x) + g.standard_t(4, n) * 0.5
    m = gam("y ~ s(x)", pl.DataFrame({"y": y, "x": x}), family=Scat(), method="REML")
    np.testing.assert_allclose(
        np.diag(m.vcov(sandwich=True))[:3],
        [0.001718046487, 0.1746792536, 1.240544125],
        rtol=1e-4,
    )


def test_sandwich_general_family():
    # mgcv gam.sandwich general-family branch (mgcv.r:4380-4382):
    # Vs = m·Vp·family$sandwich(...)·Vp + (Vp−Ve), where the meat is
    # ll(deriv=1, sandwich=TRUE)$lbb — gamlss.gH's l2 ← l1_i·l1_j reset
    # (gamlss.r:643-649), i.e. the per-observation coefficient-space
    # gradient outer-product sum. The meat is pinned against an
    # independent oracle: Σ_i g_i g_iᵀ with g_i the lb of a single-row ll
    # call (the standard gradient assembly, no sandwich path involved).
    from hea.family import cox_ph, gaulss
    from hea.R.rng import RGenerator

    g = RGenerator(6)
    n = 120
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    y = 2 * np.sin(2 * np.pi * x) + g.normal(0.0, 1.0, n) * np.exp(0.5 * (z - 0.5))
    df = pl.DataFrame({"y": y, "x": x, "z": z})
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=gaulss(), method="REML")

    fam = m.family
    X = m._md.X
    coef = np.asarray(m.coef.values, dtype=float)
    wt = m._wt
    meat = fam.sandwich(m._y_arr, X, coef, wt, lpi=m.lpi)
    meat_oracle = np.zeros_like(meat)
    for i in range(n):
        gi = fam.ll(
            m._y_arr[i : i + 1],
            X[i : i + 1, :],
            coef,
            wt[i : i + 1],
            lpi=m.lpi,
            deriv=1,
        )["lb"]
        meat_oracle += np.outer(gi, gi)
    np.testing.assert_allclose(meat, meat_oracle, rtol=1e-10, atol=1e-10)

    # the vcov surface assembles mgcv.r:4378/4382 exactly
    Vs = m.vcov(sandwich=True)
    Vp = np.asarray(m.Vp, dtype=float)
    Ve = np.asarray(m.Ve, dtype=float)
    mm = n / (n - float(m.edf_total))
    np.testing.assert_allclose(
        Vs, mm * Vp @ meat @ Vp + (Vp - Ve), rtol=1e-12, atol=1e-12
    )
    assert np.all(np.isfinite(Vs))
    np.testing.assert_allclose(Vs, Vs.T, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m.sp), [0.028233682, 10.41574], rtol=2e-4)
    np.testing.assert_allclose(float(m.edf_total), 9.87411766, rtol=1e-5)
    np.testing.assert_allclose(
        np.diag(Vs)[:4],
        [0.007162196223, 0.5282123533, 4.220706003, 0.4287573174],
        rtol=1e-4,
    )

    # families without the slot raise mgcv's stop (mgcv.r:4381) — cox_ph
    # defines NO $sandwich in mgcv (coxph.r has none).
    assert not cox_ph.has_sandwich
    with pytest.raises(NotImplementedError, match="no sandwich estimate"):
        t = np.sort(g.uniform(0, 10, 60))
        d = (g.uniform(0, 1, 60) > 0.4).astype(float)
        xc = g.uniform(0, 1, 60)
        mc = gam(
            "t ~ s(xc)",
            pl.DataFrame({"t": t, "d": d, "xc": xc}),
            family=cox_ph(),
            weights=d,
            method="REML",
        )
        mc.vcov(sandwich=True)


def test_ill_conditioned_design_matches_mgcv():
    from hea.R.rng import RGenerator

    gen = RGenerator(11)  # R-native (set.seed(11), 0-ulp)
    n = 150
    x = gen.uniform(10.0, 10.1, n)
    z = gen.uniform(0.0, 1.0, n)
    y = (
        0.5 * (x - 10.0)
        + 0.05 * (x - 10.0) ** 2
        + np.sin(2 * np.pi * z)
        + gen.normal(0.0, 0.2, n)
    )
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    m = gam("y ~ x + I(x^2) + I(x^3) + s(z)", df, method="REML")
    assert m.rank == 13 and m.p == 13  # no drop: below eps*100 tol
    np.testing.assert_allclose(m.REML_criterion / 2, -35.59408964, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.03464417846, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(m.edf)), 11.72237914, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m.coef)[:4],
        [-34137.84777, 10168.43994, -1009.697889, 33.42319007],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        m.predict(df.head(1))["fit"][0], -0.4805246562, rtol=0, atol=1e-7
    )


def _weights_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(42)
    n = 150
    x = gen.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * x
    y = f + gen.normal(0, 0.3, n)
    w = np.round(gen.uniform(0.5, 3.0, n), 6)
    w0 = w.copy()
    w0[5] = 0.0
    w0[77] = 0.0
    trials = (gen.mt.sample_int(20, n, replace=True) + 1).astype(float)  # {1..20}
    pr = 1.0 / (1.0 + np.exp(-(1.5 * np.sin(2 * np.pi * x))))
    ybin = gen.binomial(trials.astype(int), pr) / trials
    yg = gen.gamma(4.0, scale=np.exp(0.3 + np.sin(2 * np.pi * x)) / 4.0, size=n)
    lam = np.exp(0.2 + np.sin(2 * np.pi * x))
    N = gen.poisson(lam)
    ytw = np.array(
        [gen.gamma(3.0, scale=0.25, size=int(k)).sum() if k > 0 else 0.0 for k in N]
    )
    df = pl.DataFrame({"x": x, "y": y, "ybin": ybin, "yg": yg, "ytw": ytw})
    return df, w, w0, trials


def test_weights_gaussian_reml_matches_mgcv():
    df, w, _, _ = _weights_fixture()
    m = gam("y ~ s(x)", df, weights=w, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 43.4449049000, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01717485095, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.9805297900, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.1312939965, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.1347640443, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 119.8413439700, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 18.6463038300, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.AIC, 68.0116265500, rtol=0, atol=1e-5)
    p = m.predict(newdata=df[:2], se_fit=True)
    np.testing.assert_allclose(
        p["fit"].to_numpy(), [-0.1240120614, -0.0129772762], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        p["se.fit"].to_numpy(), [0.05989655466, 0.05780757341], rtol=0, atol=1e-8
    )
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(), [11.93296997, 0.362345133], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy(), [6.937955427, 0.3226310119], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc["upper"].to_numpy(), [20.52416938, 0.4069478461], rtol=1e-6
    )
    vc0 = m._compute_vcomp(rescale=False)
    np.testing.assert_allclose(
        vc0["std_dev"].to_numpy(), [2.764877815, 0.362345133], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc0["lower"].to_numpy(), [1.607529315, 0.3226310119], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc0["upper"].to_numpy(), [4.755464962, 0.4069478461], rtol=1e-6
    )
    # sp.vcov (single-formula path: the (ρ, log φ) outer Hessian) —
    # solve(hess + reg) with mgcv's elementwise reg (mgcv.r:4221-4234).
    np.testing.assert_allclose(
        m.sp_vcov(), [[0.31777315, 0.01283642], [0.01283642, 0.01403186]], rtol=1e-5
    )


def test_weights_unit_weights_equal_unweighted():
    df, _, _, _ = _weights_fixture()
    m0 = gam("y ~ s(x)", df, method="REML")
    m1 = gam("y ~ s(x)", df, weights=np.ones(df.height), method="REML")
    _assert_fp_equiv(m0.REML_criterion, m1.REML_criterion)
    _assert_fp_equiv(m0.coef, m1.coef)
    _assert_fp_equiv(m0.AIC, m1.AIC)


def test_weights_zero_weight_rows_match_mgcv():
    # Two rows with w=0: excluded from the working model (mgcv's `good`
    # mask) but still predicted; n.true stays nobs in the scale estimator
    # (gam.fit3.r:197); gaussian AIC = Inf via the −Σlog(w) term, exactly
    # as in R.
    df, _, w0, _ = _weights_fixture()
    m = gam("y ~ s(x)", df, weights=w0, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 42.0001302400, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01724336763, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.9579329700, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.1273456593, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.null_deviance, 115.9126091300, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 18.0884406700, rtol=0, atol=1e-7)
    assert np.isinf(m.AIC)
    assert np.all(np.isfinite(m.fitted_values))


def test_weights_gaussian_ml_and_gcv_match_mgcv():
    df, w, _, _ = _weights_fixture()
    m = gam("y ~ s(x)", df, weights=w, method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 40.4562298300, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.1312683055, rtol=0, atol=5e-8)
    np.testing.assert_allclose(m.AIC, 67.9331926600, rtol=0, atol=1e-4)

    g = gam("y ~ s(x)", df, weights=w)  # GCV.Cp
    np.testing.assert_allclose(g.GCV_score, 0.1372620019, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp[0], 0.06309683842, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(g.edf)), 6.5217061100, rtol=0, atol=1e-4)
    np.testing.assert_allclose(g.sigma_squared, 0.1312941190, rtol=0, atol=1e-7)
    np.testing.assert_allclose(g.AIC, 66.4288391800, rtol=0, atol=1e-5)


def test_weights_binomial_trials_match_mgcv():
    from hea.family import Binomial

    df, _, _, trials = _weights_fixture()
    m = gam("ybin ~ s(x)", df, weights=trials, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05623115534, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8716212600, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2337119773, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.null_deviance, 519.0955475400, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 168.9992855600, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[0], 0.3041718423, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.residuals[0], -0.339290643, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.edf2_total, 7.0512421100, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(np.abs(m.Vc))), 25.2730419100, rtol=1e-6)
    np.testing.assert_allclose(m.AIC, 514.3482544600, rtol=0, atol=1e-5)

    u = gam("ybin ~ s(x)", df, weights=trials, family=Binomial())  # UBRE
    np.testing.assert_allclose(u.GCV_score, 0.2137947813, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp[0], 0.137857243, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(u.edf)), 5.8914782800, rtol=0, atol=1e-4)
    np.testing.assert_allclose(u.AIC, 513.3157018800, rtol=0, atol=1e-4)


def test_weights_gamma_log_reml_matches_mgcv():
    df, w, _, _ = _weights_fixture()
    m = gam("yg ~ s(x)", df, weights=w, family=Gamma(link="log"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 145.5882797100, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05565807889, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6626274000, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.4405849487, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.1507651133, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 193.5417200300, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 69.7617899100, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m._mgcv_aic, 457.8559119000, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.AIC, 458.4093699300, rtol=0, atol=1e-6)


def test_weights_tw_matches_mgcv():
    # Extended family: weighted tw deviance/Dd chain, the Tweedie ls
    # convention (weight OUTSIDE the density at unmodified φ — mgcv
    # efam.r:3224), sig2 = exp(φ̂), the θ-gradient with weights, and
    # logLik df += n_theta.
    from hea.family import tw

    df, w, _, _ = _weights_fixture()
    m = gam("ytw ~ s(x)", df, weights=w, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 327.4641111400, rtol=0, atol=5e-5)
    np.testing.assert_allclose(m.sp[0], 0.01967446122, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.5730820200, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sigma_squared, 0.8900262323, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.3622523656, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.deviance, 242.0395602100, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m._tw_info["p_hat"], 1.220015110, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.AIC, 645.1530726700, rtol=0, atol=1e-5)


def test_weights_validation():
    df, w, _, _ = _weights_fixture()
    with pytest.raises(ValueError, match="length"):
        gam("y ~ s(x)", df, weights=w[:-1], method="REML")
    bad = w.copy()
    bad[3] = -1.0
    with pytest.raises(ValueError, match="negative"):
        gam("y ~ s(x)", df, weights=bad, method="REML")
    nan = w.copy()
    nan[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gam("y ~ s(x)", df, weights=nan, method="REML")


def _cbind_fixture():
    df, w, _, trials = _weights_fixture()
    ybin = df["ybin"].to_numpy()
    succ = np.rint(ybin * trials)
    fail = trials - succ
    d = df.with_columns(pl.Series("succ", succ), pl.Series("fail", fail))
    return d, w, trials


def test_cbind_response_equals_proportion_idiom_and_mgcv():
    from hea.family import Binomial

    d, _, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05623115534, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8716212600, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.AIC, 514.3482544600, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        m._y_arr[:3], [0.25, 0.0, 0.8333333333], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(m.prior_weights, trials, rtol=0, atol=0)
    assert m.formula == "cbind(succ, fail) ~ s(x)"
    p = gam("ybin ~ s(x)", d, weights=trials, family=Binomial(), method="REML")
    _assert_fp_equiv(m.coef, p.coef)
    _assert_fp_equiv(m.sp, p.sp)
    _assert_fp_equiv(m.REML_criterion, p.REML_criterion)
    _assert_fp_equiv(m.AIC, p.AIC)


def test_bracket_response_equals_cbind_binomial():
    """`[succ, fail] ~ s(x)` is hea-dialect sugar for `cbind(succ, fail) ~
    s(x)`. Exercises the gam intake fix: the binomial two-column rewrite is
    gated on the parsed AST, not a `"cbind" in formula` substring, so the
    bracket form (no literal "cbind") still takes the two-column path. Before
    the fix this test fails — `[succ, fail]` is mis-handled as a univariate
    response."""
    from hea.family import Binomial

    d, _, _ = _cbind_fixture()
    m_br = gam("[succ, fail] ~ s(x)", d, family=Binomial(), method="REML")
    m_cb = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(), method="REML")
    _assert_fp_equiv(m_br.coef, m_cb.coef)
    _assert_fp_equiv(m_br.sp, m_cb.sp)
    _assert_fp_equiv(m_br.REML_criterion, m_cb.REML_criterion)
    _assert_fp_equiv(m_br.AIC, m_cb.AIC)
    assert m_br.formula == "[succ, fail] ~ s(x)"


def test_cbind_with_prior_weights_matches_mgcv():
    from hea.family import Binomial

    d, w, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(), method="REML", weights=w)
    np.testing.assert_allclose(m.REML_criterion / 2, 441.7395348800, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05420557511, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.4910195100, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2673184329, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 301.4710899000, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 918.3830225500, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        m.fitted_values[:3],
        [0.3049423071, 0.3403670810, 0.7793596206],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(m._mgcv_aic, 867.0709067200, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.AIC, 867.3729279400, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.loglike, -426.0444338500, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.prior_weights, w * trials, rtol=0, atol=0)
    np.testing.assert_allclose(m.Vp[0, 0], 0.001974271296, rtol=1e-6)
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    assert label == "s(x)"
    np.testing.assert_allclose(
        [edf, ref_df, stat], [6.4910195050, 7.6159846940, 508.9404339000], rtol=1e-6
    )
    assert p_val < 1e-10
    u = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(), weights=w)
    np.testing.assert_allclose(u.GCV_score, 1.1040181364, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp[0], 0.1532665613, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(u.edf)), 6.3273479700, rtol=0, atol=1e-4)
    np.testing.assert_allclose(u.AIC, 866.2204982700, rtol=0, atol=1e-4)


def test_cbind_zero_trials_row_matches_mgcv():
    # A zero-trials row: y ← 0, weight ← 0 (R initialize), excluded from
    # the fit via the `good` mask but still predicted; df.null counts it
    # out (n.ok − 1, gam.fit3.r:843-844).
    from hea.family import Binomial

    d, _, trials = _cbind_fixture()
    trials0 = trials.copy()
    trials0[10] = 0.0
    succ0 = d["succ"].to_numpy().copy()
    succ0[10] = 0.0
    d0 = d.with_columns(pl.Series("succ0", succ0), pl.Series("fail0", trials0 - succ0))
    m = gam("cbind(succ0, fail0) ~ s(x)", d0, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 261.9908232900, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0562991662, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8638531900, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 168.4022821200, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[10], 0.5778095857, rtol=0, atol=1e-8)
    assert m._y_arr[10] == 0.0 and m.prior_weights[10] == 0.0
    assert m.df_null == 148.0
    np.testing.assert_allclose(m.AIC, 511.2769694600, rtol=0, atol=1e-5)


def test_cbind_intake_validation():
    from hea.family import Binomial, Poisson

    d, _, _ = _cbind_fixture()
    with pytest.raises(ValueError, match="family=Binomial"):
        gam("cbind(succ, fail) ~ s(x)", d, family=Poisson(), method="REML")
    with pytest.raises(ValueError, match="exactly two"):
        gam("cbind(succ, fail, x) ~ s(x)", d, family=Binomial(), method="REML")
    neg = d.with_columns((pl.col("succ") - 100.0).alias("succ_n"))
    with pytest.raises(ValueError, match="negative counts"):
        gam("cbind(succ_n, fail) ~ s(x)", neg, family=Binomial(), method="REML")
    frac = d.with_columns((pl.col("succ") + 0.4).alias("succ_f"))
    with pytest.warns(UserWarning, match="non-integer counts"):
        gam("cbind(succ_f, fail) ~ s(x)", frac, family=Binomial(), method="REML")
    tot = d.with_columns((pl.col("succ") + pl.col("fail")).alias("tot"))
    m = gam("cbind(tot - fail, fail) ~ s(x)", tot, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800, rtol=0, atol=1e-6)
    from hea.models.bam import bam

    with pytest.raises(ValueError, match="family=Binomial"):
        bam("cbind(succ, fail) ~ s(x)", d, family=Poisson())
    with pytest.raises(ValueError, match="negative counts"):
        bam("cbind(succ_n, fail) ~ s(x)", neg, family=Binomial())


def test_quasipoisson_through_gam_matches_mgcv():
    # EQL ls (fix.family.ls quasi branch), Fletcher scale with poisson
    # dvar, F-flavor s.table (scale estimated), AIC/logLik NA in R →
    # NaN here. family=quasipoisson (bare class) mirrors R's
    # function-valued family= (mgcv.r:2324).
    from hea.family import quasipoisson

    d, _, _ = _cbind_fixture()
    m = gam("succ ~ s(x)", d, family=quasipoisson, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 141.0246116800, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.2305148526, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.1846454500, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.9841906748, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 1.4583994509, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 309.2978228100, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 495.2322484000, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        m.fitted_values[:3],
        [3.2802454640, 3.6533186760, 8.4610051790],
        rtol=0,
        atol=1e-8,
    )
    assert np.isnan(m.AIC) and np.isnan(m.loglike)
    _label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose(
        [edf, ref_df, stat], [5.1846454520, 6.2624900930, 12.6559374100], rtol=1e-6
    )
    assert p_val < 1e-8
    g = gam("succ ~ s(x)", d, family=quasipoisson)
    np.testing.assert_allclose(g.GCV_score, 2.2429195192, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp[0], 0.2674208059, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(g.edf)), 6.0273140000, rtol=0, atol=1e-4)
    # Non-canonical link → full-Newton inner loop (canonical is log,
    # gam.fit3.r:2318).
    s = gam("succ ~ s(x)", d, family=quasipoisson(link="sqrt"), method="REML")
    np.testing.assert_allclose(s.REML_criterion / 2, 140.2671306200, rtol=0, atol=1e-6)
    np.testing.assert_allclose(s.sp[0], 0.2182661, rtol=1e-4)
    np.testing.assert_allclose(s.sigma_squared, 1.9665073702, rtol=0, atol=1e-8)


def test_quasibinomial_cbind_through_gam_matches_mgcv():
    from hea.family import quasibinomial

    d, _, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=quasibinomial, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, -66.3069511400, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.06605267459, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6904930000, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.0393925494, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2331078925, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 169.1760060600, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        m.fitted_values[:3],
        [0.3031568283, 0.3358585931, 0.7850271713],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(m.prior_weights, trials, rtol=0, atol=0)
    assert np.isnan(m.AIC)
    _label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose(
        [edf, ref_df, stat], [5.6904929960, 6.8217009380, 40.8698983000], rtol=1e-6
    )
    assert p_val < 1e-10
    p = gam("ybin ~ s(x)", d, weights=trials, family=quasibinomial, method="REML")
    _assert_fp_equiv(m.REML_criterion, p.REML_criterion)
    _assert_fp_equiv(m.coef, p.coef)


def _mixed_sp_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(7)
    n = 160
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    w2 = gen.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * (z - 0.5) ** 2 * 8 + 0.3 * np.cos(3 * np.pi * w2)
    y = f + gen.normal(0, 0.4, n)
    ycnt = gen.poisson(np.exp(0.3 + np.sin(2 * np.pi * x) + 0.5 * z))
    lam = np.exp(0.2 + np.sin(2 * np.pi * x))
    N = gen.poisson(lam)
    ytw = np.array(
        [gen.gamma(3.0, scale=0.25, size=int(k)).sum() if k > 0 else 0.0 for k in N]
    )
    lam2 = np.exp(0.2 + np.sin(2 * np.pi * x) + 1.2 * (z - 0.5) ** 2 * 3)
    N2 = gen.poisson(lam2)
    ytw2 = np.array(
        [gen.gamma(3.0, scale=0.25, size=int(k)).sum() if k > 0 else 0.0 for k in N2]
    )
    return pl.DataFrame(
        {
            "x": x,
            "z": z,
            "w2": w2,
            "y": y,
            "ycnt": ycnt.astype(float),
            "ytw": ytw,
            "ytw2": ytw2,
        }
    )


def test_mixed_sp_gaussian_matches_mgcv():
    # gam(sp=c(2, -1)): first sp fixed, second estimated — folded into
    # (L, lsp0) exactly like mgcv.r:1513-1538. m.sp is the FREE working
    # vector (mgcv m$sp); full_sp the per-penalty expansion (m$full.sp).
    df = _mixed_sp_fixture()
    m = gam("y ~ s(x) + s(z)", df, sp=np.array([2.0, -1.0]), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 152.1896738580, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.1401341495], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp, [2.0, 0.1401341495], rtol=1e-4)
    assert m.full_sp[0] == 2.0
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8610338694, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.3220580505, rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.2246565999, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        m.fitted_values[:2], [-0.5851315520, 0.7295423129], rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(m.Vp[0, 0], 0.002012862816, rtol=1e-6)
    # s(x, sp=2) is the same model (R: diff exactly 0), and a gam-level
    # vector is overridden by the per-smooth value (mgcv.r:1426).
    m2 = gam("y ~ s(x, sp=2) + s(z)", df, method="REML")
    _assert_fp_equiv(m2.REML_criterion, m.REML_criterion)
    _assert_fp_equiv(m2.coef, m.coef)
    m3 = gam("y ~ s(x, sp=2) + s(z)", df, sp=np.array([5.0, -1.0]), method="REML")
    _assert_fp_equiv(m3.REML_criterion, m.REML_criterion)
    m4 = gam("y ~ s(x) + s(z)", df, sp=np.array([-1.0, -1.0]), method="REML")
    m5 = gam("y ~ s(x) + s(z)", df, method="REML")
    _assert_fp_equiv(m4.REML_criterion, m5.REML_criterion)


def test_mixed_sp_zero_gcv_te_and_id_match_mgcv():
    df = _mixed_sp_fixture()
    # Fixed sp=0 → mgcv's "effective zero" replacement
    # (‖X_1‖_F²/‖S_1‖_F·eps·0.1, mgcv.r:1519-1527 — incl. the literal
    # loop-counter quirk); full.sp[0] pinned to R's exact fudge value.
    m = gam("y ~ s(x, sp=0) + s(z)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 248.0359363930, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.05506103452], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp[0], 3.230203061e-17, rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 14.8171854238, rtol=0, atol=1e-4)
    g = gam("y ~ s(x, sp=2) + s(z)", df)
    np.testing.assert_allclose(g.GCV_score, 0.3364655582, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp, [0.1158917722], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(g.edf)), 7.0386466421, rtol=0, atol=1e-4)
    t = gam("y ~ te(x, z, sp=c(1, -1))", df, method="REML")
    np.testing.assert_allclose(t.REML_criterion / 2, 125.0775025180, rtol=0, atol=1e-6)
    np.testing.assert_allclose(t.sp, [1.006184059], rtol=1e-4)
    np.testing.assert_allclose(t.full_sp, [1.0, 1.006184059], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(t.edf)), 16.5165951050, rtol=0, atol=1e-4)
    i = gam(
        "y ~ s(x, id=1) + s(z, id=1) + s(w2)",
        df,
        sp=np.array([3.0, -1.0]),
        method="REML",
    )
    np.testing.assert_allclose(i.REML_criterion / 2, 162.4801359980, rtol=0, atol=1e-6)
    np.testing.assert_allclose(i.sp, [0.2410361343], rtol=1e-4)
    np.testing.assert_allclose(i.full_sp, [3.0, 3.0, 0.2410361343], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(i.edf)), 8.1520975718, rtol=0, atol=1e-4)


def test_mixed_sp_tw_and_poisson_match_mgcv():
    from hea.family import Poisson, tw

    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x, sp=1) + s(z)", df, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 259.1938532110, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.7153525883], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp, [1.0, 0.7153525883], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.2018656475, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m._tw_info["p_hat"], 1.2491176801, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 1.1846344441, rtol=0, atol=1e-6)
    p = gam("ycnt ~ s(x, sp=0.5) + s(z)", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(p.REML_criterion / 2, 261.6789254290, rtol=0, atol=1e-6)
    np.testing.assert_allclose(p.sp, [36789.51172], rtol=1e-3)
    np.testing.assert_allclose(p.full_sp[0], 0.5, rtol=0, atol=0)
    np.testing.assert_allclose(float(np.sum(p.edf)), 5.1465787370, rtol=0, atol=1e-4)


def test_fixed_sp_unknown_scale_matches_mgcv():
    """All-fixed sp + unknown scale: the criterion must be minimized over
    log φ (mgcv's 1-D newton when lsp = [log scale], gam.fit3.r:121-123),
    NOT evaluated at the Gaussian profile φ̂ = Dp/(n−Mp) — those coincide
    only for Gaussian/EQL-shaped ls. β̂/edf/Fletcher sig2 are
    φ-independent; only the reported criterion depends on this.
    """
    from hea.family import Gamma, InverseGaussian, Tweedie

    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x)", df, family=Tweedie(p=1.5), sp=np.array([2.0]), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 275.9692309030, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 1.1707800009, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.exp(m._log_phi_hat)), 1.6764407979, rtol=1e-6)
    m05 = gam(
        "ytw2 ~ s(x)", df, family=Tweedie(p=1.5), sp=np.array([0.5]), method="REML"
    )
    np.testing.assert_allclose(
        m05.REML_criterion / 2, 271.9594278050, rtol=0, atol=1e-6
    )
    ml = gam("ytw2 ~ s(x)", df, family=Tweedie(p=1.5), sp=np.array([2.0]), method="ML")
    np.testing.assert_allclose(ml.ML_criterion / 2, 274.2404901730, rtol=0, atol=1e-6)
    dfg = df.with_columns(yg=(pl.col("y") / 3).exp())
    g = gam(
        "yg ~ s(x) + s(z)",
        dfg,
        family=Gamma(link="log"),
        sp=np.array([1.0, 4.0]),
        method="REML",
    )
    np.testing.assert_allclose(g.REML_criterion / 2, -5.0421868495, rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.sigma_squared, 0.0355890744, rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(np.sum(g.edf)), 5.0315916630, rtol=0, atol=1e-4)
    i = gam(
        "yg ~ s(x)",
        dfg,
        family=InverseGaussian(link="log"),
        sp=np.array([3.0]),
        method="REML",
    )
    np.testing.assert_allclose(i.REML_criterion / 2, 8.6185097030, rtol=0, atol=1e-6)
    np.testing.assert_allclose(i.sigma_squared, 0.0467285182, rtol=0, atol=1e-9)


def test_extended_null_deviance_find_null_dev_matches_mgcv():
    """Extended families replace null.deviance with mgcv's
    ``find.null.dev`` (efam.r:98-117: 1-D optimize over the constant ON
    THE LINK SCALE, offset in the candidate model) via the family
    postproc (nb efam.r:283, tw efam.r:3239, scat efam.r:3742) — NOT the
    standard weighted-mean value. postproc also relabels summary's Family
    line with the fitted θ.
    """
    from hea.family import nb, scat, tw

    df = _mixed_sp_fixture().with_columns(off1=0.3 * pl.col("z"))
    ms = gam("y ~ s(x)", df, family=scat(), method="REML")
    np.testing.assert_allclose(ms.null_deviance, 409.4241525690, rtol=0, atol=1e-6)
    assert ms._family_display_name() == "Scaled t(19.281,0.515)"
    mso = gam("y ~ s(x) + offset(off1)", df, family=scat(), method="REML")
    np.testing.assert_allclose(mso.null_deviance, 407.5865103560, rtol=0, atol=1e-6)
    mt = gam("ytw2 ~ s(x)", df, family=tw(), method="REML")
    np.testing.assert_allclose(mt.null_deviance, 301.8879359400, rtol=0, atol=1e-6)
    assert mt._family_display_name() == "Tweedie(p=1.237)"
    mto = gam("ytw2 ~ s(x) + offset(off1)", df, family=tw(), method="REML")
    np.testing.assert_allclose(mto.null_deviance, 298.7091359900, rtol=0, atol=1e-6)
    m5 = gam("ycnt ~ s(x)", df, family=nb(theta=5), method="REML")
    np.testing.assert_allclose(m5.null_deviance, 254.1827688399, rtol=0, atol=1e-8)
    assert m5._family_display_name() == "Negative Binomial(5)"
    m5o = gam("ycnt ~ s(x) + offset(off1)", df, family=nb(theta=5), method="REML")
    np.testing.assert_allclose(m5o.null_deviance, 253.1033229196, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        m5o.REML_criterion / 2, 261.3070751610, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        m5o.deviance_explained,
        (m5o.null_deviance - m5o.deviance) / m5o.null_deviance,
        rtol=0,
        atol=1e-12,
    )


def test_nb_tw_sqrt_link_matches_mgcv():
    """sqrt is in nb/tw's okLinks but SqrtLink had no g2g/g3g/g4g — the
    extended dDeta chain raised NotImplementedError mid-fit. Forms from fix.family.link's extended block
    (gam.fit3.r:2243-2247).
    """
    from hea.family import nb, tw

    df = _mixed_sp_fixture()
    m = gam("ycnt ~ s(x)", df, family=nb(theta=5, link="sqrt"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 262.2684702080, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.deviance, 116.2285680340, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.4727045882, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.null_deviance, 254.1827689430, rtol=0, atol=1e-8)
    mt = gam("ytw2 ~ s(x)", df, family=tw(link="sqrt"), method="REML")
    np.testing.assert_allclose(mt.REML_criterion / 2, 252.9727727650, rtol=0, atol=1e-6)
    np.testing.assert_allclose(mt._tw_info["p_hat"], 1.23714366, rtol=0, atol=1e-5)


def test_binomial_factor_response_matches_mgcv():
    """R's gam accepts a 2-level factor (or logical) binomial response
    via binomial initialize's is.factor branch (level 1 = failure);
    hea routes gam/bam response intake through the same
    ``_coerce_response`` glm uses.
    """
    df = _mixed_sp_fixture()
    from hea.R.rng import RGenerator

    gen = RGenerator(11)
    p = 1.0 / (1.0 + np.exp(-(np.sin(2 * np.pi * df["x"].to_numpy()) * 1.5)))
    yb = gen.uniform(0, 1, len(p)) < p
    df = df.with_columns(ystr=pl.Series(np.where(yb, "yes", "no")), ybool=pl.Series(yb))
    from hea.family import Binomial

    m = gam("ystr ~ s(x)", df, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 89.7247633913, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.deviance, 164.6309203600, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.4854767035, rtol=0, atol=1e-4)
    mb = gam("ybool ~ s(x)", df, family=Binomial(), method="REML")
    _assert_fp_equiv(mb.REML_criterion, m.REML_criterion)
    _assert_fp_equiv(mb.coef, m.coef)
    with pytest.raises(Exception, match="convert|cast|float"):
        gam("ystr ~ s(x)", df, method="REML")


def test_outer_hessian_matches_mgcv_reml2():
    """The outer Newton Hessian vs mgcv's analytic REML2: hea's analytic
    (ρ, log φ) block agrees to ~1e-10 and the FD θ-rows (central
    differences of the analytic gradient, h=1e-4) to ~1e-7 — the FD
    truncation band. Layouts: hea (ρ, logφ, θ) ≡ mgcv (θ, ρ, logφ),
    both V_R units.
    """
    from hea.family import tw

    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x)", df, family=tw(), method="REML")
    H = np.asarray(m._outer_info["hess"])
    perm = [2, 0, 1]  # hea (ρ,logφ,θ) → mgcv (θ,ρ,logφ)
    Hm = H[np.ix_(perm, perm)]
    R_hess = np.array(
        [
            [31.949749924, 0.011231728742, -33.078371913],
            [0.011231728742, 2.029401444600, -1.964717610290],
            [-33.078371913, -1.964717610290, 143.804523249],
        ]
    )
    np.testing.assert_allclose(Hm[0, :], R_hess[0, :], rtol=0, atol=5e-6)
    np.testing.assert_allclose(Hm[:, 0], R_hess[:, 0], rtol=0, atol=5e-6)
    np.testing.assert_allclose(Hm[1:, 1:], R_hess[1:, 1:], rtol=0, atol=1e-7)


def test_quasi_power_link_matches_r():
    """R's ``power(λ)`` link: ``g(μ) = μ^λ`` with
    R's exact factory semantics — λ ≤ 0 → log, λ = 1 → identity, link
    name "mu^round(λ,3)" — and fix.family.link's power d2link..d4link
    branch (gam.fit3.r:2329-2335). Object form only, like R (make.link
    accepts no "power(...)" string).
    """
    from hea.family import PowerLink, Quasi, power

    assert power(0).name == "log" and power(-1).name == "log"
    assert power(1).name == "identity"
    assert power(1 / 3).name == "mu^0.333"
    assert isinstance(power(0.5), PowerLink)
    df = _mixed_sp_fixture().with_columns(yg=(pl.col("y") / 3).exp())
    mq = glm("yg ~ x + z", df, family=Quasi(link=power(1 / 3), variance="mu"))
    np.testing.assert_allclose(
        mq._bhat_arr,
        [1.13069136366, -0.216498739042, 0.0362335015181],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(mq.deviance, 9.5538082355, rtol=0, atol=1e-9)
    np.testing.assert_allclose(mq.dispersion, 0.0632910836247, rtol=0, atol=1e-8)
    mg = gam(
        "yg ~ s(x) + z",
        df,
        family=Quasi(link=power(1 / 3), variance="mu"),
        method="REML",
    )
    np.testing.assert_allclose(
        mg.REML_criterion / 2, -166.3099041900, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(mg.deviance, 5.6758088545, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(mg.edf)), 7.7176279737, rtol=0, atol=1e-4)
    np.testing.assert_allclose(mg.sigma_squared, 0.0377580878, rtol=0, atol=1e-9)


def test_mixed_sp_validation():
    df = _mixed_sp_fixture()
    # mgcv's exact error for a wrong-length per-smooth sp (mgcv.r:1426).
    with pytest.raises(ValueError, match="incorrect number of smoothing"):
        gam("y ~ s(x, sp=c(1, 2)) + s(z)", df, method="REML")
    with pytest.raises(ValueError, match="sp must have length"):
        gam("y ~ s(x) + s(z)", df, sp=np.array([1.0]), method="REML")
    with pytest.raises(ValueError, match="must be numeric"):
        gam("y ~ s(x, sp='a') + s(z)", df, method="REML")
    m = gam("y ~ s(x, sp=2) + s(z, sp=0.5)", df, method="REML")
    f = gam("y ~ s(x) + s(z)", df, sp=np.array([2.0, 0.5]), method="REML")
    _assert_fp_equiv(m.REML_criterion, f.REML_criterion)
    np.testing.assert_array_equal(m.sp, f.sp)


def test_control_scale_est_matches_mgcv():
    # gam.control(scale.est=): "pearson" drops the Fletcher correction,
    # "deviance" uses dev/(n−trA) (gam.fit3.r:596-606). The fit itself
    # is untouched on this fixture (score.scale enters thresholds only)
    # — sig2 is the value-level difference.
    from hea.family import quasipoisson

    d, _, _ = _cbind_fixture()
    base = gam("succ ~ s(x)", d, family=quasipoisson, method="REML")
    p = gam(
        "succ ~ s(x)",
        d,
        family=quasipoisson,
        method="REML",
        control={"scale_est": "pearson"},
    )
    np.testing.assert_allclose(p.sigma_squared, 1.9466793481, rtol=0, atol=1e-8)
    np.testing.assert_allclose(p.REML_criterion / 2, 141.0246116800, rtol=0, atol=1e-6)
    dv = gam(
        "succ ~ s(x)",
        d,
        family=quasipoisson,
        method="REML",
        control={"scale_est": "deviance"},
    )
    np.testing.assert_allclose(dv.sigma_squared, 2.1506592518, rtol=0, atol=1e-8)
    assert base.sigma_squared not in (p.sigma_squared, dv.sigma_squared)


def test_control_newton_and_maxit_match_mgcv():
    df = _mixed_sp_fixture()
    m = gam(
        "y ~ s(x) + s(z)", df, method="REML", control={"newton": {"conv_tol": 1e-3}}
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 118.5012745590, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.01676484948, 0.06765075176], rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 11.5634548370, rtol=0, atol=1e-4)
    m2 = gam("y ~ s(x) + s(z)", df, method="REML", control={"maxit": 2})
    assert m2.REML_criterion != m.REML_criterion


def test_xt_max_knots_seed_matches_mgcv():
    from hea.R.rng import RGenerator

    gen = RGenerator(2024)
    n = 4000
    x = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x) + gen.normal(0, 0.4, n)
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x, k=20, xt=list(max.knots=1500, seed=2))", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 2026.8334280500, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.007336834857, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 16.9493402600, rtol=0, atol=1e-4)


def test_gam_control_validation():
    from hea.models.gam import gam_control

    df = _mixed_sp_fixture()
    with pytest.raises(ValueError, match="scale_est"):
        gam_control(scale_est="bogus")
    with pytest.raises(NotImplementedError, match="irls_reg"):
        gam_control(irls_reg=0.5)
    with pytest.raises(NotImplementedError, match="idLinksBases"):
        gam_control(idLinksBases=False)
    with pytest.raises(ValueError, match="newton control"):
        gam_control(newton={"bogus": 1})
    with pytest.raises(ValueError, match="nlm control"):
        gam_control(nlm={"bogus": 1})
    with pytest.raises(ValueError, match="optim control"):
        gam_control(optim={"bogus": 1})
    # mgcv gam.control nlm defaults (mgcv.r:2500-2517): ndigit from
    # epsilon, gradtol = 10*epsilon, stepmax 2, steptol 1e-4
    c = gam_control()
    assert c["nlm"] == {
        "ndigit": 7,
        "gradtol": 1e-6,
        "stepmax": 2.0,
        "steptol": 1e-4,
        "iterlim": 200,
        "check_analyticals": False,
    }
    assert c["optim"] == {"factr": 1e7}
    with pytest.raises(ValueError, match="epsilon"):
        gam_control(epsilon=0.0)
    with pytest.raises(ValueError, match="scale_est"):
        gam("y ~ s(x)", df, method="REML", control={"scale_est": "nope"})
    with pytest.raises(ValueError, match="unsupported xt entry"):
        gam("y ~ s(x, xt=list(shrink=0.5))", df, method="REML")
    m0 = gam("y ~ s(x) + s(z)", df, method="REML")
    m1 = gam("y ~ s(x) + s(z)", df, method="REML", control={})
    _assert_fp_equiv(m0.REML_criterion, m1.REML_criterion)
    _assert_fp_equiv(m0.coef, m1.coef)


def test_scale_fixed_gaussian_matches_mgcv():
    # scale=0.3 + REML: φ KNOWN — no log φ slot in the outer vector,
    # criterion at log(0.3), sig2 reported as 0.3, Vp = 0.3·A⁻¹,
    # z/Chi-sq summary flavor (scale.estimated FALSE), AIC's dev1 =
    # scale·Σwt (gam.fit3.r:848 first branch — NOT the gaussian dev
    # override).
    df = _mixed_sp_fixture()
    m = gam("y ~ s(x) + s(z)", df, method="REML", scale=0.3)
    np.testing.assert_allclose(m.REML_criterion / 2, 122.8873561270, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.02415170022, 0.1304568059], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 10.5075250592, rtol=0, atol=1e-4)
    assert m.sigma_squared == 0.3 and m.scale_estimated is False
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.2246565999, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.Vp[0, 0], 0.001875, rtol=1e-6)
    np.testing.assert_allclose(m.AIC, 285.9720683910, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.loglike, -131.7123409670, rtol=0, atol=1e-5)
    _label, edf, ref_df, stat, _p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose(
        [edf, ref_df, stat], [5.5231503566, 6.6485522295, 223.9957913612], rtol=1e-6
    )
    # GCV.Cp at fixed scale → UBRE at φ=0.3 (any family, mgcv.r:1956).
    u = gam("y ~ s(x) + s(z)", df, scale=0.3)
    np.testing.assert_allclose(u.GCV_score, -0.0661457173, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp, [0.05418197314, 0.09828929095], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(u.edf)), 9.9008106194, rtol=0, atol=1e-4)
    assert u.sigma_squared == 0.3
    np.testing.assert_allclose(u.AIC, 283.2263031720, rtol=0, atol=1e-5)


def test_scale_negative_forces_estimation_matches_mgcv():
    from hea.family import Poisson

    df = _mixed_sp_fixture()
    m = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), scale=-1)
    np.testing.assert_allclose(m.GCV_score, 1.0715298363, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.sp[0], 0.09747456578, rtol=1e-4)
    np.testing.assert_allclose(m.sp[1], 259468.0106, rtol=1e-2)  # boundary
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6204950639, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.9621181226, rtol=0, atol=1e-8)
    assert m.scale_estimated is True
    _label, edf, ref_df, stat, _p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose(
        [edf, ref_df, stat], [4.6204526515, 5.6432868481, 25.8921163282], rtol=1e-6
    )
    # Under (RE)ML, poisson/binomial scale= is silently 1 (mgcv.r:1947).
    p2 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), method="REML", scale=2)
    p0 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), method="REML")
    _assert_fp_equiv(p2.REML_criterion, p0.REML_criterion)
    assert p2.sigma_squared == 1.0 and p2.scale_estimated is False
    # Extended families: a user scale>0 fixes φ (mgcv.r:1948-1949). nb scale=2
    # matches mgcv exactly (sp/edf/theta/reml — verified vs R); here confirm tw
    # accepts scale=2 and fixes φ=2 rather than raising.
    from hea.family import tw

    mtw = gam("ytw2 ~ s(x)", df, family=tw(), method="REML", scale=2.0)
    assert float(mtw.scale) == 2.0


# ---------------------------------------------------------------------------
# C7: single-formula start= / etastart= / mustart= (gam.fit3.r:259-292).
# Start values steer the PIRLS path, not the optimum — R-verified: mgcv
# warm/perturbed starts land on the same fit to 1e-14.
# ---------------------------------------------------------------------------


def test_pirls_start_values_warm_and_invariant():
    from hea.family import Poisson

    df = _mixed_sp_fixture()
    m0 = gam("y ~ s(x) + s(z)", df, method="REML")
    m1 = gam("y ~ s(x) + s(z)", df, method="REML", start=np.asarray(m0.coef))
    m2 = gam("y ~ s(x) + s(z)", df, method="REML", mustart=m0.fitted_values)
    m3 = gam("y ~ s(x) + s(z)", df, method="REML", etastart=m0.linear_predictors)
    m4 = gam(
        "y ~ s(x) + s(z)", df, method="REML", start=np.asarray(m0.coef) + 5.0
    )  # perturbed, still valid
    for m in (m1, m2, m3, m4):
        np.testing.assert_allclose(
            m.REML_criterion, m0.REML_criterion, rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(np.asarray(m.coef), np.asarray(m0.coef), atol=1e-8)
    # Non-gaussian + GCV path too (poisson, etastart route). The
    # initial.spg seed sees the user start (mgcv.r:4591-4595), so the
    # optimizer path shifts within the stop band (~2.6e-7 here — the
    # second sp rides a flat boundary ridge, so the GCV optimum is
    # shallow and the perturbed start lands a touch further off).
    p0 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson())
    p1 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), etastart=p0.linear_predictors)
    np.testing.assert_allclose(p1.GCV_score, p0.GCV_score, rtol=0, atol=1e-6)


def test_pirls_start_values_validation():
    from hea.family import Gamma

    df = _mixed_sp_fixture()
    with pytest.raises(ValueError, match="Length of start"):
        gam("y ~ s(x) + s(z)", df, method="REML", start=np.zeros(3))
    with pytest.raises(ValueError, match="etastart must have length"):
        gam("y ~ s(x)", df, method="REML", etastart=np.zeros(3))
    with pytest.raises(ValueError, match="mustart must have length"):
        gam("y ~ s(x)", df, method="REML", mustart=np.zeros(3))
    # Unrecoverable starting values: R's intended refusal
    # (gam.fit3.r:292) — mgcv itself dies with an obscure "missing
    # value where TRUE/FALSE needed" there (verified live).
    d2 = df.with_columns((pl.col("ycnt") + 0.5).alias("ygam"))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="valid starting values"):
            gam(
                "ygam ~ s(x)",
                d2,
                family=Gamma(link="log"),
                method="REML",
                mustart=np.full(len(df), -5.0),
            )


def test_plain_quasi_identity_link_full_newton_matches_mgcv():
    # mgcv's canonical for plain quasi is "none" (fix.family.link,
    # gam.fit3.r:2322): the inner loop runs full Newton even at the
    # identity link. quasi(identity, V=mu) would take Fisher steps under
    # a link==default test — the pins differ visibly in the sp. Same-
    # optimum stopping noise leaves sp ~1.5e-6 relative off R here
    # (the criterion itself agrees to 7e-9).
    from hea.family import Quasi

    d, _, _ = _cbind_fixture()
    m = gam(
        "succ ~ s(x)", d, family=Quasi(link="identity", variance="mu"), method="REML"
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 137.1247120200, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01412671672, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.9415583200, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.9565270162, rtol=0, atol=1e-6)


def _scat_fixture():
    from hea.R.rng import RGenerator

    g = RGenerator(2)
    n = 200
    x = g.uniform(size=n)
    f = np.sin(2 * np.pi * x) + 0.5 * x
    y = f + 0.3 * g.standard_t(4, size=n)
    return pl.DataFrame({"x": x, "y": y})


def test_scat_through_gam_matches_mgcv():
    from hea.family import Scat

    df = _scat_fixture()
    m = gam("y ~ s(x)", df, family=Scat(), method="REML")
    np.testing.assert_allclose(
        m.REML_criterion / 2, 93.4279560970729, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(m.sp[0], 0.1550084027, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.8407668102, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.411261775854, rtol=0, atol=1e-7)
    nu, sig = m.family.get_theta(trans=True)
    np.testing.assert_allclose(nu, 4.23984596547, rtol=3e-6)
    np.testing.assert_allclose(sig, 0.281128469435, rtol=1e-6)
    np.testing.assert_allclose(m.deviance, 267.745438162, rtol=0, atol=3e-5)
    np.testing.assert_allclose(m.fitted_values[0], 1.14677329548, rtol=0, atol=3e-8)
    np.testing.assert_allclose(m.Vp[0, 0], 0.0005459972642, rtol=1e-6)
    np.testing.assert_allclose(m.edf2_total, 7.9703610095, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.AIC, 171.0639212244, rtol=0, atol=1e-6)


def test_scat_ml_through_gam_matches_mgcv():
    from hea.family import Scat

    df = _scat_fixture()
    m = gam("y ~ s(x)", df, family=Scat(), method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 90.4358656257, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1649328349, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.7844589623, rtol=0, atol=1e-4)
    nu, sig = m.family.get_theta(trans=True)
    np.testing.assert_allclose(nu, 4.2099380606, rtol=1e-6)
    np.testing.assert_allclose(sig, 0.2794428367, rtol=1e-6)


def test_scat_fixed_theta_fixed_sp_matches_mgcv():
    from hea.family import Scat

    df = _scat_fixture()
    fam = Scat(theta=(4.23984596546644, 0.281128469435471))
    assert fam.n_theta == 0
    m = gam("y ~ s(x)", df, family=fam, method="REML", sp=np.array([0.155008402666293]))
    np.testing.assert_allclose(
        m.REML_criterion / 2, 93.4279560970729, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(m.coef)[0], 0.411261775854293, rtol=0, atol=1e-8
    )


def test_extended_family_rejects_free_theta_with_fixed_sp():
    from hea.family import Scat

    df = _scat_fixture()
    with pytest.raises(ValueError, match="incompatible"):
        gam("y ~ s(x)", df, family=Scat(), method="REML", sp=np.array([0.1]))


def _nb_fixture():
    from hea.R.rng import RGenerator

    g = RGenerator(7)
    n = 200
    x = g.uniform(size=n)
    mu = np.exp(0.3 + np.sin(2 * np.pi * x))
    Th = 3.0
    lam = g.gamma(Th, scale=mu / Th)
    y = np.asarray(g.poisson(lam), dtype=float)
    return pl.DataFrame({"x": x, "y": y})


def test_nb_through_gam_matches_mgcv():
    from hea.family import nb

    df = _nb_fixture()
    m = gam("y ~ s(x)", df, family=nb(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 318.541902729, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0774062275045, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.33449838393, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.227494645531, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        float(m.family.get_theta(trans=True)[0]), 2.65778771157, rtol=1e-6
    )
    np.testing.assert_allclose(m.deviance, 203.567374398, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.null_deviance, 312.800417888, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.fitted_values[0], 0.656085286284, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.edf2_total, 6.32010404395, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.AIC, 633.965932303, rtol=0, atol=1e-6)


def test_nb_fixed_theta_matches_mgcv():
    from hea.family import nb

    df = _nb_fixture()
    fam = nb(theta=3.0)
    assert fam.n_theta == 0
    m = gam("y ~ s(x)", df, family=fam, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 318.617850697, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0685065177646, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.49625628987, rtol=0, atol=1e-4)


def _sl_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(5)
    n = 120
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    g = gen.mt.sample_int(6, n, replace=True)  # {0..5}
    y = np.sin(2 * np.pi * x) + 0.3 * z + gen.normal(0, 0.3, n)
    return pl.DataFrame(
        {
            "x": x,
            "z": z,
            "g": pl.Series(g.astype(str)).cast(pl.Categorical),
            "y": y,
        }
    )


def test_sl_setup_ldet_s_match_mgcv():
    from hea.models.gam import _ldet_s, _sl_setup

    df = _sl_fixture()
    m = gam("y ~ te(x, z, k=5)", df, method="REML")
    sl = _sl_setup(m._slots, m.p)
    ld = _ldet_s(sl, np.array([-1.0, 1.5]), root=True, stot=True, deriv=2)
    np.testing.assert_allclose(ld["ldetS"], 16.9871017523, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        ld["ldet1"], [7.6597767860, 13.3402232140], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(sl.lam0, [0.1721080859, 0.1719973865], rtol=1e-9)
    np.testing.assert_allclose(
        float(np.linalg.norm(m._slots[0].S)), 10.0618085492, rtol=1e-9
    )

    m2 = gam("y ~ t2(x, z, k=4)", df, method="REML")
    sl2 = _sl_setup(m2._slots, m2.p)
    assert [(b.start, b.stop, b.n_sp, b.rank) for b in sl2.blocks] == [
        (1, 5, 1, 4),
        (5, 9, 1, 4),
        (9, 13, 1, 4),
    ]
    ld2 = _ldet_s(sl2, np.array([0.2, -0.7, 1.1]), deriv=1)
    np.testing.assert_allclose(ld2["ldetS"], 2.4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ld2["ldet1"], [4.0, 4.0, 4.0], rtol=0, atol=1e-12)


def test_sl_mixed_model_matches_mgcv_geometry():
    # s(x) + te(x,z) + s(g, bs="re"): gam.side rotates the te penalties
    # into a different (equally valid) basis than mgcv's, so raw block
    # log-dets shift by a transform constant that cancels in the REML
    # criterion (fast-REML.r:294-296's own design). Pin the basis-
    # invariant quantities: block geometry/ranks, ldet1, ldet2, E rows.
    from hea.models.gam import _ldet_s, _sl_setup

    df = _sl_fixture()
    m = gam('y ~ s(x) + te(x, z, k=5) + s(g, bs="re")', df, method="REML")
    sl = _sl_setup(m._slots, m.p)
    assert [(b.start, b.stop, b.n_sp, b.rank) for b in sl.blocks] == [
        (1, 10, 1, 8),
        (10, 33, 2, 21),
        (33, 39, 1, 6),
    ]
    ld = _ldet_s(sl, np.array([0.5, -1.0, 1.5, 0.3]), root=True, stot=True, deriv=2)
    np.testing.assert_allclose(
        ld["ldet1"], [8.0, 7.6597767860, 13.3402232140, 6.0], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        np.diag(ld["ldet2"]), [0.0, 0.8960604735, 0.8960604735, 0.0], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(ld["ldet2"][1, 2], -0.8960604735, rtol=0, atol=1e-8)
    assert ld["E"].shape[0] == 35


def test_sl_machinery_invariants():
    from hea.models.gam import (
        _ldet_s,
        _sl_initial_repara,
        _sl_mult,
        _sl_repara,
        _sl_setup,
        _sl_term_mult,
    )

    df = _sl_fixture()
    m = gam('y ~ s(x) + te(x, z, k=5) + s(g, bs="re")', df, method="REML")
    sl = _sl_setup(m._slots, m.p)
    rho = np.array([0.5, -1.0, 1.5, 0.3])
    ld = _ldet_s(sl, rho, root=True, stot=True, deriv=2)

    np.testing.assert_allclose(ld["E"].T @ ld["E"], ld["S"], atol=1e-10)
    w = np.linalg.eigvalsh(0.5 * (ld["S"] + ld["S"].T))
    r_tot = sum(b.rank for b in sl.blocks)
    ld_dense = float(np.sum(np.log(np.sort(w)[::-1][:r_tot])))
    np.testing.assert_allclose(ld["ldetS"], ld_dense, rtol=0, atol=1e-8)

    h = 1e-6
    n_sp = rho.size
    fd1 = np.zeros(n_sp)
    fdH = np.zeros((n_sp, n_sp))
    for k in range(n_sp):
        rp_ = rho.copy()
        rp_[k] += h
        rm_ = rho.copy()
        rm_[k] -= h
        fd1[k] = (
            _ldet_s(_sl_setup(m._slots, m.p), rp_, deriv=0)["ldetS"]
            - _ldet_s(_sl_setup(m._slots, m.p), rm_, deriv=0)["ldetS"]
        ) / (2 * h)
        gp = _ldet_s(_sl_setup(m._slots, m.p), rp_, deriv=1)["ldet1"]
        gm = _ldet_s(_sl_setup(m._slots, m.p), rm_, deriv=1)["ldet1"]
        fdH[:, k] = (gp - gm) / (2 * h)
    np.testing.assert_allclose(ld["ldet1"], fd1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ld["ldet2"], fdH, rtol=0, atol=1e-6)

    rng = np.random.default_rng(11)
    A = rng.normal(size=(m.p, 3))
    SA, _inds = _sl_term_mult(sl, A, full=True)
    np.testing.assert_allclose(_sl_mult(sl, A), sum(SA), atol=1e-12)
    np.testing.assert_allclose(_sl_mult(sl, A), ld["S"] @ A, atol=1e-10)
    for k in range(n_sp):
        np.testing.assert_allclose(_sl_mult(sl, A, k=k), SA[k], atol=1e-12)
    SA_s, inds_s = _sl_term_mult(sl, A, full=False)
    for k in range(n_sp):
        np.testing.assert_allclose(SA_s[k], SA[k][inds_s[k]], atol=1e-12)

    beta = rng.normal(size=m.p)
    X = m._X_full
    Xr = _sl_initial_repara(sl, X, both_sides=False)
    br = _sl_initial_repara(sl, beta, both_sides=False)
    np.testing.assert_allclose(X @ beta, Xr @ br, atol=1e-8)
    Xrr = _sl_repara(ld["rp"], Xr)
    brr = _sl_repara(ld["rp"], br)
    np.testing.assert_allclose(Xr @ br, Xrr @ brr, atol=1e-8)
    np.testing.assert_allclose(_sl_repara(ld["rp"], brr, inverse=True), br, atol=1e-10)
    b_back = _sl_initial_repara(sl, br, inverse=True)
    np.testing.assert_allclose(b_back, beta, atol=1e-8)


def _mf_fixture():
    from hea.R.rng import RGenerator

    g = RGenerator(31)
    n = 150
    x = g.uniform(size=n)
    z = g.uniform(size=n)
    w = g.normal(0.0, 1.0, n)
    y = (
        np.sin(2 * np.pi * x)
        + 0.4 * w
        + g.normal(0.0, np.exp(0.3 * np.cos(2 * np.pi * z)), n)
    )
    return pl.DataFrame({"x": x, "z": z, "w": w, "y": y})


def test_multi_formula_design_matches_mgcv():
    from hea.models.gam import _prepare_multi_design

    df = _mf_fixture()
    md = _prepare_multi_design(["y ~ s(x) + w", "~ s(z)"], df)
    assert md.p == 21 and md.n_lp == 2
    assert md.nsdf == [2, 1]
    assert md.pstart == [0, 11]  # mgcv 1-based (1, 12)
    np.testing.assert_array_equal(md.lpi[0], np.arange(0, 11))
    np.testing.assert_array_equal(md.lpi[1], np.arange(11, 21))
    assert [(s.col_start, s.col_end) for s in md.slots] == [(2, 11), (12, 21)]
    assert all(s.S.shape == (9, 9) for s in md.slots)
    assert md.column_names[:3] == ["(Intercept)", "w", "s(x).1"]
    assert md.column_names[11:14] == ["(Intercept).1", "s.1(z).1", "s.1(z).2"]
    assert md.blocks[0].label == "s(x)"
    assert md.blocks[1].label == "s.1(z)"
    np.testing.assert_allclose(
        float(np.abs(md.X).sum()), 2023.2041595800, rtol=0, atol=1e-6
    )
    assert md.offsets == [None, None]
    assert md.L is None and md.n_work == 2


def test_multi_formula_lpmatrix_and_offsets():
    from hea.models.gam import _multi_lpmatrix, _prepare_multi_design

    df = _mf_fixture()
    md = _prepare_multi_design(["y ~ s(x) + w", "~ s(z)"], df)
    Xn, lpi = _multi_lpmatrix(md, df[:7])
    np.testing.assert_allclose(Xn, md.X[:7], atol=1e-12)
    assert [len(i) for i in lpi] == [11, 10]
    perm = df[::-1][:10]
    Xp, _ = _multi_lpmatrix(md, perm)
    np.testing.assert_allclose(Xp, md.X[::-1][:10], atol=1e-12)

    md2 = _prepare_multi_design(["y ~ s(x) + offset(w)", "~ s(z)"], df)
    np.testing.assert_allclose(md2.offsets[0], df["w"].to_numpy(), atol=0)
    assert md2.offsets[1] is None

    md3 = _prepare_multi_design(["y ~ s(x)", "~ s(z)", "~ w"], df)
    assert md3.n_lp == 3 and md3.p == md3.lpi[2][-1] + 1
    assert md3.nsdf == [1, 1, 2]


def test_multi_formula_validation_and_gam_guard():
    from hea.models.gam import _prepare_multi_design

    df = _mf_fixture()
    with pytest.raises(ValueError, match="at least 2"):
        _prepare_multi_design(["y ~ s(x)"], df)
    with pytest.raises(ValueError, match="response"):
        _prepare_multi_design(["~ s(x)", "~ s(z)"], df)
    with pytest.raises(NotImplementedError, match="shared-term"):
        _prepare_multi_design(["y ~ s(x)", "1 + 2 ~ s(z)"], df)
    with pytest.raises(NotImplementedError, match="general family"):
        gam(["y ~ s(x)", "~ s(z)"], df, method="REML")
    from hea.family import gaulss

    with pytest.raises(ValueError, match="list of formulas"):
        gam("y ~ s(x)", df, family=gaulss(), method="REML")
    with pytest.raises(ValueError, match="linear predictors"):
        gam(["y ~ s(x)", "~ s(z)", "~ x"], df, family=gaulss(), method="REML")


def test_gam_family_constructor_autocall():
    # mgcv accepts the family constructor — gam(family=gaulss) ≡
    # gam(family=gaulss()) via ``if (is.function(family)) family <-
    # family()`` (mgcv.r:2324). Instances pass through un-called.
    from hea.family import Gaussian, gaulss

    df = _mf_fixture()
    m = gam("y ~ s(x)", df, family=Gaussian, method="REML")
    assert isinstance(m.family, Gaussian)
    m2 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss, method="REML")
    assert isinstance(m2.family, gaulss)


# ---------------------------------------------------------------------------
# gam.vcomp rescale=TRUE default — mgcv 1.9-4 references.
# R fits read the identical data via full-precision CSV; pins are printed
# gam.vcomp() values. S.scale is recorded per penalty by _scale_penalty
# (mgcv smooth.r:3877-3884) and vcomp's default divides each sp by it
# (mgcv.r:4242-4290); rescale=False is the fitted-scaling flavor.
# (The weighted-tp case is pinned in test_weights_gaussian_reml_matches_mgcv;
# factor-only bs="re" has S.scale=1 — the Machines pins cover invariance.)
# ---------------------------------------------------------------------------


def _vcomp_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(2)  # R-native seed 7 saturates te(x1,x2)'s 2nd
    n = 240  # penalty (degenerate vcomp CI); 2 keeps both
    x0 = gen.uniform(0, 1, n)  # te components well-determined
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    fac = gen.mt.sample_int(3, n, replace=True)  # {0,1,2}
    g = gen.mt.sample_int(6, n, replace=True)  # {0..5}
    fg = np.array(["a", "b", "c"])[fac]
    gg = np.array([f"g{i}" for i in range(6)])[g]
    fb = np.where(
        fg == "a",
        np.sin(2 * np.pi * x0),
        np.where(fg == "b", np.cos(2 * np.pi * x0), x0**2 * 2.0),
    )
    y = (
        0.3
        + np.sin(2 * np.pi * x0)
        + (x1 * x2) ** 2 * 2.0
        + fb
        + 0.3 * g * x0
        + gen.normal(0, 0.4, n)
    )
    return pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "fac": fg, "g": gg, "y": y})


def test_vcomp_rescale_te_matches_mgcv():
    m = gam("y ~ s(x0) + te(x1, x2)", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(),
        [20.4386597, 0.1069975, 0.1079532, 0.7829882],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy(),
        [10.8712116, 0.03464567, 0.03010987, 0.7124269],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        vc["upper"].to_numpy(), [38.4261503, 0.3304445, 0.3870453, 0.8605381], rtol=1e-5
    )
    vc0 = m._compute_vcomp(rescale=False)
    np.testing.assert_allclose(
        vc0["std_dev"].to_numpy(),
        [4.3357825, 0.1986111, 0.1969816, 0.7829882],
        rtol=1e-5,
    )


def test_vcomp_rescale_id_linked_full_sp_matches_mgcv():
    m = gam("y ~ s(x0, by=fac, id=1)", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(), [22.0423458] * 3 + [0.6340746], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy(), [15.0467585] * 3 + [0.5775982], rtol=1e-6
    )
    np.testing.assert_allclose(
        vc["upper"].to_numpy(), [32.290344] * 3 + [0.6960731], rtol=1e-6
    )


def test_vcomp_rescale_select_null_penalty_scale_one():
    # select=TRUE appends the null-space penalty Sf with mgcv S.scale=1
    # (smooth.r:4241/4259), so its row is rescale-invariant; the main
    # penalty's row rescales as usual. Wider tolerances: the select fit
    # stops on a flatter surface.
    m = gam("y ~ s(x0)", _vcomp_fixture(), method="REML", select=True)
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(), [21.4968621, 3.1636857, 0.8169613], rtol=2e-4
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy(), [11.4222009, 0.7325736, 0.7458715], rtol=2e-3
    )
    np.testing.assert_allclose(
        vc["upper"].to_numpy(), [40.4576216, 13.6626639, 0.8948269], rtol=2e-3
    )
    vc0 = m._compute_vcomp(rescale=False)
    assert vc0["std_dev"][1] == vc["std_dev"][1]
    assert vc0["lower"][1] == vc["lower"][1]


def test_vcomp_rescale_fs_consistency_and_mgcv():
    m = gam("y ~ s(x0, g, bs='fs')", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    vc0 = m._compute_vcomp(rescale=False)
    ss = np.array([s.S_scale for s in m._slots])
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy()[:3],
        vc0["std_dev"].to_numpy()[:3] * np.sqrt(ss),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy()[:3],
        vc0["lower"].to_numpy()[:3] * np.sqrt(ss),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        vc0["std_dev"].to_numpy(),
        [4.382783625, 4.599802487, 3.526655952, 0.742592],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        vc0["lower"].to_numpy(),
        [3.098904902, 2.292909040, 1.811794896, 0.6738163137],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        vc0["upper"].to_numpy(),
        [6.198574307, 9.227659076, 6.864630332, 0.8183875445],
        rtol=1e-5,
    )


def test_fs_smooth_fit_matches_mgcv():
    df = _vcomp_fixture()
    m = gam("y ~ s(x0, g, bs='fs')", df, method="REML")
    np.testing.assert_allclose(
        np.asarray(m.sp),
        [0.028707836994574434, 0.026062865631070086, 0.04433782097095746],
        rtol=1e-6,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 307.1290091916496, rtol=1e-10)
    np.testing.assert_allclose(m.scale, 0.551442899755062, rtol=1e-9)
    np.testing.assert_allclose(np.sum(m.edf), 30.6221719808027, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:5],
        [
            1.6004302655596474,
            0.3964249627940333,
            0.6310862234821446,
            2.106833901078844,
            1.471917571333663,
        ],
        atol=1e-8,
    )
    f1 = gam("y ~ s(x0, g, bs='fs')", df, method="REML", sp=[1.0, 2.0, 0.5])
    np.testing.assert_allclose(f1.REML_criterion / 2, 346.0376552301282, rtol=1e-10)
    f2 = gam("y ~ s(x0, g, bs='fs')", df, method="REML", sp=[1.0, 0.5, 2.0])
    np.testing.assert_allclose(f2.REML_criterion / 2, 345.2185763625583, rtol=1e-10)


# ---------------------------------------------------------------------------
# summary pTerms + predict(unconditional=) — mgcv 1.9-4.
# pTerms (mgcv.r:3928-3977): one joint Wald test per whole parametric term,
# assign-exact column grouping, pinv-rank df, Chi.sq (known scale, pchisq)
# vs F (estimated scale, pf on n−Σedf). Printed via anova() exactly like
# mgcv (print.anova.gam shows pTerms.table; print.summary.gam does not).
# References from gam()+summary()$pTerms.table / predict.gam on the
# CSV-identical fixture.
# ---------------------------------------------------------------------------


def _pterms_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(11)
    n = 200
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    f4 = np.array(["a", "b", "c", "d"])[gen.mt.sample_int(4, n, replace=True)]
    feff = {"a": 0.0, "b": 0.5, "c": -0.4, "d": 0.15}
    eta = 0.4 + np.vectorize(feff.get)(f4) + 0.6 * z + np.sin(2 * np.pi * x)
    ygau = eta + gen.normal(0, 0.35, n)
    ypois = gen.poisson(np.exp(eta)).astype(float)
    g5 = np.array([f"g{i}" for i in range(5)])[gen.mt.sample_int(5, n, replace=True)]
    lo = gen.uniform(0.5, 1.5, n)
    return pl.DataFrame(
        {"x": x, "z": z, "f4": f4, "g5": g5, "lo": lo, "ygau": ygau, "ypois": ypois}
    )


def test_pterms_gaussian_F_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    rows = m._pterms_rows()
    assert [(r[0], r[1]) for r in rows] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose(
        [r[2] for r in rows], [80.0581257013, 59.5820363825], rtol=1e-6
    )
    np.testing.assert_allclose(
        [r[3] for r in rows], [1.99582307745e-33, 6.69516763716e-13], rtol=1e-5
    )


def test_pterms_poisson_chisq_matches_mgcv():
    m = gam("ypois ~ f4 + z + s(x)", _pterms_fixture(), family=Poisson(), method="REML")
    rows = m._pterms_rows()
    assert [(r[0], r[1]) for r in rows] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose(
        [r[2] for r in rows], [88.4400894000, 20.4132375000], rtol=1e-6
    )
    np.testing.assert_allclose(
        [r[3] for r in rows], [4.73776106e-19, 6.239669318e-06], rtol=1e-5
    )


def test_pterms_dropped_term_is_nan_like_mgcv():
    df = _pterms_fixture().with_columns(pl.col("z").alias("z2"))
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("ygau ~ f4 + z + z2 + s(x)", df, method="REML")
    rows = m._pterms_rows()
    assert [r[0] for r in rows] == ["f4", "z", "z2"]
    np.testing.assert_allclose(
        [rows[0][2], rows[1][2]], [80.0600926600, 59.5846767600], rtol=1e-6
    )
    assert rows[2][1] == 1
    assert np.isnan(rows[2][2]) and np.isnan(rows[2][3])


def test_pls_rank_drop_alias_twin_canonical_on_any_blas():
    from hea.models.gam import _pls_rank_drop

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    X[:, 3] = X[:, 1]  # exact duplicate
    rank, drop, _ = _pls_rank_drop(X, [], 6)
    assert rank == 5 and list(drop) == [3]
    X[:, 3] = -X[:, 1]  # exact negated alias
    rank, drop, _ = _pls_rank_drop(X, [], 6)
    assert rank == 5 and list(drop) == [3]
    X[:, 5] = X[:, 1]  # three-way: keep first only
    rank, drop, _ = _pls_rank_drop(X, [], 6)
    assert rank == 4 and list(drop) == [3, 5]


def _fit5_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(3)
    n = 220
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    w = gen.uniform(0, 1, n)
    mu = 0.4 + np.sin(2 * np.pi * x) + 0.5 * w
    sd = np.exp(-0.6 + 0.8 * np.cos(2 * np.pi * z))
    y = mu + gen.normal(0, 1, n) * sd
    return pl.DataFrame({"x": x, "z": z, "w": w, "y": y})


def _fit5_run(formulas, lsp, deriv=2):
    from hea.family import gaulss
    from hea.models.gam import (
        _gam_fit5,
        _prepare_multi_design,
        _sl_initial_repara,
        _sl_setup,
        _sym_rank,
    )

    md = _prepare_multi_design(formulas, _fit5_fixture())
    sl = _sl_setup(md.slots, md.p)
    X = _sl_initial_repara(sl, md.X, both_sides=False)
    Mp = sum(md.nsdf)
    for b, (a, bc) in zip(md.blocks, md.block_col_ranges):
        k = bc - a
        if not b.S:
            Mp += k
            continue
        Mp += k - _sym_rank(np.sum([np.asarray(s, dtype=float) for s in b.S], axis=0))
    fit = _gam_fit5(
        X,
        md.y,
        np.asarray(lsp, dtype=float),
        sl,
        family=gaulss(),
        lpi=md.lpi,
        offset=md.offsets,
        Mp=Mp,
        deriv=deriv,
    )
    return fit, Mp


def test_gam_fit5_two_sp_matches_mgcv():
    fit, Mp = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3])
    assert Mp == 5 and fit["rank"] == 21 and fit["converged"]
    np.testing.assert_allclose(fit["REML"], 231.7393228959, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fit["REML1"], [13.2496487044, 1.5371596913], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        fit["REML2"],
        [[5.44755992, -0.59313768], [-0.59313768, 2.22146868]],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(fit["l"], -197.5578307545, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fit["fitted_values"][:2],
        [[1.46550206, 2.11966446], [-0.10735620, 1.69868960]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        fit["dVkk"],
        [[9.57215971, 0.68222719], [0.68222719, 1.17817349]],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.abs(fit["db_drho"]).sum(), 6.44240433, rtol=0, atol=1e-6
    )


def test_gam_fit5_three_sp_matches_mgcv():
    fit, Mp = _fit5_run(["y ~ s(x) + s(w)", "~ s(z)"], [0.5, -0.3, 1.2])
    assert Mp == 5 and fit["rank"] == 29
    np.testing.assert_allclose(fit["REML"], 242.0725467689, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fit["REML1"], [11.6813000034, -1.4120335666, 6.7397195283], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(fit["REML2"]).ravel(),
        [
            3.80412180,
            0.12417761,
            -1.13087665,
            0.12417761,
            0.45682933,
            0.06802479,
            -1.13087665,
            0.06802479,
            4.91619318,
        ],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(fit["l"], -201.3004207932, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fit["fitted_values"][:2],
        [[1.48531489, 2.07624430], [-0.07929165, 1.74268188]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.diag(fit["dVkk"]), [9.57121693, 0.16550807, 3.20053036], rtol=0, atol=1e-6
    )


def test_gam_fit5_deriv1_trace_path_matches_deriv2():
    # deriv=1 assembles d1ldetH from the gamlss_gH TRACE-vector form
    # (fh = Hp⁻¹); deriv=2 from the full ∂H/∂ρ list. Same REML1 either
    # way (gam.fit4.r:1347-1365).
    f2, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3], deriv=2)
    f1, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3], deriv=1)
    np.testing.assert_allclose(f1["REML1"], f2["REML1"], rtol=0, atol=1e-9)
    assert f1["REML2"] is None


def test_gam_fit5_reml1_matches_finite_differences():
    base = np.array([0.5, -0.3])
    fit, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], base)
    eps = 1e-5
    g_fd = np.zeros(2)
    for k in range(2):
        lp, lm = base.copy(), base.copy()
        lp[k] += eps
        lm[k] -= eps
        fp, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], lp, deriv=0)
        fm, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], lm, deriv=0)
        g_fd[k] = (fp["REML"] - fm["REML"]) / (2 * eps)
    np.testing.assert_allclose(fit["REML1"], g_fd, rtol=0, atol=1e-6)


def test_gaulss_free_fit_through_gam_matches_mgcv():
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    assert m.converged and m.method == "REML"
    np.testing.assert_allclose(m.sp, [0.14419871, 0.22829985], rtol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2, 216.8833933770, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46270992, 2.16447805], [-0.15903017, 1.63246911]],
        rtol=0,
        atol=1e-5,
    )
    # deviance = Σ deviance-residuals² (mgcv.r:2429); null deviance
    # from gaulss's postproc (gamlss.r:910-918).
    np.testing.assert_allclose(m.deviance, 219.85105997, rtol=1e-5)
    np.testing.assert_allclose(m.null_deviance, 871.08214845, rtol=1e-5)
    assert m.rank == 21
    # GCV.Cp silently coerces to REML for general families
    # (mgcv.r:1894-1898).
    m3 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="GCV.Cp")
    assert m3.method == "REML"
    np.testing.assert_allclose(m3.REML_criterion / 2, 216.88339338, rtol=0, atol=1e-5)


def test_gaulss_post_proc_surface_matches_mgcv():
    # gam.fit5.post.proc (gam.fit4.r:1571-1719): Vp/Vc/Ve/edf/edf1/edf2
    # with both reparameterizations undone; AIC/logLik (m$aic = −2l +
    # 2Σedf, df = Σedf2 capped at #coef — logLik.gam); gam.vcomp (the
    # slice-(i) S.scale rescale rides through the multi-formula slots;
    # hea appends its conventional scale≡1 row, mgcv's gaulss table has
    # no scale row); sp.vcov = solve(hess + reg) with mgcv's literal
    # elementwise regularizer.
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    np.testing.assert_allclose(m.edf_total, 13.97060461, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        m.edf[:4], [1.0, 1.0, 0.99400183, 1.08473432], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.diag(m.Vp)[:4],
        [0.0031766700, 0.0097069400, 0.0579606100, 0.4328408500],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.diag(m.Vc)[:4],
        [0.0031954300, 0.0097356600, 0.0642987400, 0.4543290500],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.diag(m.Ve)[:2], [0.0031480100, 0.0096115700], rtol=1e-6
    )
    np.testing.assert_allclose(m.edf1_total, 16.10915241, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.edf2_total, 14.69865170, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.AIC, 406.51598096, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.loglike, -188.55933878, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.npar, 14.698652, rtol=0, atol=1e-4)
    vc = m.vcomp
    assert vc["name"].to_list() == ["s(x)", "s.1(z)", "scale"]
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy()[:2], [11.716203, 9.831598], rtol=1e-5
    )
    np.testing.assert_allclose(
        vc["lower"].to_numpy()[:2], [6.4788824, 4.0192677], rtol=1e-5
    )
    np.testing.assert_allclose(
        vc["upper"].to_numpy()[:2], [21.187206, 24.049236], rtol=1e-5
    )
    np.testing.assert_allclose(
        m.sp_vcov(), [[0.36531150, 0.00909964], [0.00909964, 0.83244672]], rtol=1e-4
    )
    m2 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        sp=np.array([2.0, 0.5]),
    )
    assert m2.sp_vcov() is None
    np.testing.assert_allclose(m2.Vc, m2.Vp, rtol=0, atol=0)


def test_gaulss_predict_and_summary_surface_matches_mgcv():
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    p = m.predict(df[:3], type="link", se_fit=True)
    np.testing.assert_allclose(
        np.c_[p["fit"], p["fit.1"]],
        [
            [1.46270992, -0.79406171],
            [-0.15903017, -0.50655307],
            [1.55899517, -1.10404635],
        ],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.c_[p["se.fit"], p["se.fit.1"]],
        [[0.06626794, 0.10166637], [0.06894145, 0.10901192], [0.08771956, 0.11470687]],
        rtol=0,
        atol=1e-6,
    )
    pr = m.predict(df[:3], type="response")
    np.testing.assert_allclose(
        np.c_[pr["fit"], pr["fit.1"]],
        [[1.46270992, 2.16447805], [-0.15903017, 1.63246911], [1.55899517, 2.92802712]],
        rtol=0,
        atol=1e-5,
    )
    pu = m.predict(df[:3], type="link", se_fit=True, unconditional=True)
    np.testing.assert_allclose(
        np.c_[pu["se.fit"], pu["se.fit.1"]],
        [[0.06677206, 0.10759165], [0.06943394, 0.11604036], [0.08817035, 0.11760738]],
        rtol=0,
        atol=1e-6,
    )
    Xl = m.predict(df[:3], type="lpmatrix")
    assert Xl.shape == (3, 21)
    np.testing.assert_allclose(np.abs(Xl).sum(), 34.86671860, rtol=1e-7)
    rows = m._smooth_significance_rows()
    assert [r[0] for r in rows] == ["s(x)", "s.1(z)"]
    np.testing.assert_allclose(
        [(r[1], r[2], r[3]) for r in rows],
        [(6.365175, 7.510126, 588.421362), (4.605429, 5.599026, 144.651351)],
        rtol=1e-4,
    )
    pt = m._pterms_rows()
    assert [(r[0], r[1]) for r in pt] == [("w", 1)]
    np.testing.assert_allclose(pt[0][2], 4.650478**2, rtol=1e-4)
    par = dict(zip(m.parametric_columns, np.asarray(m._beta_report)[m._param_idx]))
    np.testing.assert_allclose(par["(Intercept).1"], -0.58269247, rtol=0, atol=1e-6)
    m.summary()  # prints the mgcv-layout summary without error


def test_fit5_fully_penalized_summary_matches_mgcv():
    # Fully-penalized smooths (zero penalty null space after the
    # centering constraint: cc here, re below) route summary through
    # reTest → recov (mgcv.r:3599), which consumes the model R factor
    # ``b$R`` verbatim — for general families that is
    # gam.fit5.post.proc's root with R'R = −lbb, not the PIRLS
    # √W·X factor, which fit5 never stores — reading it unguarded crashes
    # summary() on any general fit with a cc/cp/re smooth. One fixture drives
    # all three recov consumptions:
    #   m1 s(x,cc)        — reTest, no random siblings (LRB branch);
    #   m2 s(x,cc)        — reTest conditioning on s(g,re) as random
    #                       (the R1/R2 split + L-inflation branch);
    #   m2 s(g,re)        — reTest on the re term itself;
    #   m2 s(v) (tp)      — testStat through the same _R_fit5.
    # Both engines run at tightened convergence (conv_tol=1e-11,
    # epsilon=1e-10) so early stops don't masquerade as disagreement.
    # On this R-native seed-31 data s(v) fully SATURATES (sp→∞, edf→1):
    # its λ rides a flat ridge where hea (3.69e10) and mgcv (3.43e10) stop
    # at different huge values, yet the s.table row (edf 1, Chi.sq 10.54)
    # is identical — so s(v)'s λ is only asserted >1e9 while s(x,cc)/s(g,re)
    # λ and the whole s.table pin tight. One uniform 5e-5 class — the circlss
    # test-pnlss-parity.R convention, widened from its 2e-6 because this
    # basin is far flatter than any there.
    # R: gam(list(y ~ ..., ~ 1), family=gaulss(), method="REML",
    # knots=list(x=c(0, 2*pi)), control=gam.control(epsilon=1e-10,
    # newton=list(conv.tol=1e-11))) on the same %.17g CSV; pins from
    # summary(b)$s.table at digits=12.
    from hea.family import gaulss
    from hea.R.rng import RGenerator

    gen = RGenerator(31)
    n = 200
    x = gen.uniform(0, 2 * np.pi, n)
    v = gen.uniform(0, 1, n)
    g = gen.mt.sample_int(8, n, replace=True)  # {0..7}
    b_g = gen.normal(0, 0.15, 8)
    y = 0.2 * np.sin(x) + 0.15 * np.cos(np.pi * v) + b_g[g] + gen.normal(0, 0.4, n)
    df = pl.DataFrame(
        {
            "x": x,
            "v": v,
            "g": pl.Series(g.astype(str)).cast(pl.Categorical),
            "y": y,
        }
    )
    kn = {"x": [0.0, 2 * np.pi]}
    ctl = {"epsilon": 1e-10, "newton": {"conv_tol": 1e-11}}
    TOL = 5e-5  # one class, ~14x the worst cross-BLAS floor (s(v) λ)

    m1 = gam(
        ['y ~ s(x, bs="cc")', "~ 1"],
        df,
        family=gaulss(),
        method="REML",
        knots=kn,
        control=ctl,
    )
    assert m1.converged
    np.testing.assert_allclose(
        m1.REML_criterion / 2, 131.957334123150, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(m1.sp, [2657.482428], rtol=TOL)
    ((label, edf, ref_df, stat, p_val),) = m1._smooth_significance_rows()
    assert label == "s(x)"
    np.testing.assert_allclose(
        [edf, ref_df, stat],
        [1.82293088063, 8.0, 6.05893727191],
        rtol=TOL,
        err_msg="m1 s(x,cc) row vs mgcv s.table",
    )
    np.testing.assert_allclose(p_val, 0.020872582751, rtol=TOL)
    m1.summary()  # the original crash site

    m2 = gam(
        ['y ~ s(x, bs="cc") + s(v) + s(g, bs="re")', "~ 1"],
        df,
        family=gaulss(),
        method="REML",
        knots=kn,
        control=ctl,
    )
    assert m2.converged
    np.testing.assert_allclose(
        m2.REML_criterion / 2, 124.466904719375, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(m2.sp[[0, 2]], [1471.3655, 42.497327], rtol=TOL)
    assert m2.sp[1] > 1e9
    rows = m2._smooth_significance_rows()
    assert [r[0] for r in rows] == ["s(x)", "s(v)", "s(g)"]
    np.testing.assert_allclose(
        [r[1:4] for r in rows],
        [
            (2.37312711793, 8.0, 19.4418827861),
            (1.00000000225, 1.00000000447, 10.5418947002),
            (5.37893803830, 7.0, 22.9558444790),
        ],
        rtol=TOL,
        err_msg="m2 rows vs mgcv s.table",
    )
    np.testing.assert_allclose(
        [r[4] for r in rows],
        [0.000165159607, 0.001167148839, 0.000089947459],
        rtol=TOL,
        err_msg="m2 p-values vs mgcv s.table",
    )
    m2.summary()


def test_gaulss_efs_optimizer_matches_mgcv():
    # available_derivs == 0 → the automatic extended-Fellner-Schall
    # outer loop (efsud, gam.fit4.r:1479-1569; mgcv.r:1907-1908's
    # optimizer switch): every fit at deriv=0, the family's ll needed
    # only to deriv 1 — the derivative-light custom-family on-ramp.
    # R: gam(list(...), family=gaulss(), optimizer="efs"). EFS's own
    # stop rules are loose by design (efs.tol = 0.1 on the REML band),
    # so tolerances here are wider than the newton pins.
    from hea.family import gaulss

    class _gaulss_efs(gaulss):
        available_derivs = 0

    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_efs(), method="REML")
    assert 3 <= m.outer_info["iter"] <= 8  # R: 5
    np.testing.assert_allclose(m.sp, [0.13952407, 0.17902162], rtol=1e-3)
    np.testing.assert_allclose(m.REML_criterion / 2, 216.9188691300, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.edf_total, 14.27664854, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46232526, 2.14669899], [-0.15974078, 1.61848595]],
        rtol=0,
        atol=1e-4,
    )
    assert m.sp_vcov() is None
    np.testing.assert_array_equal(m.Vc, m.Vp)


def test_gaulss_start_warm_restart():
    # start= (mgcv.r:1903): model-space coefficients enter the fitting
    # basis via the forward initial repara; a warm restart lands on
    # the same optimum. (The single-formula path takes start= too since
    # C7 — its wrong-length error is mgcv's gam.fit3 message; see
    # test_pirls_start_values_*.)
    from hea.family import gaulss

    df = _fit5_fixture()
    m0 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    m1 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        start=np.asarray(m0._beta),
    )
    np.testing.assert_allclose(m1.REML_criterion, m0.REML_criterion, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m1.sp, m0.sp, rtol=1e-5)
    with pytest.raises(ValueError, match="Length of start"):
        gam("y ~ s(x)", df, method="REML", start=np.zeros(11))
    with pytest.raises(NotImplementedError, match="etastart"):
        gam(
            ["y ~ s(x) + w", "~ s(z)"],
            df,
            family=gaulss(),
            method="REML",
            etastart=np.zeros(len(df)),
        )


def test_gaulss_fixed_sp_through_gam_matches_mgcv():
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        sp=np.array([2.0, 0.5]),
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 233.9898943665, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46246549, 2.08069657], [-0.09947020, 1.67028576]],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(m.sp, [2.0, 0.5])


def test_multilp_parametric_collinearity_matches_mgcv_olid():
    # mgcv's `olid` (mgcv.r:863-919, called from gam.setup.list) drops
    # unidentifiable UNPENALIZED columns across a multi-formula design,
    # then zero-pads coef on output. Its only trigger reachable through
    # hea's front end is within-formula parametric collinearity (the
    # shared-block trigger needs the '1+2~s(x)' common-term syntax, which
    # hea rejects with NotImplementedError); hea handles the reachable
    # case through the structural rank-drop (pivoted-QR detect + drop +
    # zero-fill — the same surgery mgcv's pls_fit1/gdi.c does at fit
    # level). Receipt vs mgcv 1.9-4 on gaulss with xdup = 2·x in LP1:
    # same coef count, xdup coefficient EXACTLY 0, REML to all printed
    # digits (311.964723261).
    from hea.family import gaulss
    from hea.R.rng import RGenerator

    g = RGenerator(3)
    n = 200
    x = g.uniform(0.0, 1.0, n)
    z = g.uniform(0.0, 1.0, n)
    w = g.uniform(0.0, 1.0, n)
    rn = g.normal(0.0, 1.0, n)
    df = pl.DataFrame(
        {
            "y": 1
            + x
            + np.sin(2 * np.pi * z)
            + rn * np.exp(0.3 * np.cos(2 * np.pi * w)),
            "x": x,
            "xdup": x * 2.0,
            "z": z,
            "w": w,
        }
    )
    m = gam(["y ~ x + xdup + s(z)", "~ s(w)"], df, family=gaulss(), method="REML")
    beta = np.asarray(m._beta)
    assert beta.shape[0] == 22  # mgcv ncoef
    assert beta[2] == 0.0  # xdup dropped → exact 0
    np.testing.assert_allclose(
        beta[:2], [1.11538940632, 0.996736549025], rtol=0, atol=2e-6
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 311.964723261, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.edf_total, 11.6260398718, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.0681246465194, 0.431317034396], rtol=5e-5)


def test_general_family_derivs_dispatch_guards():
    # mgcv's optimizer dispatch on available.derivs (mgcv.r:1906-1908):
    # ==1 → c("outer","bfgs") — the BFGS outer optimizer (item 7): a
    # derivs-1 family fits through gam.fit5 at deriv ≤ 1 (score + grad),
    # never the deriv-2/trHid2H path Newton needs;
    # ==0 → every fit5 call stays at deriv 0 (ll deriv ≤ 1), the fixed-sp
    # and no-smooth paths included (mgcv fits derivs-0 families only through
    # efsudr's deriv=0 calls, gam.fit4.r:1479+).
    from hea.family import gaulss

    df = _fit5_fixture()

    class _gaulss_d1(gaulss):
        available_derivs = 1

        def ll(
            self,
            y,
            X,
            coef,
            wt,
            *,
            lpi,
            offset=None,
            deriv=0,
            d1b=None,
            d2b=None,
            fh=None,
            D=None,
        ):
            assert deriv <= 2, f"bfgs asked a derivs-1 family for ll(deriv={deriv})"
            return super().ll(
                y,
                X,
                coef,
                wt,
                lpi=lpi,
                offset=offset,
                deriv=deriv,
                d1b=d1b,
                d2b=d2b,
                fh=fh,
                D=D,
            )

    m_bfgs = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d1(), method="REML")
    m_newton = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    np.testing.assert_allclose(
        m_bfgs.REML_criterion, m_newton.REML_criterion, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(m_bfgs._beta), np.asarray(m_newton._beta), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(m_bfgs.sp, m_newton.sp, rtol=1e-3)
    assert m_bfgs.outer_info["conv"] == "full convergence"

    class _gaulss_d0(gaulss):
        available_derivs = 0

        def ll(
            self,
            y,
            X,
            coef,
            wt,
            *,
            lpi,
            offset=None,
            deriv=0,
            d1b=None,
            d2b=None,
            fh=None,
            D=None,
        ):
            assert deriv <= 1, f"derivs-0 family asked for ll(deriv={deriv})"
            return super().ll(
                y,
                X,
                coef,
                wt,
                lpi=lpi,
                offset=offset,
                deriv=deriv,
                d1b=d1b,
                d2b=d2b,
                fh=fh,
                D=D,
            )

    m2 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        sp=np.array([2.0, 0.5]),
    )
    m0 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=_gaulss_d0(),
        method="REML",
        sp=np.array([2.0, 0.5]),
    )
    np.testing.assert_allclose(m0.REML_criterion, m2.REML_criterion, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(m0._beta), np.asarray(m2._beta), rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(m0.Vp), np.asarray(m2.Vp), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(m0.Vc, m0.Vp)
    assert m0.sp_vcov() is None

    m0p = gam(["y ~ w", "~ 1"], df, family=_gaulss_d0(), method="REML")
    m2p = gam(["y ~ w", "~ 1"], df, family=gaulss(), method="REML")
    np.testing.assert_allclose(
        m0p.REML_criterion, m2p.REML_criterion, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(m0p._beta), np.asarray(m2p._beta), rtol=0, atol=1e-10
    )


def test_optimizer_knob_efs_and_validation():
    # gam(optimizer=) — mgcv's intake and dispatch: first element
    # "outer"|"efs" with estimate.gam's "unknown optimizer" error
    # (mgcv.r:1913), second element defaulting to "newton" with
    # gam.outer's "unknown outer optimization method." (mgcv.r:
    # 1643-1644), efs forcing method="REML" (mgcv.r:1914), and the
    # available.derivs==1 coercion skipped when efs is requested
    # (mgcv.r:1907). newton/efs/nlm/optim are ported (nlm/optim pins:
    # test_nlm_optim_* below); standard-family bfgs raises (C9).
    from hea.family import gaulss

    df = _fit5_fixture()

    class _gaulss_d0(gaulss):
        available_derivs = 0

    m_auto = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d0(), method="REML")
    m_knob = gam(
        ["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML", optimizer="efs"
    )
    _assert_fp_equiv(m_knob._beta, m_auto._beta)
    _assert_fp_equiv(m_knob.sp, m_auto.sp)
    assert m_knob.optimizer == ("efs", "newton")
    np.testing.assert_allclose(
        m_knob.REML_criterion / 2, 216.9188691300, rtol=0, atol=1e-3
    )
    assert m_knob.sp_vcov() is None  # deriv-0 fit: no outer hess

    # efs coerces the method like mgcv.r:1914 (the general path is
    # REML-coerced anyway, mgcv.r:1894 — the fit must be identical)
    m_gcv = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="GCV.Cp",
        optimizer="efs",
    )
    assert m_gcv.method == "REML"
    _assert_fp_equiv(m_gcv._beta, m_knob._beta)

    # derivs==1 + optimizer="efs" is legal (mgcv.r:1907 only coerces
    # to bfgs when efs was NOT requested); ll never asked past deriv 1
    class _gaulss_d1(gaulss):
        available_derivs = 1

        def ll(
            self,
            y,
            X,
            coef,
            wt,
            *,
            lpi,
            offset=None,
            deriv=0,
            d1b=None,
            d2b=None,
            fh=None,
            D=None,
        ):
            assert deriv <= 1, f"efs asked a derivs-1 family for ll(deriv={deriv})"
            return super().ll(
                y,
                X,
                coef,
                wt,
                lpi=lpi,
                offset=offset,
                deriv=deriv,
                d1b=d1b,
                d2b=d2b,
                fh=fh,
                D=D,
            )

    m_d1 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=_gaulss_d1(),
        method="REML",
        optimizer="efs",
    )
    _assert_fp_equiv(m_d1._beta, m_auto._beta)

    with pytest.raises(ValueError, match="unknown optimizer"):
        gam("y ~ s(x)", df, method="REML", optimizer="perf")
    with pytest.raises(ValueError, match="unknown outer optimization method"):
        gam("y ~ s(x)", df, method="REML", optimizer=("outer", "magic"))
    m_bfgs = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df,
        family=gaulss(),
        method="REML",
        optimizer=("outer", "bfgs"),
    )
    m_newton = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    np.testing.assert_allclose(
        m_bfgs.REML_criterion, m_newton.REML_criterion, rtol=0, atol=1e-6
    )
    assert m_bfgs.optimizer == ("outer", "bfgs")
    with pytest.raises(NotImplementedError, match="general families only"):
        gam("y ~ s(x)", df, method="REML", optimizer=("outer", "bfgs"))
    # single-formula efs → efsudr (gam.fit4.r:822): method coerced to
    # REML, and the Fellner-Schall optimum agrees with newton's to the
    # EFS stopping tolerance (exact mgcv pins live in
    # test_efsudr_*_matches_mgcv below).
    m_efs1 = gam("y ~ s(x)", df, method="GCV.Cp", optimizer="efs")
    assert m_efs1.optimizer == ("efs", "newton")
    assert m_efs1.method == "REML"
    m_n1 = gam("y ~ s(x)", df, method="REML")
    np.testing.assert_allclose(
        m_efs1.REML_criterion, m_n1.REML_criterion, rtol=0, atol=2e-2
    )

    m_def = gam("y ~ s(x)", df, method="REML")
    m_opt = gam("y ~ s(x)", df, method="REML", optimizer=("outer", "newton"))
    _assert_fp_equiv(m_opt._beta, m_def._beta)
    assert m_def.optimizer == ("outer", "newton")


def test_efs_maxit_control_caps_outer_loop():
    # hea-only knob: the EFS outer-loop cap (mgcv hard-codes
    # ``for (iter in 1:200)`` in efsud, gam.fit4.r:1493). Default is
    # 200 — identical to mgcv — so cross-engine parity is unchanged;
    # the knob exists only for hea-native fits of hard multi-LP
    # families that need >200 EFS steps to satisfy efs_tol.
    from hea.family import gaulss
    from hea.models.gam import gam_control

    df = _fit5_fixture()
    forms = ["y ~ s(x) + w", "~ s(z)"]

    m_def = gam(forms, df, family=gaulss(), method="REML", optimizer="efs")
    assert m_def.outer_info["conv"] == "full convergence"
    assert m_def.outer_info["iter"] == 5
    m_big = gam(
        forms,
        df,
        family=gaulss(),
        method="REML",
        optimizer="efs",
        control={"efs_maxit": 500},
    )
    _assert_fp_equiv(m_big._beta, m_def._beta)
    _assert_fp_equiv(m_big.sp, m_def.sp)
    assert m_big.outer_info["conv"] == "full convergence"
    assert m_big.outer_info["iter"] == 5

    m_cap = gam(
        forms,
        df,
        family=gaulss(),
        method="REML",
        optimizer="efs",
        control={"efs_maxit": 3},
    )
    assert m_cap.outer_info["conv"] == "iteration limit reached"
    assert m_cap.outer_info["iter"] == 3

    with pytest.raises(ValueError, match="efs_maxit"):
        gam_control(efs_maxit=0)
    with pytest.raises(ValueError, match="efs_maxit"):
        gam(
            forms,
            df,
            family=gaulss(),
            method="REML",
            optimizer="efs",
            control={"efs_maxit": -1},
        )


def test_efsudr_gaussian_poisson_matches_mgcv():
    # Single-formula optimizer="efs" → efsudr (gam.fit4.r:822-938), the
    # gam.fit3 (regular-family) branch. R refs (mgcv 1.9-4), data
    # reproduced bit-for-bit by hea.R.rng:
    #   set.seed(2); x<-runif(200); y<-sin(2*pi*x)+.5*x+rnorm(200,0,.3)
    #   gam(y~s(x), method="REML", optimizer="efs")
    #     sp 0.01527936133  Σedf 7.951206296  sig2 0.0950333223
    #     REML 62.74910762, 5 iters, score.hist 120.42429 65.804633
    #     62.75026 62.74912 62.749108 (full convergence)
    #   poisson (same x stream): y<-rpois(200, exp(0.6*sin(2*pi*x)))
    #     sp 0.199587407  Σedf 5.129674639  REML 269.6455238, 7 iters
    from hea.family import Poisson
    from hea.R.rng import RGenerator

    g = RGenerator(2)
    n = 200
    x = g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.5 * x + g.normal(0, 0.3, n)
    m = gam("y ~ s(x)", pl.DataFrame({"y": y, "x": x}), method="REML", optimizer="efs")
    np.testing.assert_allclose(m.sp[0], 0.01527936133, rtol=1e-8)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.951206296, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.0950333223, rtol=0, atol=1e-9)
    np.testing.assert_allclose(m.REML_criterion / 2, 62.74910762, rtol=0, atol=1e-6)
    info = m._outer_info
    assert info["conv"] == "full convergence" and info["iter"] == 5
    np.testing.assert_allclose(
        info["score_hist"],
        [120.42429, 65.804633, 62.75026, 62.74912, 62.749108],
        rtol=1e-6,
    )
    g = RGenerator(2)
    x = g.uniform(0, 1, n)
    yp = g.poisson(np.exp(0.6 * np.sin(2 * np.pi * x))).astype(float)
    mp = gam(
        "y ~ s(x)",
        pl.DataFrame({"y": yp, "x": x}),
        family=Poisson(),
        method="REML",
        optimizer="efs",
    )
    np.testing.assert_allclose(mp.sp[0], 0.199587407, rtol=1e-8)
    np.testing.assert_allclose(float(np.sum(mp.edf)), 5.129674639, rtol=0, atol=1e-6)
    np.testing.assert_allclose(mp.REML_criterion / 2, 269.6455238, rtol=0, atol=1e-6)
    assert mp._outer_info["iter"] == 7
    # mgcv computes NO sp-uncertainty pieces for efs fits (deriv-0
    # object → gam.fit3.post.proc skips edf2/Vc, gam.fit3.r:978).
    np.testing.assert_allclose(m.edf2, m.edf)
    np.testing.assert_allclose(m.Vc, m.Vp)


def test_efsudr_nb_matches_mgcv():
    # efsudr's gam.fit4 (extended-family) branch: nb θ estimated
    # in-PIRLS by estimate.theta at each accepted iterate
    # (gam.fit4.r:507-515). R ref (mgcv 1.9-4): set.seed(3);
    # x<-runif(200); y<-rnbinom(200, size=3, mu=exp(.8*sin(2*pi*x)+.5));
    # gam(y~s(x), family=nb(), method="REML", optimizer="efs") →
    # sp 0.1130010588, Σedf 5.430232799, REML 356.8301594, 5 iters,
    # score.hist 360.6389944 357.3592854 356.8369431 356.8301811
    # 356.8301594. (mgcv's summary Theta 2.5148 comes from a REJECTED
    # step-extension trial's θ leaked into the family env; the returned
    # object — sp/edf/REML/Vp — is the accepted fit's, whose θ hea
    # keeps: exp(θ̂) 2.5146138.)
    from hea.family import nb
    from hea.R.rng import RGenerator

    g = RGenerator(3)
    n = 200
    x = g.uniform(0, 1, n)
    mu = np.exp(0.8 * np.sin(2 * np.pi * x) + 0.5)
    y = g.mt.rnbinom_n(np.full(n, 3.0), mu).astype(float)
    m = gam(
        "y ~ s(x)",
        pl.DataFrame({"y": y, "x": x}),
        family=nb(),
        method="REML",
        optimizer="efs",
    )
    np.testing.assert_allclose(m.sp[0], 0.1130010588, rtol=1e-8)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.430232799, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.REML_criterion / 2, 356.8301594, rtol=0, atol=1e-6)
    info = m._outer_info
    assert info["conv"] == "full convergence" and info["iter"] == 5
    np.testing.assert_allclose(
        info["score_hist"],
        [360.6389944, 357.3592854, 356.8369431, 356.8301811, 356.8301594],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        float(np.exp(m.family.get_theta()[0])), 2.514613816, rtol=1e-6
    )


def test_efsudr_tw_matches_mgcv():
    # efsudr's hardest path: tw — scale-unknown extended family, so
    # estimate.theta jointly updates (θ_p, log φ) inside PIRLS
    # (family$scale<0 branch, gam.fit4.r:509-513) and efsudr edf-
    # corrects φ for the update (gam.fit4.r:866-871). Poisson-gamma
    # compound data; R ref (mgcv 1.9-4) fit on the identical CSV:
    # sp 0.2209121294, Σedf 5.606763574, θ̂ -1.128564843
    # (p̂ 1.24953753), sig2 1.42810311, REML 570.5731462, 7 iters.
    from hea.family import tw
    from hea.R.rng import RGenerator

    g = RGenerator(5)
    n = 300
    x = g.uniform(0, 1, n)
    lam = np.exp(0.5 * np.sin(2 * np.pi * x) + 0.4)
    N = g.poisson(lam)
    y = np.array(
        [g.gamma(3.0, scale=0.4, size=int(k)).sum() if k > 0 else 0.0 for k in N]
    )
    m = gam(
        "y ~ s(x)",
        pl.DataFrame({"y": y, "x": x}),
        family=tw(),
        method="REML",
        optimizer="efs",
    )
    np.testing.assert_allclose(m.sp[0], 0.2209121294, rtol=1e-7)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.606763574, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(m.family.get_theta()[0]), -1.128564843, rtol=1e-8)
    np.testing.assert_allclose(m.family.p, 1.24953753, rtol=1e-8)
    np.testing.assert_allclose(m.sigma_squared, 1.42810311, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.REML_criterion / 2, 570.5731462, rtol=0, atol=1e-6)
    assert m._outer_info["iter"] == 7


def test_estimate_theta_tw_joint_scale_matches_mgcv():
    # bam._estimate_theta's scale<0 branch (efam.r:5-96) — dormant
    # until efsudr: joint (θ, log φ) Newton for tw. Two bugs surfaced:
    # colSums(as.matrix(Dth)) needs the n×1 COLUMN reading for 1-θ
    # families, and tw.dev_resids/ls_extended must honor the PASSED θ
    # exactly (ls_extended's old allclose skip evaluated the chain rule
    # up to ~1e-5 off, stalling the Newton at the optimum). R ref:
    # mgcv:::estimate.theta(0, tw-family, y, mu=mean(y), scale=-1,
    # tol=1e-7) → θ -0.949822462026, log φ 0.475006868998 (data as in
    # test_efsudr_tw_matches_mgcv).
    from hea.family import tw as _tw
    from hea.models.bam import _estimate_theta
    from hea.R.rng import RGenerator

    g = RGenerator(5)
    n = 300
    x = g.uniform(0, 1, n)
    lam = np.exp(0.5 * np.sin(2 * np.pi * x) + 0.4)
    N = g.poisson(lam)
    y = np.array(
        [g.gamma(3.0, scale=0.4, size=int(k)).sum() if k > 0 else 0.0 for k in N]
    )
    fam = _tw()
    fam.set_theta([0.0])
    mu = np.full(n, float(np.mean(y)))
    out = _estimate_theta(fam, y, mu, scale=-1.0, wt=np.ones(n), tol=1e-7)
    np.testing.assert_allclose(out, [-0.949822462026, 0.475006868998], rtol=1e-9)


def test_nlm_optim_gaussian_matches_mgcv():
    # gam.outer's nlm/optim branch (mgcv.r:1692-1717): R's own
    # optimizers (uncmin/L-BFGS-B, ported bit-exact in hea.R.uncmin /
    # hea.R.lbfgsb) driving gam2objective/gam2derivative/gam4objective.
    # R ref (mgcv 1.9-4, R 4.6.0): set.seed(2); n=120; x0<-runif(n);
    # x1<-runif(n); f<-0.2*x0^11*(10*(1-x0))^6+10*(10*x0)^3*(1-x0)^10;
    # y<-f+2*sin(2*pi*x1)+rnorm(n)*2
    #   optimizer=c("outer","nlm"):  sp .0210021359485741
    #     .521377903996854, REML 269.93183136117, edf 9.88096324336631,
    #     sig2 4.3917390183326, code 5 (stepmax hit 5x — mgcv's
    #     gam.control stepmax=2 genuinely stalls nlm here), 23 iters
    #   optimizer=c("outer","optim"): sp .015299488978041
    #     .160238650281232, REML 269.356777836228, edf 11.2063972648132,
    #     sig2 4.27630344586604, convergence 0, counts 20 20
    from hea.R.rng import RGenerator

    g = RGenerator(2)
    n = 120
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    f = 0.2 * x0**11 * (10 * (1 - x0)) ** 6 + 10 * (10 * x0) ** 3 * (1 - x0) ** 10
    y = f + 2.0 * np.sin(2 * np.pi * x1) + g.normal(0.0, 2.0, n)
    df = pl.DataFrame({"y": y, "x0": x0, "x1": x1})

    m = gam("y ~ s(x0) + s(x1)", df, method="REML", optimizer=("outer", "nlm"))
    np.testing.assert_allclose(m.sp, [0.0210021359485741, 0.521377903996854], rtol=1e-8)
    np.testing.assert_allclose(m.REML_criterion / 2, 269.93183136117, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(m.edf_total), 9.88096324336631, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 4.3917390183326, rtol=1e-8)
    assert m._outer_info["code"] == 5
    assert m._outer_info["iterations"] == 23
    # deriv-0 final fit (the closing gam2objective, mgcv.r:1711):
    # edf2/Vc are NULL in mgcv → hea's magic-style fallbacks
    np.testing.assert_allclose(m.edf2, m.edf)
    np.testing.assert_allclose(m.Vc, m.Vp)
    assert m.sp_vcov() is None

    mo = gam("y ~ s(x0) + s(x1)", df, method="REML", optimizer=("outer", "optim"))
    np.testing.assert_allclose(mo.sp, [0.015299488978041, 0.160238650281232], rtol=1e-8)
    np.testing.assert_allclose(
        mo.REML_criterion / 2, 269.356777836228, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(float(mo.edf_total), 11.2063972648132, rtol=0, atol=1e-6)
    np.testing.assert_allclose(mo.sigma_squared, 4.27630344586604, rtol=1e-8)
    info = mo._outer_info
    assert info["convergence"] == 0
    assert info["counts"] == {"function": 20, "gradient": 20}
    assert info["message"].startswith("CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH")
    np.testing.assert_allclose(mo.edf2, mo.edf)


def test_nlm_optim_poisson_ubre_and_reml_matches_mgcv():
    from hea.family import Poisson
    from hea.R.rng import RGenerator

    g = RGenerator(2)
    n = 200
    x = g.uniform(0, 1, n)
    y = g.poisson(np.exp(0.6 * np.sin(2 * np.pi * x))).astype(float)
    df = pl.DataFrame({"y": y, "x": x})

    m = gam("y ~ s(x)", df, family=Poisson(), method="REML", optimizer=("outer", "nlm"))
    np.testing.assert_allclose(m.sp[0], 0.194798912292, rtol=1e-7)
    np.testing.assert_allclose(m.REML_criterion / 2, 269.645140976, rtol=0, atol=1e-6)
    assert m._outer_info["code"] == 1
    assert m._outer_info["iterations"] == 6

    mo = gam(
        "y ~ s(x)", df, family=Poisson(), method="REML", optimizer=("outer", "optim")
    )
    np.testing.assert_allclose(mo.sp[0], 0.194796680761, rtol=1e-8)
    np.testing.assert_allclose(mo.REML_criterion / 2, 269.645140976, rtol=0, atol=1e-6)
    assert mo._outer_info["counts"] == {"function": 7, "gradient": 7}

    mg = gam(
        "y ~ s(x)", df, family=Poisson(), method="GCV.Cp", optimizer=("outer", "nlm")
    )
    np.testing.assert_allclose(mg.sp[0], 0.258259939929, rtol=1e-7)
    np.testing.assert_allclose(mg.GCV_score, 0.144185096525, rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(mg.edf_total), 4.89211534121, rtol=0, atol=1e-6)
    assert mg._outer_info["code"] == 1
    assert mg._outer_info["iterations"] == 7

    mgo = gam(
        "y ~ s(x)", df, family=Poisson(), method="GCV.Cp", optimizer=("outer", "optim")
    )
    np.testing.assert_allclose(mgo.sp[0], 0.258254628985, rtol=1e-7)
    np.testing.assert_allclose(mgo.GCV_score, 0.144185096523, rtol=0, atol=1e-9)
    assert mgo._outer_info["convergence"] == 0
    assert mgo._outer_info["counts"] == {"function": 7, "gradient": 7}


def test_nlm_optim_nb_tw_matches_mgcv():
    from hea.family import nb, tw
    from hea.R.rng import RGenerator

    g = RGenerator(3)
    n = 200
    x = g.uniform(0, 1, n)
    mu = np.exp(0.8 * np.sin(2 * np.pi * x) + 0.5)
    y = g.mt.rnbinom_n(np.full(n, 3.0), mu).astype(float)
    dfn = pl.DataFrame({"y": y, "x": x})

    m = gam("y ~ s(x)", dfn, family=nb(), method="REML", optimizer=("outer", "nlm"))
    np.testing.assert_allclose(m.sp[0], 0.111885869757, rtol=1e-8)
    np.testing.assert_allclose(m.REML_criterion / 2, 356.788022059, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        float(np.exp(m.family.get_theta()[0])), 2.33156810451, rtol=1e-8
    )
    assert m._outer_info["code"] == 1
    assert m._outer_info["iterations"] == 10

    mo = gam("y ~ s(x)", dfn, family=nb(), method="REML", optimizer=("outer", "optim"))
    np.testing.assert_allclose(mo.sp[0], 0.111890801175, rtol=2e-3)
    np.testing.assert_allclose(mo.REML_criterion / 2, 356.788022061, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        float(np.exp(mo.family.get_theta()[0])), 2.3315789377, rtol=1e-3
    )
    assert mo._outer_info["convergence"] == 0

    g = RGenerator(5)
    n = 300
    x = g.uniform(0, 1, n)
    lam = np.exp(0.5 * np.sin(2 * np.pi * x) + 0.4)
    N = g.poisson(lam)
    y = np.array(
        [g.gamma(3.0, scale=0.4, size=int(k)).sum() if k > 0 else 0.0 for k in N]
    )
    dft = pl.DataFrame({"y": y, "x": x})

    mt = gam("y ~ s(x)", dft, family=tw(), method="REML", optimizer=("outer", "nlm"))
    np.testing.assert_allclose(mt.sp[0], 0.220459770958, rtol=5e-7)
    np.testing.assert_allclose(mt.REML_criterion / 2, 570.55170345, rtol=0, atol=1e-6)
    np.testing.assert_allclose(mt.family.p, 1.2530740822, rtol=1e-8)
    np.testing.assert_allclose(mt.sigma_squared, 1.44666399882, rtol=1e-8)
    assert mt._outer_info["code"] == 1
    assert mt._outer_info["iterations"] == 19

    mto = gam("y ~ s(x)", dft, family=tw(), method="REML", optimizer=("outer", "optim"))
    np.testing.assert_allclose(mto.sp[0], 0.220459657052, rtol=2e-3)
    np.testing.assert_allclose(mto.REML_criterion / 2, 570.55170345, rtol=0, atol=1e-5)
    np.testing.assert_allclose(mto.family.p, 1.25307407149, rtol=1e-4)
    assert mto._outer_info["convergence"] == 0


def test_general_family_authoring_contract():
    from itertools import combinations_with_replacement

    from scipy.special import digamma, gammaln, polygamma

    from hea.family import (
        GeneralFamily,
        IdentityLink,
        Link,
        gamlss_etamu,
        gamlss_gH,
        trind_generator,
    )

    class _ShiftedLogLink(Link):
        name = "slog"
        b = 0.01

        def link(self, mu):
            return np.log(np.asarray(mu, dtype=float) - self.b)

        def linkinv(self, eta):
            return self.b + np.exp(np.clip(np.asarray(eta, dtype=float), -700.0, 700.0))

        def mu_eta(self, eta):
            return np.maximum(
                np.exp(np.clip(np.asarray(eta, dtype=float), -700.0, 700.0)),
                np.finfo(float).eps,
            )

        def d2link(self, mu):
            return -1.0 / (np.asarray(mu, dtype=float) - self.b) ** 2

        def d3link(self, mu):
            return 2.0 / (np.asarray(mu, dtype=float) - self.b) ** 3

        def d4link(self, mu):
            return -6.0 / (np.asarray(mu, dtype=float) - self.b) ** 4

    class _TanhLink(Link):
        name = "tanh"

        def link(self, mu):
            return np.arctanh(np.asarray(mu, dtype=float))

        def linkinv(self, eta):
            eps = np.finfo(float).eps
            return np.clip(np.tanh(np.asarray(eta, dtype=float)), -1.0 + eps, 1.0 - eps)

        def mu_eta(self, eta):
            a = np.exp(-2.0 * np.abs(np.asarray(eta, dtype=float)))
            return np.maximum(4.0 * a / (1.0 + a) ** 2, np.finfo(float).eps)

        def d2link(self, mu):
            mu = np.asarray(mu, dtype=float)
            return 2.0 * mu / (1.0 - mu * mu) ** 2

        def d3link(self, mu):
            mu = np.asarray(mu, dtype=float)
            return (2.0 + 6.0 * mu * mu) / (1.0 - mu * mu) ** 3

        def d4link(self, mu):
            mu = np.asarray(mu, dtype=float)
            return 24.0 * mu * (1.0 + mu * mu) / (1.0 - mu * mu) ** 4

    class _TLSS(GeneralFamily):
        name = "tlss-dummy"
        n_lp = 3
        available_derivs = 0
        scale_known = True
        n_theta = 0

        def __init__(self):
            super().__init__([IdentityLink(), _ShiftedLogLink(), _TanhLink()])
            self.tri = trind_generator(3)
            self.seen = {
                "deriv": [],
                "use_unscaled": [],
                "postproc_kwargs": None,
                "offsets": None,
                "lpi_cols": None,
                "wt_len": None,
            }

        @staticmethod
        def _l0(y, mu, sigma, lam):
            nu = 6.0 + 4.0 * lam
            z = (y - mu) / sigma
            return (
                gammaln((nu + 1.0) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * np.log(nu * np.pi)
                - np.log(sigma)
                - (nu + 1.0) / 2.0 * np.log1p(z * z / nu)
            )

        @staticmethod
        def _lp_derivs(y, mu, sigma, lam):
            nu = 6.0 + 4.0 * lam
            z = (y - mu) / sigma
            q = nu + z * z
            g = (nu + 1.0) * z / q
            dg_dz = (nu + 1.0) * (nu - z * z) / q**2
            dg_dnu = z * (z * z - 1.0) / q**2
            dC = 0.5 * digamma((nu + 1.0) / 2.0) - 0.5 * digamma(nu / 2.0) - 0.5 / nu
            d2C = (
                0.25 * polygamma(1, (nu + 1.0) / 2.0)
                - 0.25 * polygamma(1, nu / 2.0)
                + 0.5 / nu**2
            )
            l_nu = dC - 0.5 * np.log1p(z * z / nu) + (nu + 1.0) * z * z / (2.0 * nu * q)
            l_nunu = (
                d2C
                + z * z / (2.0 * nu * q)
                - z * z * (nu * nu + 2.0 * nu + z * z) / (2.0 * nu * nu * q * q)
            )
            d1 = {"mu": g / sigma, "sigma": (g * z - 1.0) / sigma, "lam": 4.0 * l_nu}
            d2 = {
                ("mu", "mu"): -dg_dz / sigma**2,
                ("mu", "sigma"): -(g + z * dg_dz) / sigma**2,
                ("mu", "lam"): 4.0 * dg_dnu / sigma,
                ("sigma", "sigma"): (1.0 - 2.0 * g * z - z * z * dg_dz) / sigma**2,
                ("sigma", "lam"): 4.0 * z * dg_dnu / sigma,
                ("lam", "lam"): 16.0 * l_nunu,
            }
            return d1, d2

        def ll(
            self,
            y,
            X,
            coef,
            wt,
            *,
            lpi,
            offset=None,
            deriv=0,
            d1b=None,
            d2b=None,
            fh=None,
            D=None,
        ):
            self.seen["deriv"].append(deriv)
            self.seen["wt_len"] = None if wt is None else len(wt)
            self.seen["offsets"] = offset
            self.seen["lpi_cols"] = sorted(int(c) for ix in lpi for c in np.asarray(ix))
            assert deriv <= 1, f"derivs-0 family asked for ll(deriv={deriv})"
            y = np.asarray(y, dtype=float)
            X = np.asarray(X, dtype=float)
            coef = np.asarray(coef, dtype=float)
            jj = [np.asarray(ix, dtype=int) for ix in lpi]
            etas = []
            for j in range(3):
                eta = X[:, jj[j]] @ coef[jj[j]]
                if offset is not None and offset[j] is not None:
                    eta = eta + offset[j]
                etas.append(eta)
            mu = self.links[0].linkinv(etas[0])
            sigma = self.links[1].linkinv(etas[1])
            lam = self.links[2].linkinv(etas[2])
            w = np.ones_like(y) if wt is None else np.asarray(wt, dtype=float)
            l0 = self._l0(y, mu, sigma, lam)
            ret = {"l": float(np.sum(w * l0))}
            if deriv == 0:
                return ret
            names = ("mu", "sigma", "lam")
            params = {"mu": mu, "sigma": sigma, "lam": lam}
            d1, d2 = self._lp_derivs(y, mu, sigma, lam)
            shape = y.shape
            l1 = np.column_stack([np.broadcast_to(d1[p], shape) for p in names])
            l2 = np.column_stack(
                [
                    np.broadcast_to(d2[k], shape)
                    for k in combinations_with_replacement(names, 2)
                ]
            )
            l1 = l1 * w[:, None]
            l2 = l2 * w[:, None]
            ig1 = np.column_stack(
                [lnk.mu_eta(eta) for lnk, eta in zip(self.links, etas)]
            )
            g2 = np.column_stack(
                [lnk.d2link(params[name]) for lnk, name in zip(self.links, names)]
            )
            tri = self.tri
            de = gamlss_etamu(
                l1,
                l2,
                None,
                None,
                ig1,
                g2,
                None,
                None,
                tri["i2"],
                tri["i3"],
                tri["i4"],
                deriv - 1,
            )
            gh = gamlss_gH(
                X,
                jj,
                de["l1"],
                de["l2"],
                tri["i2"],
                l3=de["l3"],
                i3=tri["i3"],
                l4=de["l4"],
                i4=tri["i4"],
                d1b=d1b,
                d2b=d2b,
                deriv=deriv - 1,
                fh=fh,
                D=D,
            )
            ret.update(gh)
            return ret

        def _null_params(self, y):
            y = np.asarray(y, dtype=float)
            return (float(np.median(y)), max(float(np.std(y)) * 0.8, 0.05), 0.0)

        def initialize_coef(self, y, X, lpi, E=None, offset=None, use_unscaled=False):
            self.seen["use_unscaled"].append(bool(use_unscaled))
            y = np.asarray(y, dtype=float)
            X = np.asarray(X, dtype=float)
            jj = [np.asarray(ix, dtype=int) for ix in lpi]
            n, p = X.shape
            if E is None:
                E = np.zeros((0, p))
            start = np.zeros(p)
            for j, (lnk, p0) in enumerate(zip(self.links, self._null_params(y))):
                target = np.full(n, float(lnk.link(p0)))
                if offset is not None and offset[j] is not None:
                    target = target - offset[j]
                cols = jj[j]
                xa = np.vstack([X[:, cols], E[:, cols]])
                ta = np.concatenate([target, np.zeros(E.shape[0])])
                b, *_ = np.linalg.lstsq(xa, ta, rcond=None)
                start[cols] = np.where(np.isfinite(b), b, 0.0)
            return start

        def postproc(
            self, y, prior_weights, fitted, linear_predictors, offset, intercept
        ):
            self.seen["postproc_kwargs"] = {
                "prior_weights": np.shape(prior_weights),
                "fitted": np.shape(fitted),
                "linear_predictors": np.shape(linear_predictors),
                "intercept": intercept,
            }
            y = np.asarray(y, dtype=float)
            f0 = np.broadcast_to(np.asarray(self._null_params(y)), (y.shape[0], 3))
            r0 = self.residuals(y, f0)
            return {"null_deviance": float(np.sum(r0 * r0))}

        def residuals(self, y, fitted, type: str = "deviance"):
            y = np.asarray(y, dtype=float)
            fitted = np.asarray(fitted, dtype=float)
            mu, sigma, lam = (fitted[:, 0], fitted[:, 1], fitted[:, 2])
            rsd = y - mu
            if type == "response":
                return rsd
            if type == "pearson":
                nu = 6.0 + 4.0 * lam
                return rsd / (sigma * np.sqrt(nu / (nu - 2.0)))
            l_sat = self._l0(y, y, sigma, lam)
            l_obs = self._l0(y, mu, sigma, lam)
            return np.sign(rsd) * np.sqrt(2.0 * np.clip(l_sat - l_obs, 0.0, None))

    fam = _TLSS()
    rng = np.random.default_rng(11)
    n = 40
    Xs = np.hstack(
        [
            np.ones((n, 1)),
            rng.normal(size=(n, 2)),
            np.ones((n, 1)),
            rng.normal(size=(n, 1)),
            np.ones((n, 1)),
        ]
    )
    lpi = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 6)]
    yv = 0.5 + Xs[:, 1] * 0.4 + rng.standard_t(df=6, size=n) * 0.7
    coef = np.array([0.3, 0.2, -0.1, np.log(0.8 - 0.01), 0.05, 0.1])
    wt = np.ones(n)
    base = fam.ll(yv, Xs, coef, wt, lpi=lpi, deriv=1)
    h = 1e-6
    fd_lb = np.empty(6)
    fd_lbb = np.empty((6, 6))
    for k in range(6):
        cp = coef.copy()
        cm = coef.copy()
        cp[k] += h
        cm[k] -= h
        fd_lb[k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=0)["l"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=0)["l"]
        ) / (2 * h)
        fd_lbb[:, k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=1)["lb"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=1)["lb"]
        ) / (2 * h)
    np.testing.assert_allclose(base["lb"], fd_lb, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        base["lbb"], 0.5 * (fd_lbb + fd_lbb.T), rtol=5e-5, atol=5e-6
    )

    n = 320
    x = rng.uniform(size=n)
    off1 = 0.3 * rng.normal(size=n)
    y = 1.0 + np.sin(2 * np.pi * x) + off1 + rng.standard_t(df=7.0, size=n) * 0.7
    df = pl.DataFrame({"y": y, "x": x, "off1": off1})
    fam2 = _TLSS()
    m = gam(["y ~ s(x) + offset(off1)", "~ 1", "~ 1"], df, family=fam2, method="REML")

    assert max(fam2.seen["deriv"]) <= 1
    assert set(fam2.seen["use_unscaled"]) == {False, True}
    off_seen = fam2.seen["offsets"]
    assert isinstance(off_seen, (list, tuple)) and len(off_seen) == 3
    np.testing.assert_allclose(np.asarray(off_seen[0]), off1, rtol=0, atol=1e-12)
    assert off_seen[1] is None and off_seen[2] is None
    assert fam2.seen["lpi_cols"] == list(range(len(np.asarray(m._beta))))
    assert fam2.seen["wt_len"] == n
    assert m.outer_info["conv"] in ("full convergence", "iteration limit reached")
    pk = fam2.seen["postproc_kwargs"]
    assert pk == {
        "prior_weights": (n,),
        "fitted": (n, 3),
        "linear_predictors": (n, 3),
        "intercept": True,
    }
    assert np.isfinite(m.null_deviance) and m.null_deviance > 0
    np.testing.assert_allclose(
        m.deviance, float(np.sum(np.asarray(m.residuals) ** 2)), rtol=1e-12
    )
    np.testing.assert_array_equal(
        np.asarray(m.residuals_of("deviance")), np.asarray(m.residuals)
    )
    fitted = np.asarray(m.fitted_values)
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("response")), y - fitted[:, 0], rtol=0, atol=1e-12
    )
    truth = 1.0 + np.sin(2 * np.pi * x) + off1
    assert np.corrcoef(fitted[:, 0], truth)[0, 1] > 0.95
    assert 0.3 < float(np.median(fitted[:, 1])) < 1.5
    assert float(np.max(np.abs(fitted[:, 2]))) < 0.999
    pred = m.predict(df[:5])
    assert pred.shape[0] == 5
    m.summary()

    fam3 = _TLSS()
    m_cc = gam(
        ['y ~ s(x, bs="cc")', "~ 1", "~ 1"],
        df,
        family=fam3,
        method="REML",
        knots={"x": [0.0, 1.0]},
    )
    ((label, _edf, ref_df, stat, p_cc),) = m_cc._smooth_significance_rows()
    assert label == "s(x)" and ref_df > 0
    assert np.isfinite(stat) and 0.0 <= p_cc <= 1.0
    m_cc.summary()


def test_general_family_newton_reml_nlp4_robustness():
    from itertools import combinations_with_replacement
    from itertools import product as iproduct

    from hea.family import (
        GeneralFamily,
        IdentityLink,
        Link,
        LogLink,
        gamlss_etamu,
        gamlss_gH,
        trind_generator,
    )

    K = 4
    AQ = np.ones(K)  # per-parameter quadratic curvature
    DQ = 1.0  # quartic floor -> l0 bounded above
    CC = 0.5  # all-distinct coupling (|CC|/4 < DQ)

    class _TanhLink(Link):
        name = "tanh"

        def link(self, mu):
            return np.arctanh(np.asarray(mu, float))

        def linkinv(self, eta):
            eps = np.finfo(float).eps
            return np.clip(np.tanh(np.asarray(eta, float)), -1 + eps, 1 - eps)

        def mu_eta(self, eta):
            a = np.exp(-2.0 * np.abs(np.asarray(eta, float)))
            return np.maximum(4.0 * a / (1.0 + a) ** 2, np.finfo(float).eps)

        def d2link(self, mu):
            mu = np.asarray(mu, float)
            return 2.0 * mu / (1.0 - mu * mu) ** 2

        def d3link(self, mu):
            mu = np.asarray(mu, float)
            return (2.0 + 6.0 * mu * mu) / (1.0 - mu * mu) ** 3

        def d4link(self, mu):
            mu = np.asarray(mu, float)
            return 24.0 * mu * (1.0 + mu * mu) / (1.0 - mu * mu) ** 4

    def _u(y, mus):
        u = mus - np.array([0.0, 1.0, 0.0, 0.0])[None, :]
        u[:, 0] = mus[:, 0] - y
        return u

    def _l0(y, mus):
        u = _u(y, mus)
        return (
            -0.5 * (AQ[None, :] * u * u).sum(1)
            - DQ * (u**4).sum(1)
            + CC * np.prod(u, axis=1)
        )

    def _partial(u, idx):
        n = u.shape[0]
        s = set(idx)
        if len(idx) == 1:
            k = idx[0]
            prod = np.full(n, CC)
            for j in range(K):
                if j != k:
                    prod = prod * u[:, j]
            return -AQ[k] * u[:, k] - 4.0 * DQ * u[:, k] ** 3 + prod
        if len(s) != len(idx):  # some repeated index
            if len(idx) == 2 and len(s) == 1:
                return -AQ[idx[0]] - 12.0 * DQ * u[:, idx[0]] ** 2
            if len(idx) == 3 and len(s) == 1:
                return -24.0 * DQ * u[:, idx[0]]
            if len(idx) == 4 and len(s) == 1:
                return np.full(n, -24.0 * DQ)
            return np.zeros(n)
        missing = [j for j in range(K) if j not in s]  # all distinct
        prod = np.full(n, CC)
        for j in missing:
            prod = prod * u[:, j]
        return prod

    class _K4(GeneralFamily):
        name = "k4-dummy"
        n_lp = 4
        available_derivs = 2
        scale_known = True
        n_theta = 0

        def __init__(self):
            super().__init__([IdentityLink(), LogLink(), IdentityLink(), _TanhLink()])
            self.tri = trind_generator(4)
            self.seen = {"deriv": [], "lpi_cols": None}

        def ll(
            self,
            y,
            X,
            coef,
            wt,
            *,
            lpi,
            offset=None,
            deriv=0,
            d1b=None,
            d2b=None,
            fh=None,
            D=None,
        ):
            self.seen["deriv"].append(deriv)
            y = np.asarray(y, float)
            X = np.asarray(X, float)
            coef = np.asarray(coef, float)
            jj = [np.asarray(ix, int) for ix in lpi]
            self.seen["lpi_cols"] = sorted(int(c) for ix in jj for c in ix)
            etas = []
            for j in range(K):
                eta = X[:, jj[j]] @ coef[jj[j]]
                if offset is not None and offset[j] is not None:
                    eta = eta + offset[j]
                etas.append(eta)
            mus = np.column_stack([lnk.linkinv(e) for lnk, e in zip(self.links, etas)])
            w = np.ones_like(y) if wt is None else np.asarray(wt, float)
            ret = {"l": float(np.sum(w * _l0(y, mus)))}
            if deriv == 0:
                return ret
            u = _u(y, mus)
            packs = [
                list(combinations_with_replacement(range(K), m)) for m in (1, 2, 3, 4)
            ]
            l1, l2, l3, l4 = (
                np.column_stack([_partial(u, c) for c in cs]) * w[:, None]
                for cs in packs
            )
            ig1 = np.column_stack([lnk.mu_eta(e) for lnk, e in zip(self.links, etas)])
            g2 = np.column_stack(
                [lnk.d2link(mus[:, k]) for k, lnk in enumerate(self.links)]
            )
            g3 = np.column_stack(
                [lnk.d3link(mus[:, k]) for k, lnk in enumerate(self.links)]
            )
            g4 = np.column_stack(
                [lnk.d4link(mus[:, k]) for k, lnk in enumerate(self.links)]
            )
            t = self.tri
            de = gamlss_etamu(
                l1, l2, l3, l4, ig1, g2, g3, g4, t["i2"], t["i3"], t["i4"], deriv - 1
            )
            if deriv > 1 and d1b is None:
                return {**ret, "_de": de, "_ig1": ig1, "_u": u}
            gh = gamlss_gH(
                X,
                jj,
                de["l1"],
                de["l2"],
                t["i2"],
                l3=de["l3"],
                i3=t["i3"],
                l4=de["l4"],
                i4=t["i4"],
                d1b=d1b,
                d2b=d2b,
                deriv=deriv - 1,
                fh=fh,
                D=D,
            )
            ret.update(gh)
            return ret

        def initialize_coef(self, y, X, lpi, E=None, offset=None, use_unscaled=False):
            y = np.asarray(y, float)
            X = np.asarray(X, float)
            jj = [np.asarray(ix, int) for ix in lpi]
            n, p = X.shape
            if E is None:
                E = np.zeros((0, p))
            start = np.zeros(p)
            tgt0 = [float(np.median(y)), 1.0, 0.0, 0.0]
            for j, lnk in enumerate(self.links):
                target = np.full(n, float(lnk.link(tgt0[j])))
                if offset is not None and offset[j] is not None:
                    target = target - offset[j]
                cols = jj[j]
                xa = np.vstack([X[:, cols], E[:, cols]])
                ta = np.concatenate([target, np.zeros(E.shape[0])])
                b, *_ = np.linalg.lstsq(xa, ta, rcond=None)
                start[cols] = np.where(np.isfinite(b), b, 0.0)
            return start

        def postproc(
            self, y, prior_weights, fitted, linear_predictors, offset, intercept
        ):
            y = np.asarray(y, float)
            r0 = y - float(np.median(y))
            return {"null_deviance": float(np.sum(r0 * r0))}

        def residuals(self, y, fitted, type: str = "deviance"):
            y = np.asarray(y, float)
            fitted = np.asarray(fitted, float)
            return y - fitted[:, 0]

    fam = _K4()
    rng = np.random.default_rng(7)
    n = 36
    Xs = np.hstack(
        [
            np.ones((n, 1)),
            rng.normal(size=(n, 2)),  # LP1: 3
            np.ones((n, 1)),
            rng.normal(size=(n, 1)),  # LP2: 2
            np.ones((n, 1)),  # LP3: 1
            np.ones((n, 1)),
            rng.normal(size=(n, 1)),
        ]
    )  # LP4: 2
    lpi = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 6), np.arange(6, 8)]
    p = Xs.shape[1]
    yv = rng.normal(size=n) * 0.5 + 0.3
    coef = rng.normal(size=p) * 0.2
    wt = np.ones(n)
    base = fam.ll(yv, Xs, coef, wt, lpi=lpi, deriv=1)
    h = 1e-6
    fd_lb = np.empty(p)
    fd_lbb = np.empty((p, p))
    for k in range(p):
        cp = coef.copy()
        cm = coef.copy()
        cp[k] += h
        cm[k] -= h
        fd_lb[k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=0)["l"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=0)["l"]
        ) / (2 * h)
        fd_lbb[:, k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=1)["lb"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=1)["lb"]
        ) / (2 * h)
    np.testing.assert_allclose(base["lb"], fd_lb, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(
        base["lbb"], 0.5 * (fd_lbb + fd_lbb.T), rtol=5e-5, atol=5e-6
    )

    dd = fam.ll(yv, Xs, coef, wt, lpi=lpi, deriv=4)
    de, ig1, u = dd["_de"], dd["_ig1"], dd["_u"]
    cols3 = list(combinations_with_replacement(range(K), 3))
    cols4 = list(combinations_with_replacement(range(K), 4))
    ad4 = cols4.index((0, 1, 2, 3))
    ad3 = cols3.index((0, 1, 2))
    want4 = CC * ig1[:, 0] * ig1[:, 1] * ig1[:, 2] * ig1[:, 3]
    want3 = CC * u[:, 3] * ig1[:, 0] * ig1[:, 1] * ig1[:, 2]
    np.testing.assert_allclose(de["l4"][:, ad4], want4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(de["l3"][:, ad3], want3, rtol=0, atol=1e-12)
    etas0 = np.column_stack([Xs[:, lpi[j]] @ coef[lpi[j]] for j in range(K)])
    hs = 8e-3
    acc = np.zeros(n)
    for signs in iproduct((1, -1), repeat=K):
        et = etas0 + hs * np.array(signs)[None, :]
        mus = np.column_stack(
            [lnk.linkinv(et[:, j]) for j, lnk in enumerate(fam.links)]
        )
        acc += float(np.prod(signs)) * _l0(yv, mus)
    np.testing.assert_allclose(de["l4"][:, ad4], acc / (2 * hs) ** 4, rtol=0, atol=1e-4)

    N = 300
    x = rng.uniform(size=N)
    y = np.sin(2 * np.pi * x) + rng.normal(size=N) * 0.3
    df = pl.DataFrame({"y": y, "x": x})
    fam2 = _K4()
    m = gam(["y ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family=fam2, method="REML")
    assert 4 in fam2.seen["deriv"] and 3 in fam2.seen["deriv"]
    assert m.outer_info["conv"] == "full convergence"
    assert fam2.seen["lpi_cols"] == list(range(len(np.asarray(m._beta))))
    fitted = np.asarray(m.fitted_values)
    assert fitted.shape == (N, 4)
    assert np.corrcoef(fitted[:, 0], np.sin(2 * np.pi * x))[0, 1] > 0.95
    assert np.all(np.isfinite(np.asarray(m.Vp)))
    assert np.isfinite(m.REML_criterion)
    assert np.isfinite(m.null_deviance) and m.null_deviance > 0
    np.testing.assert_allclose(
        m.deviance, float(np.sum(np.asarray(m.residuals) ** 2)), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("response")), y - fitted[:, 0], rtol=0, atol=1e-12
    )
    assert m.predict(df[:5]).shape[0] == 5
    m.summary()


def _twlss_fixture():
    from hea.family import _r_tweedie
    from hea.R.rng import RGenerator

    gen = RGenerator(9)
    n = 300
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    w = gen.uniform(size=n)
    mu = np.exp(0.3 + np.sin(2 * np.pi * x) + 0.3 * w)
    y = _r_tweedie(gen, mu, p=1.55, phi=0.9)
    return pl.DataFrame({"y": y, "x": x, "z": z, "w": w})


def _gammals_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(11)
    n = 250
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    mean = np.exp(0.5 + np.sin(2 * np.pi * x))
    sigma = np.exp(-1.0 + 0.6 * z)
    y = gen.gamma(shape=1.0 / sigma, scale=mean * sigma)
    return pl.DataFrame({"y": y, "x": x, "z": z})


def _gumbls_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(13)
    n = 250
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    loc = 0.5 + np.sin(2 * np.pi * x)
    logbeta = -0.5 + 0.4 * z
    u = gen.uniform(size=n)
    y = loc - np.exp(logbeta) * np.log(-np.log(u))
    return pl.DataFrame({"y": y, "x": x, "z": z})


def _gevlss_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(17)
    n = 300
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    mu = 0.5 + np.sin(2 * np.pi * x)
    logsig = -0.5 + 0.3 * z
    xi = 0.1
    u = gen.uniform(size=n)
    y = mu + ((-np.log(u)) ** (-xi) - 1.0) * np.exp(logsig) / xi
    return pl.DataFrame({"y": y, "x": x, "z": z})


def test_twlss_through_gam_matches_mgcv():
    # R: gam(list(y ~ s(x) + w, ~ 1, ~ s(z)), family=twlss(),
    # method="REML") — available.derivs=0, so mgcv coerces the
    # optimizer to efs (mgcv.r:1908) and hea auto-dispatches the same
    # way; tolerances sit inside EFS's own stop band (efs.tol = 0.1),
    # like the gaulss efs pins. m2 puts a covariate on the θ (index)
    # predictor — per-row p through the vectorized ldTweedie series
    # (the C_tweedious2 case) end-to-end.
    from hea.family import twlss

    df = _twlss_fixture()
    m1 = gam(["y ~ s(x) + w", "~ 1", "~ s(z)"], df, family=twlss(), method="REML")
    np.testing.assert_allclose(m1.REML_criterion / 2, 515.6445331249, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m1.sp[0], 0.1810879219, rtol=1e-4)
    assert m1.sp[1] > 500.0
    np.testing.assert_allclose(m1.edf_total, 10.2292798004, rtol=0, atol=2e-3)
    rows = m1._smooth_significance_rows()
    np.testing.assert_allclose(rows[1][1], 1.0070551110, rtol=0, atol=5e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m1._beta)[:4]),
        np.abs([0.3636872521, 0.2340139050, 1.7497627520, 0.2383358048]),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[:2],
        [
            [4.1009048580, -0.0109253440, -0.1731325510],
            [2.7718950940, -0.0109253440, -0.0679816568],
        ],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(m1.deviance, 357.7929921000, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m1.null_deviance, 567.2259056000, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m1.residuals)[:3],
        [-1.1855994500, -0.2774933634, -0.4901982250],
        rtol=0,
        atol=1e-4,
    )
    assert 6 <= m1.outer_info["iter"] <= 12  # R: 9
    np.testing.assert_allclose(np.asarray(m1.Vp)[0, 0], 0.0102796362, rtol=0, atol=1e-6)
    assert m1.sp_vcov() is None  # deriv-0 fit
    pred = m1.predict(df[:3])
    np.testing.assert_allclose(
        pred["fit"].to_numpy(), np.asarray(m1.fitted_values)[:3, 0], rtol=0, atol=1e-10
    )
    m1.summary()

    m2 = gam(["y ~ s(x)", "~ z", "~ 1"], df, family=twlss(), method="REML")
    np.testing.assert_allclose(m2.REML_criterion / 2, 514.1250577000, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m2.sp, [0.1885776983], rtol=1e-4)
    np.testing.assert_allclose(m2.edf_total, 9.1773799040, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m2._beta)[10:13],
        [-0.0034247063, -0.0106819944, -0.1195655320],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(m2.fitted_values)[0],
        [3.8117852460, -0.0062579028, -0.1195655320],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(m2.deviance, 358.1491851000, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m2.null_deviance, 564.4292756000, rtol=0, atol=1e-3)


def test_twlss_prior_weights_honoured():
    # hea's twlss honours gam(weights=) as a weighted log-likelihood — a
    # defined divergence from mgcv, whose twlss ll DROPS prior weights
    # (gamlss.r:2556, wt unread). There is therefore no mgcv oracle for a
    # weighted twlss fit; the contract is the duplication identity —
    # weighting a row by integer w equals the unweighted fit on the
    # row-duplicated design. That is exact for a parametric model; a
    # data-driven s(x) basis shifts under duplication (quantile knots move),
    # so the smooth fit is only approximately invariant and is not asserted.
    from hea.family import twlss

    df = _twlss_fixture()
    pw = np.tile([1.0, 2.0], 150)
    reps = pw.astype(int)
    dfd = pl.DataFrame({c: np.repeat(df[c].to_numpy(), reps) for c in df.columns})

    mu = gam(["y ~ s(x)", "~ 1", "~ 1"], df, family=twlss(), method="REML")
    np.testing.assert_allclose(mu.REML_criterion / 2, 514.2855621000, rtol=0, atol=1e-4)
    np.testing.assert_allclose(mu.sp, [0.1886027173], rtol=1e-4)
    np.testing.assert_allclose(mu.deviance, 358.1582964000, rtol=0, atol=1e-3)
    np.testing.assert_allclose(mu.null_deviance, 564.4194261000, rtol=0, atol=1e-3)

    mwp = gam(["y ~ w", "~ 1", "~ 1"], df, family=twlss(), weights=pw)
    mdp = gam(["y ~ w", "~ 1", "~ 1"], dfd, family=twlss())
    np.testing.assert_allclose(
        np.asarray(mwp._beta), np.asarray(mdp._beta), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(mwp.deviance, mdp.deviance, rtol=0, atol=1e-4)

    mup = gam(["y ~ w", "~ 1", "~ 1"], df, family=twlss())
    assert np.max(np.abs(np.asarray(mwp._beta) - np.asarray(mup._beta))) > 1e-3
    # 4. Deviance residuals carry the prior weights as a per-row √w scaling
    #    (√(2(yθ−κ)w/φ); object$prior.weights, gamlss.r:2541) — mgcv-faithful
    #    and unchanged by the likelihood-weighting fix. Verified on ONE fitted
    #    object (μ/θ/φ identical) so the scaling is exact to ~1 ulp and
    #    BLAS-independent.
    yv = np.asarray(df["y"], dtype=float)
    fit = np.asarray(mu.fitted, dtype=float)
    r_un = twlss().residuals(yv, fit, type="deviance")
    r_wt = twlss().residuals(yv, fit, type="deviance", prior_weights=pw)
    np.testing.assert_allclose(r_wt, r_un * np.sqrt(pw), rtol=1e-12)


def test_gammals_through_gam_matches_mgcv():
    # R: gam(list(y ~ s(x), ~ s(z)), family=gammals(), method="REML") on
    # the set.seed(11) fixture — gammals is available.derivs=2, so this
    # drives the full outer Newton (REML1/REML2 ⇒ ll deriv 3/4) and the
    # new family$predict hook (predict.gam, mgcv.r:3171-3198).
    from hea.family import gammals

    df = _gammals_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=gammals(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 376.9788523, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.1010813233, rtol=1e-4)
    assert m.sp[1] > 500.0
    np.testing.assert_allclose(m.edf_total, 8.586757468, rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.5351402153, 2.185161584, 0.1922625554, 0.1110607536]),
        rtol=0,
        atol=1e-4,
    )
    # fitted matrix is (mean, log σ): col 0 exponentiated by postproc
    # (gamlss.r:2739) — mgcv's object$fitted.values after the in-place
    # rewrite. R: rows 1-2 = (4.5658, -0.3322), (1.8810, -0.7676).
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[4.56583108, -0.3321797963], [1.881021014, -0.7676485103]],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(m.deviance, 272.1383283, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 532.7823068, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.002131979348, rtol=0, atol=1e-6
    )

    pr = m.predict(df[:3], se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [4.56583108, 1.881021014, 1.385835927], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        pr["fit.1"].to_numpy(),
        [-0.3321797963, -0.7676485103, -0.3454239398],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.4412707648, 0.4300296758, 0.1464255231],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pr["se.fit.1"].to_numpy(),
        [0.149232104, 0.1060445032, 0.1444384638],
        rtol=0,
        atol=1e-5,
    )
    pr_all = m.predict(type="response")
    np.testing.assert_allclose(
        pr_all["fit"].to_numpy(), np.asarray(m.fitted_values)[:, 0], rtol=0, atol=1e-9
    )
    pr_lnk = m.predict(df[:2], type="link")
    np.testing.assert_allclose(
        pr_lnk["fit"].to_numpy(),
        np.log(np.asarray(m.fitted_values)[:2, 0]),
        rtol=0,
        atol=1e-9,
    )
    m.summary()


def test_gumbls_through_gam_matches_mgcv():
    from hea.family import gumbls

    df = _gumbls_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=gumbls(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 355.4765655, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.1329424295, rtol=1e-3)
    assert m.sp[1] > 500.0  # flat λ→∞ direction (s(z) on log β)
    np.testing.assert_allclose(m.edf_total, 8.327469431, rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.4872208906, -1.993640641, 0.2718586452, 0.1469972102]),
        rtol=0,
        atol=1e-3,
    )
    # fitted matrix is (mean, log β): col 0 = location + e^β·γ, added in
    # place by postproc (gamlss.r:3070). R: rows 1-2 of fitted.values.
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[-0.09536002225, -0.2589687628], [1.836117586, -0.4267671006]],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(m.deviance, 291.9068221, rtol=0, atol=1e-2)
    assert np.isnan(m.null_deviance)  # mgcv leaves gumbls null dev NA
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00270728526, rtol=0, atol=1e-5)

    pr = m.predict(df[:3], se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [-0.5408822784, 1.45941768, 1.0853551], rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        pr["fit.1"].to_numpy(),
        [-0.2589687628, -0.4267671006, -0.2658441114],
        rtol=0,
        atol=1e-4,
    )
    assert abs(float(pr["fit"][0]) - float(m.fitted_values[0, 0])) > 0.1
    m.summary()


def test_gevlss_through_gam_matches_mgcv():
    from hea.family import gevlss

    df = _gevlss_fixture()
    m = gam(["y ~ s(x)", "~ s(z)", "~ 1"], df, family=gevlss(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 400.7915694, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.09316907739, rtol=1e-3)
    assert m.sp[1] > 50.0  # s(z)-on-log-σ flat-ish direction
    np.testing.assert_allclose(m.edf_total, 10.54442778, rtol=0, atol=2e-3)
    np.testing.assert_allclose(np.asarray(m._beta)[-1], 1.184566711, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [
            [1.268048747, -0.3566666145, 0.1486518752],
            [0.02079092758, -0.4185000686, 0.1486518752],
        ],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        float(np.sum(np.diag(np.asarray(m.Vp)))), 2.90608114, rtol=0, atol=1e-2
    )
    assert np.isnan(m.null_deviance)  # mgcv leaves gevlss null dev NA
    assert 4 <= m.outer_info["iter"] <= 10  # R: 6

    pr = m.predict(type="response")
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), np.asarray(m.fitted_values)[:, 0], rtol=0, atol=1e-9
    )
    m.summary()


def test_gevlss_rank_deficient_matches_mgcv():
    # A rank-deficient general design (duplicate LP1 covariate) drives
    # gam.fit5's end-stage balanced-Hessian rank check → parameter drop +
    # lpi reindex (gam.fit4.r:1150-1199) — a path no other ported family
    # fires. R: gam(list(y ~ x + xb, ~ s(z), ~ 1), gevlss()) drops the
    # duplicate (rank 13/14, coef 0).
    from hea.family import gevlss

    base = _gevlss_fixture()
    df = base.with_columns(pl.col("x").alias("xb"))
    m = gam(["y ~ x + xb", "~ s(z)", "~ 1"], df, family=gevlss(), method="REML")
    assert m.rank == 13 and len(np.asarray(m._beta)) == 14
    np.testing.assert_allclose(m.REML_criterion / 2, 448.9062817, rtol=0, atol=1e-3)
    beta = np.asarray(m._beta)
    np.testing.assert_allclose(beta[2], 0.0, rtol=0, atol=0)  # dropped
    np.testing.assert_allclose(beta[0], 1.3078499, rtol=0, atol=1e-3)
    np.testing.assert_allclose(beta[1], -1.9194328, rtol=0, atol=1e-3)


def _cox_ph_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(7)
    n = 200
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    lp = 0.7 * np.sin(2 * np.pi * x) + 0.4 * z
    u = gen.uniform(size=n)
    time = -np.log(u) / np.exp(lp)
    ev = gen.uniform(size=n)
    event = (ev < 0.75).astype(float)
    return pl.DataFrame({"time": time, "x": x, "z": z, "event": event})


def test_cox_ph_through_gam_matches_mgcv():
    from hea.family import cox_ph

    df = _cox_ph_fixture()
    m = gam("time ~ s(x) + z", df, family=cox_ph(), weights=df["event"], method="REML")
    assert m.column_names == ["z"] + [f"s(x).{i}" for i in range(1, 10)]
    assert "(Intercept)" not in m.column_names
    np.testing.assert_allclose(
        m.REML_criterion / 2, 619.504025083614, rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.04951670043, rtol=0, atol=1e-2)
    np.testing.assert_allclose(m.sp[0], 0.0888380593715, rtol=2e-3, atol=0)
    np.testing.assert_allclose(m.null_deviance, 251.817485857, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 244.337920299, rtol=0, atol=1e-3)
    beta = np.asarray(m._beta)
    np.testing.assert_allclose(beta[0], 0.5347033442229, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.abs(beta[1:]),
        np.abs(
            [
                1.1918412609676,
                0.9055424491633,
                -0.0820593633819,
                -0.1120411669932,
                -0.0370197861409,
                -0.3200248848103,
                -0.1442647042816,
                0.7227752645372,
                0.5849763355643,
            ]
        ),
        rtol=0,
        atol=2e-3,
    )
    fitted = np.asarray(m.fitted_values)
    assert fitted.shape == (200,)
    np.testing.assert_allclose(
        fitted[:5],
        [
            0.984691862732,
            0.734034958684,
            0.779977961331,
            0.889916922677,
            0.544964819219,
        ],
        rtol=0,
        atol=1e-5,
    )


def test_cox_ph_residuals_predict_match_mgcv():
    # cox.ph deviance/martingale residuals and the survivor-function
    # prediction hook (predict.gam → family$predict, coxph.r:199-245),
    # which needs the new event times from newdata.
    from hea.family import cox_ph

    df = _cox_ph_fixture()
    m = gam("time ~ s(x) + z", df, family=cox_ph(), weights=df["event"], method="REML")
    np.testing.assert_allclose(
        np.asarray(m.residuals)[:5],
        [2.52471540, 0.98282250, 1.13211646, -0.48296411, 0.46087840],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("martingale"))[:5],
        [0.98457348, 0.69080138, 0.75151039, -0.11662717, 0.39296596],
        rtol=0,
        atol=1e-4,
    )
    with pytest.raises(NotImplementedError, match="score/schoenfeld"):
        m.residuals_of("score")
    nd = df[:6].select(["time", "x", "z"])
    pr = m.predict(nd, type="response", se_fit=True)
    np.testing.assert_allclose(
        np.asarray(pr["fit"]),
        [
            0.9846918627,
            0.7340349587,
            0.7799779613,
            0.8899169227,
            0.5449648192,
            0.0735293562,
        ],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(pr["se.fit"]),
        [
            0.00851784061,
            0.05314490011,
            0.05525484119,
            0.03550937447,
            0.06468752039,
            0.04388060327,
        ],
        rtol=0,
        atol=1e-4,
    )


def test_cox_ph_pbc_textbook_matches_mgcv():
    # Wood (2017) GAMs §7.8, "Primary biliary cirrhosis survival
    # analysis" (book p.377-378): the cox.ph fixed-covariates model on
    # the survival::pbc data. Status==2 (death) is the event; transplant
    # /censored are censored. The selected model (after dropping
    # alk.phos, ast, stage) is
    #   gam(time ~ trt + sex + s(sqrt(protime)) + s(platelet) + s(age)
    #       + s(bili) + s(albumin), weights=(status==2), family=cox.ph)
    # Pinned to CURRENT mgcv (1.9.4) on complete cases (n=308 — mgcv's
    # na.omit on the model frame; hea needs weights aligned to the
    # dropped rows, so we filter explicitly). The three near-linear
    # smooths sit on flat λ→∞ ridges (edf≈1, huge disparate sp) so only
    # their edf/Chi²/p are pinned, not sp; s(age)/s(bili) are tight.
    from hea.family import cox_ph

    pbc = load_dataset("survival", "pbc")
    mv = [
        "time",
        "status",
        "trt",
        "sex",
        "protime",
        "platelet",
        "age",
        "bili",
        "albumin",
    ]
    cc = pbc.drop_nulls(subset=mv)
    assert cc.height == 308
    status1 = (cc["status"] == 2).cast(pl.Float64)
    m = gam(
        "time ~ trt + sex + s(sqrt(protime)) + s(platelet) + s(age) "
        "+ s(bili) + s(albumin)",
        cc,
        family=cox_ph(),
        weights=status1,
        method="REML",
    )
    assert "(Intercept)" not in m.column_names
    assert m.column_names[:2] == ["trt", "sexf"]
    np.testing.assert_allclose(
        m.REML_criterion / 2, 547.256744526281, rtol=0, atol=5e-2
    )
    np.testing.assert_allclose(float(np.sum(m.edf)), 15.3095029449, rtol=0, atol=5e-2)
    np.testing.assert_allclose(m.null_deviance, 413.412618165, rtol=0, atol=1e-2)
    beta = dict(zip(m.column_names, np.asarray(m._beta)))
    np.testing.assert_allclose(beta["trt"], 0.06715634, rtol=0, atol=2e-3)
    np.testing.assert_allclose(beta["sexf"], -0.49515759, rtol=0, atol=3e-3)
    rows = {r[0]: r for r in m._smooth_significance_rows()}
    np.testing.assert_allclose(rows["s(age)"][1], 6.042792, rtol=0, atol=2e-2)
    np.testing.assert_allclose(rows["s(age)"][3], 29.417277, rtol=0, atol=0.3)
    np.testing.assert_allclose(rows["s(bili)"][1], 4.264429, rtol=0, atol=2e-2)
    np.testing.assert_allclose(rows["s(bili)"][3], 89.540337, rtol=0, atol=0.5)
    for lab, chi in (
        ("s(sqrt(protime))", 13.333751),
        ("s(platelet)", 5.787376),
        ("s(albumin)", 31.086251),
    ):
        np.testing.assert_allclose(rows[lab][1], 1.0, rtol=0, atol=2e-2)
        np.testing.assert_allclose(rows[lab][3], chi, rtol=0, atol=0.3)
    for lab in rows:
        assert 0.0 <= rows[lab][4] <= 1.0
    m.summary()
    assert "sqrt(protime)" in m.data.columns
    fig = m.plot_smooth()
    assert len(fig.axes) == 5
    plt.close(fig)


def _ziplss_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(31)
    n = 300
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    gamma = 0.6 + np.sin(2 * np.pi * x)
    eta = 1.2 * np.sin(2 * np.pi * z)
    p = 1.0 - np.exp(-np.exp(eta))
    y = gen.poisson(np.exp(gamma)) * (gen.uniform(size=n) < p)
    return pl.DataFrame({"y": y.astype(float), "x": x, "z": z})


def test_ziplss_through_gam_matches_mgcv():
    from hea.family import ziplss

    df = _ziplss_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=ziplss(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 424.2451262, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp, [0.1515432173, 0.1131497668], rtol=1e-4)
    np.testing.assert_allclose(m.edf_total, 11.14924342, rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.6705247844, -1.655109582, 0.3424055272, -0.0520844202]),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.4844884226, 0.2271978785], [0.3148140516, -0.2381554759]],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(m.deviance, 500.745984, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 674.6005762, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.006223781606, rtol=0, atol=1e-6
    )

    # family$predict hook: type="response" returns the single column
    # E(y) = p·μ (mgcv emits one fit column for ziplss, not n_lp), with a
    # delta-method SE — note mgcv reuses gamma's variance for the eta term
    # (gamlss.r:1718), reproduced bug-for-bug so the SE matches predict.gam.
    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [1.445763757, 1.001536501, 2.231325171], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.1624539204, 0.2429565392, 0.2415688359],
        rtol=0,
        atol=1e-5,
    )
    pr_all = m.predict(type="response")
    np.testing.assert_allclose(
        pr_all["fit"].to_numpy()[:3],
        [1.445763757, 1.001536501, 2.231325171],
        rtol=0,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [-1.584347308, -1.255452233, 0.9717809043, -1.168783967, -1.423877013],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        m.residuals_of("response")[:5],
        [-1.445763757, -1.001536501, 1.768674829, -1.500289883, -0.3922964182],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(np.sum(m.residuals_of("deviance") ** 2)), m.deviance, rtol=0, atol=1e-9
    )
    m.summary()


def _multinom_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(33)
    n = 400
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    eta1 = 1.4 * np.sin(2 * np.pi * x)
    eta2 = 1.2 * np.cos(2 * np.pi * z) - 0.3
    ee = np.exp(np.column_stack([np.zeros(n), eta1, eta2]))
    P = ee / ee.sum(axis=1, keepdims=True)
    u = gen.uniform(0, 1, n)
    y = (np.cumsum(P, axis=1) > u[:, None]).argmax(axis=1).astype(float)
    return pl.DataFrame({"y": y, "x": x, "z": z})


def test_multinom_through_gam_matches_mgcv():
    from hea.family import multinom

    df = _multinom_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=multinom(2), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 252.9000533, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp, [0.05569991114, 0.1433165651], rtol=1e-4)
    np.testing.assert_allclose(m.edf_total, 10.00551406, rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([-0.01450008726, 2.368136713, 0.7424680446, 0.4313548243]),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.07480023513, 0.02199777687], [0.3881200015, -1.508606099]],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(m.deviance, 762.0647414, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 871.2817292, rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.01566970018, rtol=0, atol=1e-6)

    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(
        [pr["fit"][0], pr["fit.1"][0], pr["fit.2"][0]],
        [0.3225899731, 0.347645165, 0.3297648618],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        [pr["se.fit"][0], pr["se.fit.1"][0], pr["se.fit.2"][0]],
        [0.03781883063, 0.05422039848, 0.0523534205],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pr["fit"].to_numpy() + pr["fit.1"].to_numpy() + pr["fit.2"].to_numpy(),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [1.453666372, -1.408229965, -1.357551087, 1.078630867, 1.19190966],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        float(np.sum(m.residuals_of("deviance") ** 2)), m.deviance, rtol=0, atol=1e-9
    )
    m.summary()


def _mvn_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(52)
    n = 200
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    e1 = gen.normal(size=n)
    e2 = gen.normal(size=n)
    y1 = 2 * np.sin(np.pi * x) + e1
    y2 = 0.6 * np.sin(np.pi * x) + np.exp(1.5 * z) - 2 + 0.5 * e1 + 0.8 * e2
    return pl.DataFrame({"y1": y1, "y2": y2, "x": x, "z": z})


def test_mvn_through_gam_matches_mgcv():
    from hea.family import mvn

    df = _mvn_fixture()
    m = gam(["y1 ~ s(x)", "y2 ~ s(z)"], df, family=mvn(d=2))
    assert m.method == "REML"
    np.testing.assert_allclose(m.REML_criterion / 2, 205.7441216777, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.2006855377, 3.4818966647], rtol=2e-3)
    np.testing.assert_allclose(m.edf_total, 11.51223604, rtol=0, atol=2e-3)
    np.testing.assert_allclose(m.deviance, 400.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.null_deviance, 627.91988172, rtol=0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(m._beta)[-3:],
        [0.12714762, -0.57915894, -0.04044921],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[[0, 1, 10, 11]]),
        np.abs([1.249452, 0.033666, 0.559357, 0.115879]),
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.826858, 1.016451], [0.453129, -0.089196]],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00528745, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m.Vp)[22, 22], 0.00251721, rtol=0, atol=1e-6)
    assert m.outer_info["iter"] == 4
    assert m.outer_info["conv"] == "full convergence"

    pr = m.predict(df[:2], se_fit=True, type="response")
    np.testing.assert_allclose(
        [pr["fit"][0], pr["fit.1"][0]], [0.826858, 1.016451], rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        [pr["se.fit"][0], pr["se.fit.1"][0], pr["se.fit"][1], pr["se.fit.1"][1]],
        [0.140909, 0.103364, 0.182820, 0.105579],
        rtol=0,
        atol=1e-4,
    )

    rr = np.asarray(m.residuals_of("deviance"))
    assert rr.shape == (200, 2)
    np.testing.assert_allclose(float(np.sum(rr**2)), m.deviance, rtol=0, atol=1e-8)
    m.summary()


def test_shash_through_gam_matches_mgcv():
    from hea.family import _r_tweedie, shash  # noqa: F401
    from hea.R.rng import RGenerator

    gen = RGenerator(21)
    n = 400
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    mu_t = 1.0 + np.sin(2 * np.pi * x)
    sig_t = np.exp(-0.4 + 0.5 * z)
    u = gen.normal(0, 1, n)
    y = mu_t + sig_t * np.sinh(np.arcsinh(u) + 0.4)
    df = pl.DataFrame({"y": y, "x": x, "z": z})

    m1 = gam(["y ~ s(x)", "~ s(z)", "~ 1", "~ 1"], df, family=shash(), method="REML")
    np.testing.assert_allclose(m1.REML_criterion / 2, 546.1724511286, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m1.sp[0], 0.1910175024, rtol=1e-4)
    np.testing.assert_allclose(m1.sp[1], 25838.09103, rtol=2e-2)
    np.testing.assert_allclose(m1.edf_total, 10.2633615130, rtol=0, atol=5e-3)
    b = np.asarray(m1._beta)
    np.testing.assert_allclose(b[0], 1.0060261008, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.abs(b[1:3]), np.abs([1.5298088420, -0.1395345089]), rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(
        [b[20], b[21]], [0.4312252846, -0.0635499741], rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[0],
        [0.2022640409, -0.0442752319, 0.4312252846, -0.0635499741],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(m1.deviance, 1051.49370326, rtol=1e-4)
    assert np.isnan(m1.null_deviance)  # mgcv: NULL (no postproc)
    np.testing.assert_allclose(
        np.asarray(m1.residuals)[:3],
        [-1.7047382825, -1.2148037602, 1.9049698110],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m1.Vp)[0, 0], 0.007045588271, rtol=0, atol=1e-5
    )
    assert 2 <= m1.outer_info["iter"] <= 8  # R: 3
    qq = m1._qq_gam_quantiles(type="deviance", rep=0, s_rep=2, seed=1)
    assert qq["Dq"] is not None and np.all(np.isfinite(qq["Dq"]))
    qq2 = m1._qq_gam_quantiles(type="deviance", rep=3, level=0, seed=1)
    assert qq2["Dq"] is not None and np.all(np.isfinite(qq2["Dq"]))
    m1.summary()
    pred = m1.predict(df[:3])
    np.testing.assert_allclose(
        pred["fit"].to_numpy(), np.asarray(m1.fitted_values)[:3, 0], rtol=0, atol=1e-10
    )

    # mgcv's shash ll rejects offsets outright (gamlss.r:3470)
    with pytest.raises(NotImplementedError, match="offset not still"):
        gam(
            ["y ~ s(x) + offset(z)", "~ s(z)", "~ 1", "~ 1"],
            df,
            family=shash(),
            method="REML",
        )

    m2 = gam(
        ["y ~ s(x)", "~ s(z)", "~ 1", "~ 1"],
        df,
        family=shash(),
        method="REML",
        optimizer="efs",
    )
    np.testing.assert_allclose(m2.REML_criterion / 2, 546.1756937219, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        m2.sp, [0.18448478, 2613.7345], rtol=1e-2
    )  # s(z) flattish efs ridge
    np.testing.assert_allclose(m2.edf_total, 10.31418966, rtol=0, atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(m2.fitted_values)[0],
        [0.2018576699, -0.0445263220, 0.4309091585, -0.0641118029],
        rtol=0,
        atol=1e-3,
    )


def test_predict_unconditional_se_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    p = m.predict(df[:3], se_fit=True)
    pu = m.predict(df[:3], se_fit=True, unconditional=True)
    np.testing.assert_allclose(
        p["se.fit"].to_numpy(), [0.0774989332, 0.1475501829, 0.0806144988], rtol=1e-6
    )
    np.testing.assert_allclose(
        pu["se.fit"].to_numpy(), [0.0777513533, 0.1479764183, 0.0811005933], rtol=1e-6
    )
    mg = gam("ygau ~ f4 + z + s(x)", df, method="GCV.Cp")
    with pytest.warns(UserWarning, match="not available"):
        pg = mg.predict(df[:3], se_fit=True, unconditional=True)
    np.testing.assert_array_equal(
        pg["se.fit"].to_numpy(), mg.predict(df[:3], se_fit=True)["se.fit"].to_numpy()
    )


# ---------------------------------------------------------------------------
# predict type="terms"/"iterms" + terms=/exclude= — mgcv 1.9-4
# pins from predict.gam on the CSV-identical fixture, newdata = rows 1:6.
# terms=/exclude= zero the de-selected terms' design columns for EVERY type
# (mgcv.r:2993-3026); for type="terms"/"iterms" a trailing column selection
# applies with warn-and-ignore semantics (mgcv.r:3257-3284). iterms differs
# from terms only in constrained smooths' SEs (cmX "carry the intercept",
# mgcv.r:3072-3081).
# ---------------------------------------------------------------------------


def test_predict_terms_and_iterms_match_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    nd = df.head(6)
    pt = m.predict(nd, type="terms")
    assert pt.columns == ["f4", "z", "s(x)"]
    np.testing.assert_allclose(
        pt.to_numpy(),
        [
            [0.0000000000, 0.2377973618, 0.8729497169],
            [0.7342117915, 0.2155591008, -0.0446679116],
            [0.2712112896, 0.4304817651, -0.1832820318],
            [-0.3299837116, 0.1456983896, 0.0241175461],
            [0.7342117915, 0.5232056834, 0.2809888679],
            [0.7342117915, 0.6593732328, -0.4352664652],
        ],
        atol=1e-7,
    )
    pts = m.predict(nd, type="terms", se_fit=True)
    assert pts.columns == ["f4", "z", "s(x)", "se.f4", "se.z", "se.s(x)"]
    np.testing.assert_allclose(
        pts.to_numpy()[:, 3:],
        [
            [0.0000000000, 0.0308069966, 0.0530786832],
            [0.0738603615, 0.0279259973, 0.1364511756],
            [0.0750765806, 0.0557695434, 0.0604690770],
            [0.0734874123, 0.0188754398, 0.1154736067],
            [0.0738603615, 0.0677820628, 0.0698835544],
            [0.0738603615, 0.0854227684, 0.0966170141],
        ],
        atol=1e-7,
    )
    pti = m.predict(nd, type="iterms", se_fit=True)
    np.testing.assert_allclose(pti.to_numpy()[:, :3], pt.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        pti.to_numpy()[:, 3:],
        [
            [0.0000000000, 0.0308069966, 0.0587771513],
            [0.0738603615, 0.0279259973, 0.1387671800],
            [0.0750765806, 0.0557695434, 0.0655279801],
            [0.0734874123, 0.0188754398, 0.1182013568],
            [0.0738603615, 0.0677820628, 0.0743042265],
            [0.0738603615, 0.0854227684, 0.0998611753],
        ],
        atol=1e-7,
    )
    pti2 = m.predict(nd, type="iterms", se_fit=True, iterms_type=2)
    np.testing.assert_allclose(pti2.to_numpy(), pti.to_numpy(), atol=1e-10)


def test_predict_terms_select_exclude_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    nd = df.head(6)
    sx = [
        0.8729497169,
        -0.0446679116,
        -0.1832820318,
        0.0241175461,
        0.2809888679,
        -0.4352664652,
    ]
    zc = [
        0.2377973618,
        0.2155591008,
        0.4304817651,
        0.1456983896,
        0.5232056834,
        0.6593732328,
    ]
    sel = m.predict(nd, type="terms", terms="s(x)")
    assert sel.columns == ["s(x)"]
    np.testing.assert_allclose(sel["s(x)"].to_numpy(), sx, atol=1e-7)
    selp = m.predict(nd, type="terms", terms=["z", "s(x)"])
    assert selp.columns == ["z", "s(x)"]
    np.testing.assert_allclose(selp.to_numpy(), np.column_stack([zc, sx]), atol=1e-7)
    exc = m.predict(nd, type="terms", exclude="f4")
    assert exc.columns == ["z", "s(x)"]
    np.testing.assert_allclose(exc.to_numpy(), np.column_stack([zc, sx]), atol=1e-7)
    with pytest.warns(UserWarning, match="non-existent terms"):
        wt = m.predict(nd, type="terms", terms="nope")
    assert wt.columns == ["f4", "z", "s(x)"]
    assert float(np.abs(wt.to_numpy()).max()) == 0.0
    with pytest.warns(UserWarning, match="non-existent exclude"):
        we = m.predict(nd, type="terms", exclude="nope")
    np.testing.assert_allclose(
        we.to_numpy(), m.predict(nd, type="terms").to_numpy(), rtol=1e-12
    )


def test_predict_link_response_terms_exclude_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    nd = df.head(6)
    le = m.predict(nd, type="link", exclude="s(x)", se_fit=True)
    np.testing.assert_allclose(
        le["fit"].to_numpy(),
        [
            0.5736263216,
            1.2855998521,
            1.0375220145,
            0.1515436379,
            1.5932464347,
            1.7294139842,
        ],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        le["se.fit"].to_numpy(),
        [
            0.0569307847,
            0.0536362256,
            0.0533644748,
            0.0587270876,
            0.0545917647,
            0.0635634911,
        ],
        atol=1e-7,
    )
    re_ = m.predict(nd, type="response", exclude="s(x)")
    np.testing.assert_allclose(re_["fit"].to_numpy(), le["fit"].to_numpy(), rtol=1e-12)
    ni = m.predict(nd, type="link", exclude="(Intercept)")
    np.testing.assert_allclose(
        ni["fit"].to_numpy(),
        [
            1.1107470787,
            0.9051029807,
            0.5184110229,
            -0.1601677758,
            1.5384063428,
            0.9583185591,
        ],
        atol=1e-7,
    )
    lt = m.predict(nd, type="link", terms="s(x)")
    np.testing.assert_allclose(
        lt["fit"].to_numpy(),
        [
            0.8729497169,
            -0.0446679116,
            -0.1832820318,
            0.0241175461,
            0.2809888679,
            -0.4352664652,
        ],
        atol=1e-7,
    )
    lz = m.predict(nd, type="link", terms="z")
    np.testing.assert_allclose(
        lz["fit"].to_numpy(),
        [
            0.2377973618,
            0.2155591008,
            0.4304817651,
            0.1456983896,
            0.5232056834,
            0.6593732328,
        ],
        atol=1e-7,
    )


def test_predict_iterms_unconstrained_smooth_fallback():
    # s(g5, bs="re") has no absorbed constraint (nCons == 0): its iterms
    # SE must equal its terms SE (mgcv.r:3072 gate), while s(x)'s widens.
    # The re component fits to σ ≈ 0 here — its λ stops on a flat REML
    # boundary (hea 763382 vs R 375898 with REML values 2.5e-6 apart), so
    # the s(g5) columns themselves are boundary noise and only their
    # structure is asserted; everything else pins tight.
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x) + s(g5, bs='re')", df, method="REML")
    nd = df.head(6)
    pt = m.predict(nd, type="terms", se_fit=True)
    pi_ = m.predict(nd, type="iterms", se_fit=True)
    assert pt.columns[:4] == ["f4", "z", "s(x)", "s(g5)"]
    np.testing.assert_array_equal(pi_["se.s(g5)"].to_numpy(), pt["se.s(g5)"].to_numpy())
    assert float(np.abs(pt["s(g5)"].to_numpy()).max()) < 1e-4
    np.testing.assert_allclose(
        pt["se.s(x)"].to_numpy(),
        [
            0.0530787002,
            0.1364512782,
            0.0604690950,
            0.1154736894,
            0.0698835753,
            0.0966170952,
        ],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        pi_["se.s(x)"].to_numpy(),
        [
            0.0587771673,
            0.1387672811,
            0.0655279973,
            0.1182014378,
            0.0743042467,
            0.0998612540,
        ],
        rtol=1e-4,
    )


def test_predict_terms_offset_poisson_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ypois ~ z + s(x) + offset(log(lo))", df, family=Poisson(), method="REML")
    nd = df.head(6)
    o3 = m.predict(nd, type="link", exclude="s(x)")
    np.testing.assert_allclose(
        o3["fit"].to_numpy(),
        [
            0.7647704612,
            1.1728571194,
            0.4727620017,
            1.1699203351,
            1.1000742402,
            1.6116984342,
        ],
        atol=1e-7,
    )
    t3 = m.predict(nd, type="terms")
    assert t3.columns == ["z", "s(x)"]
    np.testing.assert_allclose(
        t3.to_numpy()[0], [0.2172354052, 0.9256322997], atol=1e-7
    )


def test_predict_terms_multi_lp_gaulss_matches_mgcv():
    from hea.family import gaulss

    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(), method="REML")
    nd = df.head(6)
    t4 = m.predict(nd, type="terms")
    assert t4.columns == ["w", "s(x)", "s.1(z)"]
    np.testing.assert_allclose(
        t4.to_numpy(),
        [
            [0.26576084, 0.71070807, -0.21136924],
            [0.23860749, -0.88387867, 0.07613941],
            [0.43708020, 0.63567395, -0.52135388],
            [0.31992156, 0.87529617, -0.86759606],
            [0.07666149, -0.57645871, -0.28874762],
            [0.27264823, -0.58786727, -0.34349805],
        ],
        atol=1e-6,
    )
    with pytest.warns(UserWarning, match="iterms not available"):
        i4 = m.predict(nd, type="iterms")
    np.testing.assert_array_equal(i4.to_numpy(), t4.to_numpy())
    e4 = m.predict(nd, type="link", exclude="s.1(z)", se_fit=True)
    np.testing.assert_allclose(
        e4["fit"].to_numpy()[:2], [1.46270992, -0.15903017], atol=1e-6
    )
    np.testing.assert_allclose(
        e4["fit.1"].to_numpy()[:2], [-0.58269247, -0.58269247], atol=1e-6
    )
    np.testing.assert_allclose(
        e4["se.fit"].to_numpy()[:2], [0.06626794, 0.06894145], atol=1e-6
    )
    np.testing.assert_allclose(
        e4["se.fit.1"].to_numpy()[:2], [0.04918413, 0.04918413], atol=1e-6
    )
    r4 = m.predict(nd, type="response", exclude="s(x)", se_fit=True)
    np.testing.assert_allclose(
        r4["fit"].to_numpy()[:2], [0.75200185, 0.72484850], atol=1e-6
    )
    np.testing.assert_allclose(
        r4["fit.1"].to_numpy()[:2], [2.16447805, 1.63246911], atol=1e-6
    )
    np.testing.assert_allclose(
        r4["se.fit"].to_numpy()[:2], [0.02961727, 0.02850520], atol=1e-6
    )
    np.testing.assert_allclose(
        r4["se.fit.1"].to_numpy()[:2], [0.21529159, 0.17505347], atol=1e-6
    )


def test_predict_na_action_block_size_matches_mgcv():
    """predict(na_action=, block_size=, newdata_guaranteed=) — the
    predict.gam newdata-stage args (mgcv.r:2692-2830 + the napredict
    round trip). na.pass (default) evaluates complete rows and returns
    NaN rows at NA positions — including NaN lpmatrix rows
    (mgcv.r:3303); na.omit/na.exclude drop the rows; na.fail raises;
    None skips NA processing. Response-column NAs are tolerated (mgcv's
    naresp dance). block.size chunks the design build (results equal to
    BLAS rounding, as in R); newdata.guaranteed skips the processing.
    mgcv 1.9-4 pins on the seed-8 recipe.
    """
    from hea.models.bam import bam as _bam
    from hea.R.rng import RGenerator

    g = RGenerator(8)
    n = 120
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.5 * z + g.normal(0, 1, n) * 0.3
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    m = gam("y ~ s(x) + z", df, method="REML")

    nd = pl.DataFrame({"x": [0.1, None, 0.5, 0.9], "z": [0.2, 0.3, None, 0.6]})
    p1 = m.predict(nd, type="link", se_fit=True)  # na.pass
    np.testing.assert_allclose(
        p1["fit"].to_numpy()[[0, 3]], [0.6967545055, -0.2952746326], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        p1["se.fit"].to_numpy()[[0, 3]], [0.0848671384, 0.0713200030], rtol=0, atol=1e-8
    )
    assert np.isnan(p1["fit"].to_numpy()[[1, 2]]).all()
    assert np.isnan(p1["se.fit"].to_numpy()[[1, 2]]).all()
    p2 = m.predict(nd, type="link", na_action="na.omit")
    assert p2.height == 2
    np.testing.assert_allclose(
        p2["fit"].to_numpy(), [0.6967545055, -0.2952746326], rtol=0, atol=1e-8
    )
    with pytest.raises(ValueError, match="na_action='na.fail'"):
        m.predict(nd, type="link", na_action="na.fail")
    with pytest.raises(ValueError, match="unrecognised na.action"):
        m.predict(nd, type="link", na_action="bogus")
    # lpmatrix gets NaN rows too (mgcv napredicts H, mgcv.r:3303)
    H = m.predict(nd, type="lpmatrix")
    assert H.shape == (4, 11)
    assert np.isnan(H[[1, 2]]).all() and not np.isnan(H[[0, 3]]).any()
    t1 = m.predict(nd, type="terms")
    assert np.isnan(t1.to_numpy()[[1, 2]]).all()
    assert not np.isnan(t1.to_numpy()[[0, 3]]).any()

    nd2 = pl.DataFrame({"x": g.uniform(0, 1, 50), "z": g.uniform(0, 1, 50)})
    a = m.predict(nd2, type="link", se_fit=True)
    b = m.predict(nd2, type="link", se_fit=True, block_size=7)
    np.testing.assert_allclose(a["fit"].to_numpy(), b["fit"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        a["se.fit"].to_numpy(), b["se.fit"].to_numpy(), rtol=1e-10
    )
    c = m.predict(nd2, type="link", newdata_guaranteed=True)
    np.testing.assert_allclose(a["fit"].to_numpy(), c["fit"].to_numpy(), rtol=0, atol=0)
    d = m.predict(nd2, type="link", na_action=None)
    np.testing.assert_allclose(a["fit"].to_numpy(), d["fit"].to_numpy(), rtol=0, atol=0)

    mb = _bam("y ~ s(x) + z", df, discrete=True)
    pb = mb.predict(nd, type="link")
    assert np.isnan(pb["fit"].to_numpy()[[1, 2]]).all()
    pb2 = mb.predict(nd, type="response", na_action="na.omit")
    assert pb2.height == 2


def test_gam_min_sp_matches_mgcv():
    """gam(min.sp=)/gam(H=) — the fixed additive penalty (mgcv.r:1465-
    1508). min.sp[k] folds a FIXED block min.sp[k]·S_k into H, lower-
    bounding smooth k's EFFECTIVE penalty; the estimated sp stays free so
    the reported sp is only the estimated part (it can fall below min.sp).
    The fixed penalty enters the fit AND — as an extra un-parameterised
    square root — the (RE)ML log|Sλ|₊ (reparam fixed_penalty root) / the
    magic GCV St (mgcv passes G$H to magic, mgcv.r:2620). Since
    Σ min.sp[k]·S_k has range ⊆ span{S_k}, Mp is unchanged. mgcv 1.9-4
    pins (set.seed(7) frame; min.sp=c(0.05, 0.5)).
    """
    from hea.R.rng import RGenerator

    g = RGenerator(7)
    n = 120
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.5 * z**2 + g.normal(0, 0.3, n)
    d = {"x": x, "z": z, "y": y}

    m0 = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML")
    m1 = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", min_sp=[0.05, 0.5])
    assert m1._Mp == m0._Mp
    assert m1._UrS is not None and len(m1._UrS) == 3  # 2 S-roots + 1 H-root
    assert m1.sp[0] < 1e-5
    np.testing.assert_allclose(m1.sp[1], 1.298407176, rtol=1e-4)
    np.testing.assert_allclose(m1.REML_criterion / 2.0, 40.7634154362, rtol=1e-4)
    np.testing.assert_allclose(m1.edf_total, 8.138304573, rtol=1e-5)
    np.testing.assert_allclose(m1.scale, 0.08287281392, rtol=1e-5)
    np.testing.assert_allclose(m1.Vp[0, 0], 0.0006906067827, rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[:3],
        [-0.04136038139, 0.732504374, 0.753032283],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(m1.bhat).reshape(-1)[0], 0.2152093634, rtol=1e-4
    )

    mg = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="GCV.Cp", min_sp=[0.05, 0.5])
    assert mg.sp[0] < 1e-6  # floored to ~0
    np.testing.assert_allclose(mg.sp[1], 1.8661594, rtol=1e-4)
    np.testing.assert_allclose(mg.edf_total, 8.0031678, rtol=1e-5)
    np.testing.assert_allclose(mg.scale, 0.0829456439, rtol=1e-5)

    Hmat = np.asarray(m1._H_fixed)
    mh = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", H=Hmat)
    np.testing.assert_allclose(mh.REML_criterion, m1.REML_criterion, rtol=0, atol=1e-9)
    np.testing.assert_allclose(mh.edf_total, m1.edf_total, rtol=0, atol=1e-9)

    # ---- validation (mgcv.r:1466-1469) --------------------------------
    with pytest.raises(ValueError, match="length of min.sp is wrong"):
        gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", min_sp=[0.05])
    with pytest.raises(ValueError, match="must be non negative"):
        gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", min_sp=[0.1, -0.2])
    with pytest.raises(ValueError, match="NA's in min.sp"):
        gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", min_sp=[np.nan, 0.1])
    with pytest.raises(ValueError, match="H has wrong dimension"):
        gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", H=np.eye(5))


def test_gam_parapen_matches_mgcv():
    """gam(paraPen=) — estimated penalties on parametric terms (mgcv's
    parametricPenalty, mgcv.r:767-836, merged mgcv.r:1180-1509). Each
    penalty is PREPENDED to the smooth penalties with its own working sp
    (reported sp is paraPen-first, matching mgcv), is ESTIMATED (so it
    enters Mp — the null space shrinks by its rank — and the reparam UrS
    like a smooth), and folds through every rail. Pins: mgcv 1.9-4,
    set.seed(11) frame.
    """
    import polars as pl

    from hea.R.rng import RGenerator

    g = RGenerator(11)
    n = 200
    x0 = g.uniform(0, 1, n)
    x1 = g.uniform(0, 1, n)
    x2 = g.uniform(0, 1, n)
    y = (
        2 * np.sin(np.pi * x0)
        + 0.5 * x1
        + 0.3 * x2
        - 0.4 * x1 * x2
        + g.normal(0, 1, n) * 0.5
    )
    d = {"y": y, "x0": x0, "x1": x1, "x2": x2}

    mA = gam(
        "y ~ s(x0) + x1 + x2",
        d,
        method="REML",
        paraPen={"x1": {"S": np.array([[1.0]])}},
    )
    assert mA._Mp == 3
    assert len(mA._slots) == 2
    assert [s.block.label for s in mA._slots] == ["x1", "s(x0)"]
    np.testing.assert_allclose(
        np.asarray(mA.sp), [1.686864106, 0.06781100753], rtol=1e-4
    )
    np.testing.assert_allclose(mA.edf_total, 7.873517, rtol=1e-5)
    np.testing.assert_allclose(mA.scale, 0.274444, rtol=1e-5)
    np.testing.assert_allclose(mA.REML_criterion / 2.0, 165.024291013, rtol=1e-6)
    np.testing.assert_allclose(
        np.diag(np.asarray(mA.Vp))[:3],
        [0.009836403186, 0.01655395247, 0.01721800529],
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(mA.coefficients)[:3], [1.3446392, 0.38228369, 0.24055995], rtol=1e-4
    )
    pe = mA.pen_edf()
    assert list(pe["name"]) == ["s(x0)"]
    np.testing.assert_allclose(pe["edf"][0], 4.97526516, rtol=1e-4)

    fac = np.array([str(1 + (i % 3)) for i in range(n)])
    df = pl.DataFrame({"y": y, "x0": x0, "fac": fac})
    mF = gam("y ~ s(x0) + fac", df, method="REML", paraPen={"fac": {"S": np.eye(2)}})
    assert mF._Mp == 2  # both fac dummies penalized
    np.testing.assert_allclose(
        np.asarray(mF.sp), [534210.5834, 0.07204428643], rtol=1e-3
    )
    np.testing.assert_allclose(mF.edf_total, 5.924066791, rtol=1e-5)
    np.testing.assert_allclose(mF.scale, 0.2906750491, rtol=1e-5)
    np.testing.assert_allclose(mF.REML_criterion / 2.0, 168.837993008, rtol=1e-6)

    # ---- CASE C: a FIXED paraPen sp folds out of the estimated vector --
    # paraPen sp=5 is folded into lsp0 (= log 5) and excluded from the
    # reported sp; full.sp still carries it (mgcv.r:1515-1531).
    mC = gam(
        "y ~ s(x0) + x1 + x2",
        d,
        method="REML",
        paraPen={"x1": {"S": np.array([[1.0]]), "sp": 5.0}},
    )
    assert mC._n_work == 1  # only s(x0) is estimated
    np.testing.assert_allclose(np.asarray(mC._lsp0)[0], np.log(5.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mC.sp), [0.06917407842], rtol=1e-4)
    np.testing.assert_allclose(np.asarray(mC.full_sp), [5.0, 0.06917407842], rtol=1e-4)
    np.testing.assert_allclose(mC.edf_total, 7.703656498, rtol=1e-5)

    with pytest.raises(ValueError, match="wrong dimension"):
        gam(
            "y ~ s(x0) + x1", d, method="REML", paraPen={"x1": {"S": np.eye(2)}}
        )  # 2×2 on a 1-col term
    with pytest.raises(ValueError, match="rank' has wrong length"):
        gam(
            "y ~ s(x0) + x1",
            d,
            method="REML",
            paraPen={"x1": {"S": np.array([[1.0]]), "rank": [1, 2]}},
        )
    with pytest.raises(ValueError, match="sp' dimension wrong"):
        gam(
            "y ~ s(x0) + x1",
            d,
            method="REML",
            paraPen={"x1": {"S": np.array([[1.0]]), "sp": [1.0, 2.0]}},
        )
    with pytest.raises(ValueError, match="not matched to parametric"):
        gam(
            "y ~ s(x0) + x1",
            d,
            method="REML",
            paraPen={"zzz": {"S": np.array([[1.0]])}},
        )


def test_gam_fit_false_and_G_reuse():
    """gam(fit=FALSE) returns an unfitted 'prefit' carrying the full setup
    (mgcv.r:2384-2387, class gam.prefit); gam(G=prefit) skips the whole
    basis/penalty construction and runs only estimate.gam on it
    (mgcv.r:2267). Since setup is fit-independent, the G= refit must be
    bit-identical to the direct fit. Single-LP dense only.
    """
    from hea.family import gaulss
    from hea.R.rng import RGenerator

    g = RGenerator(7)
    n = 120
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.5 * z**2 + g.normal(0, 0.3, n)
    d = {"x": x, "z": z, "y": y}

    m = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML")
    G = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="REML", fit=False)
    assert G._is_prefit
    assert G._Mp == m._Mp and len(G._slots) == len(m._slots)
    assert G._X_full.shape == m._X_full.shape
    m2 = gam(None, None, G=G)
    assert not m2._is_prefit
    np.testing.assert_allclose(m2.sp, m.sp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m2.coefficients, m.coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m2.Vp, m.Vp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m2.edf_total, m.edf_total, rtol=0, atol=1e-12)

    a1 = gam(
        "y ~ s(x,k=10) + z", d, method="REML", paraPen={"z": {"S": np.array([[1.0]])}}
    )
    b1 = gam(
        None,
        None,
        G=gam(
            "y ~ s(x,k=10) + z",
            d,
            method="REML",
            paraPen={"z": {"S": np.array([[1.0]])}},
            fit=False,
        ),
    )
    np.testing.assert_allclose(b1.coefficients, a1.coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b1.sp, a1.sp, rtol=0, atol=1e-12)
    a2 = gam("y ~ s(x,k=10) + s(z,k=10)", d, method="GCV.Cp")
    b2 = gam(
        None, None, G=gam("y ~ s(x,k=10) + s(z,k=10)", d, method="GCV.Cp", fit=False)
    )
    np.testing.assert_allclose(b2.coefficients, a2.coefficients, rtol=0, atol=1e-12)

    with pytest.raises(NotImplementedError, match="single-formula dense"):
        gam(["y ~ s(x)", "~ s(z)"], d, family=gaulss(), fit=False)
    with pytest.raises(ValueError, match="prefit"):
        gam(None, None, G=m)  # m is fitted, not a prefit


def test_gam_in_out_and_drop_intercept_match_mgcv():
    """gam(in.out=) — the estimate.gam warm start (mgcv.r:2005-2010,
    2028-2032): lsp seeds at log(in.out$sp) (WORKING length, no
    projection) and log scale at log(in.out$scale); both entries are
    validated unconditionally with mgcv's stop. gam(drop.intercept=) —
    mgcv.r:1163-1171: the parametric matrix keeps its factor contrast
    coding but the assign==0 column is deleted. mgcv 1.9-4 pins.
    """
    from hea.family import gaulss
    from hea.R.rng import RGenerator

    g = RGenerator(8)
    n = 120
    x = g.uniform(0, 1, n)
    z = g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.5 * z + g.normal(0, 1, n) * 0.3
    df = pl.DataFrame({"x": x, "z": z, "y": y})

    g1 = gam("y ~ s(x) + z", df, method="REML")
    g2 = gam(
        "y ~ s(x) + z",
        df,
        method="REML",
        in_out={"sp": g1.sp * 4, "scale": g1.sigma_squared * 2},
    )
    np.testing.assert_allclose(g2.sp, g1.sp, rtol=1e-5)
    np.testing.assert_allclose(g2.REML_criterion, g1.REML_criterion, rtol=0, atol=1e-7)
    np.testing.assert_allclose(g1.sp, [0.008239710144], rtol=1e-4)
    for bad in (
        {"sp": [1.0]},  # no scale
        {"sp": [1.0, 2.0], "scale": 1.0},
    ):  # wrong length
        with pytest.raises(ValueError, match="in.out incorrect"):
            gam("y ~ s(x) + z", df, method="REML", in_out=bad)

    df5 = _fit5_fixture()
    m0 = gam(["y ~ s(x) + w", "~ s(z)"], df5, family=gaulss(), method="REML")
    m1 = gam(
        ["y ~ s(x) + w", "~ s(z)"],
        df5,
        family=gaulss(),
        method="REML",
        in_out={"sp": m0.sp * 3, "scale": 1.0},
    )
    np.testing.assert_allclose(m1.sp, m0.sp, rtol=1e-5)
    np.testing.assert_allclose(m1.REML_criterion, m0.REML_criterion, rtol=0, atol=1e-6)

    g2 = RGenerator(8)
    x2 = g2.uniform(0, 1, n)
    f2 = pl.Series("f", ["a", "b", "c"] * 40, dtype=pl.Enum(["a", "b", "c"]))
    eff = np.array([0.0, 0.5, -0.3])[np.tile([0, 1, 2], 40)]
    y2 = np.sin(2 * np.pi * x2) + eff + g2.normal(0, 1, n) * 0.3
    df2 = pl.DataFrame({"x": x2, "f": f2, "y": y2})
    m = gam("y ~ f + s(x)", df2, method="REML", drop_intercept=True)
    assert m.p == 11 and m.column_names[:2] == ["fb", "fc"]
    np.testing.assert_allclose(m.sp, [0.006473414299], rtol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2, 50.3073755956, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.bhat).reshape(-1)[:2],
        [0.4613980081, -0.3049279011],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(m.edf_total, 9.01308391, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2], [0.14532835, 1.47009894], rtol=0, atol=1e-6
    )
    p = m.predict(df2.head(3), type="link")
    np.testing.assert_allclose(
        p["fit"].to_numpy(), [0.14532835, 1.47009894, -1.45408158], rtol=0, atol=1e-6
    )
    m_base = gam("y ~ f + s(x)", df2, method="REML")
    assert m_base.p == 12


# ---------------------------------------------------------------------------
# qq.gam + gam.check plots + multi-LP check — mgcv 1.9-4.
# qq.gam (plots.r:94): family-correct theoretical residual quantiles — the
# qf direct path (averaged over s_rep randomized uniform grids), the rd
# simulation path (rep>0, level band), and the qqnorm fallback when the
# family has neither hook (gaulss — same in mgcv). mgcv randomizes via R's
# RNG and hea via numpy, so shuffle-dependent quantiles are pinned within
# R's own seed-to-seed band (measured 0.07 for the poisson fit below);
# the unweighted-gaussian direct path is shuffle-INVARIANT and pins exactly.
# ---------------------------------------------------------------------------


def test_qq_gam_gaussian_direct_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    q = m._qq_gam_quantiles(seed=0)
    np.testing.assert_allclose(
        q["Dq"][:5],
        [-1.0022383513, -0.8684696298, -0.8002824190, -0.7527795603, -0.7157525518],
        atol=1e-9,
    )
    np.testing.assert_allclose(q["Dq"][99], -0.0022374646, atol=1e-9)
    np.testing.assert_allclose(q["Dq"][199], 1.0022383513, atol=1e-9)
    assert q["lim"] is None
    np.testing.assert_allclose(
        np.sort(q["D"])[:3], [-1.0144419914, -0.8464473778, -0.7445768179], rtol=1e-6
    )


def test_qq_gam_poisson_direct_matches_mgcv_exactly():
    m = gam("ypois ~ z + s(x)", _pterms_fixture(), family=Poisson(), method="REML")
    q = m._qq_gam_quantiles(seed=1)
    np.testing.assert_allclose(
        q["Dq"][:5],
        [-2.7744499905, -2.3318922820, -2.1897458268, -2.0856308712, -1.9926249497],
        atol=2e-9,
    )
    np.testing.assert_allclose(
        np.asarray(q["Dq"])[[49, 99, 149, 199]],
        [-0.9523298576, -0.1019956073, 0.6055969984, 2.7228860917],
        atol=2e-9,
    )
    assert np.all(np.diff(q["Dq"]) >= 0)


def test_qq_gam_simulation_branch_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    q = m._qq_gam_quantiles(rep=20, level=0.9, seed=1)
    np.testing.assert_allclose(
        np.asarray(q["Dq"])[[0, 99, 199]], [-1.085751, -0.007870, 0.977612], atol=0.25
    )
    assert q["lim"].shape == (2, 200)
    assert np.all(q["lim"][0] <= q["lim"][1])
    q2 = m._qq_gam_quantiles(rep=5, level=1, seed=1)
    assert q2["dm"].shape == (200, 5) and q2["lim"] is None


def test_qq_gam_plot_and_gaulss_simulation():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    ax = m.qq_gam(seed=0)
    assert ax.get_xlabel() == "theoretical quantiles"
    plt.close("all")
    from hea.family import gaulss

    mg = gam(
        ["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML"
    )
    # gaulss HAS rd in mgcv (gamlss.r:1089) — qq.gam takes the
    # simulation path (no qf → rep=50 via rd), NOT a qqnorm fallback.
    qq = mg._qq_gam_quantiles(seed=0)
    assert qq["Dq"] is not None
    n = qq["Dq"].size
    a = 0.5
    pp = (np.arange(1, n + 1) - a) / n
    from scipy.stats import norm as _norm

    ref = _norm.ppf(np.clip(pp, 1e-12, 1 - 1e-12))
    inner = slice(n // 10, -n // 10)  # compare away from the tails
    np.testing.assert_allclose(qq["Dq"][inner], ref[inner], atol=0.25)
    ax = mg.qq_gam(seed=0)
    assert ax.get_title() == "QQ plot of residuals"
    plt.close("all")


def test_gaulss_residuals_match_mgcv():
    from hea.family import gaulss

    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML")
    rd = m.residuals_of("deviance")
    rr = m.residuals_of("response")
    np.testing.assert_allclose(
        rd[:5],
        [-1.40666569, -0.75618930, 0.40399967, -1.27152202, 0.33599667],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        rr[:5],
        [-0.64988679, -0.46321814, 0.13797675, -0.31089046, 0.14392372],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(m.residuals_of("pearson"), rd)
    with pytest.raises(ValueError, match="gaulss residuals"):
        m.residuals_of("working")


def test_gaulss_check_and_k_check_match_mgcv(capsys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hea.family import gaulss

    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML")
    kt = m._k_check(seed=0)
    assert kt[""].to_list() == ["s(x)", "s.1(z)"]
    np.testing.assert_allclose(kt["k'"].to_numpy(), [9.0, 9.0])
    np.testing.assert_allclose(
        kt["edf"].to_numpy(), [6.36517535, 4.60542926], rtol=1e-6
    )
    np.testing.assert_allclose(
        kt["k-index"].to_numpy(), [1.01668913, 1.01275279], rtol=1e-6
    )
    np.testing.assert_allclose(kt["p-value"].to_numpy(), [0.620, 0.525], atol=1e-12)
    m.check(seed=0, plots=False)
    out = capsys.readouterr().out
    assert "Method: REML   Optimizer: outer newton" in out
    assert "full convergence after" in out
    assert "Basis dimension (k) checking" in out
    assert "s.1(z)" in out
    axes = m.plot_check(seed=0)
    assert axes.shape == (2, 2)
    # gaulss has rd (gamlss.r:1089) → simulation-path QQ, like mgcv.
    assert axes[0, 0].get_title() == "QQ plot of residuals"
    plt.close("all")
    with pytest.raises(NotImplementedError, match="plot_smooth"):
        m.plot()


def test_family_qf_rd_unit_values_match_R():
    from hea.family import Binomial, Gaussian

    g = Gamma()
    np.testing.assert_allclose(
        g.qf(np.array([0.1, 0.5, 0.9]), np.array([2.0, 2.0, 2.0]), 1.0, 0.3),
        [0.7860239435, 1.8039491853, 3.4688989388],
        rtol=1e-9,
    )
    b = Binomial()
    np.testing.assert_allclose(
        b.qf(
            np.array([0.1, 0.5, 0.9]),
            np.array([0.4, 0.4, 0.4]),
            np.array([7.0, 7.0, 7.0]),
            1.0,
        ),
        [0.1428571429, 0.4285714286, 0.5714285714],
        rtol=1e-9,
    )
    p = Poisson()
    np.testing.assert_allclose(
        p.qf(np.array([0.1, 0.5, 0.9]), np.array([3.5, 3.5, 3.5]), 1.0, 1.0),
        [1.0, 3.0, 6.0],
    )
    gau = Gaussian()
    np.testing.assert_allclose(gau.qf(0.75, 1.2, 2.0, 0.5), 1.5372448751, rtol=1e-9)
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    mu = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(g.rd(rng1, mu, 1.0, 0.3), g.rd(rng2, mu, 1.0, 0.3))


# ---------------------------------------------------------------------------
# summary(freq=, dispersion=) + anova passthrough — mgcv 1.9-4.
# freq=TRUE swaps Ve for Vp in the PARAMETRIC tables only (mgcv.r:3890;
# the smooth tests always use Vp). dispersion= rescales every covariance
# by dispersion/sig2 and forces est.disp=FALSE — z/Chi.sq forms, testStat
# res.df=-1, reTest fed the rescaled covariances with sig2 untouched
# (mgcv.r:3895-3899) — and prints as the Scale est. Pins from
# summary(m, freq=, dispersion=)$p.table/pTerms.table/s.table on the
# CSV-identical fixtures.
# ---------------------------------------------------------------------------


def test_summary_freq_dispersion_parametric_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    np.testing.assert_allclose(
        m._se_report_for(True, None)[:5],
        [0.0729556790, 0.0735110295, 0.0728551167, 0.0746810309, 0.0932363029],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        m._se_report_for(False, 2.0)[:5],
        [0.2906511136, 0.2925519833, 0.2910747763, 0.2973692805, 0.3704794923],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        m._se_report_for(True, 0.5)[:5],
        [0.1444843226, 0.1455841607, 0.1442851649, 0.1479012779, 0.1846488750],
        rtol=1e-7,
    )
    np.testing.assert_array_equal(m._se_report_for(False, None), m._se_report)
    rows_f = m._pterms_rows(freq=True)
    assert [(r[0], r[1]) for r in rows_f] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose(
        [r[2] for r in rows_f], [80.7093362000, 59.9639681100], rtol=1e-6
    )
    rows_d = m._pterms_rows(dispersion=2.0)
    np.testing.assert_allclose(
        [r[2] for r in rows_d], [15.3088806600, 3.7978001470], rtol=1e-6
    )
    np.testing.assert_allclose(
        [r[3] for r in rows_d], [0.00157083954, 0.05131996646], rtol=1e-5
    )


def test_summary_dispersion_smooth_tables_match_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    sm = m._smooth_significance_rows(dispersion=2.0)
    np.testing.assert_allclose(
        [sm[0][1], sm[0][2], sm[0][3]],
        [6.7270776850, 7.8393200430, 51.6659932800],
        rtol=1e-5,
    )
    assert sm[0][4] < 1e-12
    # freq= never reaches the smooth table (mgcv.r:4014 hard-codes Vp).
    sm0 = m._smooth_significance_rows()
    sm0b = m._smooth_significance_rows(dispersion=None)
    assert sm0 == sm0b
    mp = gam("ypois ~ z + s(x)", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(
        mp._se_report_for(False, 1.5)[:2], [0.1157668914, 0.1779806259], rtol=1e-7
    )
    smp = mp._smooth_significance_rows(dispersion=1.5)
    np.testing.assert_allclose(smp[0][3], 160.5454171000, rtol=1e-6)


def test_summary_dispersion_re_test_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x) + s(g5, bs='re')", _pterms_fixture(), method="REML")
    sm_d = m._smooth_significance_rows(dispersion=2.0)
    sm_0 = m._smooth_significance_rows()
    np.testing.assert_allclose(sm_d[0][3], 51.6659866850, rtol=1e-5)
    np.testing.assert_allclose(sm_d[1][4], 0.9985281201, rtol=1e-6)
    np.testing.assert_allclose(sm_d[1][3], sm_0[1][3] * sm_0[1][2], rtol=1e-10)


def test_summary_re_test_false_drops_re_rows(capsys):
    # summary.gam(re.test=FALSE) (mgcv.r:3858 formal, 4024 gate): the
    # reTest-path smooths get res <- NULL and their rows are DROPPED —
    # not rerouted to testStat. mgcv receipt (1.9-4): s.table goes 2
    # rows -> 1, and the surviving s(x) row is all.equal-identical.
    m = gam("ygau ~ f4 + z + s(x) + s(g5, bs='re')", _pterms_fixture(), method="REML")
    r1 = m._smooth_significance_rows()
    r0 = m._smooth_significance_rows(re_test=False)
    assert [r[0] for r in r1] == ["s(x)", "s(g5)"]
    assert [r[0] for r in r0] == ["s(x)"]
    assert r0[0] == r1[0]  # surviving row bit-identical
    m.summary(re_test=False)
    out = capsys.readouterr().out
    assert "s(g5)" not in out and "s(x)" in out


def test_summary_freq_dispersion_gaulss_and_print(capsys):
    from hea.family import gaulss

    mg = gam(
        ["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML"
    )
    idx = mg._param_idx
    np.testing.assert_allclose(
        mg._se_report_for(True, None)[idx],
        [0.0561071082, 0.0980386275, 0.0488761346],
        rtol=1e-6,
    )
    m1 = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    m1.summary(dispersion=2.0)
    out = capsys.readouterr().out
    assert "Scale est. = 2  " in out
    assert "Chi.sq" in out and "z value" in out
    m1.summary(freq=True)
    out_f = capsys.readouterr().out
    assert "t value" in out_f and "F" in out_f


def test_anova_gam_freq_dispersion_passthrough(capsys):
    from hea.R import anova

    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    anova(m, dispersion=2.0)
    out = capsys.readouterr().out
    assert "Chi.sq" in out and "15.3" in out
    anova(m, freq=True)
    out_f = capsys.readouterr().out
    assert "80.7" in out_f or "80.71" in out_f
    m2 = gam("ygau ~ z + s(x)", _pterms_fixture(), method="REML")
    with pytest.raises(TypeError, match="single-gam"):
        anova(m, m2, dispersion=2.0)


# ---------------------------------------------------------------------------
# concurvity — mgcv.r:3340-3423, R pins on the CSV fixtures.
# Blocks = each smooth + mgcv's "para" block, which (evaluation-order
# quirk: stop <- c(min(start)-1, stop) AFTER start was prepended) is just
# the FIRST design column — ported bug-for-bug; single column ⇒ the three
# para measures coincide, pairwise para rows are exact zeros vs centered
# smooths, and a multi-LP fit's duplicated LP2 intercept drives para to 1.
# ---------------------------------------------------------------------------


def test_concurvity_full_and_pairwise_match_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + s(x) + s(z)", df, method="REML")
    cf = m.concurvity()
    assert cf.columns == ["", "para", "s(x)", "s(z)"]
    assert cf[""].to_list() == ["worst", "observed", "estimate"]
    np.testing.assert_allclose(cf["para"].to_numpy(), [0.7888546768] * 3, rtol=1e-7)
    np.testing.assert_allclose(  # worst, estimate: mgcv
        cf["s(x)"].to_numpy()[[0, 2]], [0.1690506098, 0.0507947318], rtol=1e-5
    )
    np.testing.assert_allclose(
        cf["s(x)"].to_numpy()[1],  # observed: hea fit
        0.0689315647,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        cf["s(z)"].to_numpy()[[0, 2]], [0.1408198865, 0.0737227575], rtol=1e-5
    )
    np.testing.assert_allclose(cf["s(z)"].to_numpy()[1], 0.0702257390, rtol=1e-5)
    cp = m.concurvity(full=False)
    assert set(cp) == {"worst", "observed", "estimate"}
    W = cp["worst"]
    assert W[""].to_list() == ["para", "s(x)", "s(z)"]
    np.testing.assert_allclose(np.diag(W.to_numpy()[:, 1:].astype(float)), np.ones(3))
    assert float(W["s(x)"][0]) < 1e-12 and float(W["para"][1]) < 1e-12
    np.testing.assert_allclose(float(W["s(z)"][1]), 0.1289878965, rtol=1e-6)
    np.testing.assert_allclose(
        float(cp["observed"]["s(z)"][1]),  # hea fit
        0.0639043896,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(cp["observed"]["s(x)"][2]),  # hea fit
        0.0427210945,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(cp["estimate"]["s(z)"][1]),  # mgcv
        0.0640022213,
        rtol=1e-5,
    )


def test_concurvity_correlated_and_intercept_only_para():
    df = _pterms_fixture().with_columns(((pl.col("x") + pl.col("z")) / 2).alias("xc"))
    m = gam("ygau ~ s(x) + s(xc)", df, method="REML")
    cf = m.concurvity()
    assert float(np.abs(cf["para"].to_numpy()).max()) < 1e-12
    np.testing.assert_allclose(
        cf["s(x)"].to_numpy(), [0.5448962547, 0.3866319478, 0.4592832865], rtol=1e-5
    )
    np.testing.assert_allclose(
        cf["s(xc)"].to_numpy(), [0.5448962547, 0.5301830175, 0.4381152160], rtol=1e-5
    )
    cp = m.concurvity(full=False)
    np.testing.assert_allclose(float(cp["worst"]["s(xc)"][1]), 0.5448962547, rtol=1e-6)


def test_concurvity_multi_lp_gaulss_matches_mgcv():
    from hea.family import gaulss

    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML")
    cf = m.concurvity()
    assert cf.columns == ["", "para", "s(x)", "s.1(z)"]
    np.testing.assert_allclose(cf["para"].to_numpy(), np.ones(3), atol=1e-10)
    # The duplicated intercept makes the stacked X EXACTLY rank-deficient
    # (σ_min ≈ 3e-15), and the FULL measures run an unpivoted QR over it
    # (mgcv.r:3376 "No pivoting!!") — a 1e-14 perturbation of X moves
    # these values by ~5e-3, so they are platform noise at that scale in
    # mgcv too (CI/OpenBLAS measured 3.9e-4 from the Mac pins). Pin the
    # noise band, not the digits.
    np.testing.assert_allclose(
        cf["s(x)"].to_numpy(), [0.11186354, 0.06067571, 0.05768171], atol=0.02
    )
    np.testing.assert_allclose(
        cf["s.1(z)"].to_numpy(), [0.12120035, 0.03000198, 0.02321672], atol=0.02
    )
    cp = m.concurvity(full=False)
    np.testing.assert_allclose(
        float(cp["estimate"]["s.1(z)"][1]), 0.01809372, rtol=1e-5
    )
    m0 = gam("ygau ~ f4 + z", _pterms_fixture(), method="REML")
    with pytest.raises(ValueError, match="nothing to do"):
        m0.concurvity()


# ---------------------------------------------------------------------------
# influence / cooks_distance accessors — mgcv.r:4415/4212.
# influence.gam returns model$hat (the penalized hat diagonal, Σ = edf);
# cooks.distance.gam = (pearson/(1−hat))²·hat/(φ̂·Σedf). General-family
# fits have NULL hat in mgcv (influence empty, cooks all-NA) — hea raises.
# ---------------------------------------------------------------------------


def test_influence_cooks_distance_match_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    h = m.influence()
    np.testing.assert_allclose(
        h[:5],
        [0.0471134262, 0.1707783228, 0.0509776201, 0.1307545683, 0.0666771951],
        rtol=1e-6,
    )
    np.testing.assert_allclose(h.sum(), m.edf_total, rtol=1e-10)
    cd = m.cooks_distance()
    np.testing.assert_allclose(
        cd[:5],
        [0.0057238038, 0.0282487724, 0.0032349746, 0.0662443276, 0.0168205445],
        rtol=1e-6,
    )
    np.testing.assert_allclose(cd.max(), 0.0662443276, rtol=1e-6)
    assert int(np.argmax(cd)) == 3  # R's which.max = 4, 1-based
    mp = gam("ypois ~ z + s(x)", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(
        mp.influence()[:3], [0.0314975428, 0.1067972886, 0.0280958236], rtol=1e-6
    )
    np.testing.assert_allclose(
        mp.cooks_distance()[:3], [0.0004029915, 0.0282574127, 0.0082311885], rtol=1e-6
    )
    from hea.family import gaulss

    mg = gam(
        ["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(), method="REML"
    )
    with pytest.raises(NotImplementedError, match="general-family"):
        mg.influence()
    with pytest.raises(NotImplementedError, match="general-family"):
        mg.cooks_distance()


def _betar_fixture():
    from scipy.stats import beta as _B

    from hea.R.rng import RGenerator

    gen = RGenerator(71)
    n = 250
    x = gen.uniform(0, 1, n)
    u = gen.uniform(0, 1, n)
    mu = 1.0 / (1.0 + np.exp(-(0.8 * np.sin(2 * np.pi * x) - 0.3)))
    y = _B.ppf(u, 12.0 * mu, 12.0 * (1.0 - mu))
    return pl.DataFrame({"y": y, "x": x})


@pytest.mark.filterwarnings("ignore:Fitting terminated with step failure:UserWarning")
def test_betar_through_gam_matches_mgcv():
    from hea.family import betar

    df = _betar_fixture()
    m = gam("y ~ s(x)", df, family=betar(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, -161.49936705, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.1881112502], rtol=1e-4)
    np.testing.assert_allclose(
        float(np.exp(m.family.get_theta()[0])), 14.41333656, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(m.edf_total, 6.82912381, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.deviance, 234.26619683, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.null_deviance, 468.30180613, rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00110098, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:3]),
        np.abs([0.236147, 1.665602, 0.753554]),
        rtol=0,
        atol=1e-4,
    )

    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [0.5893680, 0.3717789, 0.5918235], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(), [0.0179293, 0.0206123, 0.0178807], rtol=0, atol=1e-5
    )

    np.testing.assert_allclose(
        np.asarray(m.residuals_of("deviance"))[:3],
        [-0.179788, -0.679496, 1.748929],
        rtol=0,
        atol=1e-5,
    )

    # the other three okLinks exercise the probit/cloglog/cauchit
    # g2g/g3g/g4g extended forms (gam.fit3.r:2249-2303) in the extended
    # IRLS — REML matched to R per link.
    for lk, reml_ref in (
        ("probit", -160.5668040),
        ("cloglog", -160.8250744),
        ("cauchit", -160.9522179),
    ):
        ml = gam("y ~ s(x)", df, family=betar(link=lk), method="REML")
        np.testing.assert_allclose(ml.REML_criterion / 2, reml_ref, rtol=0, atol=1e-4)


def _ocat_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(54)
    n = 300
    x = gen.uniform(0, 1, n)
    f0 = 2.0 * np.sin(2 * np.pi * x) - 0.3
    u = gen.uniform(0, 1, n)
    lat = f0 + np.log(u / (1.0 - u))
    alpha = np.array([-np.inf, -1.0, 0.1, 1.1, np.inf])
    y = np.zeros(n, dtype=int)
    for i in range(4):
        y[(lat > alpha[i]) & (lat <= alpha[i + 1])] = i
    return pl.DataFrame({"y": y, "x": x})


def test_ocat_through_gam_matches_mgcv():
    from hea.family import ocat

    df = _ocat_fixture()
    m = gam("y ~ s(x)", df, family=ocat(R=4), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 311.1907401388, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.017735245992525465], rtol=1e-4)
    np.testing.assert_allclose(
        m.family.get_theta(),
        [0.10714304251230537, -0.16319455867953944],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        m.family.get_theta(trans=True),
        [-1.0, 0.113093462777623, 0.96251937193122572],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(m.edf_total, 6.5603973369, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 600.2801127222, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.null_deviance, 802.6873793845, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.015303883954, rtol=0, atol=1e-6
    )
    assert float(np.asarray(m._beta)[0]) < 0
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs(
            [
                -0.4253831825912549,
                -5.7504861424827149,
                0.64506034454421934,
                0.72245036970946885,
            ]
        ),
        rtol=0,
        atol=1e-3,
    )
    assert m._postproc["family_name"] == "Ordered Categorical(-1,0.11,0.96)"

    pr = m.predict(pl.DataFrame({"x": [0.2, 0.5, 0.8]}), type="response", se_fit=True)
    fit = np.column_stack(
        [pr[c].to_numpy() for c in ("fit", "fit.1", "fit.2", "fit.3")]
    )
    se = np.column_stack(
        [pr[c].to_numpy() for c in ("se.fit", "se.fit.1", "se.fit.2", "se.fit.3")]
    )
    np.testing.assert_allclose(fit.sum(axis=1), [1, 1, 1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        fit[0],
        [
            0.0586729857099971,
            0.10079153292851062,
            0.1478313777963898,
            0.69270410356510248,
        ],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        fit[2],
        [
            0.8502720903780735,
            0.095037812305111879,
            0.030545553826115346,
            0.024144543490699277,
        ],
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        se[0],
        [
            0.016432665667876309,
            0.023446808696688203,
            0.023454075265177548,
            0.063333549629742053,
        ],
        rtol=0,
        atol=1e-5,
    )

    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:5],
        [
            1.6159625596856027,
            2.6002725708653465,
            -0.55428957625776465,
            0.85980858568121443,
            1.0698318351438267,
        ],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(float(np.sum(dres**2)), m.deviance, rtol=0, atol=1e-8)


def _ziP_fixture():
    from scipy.stats import poisson

    from hea.R.rng import RGenerator

    gen = RGenerator(7)
    n = 400
    x = gen.uniform(0, 1, n)
    u_count = gen.uniform(0, 1, n)
    u_pres = gen.uniform(0, 1, n)
    gamma = 1.5 * np.sin(2 * np.pi * x) + 0.2
    p = 1.0 - np.exp(-np.exp(-1.0 + 1.6 * gamma))
    y = np.where(u_pres < p, poisson.ppf(u_count, np.exp(gamma)), 0.0)
    return pl.DataFrame({"y": y.astype(int), "x": x})


def test_ziP_through_gam_matches_mgcv():
    from hea.family import ziP

    df = _ziP_fixture()
    m = gam("y ~ s(x)", df, family=ziP(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 422.7206170258, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.083619872403188383], rtol=2e-3)
    np.testing.assert_allclose(
        m.family.get_theta(),
        [-3.2949401719022888, 1.2258053436229546],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(m.edf_total, 6.9362279961, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 236.9705123525, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.null_deviance, 761.4398775608, rtol=0, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.00352127204674, rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        m.fitted_values[:4],
        [
            0.49477606138778263,
            1.1932554070862351,
            1.2263867543910303,
            0.98869377799869562,
        ],
        rtol=0,
        atol=1e-4,
    )
    assert m._postproc["family_name"] == "Zero inflated Poisson(-3.295,3.407)"
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:3]),
        np.abs([0.58794071413230287, 1.9594814883910816, 0.61803791743402448]),
        rtol=0,
        atol=2e-3,
    )

    pr = m.predict(pl.DataFrame({"x": [0.2, 0.5, 0.8]}), type="response", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [4.7469199014934222, 0.43993064312301344, 0.03157416057280106],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.27815576114349722, 0.12612620416309303, 0.020997731734616366],
        rtol=0,
        atol=1e-3,
    )

    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:6],
        [
            -0.63250442710168497,
            0.56473914672204939,
            0.88371319063094189,
            -1.4671154974843927,
            -0.024871912805149807,
            -0.2035584951393099,
        ],
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_allclose(float(np.sum(dres**2)), m.deviance, rtol=0, atol=1e-6)
    rres = np.asarray(m.residuals_of("response"))
    np.testing.assert_allclose(
        rres[:4],
        [
            -0.3688958956021644,
            0.97031232323179584,
            1.7883819521639648,
            -1.9008466174667389,
        ],
        rtol=0,
        atol=2e-3,
    )

    # no.r.sq (efam.r:4142; ocat too, efam.r:3080): summary.gam sets
    # r.sq NULL (mgcv.r:4055) and the print shows Deviance explained
    # without an R-sq.(adj) — hea mirrors via NaN r² at fit time.
    from hea.family import ocat

    assert ocat(R=4).no_r_sq is True
    assert np.isnan(m.r_squared) and np.isnan(m.r_squared_adjusted)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.summary()
    out_txt = buf.getvalue()
    assert "R-sq" not in out_txt and "Deviance explained" in out_txt


def test_ziP_b_nonzero_through_gam_matches_mgcv():
    from hea.family import ziP

    df = _ziP_fixture()
    m = gam("y ~ s(x)", df, family=ziP(b=0.3), method="REML")
    np.testing.assert_allclose(
        m.REML_criterion / 2, 422.72061702752336, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(m.sp, [0.083621695803168888], rtol=2e-3)
    np.testing.assert_allclose(
        m.family.get_theta(),
        [-3.2949850446340050, 1.1336412686440676],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        m.family.get_theta(trans=True),
        [-3.2949850446340050, 3.4069491644063006],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(m.edf_total, 6.936220446801042, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 236.97051074207116, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.null_deviance, 761.4387750977171, rtol=0, atol=5e-3)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.0035211998176852485, rtol=0, atol=1e-5
    )
    assert m._postproc["family_name"] == "Zero inflated Poisson(-3.295,3.407)"
    pr = m.predict(pl.DataFrame({"x": [0.2, 0.5, 0.8]}), type="response", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [4.74691676164069154, 0.43993115801732319, 0.03157427031317666],
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.278155043412824876, 0.126126458821157666, 0.020997829842324959],
        rtol=0,
        atol=1e-3,
    )


def _cnorm_fixture():
    from hea.R.rng import RGenerator

    gen = RGenerator(6)
    n = 200
    x = gen.uniform(0, 1, n)
    ys = 2.0 * np.sin(np.pi * x) + gen.normal(size=n) * 1.3
    y1 = ys.copy()
    y2 = ys.copy()
    m = ys < -1.0
    y1[m] = -1.0
    y2[m] = -np.inf
    m = ys > 3.0
    y1[m] = 3.0
    y2[m] = np.inf
    return pl.DataFrame({"x": x, "y1": y1, "y2": y2})


def test_cnorm_through_gam_matches_mgcv():
    from hea.family import cnorm

    df = _cnorm_fixture()
    m = gam("cbind(y1, y2) ~ s(x)", df, family=cnorm(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 320.1707496643, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.321192647856], rtol=2e-3)
    np.testing.assert_allclose(
        m.family.get_theta(), [0.228565516296], rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        m.family.get_theta(True), [1.25679586304], rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(m.edf_total, 4.3945575588, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 229.8452755840, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.null_deviance, 275.8797556247, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.scale, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(m.Vp)[0, 0], 0.00809956779924, rtol=0, atol=1e-5
    )
    assert m._family_display_name() == "cnorm(1.257)"
    np.testing.assert_allclose(
        np.sum(np.abs(np.asarray(m._beta))), 7.23645196361, rtol=0, atol=2e-3
    )

    pr = m.predict(
        pl.DataFrame({"x": [0.1, 0.3, 0.5, 0.7, 0.9]}), type="response", se_fit=True
    )
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [0.52039066828, 1.40609052147, 1.80902094522, 1.54790986302, 0.64630975498],
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.186345591375, 0.17266137593, 0.174231865377, 0.177779241717, 0.172302773093],
        rtol=0,
        atol=1e-3,
    )

    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:5],
        [
            -0.875033484349,
            0.0785548636183,
            0.221766506956,
            -0.837126217578,
            -0.0758774624658,
        ],
        rtol=0,
        atol=2e-3,
    )
    np.testing.assert_allclose(float(np.sum(dres**2)), m.deviance, rtol=0, atol=1e-6)
    rres = np.asarray(m.residuals_of("response"))
    np.testing.assert_allclose(
        rres[:5],
        [
            -1.09973846315,
            0.0987274276175,
            0.278715228504,
            -1.0520967671,
            -0.0953624809253,
        ],
        rtol=0,
        atol=2e-3,
    )


# =============================================================================
# Smooth-class extras
# -----------------------------------------------------------------------------
# S1a — s(..., pc=) point constraints. mgcv smooth.construct3 (smooth.r:3676-
# 3679) REPLACES the default sum-to-zero identifiability constraint
# (C = colMeans(X)) with the smooth's basis row evaluated AT the pc point
# (always.apply=TRUE), so the smooth passes through 0 there and the intercept
# absorbs the shift. Was SILENTLY IGNORED before (pc= unparsed → fit identical
# to s(x)). Pins: mgcv 1.9-4.
# =============================================================================


def _pc_fixture(seed: int = 6, n: int = 100) -> pl.DataFrame:
    from hea.R.rng import RGenerator

    g = RGenerator(seed)
    x = g.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + g.normal(0.0, 1.0, n) * 0.3
    return pl.DataFrame({"x": x, "y": y})


def test_s_pc_point_constraint_matches_mgcv():
    """s(x, pc=0.5): the point constraint reparameterizes — fit is
    identifiability-invariant (predictions == s(x)) but the smooth term passes
    through 0 at x=0.5 and the intercept becomes the smooth's value there."""
    d = _pc_fixture()
    m = gam("y ~ s(x, k=10, pc=0.5)", d, method="REML")
    assert m.REML_criterion / 2 == pytest.approx(25.3327898083, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(7.5711903486, rel=1e-5)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(0.1249797454, abs=1e-6)
    tp = m.predict(pl.DataFrame({"x": [0.5]}), type="terms")
    assert abs(float(np.asarray(tp).ravel()[0])) < 1e-9
    pred = float(np.asarray(m.predict(pl.DataFrame({"x": [0.5]}))).ravel()[0])
    assert pred == pytest.approx(0.1249797454, abs=1e-6)

    m0 = gam("y ~ s(x, k=10)", d, method="REML")
    assert float(np.asarray(m0.coef)[0]) == pytest.approx(-0.0073841475, abs=1e-6)
    assert abs(float(np.asarray(m.coef)[0]) - float(np.asarray(m0.coef)[0])) > 0.1


def test_s_pc_with_by_factor_matches_mgcv():
    """pc= shares one always-applied constraint across by-factor levels —
    each level's smooth passes through 0 at the point. R-pinned end-to-end."""
    from hea.R.rng import RGenerator

    g = RGenerator(11)
    x = g.uniform(0.0, 1.0, 200)
    f = np.array(["a", "b"] * 100)  # deterministic f (RNG-reproducible)
    y = np.sin(2 * np.pi * x) + (f == "b") * 0.5 + g.normal(0.0, 1.0, 200) * 0.3
    d = pl.DataFrame({"x": x, "f": f, "y": y})
    m = gam("y ~ f + s(x, k=8, by=f, pc=0.4)", d, method="REML")
    assert m.REML_criterion / 2 == pytest.approx(64.78112149, abs=1e-5)
    assert float(np.sum(m.edf)) == pytest.approx(13.85115823, rel=1e-5)


def test_s_pc_unsupported_forms_raise():
    """Honest raises (no silent mis-fit) for the pc forms hea doesn't port:
    the general list-of-lists constraint, t2()/matrix-arg te(), a numeric by=,
    and a covariate-count mismatch."""
    d = _pc_fixture()
    d = d.with_columns(z=pl.col("x") * 0.5 + 0.1)
    with pytest.raises(NotImplementedError, match="general"):
        gam("y ~ s(x, pc=list(1))", d, method="REML")
    with pytest.raises(NotImplementedError, match="t2"):
        gam("y ~ t2(x, z, pc=0.5)", d, method="REML")
    with pytest.raises(ValueError, match="one value per smooth covariate"):
        gam("y ~ s(x, pc=c(0.5, 0.3))", d, method="REML")


# -----------------------------------------------------------------------------
# S1b — te/ti fx=TRUE (fixed / unpenalized tensor). mgcv builds every margin
# penalized then drops the corresponding TENSOR penalty for fx margins
# (smooth.r:462, 830). hea was forwarding fx INTO the marginal s() call,
# emptying its penalty and crashing the one-penalty-per-margin assembly. t2 has
# NO fx (t2() hardcodes fx<-FALSE, smooth.r:539) — honest raise. Pins: mgcv 1.9-4.
# -----------------------------------------------------------------------------


def _tensor_fx_fixture(n: int = 200) -> pl.DataFrame:
    from hea.R.rng import RGenerator

    g = RGenerator(7)
    x = g.uniform(0.0, 1.0, n)
    z = g.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * z) + g.normal(0.0, 1.0, n) * 0.3
    return pl.DataFrame({"x": x, "z": z, "y": y})


@pytest.mark.parametrize(
    "form, edf, coef0, dev",
    [
        ("te(x, z, k=4, fx=TRUE)", 16.0, -0.12128656, 22.762519),
        ("te(x, z, k=4, fx=c(TRUE,FALSE))", 15.16135371, -0.12128656, 22.840118),
        ("ti(x, z, k=4, fx=TRUE)", 10.0, -0.24088277, 179.766597),
    ],
)
def test_te_ti_fx_matches_mgcv(form, edf, coef0, dev):
    """Fixed/partly-fixed tensor smooths fit unpenalized like mgcv."""
    m = gam(f"y ~ {form}", _tensor_fx_fixture(), method="REML")
    assert float(np.sum(m.edf)) == pytest.approx(edf, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(coef0, abs=1e-6)
    assert float(m.deviance) == pytest.approx(dev, rel=1e-6)


def test_t2_fx_raises():
    """t2 has no fx in mgcv (hardcoded FALSE) — honest raise, not a mis-fit."""
    with pytest.raises(NotImplementedError, match="t2.. does not support fx"):
        gam("y ~ t2(x, z, k=3, fx=TRUE)", _tensor_fx_fixture(), method="REML")


# -----------------------------------------------------------------------------
# S1c — matrix-argument (summation convention) te()/ti()/s() with a numeric
# by= (anisotropic signal regression / distributed-lag / receptive-field
# models). mgcv smoothCon (smooth.r:3877-4051) for a matrix argument:
#   * scale.penalty on the LONG-FORM (n*m, p) X — *before* the by-multiply
#     (3879); scaling the row-summed X gives a different S.scale.
#   * numeric by-multiply on the long form, then row-block summation (3997).
#   * the centering constraint is DROPPED when the by-matrix row-sums vary
#     (sd(L1) > mean(L1)·eps·1000, 3925-3943) — so the smooth keeps all its
#     raw columns (s k=6 → 7 coefs incl. intercept; te 5×4 → 21).
#   * check.rank on the summed design (4035).
# Factor by= and pc= with matrix args are rejected (mgcv stops on factor by,
# smooth.r:3970). Fixed-sp pins validate the design+penalty+scale.penalty
# independent of the outer optimiser. Pins: mgcv 1.9-4.
# -----------------------------------------------------------------------------


def _rf_matrix_fixture(seed: int = 42, n: int = 120, nlag: int = 6, nx: int = 4):
    """Receptive-field-style matrix-arg fixture (R-bit-exact RNG draw order:
    Stim1, Stim2, y-noise, then the no-by covariates A, B)."""
    from hea.formula import normalize_data
    from hea.R.rng import RGenerator

    g = RGenerator(seed)
    Stim1 = g.normal(0.0, 1.0, n * nlag).reshape(n, nlag, order="F")
    Lag = np.tile(np.arange(nlag, dtype=float), (n, 1))
    m2 = nlag * nx
    Stim2 = g.normal(0.0, 1.0, n * m2).reshape(n, m2, order="F")
    lag2 = np.array([lg for x in range(nx) for lg in range(nlag)], dtype=float)
    xc = np.array([x for x in range(nx) for lg in range(nlag)], dtype=float)
    Lag2 = np.tile(lag2, (n, 1))
    Xc = np.tile(xc, (n, 1))
    eta = Stim1 @ np.exp(-np.arange(nlag) / 2.0) * 1.2
    y = eta + g.normal(0.0, 1.0, n) * 0.5
    A = g.uniform(0.0, 1.0, n * 8).reshape(n, 8, order="F")
    B = g.uniform(0.0, 1.0, n * 8).reshape(n, 8, order="F")
    return normalize_data(
        {
            "y": y,
            "Stim1": Stim1,
            "Lag": Lag,
            "Stim2": Stim2,
            "Lag2": Lag2,
            "Xc": Xc,
            "A": A,
            "B": B,
        }
    )


def test_s_matrix_arg_by_matches_mgcv():
    """1-D matrix-arg s(Lag, by=Stim) (temporal RF). The varying by-matrix
    row-sums drop the centering constraint ⇒ 7 coefs (1 intercept + 6 raw),
    not 6. Free fit lands in the flat-optimum band; sp= is exact."""
    d = _rf_matrix_fixture()
    m = gam("y ~ s(Lag, by=Stim1, k=6)", d, method="REML")
    assert len(np.asarray(m.coef)) == 7  # no constraint dropped
    assert float(np.sum(m.edf)) == pytest.approx(5.42301495, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(95.1808698, abs=1e-4)
    assert float(m.scale) == pytest.approx(0.244131395, rel=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:3],
        [-0.111997576, -2.01465219, 0.78631313],
        atol=1e-4,
    )
    mf = gam("y ~ s(Lag, by=Stim1, k=6)", d, method="REML", sp=[0.5])
    assert float(np.sum(mf.edf)) == pytest.approx(4.83969374, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(95.7033854, abs=1e-6)
    assert float(mf.scale) == pytest.approx(0.24526852, rel=1e-6)
    assert float(mf.deviance) == pytest.approx(28.2451979, rel=1e-6)


def test_te_matrix_arg_by_matches_mgcv():
    """Anisotropic 2-D matrix-arg te(Lag, Xc, by=Stim) — the distributed-lag /
    spatiotemporal RF model (was NotImplementedError). Per-margin smoothing,
    no centering constraint (21 = 1 + 5·4 coefs). predict() reproduces the
    in-sample fit through the BasisSpec replay (raw → by → row-sum → no
    absorb). Free + fixed-sp pins, mgcv 1.9-4."""
    d = _rf_matrix_fixture()
    m = gam("y ~ te(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML")
    assert len(np.asarray(m.coef)) == 21
    assert float(np.sum(m.edf)) == pytest.approx(6.38619524, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(222.256481, abs=1e-4)
    assert float(m.scale) == pytest.approx(2.21967292, rel=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:3],
        [-0.143668096, -0.0772378245, 0.647804128],
        atol=1e-5,
    )
    from hea.formula import normalize_data

    nd = normalize_data(
        {
            "Lag2": d["Lag2"].to_numpy()[:4],
            "Xc": d["Xc"].to_numpy()[:4],
            "Stim2": d["Stim2"].to_numpy()[:4],
        }
    )
    pr = np.asarray(m.predict(nd)).ravel()
    np.testing.assert_allclose(
        pr, [-0.143668096, -0.0772378245, 0.647804128, 0.193445483], atol=1e-5
    )
    mf = gam("y ~ te(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML", sp=[0.3, 2.0])
    assert float(np.sum(mf.edf)) == pytest.approx(20.8201247, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(261.321131, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.38783979, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(mf.coef)[:5],
        [-0.121393561, -0.172502089, 0.102475379, -0.0678943387, -0.0657378187],
        atol=1e-6,
    )


def test_ti_matrix_arg_by_matches_mgcv():
    """Matrix-arg ti(Lag, Xc, by=Stim) — the pure-interaction tensor (centered
    margins, no outer constraint). Free + fixed-sp, mgcv 1.9-4."""
    d = _rf_matrix_fixture()
    m = gam("y ~ ti(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML")
    assert len(np.asarray(m.coef)) == 13
    assert float(np.sum(m.edf)) == pytest.approx(2.67806372, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(221.768248, abs=1e-4)
    mf = gam("y ~ ti(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML", sp=[0.4, 1.1])
    assert float(np.sum(mf.edf)) == pytest.approx(12.64987, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(240.652234, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.38504158, rel=1e-6)


def test_te_matrix_arg_no_by_scale_penalty_matches_mgcv():
    """No-by matrix-arg te(A, B): scale.penalty is computed on the long form,
    not the row-summed X. Fixed sp pins coef-level, mgcv 1.9-4."""
    d = _rf_matrix_fixture()
    mf = gam("y ~ te(A, B, k=c(4,4))", d, method="REML", sp=[0.7, 1.3])
    assert len(np.asarray(mf.coef)) == 16
    assert float(np.sum(mf.edf)) == pytest.approx(14.6812672, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(234.960699, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.39505718, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(mf.coef)[:4],
        [-0.0446469446, 0.119795764, 0.0241560802, 0.141054417],
        atol=1e-6,
    )


def test_matrix_arg_by_unsupported_forms_raise():
    """Honest raises (no silent mis-fit): factor by= (mgcv stops too,
    smooth.r:3970), t2() matrix args (no summation port), and pc= with a
    matrix argument."""
    d = _rf_matrix_fixture()
    d = d.with_columns(fac=pl.Series((np.arange(d.height) % 2).astype(str)))
    with pytest.raises(NotImplementedError, match="factor by="):
        gam("y ~ te(Lag2, Xc, by=fac, k=c(5,4))", d, method="REML")
    with pytest.raises(NotImplementedError, match="t2.. with matrix arguments"):
        gam("y ~ t2(Lag2, Xc, by=Stim2)", d, method="REML")
    with pytest.raises(NotImplementedError, match="pc="):
        gam("y ~ te(Lag2, Xc, by=Stim2, pc=c(1.0, 1.0), k=c(5,4))", d, method="REML")


# -----------------------------------------------------------------------------
# S2 — bs="mrf" (Markov random field). Region indicator basis + graph-Laplacian
# penalty from a neighbour list (or a supplied penalty matrix). xt threaded via
# gam(xt={region: {...}}) — the object-arg channel, like knots=. Default k =
# #regions (full rank). Source: smooth.r:2726-2875. Pins: mgcv 1.9-4.
# -----------------------------------------------------------------------------

_MRF_NB = {
    "0": ["1", "3"],
    "1": ["0", "2", "4"],
    "2": ["1", "5"],
    "3": ["0", "4", "6"],
    "4": ["1", "3", "5", "7"],
    "5": ["2", "4", "8"],
    "6": ["3", "7"],
    "7": ["4", "6", "8"],
    "8": ["5", "7"],
}


def _mrf_fixture(n: int = 180) -> pl.DataFrame:
    from hea.R.rng import RGenerator

    g = RGenerator(5)
    region = np.arange(n) % 9  # rep(0:8, length.out=n)
    reff = np.array([-1, -0.5, 0.2, 0.8, 0, -0.8, 0.5, 1, -1.2])
    y = reff[region] + g.normal(0.0, 1.0, n) * 0.5
    return pl.DataFrame({"region": region, "y": y})


def test_mrf_through_gam_matches_mgcv():
    """3x3-grid MRF (rook adjacency): graph-Laplacian penalty from nb, full rank
    (k = #regions). R-pinned end-to-end + predict + penalty=-form equivalence."""
    d = _mrf_fixture()
    m = gam('y ~ s(region, bs="mrf")', d, method="REML", xt={"region": {"nb": _MRF_NB}})
    assert m.REML_criterion / 2 == pytest.approx(147.98002060, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(8.83057332, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(-0.10822060, abs=1e-6)
    assert float(m.scale) == pytest.approx(0.24834096, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.coef),
        [
            -0.108221,
            -0.238663,
            0.354659,
            1.102483,
            0.614502,
            -0.459381,
            0.733006,
            1.408727,
            -0.957989,
        ],
        atol=1e-5,
    )
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 4, 8]}))).ravel()
    assert pr[0] == pytest.approx(-0.960669, abs=1e-5)

    S = np.zeros((9, 9))
    for i, ns in _MRF_NB.items():
        S[int(i), int(i)] = len(ns)
        for j in ns:
            S[int(i), int(j)] = -1.0
    m2 = gam('y ~ s(region, bs="mrf")', d, method="REML", xt={"region": {"penalty": S}})
    assert m2.REML_criterion == pytest.approx(m.REML_criterion, abs=1e-9)


def test_mrf_low_rank_matches_mgcv():
    """Low-rank MRF (k=4 < 9 regions): natural-parameter truncation via
    nat.param(type=0), keeping the 4 least-penalized basis directions.
    """
    d = _mrf_fixture()
    m = gam(
        'y ~ s(region, bs="mrf", k=4)', d, method="REML", xt={"region": {"nb": _MRF_NB}}
    )
    assert m.REML_criterion / 2 == pytest.approx(203.18732238, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(3.93834728, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(-0.10822060, abs=1e-6)
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 1, 2]}))).ravel()
    np.testing.assert_allclose(pr, [-0.677186, -0.390348, -0.103510], atol=1e-5)


_MRF_POLYS_NB = {
    "0": ["1", "3", "4"],
    "1": ["0", "2", "3", "4", "5"],
    "2": ["1", "4", "5"],
    "3": ["0", "1", "4", "6", "7"],
    "4": ["0", "1", "2", "3", "5", "6", "7", "8"],
    "5": ["1", "2", "4", "7", "8"],
    "6": ["3", "4", "7"],
    "7": ["3", "4", "5", "6", "8"],
    "8": ["4", "5", "7"],
}


def _mrf_grid_polys() -> dict:
    """3x3 grid of closed unit squares; region r at (row=r//3, col=r%3)."""
    polys = {}
    for r in range(9):
        row, col = r // 3, r % 3
        polys[str(r)] = np.array(
            [
                [col, row],
                [col + 1, row],
                [col + 1, row + 1],
                [col, row + 1],
                [col, row],
            ],
            dtype=float,
        )
    return polys


def test_mrf_polys_pol2nb_matches_mgcv():
    """polys= path: derive the neighbour list from boundary polygons via the
    pol2nb port (smooth.r:2668-2723), then build the graph-Laplacian penalty.
    The neighbour structure (queen adjacency) is pinned exactly vs mgcv:::pol2nb;
    the end-to-end fit + predict are pinned vs mgcv; and polys= is shown
    equivalent to supplying the derived nb= directly."""
    from hea.formula import _pol2nb

    polys = _mrf_grid_polys()
    assert _pol2nb(polys) == _MRF_POLYS_NB

    d = _mrf_fixture()
    m = gam(
        'y ~ s(region, bs="mrf")', d, method="REML", xt={"region": {"polys": polys}}
    )
    assert m.REML_criterion / 2 == pytest.approx(147.85997000, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(8.83460016, rel=1e-6)
    assert float(m.scale) == pytest.approx(0.24833166, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.coef),
        [
            -0.108221,
            -0.236804,
            0.360208,
            1.106102,
            0.604543,
            -0.454536,
            0.729667,
            1.411282,
            -0.959663,
        ],
        atol=1e-5,
    )
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 4, 8]}))).ravel()
    np.testing.assert_allclose(pr, [-0.961820, 0.282922, -1.281283], atol=1e-5)

    m2 = gam(
        'y ~ s(region, bs="mrf")',
        d,
        method="REML",
        xt={"region": {"nb": _MRF_POLYS_NB}},
    )
    assert m2.REML_criterion == pytest.approx(m.REML_criterion, abs=1e-9)


def test_mrf_unsupported_and_validation_raise():
    """Honest boundaries: missing xt; malformed polys vertex matrix;
    wrong-dimension penalty."""
    d = _mrf_fixture()
    with pytest.raises(ValueError, match="needs xt"):
        gam('y ~ s(region, bs="mrf")', d, method="REML")
    with pytest.raises(ValueError, match="2-column"):
        gam(
            'y ~ s(region, bs="mrf")',
            d,
            method="REML",
            xt={"region": {"polys": {"0": [[0.0, 0.0, 0.0]]}}},
        )
    with pytest.raises(ValueError, match="expected"):
        gam(
            'y ~ s(region, bs="mrf")',
            d,
            method="REML",
            xt={"region": {"penalty": np.eye(5)}},
        )


# =============================================================================
# B7 — print(gam) mgcv layout (mgcv.r:2443-2467, print.gam) and the no-penalty
# (RE)ML/GCV score. Family/Formula → per-smooth estimated edf (round-4/3-sig,
# 7 per line, ``total =``) → ``{method} score:`` → optional ``rank: r/p``. Each
# repr is pinned BYTE-FOR-BYTE against mgcv 1.9-4's print() output; the score
# values are mgcv-exact (gam fits match mgcv). Pins: mgcv 1.9-4.
# -----------------------------------------------------------------------------


def _repr_fixture_11() -> pl.DataFrame:
    """set.seed(11); runif(x), runif(x2), runif(x3 unused), rnorm(y), rpois(yc)
    — the bit-exact R stream (hea.R.rng)."""
    from hea.R.rng import RGenerator

    g = RGenerator(11)
    n = 220
    x = g.uniform(0, 1, n)
    x2 = g.uniform(0, 1, n)
    g.uniform(0, 1, n)  # x3 — consumes the stream
    grp = np.array(["a", "b"] * (n // 2))
    y = np.sin(2 * np.pi * x) + 0.6 * x2 + g.normal(0, 1, n) * 0.3
    yc = g.poisson(np.exp(0.4 + np.sin(2 * np.pi * x)))
    return pl.DataFrame({"y": y, "x": x, "x2": x2, "grp": grp, "yc": yc})


def test_print_gam_layout_matches_mgcv():
    """print.gam layout pinned byte-for-byte vs mgcv: REML / multi-smooth+factor
    / GCV (scale unknown) / no-smooth (Total model d.f.) / UBRE (poisson)."""
    d = _repr_fixture_11()

    assert repr(gam("y ~ s(x)", d, method="REML")).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "y ~ s(x)",
        "",
        "Estimated degrees of freedom:",
        "6.92  total = 7.92 ",
        "",
        "REML score: 101.8412     ",
    ]
    assert repr(gam("y ~ s(x) + s(x2) + grp", d, method="REML")).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "y ~ s(x) + s(x2) + grp",
        "",
        "Estimated degrees of freedom:",
        "7.28 1.49  total = 10.77 ",
        "",
        "REML score: 74.98641     ",
    ]
    assert repr(gam("y ~ s(x) + s(x2)", d)).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "y ~ s(x) + s(x2)",
        "",
        "Estimated degrees of freedom:",
        "6.08 1.43  total = 8.51 ",
        "",
        "GCV score: 0.1006444     ",
    ]
    assert repr(gam("y ~ x + x2", d, method="REML")).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "y ~ x + x2",
        "Total model degrees of freedom 3 ",
        "",
        "REML score: 185.1315     ",
    ]
    assert repr(gam("yc ~ s(x)", d, family=Poisson())).split("\n") == [
        "",
        "Family: poisson ",
        "Link function: log ",
        "",
        "Formula:",
        "yc ~ s(x)",
        "",
        "Estimated degrees of freedom:",
        "4.73  total = 5.73 ",
        "",
        "UBRE score: 0.1213043     ",
    ]


def test_print_gam_layout_wrap_general_and_rank():
    """The fiddly print.gam corners: edf wrapping (7 per line, >7 smooths), a
    general family (multi-link + one-sided LP formula), and the rank-deficient
    ``rank: r/p`` tail. All pinned byte-for-byte vs mgcv 1.9-4."""
    from hea.family import gaulss
    from hea.R.rng import RGenerator

    g = RGenerator(21)
    n = 400
    X = g.uniform(0, 1, n * 8).reshape(n, 8, order="F")  # R fills column-major
    cols = {f"v{j}": X[:, j] for j in range(8)}
    v0, v1, v2 = cols["v0"], cols["v1"], cols["v2"]
    y = np.sin(2 * np.pi * v0) + np.cos(2 * np.pi * v1) + v2 + g.normal(0, 1, n) * 0.4
    y2 = 2 + 0.5 * v0 + g.normal(0, 1, n) * (0.2 + 0.4 * v1)
    d = pl.DataFrame({**cols, "y": y, "y2": y2})

    mw = gam(
        "y ~ s(v0) + s(v1) + s(v2) + s(v3) + s(v4) + s(v5) + s(v6) + s(v7)",
        d,
        method="REML",
    )
    assert repr(mw).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "y ~ s(v0) + s(v1) + s(v2) + s(v3) + s(v4) + s(v5) + s(v6) + s(v7)",
        "",
        "Estimated degrees of freedom:",
        "7.18 6.98 1.00 1.80 1.00 1.00 1.00 ",  # wraps after the 7th
        "1.86  total = 22.82 ",
        "",
        "REML score: 242.7806     ",
    ]

    mg = gam(["y2 ~ s(v0)", "~s(v1)"], d, family=gaulss(), method="REML")
    assert repr(mg).split("\n") == [
        "",
        "Family: gaulss ",
        "Link function: identity logb ",
        "",
        "Formula:",
        "y2 ~ s(v0)",
        "~s(v1)",
        "",
        "Estimated degrees of freedom:",
        "1.00 1.46  total = 4.46 ",
        "",
        "REML score: 199.7328     ",
    ]

    g2 = RGenerator(1)
    m = 60
    xa = g2.uniform(0, 1, m)
    xb = 2 * xa
    yy = xa + g2.normal(0, 1, m) * 0.2
    dr = pl.DataFrame({"yy": yy, "xa": xa, "xb": xb})
    with pytest.warns(UserWarning, match="rank deficient"):
        mr = gam("yy ~ xa + xb", dr, method="REML")
    assert repr(mr).split("\n") == [
        "",
        "Family: gaussian ",
        "Link function: identity ",
        "",
        "Formula:",
        "yy ~ xa + xb",
        "Total model degrees of freedom 2 ",
        "",
        "REML score: -14.09846     rank: 2/3",
    ]


def test_no_penalty_score_matches_mgcv():
    """No-smooth (no-penalty) fits still report mgcv's (RE)ML/GCV/UBRE score —
    the criterion's φ̂ is profiled exactly as mgcv's reduction-to-Gaussian does
    (REML: Dp/(n−Mp), ML: Dp/n; GCV/UBRE estimate φ internally). Pins: mgcv
    1.9-4. Previously hea returned NaN here."""
    from hea.R.rng import RGenerator

    g = RGenerator(11)
    n = 220
    x = g.uniform(0, 1, n)
    x2 = g.uniform(0, 1, n)
    g.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + 0.6 * x2 + g.normal(0, 1, n) * 0.3
    yc = g.poisson(np.exp(0.4 + 0.7 * x))
    d = pl.DataFrame({"y": y, "x": x, "x2": x2, "yc": yc})

    assert gam("y ~ x + x2", d, method="REML").REML_criterion / 2 == pytest.approx(
        185.1315, abs=1e-4
    )
    assert gam("y ~ x + x2", d, method="ML").ML_criterion / 2 == pytest.approx(
        180.5841, abs=1e-4
    )
    assert gam("y ~ x + x2", d).GCV_score == pytest.approx(0.3107573, abs=1e-7)
    assert gam(
        "yc ~ x", d, family=Poisson(), method="REML"
    ).REML_criterion / 2 == pytest.approx(383.5616, abs=1e-4)
    assert gam("yc ~ x", d, family=Poisson()).GCV_score == pytest.approx(
        0.154332, abs=1e-6
    )


def test_softplus_poisson_gam_matches_mgcv():
    """Poisson gam with the softplus link (Thread A), R-pinned EXACT against
    mgcv 1.9-4. mgcv has no softplus link, but ``fix.family.link`` returns the
    family unchanged when it already carries d2link/d3link/d4link — so the
    oracle is an mgcv gam whose family was handed the same analytic softplus
    link derivatives hea uses (``tests/r_oracle/softplus_link.R``). Validates
    the full REML path (non-canonical inner Newton + the link's 2nd
    derivative), not just the link algebra."""
    from hea.R.rng import RGenerator

    g = RGenerator(3)
    n = 200
    x = g.uniform(0.0, 1.0, n)
    y = g.poisson(np.log1p(np.exp(1.0 + 1.5 * np.sin(2 * np.pi * x)))).astype(float)
    d = pl.DataFrame({"y": y, "x": x})
    m = gam("y ~ s(x)", d, family=Poisson(link="softplus"), method="REML")
    assert float(np.sum(m.edf)) == pytest.approx(5.718371, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(306.8628, abs=1e-3)
    assert float(m.deviance) == pytest.approx(231.490028, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(1.2129598, abs=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:4],
        [2.3735475, 0.5884992, 2.1839568, 2.5833613],
        atol=1e-5,
    )


# ===========================================================================
# Phase-4 (F1) — method= extensions: GACV.Cp + P-REML / P-ML
#
# mgcv adds four criteria over hea's {REML, ML, GCV.Cp}: GACV.Cp, P-REML,
# P-ML (F1, ported here) and NCV/QNCV (F2, deferred). The new surface is only
# the scale-UNKNOWN standard-family path — for known scale (binomial/poisson,
# or gam(scale>0)) GACV.Cp degenerates to UBRE and P-REML/P-ML to REML/ML
# (mgcv.r:1956-1970), and extended families coerce everything to REML
# (mgcv.r:1892). Pins: mgcv 1.9-4.
# ===========================================================================


def _score_of(b):
    """The optimized smoothness criterion (mgcv's b$gcv.ubre), whatever it is."""
    if b.method in ("REML", "P-REML"):
        return b.REML_criterion / 2.0
    if b.method in ("ML", "P-ML"):
        return b.ML_criterion / 2.0
    return b.GCV_score


def test_preml_pml_gacv_gaussian_match_mgcv():
    """mcycle gaussian-identity, the three new criteria pinned to mgcv 1.9-4
    (gam(accel ~ s(times, k=20), method=...)). sp tolerance is loose-ish —
    the optimum is flat in log λ so the converged sp carries ~1e-5 stopping
    noise (the same character as REML) — while edf/scale/score pin tight."""
    d = load_dataset("MASS", "mcycle")
    pins = {
        "P-REML": (0.001201225072, 13.34821187, 511.085904, 618.2469636),
        "P-ML": (0.00125673145, 13.23891672, 511.118452, 624.9696944),
        "GACV.Cp": (0.002051451768, 12.06929732, 513.1780453, 559.7472062),
    }
    for method, (sp_t, edf_t, sc_t, score_t) in pins.items():
        b = gam("accel ~ s(times, k=20)", d, method=method)
        assert float(b.sp[0]) == pytest.approx(sp_t, rel=2e-3)
        assert float(b.edf_total) == pytest.approx(edf_t, rel=1e-3)
        assert float(b.sigma_squared) == pytest.approx(sc_t, rel=1e-3)
        assert _score_of(b) == pytest.approx(score_t, rel=1e-5)


def test_preml_pml_gacv_gamma_match_mgcv():
    """trees + Gamma(log), two smooths, the three new criteria pinned to mgcv.
    The Height smooth is essentially linear (fully penalized → a huge, flat
    sp), so only the well-determined Girth sp[1] is pinned tight; edf/scale/
    score pin to ~1e-6."""
    d = load_dataset("datasets", "trees")
    pins = {
        "P-REML": (0.2044224642, 4.760659811, 0.006827033793, 78.03936884),
        "P-ML": (0.3019696627, 4.501993974, 0.006875739433, 69.64771768),
        "GACV.Cp": (0.2793119364, 4.552183321, 0.006863431057, 0.007903434983),
    }
    for method, (sp1_t, edf_t, sc_t, score_t) in pins.items():
        b = gam(
            "Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"), method=method
        )
        assert float(b.sp[1]) == pytest.approx(sp1_t, rel=1e-4)
        assert float(b.edf_total) == pytest.approx(edf_t, rel=1e-5)
        assert float(b.sigma_squared) == pytest.approx(sc_t, rel=1e-5)
        assert _score_of(b) == pytest.approx(score_t, rel=1e-6)


def test_sp_vcov_preml_pml_matches_mgcv():
    """sp.vcov (mgcv.r:4221-4234) on the P-REML/P-ML rails — the method
    gate is mgcv's {ML, P-ML, REML, P-REML, fREML} list, so the Pearson
    criteria must return solve(hess+reg), not None. Pinned to mgcv 1.9-4
    on trees/Gamma at both the default and (edge.correct=FALSE, reg=1e-2)
    signatures (edge_correct falls through on any fit without an
    edge-corrected Hessian — mgcv's own behavior); GCV fits return NULL."""
    d = load_dataset("datasets", "trees")
    pins = {
        "P-REML": (
            [[981.216889066, -1.417975141], [-1.417975141, 1.431754248]],
            [[101.223338570, -1.430350665], [-1.430350665, 1.431754074]],
        ),
        "P-ML": (
            [[977.442749011, -1.399349379], [-1.399349379, 1.430640697]],
            [[101.179725562, -1.427441803], [-1.427441803, 1.430639797]],
        ),
    }
    for method, (V_t, V2_t) in pins.items():
        b = gam(
            "Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"), method=method
        )
        np.testing.assert_allclose(b.sp_vcov(), V_t, rtol=1e-6)
        np.testing.assert_allclose(
            b.sp_vcov(edge_correct=False, reg=1e-2), V2_t, rtol=1e-6
        )
    bg = gam(
        "Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"), method="GCV.Cp"
    )
    assert bg.sp_vcov() is None


def test_mroot_svd_matches_mgcv():
    """mroot(method=) (mgcv.r:4444-4470): the "svd" branch (symmetric
    eigen, rank from values > max·eps), the non-symmetric stop, and the
    unknown-method stop. Factor pinned to mgcv 1.9-4 (set.seed(1),
    A = B0 B0' with B0 = matrix(rnorm(28), 7, 4) — LAPACK eigenvector
    signs agree)."""
    from hea.models.gam import _mroot
    from hea.R.rng import RGenerator

    g = RGenerator(1)
    B0 = g.normal(0.0, 1.0, 28).reshape(4, 7).T  # R's col-major fill
    A = B0 @ B0.T
    Bs = _mroot(A, method="svd")
    assert Bs.shape == (7, 4)  # rank detected = 4
    np.testing.assert_allclose(Bs @ Bs.T, A, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Bs[:3, 0], [-0.9172445, -0.5126732, 1.695191], rtol=1e-6)
    Bc = _mroot(A, method="chol")
    np.testing.assert_allclose(Bc @ Bc.T, A, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="not symmetric"):
        _mroot(B0[:4, :4])
    with pytest.raises(ValueError, match="not recognised"):
        _mroot(A, method="qr")


def test_preml_pml_gacv_fixed_tweedie_match_mgcv():
    """Fixed-power Tweedie(p) is a *standard* exponential family (NOT
    extended), so GACV.Cp / P-REML / P-ML are valid for it and mgcv does not
    coerce to REML. Pinned on trees (Tweedie(p=1.5, log), Girth k=8)."""
    d = load_dataset("datasets", "trees")
    pins = {
        "GACV.Cp": (1.581523002, 3.68659722, 0.07249915935, 0.07999616904),
        "P-REML": (2.063876052, 3.513077485, 0.07298142699, 85.43315024),
        "P-ML": (4.816226123, 3.013187879, 0.07506138845, 80.60773045),
    }
    for method, (sp_t, edf_t, sc_t, score_t) in pins.items():
        b = gam("Volume ~ s(Girth, k=8)", d, family=Tweedie(p=1.5), method=method)
        assert b.method == method  # not coerced — fixed Tweedie
        assert float(b.sp[0]) == pytest.approx(sp_t, rel=2e-3)
        assert float(b.edf_total) == pytest.approx(edf_t, rel=1e-4)
        assert float(b.sigma_squared) == pytest.approx(sc_t, rel=1e-4)
        assert _score_of(b) == pytest.approx(score_t, rel=1e-5)


def _synth_poisson_frame():
    """Deterministic count frame for the known-scale reduction checks
    (self-contained — no mgcv pin needed, the assertions are reduction
    identities)."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(size=120))
    yp = rng.poisson(np.exp(0.3 + 1.2 * np.sin(2 * np.pi * x)))
    return pl.DataFrame({"x": x, "yp": yp.astype(float)})


def test_preml_collapses_to_reml_when_scale_known():
    """Known scale (poisson) collapses P-REML→REML and P-ML→ML
    (mgcv.r:1968-1970): the resolved method string flips and the fit is
    identical to the plain criterion."""
    df = _synth_poisson_frame()
    b_p = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="P-REML")
    b_r = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="REML")
    assert b_p.method == "REML"
    np.testing.assert_allclose(np.asarray(b_p.sp), np.asarray(b_r.sp), rtol=1e-9)
    assert b_p.REML_criterion == pytest.approx(b_r.REML_criterion, rel=1e-9)

    b_pm = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="P-ML")
    b_ml = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="ML")
    assert b_pm.method == "ML"
    assert b_pm.ML_criterion == pytest.approx(b_ml.ML_criterion, rel=1e-9)


def test_gacv_collapses_to_ubre_when_scale_known():
    """Known scale (poisson) makes GACV.Cp degenerate to UBRE — same fit and
    score as GCV.Cp (mgcv.r:1956)."""
    df = _synth_poisson_frame()
    b_g = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="GACV.Cp")
    b_c = gam("yp ~ s(x, k=12)", df, family=Poisson(), method="GCV.Cp")
    np.testing.assert_allclose(np.asarray(b_g.sp), np.asarray(b_c.sp), rtol=1e-7)
    assert b_g.GCV_score == pytest.approx(b_c.GCV_score, rel=1e-9)
    assert b_g._print_score()[0] == "UBRE"


def test_extended_family_coerces_exotic_methods_to_reml():
    """Estimated-power tw() (and scat/nb) is an extended.family — gam.fit4 has
    no GCV/UBRE/GACV/Pearson-Laplace path, so mgcv coerces any criterion other
    than REML/ML to REML (mgcv.r:1892). hea matches (silently, like mgcv)."""
    d = load_dataset("datasets", "trees")
    for method in ("GACV.Cp", "P-REML", "P-ML", "GCV.Cp"):
        b = gam("Volume ~ s(Girth, k=8)", d, family=tw(), method=method)
        assert b.method == "REML"


def test_ncv_method_raises_not_implemented():
    """NCV/QNCV are valid mgcv methods but are not ported."""
    d = load_dataset("MASS", "mcycle")
    for method in ("NCV", "QNCV"):
        with pytest.raises(NotImplementedError, match="NCV"):
            gam("accel ~ s(times)", d, method=method)


def test_te_multi_dim_margin_d_arg():
    """te(..., d=c(1,2)) groups covariates into margins of given dims (mgcv
    smooth.r:399-414): one sp per MARGIN, default k=5^d, cr/ps promoted to tp."""
    rng = np.random.default_rng(3)
    n = 400
    d = pl.DataFrame(
        {c: rng.uniform(size=n) for c in "xzw"} | {"y": rng.normal(size=n)}
    )
    m = gam("y ~ te(x, z, w, d=c(1,2))", d, method="REML")  # 1D lag + 2D space
    assert np.asarray(m.sp).size == 2  # one sp per margin
    m2 = gam("y ~ te(x, z, w, d=c(1,2), bs=c('cr','tp'))", d, method="REML")
    assert np.asarray(m2.sp).size == 2
    with pytest.raises(ValueError, match="sum to"):
        gam("y ~ te(x, z, w, d=c(1,1))", d, method="REML")  # sums to 2 != 3


def test_ad_adaptive_smooth_1d_2d():
    """bs="ad" adaptive smooth (mgcv): 1-D builds 5 adaptive-wiggliness
    penalties, 2-D builds a 3x3=9 grid, and >2-D raises (ad is 1-D/2-D only).
    Each penalty carries its own sp so the smoothness varies over the domain."""
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(0.0, 1.0, n)
    f = np.exp(-((x - 0.5) ** 2) / (2 * 0.05**2))  # compact bump (RF-like)
    d = pl.DataFrame(
        {
            "x": x,
            "y": f + rng.normal(0.0, 0.1, n),
            "z": rng.uniform(size=n),
            "w": rng.uniform(size=n),
        }
    )
    m1 = gam("y ~ s(x, bs='ad')", d, method="REML")
    assert np.asarray(m1.sp).size == 5  # 5 adaptive wiggliness penalties
    assert np.isfinite(np.asarray(m1.fitted_values)).all()
    m2 = gam("y ~ s(x, z, bs='ad')", d, method="REML")
    assert np.asarray(m2.sp).size == 9  # 3x3 wiggliness grid
    assert np.isfinite(np.asarray(m2.fitted_values)).all()
    with pytest.raises(ValueError, match="ad smooth"):
        gam("y ~ s(x, z, w, bs='ad')", d, method="REML")


def test_te_ad_multi_penalty_margin():
    """Multi-penalty te margin — a hea extension beyond mgcv, which refuses it
    (smooth.r:773). A 1-D cr time margin x a 2-D *ad* space margin: cr(1) +
    2D-ad(3x3=9) = 10 sp, each margin penalty lifted independently through
    tensor.prod.penalties. Fixed sp (single PIRLS) keeps it fast/deterministic."""
    rng = np.random.default_rng(5)
    n = 400
    d = pl.DataFrame(
        {c: rng.uniform(size=n) for c in "xzw"} | {"y": rng.normal(size=n)}
    )
    m = gam(
        "y ~ te(x, z, w, d=c(1,2), bs=c('cr','ad'), k=c(5,5))",
        d,
        method="REML",
        sp=[1.0] * 10,
    )
    assert np.asarray(m.sp).size == 10
    assert np.isfinite(np.asarray(m.coef)).all()


# ===========================================================================
# negbin — fixed-θ NB through gam (audit-2 B10; mgcv negbin(),
# gam.fit3.r:2564-2642 + estimate.gam mgcv.r:1963-1979 + gam.outer:1649).
# Pins: mgcv 1.9-4 on the RGenerator(66) recipe (see test_bam.py
# ``_method_probe_frame`` for the bit-identical R data).
# ===========================================================================


def _negbin_probe_frame(n: int = 300) -> pl.DataFrame:
    """RGenerator(66) counts — bit-matches ``set.seed(66); x* <- runif(n);
    e <- rnorm(n,0,.2); mu <- 2+6*(x0-.5)^2+4*(x1-.25)^2; yp <- rpois(n,mu)``
    (same recipe as test_bam._method_probe_frame; mu dyadic-exact so the
    rpois stream is bit-identical to R's)."""
    from hea.R.rng import RGenerator

    g = RGenerator(66)
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    g.uniform(0.0, 1.0, n)  # x2 — drawn to keep the stream aligned
    g.normal(0.0, 0.2, n)  # e
    mu = 2 + 6 * (x0 - 0.5) ** 2 + 4 * (x1 - 0.25) ** 2
    yp = np.asarray(g.poisson(mu), dtype=float)
    return pl.DataFrame({"x0": x0, "x1": x1, "yp": yp})


def test_negbin_through_gam_matches_mgcv():
    """gam(negbin(2)) across methods vs mgcv 1.9-4. estimate.gam forces
    φ = 1 whatever scale= says ("scale <- 1; ## no choice", mgcv.r:1963-1966
    + 1975-1979) and GCV.Cp/GACV.Cp become UBRE; P-REML collapses to REML
    (known scale). The θ-vector form errors at fit time with gam.outer's
    message (mgcv.r:1649-1650) — the range search is deprecated.r-only."""
    from hea.family import negbin

    df = _negbin_probe_frame()
    m = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="REML")
    np.testing.assert_allclose(m.sp, [2.674656889, 7.421293859], rtol=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(5.382552732, rel=1e-7)
    assert m.REML_criterion / 2 == pytest.approx(640.428625352, rel=1e-9)
    assert m.deviance == pytest.approx(159.8165023, rel=1e-8)
    assert m.null_deviance == pytest.approx(184.3636025, rel=1e-8)
    assert m.AIC == pytest.approx(1275.357879, rel=1e-8)
    assert m.scale_estimated is False and m.sigma_squared == 1.0
    assert float(np.asarray(m.coefficients)[0]) == pytest.approx(
        1.11546079869, rel=1e-7
    )
    assert float(m.Vp[0, 0]) == pytest.approx(0.00278642638887, rel=1e-6)
    np.testing.assert_allclose(
        m.fitted_values[:3], [3.47125220165, 3.15303269855, 2.27727871824], rtol=1e-8
    )
    assert m.family.name == "Negative Binomial(2)"

    m_u = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="GCV.Cp")
    np.testing.assert_allclose(m_u.sp, [4.34280477, 9.294872256], rtol=1e-6)
    assert float(np.sum(m_u.edf)) == pytest.approx(4.995203191, rel=1e-7)
    assert m_u.GCV_score == pytest.approx(-0.431835854591, rel=1e-9)
    assert m_u.AIC == pytest.approx(1273.055941, rel=1e-8)
    for kw in ({"scale": -1.0}, {"scale": 5.0}):
        m_s = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="GCV.Cp", **kw)
        np.testing.assert_allclose(m_s.sp, m_u.sp, rtol=0)
        assert m_s.GCV_score == m_u.GCV_score
        assert m_s.sigma_squared == 1.0
    m_ga = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="GACV.Cp")
    np.testing.assert_allclose(m_ga.sp, m_u.sp, rtol=0)
    assert m_ga.GCV_score == m_u.GCV_score

    m_ml = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="ML")
    assert float(m_ml.sp[0]) == pytest.approx(4.765964897, rel=1e-5)
    assert float(m_ml.sp[1]) == pytest.approx(36896.5181, rel=1e-3)
    assert m_ml.ML_criterion / 2 == pytest.approx(636.030243645, rel=1e-9)
    m_pr = gam("yp ~ s(x0) + s(x1)", df, family=negbin(2), method="P-REML")
    assert m_pr.method == "REML"
    np.testing.assert_allclose(m_pr.sp, m.sp, rtol=0)

    m_sq = gam("yp ~ s(x0) + s(x1)", df, family=negbin(3.7, link="sqrt"), method="REML")
    np.testing.assert_allclose(m_sq.sp, [3.11556099, 6.800298143], rtol=1e-6)
    assert m_sq.REML_criterion / 2 == pytest.approx(617.091489985, rel=1e-9)
    assert m_sq.deviance == pytest.approx(205.8400679, rel=1e-8)
    assert m_sq.AIC == pytest.approx(1226.504486, rel=1e-8)
    assert m_sq.family.name == "Negative Binomial(3.7)"
    m_iv = gam(
        "yp ~ s(x0) + s(x1)", df, family=negbin(2, link="inverse"), method="REML"
    )
    np.testing.assert_allclose(m_iv.sp, [44.36693182, 196.9028291], rtol=1e-6)
    assert m_iv.REML_criterion / 2 == pytest.approx(644.704911279, rel=1e-9)

    with pytest.raises(ValueError, match="single value for theta or use nb"):
        gam("yp ~ s(x0)", df, family=negbin([2, 9]), method="REML")


# ===========================================================================
# cpois — censored Poisson through gam (audit-2 B11; mgcv cpois(),
# efam.r:344-537 + dppois:312-339). Pins: mgcv 1.9-4 on the RGenerator(66)
# recipe with deterministic yp%%7 censoring. The free-fit sp pins carry the
# outer-Newton ENDPOINT scatter (both sides stop at "full convergence" with
# |grad| ~1e-5 on a flat criterion); the criterion SURFACE is pinned tight
# through fixed-sp refits (receipts: hea at R's sp 451.70535520573 ≡ R to
# 2e-13, R at hea's sp 451.70535520700747 ≡ hea to 4e-12; ML likewise).
# ===========================================================================


def _cpois_probe_frame(n: int = 300) -> pl.DataFrame:
    """RGenerator(66) censored counts — bit-matches ``set.seed(66)`` +
    the ``_negbin_probe_frame`` stream, then deterministic censoring:
    ``m7 <- yp %% 7``; m7==1 → interval ``[yp, yp+3]``, m7==2 → left
    (−∞), m7==3 → right (+∞), else uncensored. 127 unc (incl. 20 zero
    counts — the mustart-0 quirk rows), 44 int, 73 left, 56 right."""
    from hea.R.rng import RGenerator

    g = RGenerator(66)
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    g.uniform(0.0, 1.0, n)  # x2 — drawn to keep the stream aligned
    g.normal(0.0, 0.2, n)  # e
    mu = 2 + 6 * (x0 - 0.5) ** 2 + 4 * (x1 - 0.25) ** 2
    yp = np.asarray(g.poisson(mu), dtype=float)
    m7 = yp % 7
    yat = yp.copy()
    yat[m7 == 1] = yp[m7 == 1] + 3
    yat[m7 == 2] = -np.inf
    yat[m7 == 3] = np.inf
    return pl.DataFrame({"x0": x0, "x1": x1, "yp": yp, "yat": yat})


def test_cpois_through_gam_matches_mgcv():
    """gam(cbind(y,yat), cpois()) vs mgcv 1.9-4: REML/ML free fits +
    fixed-sp criterion-surface pins, the extended-family method coercion,
    sqrt link, and predict(se)."""
    from hea.family import cpois

    df = _cpois_probe_frame()
    m = gam("cbind(yp, yat) ~ s(x0) + s(x1)", df, family=cpois(), method="REML")
    np.testing.assert_allclose(m.sp, [0.334933439083, 0.471071335086], rtol=5e-5)
    assert float(np.sum(m.edf)) == pytest.approx(10.680586348, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(451.705355206, rel=1e-9)
    assert m.deviance == pytest.approx(443.753510916, rel=1e-6)
    assert m.null_deviance == pytest.approx(517.252889711, rel=1e-9)
    assert m.AIC == pytest.approx(886.522280925, rel=1e-6)
    assert float(np.asarray(m.coefficients)[0]) == pytest.approx(
        1.27449709949, rel=1e-6
    )
    assert float(m.Vp[0, 0]) == pytest.approx(0.00116059616683, rel=1e-5)
    np.testing.assert_allclose(
        m.fitted_values[:3], [3.42613761818, 4.58566853818, 2.32324213843], rtol=1e-5
    )
    assert m.family.name == "cpois"
    assert m.scale_estimated is False and m.sigma_squared == 1.0
    m_fix = gam(
        "cbind(yp, yat) ~ s(x0) + s(x1)",
        df,
        family=cpois(),
        method="REML",
        sp=[0.33493343908293738, 0.4710713350860859],
    )
    assert m_fix.REML_criterion / 2 == pytest.approx(451.70535520573532, rel=1e-12)

    nd = pl.DataFrame({"x0": [0.25, 0.75], "x1": [0.4, 0.1]})
    pr = m.predict(nd, type="link", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [1.1551725162, 1.19871334477], rtol=1e-5
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(), [0.0997157367767, 0.113985014911], rtol=1e-4
    )

    # Extended families silently coerce non-REML/ML/NCV methods to REML
    # (mgcv.r:1892) — same optimum as the REML fit, method relabeled.
    m_g = gam("cbind(yp, yat) ~ s(x0) + s(x1)", df, family=cpois(), method="GCV.Cp")
    assert m_g.method == "REML"
    np.testing.assert_allclose(m_g.sp, m.sp, rtol=0)

    m_ml = gam("cbind(yp, yat) ~ s(x0) + s(x1)", df, family=cpois(), method="ML")
    assert float(m_ml.sp[0]) == pytest.approx(0.753071814931, rel=1e-6)
    assert float(m_ml.sp[1]) > 1e4
    assert m_ml.ML_criterion / 2 == pytest.approx(447.311345431, rel=1e-7)
    m_mlfix = gam(
        "cbind(yp, yat) ~ s(x0) + s(x1)",
        df,
        family=cpois(),
        method="ML",
        sp=[0.753071814931, 295490.938456],
    )
    assert m_mlfix.ML_criterion / 2 == pytest.approx(447.311345431, rel=1e-11)

    m_sq = gam(
        "cbind(yp, yat) ~ s(x0) + s(x1)", df, family=cpois(link="sqrt"), method="REML"
    )
    np.testing.assert_allclose(m_sq.sp, [0.419650906293, 0.590390103814], rtol=3e-4)
    assert m_sq.REML_criterion / 2 == pytest.approx(451.786394307, rel=1e-9)
    assert m_sq.deviance == pytest.approx(445.244665235, rel=1e-5)
    assert m_sq.AIC == pytest.approx(887.363214464, rel=1e-6)
    assert float(np.asarray(m_sq.coefficients)[0]) == pytest.approx(
        1.90180298416, rel=1e-6
    )


def test_cpois_uncensored_matches_mgcv_and_poisson():
    """1-column response ⇒ all-uncensored: the cpois likelihood is the
    plain Poisson one, and the REML criterion lands on gam(poisson())'s
    to all printed digits (mgcv receipt: 600.190785786 both ways). The
    well-conditioned surface also pins sp at 1e-6 here."""
    from hea.family import cpois

    df = _cpois_probe_frame()
    m = gam("yp ~ s(x0) + s(x1)", df, family=cpois(), method="REML")
    np.testing.assert_allclose(m.sp, [2.18314684842, 3.233362839], rtol=1e-6)
    assert m.REML_criterion / 2 == pytest.approx(600.190785786, rel=1e-10)
    assert m.deviance == pytest.approx(343.739701203, rel=1e-10)
    assert m.null_deviance == pytest.approx(411.270336126, rel=1e-9)
    assert m.AIC == pytest.approx(1189.8806843, rel=1e-9)
    assert float(np.asarray(m.coefficients)[0]) == pytest.approx(
        1.11582317903, rel=1e-8
    )
    from hea.family import Poisson

    mp = gam("yp ~ s(x0) + s(x1)", df, family=Poisson(), method="REML")
    assert m.REML_criterion / 2 == pytest.approx(mp.REML_criterion / 2, rel=1e-10)


# ===========================================================================
# clog — censored logistic through gam (audit-2 B12; mgcv clog(),
# efam.r:2192-2612). Pins: mgcv 1.9-4 on the RGenerator(66) recipe with
# deterministic floor(y*10)%%7 censoring; identity link (the default), the
# log-scale θ estimated jointly (free-fit values matched R at ~1e-11 at pin
# time — well-conditioned surface, unlike cpois' flat one).
# ===========================================================================


def _clog_probe_frame(n: int = 300) -> pl.DataFrame:
    """RGenerator(66) censored continuous response — bit-matches
    ``set.seed(66); x0,x1,x2 <- runif; e <- rnorm(n,0,.2);
    y <- f + 2.5*e`` then m7 = floor(y*10) %% 7: m7==1 → interval
    [y, y+1.5], m7==2 → left (−∞), m7==3 → right (+∞). 164 unc / 46
    int / 36 left / 54 right."""
    from hea.R.rng import RGenerator

    g = RGenerator(66)
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    g.uniform(0.0, 1.0, n)  # x2 — stream alignment
    e = g.normal(0.0, 0.2, n)
    f = 2 + 6 * (x0 - 0.5) ** 2 + 4 * (x1 - 0.25) ** 2
    y = f + 2.5 * e
    m7 = np.floor(y * 10) % 7
    yat = y.copy()
    yat[m7 == 1] = y[m7 == 1] + 1.5
    yat[m7 == 2] = -np.inf
    yat[m7 == 3] = np.inf
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y, "yat": yat})


def test_clog_through_gam_matches_mgcv():
    """gam(cbind(y,yat), clog()) vs mgcv 1.9-4: REML/ML with the
    log-scale θ estimated jointly, fixed-θ, and predict(se). Note the
    NEGATIVE AICs — clog's aic slot carries only the saturated pieces
    (mgcv reports it verbatim; replicated bug-for-bug)."""
    from hea.family import clog

    df = _clog_probe_frame()
    m = gam("cbind(y, yat) ~ s(x0) + s(x1)", df, family=clog(), method="REML")
    np.testing.assert_allclose(m.sp, [0.503914238939, 0.820442672981], rtol=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(9.30196740092, rel=1e-8)
    assert m.REML_criterion / 2 == pytest.approx(252.18910703663536, rel=1e-10)
    assert m.deviance == pytest.approx(353.679863947, rel=1e-9)
    assert m.null_deviance == pytest.approx(718.82482886, rel=1e-9)
    assert m.AIC == pytest.approx(-519.601936491, rel=1e-9)
    assert float(np.asarray(m.coefficients)[0]) == pytest.approx(3.1725799995, rel=1e-9)
    assert float(m.Vp[0, 0]) == pytest.approx(0.00132715696638, rel=1e-8)
    np.testing.assert_allclose(
        m.fitted_values[:3], [3.61104181178, 3.05193246991, 2.1926353359], rtol=1e-9
    )
    np.testing.assert_allclose(m.family.get_theta(True), [0.334666712875], rtol=1e-8)
    assert m._family_display_name() == "clog(0.335)"
    assert m.scale_estimated is False and m.sigma_squared == 1.0

    nd = pl.DataFrame({"x0": [0.25, 0.75], "x1": [0.4, 0.1]})
    pr = m.predict(nd, type="link", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [2.55649773675, 2.66064951599], rtol=1e-9
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(), [0.0948692128224, 0.110975005354], rtol=1e-8
    )

    m_ml = gam("cbind(y, yat) ~ s(x0) + s(x1)", df, family=clog(), method="ML")
    np.testing.assert_allclose(m_ml.sp, [0.627031555174, 0.99948304632], rtol=1e-6)
    assert m_ml.ML_criterion / 2 == pytest.approx(249.069524633, rel=1e-10)
    assert m_ml.AIC == pytest.approx(-525.157185499, rel=1e-9)
    np.testing.assert_allclose(m_ml.family.get_theta(True), [0.333061106696], rtol=1e-8)

    m_f = gam(
        "cbind(y, yat) ~ s(x0) + s(x1)", df, family=clog(theta=0.6), method="REML"
    )
    assert m_f.family.n_theta == 0
    np.testing.assert_allclose(m_f.sp, [0.530146384137, 0.857154330129], rtol=1e-6)
    assert m_f.REML_criterion / 2 == pytest.approx(289.556661799, rel=1e-10)
    assert m_f.deviance == pytest.approx(212.387856373, rel=1e-9)
    assert m_f.AIC == pytest.approx(51.1549557883, rel=1e-9)
    np.testing.assert_allclose(m_f.family.get_theta(True), [0.6], rtol=0)


def test_clog_uncensored_matches_mgcv():
    """1-column response ⇒ all-uncensored logistic regression with σ
    estimated (mgcv receipt: σ̂ 0.269056124913, crit 218.917036874)."""
    from hea.family import clog

    df = _clog_probe_frame()
    m = gam("y ~ s(x0) + s(x1)", df, family=clog(), method="REML")
    np.testing.assert_allclose(m.sp, [0.673977297993, 0.8092702128], rtol=1e-6)
    assert m.REML_criterion / 2 == pytest.approx(218.917036874, rel=1e-10)
    assert m.deviance == pytest.approx(352.523701203, rel=1e-9)
    assert m.null_deviance == pytest.approx(1019.77530518, rel=1e-9)
    assert m.AIC == pytest.approx(-720.518600427, rel=1e-9)
    np.testing.assert_allclose(m.family.get_theta(True), [0.269056124913], rtol=1e-8)


# ===========================================================================
# bcg — censored Box-Cox Gaussian through gam (audit-2 B13; mgcv bcg(),
# efam.r:1477-2170). Pins: mgcv 1.9-4 on the RGenerator(66) recipe with a
# positive lognormal-ish response and deterministic floor(y*10)%%7
# censoring under bcg's conventions (left = yat≤0). Both θ = (λ, log σ)
# estimated jointly — the first 2-θ censored family through the outer
# Newton. NOTE mgcv's own bcg(theta=c(λ,σ>0)) FIXED-θ fit crashes inside
# gam.fit4 ("subscript out of bounds": the length-3 .Theta quirk breaks
# its nt bookkeeping), so only free-θ fits are pinnable.
# ===========================================================================


def _bcg_probe_frame(n: int = 300) -> pl.DataFrame:
    """RGenerator(66) censored positive response — bit-matches
    ``set.seed(66)``: ``y <- exp(f/3 + e)``, then m7 = floor(y*10)%%7:
    m7==1 → interval [y, 1.5y], m7==2 → LEFT (yat=0), m7==3 → right
    (+∞). 154 unc / 46 int / 49 left / 51 right."""
    from hea.R.rng import RGenerator

    g = RGenerator(66)
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    g.uniform(0.0, 1.0, n)  # x2 — stream alignment
    e = g.normal(0.0, 0.2, n)
    f = 2 + 6 * (x0 - 0.5) ** 2 + 4 * (x1 - 0.25) ** 2
    y = np.exp(f / 3 + e)
    m7 = np.floor(y * 10) % 7
    yat = y.copy()
    yat[m7 == 1] = y[m7 == 1] * 1.5
    yat[m7 == 2] = 0.0
    yat[m7 == 3] = np.inf
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y, "yat": yat})


def test_bcg_through_gam_matches_mgcv():
    """gam(cbind(y,yat), bcg()) vs mgcv 1.9-4: REML/ML with (λ, log σ)
    estimated jointly, predict(se), and the bc-scale deviance-residual
    sign (mgcv's attr(d,"sign") through residuals_extended)."""
    from hea.family import bcg

    df = _bcg_probe_frame()
    m = gam("cbind(y, yat) ~ s(x0) + s(x1)", df, family=bcg(), method="REML")
    np.testing.assert_allclose(m.sp, [2.87916620956, 2.31564146682], rtol=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(9.77314629757, rel=1e-8)
    assert m.REML_criterion / 2 == pytest.approx(264.60403262326531, rel=1e-10)
    assert m.deviance == pytest.approx(323.212958908, rel=1e-9)
    assert m.null_deviance == pytest.approx(711.364321966, rel=1e-9)
    assert m.AIC == pytest.approx(672.225593252, rel=1e-9)
    assert float(np.asarray(m.coefficients)[0]) == pytest.approx(
        1.13843516264, rel=1e-8
    )
    assert float(m.Vp[0, 0]) == pytest.approx(0.000257925642007, rel=1e-8)
    np.testing.assert_allclose(
        m.fitted_values[:3], [1.31740169455, 1.05356113639, 0.720764722064], rtol=1e-9
    )
    np.testing.assert_allclose(
        m.family.get_theta(True), [0.14167620229, 0.252280924883], rtol=1e-8
    )
    assert m._family_display_name() == "bcg(0.142,0.252)"

    nd = pl.DataFrame({"x0": [0.25, 0.75], "x1": [0.4, 0.1]})
    pr = m.predict(nd, type="link", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), [0.898999603746, 0.917437138588], rtol=1e-9
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(), [0.0435961564601, 0.0497767557132], rtol=1e-8
    )
    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [
            -1.46108218233,
            0.0254703400613,
            -0.761915784925,
            1.05636224341,
            -1.27588758661,
        ],
        rtol=1e-9,
    )

    m_ml = gam("cbind(y, yat) ~ s(x0) + s(x1)", df, family=bcg(), method="ML")
    np.testing.assert_allclose(m_ml.sp, [4.03433822373, 3.60488363604], rtol=1e-6)
    assert m_ml.ML_criterion / 2 == pytest.approx(259.058763248, rel=1e-10)
    assert m_ml.AIC == pytest.approx(673.705114094, rel=1e-9)
    np.testing.assert_allclose(
        m_ml.family.get_theta(True), [0.0719738141189, 0.233744887195], rtol=1e-7
    )


def test_bcg_uncensored_matches_mgcv():
    """1-column response ⇒ all-uncensored Box-Cox regression. This path
    has Deta3 ≡ 0 (gaussian-like uncensored case) with free θ — the
    needs_w=False + θ-rows corner that used to crash `_reml_hessian`
    (unbound K)."""
    from hea.family import bcg

    df = _bcg_probe_frame()
    m = gam("y ~ s(x0) + s(x1)", df, family=bcg(), method="REML")
    np.testing.assert_allclose(m.sp, [4.16011737382, 4.75457719152], rtol=1e-6)
    assert m.REML_criterion / 2 == pytest.approx(248.098356183, rel=1e-10)
    assert m.deviance == pytest.approx(290.271254572, rel=1e-9)
    assert m.null_deviance == pytest.approx(887.11924597, rel=1e-9)
    assert m.AIC == pytest.approx(869.100294135, rel=1e-9)
    np.testing.assert_allclose(
        m.family.get_theta(True), [0.123209217613, 0.214649223507], rtol=1e-8
    )


# ===========================================================================
# gfam (grouped families) — mgcv gfam() (gfam.r:3-604) through gam. The
# response is cbind(y, index); component scale parameters join θ as
# log-scales. Pins: live mgcv 1.9-4 on the RGenerator(66)-matched frame
# below (runif/rnorm/rpois streams bit-identical; index and tw/binomial
# responses dyadic-exact transforms of the shared rpois stream).
# ===========================================================================


def _gfam_probe_frame(n: int = 210):
    """Bit-matches the R recipe (set.seed(66)):"""
    from hea.R.rng import RGenerator

    g = RGenerator(66)
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 1.0, n)
    x2 = g.uniform(0.0, 1.0, n)
    e = g.normal(0.0, 0.2, n)
    f = np.sin(2 * np.pi * x0) * 0.7 + (x1 - 0.5) ** 2 * 2 + 0.3 * np.sin(np.pi * x2)
    mu = 2 + 6 * (x0 - 0.5) ** 2 + 4 * (x1 - 0.25) ** 2
    yp = np.asarray(g.poisson(mu), dtype=float)
    fin = yp % 3 + 1
    y = np.zeros(n)
    i1, i2, i3 = fin == 1, fin == 2, fin == 3
    y[i1] = (yp[i1] > 2).astype(float)
    y[i2] = (yp[i2] / 4 + np.round(x2[i2] * 128) / 128) * (yp[i2] != 3)
    y[i3] = (f + e)[i3]
    return pl.DataFrame(
        {"y": y, "fin": fin, "x0": x0, "x1": x1, "x2": x2, "yp": yp, "f": f, "e": e}
    )


def test_gfam_through_gam_matches_mgcv():
    """binomial + tw + gaussian (the ?gfam example combo): REML and ML,
    prediction with the family index supplied in newdata. The tw power
    runs into its p→b boundary here (p̂ ≈ 1.99), where the criterion is
    flat in twθ: R itself parks at twθ = 12.594 (free) vs 12.785 (its
    own fixed-sp refit, Δcrit 4.6e-8 rel) — hea's endpoint (12.17,
    Δcrit 1.5e-7) sits in the same flat tail, so twθ is pinned in
    p-space and the fit-level pins carry flat-direction tolerances.
    The exponential-pair and gaussian+tw tests below pin the same
    machinery at 1e-11..1e-14 where no flat direction exists."""
    from hea.family import Binomial, Gaussian, gfam, tw

    df = _gfam_probe_frame()
    m = gam(
        "cbind(y, fin) ~ s(x0) + s(x1) + s(x2)",
        df,
        family=gfam([Binomial(), tw(), Gaussian()]),
        method="REML",
    )
    np.testing.assert_allclose(
        m.sp, [0.36453856569, 3.55925798119, 6.79134855377], rtol=1e-8
    )
    th = m.family.get_theta()
    np.testing.assert_allclose(th[1:], [-0.934387658144, -3.156660593], rtol=1e-8)
    p_hat = m.family.get_fl()[1]._p_of_theta(th[0])
    assert th[0] > 8.0
    assert p_hat == pytest.approx(1.9899973520, abs=5e-5)
    assert m.REML_criterion / 2 == pytest.approx(139.128461856, rel=2e-6)
    assert float(np.sum(m.edf)) == pytest.approx(12.9027518681, rel=1e-6)
    assert m.deviance == pytest.approx(217.86886093, rel=5e-6)
    assert m.null_deviance == pytest.approx(98.9895272305, rel=1e-6)
    assert m.AIC == pytest.approx(263.478873347, rel=5e-6)
    assert m._family_display_name() == "gfam{binomial,Tweedie(p=1.99),gaussian}"
    np.testing.assert_allclose(
        m.fitted_values[:5],
        [
            0.645263508154,
            -0.181620539483,
            0.572424567087,
            2.06991168691,
            -0.227087476246,
        ],
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [
            0.936051821046,
            -0.888627642797,
            1.05629002909,
            0.169576783767,
            -0.227785060989,
        ],
        rtol=5e-5,
    )

    # response-scale prediction: the family index rides newdata under
    # the second cbind arg (mgcv reads it off the newdata response,
    # mgcv.r:2819 → gfam.r:493-498).
    nd = pl.DataFrame(
        {
            "fin": [1.0, 2, 3, 2],
            "x0": [0.2, 0.5, 0.8, 0.35],
            "x1": [0.3, 0.6, 0.4, 0.7],
            "x2": [0.25, 0.45, 0.85, 0.15],
        }
    )
    pr = m.predict(nd, se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [0.696213221734, 1.43883576842, -0.372387039487, 2.01534813817],
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.0152952979359, 0.111422504552, 0.0836168410102, 0.145180900319],
        rtol=5e-6,
    )
    np.testing.assert_allclose(
        m.predict(nd, type="link")["fit"].to_numpy(),
        [0.829329898447, 0.363834292425, -0.372387039487, 0.700791953759],
        rtol=5e-6,
    )
    # no family index anywhere → gfam's own error (gfam.r:492).
    with pytest.raises(ValueError, match="no family index"):
        m.predict(nd.drop("fin"))
    with pytest.raises(NotImplementedError):
        m.residuals_of("pearson")

    m2 = gam(
        "cbind(y, fin) ~ s(x0) + s(x1) + s(x2)",
        df,
        family=gfam([Binomial(), tw(), Gaussian()]),
        method="ML",
    )
    np.testing.assert_allclose(
        m2.sp, [0.406760049566, 4.67356472978, 11.4456735007], rtol=1e-7
    )
    np.testing.assert_allclose(
        m2.family.get_theta()[1:], [-0.933804586328, -3.18907237988], rtol=1e-8
    )
    assert m2.ML_criterion / 2 == pytest.approx(133.735499889, rel=2e-6)
    assert m2._family_display_name() == "gfam{binomial,Tweedie(p=1.99),gaussian}"


def test_gfam_exponential_pair_matches_mgcv():
    """poisson + gaussian — exponential members only, so the single θ
    is the gaussian free log-scale. No flat direction: everything is
    pinned at the levels measured (θ 9e-14, criterion 6e-14)."""
    from hea.family import Gaussian, Poisson, gfam

    df = _gfam_probe_frame()
    i1 = df["fin"].to_numpy() == 1.0
    y2 = np.where(i1, df["yp"].to_numpy(), df["f"].to_numpy() + df["e"].to_numpy())
    df2 = df.with_columns(pl.Series("y", y2), pl.Series("fin", np.where(i1, 1.0, 2.0)))
    m = gam(
        "cbind(y, fin) ~ s(x0) + s(x1) + s(x2)",
        df2,
        family=gfam([Poisson(), Gaussian()]),
        method="REML",
    )
    np.testing.assert_allclose(
        m.sp, [0.362319563763, 5.72221508129, 6.29915779122], rtol=1e-8
    )
    np.testing.assert_allclose(m.family.get_theta(), [-3.35392636346], rtol=1e-10)
    assert m.REML_criterion / 2 == pytest.approx(188.374808536, rel=1e-11)
    assert float(np.sum(m.edf)) == pytest.approx(15.1278425247, rel=1e-10)
    assert m.deviance == pytest.approx(364.685364739, rel=1e-11)
    assert m._family_display_name() == "gfam{poisson,gaussian}"


def test_gfam_gaussian_tw_theta_walk_quirk_matches_mgcv():
    """gaussian + tw with the tw power INTERIOR (p̂ = 1.242): tight pins
    throughout. Also the putTheta-walk quirk receipt (gfam.r:66-74): the
    R loop advances i0 only for extended members, so with the gaussian
    free-scale slot FIRST, tw's stored θ is set from the wrong positions
    — the display label's p comes from p(gauss log σ̂²), identically in
    mgcv and hea ("Tweedie(p=1.242)" instead of p(θ̂_tw) = 1.278)."""
    from hea.family import Gaussian, gfam, tw

    df = _gfam_probe_frame()
    i3 = df["fin"].to_numpy() == 3.0
    yp = df["yp"].to_numpy()
    x2 = df["x2"].to_numpy()
    y3 = np.where(
        i3, df["f"].to_numpy() + df["e"].to_numpy(), yp / 4 + np.round(x2 * 128) / 128
    )
    df3 = df.with_columns(pl.Series("y", y3), pl.Series("fin", np.where(i3, 1.0, 2.0)))
    m = gam(
        "cbind(y, fin) ~ s(x0) + s(x1) + s(x2)",
        df3,
        family=gfam([Gaussian(), tw()]),
        method="REML",
    )
    np.testing.assert_allclose(
        m.sp, [6.05489847015, 7.62000140649, 260.164562235], rtol=1e-8
    )
    np.testing.assert_allclose(
        m.family.get_theta(),
        [-1.17033174973, -1.10577520975, -1.77109552965],
        rtol=1e-9,
    )
    assert m.REML_criterion / 2 == pytest.approx(160.741637692, rel=1e-11)
    assert m._family_display_name() == "gfam{gaussian,Tweedie(p=1.242)}"


def test_gam_bam_predict_uses_fit_xlevels():
    """``predict`` on a frame missing a factor level must code the parametric
    contrast on the FIT's levels (R's ``xlevels``, lm.R:79/695) and reject an
    unseen one — for every gam/bam rail.
    """
    from hea.models.bam import bam

    rng = np.random.default_rng(5)
    n = 400
    g = rng.choice(["a", "b", "c"], n)
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(0.5 + (g == "b") * 0.4 + (g == "c") * 0.9 + np.sin(x)))
    d = pl.DataFrame({"y": y.astype(float), "g": g, "x": x}).with_columns(
        pl.col("g").cast(pl.Enum(["a", "b", "c"]))
    )
    f = "y ~ g + s(x, k=6)"
    models = {
        "gam": gam(f, d, family=Poisson()),
        "bam-dense": bam(f, d, family=Poisson(), discrete=False),
        "bam-discrete": bam(f, d, family=Poisson(), discrete=True),
    }
    allrows = pl.DataFrame({"g": ["a", "b", "c"], "x": [0.3, 0.3, 0.3]})
    for name, m in models.items():
        both = m.predict(newdata=allrows, type="response")["fit"].to_numpy()
        for i, lv in enumerate(["a", "b", "c"]):
            solo = m.predict(
                newdata=pl.DataFrame({"g": [lv], "x": [0.3]}), type="response"
            )["fit"].to_numpy()
            np.testing.assert_allclose(solo[0], both[i], rtol=1e-10, err_msg=name)
        with pytest.raises(ValueError, match="factor g has new level zz"):
            m.predict(newdata=pl.DataFrame({"g": ["zz"], "x": [0.3]}))


def test_gam_bam_predict_replays_training_predvars():
    """``poly``/``bs``/``ns``/``scale`` in the PARAMETRIC part must replay the
    fit's basis on newdata (R's ``predvars``), not be recomputed from it.
    """
    from hea.models.bam import bam

    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(0, 10, n)
    z = rng.uniform(size=n)
    d = pl.DataFrame(
        {
            "y": 1 + 0.5 * x - 0.03 * x**2 + np.sin(z) + rng.normal(0, 0.2, n),
            "x": x,
            "z": z,
        }
    )
    f = "y ~ poly(x,2) + s(z)"
    for name, m in (
        ("gam", gam(f, d)),
        ("bam-dense", bam(f, d, discrete=False)),
        ("bam-discrete", bam(f, d, discrete=True)),
    ):
        head = m.predict(newdata=d.head(3), type="response")["fit"].to_numpy()
        np.testing.assert_allclose(
            head, np.asarray(m.fitted)[:3], rtol=1e-9, err_msg=name
        )


def test_predict_checks_variable_types_against_fit():
    """R's ``.checkMFClasses`` (models.R:401-434, called from lm.R:697): a
    variable whose type changed between fit and predict is refused, with R's
    message. ``factor`` supplied where the fit saw ``character`` is allowed —
    that is what R's own model.frame coercion produces."""
    from hea.family import Poisson
    from hea.models.glm import glm

    rng = np.random.default_rng(0)
    n = 200
    g = rng.choice(["a", "b"], n)
    x = rng.uniform(size=n)
    d = pl.DataFrame(
        {
            "y": rng.poisson(np.exp(1 + (g == "b") * 1.5 + 0.3 * x)).astype(float),
            "g": g,
            "x": x,
        }
    )
    m = glm("y ~ g + x", d, family=Poisson())
    with pytest.raises(
        ValueError,
        match=r"variable 'x' was fitted with type \"numeric\" "
        r"but type \"character\" was supplied",
    ):
        m.predict(pl.DataFrame({"g": ["a"], "x": ["0.5"]}))
    with pytest.raises(
        ValueError, match=r"variables 'g', 'x' were specified with different types"
    ):
        m.predict(pl.DataFrame({"g": [1.0], "x": ["0.5"]}))
    as_enum = pl.DataFrame(
        {"g": pl.Series(["a"], dtype=pl.Enum(["a", "b"])), "x": [0.5]}
    )
    as_str = pl.DataFrame({"g": ["a"], "x": [0.5]})
    np.testing.assert_allclose(
        np.asarray(m.predict(as_enum, type="response")).ravel(),
        np.asarray(m.predict(as_str, type="response")).ravel(),
        rtol=1e-14,
    )


def test_check_reports_mgcv_method_and_optimizer():
    """``gam.check``'s header is ``b$method`` + ``b$optimizer`` (plots.r:300).
    mgcv.r:2426, a REPORTING rule: an additive model (Gaussian+identity) on a
    mgcv forces through outer looping (mgcv.r:1933). bam's own rails label
    themselves (bam.r:784/1249). Expected column is live-R output.
    """
    import contextlib
    import io

    from hea.models.bam import bam

    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(size=n)
    d = pl.DataFrame(
        {
            "x": x,
            "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n),
            "yp": rng.poisson(np.exp(np.sin(2 * np.pi * x))).astype(float),
        }
    )

    def hdr(m):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.check(plots=False)
        return buf.getvalue().splitlines()[0]

    cases = [
        (gam("y ~ s(x)", d, method="GCV.Cp"), "Method: GCV   Optimizer: magic"),
        (
            gam("yp ~ s(x)", d, family=Poisson(), method="GCV.Cp"),
            "Method: UBRE   Optimizer: outer newton",
        ),
        (gam("y ~ s(x)", d, method="GACV.Cp"), "Method: GACV   Optimizer: magic"),
        (gam("y ~ s(x)", d, method="REML"), "Method: REML   Optimizer: outer newton"),
        (
            bam("y ~ s(x)", d, method="fREML", discrete=False),
            "Method: fREML   Optimizer: perf newton",
        ),
        (
            bam("y ~ s(x)", d, method="fREML", discrete=True),
            "Method: fREML   Optimizer: perf chol",
        ),
    ]
    for m, expected in cases:
        assert hdr(m) == expected

    # The discrete rail reports bgam.fitd's prop grad/hess (bam.r:884), not a
    # fabricated "fixed by user" line.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bam("y ~ s(x)", d, method="fREML", discrete=True).check(plots=False)
    txt = buf.getvalue()
    assert "$grad" in txt and "$hess" in txt
    assert "fixed by user" not in txt


def test_plot_smooth_xlim_sets_evaluation_range():
    """mgcv's ``xlim`` picks the range the term is EVALUATED over
    (``xx <- seq(xlim[1],xlim[2],length=n)``, plots.r:930-931), so a range
    wider than the data extrapolates rather than cropping the axis. Grid pinned
    against ``plot(b, n=10, xlim=c(-0.5,1.5))``."""
    rng = np.random.default_rng(1)
    n = 300
    x = rng.uniform(0, 1, n)
    d = pl.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)})
    m = gam("y ~ s(x, k=8)", d)

    xs, _ = _plot_curve(m, select=0, rug=False, n_grid_1d=10, xlim=(-0.5, 1.5))
    np.testing.assert_allclose(xs, np.linspace(-0.5, 1.5, 10), rtol=1e-12)
    assert xs[0] < x.min() and xs[-1] > x.max()  # genuinely extrapolated

    xs0, _ = _plot_curve(m, select=0, rug=False, n_grid_1d=10)
    np.testing.assert_allclose(xs0[[0, -1]], [x.min(), x.max()], rtol=1e-12)
