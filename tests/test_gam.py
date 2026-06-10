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

from conftest import load_dataset
from hea.models import gam, glm
from hea.family import Gamma, Tweedie, tw
from hea.models.gam import VisResult

matplotlib.use("Agg")  # headless — must be set before pyplot import below.
import matplotlib.pyplot as plt   # noqa: E402


# =============================================================================
# 1. mgcv-oracle parity
# =============================================================================


def _allclose(actual, expected, *, atol, name=""):
    np.testing.assert_allclose(actual, expected, atol=atol,
                               err_msg=f"{name}: {actual} vs {expected}")


def _assert_param(m, col, est, *, atol=5e-3):
    if col not in m.bhat.columns:
        raise KeyError(f"{col!r} not in {list(m.bhat.columns)!r}")
    np.testing.assert_allclose(m.bhat[col][0], est, atol=atol,
                               err_msg=f"param[{col}]")


# ---------------------------------------------------------------------------
# 1) MASS::mcycle — single tp smooth, REML
# ---------------------------------------------------------------------------


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
    # mgcv's logLik.gam profiles σ² out at the MLE rss/n (not the unbiased
    # rss/(n-edf) reported as $sig2); pin both to lock that convention down.
    _allclose(m.loglike, -597.8345, atol=5e-3, name="loglike")
    _assert_param(m, "(Intercept)", -25.54586, atol=5e-3)
    _allclose(m.edf_by_smooth["s(times)"], 8.624691, atol=5e-4, name="edf[s(times)]")


# ---------------------------------------------------------------------------
# 2) MASS::mcycle — single tp smooth, GCV.Cp
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3) gamSim eg1 — four tp smooths, REML
# ---------------------------------------------------------------------------


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

    # Per-smooth edf — sp[3] is on a flat ridge so the s(x3) edf pins to ~1.0
    # (the linear fallthrough). The other three are well-determined.
    _allclose(m.edf_by_smooth["s(x0)"], 3.020970, atol=5e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.843246, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.019844, atol=5e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 1.001421, atol=5e-2, name="edf[s(x3)]")


# ---------------------------------------------------------------------------
# 4) gamSim eg1 — tensor-product te(x1,x2), REML
# ---------------------------------------------------------------------------


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

    # te has 2 marginal penalties → 2 smoothing parameters
    assert m.sp.shape == (3,)  # 1 for s(x0) + 2 for te
    _allclose(m.sp[0], 1.492971, atol=5e-3, name="sp[s(x0)]")
    _allclose(m.sp[1], 33.05461, atol=1e-1, name="sp[te-1]")
    _allclose(m.sp[2], 0.0882241, atol=5e-3, name="sp[te-2]")


# ---------------------------------------------------------------------------
# 5) by=factor — synthetic data, REML
# ---------------------------------------------------------------------------


def test_byfactor_smooth_REML():
    """gam(y ~ g + s(x, by=g), data=<synth>, method="REML")

    by=factor produces one smooth block per factor level (3 here →
    sp has length 3, edf rolls up per block, identifiability handled
    via mgcv's id="" + parametric main-effect g).
    """
    d = load_dataset("synthetic", "seed_synth_gam_by_factor")
    # Re-cast g as Enum since the schema sidecar may not exist for this synth file.
    if d.schema["g"] != pl.Enum(["A", "B", "C"]):
        d = d.with_columns(pl.col("g").cast(pl.Enum(["A", "B", "C"])))
    m = gam("y ~ g + s(x, by=g)", d, method="REML")

    assert m.n == 300
    assert m.sp.shape == (3,)
    _allclose(m.edf_total, 21.36070, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.04265686, atol=5e-4, name="sigma2")
    _allclose(m.REML_criterion / 2, -9.890208, atol=5e-3, name="REML/2")
    _allclose(m.r_squared_adjusted, 0.9164980, atol=5e-3, name="r2adj")
    _assert_param(m, "(Intercept)",  0.02332958, atol=5e-3)
    _assert_param(m, "gB",          -0.06749164, atol=5e-3)
    _assert_param(m, "gC",           0.63793878, atol=5e-3)

    _allclose(m.edf_by_smooth["s(x):gA"], 6.953522, atol=5e-3, name="edf[s(x):gA]")
    _allclose(m.edf_by_smooth["s(x):gB"], 6.745235, atol=5e-3, name="edf[s(x):gB]")
    _allclose(m.edf_by_smooth["s(x):gC"], 4.661939, atol=5e-3, name="edf[s(x):gC]")


# ---------------------------------------------------------------------------
# 6) MASS::mcycle — P-spline (bs="ps") smooth, REML
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 7) gamSim eg1 — overlap case: s(x1)+s(x2)+te(x1,x2) requires gam.side
# ---------------------------------------------------------------------------


def test_gamSim_eg1_overlap_gamSide_REML():
    """gam(y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2), method='REML')

    The te(x1, x2) marginals overlap the main-effect smooths s(x1) and s(x2).
    Without identifiability constraints the joint design would be rank-deficient
    along the te marginals. mgcv handles this in `gam.side`: it builds X1 from
    the intercept + every strict-subset smooth (here s(x1) and s(x2)), then
    QR-with-pivoting picks the te columns that are linearly dependent on X1
    and deletes them (along with the matching rows/cols of each marginal S).

    For this dataset gam.side drops 2 te columns (24 → 22), so the full design
    has p = 42 columns. Pinning p exercises that path end-to-end.

    The REML surface has multiple near-equivalent optima differing in how
    they distribute penalty between the te marginals and the main-effect
    s(x1)/s(x2). hea and mgcv land at different optima, so the per-marginal
    sp's diverge. Overall fit quantities (σ², REML, r², intercept, x0)
    still agree closely, and edfs land within ~0.34.
    """
    d = load_dataset("mgcv", "gamSim_eg1")
    m = gam("y ~ x0 + s(x1, bs='cr') + s(x2) + te(x1, x2)", d, method="REML")

    assert m.n == 400
    # gam.side must drop 2 te columns: intercept + x0 + s(x1)[9] + s(x2)[9] + te[24-2]
    assert m.bhat.shape[1] == 42, f"gam.side drop failed: p={m.bhat.shape[1]} (expected 42)"

    # 4 sp's: s(x1), s(x2), te-marginal-1, te-marginal-2
    assert m.sp.shape == (4,)
    _allclose(m.sp[1], 7.998938e-03, atol=5e-4, name="sp[s(x2)]")

    # Tight: overall fit
    _allclose(m.sigma_squared, 4.149471, atol=5e-2, name="sigma2")
    _allclose(m.r_squared_adjusted, 0.697276, atol=5e-3, name="r2adj")
    _allclose(m.REML_criterion / 2, 866.7819, atol=5e-1, name="REML/2")
    _assert_param(m, "(Intercept)", 7.642771, atol=5e-3)
    _assert_param(m, "x0", 0.394401, atol=5e-3)

    # Looser: edfs (multi-modal sp surface — mgcv vs hea land at different optima)
    _allclose(m.edf_total, 13.836828, atol=5e-1, name="edf_total")
    _allclose(m.edf_by_smooth["s(x1)"], 2.790683, atol=2e-1, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 8.044964, atol=5e-2, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["te(x1,x2)"], 1.001181, atol=5e-1, name="edf[te]")


# ---------------------------------------------------------------------------
# 8) nlme::Machines — re smooths (Wood 2017 §6.5 example)
# ---------------------------------------------------------------------------


def test_machines_re_smooths_REML():
    """gam(score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re'),
       data=Machines, method='REML') and the by=Machine variant.

    Two random-effect formulations from Wood 2017 §6.5. Exercises:

      - bs='re' on a single factor (one column per Worker level)
      - bs='re' on a Machine:Worker interaction (one column per cell)
      - bs='re' with by=factor (one block per Machine level)

    All three paths require Worker/Machine to be pl.Enum factors. With raw
    CSV dtypes (Int64/Utf8) they silently degrade to single-column random
    *slopes*, blowing edf to ~5 and AIC by ~170.

    Pins target mgcv's published values directly: gam.side is now skipped
    for `bs='re'` smooths (matching mgcv's `side.constrain=FALSE` on re
    smooths), so the design has all 27 cols, the REML optimum lands at
    mgcv's sp's, and edf/loglike/sp/coefficients agree to 4-5 digits.
    AIC uses df = sum(edf2)+1 (Wood 2017 §6.11.3) with edf2 including
    both Vc1 (∂β/∂ρ propagation) and Vc2 (Cholesky-derivative
    correction) — the full mgcv ``gam.fit3.post.proc`` decomposition.
    """
    d = load_dataset("nlme", "Machines")

    b1 = gam("score ~ Machine + s(Worker, bs='re') + s(Machine, Worker, bs='re')",
             data=d, method="REML")
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
    _allclose(b1.edf_total,  17.76461, atol=5e-5, name="b1.edf")
    # edf < edf2 < edf1 — sp uncertainty inflates df, capped by tr(2F-F²).
    _allclose(b1.edf1_total, 17.99523, atol=5e-5, name="b1.edf1")
    _allclose(b1.edf2_total, 17.85995, atol=5e-5, name="b1.edf2")
    assert b1.edf_total < b1.edf2_total <= b1.edf1_total
    _allclose(b1.sigma_squared, 0.92463,  atol=5e-5, name="b1.sigma2")
    _allclose(b1.loglike,    -63.73532,   atol=5e-4, name="b1.loglike")
    _allclose(b1.AIC,        165.19055,   atol=5e-4, name="b1.AIC")
    _assert_param(b1, "(Intercept)", 52.3556, atol=5e-3)
    # both blocks should have meaningful edf — degraded path would give ~1
    assert b1.edf_by_smooth["s(Worker)"] > 3.0
    assert b1.edf_by_smooth["s(Machine,Worker)"] > 8.0

    # vcomp: matches mgcv to 4-5 decimals on points and CIs.
    vc = b1.vcomp
    assert vc.shape == (3, 4)
    assert vc["name"].to_list() == ["s(Worker)", "s(Machine,Worker)", "scale"]
    expected = {
        "s(Worker)":         (4.78106, 2.24987, 10.15997),
        "s(Machine,Worker)": (3.72952, 2.38281,  5.83737),
        "scale":             (0.96158, 0.76325,  1.21143),
    }
    for nm, (sd, lo, hi) in expected.items():
        row = vc.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"vcomp {nm}.std")
        _allclose(row["lower"],   lo, atol=5e-4, name=f"vcomp {nm}.lo")
        _allclose(row["upper"],   hi, atol=5e-4, name=f"vcomp {nm}.hi")

    b2 = gam("score ~ Machine + s(Worker, bs='re') + s(Worker, bs='re', by=Machine)",
             data=d, method="REML")
    assert b2.n == 54
    # by=Machine produces one block per level → 3 extra sp's, total 4
    assert b2.sp.shape == (4,)
    _allclose(b2.edf_total,  17.64453, atol=5e-5, name="b2.edf")
    _allclose(b2.edf2_total, 17.98557, atol=5e-5, name="b2.edf2")
    _allclose(b2.sigma_squared, 0.92463, atol=5e-5, name="b2.sigma2")
    _allclose(b2.loglike,    -63.82464,  atol=5e-4, name="b2.loglike")
    _allclose(b2.AIC,        165.62043,  atol=5e-4, name="b2.AIC")

    vc2 = b2.vcomp
    assert vc2.shape == (5, 4)
    assert vc2["name"].to_list() == [
        "s(Worker)", "s(Worker):MachineA", "s(Worker):MachineB",
        "s(Worker):MachineC", "scale",
    ]
    expected_b2 = {
        "s(Worker)":          (3.78595, 1.79873,  7.96861),
        "s(Worker):MachineA": (1.94032, 0.25319, 14.86973),
        "s(Worker):MachineB": (5.87402, 2.98833, 11.54628),
        "s(Worker):MachineC": (2.84547, 0.82993,  9.75584),
        "scale":              (0.96158, 0.76325,  1.21143),
    }
    for nm, (sd, lo, hi) in expected_b2.items():
        row = vc2.filter(pl.col("name") == nm).row(0, named=True)
        _allclose(row["std_dev"], sd, atol=5e-4, name=f"b2 vcomp {nm}.std")
        _allclose(row["lower"],   lo, atol=5e-4, name=f"b2 vcomp {nm}.lo")
        # The per-level upper CI bounds sit on flat REML directions and
        # are exp-amplified, so they're the band-noise-limited quantities
        # here. Measured: seeding at initial.spg (identical to mgcv's
        # seed to 1e-9), hea and mgcv stop with REML values 2.6e-8 apart
        # — both far inside the |Δscore| ≤ score.scale·conv.tol ≈ 1e-4
        # band — yet the by-level sp's differ up to 4.7e-4 relative along
        # the flat directions, moving MachineA's bound by ~7e-4 relative
        # and MachineC's by ~1e-4. Irreducible without bit-identical
        # arithmetic; the by-level bounds get a band-derived rtol.
        if nm.startswith("s(Worker):Machine"):
            np.testing.assert_allclose(
                row["upper"], hi, rtol=2e-3,
                err_msg=f"b2 vcomp {nm}.hi: {row['upper']} vs {hi}",
            )
        else:
            _allclose(row["upper"], hi, atol=5e-4,
                      name=f"b2 vcomp {nm}.hi")


def test_data_helper_applies_schema_sidecar():
    """`hea.data()` must restore R's factor type via the JSON schema sidecar.

    Without it, factor columns come back from CSV as Int64/Utf8 and bs='re'
    / by=factor / fs / sz smooths silently take the non-factor fallthrough
    path — which is the Machines b1/b2 footgun (AIC ~337 instead of ~165).
    """
    from hea import data
    d = data("Machines", "nlme")
    assert isinstance(d.schema["Worker"], pl.Enum), \
        f"Worker should be pl.Enum, got {d.schema['Worker']}"
    assert isinstance(d.schema["Machine"], pl.Enum), \
        f"Machine should be pl.Enum, got {d.schema['Machine']}"


def test_factor_helper():
    """`hea.R.factor()` is the polars equivalent of R's factor() — the
    user-side fix for wild-data Int64-stored factor columns.
    """
    from hea.R import factor
    from hea.formula import _ORDERED_COLS_CV, set_ordered_cols

    # Bypass `hea.data` (which applies our schema sidecar) to simulate the
    # wild-data scenario where factor info has been stripped — exactly what
    # rdatasets gives us out of the box.
    import rdatasets
    df = pl.from_pandas(rdatasets.data("nlme", "Machines")).drop("rownames")
    assert df.schema["Worker"] == pl.Int64  # the wild-data scenario

    # Auto-detect levels, alphanumeric sort
    out = factor(df["Worker"])
    assert isinstance(out.dtype, pl.Enum)
    assert out.dtype.categories.to_list() == ["1", "2", "3", "4", "5", "6"]
    assert out.name == "Worker"  # preserved → with_columns replaces

    # Explicit levels override sort order
    out2 = factor(df["Worker"], levels=["6", "2", "4", "1", "3", "5"])
    assert out2.dtype.categories.to_list() == ["6", "2", "4", "1", "3", "5"]

    # Casting fixes the s(...,bs='re') breakage end-to-end
    set_ordered_cols(frozenset())  # clean slate
    df_fixed = df.with_columns(factor(df["Worker"]))
    m = gam("score ~ Machine + s(Worker, bs='re')", data=df_fixed, method="REML")
    # degraded path would give edf ~ 2; correct path gives ~5
    assert m.edf_total > 4.0, f"factor() didn't fix the re basis: edf={m.edf_total}"

    # ordered=True adds to contextvar; ordered=False leaves it alone
    set_ordered_cols(frozenset())
    factor(df["Worker"], ordered=True)
    assert "Worker" in _ORDERED_COLS_CV.get()
    factor(df["Worker"], ordered=False)
    assert "Worker" in _ORDERED_COLS_CV.get(), "ordered=False shouldn't unregister"

    # labels= dict: reorder + rename in one pass (R's factor(x, levels=, labels=))
    test = pl.Series("test", [0, 1, 1, 0, 1])
    out_l = factor(test, labels={0: "negative", 1: "positive"})
    assert out_l.dtype.categories.to_list() == ["negative", "positive"]
    assert out_l.to_list() == ["negative", "positive", "positive", "negative", "positive"]
    assert out_l.name == "test"

    # dict insertion order controls level order (= reference level)
    out_rev = factor(test, labels={1: "positive", 0: "negative"})
    assert out_rev.dtype.categories.to_list() == ["positive", "negative"]

    # column value missing from labels keys → replace_strict errors
    with pytest.raises(pl.exceptions.InvalidOperationError):
        factor(pl.Series([0, 1, 2]), labels={0: "a", 1: "b"})

    # labels and levels together is a usage error
    with pytest.raises(ValueError, match="not both"):
        factor(test, levels=[0, 1], labels={0: "a", 1: "b"})

    # passing a dict to levels= is the easy typo — fail loudly, not silently
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

    # Auto-detect levels via mutate(str)
    out = hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor("g"))
    assert isinstance(out.schema["g"], pl.Enum)
    assert out.schema["g"].categories.to_list() == ["a", "b"]

    # Explicit levels via mutate(str, levels=)
    out2 = hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor("g", levels=["b", "a"]))
    assert out2.schema["g"].categories.to_list() == ["b", "a"]

    # labels= rename in one pass
    out3 = hea.tidy.DataFrame._from_pydf(df._df).mutate(
        g=factor("g", labels={"a": "Alpha", "b": "Bravo"})
    )
    assert out3["g"].to_list() == ["Bravo", "Alpha", "Bravo", "Alpha"]

    # pl.Expr form also resolves
    out4 = hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor(pl.col("g")))
    assert isinstance(out4.schema["g"], pl.Enum)

    # select() integration: rename + factor in one verb
    out5 = hea.tidy.DataFrame._from_pydf(df._df).select("x", grp=factor("g"))
    assert out5.columns == ["x", "grp"]
    assert isinstance(out5.schema["grp"], pl.Enum)

    # strict= threads through deferred path
    df_typo = pl.DataFrame({"g": ["a", "b", "x", "a"]})
    out6 = hea.tidy.DataFrame._from_pydf(df_typo._df).mutate(
        g=factor("g", levels=["a", "b"])
    )
    assert out6["g"].to_list() == ["a", "b", None, "a"]
    with pytest.raises(pl.exceptions.InvalidOperationError):
        hea.tidy.DataFrame._from_pydf(df_typo._df).mutate(
            g=factor("g", levels=["a", "b"], strict=True)
        )

    # Auto-detect raises a clear error if the column isn't in the frame
    with pytest.raises(ValueError, match="auto-detect levels"):
        hea.tidy.DataFrame._from_pydf(df._df).mutate(g=factor("missing"))


# Tests for parse_number / if_else / case_when moved to test_dataframe.py
# (they ship from hea.dataframe with the tidyverse port).


# ---------------------------------------------------------------------------
# Cross-cutting: sp passthrough reproduces a fixed-sp fit
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Regression: PIRLS init must produce a valid (β_null, η_null, μ_null) for
# canonical inverse-Gaussian (link = 1/μ²). The previous baseline β=0 ⇒ η=0
# is invalid for this link (valideta requires η>0 finite), and step-halving
# toward η_old=0 cannot escape the invalid region — the fit raised
# `FloatingPointError: PIRLS step halving failed (validity)`. The fix is
# mgcv's null.coef pattern: project a constant valid η onto colspan(X).
# ---------------------------------------------------------------------------


def test_pirls_init_canonical_inverse_gaussian():
    """IG canonical fit on Wald-distributed data must converge."""
    from hea.family import inverse_gaussian
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    mu = 1.5 + 0.5 * np.sin(2 * np.pi * x)               # ∈ [1.0, 2.0], strictly positive
    y = rng.wald(mean=mu, scale=1.0)
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", df, family=inverse_gaussian(), method="REML")
    assert m.n == n
    assert np.all(np.isfinite(m._beta))
    assert np.all(np.isfinite(m.fitted))
    # m.fitted is μ = linkinv(η). For canonical IG link (1/μ²) μ>0 ⇔ η>0,
    # so this also serves as a valideta check on the converged fit.
    assert np.all(m.fitted > 0)
    assert np.all(m.linear_predictors > 0)
    # Phase 2.2 wiring: unknown-scale family ⇒ log φ enters the outer
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
    # mgcv reference on this exact dataset (R, mgcv 1.9-4):
    #   gam(y ~ s(x), family=inverse.gaussian(), method="REML")
    #   m$sig2 = 0.969595916461   (plain Pearson: 0.939454279371)
    np.testing.assert_allclose(m.sigma_squared, 0.969595916461, rtol=1e-6)
    np.testing.assert_allclose(m._pearson_scale, 0.939454279371, rtol=1e-6)
    # Intercept ≈ link(mean(y)) = 1/mean(y)² for an intercept-only fit;
    # with a smooth that captures most of the signal it lands near
    # link(mean(mu_true)) = 1/1.5² ≈ 0.444.
    intercept = m.bhat["(Intercept)"][0]
    assert 0.30 < intercept < 0.60


# ---------------------------------------------------------------------------
# Phase 1.9 — non-Gaussian post-fit smoke. trees + Gamma(log) is the canonical
# small-n GLM example; mgcv's r.sq, deviance_explained, and null_deviance only
# depend on (y, μ, wt, family) and family.dev_resids, so those land at mgcv's
# values even before the REML score is family-aware (Phase 2). sp/edf/AIC
# still depend on the (Gaussian-only) REML score, so they're pinned at hea's
# current values with a TODO — Phase 4's mgcv-oracle battery tightens them.
# ---------------------------------------------------------------------------


def test_trees_gamma_log_smoke():
    """trees + Gamma(log), method='REML': pin family-agnostic post-fit values
    against mgcv (those that don't depend on sp), and hea's current
    sp-dependent values as a regression guard until Phase 2 lands."""
    from hea.family import Gamma
    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d, family=Gamma(link="log"),
            method="REML")

    # Family / link plumbing.
    assert m.family.name == "Gamma"
    assert m.family.link.name == "log"
    assert m.family.scale_known is False

    # μ vs η: log-link ⇒ μ = exp(η), strictly positive.
    assert np.all(m.fitted_values > 0)
    np.testing.assert_allclose(
        m.fitted_values, np.exp(m.linear_predictors), atol=1e-12,
    )
    assert m.fitted is m.fitted_values or np.array_equal(m.fitted, m.fitted_values)

    # df: n=31, intercept-only null ⇒ df_null = n-1 = 30.
    assert m.n == 31
    np.testing.assert_allclose(m.df_null, 30.0, atol=0.0)

    # mgcv reference values (R 4.5.3, mgcv 1.9-3) at the converged fit:
    #   sp        = (15742.67387, 0.2112713142)
    #   edf_total = 4.738161, edf2_total = 5.270166
    #   scale     = m$reml.scale = 0.0068696749 (m$scale = 0.0068300304)
    #   deviance  = 0.1805645860, null_deviance = 8.3172012147
    #   r2_adj    = 0.9744391060, dev_expl = 0.9782902227
    #   AIC       = 144.3438870069, logLik = -65.9017771491
    #   intercept = 3.2756440543

    # Tight pins (independent of optimizer convergence trajectory).
    np.testing.assert_allclose(m.r_squared_adjusted, 0.9744391060, atol=5e-5)
    np.testing.assert_allclose(m.deviance_explained, 0.9782902227, atol=5e-5)
    np.testing.assert_allclose(m.null_deviance, 8.3172012147, atol=5e-7)
    np.testing.assert_allclose(m.deviance, 0.1805645860, atol=5e-4)
    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756440543, atol=5e-3)

    # Looser pins on optimizer-dependent quantities. Phase 2.2 is using
    # L-BFGS-B with finite-difference gradients on the (ρ, log φ) outer
    # vector; the score has a long flat plateau in the Height-smooth
    # direction (its smooth saturates well before sp[0] hits the upper
    # rho bound), so sp[0] reproducibly lands ~50× larger than mgcv's
    # analytical-Newton answer while edf/scale/deviance agree to ~5e-3.
    # Phase 3 (analytical (ρ, log φ) gradients/Hessian) will tighten this.
    np.testing.assert_allclose(m.sp[1], 0.2112713142, rtol=2e-3)
    np.testing.assert_allclose(m.edf_total,  4.738161, atol=5e-2)
    np.testing.assert_allclose(m.edf2_total, 5.270166, atol=5e-2)
    np.testing.assert_allclose(m.scale,      0.0068696749, atol=5e-5)
    np.testing.assert_allclose(m.sigma_squared, m.scale, atol=0.0)
    np.testing.assert_allclose(m.logLik, -65.9017771491, atol=2e-2)
    np.testing.assert_allclose(m.AIC,    144.3438870069, atol=1e-1)

    # AIC.default identity: AIC = -2·logLik + 2·npar (by construction).
    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    np.testing.assert_allclose(m.BIC, -2.0 * m.logLik + np.log(m.n) * m.npar,
                               atol=1e-10)

    # Intercept: log(weighted_mean(Volume)) ≈ log(30.17) ≈ 3.408 for an
    # intercept-only fit; with two smooths absorbing most of the signal
    # the fitted intercept lands near 3.276.
    np.testing.assert_allclose(m.bhat["(Intercept)"][0], 3.2756425861, atol=5e-5)

    # First-five fitted μ vs mgcv reference — Phase 2.2 lands within ~5e-4
    # of mgcv even with the FD optimizer plateau (the smooths matter for μ,
    # not the saturated Height direction).
    np.testing.assert_allclose(
        m.fitted_values[:5],
        [10.62414379, 10.36186212, 10.41212209, 16.42891707, 19.68356227],
        atol=5e-3,
    )

    # Gamma(log) residual identities:
    #   working = (y-μ)/(dμ/dη) = (y-μ)/μ
    #   pearson = (y-μ)·√(wt/V) = (y-μ)/μ        (V=μ², wt=1)
    # ⇒ pearson == working for log Gamma.
    pearson = m.residuals_of("pearson")
    working = m.residuals_of("working")
    np.testing.assert_allclose(pearson, working, atol=1e-12)
    response = m.residuals_of("response")
    np.testing.assert_allclose(response, m._y_arr - m.fitted_values, atol=0.0)
    # Default residuals = deviance residuals.
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
    # η == μ for identity link.
    np.testing.assert_allclose(m.linear_predictors, mu, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("response"), target, atol=0.0)
    np.testing.assert_allclose(m.residuals_of("deviance"), target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("pearson"),  target, atol=1e-12)
    np.testing.assert_allclose(m.residuals_of("working"),  target, atol=1e-12)
    # m.residuals defaults to deviance residuals.
    np.testing.assert_array_equal(m.residuals, m.residuals_of("deviance"))
    # Deviance residual identity: Σ d_i² = m.deviance for Gaussian (V=1).
    np.testing.assert_allclose(np.sum(m.residuals_of("deviance") ** 2),
                               m.deviance, atol=1e-9)
    # AIC.default self-consistency.
    np.testing.assert_allclose(m.AIC, -2.0 * m.logLik + 2.0 * m.npar, atol=1e-10)
    # Bad type raises.
    with pytest.raises(ValueError):
        m.residuals_of("partial")


def test_reml_finite_for_trees_gamma_log():
    """Sanity: for the converged Gamma(log) fit, `_reml` returns a
    finite value at the hea-current sp. (Phase 2.2 makes φ̂ a joint outer
    variable; this just ensures the formula is wired up correctly.)"""
    from hea.models import gam
    from hea.family import Gamma
    d = load_dataset("R", "trees")
    m = gam("Volume ~ s(Height) + s(Girth)", d,
            family=Gamma(link="log"), method="REML")
    log_phi = float(np.log(m.scale))
    v = m._reml(m._rho_hat, log_phi)
    assert np.isfinite(v)


# ---------------------------------------------------------------------------
# gam.check() — port of mgcv::gam.check / mgcv::k.check
# ---------------------------------------------------------------------------


def test_kcheck_mcycle_matches_mgcv():
    """k.check on `accel ~ s(times)` (1D smooth, REML).

    mgcv pin (n.rep=10000, see development log):
        s(times)  k'=9   edf=8.62469100  k-index=1.14736165
    edf and k-index are deterministic in the residuals + covariate; we
    pin them tightly. The p-value is a permutation tail and depends on
    the RNG draw — pin it to a wide-enough band that the test stays
    robust across RNG seeds.
    """
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    ktab = m._k_check(seed=0, n_rep=2000)
    assert ktab[""].to_list() == ["s(times)"]
    np.testing.assert_allclose(ktab["k'"].to_list(),     [9.0],          atol=0)
    np.testing.assert_allclose(ktab["edf"].to_list(),    [8.62469100],   atol=5e-5)
    np.testing.assert_allclose(ktab["k-index"].to_list(),[1.14736165],   atol=5e-5)
    # mgcv reports ~0.95 with 10k reps; permutation noise widens the band.
    assert 0.85 < ktab["p-value"][0] <= 1.0


def test_kcheck_handles_no_smooths_returns_none():
    """k.check is undefined when there are no smooth blocks. Mirrors
    mgcv: `k.check` returns NULL → `gam.check` skips the table."""
    d = load_dataset("R", "trees")
    m = gam("Volume ~ Height + Girth", d, method="REML")
    assert m._k_check() is None


def test_check_prints_convergence_block(capsys):
    """`gam.check()` runs end-to-end and emits the mgcv-style header.

    The exact gradient/eigenvalue numbers are not pinned (those are
    determined by the converged ρ̂ and would shift if the optimizer is
    re-tuned later); we only verify the structural lines are there.
    """
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    m.check(seed=0, k_rep=200)
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
    m.check()
    out = capsys.readouterr().out
    assert "Model required no smoothing parameter selection" in out
    assert "Basis dimension" not in out


# ---------------------------------------------------------------------------
# LHS expressions — `y^0.25 ~ ...`, `log(y) ~ ...`, `I(y/100) ~ ...`
# ---------------------------------------------------------------------------


def test_lhs_power_brain_matches_mgcv():
    """Wood §7.2: `gam(medFPQ^.25 ~ s(Y, X, k=100), data=brain)`.

    mgcv pins on the trimmed dataset (medFPQ > 5e-5, n=1565):
        edf_total ≈ 65.176, sigma2 ≈ 0.039541, GCV ≈ 0.041259
    """
    d = load_dataset("gamair", "brain").filter(pl.col("medFPQ") > 5e-5)
    m = gam("medFPQ^.25 ~ s(Y, X, k=100)", d)
    assert m.n == 1565
    assert m.y.name == "medFPQ^0.25"
    _allclose(m.edf_total,     65.1763,  atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.039541, atol=5e-6, name="sigma2")
    _allclose(m.GCV_score,     0.041259, atol=5e-6, name="GCV")


def test_lhs_log_matches_manual_transform():
    """`log(y) ~ ...` should be identical to pre-computing log(y) in
    polars and fitting `log_y ~ ...` on the same RHS."""
    d = load_dataset("R", "trees")
    m_lhs = gam("log(Volume) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns(pl.col("Volume").log().alias("log_v"))
    m_pre = gam("log_v ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    np.testing.assert_allclose(m_lhs.sp,     m_pre.sp,     atol=0)
    np.testing.assert_allclose(m_lhs._beta,  m_pre._beta,  atol=1e-12)
    assert m_lhs.y.name == "log(Volume)"


def test_lhs_I_div_matches_manual_transform():
    """`I(y/100) ~ ...` is just an unwrap; should equal pre-computing
    y/100. Also verifies the deparsed label survives I()."""
    d = load_dataset("R", "trees")
    m_lhs = gam("I(Volume / 100) ~ s(Height) + s(Girth)", d, method="REML")
    d2 = d.with_columns((pl.col("Volume") / 100.0).alias("v100"))
    m_pre = gam("v100 ~ s(Height) + s(Girth)", d2, method="REML")
    np.testing.assert_allclose(m_lhs.fitted, m_pre.fitted, atol=1e-12)
    # Deparser inserts spaces around `/`; mgcv shows `I(Volume/100)` instead,
    # but both reduce to the same column transform — the visible label is
    # the deparser's choice, which is acceptable.
    assert "Volume" in m_lhs.y.name and "100" in m_lhs.y.name


def test_lhs_unsupported_function_raises():
    """An unsupported function on the LHS should error with a helpful
    message naming the allowed transforms."""
    d = load_dataset("R", "trees")
    with pytest.raises(NotImplementedError, match="not supported"):
        gam("foo(Volume) ~ s(Height)", d, method="REML")


def test_lhs_cbind_raises():
    """cbind() multi-column response is not implemented yet — error clearly."""
    d = load_dataset("R", "trees")
    with pytest.raises(NotImplementedError, match="cbind"):
        gam("cbind(Volume, Height) ~ s(Girth)", d, method="REML")


def test_lhs_unknown_column_raises():
    """Reference to a non-existent column inside an LHS expression."""
    d = load_dataset("R", "trees")
    with pytest.raises(KeyError, match="nope"):
        gam("log(nope) ~ s(Height)", d, method="REML")


def test_lhs_na_omit_drops_lhs_referenced_columns():
    """If the LHS expression touches a column that has NAs, those rows
    must be dropped before evaluating the response — otherwise polars
    would surface NaN through the transform."""
    d = pl.DataFrame({
        "a":  [1.0, 4.0, None, 16.0, 25.0, 36.0,  49.0, 64.0,  81.0, 100.0],
        "x":  [1.0, 2.0, 3.0,   4.0,  5.0,  6.0,   7.0,  8.0,   9.0,  10.0],
    })
    m = gam("sqrt(a) ~ s(x, k=4)", d, method="REML")
    # The NA row in `a` was dropped; n is 9, not 10.
    assert m.n == 9
    np.testing.assert_allclose(np.asarray(m.y.to_list()),
                               np.sqrt([1, 4, 16, 25, 36, 49, 64, 81, 100]),
                               atol=1e-12)


def test_check_outer_info_is_populated_after_fit():
    """`_outer_info` should be filled with grad/hess/score/iter after
    a smooth fit, and remain None for the no-smooth path."""
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    info = m._outer_info
    assert info is not None
    assert info["iter"] >= 1
    # Gaussian REML puts (ρ, log φ) on the outer vector — for one smooth
    # that's length 2; known-scale families would be length 1.
    g = info["grad"]
    H = info["hess"]
    assert g.size >= len(m.sp)
    assert H.shape == (g.size, g.size)
    assert np.isfinite(info["score"])
    # mcycle's REML surface is well-behaved → Hessian PD at optimum.
    ev = np.linalg.eigvalsh(0.5 * (info["hess"] + info["hess"].T))
    assert ev.min() > 0

    d2 = load_dataset("R", "trees")
    m2 = gam("Volume ~ Height + Girth", d2, method="REML")
    assert m2._outer_info is None


# ---------------------------------------------------------------------------
# offset(...) — both via formula and via offset= kwarg.
#
# Identity check: a parametric-only formula's gam fit must exactly match
# the equivalent glm fit (gam reduces to glm when there are no smooths).
# Offset-shift check: predict with newdata re-evaluates formula offsets.
# Parity check: mgcv pinned values for a small Poisson+offset GAM.
# ---------------------------------------------------------------------------


def test_gam_offset_in_formula_matches_glm():
    """No smooths → gam == glm. Offset(...) inside the formula must
    propagate identically through both."""
    from hea.models import glm
    from hea.family import Quasi
    d = load_dataset("MASS", "quine")  # count data
    # Synthetic offset column to exercise the path.
    d = d.with_columns(off=pl.lit(0.3) * pl.col("Days").cast(pl.Float64).clip(lower_bound=1).log())
    formula = "Days ~ offset(off) + Sex + Age"
    fam = Quasi(link="log", variance="mu")
    b_glm = glm(formula, family=fam, data=d)
    b_gam = gam(formula, family=fam, data=d, method="REML")
    np.testing.assert_allclose(
        b_gam._beta, b_glm._bhat_arr, atol=1e-10,
    )
    np.testing.assert_allclose(b_gam.deviance, b_glm.deviance, atol=1e-10)
    np.testing.assert_allclose(
        b_gam.fitted_values, b_glm.fitted_values, atol=1e-10,
    )


def test_gam_offset_kwarg_equivalent_to_formula_offset():
    """offset(off) in formula should give the same fit as offset=off kwarg."""
    rng = np.random.default_rng(0)
    n = 100
    d = pl.DataFrame({
        "y": rng.poisson(3.0, n).astype(float),
        "x": rng.standard_normal(n),
        "off_col": rng.uniform(0.0, 1.0, n),
    })
    from hea.family import Poisson
    a = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    b = gam("y ~ x", family=Poisson(), data=d, method="REML",
            offset=d["off_col"].to_numpy())
    np.testing.assert_allclose(a._beta, b._beta, atol=1e-10)
    np.testing.assert_allclose(a.deviance, b.deviance, atol=1e-10)


def test_gam_gamma_kwarg_matches_mgcv_on_trees():
    """``gamma=`` (mgcv's smoothing-strength multiplier) — Wood §4.6 cites
    ``gamma=1.4`` as a reasonable default for over-fit protection.

    Pinned: trees + Gamma(log), GCV.Cp and REML, both γ=1 and γ=1.4.
    Criterion values come from mgcv 1.9.4 directly.
    """
    from hea.family import Gamma
    trees = load_dataset("mgcv", "trees")

    # GCV.Cp path
    m_gcv_1 = gam("Volume ~ s(Height) + s(Girth)",
                  family=Gamma(link="log"), data=trees,
                  method="GCV.Cp", gamma=1.0)
    np.testing.assert_allclose(m_gcv_1.GCV_score, 0.008082356, atol=1e-6)
    np.testing.assert_allclose(m_gcv_1.sp[1], 0.342711, atol=1e-4)

    m_gcv_14 = gam("Volume ~ s(Height) + s(Girth)",
                   family=Gamma(link="log"), data=trees,
                   method="GCV.Cp", gamma=1.4)
    np.testing.assert_allclose(m_gcv_14.GCV_score, 0.009228008, atol=1e-6)
    np.testing.assert_allclose(m_gcv_14.sp[1], 0.524542, atol=1e-4)
    # γ>1 produces smoother fits — sp[1] (Girth) increases.
    assert m_gcv_14.sp[1] > m_gcv_1.sp[1]

    # REML path — hea's REML_criterion is -2·V_R; mgcv's b$gcv.ubre is V_R.
    m_reml_1 = gam("Volume ~ s(Height) + s(Girth)",
                   family=Gamma(link="log"), data=trees,
                   method="REML", gamma=1.0)
    np.testing.assert_allclose(m_reml_1.REML_criterion / 2, 78.00469, atol=1e-3)

    m_reml_14 = gam("Volume ~ s(Height) + s(Girth)",
                    family=Gamma(link="log"), data=trees,
                    method="REML", gamma=1.4)
    np.testing.assert_allclose(m_reml_14.REML_criterion / 2, 59.35457, atol=1e-3)


def test_plot_smooth_dispatches_2d_to_contour():
    """``plot_smooth`` should auto-render contour for 2D smooths
    (Wood 2017 Fig. 4.14 — bold/dashed/dotted contours + data scatter)."""
    import matplotlib
    matplotlib.use("Agg")
    from hea.family import Gamma
    trees = load_dataset("mgcv", "trees")
    ct5 = gam("Volume ~ s(Height, Girth, k=25)",
              family=Gamma(link="log"), data=trees)
    fig = ct5.plot_smooth(too_far=0.1)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # Title should carry the smooth label + edf, mgcv-style.
    assert "s(Height,Girth," in ax.get_title()
    assert ax.get_xlabel() == "Height"
    assert ax.get_ylabel() == "Girth"

    # Mixed 1D + 2D: panel 0 is 1D (no title, ylabel carries the label),
    # panel 1 is 2D (title carries the label).
    m = gam("Volume ~ s(Height) + s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)
    fig2 = m.plot_smooth(too_far=0.1)
    assert len(fig2.axes) == 2
    assert fig2.axes[0].get_title() == ""        # 1D panel
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
            .replace_strict([1, 2, 3], ["small", "medium", "large"],
                            return_dtype=pl.Enum(["small", "medium", "large"])),
    )
    ct7 = gam("Volume ~ Hclass + s(Girth)",
              family=Gamma(link="log"), data=trees)
    fig = ct7.plot_smooth(all_terms=True)
    assert len(fig.axes) == 2

    # Panel 0: smooth s(Girth)
    assert fig.axes[0].get_xlabel() == "Girth"
    assert "s(Girth," in fig.axes[0].get_ylabel()

    # Panel 1: parametric Hclass (factor termplot)
    assert fig.axes[1].get_xlabel() == "Hclass"
    assert fig.axes[1].get_ylabel() == "Partial for Hclass"
    # x-tick labels are the level names in factor order.
    xticks = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert xticks == ["small", "medium", "large"]

    # all_terms=False (default) → only the smooth panel.
    fig2 = ct7.plot_smooth()
    assert len(fig2.axes) == 1


def test_plot_smooth_select_by_name_and_list():
    """``select=`` accepts a smooth label, a list of labels, or a list of
    ints; ordering follows the list."""
    import matplotlib
    matplotlib.use("Agg")
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")

    # Single string → one panel.
    fig = m.plot_smooth(select="s(x2)")
    assert len(fig.axes) == 1
    assert "s(x2," in fig.axes[0].get_ylabel()

    # List of strings → panels in given order. Reverse formula order to
    # verify ordering is honored.
    fig = m.plot_smooth(select=["s(x3)", "s(x1)"])
    assert len(fig.axes) == 2
    assert "s(x3," in fig.axes[0].get_ylabel()
    assert "s(x1," in fig.axes[1].get_ylabel()

    # Mixed int + str works.
    fig = m.plot_smooth(select=[0, "s(x3)"])
    assert len(fig.axes) == 2
    assert "s(x1," in fig.axes[0].get_ylabel()
    assert "s(x3," in fig.axes[1].get_ylabel()

    # Unknown name lists the available labels.
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
    m = gam("Volume ~ s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    # scheme=1 → persp axes; zlabel carries the smooth label.
    fig = m.plot_smooth(scheme=1)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], Axes3D)
    assert "s(Height,Girth," in fig.axes[0].get_zlabel()

    # scheme=0 (default) keeps the contour rendering.
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
    m = gam("Volume ~ s(Height) + s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    # 1D, 2D-persp — last panel must be 3D, first 2D.
    fig = m.plot_smooth(scheme=[0, 1])
    assert len(fig.axes) == 2
    assert not isinstance(fig.axes[0], Axes3D)
    assert isinstance(fig.axes[1], Axes3D)

    # Length mismatch raises.
    with pytest.raises(ValueError, match="scheme list must have length 2"):
        m.plot_smooth(scheme=[0, 1, 0])


def test_plot_smooth_ax_3d_required_for_persp():
    """Passing ``ax=`` for a 2D scheme=1 panel demands a 3D Axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.family import Gamma
    trees = load_dataset("mgcv", "trees")
    m = gam("Volume ~ s(Height, Girth, k=20)",
            family=Gamma(link="log"), data=trees)

    fig, ax2d = plt.subplots()
    with pytest.raises(TypeError, match="3D Axes"):
        m.plot_smooth(scheme=1, ax=ax2d)

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    out = m.plot_smooth(scheme=1, ax=ax3d)
    assert out is ax3d


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
    d = pl.DataFrame({
        "y": rng.poisson(4.0, n).astype(float),
        "x": rng.standard_normal(n),
        "off_col": rng.uniform(0.5, 1.5, n),
    })
    from hea.family import Poisson
    m = gam("y ~ offset(off_col) + x", family=Poisson(), data=d, method="REML")
    # Same X but a different offset column → η̂ should shift by exactly Δoffset.
    new = d.with_columns((pl.col("off_col") + 2.0).alias("off_col"))
    eta_orig = m.predict(type="link")["fit"].to_numpy()
    eta_new = m.predict(new, type="link")["fit"].to_numpy()
    np.testing.assert_allclose(eta_new - eta_orig, 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# select=TRUE — mgcv's null-space penalty for term selection
# ---------------------------------------------------------------------------


def test_select_true_doubles_n_sp():
    """select=TRUE adds one null-space penalty per smooth → n_sp doubles."""
    d = load_dataset("synthetic", "seed_synth_basic")
    m_off = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML")
    m_on  = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)
    assert len(m_off.sp) == 3
    assert len(m_on.sp) == 6


def test_select_true_three_smooth_REML():
    """gam(y ~ s(x1)+s(x2)+s(x3), data=seed_synth_basic, method="REML", select=TRUE)

    Pinned to mgcv's converged values. s(x3) is signal-free in this data, so
    select=TRUE shrinks its edf to ~0 — the whole point of the null-space
    penalty.
    """
    d = load_dataset("synthetic", "seed_synth_basic")
    m = gam("y ~ s(x1) + s(x2) + s(x3)", d, method="REML", select=True)

    # mgcv-converged scalars (from gam(..., select=TRUE) on seed_synth_basic):
    _allclose(m.edf_total, 2.912088577, atol=5e-3, name="edf_total")
    _allclose(m.sigma_squared, 0.8940008109, atol=5e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 277.0814067, atol=5e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 1.091137918, atol=5e-3)

    # Per-smooth edf — both implementations land in the flat plateau where
    # the heavily-shrunk null-space sps drift; pin only the well-determined
    # active edf and assert s(x3) is essentially shrunk out.
    _allclose(m.edf_by_smooth["s(x1)"], 0.9739738079, atol=5e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 0.9379440321, atol=5e-3, name="edf[s(x2)]")
    assert m.edf_by_smooth["s(x3)"] < 1e-2, \
        f"s(x3) should be selected out, got edf={m.edf_by_smooth['s(x3)']}"


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
    m_fix  = gam("accel ~ s(times)", d, method="REML", select=True, sp=m_free.sp)
    # β at the same rho is independent of φ — fitted values match exactly.
    np.testing.assert_allclose(m_fix.fitted, m_free.fitted, atol=1e-10)
    np.testing.assert_allclose(m_fix.edf_total, m_free.edf_total, atol=1e-10)
    # σ² and REML are profile-out-identical only at the *exact* gradient zero;
    # the optimizer stops at gradient ≈ 0 → expect ~1e-6 relative agreement.
    np.testing.assert_allclose(m_fix.sigma_squared, m_free.sigma_squared, rtol=1e-5)
    np.testing.assert_allclose(m_fix.REML_criterion, m_free.REML_criterion, rtol=1e-7)


def test_select_true_at_mgcv_sp_matches_mgcv():
    """At a fixed sp vector, hea's select=TRUE fit must reproduce mgcv's
    post-fit numbers — checks the null-space penalty math directly,
    bypassing optimizer convergence differences.

    mgcv-converged sp on `gamSim_eg1` for `y ~ s(x0)+s(x1)+s(x2)+s(x3)` with
    `select=TRUE, method="REML"`:
    """
    d = load_dataset("mgcv", "gamSim_eg1")
    sp_mgcv = np.array([
        2.521010255, 423334.7801,    # s(x0): wig, null
        1.843214985, 1.820731653,    # s(x1): wig, null
        0.00569866453, 47639.04804,  # s(x2): wig, null
        84968.55542, 131.2834178,    # s(x3): wig, null (essentially zeroed)
    ])
    m = gam("y ~ s(x0) + s(x1) + s(x2) + s(x3)",
            d, method="REML", select=True, sp=sp_mgcv)

    # mgcv post-fit at this sp — bit-perfect targets:
    _allclose(m.edf_total, 14.45446565, atol=1e-3, name="edf_total")
    _allclose(m.sigma_squared, 3.933035582, atol=1e-3, name="sigma2")
    _allclose(m.REML_criterion / 2, 868.3979813, atol=1e-3, name="REML/2")
    _assert_param(m, "(Intercept)", 7.833279497, atol=1e-3)
    _allclose(m.edf_by_smooth["s(x0)"], 2.418051213, atol=1e-3, name="edf[s(x0)]")
    _allclose(m.edf_by_smooth["s(x1)"], 2.839713272, atol=1e-3, name="edf[s(x1)]")
    _allclose(m.edf_by_smooth["s(x2)"], 7.448219388, atol=1e-3, name="edf[s(x2)]")
    _allclose(m.edf_by_smooth["s(x3)"], 0.7484817774, atol=1e-3, name="edf[s(x3)]")


# ---------------------------------------------------------------------------
# Summary p-value dispatch — known-scale family (binomial), select=TRUE
# ---------------------------------------------------------------------------


def test_select_true_binomial_summary_matches_mgcv():
    """hea's summary() must dispatch on ``family.scale_known``: known-scale
    families use the Wald z-test for parametric coefficients and the Wood
    (2013) reTest with Davies' weighted-χ² CDF for smooth significance,
    not t/F. Pinned to mgcv on wesdr at mgcv's converged sp.
    """
    from scipy.stats import norm
    from hea.family import Binomial
    d = load_dataset("gamair", "wesdr")
    sp_mgcv = np.array([
        0.0164113465035,  4.59199813892,  # s(dur): wig, null
        1793.09515417,    0.953183305109, # s(gly): wig, null
        0.0458306723482,  5.7780644155,   # s(bmi): wig, null
    ])
    m = gam("ret ~ s(dur,k=5) + s(gly,k=5) + s(bmi,k=5)",
            d, family=Binomial(), method="REML", select=True, sp=sp_mgcv)

    # Family / scale-known dispatch flag.
    assert m.family.scale_known is True

    # mgcv post-fit at the same sp:
    _allclose(m.edf_total, 7.430392736, atol=1e-3, name="edf_total")
    # mgcv's `b$gcv.ubre` (printed in summary as `-REML`) — for binomial it's
    # the REML/2 we report, since hea's `REML_criterion` doubles mgcv's value.
    _allclose(m.REML_criterion / 2, 389.4888704, atol=1e-3, name="REML/2")

    # Parametric: z-test, not t-test (binomial → φ ≡ 1).
    j = list(m.bhat.columns).index("(Intercept)")
    est = float(m._beta[j])
    se = float(m._se[j])
    z = est / se
    p_z = 2.0 * norm.sf(abs(z))
    _allclose(est, -0.4150103, atol=1e-3, name="intercept")
    _allclose(se, 0.0887844, atol=1e-3, name="intercept SE")
    _allclose(z, -4.674361, atol=5e-3, name="z")
    _allclose(p_z, 2.948704e-06, atol=5e-7, name="Pr(>|z|)")

    # Smooth significance via reTest (mgcv summary.gam reTest path):
    # mgcv pins (edf, Ref.df, Chi.sq, p-value):
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
        _allclose(float(m.edf[a:bcol].sum()), edf_t,
                  atol=1e-3, name=f"edf[{label}]")
        # Ref.df = effective rank from Davies' eigenvalue truncation.
        # Under select=TRUE this is the basis dimension (= ncol of the
        # smooth's design block).
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

    Scale-known family. Even with φ ≡ 1, REML and ML pick *different* sp
    because the Hessian log-det differs (REML uses log|H+S|; ML uses
    log|H_pp+S_pp|, the range-only block — see mgcv ``MLpenalty1`` in
    gdi.c:1532-1680). mgcv's pins:
        REML sp ≈ (0.0565, 4205, 0.1277), edf 9.117, score 386.350
        ML   sp ≈ (0.0787, 34055, 0.2153), edf 8.417, score 384.004
    """
    from hea.family import Binomial
    d = load_dataset("gamair", "wesdr")
    m_ml = gam("ret ~ s(dur) + s(gly) + s(bmi)",
               d, family=Binomial(), method="ML")

    _allclose(m_ml.ML_criterion / 2, 384.0036, atol=5e-3, name="ML/2")
    _allclose(m_ml.edf_total, 8.416686, atol=5e-3, name="edf_total")
    _assert_param(m_ml, "(Intercept)", -0.4176841, atol=5e-3)
    _allclose(m_ml.sp[0], 0.07866319, atol=5e-3, name="sp[s(dur)]")
    _allclose(m_ml.sp[2], 0.2152721,  atol=5e-3, name="sp[s(bmi)]")
    # sp[1] for s(gly) is on a ~flat ridge (mgcv 34055, hea > 1e7) — both
    # are effectively "fully smoothed", so don't pin its absolute value;
    # the resulting fit (edf, score) is what matches.


def test_method_validation():
    """gam() rejects bogus method strings before doing any work."""
    d = load_dataset("MASS", "mcycle")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="UBRE")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="GACV.Cp")
    with pytest.raises(ValueError, match="REML.*ML.*GCV"):
        gam("accel ~ s(times)", d, method="P-REML")


# ---------------------------------------------------------------------------
# Tweedie / tw — end-to-end fits on a synthetic compound Poisson-Gamma
# response. Mirrors the mgcv mack/egg-count workflow at small n; checks
# that p̂ recovers the truth and tw() never scores worse than Tweedie(p̂_init).
# ---------------------------------------------------------------------------


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
    """User's literal R formula: ``s(I(b.depth^.5))`` and ``offset(...)``.

    Pins the joint-Newton tw() fit on the gamair::mack complete-cases
    subset to mgcv 1.9-4. Generated with:
        m <- gam(egg.count ~ s(lon, lat, k=100) + s(I(b.depth^.5)) + s(c.dist)
                              + s(salinity) + s(temp.surf) + s(temp.20m)
                              + offset(log.net.area),
                 data=mack, family=tw(), method="REML", select=TRUE)

    The expression ``I(b.depth^.5)`` is materialised into a synthesised
    column ``"I(b.depth^0.5)"`` by ``materialize_smooths`` /
    ``_smooth_arg_expr_map``; predict-time replay re-evaluates the AST
    against new data via the same machinery.
    """
    mack = load_dataset("gamair", "mack")
    keep_cols = ["egg.count", "lon", "lat", "b.depth", "c.dist",
                 "salinity", "temp.surf", "temp.20m", "net.area"]
    mack = mack.drop_nulls(subset=keep_cols)
    mack = mack.with_columns(log_net_area=pl.col("net.area").log())

    m = gam(
        "egg.count ~ s(lon, lat, k=100) + s(I(b.depth^0.5)) + s(c.dist) "
        "+ s(salinity) + s(temp.surf) + s(temp.20m) + offset(log_net_area)",
        mack, family=tw(), method="REML", select=True,
    )
    info = m._tw_info
    assert info is not None
    np.testing.assert_allclose(info["p_hat"], 1.33307185396394, atol=1e-3)
    np.testing.assert_allclose(m.REML_criterion / 2,
                               927.776776447335, atol=5e-3)
    # mgcv reports edf for the I(...) smooth under its deparsed label.
    assert "s(I(b.depth^0.5))" in m.edf_by_smooth
    np.testing.assert_allclose(
        m.edf_by_smooth["s(I(b.depth^0.5))"], 2.37609109, atol=5e-2,
    )
    np.testing.assert_allclose(m.edf_total, 47.4833915, atol=5e-2)


def test_gam_tw_mack_mgcv_oracle():
    """Pin tw() joint outer-Newton output against mgcv 1.9-4 on gamair::mack.

    Generated with:
        library(gamair); data(mack)
        mack$log.net.area <- log(mack$net.area)
        keep <- complete.cases(mack[, c("egg.count", "lon", "lat",
                                        "b.depth", "c.dist", "salinity",
                                        "temp.surf", "temp.20m",
                                        "log.net.area")])
        m <- gam(egg.count ~ s(lon, lat, k=20) + s(temp.surf),
                 data=mack[keep,], family=tw(), method="REML",
                 offset=log.net.area)

    p̂ matches to ~6 digits, REML/2 to 7 digits, scale to ~5 digits.
    sp[1] (temp.surf) sits on the flat-ridge tail where mgcv and hea both
    effectively fully smooth; only the resulting REML/edf are pinned there,
    not the absolute sp value.
    """
    mack = load_dataset("gamair", "mack")
    keep_cols = ["egg.count", "lon", "lat", "b.depth", "c.dist",
                 "salinity", "temp.surf", "temp.20m", "net.area"]
    mack = mack.drop_nulls(subset=keep_cols)
    mack = mack.with_columns(log_net_area=pl.col("net.area").log())

    m = gam(
        "egg.count ~ s(lon, lat, k=20) + s(temp.surf)",
        mack, family=tw(), method="REML",
        offset=mack["log_net_area"].to_numpy().tolist(),
    )
    info = m._tw_info
    assert info is not None
    np.testing.assert_allclose(info["p_hat"], 1.39920632555438, atol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2,
                               945.744274311548, atol=1e-4)
    np.testing.assert_allclose(np.exp(info["log_phi_hat"]),
                               4.00764107362287, rtol=5e-4)
    np.testing.assert_allclose(m.edf_total, 17.9986147698585, atol=5e-2)
    np.testing.assert_allclose(m.sp[0], 0.161829581092981, rtol=5e-3)
    # sp[1] for s(temp.surf) sits in a flat tail (mgcv: 5.62, hea: 5.72) —
    # both are effectively "fully smoothed"; pin the resulting fit (REML,
    # edf above) instead of the sp itself.


def test_gam_fit_tw_score_no_worse_than_fixed_p():
    """Joint outer Newton over (ρ, log φ, θ_tw) only accepts steps that
    improve the criterion, so tw()'s REML score should be ≤ Tweedie(1.5)'s."""
    rng = np.random.default_rng(7)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=300, p_true=1.4)
    df = pl.DataFrame({"y": y, "x": x})
    m_fixed = gam("y ~ s(x, k=8)", df, family=Tweedie(p=1.5), method="REML")
    m_tw = gam("y ~ s(x, k=8)", df, family=tw(), method="REML")
    assert m_tw.REML_criterion <= m_fixed.REML_criterion + 1e-6


def test_tw_rejects_gcv_method():
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="REML"):
        gam("y ~ s(x, k=6)", df, family=tw(), method="GCV.Cp")


def test_tw_rejects_fixed_sp():
    rng = np.random.default_rng(0)
    x, y, _ = _simulate_compound_poisson_gamma(rng, n=100)
    df = pl.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="incompatible"):
        gam("y ~ s(x, k=6)", df, family=tw(), method="REML", sp=np.array([0.1]))


# =============================================================================
# 2. vis() — port of mgcv's vis.gam
# =============================================================================


@pytest.fixture(scope="module")
def trees_te():
    """trees with a 2D tensor smooth — the canonical vis.gam example."""
    data = (
        load_dataset("mgcv", "trees")
        .rename({"Volume": "vol", "Girth": "g", "Height": "h"})
    )
    m = gam("vol ~ te(g, h)", data=data, method="REML")
    return m, data


@pytest.fixture(scope="module")
def factor_model():
    """A model with one numeric and one factor RHS variable."""
    rng = np.random.RandomState(0)
    df = pl.DataFrame({
        "y": rng.randn(120),
        "x": rng.rand(120),
        "g": (["a", "b", "c"] * 40),
    })
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
    # Identity link → response == link, identical SEs
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
    # No false-positives: every kept cell in v1 == v0 (NaN-only diff)
    keep = ~np.isnan(v1.fit)
    assert np.allclose(v0.fit[keep], v1.fit[keep])


def test_cond_overrides_typical_value():
    """`cond={var: val}` shifts the held-fixed value, changing the surface."""
    rng = np.random.RandomState(1)
    df = pl.DataFrame({
        "y": rng.randn(80),
        "x1": rng.rand(80),
        "x2": rng.rand(80),
        "x3": rng.rand(80),
    })
    m = gam("y ~ s(x1) + s(x2) + s(x3)", data=df, method="REML")
    # x3 is held at median by default; override and the surface changes.
    v_default = m.vis(view=("x1", "x2"), n_grid=8)
    v_override = m.vis(view=("x1", "x2"), n_grid=8, cond={"x3": 0.9})
    # With purely-additive smooths the *shape* of the surface over (x1, x2)
    # only differs by an offset (the s(x3) at x3=median vs x3=0.9). So check
    # that fit_default - fit_override is a constant.
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
    df = pl.DataFrame({"y": np.arange(10.0), "x": np.arange(10.0)})
    m = gam("y ~ s(x)", data=df, method="REML")
    with pytest.raises(ValueError):
        # only one RHS var with variation — auto-pick fails
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
    # levels appear repeated in the grid; the unique non-empty labels must
    # be a subset of {"a", "b", "c"}.
    assert {t for t in yticks if t} <= {"a", "b", "c"}
    plt.close("all")


def test_invalid_plot_kind(trees_te):
    m, _ = trees_te
    v = m.vis(view=("g", "h"), n_grid=5)
    with pytest.raises(ValueError):
        v.plot(kind="surface")  # only contour/persp supported


# =============================================================================
# 3. get_difference() — port of itsadug::get_difference
# =============================================================================
#
# For each case under ``tests/fixtures/itsadug_plot_diff/<case_id>/``, we
# re-fit the same model in hea, replay the same ``(comp, cond, f, sim_ci,
# rm_ranef)`` arguments, and compare the per-row ``difference`` and ``CI``
# against R's output. For the ``sim_ci=True`` case we also check the
# deterministic ``se_fit`` (= ``sqrt(rowSums((X1-X2) Vc (X1-X2)^T))``) to
# high precision and the empirical ``crit`` to a loose tolerance —
# Python and R don't share an RNG, so the quantile of the
# max-abs-standardized-deviation envelope only matches to the Monte-Carlo
# SE.
#
# Fixtures are baked once via
# ``Rscript tests/scripts/itsadug_plot_diff_fixtures.R``; re-run that
# script if ``itsadug`` or the model design changes.

_ITSADUG_ROOT = Path(__file__).parent / "fixtures" / "itsadug_plot_diff"
_ITSADUG_MODEL_DIR = _ITSADUG_ROOT / "_model"
_ITSADUG_CASE_DIRS = sorted(
    p for p in _ITSADUG_ROOT.iterdir()
    if p.is_dir() and not p.name.startswith("_")
) if _ITSADUG_ROOT.exists() else []
_ITSADUG_CASE_IDS = [p.name for p in _ITSADUG_CASE_DIRS]


def _itsadug_load_data() -> pl.DataFrame:
    """Load the synthetic dataset and re-attach factor levels — CSV
    round-trip drops R's factor type, but hea is happy with either pl.Utf8
    or pl.Enum at materialize time. We use pl.Enum with the explicit level
    order R wrote (A,B,C / Y,Z) for parity with mgcv's contrasts.
    """
    df = pl.read_csv(_ITSADUG_MODEL_DIR / "data.csv", null_values="NA")
    df = df.with_columns([
        df["group"].cast(pl.Enum(["A", "B", "C"])),
        df["cohort"].cast(pl.Enum(["Y", "Z"])),
    ])
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
        # numeric arrays come through as list-of-numbers; string fixers
        # (cohort="Y") come through as list-of-strings.
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
    }


@pytest.mark.parametrize("case_id", _ITSADUG_CASE_IDS)
def test_get_difference_matches_itsadug(itsadug_fitted_model, case_id: str):
    m, _ = itsadug_fitted_model
    case_dir = _ITSADUG_ROOT / case_id
    args = _itsadug_parse_args(case_dir / "args.json")
    ref = pl.read_csv(case_dir / "diff_table.csv", null_values="NA")

    # rm.ranef arrives as a Python bool here — get_difference accepts it
    # as-is. None of our cases exercise the substring-grep mode (we don't
    # have ranef smooths in this v1 fixture set).
    rm_ranef = args["rm_ranef"]
    if not isinstance(rm_ranef, bool):
        rm_ranef = list(rm_ranef) if isinstance(rm_ranef, list) else rm_ranef

    res = m.get_difference(
        comp=args["comp"],
        cond=args["cond"],
        f=args["f"],
        sim_ci=args["sim_ci"],
        rm_ranef=rm_ranef,
        rng=20260430,  # deterministic for the sim.ci path; loose tol on crit
        n_sim=10_000,
    )

    assert res.difference.shape[0] == args["n_grid"], (
        f"{case_id}: got {res.difference.shape[0]} grid rows, "
        f"want {args['n_grid']}"
    )

    # difference is a basis-invariant linear functional of β̂, so the
    # only source of disagreement with mgcv is REML convergence drift
    # in the smoothing parameters. For this dataset, hea and mgcv agree
    # on sp[1]/sp[2] to ~1e-6 relative but on sp[3] (group=C, flat
    # signal) only to ~1% — the REML loglik is very flat there, and
    # both solvers stop in slightly different places. The resulting
    # difference noise lands at ~5e-5 absolute (5e-3 relative near
    # zero crossings) — orders of magnitude below the CI half-width
    # itself (~0.27 here), so still a tight oracle in any practical sense.
    np.testing.assert_allclose(
        res.difference,
        ref["difference"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: difference diverges from itsadug",
    )

    # CI is f * sqrt(diag(p Vp p^T)) — same convergence-drift bound.
    np.testing.assert_allclose(
        res.ci,
        ref["CI"].to_numpy(),
        rtol=1e-3,
        atol=2e-4,
        err_msg=f"{case_id}: CI diverges from itsadug",
    )

    if args["sim_ci"]:
        assert args["has_sim_ci_col"], "fixture mislabel: sim.ci=TRUE but no sim.CI column"
        assert res.sim_ci is not None and res.crit is not None

        # se_fit = sim_ci / crit — deterministic given Vc and p. Same
        # convergence-drift bound as ``CI``: tight in absolute terms,
        # bounded by REML sp agreement.
        ref_se_fit = pl.read_csv(case_dir / "se_fit.csv")["se_fit"].to_numpy()
        ours_se_fit = res.sim_ci / res.crit
        np.testing.assert_allclose(
            ours_se_fit,
            ref_se_fit,
            rtol=1e-3,
            atol=2e-4,
            err_msg=f"{case_id}: simultaneous se_fit diverges",
        )

        # crit is an empirical 0.95 quantile over n_sim=10000 MVN draws.
        # Cross-RNG comparison: the two implementations sample
        # independently so the quantile differs by Monte-Carlo SE. The
        # standard error of the 0.95 quantile of the MASD with n=10000
        # draws is roughly 0.5–2% of the value; allow 5% for safety.
        ref_crit = float(pl.read_csv(case_dir / "crit.csv")["crit"][0])
        np.testing.assert_allclose(
            res.crit, ref_crit, rtol=0.05,
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
        comp=args_b["comp"], cond=args_b["cond"], f=args_b["f"],
        sim_ci=False, rm_ranef=True,
    )
    res_y = m.get_difference(
        comp=args_y["comp"], cond=args_y["cond"], f=args_y["f"],
        sim_ci=False, rm_ranef=True,
    )
    np.testing.assert_allclose(res_y.difference, res_b.difference,
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_y.ci, res_b.ci,
                                rtol=1e-12, atol=1e-12)


# =============================================================================
# Tier-1 mgcv-parity fixes (testStat mixture p-values + √W design, Fletcher
# scale, rank detection, id= guard, reTest with sibling random effects).
#
# Reference values: R mgcv 1.9-4 run locally on the exact CSV each numpy
# generator below reproduces (tests never call R). The testStat pins
# discriminate against the old single-statistic F-only path, which is ~15%
# off on the low-p Gaussian case below.
# =============================================================================


def _borderline_gaussian(seed: int, amp: float, n: int = 130) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = amp * np.sin(4 * np.pi * x) + rng.normal(0, 1, n)
    return pl.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize(
    "seed, amp, expected",
    [
        # (edf, Ref.df, F, p) from mgcv summary(gam(y ~ s(x), method="REML"))
        (23, 0.45, (2.920565601204, 3.635185663159,
                    0.707551676855, 0.489004131254)),
        (31, 0.50, (5.62568065646878, 6.76317580531274,
                    4.83126594787145, 0.00011153338537)),
    ],
)
def test_teststat_mixture_pvalue_gaussian_matches_mgcv(seed, amp, expected):
    """Wood (2013) testStat: fractional-rank mixture reference distribution
    (psum.chisq) + d/d1 averaging — not the pf(F, rank, res.df) fallback."""
    m = gam("y ~ s(x)", _borderline_gaussian(seed, amp), method="REML")
    label, edf, ref_df, stat_col, p_val = m._smooth_significance_rows()[0]
    assert label == "s(x)"
    np.testing.assert_allclose(
        [edf, ref_df, stat_col, p_val], expected, rtol=5e-4,
        err_msg="s(x) row vs mgcv s.table",
    )


def test_teststat_mixture_pvalue_poisson_matches_mgcv():
    """Known-scale branch: chi-squared mixture via psum.chisq, and the
    statistic built on the √W-weighted design (mgcv tests against object$R,
    the QR factor of √W·X — unweighted X is only its legacy fallback)."""
    from hea.family import Poisson
    rng = np.random.default_rng(4)
    n = 160
    x = rng.uniform(0, 1, n)
    y = rng.poisson(np.exp(0.30 * np.sin(4 * np.pi * x)))
    d = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", d, family=Poisson(), method="REML")
    label, edf, ref_df, stat_col, p_val = m._smooth_significance_rows()[0]
    # mgcv: edf, Ref.df, Chi.sq, p-value
    np.testing.assert_allclose(
        [edf, ref_df, stat_col, p_val],
        (3.155091929152, 3.902446682533, 6.195704535045, 0.222514468845),
        rtol=5e-4, err_msg="s(x) row vs mgcv s.table (poisson)",
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
    assert m.p == 11          # internal (reduced) parameter count
    assert m._p_orig == 12    # original — what check()/summary report
    # one of the twins is zero-filled, the other carries the effect
    b1 = float(np.asarray(m.bhat["x1"])[0])
    b2 = float(np.asarray(m.bhat["x2"])[0])
    assert (b1 == 0.0) != (b2 == 0.0)
    assert abs(b1 + b2 - 2.0) < 0.25


def test_rank_deficient_drop_matches_mgcv_exactly():
    # Same construction at rng(3): mgcv (1.9-4) keeps one twin with
    # coefficient 1.98453773, rank 11, REML 22.46719612, sig2 0.07442324,
    # first prediction 0.84075695. The criterion keeps the *pre-drop*
    # Mp and log|S| basis (G$Mp and UrS are setup-time quantities) —
    # using post-drop Mp shifts REML by exactly ΔMp/2·log(2πφ̂) ≈ 0.38.
    rng = np.random.default_rng(3)
    n = 80
    x1 = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = 1.96 * x1 + np.sin(2 * np.pi * z) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x1.copy(), "z": z, "y": y})
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("y ~ x1 + x2 + s(z)", df, method="REML")
    kept = (float(np.asarray(m.bhat["x2"])[0])
            or float(np.asarray(m.bhat["x1"])[0]))
    np.testing.assert_allclose(kept, 1.98453773, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.REML_criterion / 2, 22.46719612,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.07442324,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.sp[0], 0.0085086098, rtol=1e-5)
    np.testing.assert_allclose(
        m.predict(df.head(1))["fit"][0], 0.84075695, rtol=0, atol=1e-6,
    )
    # t value of the dropped twin is NaN, like mgcv's p.table
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
    sharing (pooled knots over [0, 3] differ from each smooth's own)."""
    rng = np.random.default_rng(13)
    n = 250
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 3, n)
    y = np.sin(2 * np.pi * x0) + np.sin(2 * np.pi * x1 / 3) \
        + rng.normal(0, 0.35, n)
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y})


@pytest.mark.parametrize(
    "formula, exp_sp, exp_edf, exp_reml",
    [
        # mgcv references on the exact _id_linked_data() CSV:
        #   gam(y ~ s(x0, bs=..., id=1) + s(x1, bs=..., id=1), method="REML")
        ("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)",
         2.73138803, (5.388691, 7.840385), 111.9803343),
        ("y ~ s(x0, id=1) + s(x1, id=1)",          # tp (default basis)
         0.000950795685, (4.156817, 8.746249), 116.2335752),
    ],
)
def test_id_links_smoothing_parameters_matches_mgcv(
    formula, exp_sp, exp_edf, exp_reml,
):
    """mgcv id= semantics: ONE working λ shared across the linked smooths
    (L-matrix), bases built from POOLED covariate values (idLinksBases),
    penalties rescaled and constrained against the pooled construction —
    sp, per-smooth edf, and the REML score all pin to mgcv."""
    m = gam(formula, _id_linked_data(), method="REML")
    assert len(m.sp) == 1                 # working sp (mgcv's m$sp)
    assert len(m._slots) == 2             # two penalties share it
    np.testing.assert_allclose(np.exp(m._rho_hat), [m.sp[0]] * 2, rtol=1e-12)
    np.testing.assert_allclose(m.sp[0], exp_sp, rtol=1e-4)
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()), exp_edf, rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, exp_reml, rtol=1e-6)
    # The shared raw basis must replay at predict time (renamed view for
    # the second smooth).
    pred = m.predict(_id_linked_data().head(40))["fit"].to_numpy()
    np.testing.assert_allclose(pred, m.fitted[:40], rtol=1e-10)


def test_id_by_factor_single_lambda_matches_mgcv():
    """``s(x2, by=fac, id=1)``: all by-level blocks share one λ — the
    canonical id idiom (mgcv gam.models docs; fixture mgcv_0080's formula).
    mgcv reference: sp=(0.0133409281, 0.0206449404), full.sp repeats the
    first across the three level blocks; -REML=189.4203017,
    scale=0.15601042."""
    rng = np.random.default_rng(5)
    n = 300
    x2 = rng.uniform(0, 1, n)
    x0 = rng.uniform(0, 1, n)
    fac = rng.integers(1, 4, n)
    fl = np.array([0.0, 1.0, 2.0])[fac - 1]
    amp = np.where(fac == 1, 1.0, np.where(fac == 2, 1.5, 0.5))
    y = fl + amp * np.sin(2 * np.pi * x2) + np.cos(2 * np.pi * x0) \
        + rng.normal(0, 0.4, n)
    df = pl.DataFrame({
        "x2": x2, "x0": x0, "fac": [f"f{i}" for i in fac], "y": y,
    }).with_columns(pl.col("fac").cast(pl.Enum(["f1", "f2", "f3"])))
    m = gam("y ~ fac + s(x2, by=fac, id=1) + s(x0)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(
        m.sp, [0.0133409281, 0.0206449404], rtol=1e-4,
    )
    np.testing.assert_allclose(            # full.sp expansion
        np.exp(m._rho_hat),
        [m.sp[0], m.sp[0], m.sp[0], m.sp[1]], rtol=1e-12,
    )
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()),
        [5.9821444, 5.9008228, 5.7796811, 6.4424672], rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 189.4203017, rtol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.15601042, rtol=1e-5)


def test_id_tensor_smooths_match_mgcv():
    """id across te() smooths links pairwise (1st penalty ↔ 1st, 2nd ↔
    2nd) with pooled marginal bases. mgcv reference: sp[0]=0.1522418706,
    sp[1] on the flat λ→∞ tail (mgcv 3.58e6 — only its order of magnitude
    is determined); -REML=77.67269936; per-smooth edf 8.760963/8.7305329;
    scale 0.090709176."""
    rng = np.random.default_rng(21)
    n = 220
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    z, u = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x0) * np.cos(np.pi * z)
         + 0.8 * np.sin(2 * np.pi * x1) * np.cos(np.pi * u)
         + rng.normal(0, 0.3, n))
    df = pl.DataFrame({"x0": x0, "x1": x1, "z": z, "u": u, "y": y})
    m = gam("y ~ te(x0, z, id=1) + te(x1, u, id=1)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(m.sp[0], 0.1522418706, rtol=1e-4)
    assert m.sp[1] > 1e5                  # flat saturation tail
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()), [8.760963, 8.7305329], rtol=1e-4,
    )
    np.testing.assert_allclose(m.REML_criterion / 2, 77.67269936, rtol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.090709176, rtol=1e-5)


def test_id_fixed_sp_takes_working_length():
    """``sp=`` supplies the *working* parameters (mgcv semantics): one
    value drives both linked penalties. mgcv reference at sp=2.0:
    sum(edf)=14.594997."""
    d = _id_linked_data()
    m = gam("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)", d, sp=[2.0])
    np.testing.assert_allclose(m.edf_total, 14.594997, rtol=1e-5)
    with pytest.raises(ValueError, match="length 1"):
        gam("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)", d,
            sp=[2.0, 3.0])


def test_id_singleton_is_noop():
    """An id used by a single smooth links nothing — same model as no id."""
    d = _id_linked_data()
    m1 = gam("y ~ s(x0, id=9) + s(x1)", d, method="REML")
    m0 = gam("y ~ s(x0) + s(x1)", d, method="REML")
    assert len(m1.sp) == 2
    np.testing.assert_allclose(m1.sp, m0.sp, rtol=1e-10)
    np.testing.assert_allclose(
        m1.REML_criterion, m0.REML_criterion, rtol=1e-12,
    )


def test_bam_still_rejects_id():
    """bam has no L-matrix layer yet — must refuse rather than silently
    fit independent λ's."""
    from hea.models.bam import bam
    d = _id_linked_data()
    with pytest.raises(NotImplementedError, match="id="):
        bam("y ~ s(x0, id=1) + s(x1, id=1)", d)


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
    rng = np.random.default_rng(7)
    n = 250
    x = rng.uniform(0, 1, n)
    g1 = rng.integers(0, 8, n)
    g2 = rng.integers(0, 6, n)
    b1 = rng.normal(0, 0.5, 8)
    b2 = rng.normal(0, 0.09, 6)
    y = np.sin(2 * np.pi * x) + b1[g1] + b2[g2] + rng.normal(0, 0.5, n)
    df = pl.DataFrame({
        "x": x,
        "g1": [f"a{i}" for i in g1],
        "g2": [f"b{i}" for i in g2],
        "y": y,
    }).with_columns(
        pl.col("g1").cast(pl.Enum([f"a{i}" for i in range(8)])),
        pl.col("g2").cast(pl.Enum([f"b{i}" for i in range(6)])),
    )
    m = gam("y ~ s(x) + s(g1, bs='re') + s(g2, bs='re')", df, method="REML")
    rows = {r[0]: r[1:] for r in m._smooth_significance_rows()}
    # mgcv s.table (edf, Ref.df, F, p) on this exact dataset:
    np.testing.assert_allclose(
        rows["s(g1)"],
        (6.90386745017, 7.0, 88.75859491707, 0.0),
        rtol=5e-4, atol=1e-12, err_msg="s(g1) vs mgcv",
    )
    np.testing.assert_allclose(
        rows["s(g2)"],
        (3.5166733495327, 5.0, 3.5106246383545, 0.0198880041678),
        rtol=5e-4, err_msg="s(g2) vs mgcv",
    )


# ---------------------------------------------------------------------------
# 1.6 PIRLS control parity (gam.fit3 inner loop): full-Newton steps for
# non-canonical links, signed Newton weights in the score (no wholesale
# Fisher fallback), fix.family starting values, maxit/gradient-check.
# ---------------------------------------------------------------------------

def _noncanonical_pirls_data():
    rng = np.random.default_rng(101)
    n = 400
    x = rng.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * x ** 2
    mu = np.exp(0.3 + f)
    y_gamma = rng.gamma(shape=3.0, scale=mu / 3.0)
    y_glog = mu + rng.normal(0, 1.0, n)
    return pl.DataFrame({"x": x, "yg": y_gamma, "yn": y_glog})


def test_pirls_noncanonical_gamma_log_matches_mgcv():
    # gam(yg ~ s(x), Gamma(log), REML) — non-canonical link, so the inner
    # loop takes full-Newton steps (gam.fit3.r:118). mgcv 1.9-4 reference.
    df = _noncanonical_pirls_data()
    m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 516.0426205989,
                               rtol=0, atol=1e-6, err_msg="REML")
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.9636897232,
                               rtol=0, atol=5e-4, err_msg="edf")
    np.testing.assert_allclose(m.sigma_squared, 0.357718681258,
                               rtol=0, atol=5e-6, err_msg="sig2")


def test_pirls_gaussian_log_negative_newton_weights_match_mgcv():
    # gaussian(link="log") with y ≤ 0 in 29 rows: needs mgcv's fix.family
    # starting values (mustart = pmax(y, .01·sd(y)), gam.fit3.r:2550), and
    # at convergence 26 rows carry *negative* Newton weights — mgcv keeps
    # the signed weights in the REML score (gam.fit3.r:505-515). A
    # Fisher-fallback score is off by ~0.06 on this criterion.
    df = _noncanonical_pirls_data()
    from hea.family import Gaussian
    m = gam("yn ~ s(x)", df, family=Gaussian(link="log"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 593.6963879213,
                               rtol=0, atol=1e-6, err_msg="REML")
    np.testing.assert_allclose(m.sp[0], 0.09667572681,
                               rtol=1e-3, err_msg="sp")
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.6908929715,
                               rtol=0, atol=1e-3, err_msg="edf")
    np.testing.assert_allclose(m.sigma_squared, 1.064863735326,
                               rtol=0, atol=5e-6, err_msg="sig2")
    b0 = float(np.asarray(m.coef)[0])
    np.testing.assert_allclose(b0, 0.5736692935, rtol=0, atol=1e-5,
                               err_msg="intercept")


def test_pirls_clean_fits_emit_no_warnings():
    # converged/boundary/warn plumbing: a healthy fit must surface nothing.
    import warnings as w
    df = _noncanonical_pirls_data()
    with w.catch_warnings():
        w.simplefilter("error")
        m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    assert float(np.sum(m.edf)) > 1.0


# ---------------------------------------------------------------------------
# 1.7 micro-divergences: null deviance semantics (offset / no-intercept),
# scaled.pearson residuals.
# ---------------------------------------------------------------------------

def test_null_deviance_offset_and_no_intercept_match_mgcv():
    # mgcv: gam.fit3 always runs with intercept=TRUE (mgcv.r:1667) so the
    # base null deviance is dev(weighted-mean) for every formula; for
    # intercept+offset models estimate.gam refits glm(y ~ offset(off))
    # (mgcv.r:2072-2075). df.null = n-1 always. mgcv 1.9-4 references.
    rng = np.random.default_rng(33)
    n = 200
    x = rng.uniform(0, 1, n)
    expo = rng.uniform(0.5, 2.0, n)
    mu = expo * np.exp(0.4 + np.sin(2 * np.pi * x))
    y = rng.poisson(mu).astype(float)
    df = pl.DataFrame({"x": x, "expo": expo, "y": y})
    from hea.family import Poisson

    m = gam("y ~ s(x) - 1 + offset(log(expo))", df, family=Poisson(),
            method="REML")
    np.testing.assert_allclose(m.null_deviance, 418.3499279531,
                               rtol=0, atol=1e-7)
    assert m.df_null == n - 1

    m2 = gam("y ~ s(x) + offset(log(expo))", df, family=Poisson(),
             method="REML")
    np.testing.assert_allclose(m2.null_deviance, 410.0056332077,
                               rtol=0, atol=1e-7)

    # scaled.pearson = pearson/√φ̂ (mgcv.r:3457); φ=1 for Poisson so the
    # no-intercept fit pins R's residuals(m, "scaled.pearson") directly.
    r = m.residuals_of("scaled.pearson")
    np.testing.assert_allclose(
        r[:3], [0.2833041422, 1.0547593514, 0.9580616495],
        rtol=0, atol=1e-5,
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
    # R: set.seed(1); runif(5)
    r = _RUnif(1)
    np.testing.assert_array_equal(
        [r.unif_rand() for _ in range(5)],
        [0.26550866314209998, 0.37212389963679016, 0.57285336335189641,
         0.90820778999477625, 0.2016819310374558],
    )
    # R: set.seed(1); sample(4000, 8)
    np.testing.assert_array_equal(
        _RUnif(1).sample_int(4000, 8) + 1,
        [1017, 3908, 679, 2177, 930, 1533, 471, 2347],
    )
    # R: set.seed(42); sample(10, 10)  (full permutation)
    np.testing.assert_array_equal(
        _RUnif(42).sample_int(10, 10) + 1,
        [1, 5, 10, 8, 2, 4, 6, 9, 7, 3],
    )
    # R: set.seed(1); sum(sample(3000, 2000)) — crosses an MT refill and
    # several pool-size power-of-two boundaries.
    s = _RUnif(1).sample_int(3000, 2000) + 1
    assert s.sum() == 2979991
    assert s[:5].tolist() == [1017, 679, 2177, 930, 1533]
    assert s[-3:].tolist() == [2694, 2568, 1897]


def test_tp_max_knots_subsample_matches_mgcv():
    # n=4000 > max.knots=2000: before the subsample port hea used all
    # unique rows and sp was 5-12% off mgcv; now the knot sets are
    # identical. mgcv 1.9-4: gam(y ~ s(x, k=20), REML).
    rng = np.random.default_rng(2024)
    n = 4000
    x = rng.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)
         + rng.normal(0, 0.4, n))
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x, k=20)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 2046.4493253413,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.009114465577, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 16.9714122248,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sigma_squared, 0.159927692016,
                               rtol=0, atol=1e-8)


def _tp_ds_2d_data():
    rng = np.random.default_rng(77)
    n = 3000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x1) * np.cos(np.pi * x2) + (x1 - 0.5) ** 2
         + rng.normal(0, 0.3, n))
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_tp_2d_subsample_matches_mgcv():
    # 2-D exercises uniquecombs(·,TRUE)'s C-locale string-sort row order
    # (the sample indexes into those rows). mgcv 1.9-4 references.
    df = _tp_ds_2d_data()
    m = gam("y ~ s(x1, x2, k=40)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 699.2650646040,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1574089045, rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 38.2145858974,
                               rtol=0, atol=1e-4)


def test_ds_subsample_and_reported_sp_match_mgcv():
    # Two fixes pinned here: the seeded ds knot subsample at n=3000, and
    # the Rlanczos basis in _lowrank_kernel_reduce — with a dense eigh
    # basis the *reported* sp was off mgcv by a model-invariant 7-18%
    # factor (identical REML/edf/fits) because clustered eigenvalues give
    # eigh a different orthonormal basis than mgcv's Lanczos, shifting
    # ‖S‖ and the sp that compensates for it.
    df = _tp_ds_2d_data()
    m3 = gam("y ~ s(x1, x2, bs='ds', k=40)", df, method="REML")
    np.testing.assert_allclose(m3.REML_criterion / 2, 696.2092773464,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m3.sp[0], 0.03414280754, rtol=1e-5)

    m15 = gam("y ~ s(x1, x2, bs='ds', k=40)", df.head(1500), method="REML")
    np.testing.assert_allclose(m15.REML_criterion / 2, 351.9531642893,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m15.sp[0], 0.02727861846, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2.3 outer-optimization seeding: initial.spg (√w-weighted initial.sp +
# global 0.4-rebalance) for every outer method, log(null.scale/10) for the
# φ slot. Seeds pinned directly against mgcv:::initial.spg / get.null.coef.
# ---------------------------------------------------------------------------

def test_initial_spg_seed_matches_mgcv():
    df = _noncanonical_pirls_data()
    m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    # R: log(mgcv:::initial.spg(G$X, G$y, G$w, G$family, G$S, G$rank,
    #        G$off, E=G$Eb)) on the same fixture (non-canonical weights).
    np.testing.assert_allclose(m._initial_sp_rho(), [0.7588679789],
                               rtol=0, atol=1e-8)
    # log(null.scale/10), null.scale = Σ dev_resids(y, ȳ)/n
    y = df["yg"].to_numpy()
    mu0 = np.full(len(y), y.mean())
    ns = float(np.sum(m.family.dev_resids(y, mu0, np.ones(len(y))))) / len(y)
    np.testing.assert_allclose(np.log(ns / 10.0), -2.5423759619,
                               rtol=0, atol=1e-8)

    # Two smooths — exercises the shared ×10 rebalance loop.
    df2 = _tp_ds_2d_data()
    m2 = gam("y ~ s(x1, x2, k=40) + s(x1, k=10)", df2, method="REML")
    np.testing.assert_allclose(
        m2._initial_sp_rho(), [1.4926773694, 2.5814449388],
        rtol=0, atol=1e-8,
    )

    # The seed moves the optimizer onto mgcv's stopping point: mcycle edf
    # was 8.624673 (hea's old path) and is mgcv's 8.624691 now.
    d = load_dataset("MASS", "mcycle")
    m3 = gam("accel ~ s(times)", d, method="REML")
    np.testing.assert_allclose(m3._initial_sp_rho(), [-1.0127433870],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(m3.edf)) - 1.0, 8.624691,
                               rtol=0, atol=2e-5)


# ---------------------------------------------------------------------------
# 2.4 tw() sp-uncertainty pieces: extended-family sig2 = exp(φ̂) reporting,
# family-θ column of db.drho in Vc/edf2, ML projected-Hessian θ term.
# ---------------------------------------------------------------------------

def _tw_24_data():
    rng = np.random.default_rng(55)
    n = 500
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x))
    y = rng.gamma(shape=2.0, scale=mu / 2.0)
    y[rng.uniform(size=n) < 0.08] = 0.0
    return pl.DataFrame({"x": x, "y": y})


def test_tw_reml_scale_and_vc_match_mgcv():
    # mgcv-extended families report sig2 = exp(φ̂_REML) (gam.outer's
    # scale.est), NOT the Fletcher estimator — Fletcher applied to tw was
    # 0.56% off, dragging Vp/Vc with it. mgcv 1.9-4 references.
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 826.7923690210,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.87769401,
                               rtol=0, atol=1e-6)
    assert m.sigma_squared == np.exp(m._log_phi_hat)
    np.testing.assert_allclose(m.sp[0], 0.11942987, rtol=1e-4)
    # Vc/edf2 with the family-θ column of db.drho; residual vs mgcv is
    # p̂-ridge noise (criteria agree to 6e-8, p̂ differs 1.5e-3).
    np.testing.assert_allclose(np.diag(m.Vc)[0], 0.0014742061, rtol=1e-3)
    np.testing.assert_allclose(m.edf2_total, 7.05826502, rtol=0, atol=2e-3)


def test_tw_db_dtheta_column_matches_finite_differences():
    # ∂β̂/∂θ_tw (the family-θ column of mgcv's db.drho): analytic
    # −A⁻¹X'(Dmuth·μ')/2 vs central differences of the PIRLS fixed point
    # over θ at fixed ρ. FD noise floor ≈ PIRLS tol/δ ≈ 1e-3 relative.
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
    # method="ML" + tw: the ∂log|M_proj|/∂p projected-Hessian term was
    # skipped, biasing the gradient — hea stopped with the ML criterion
    # 3e-5 off mgcv and φ̂ 0.8% off. With the term: criterion to 2.4e-8,
    # φ̂ to 8 digits. mgcv 1.9-4 references.
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 824.6125277246,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.87574253,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.13126347, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2.5 gam.control(edge.correct=): post-convergence walk of Hessian-flat
# smoothing parameters + the k=2 Vc recomputation with the weaker 1e-7 Vr
# prior (gam.fit3.r:1670-1716, post.proc K loop).
# ---------------------------------------------------------------------------

def test_edge_correct_vc_matches_mgcv():
    # On both fixtures mgcv's flat set is empty (lsp1 == lsp) and the
    # corrected Vc differs from the plain one purely through the k=2
    # 1e-7 prior — a 2.5x change on the flat smooth's null-space-adjacent
    # entries in the second fixture. mgcv 1.9-4 references.
    rng = np.random.default_rng(99)
    n = 300
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m1 = gam("y ~ s(x1) + s(x2)", df, method="REML", edge_correct=True)
    np.testing.assert_allclose(
        np.diag(m1.Vc)[[1, 9, 10]],
        [0.022485405, 0.084568825, 0.0027264787], rtol=5e-4,
    )
    # edf2 keeps the fitted-model (k=1) value.
    np.testing.assert_allclose(m1.edf2_total, 11.06057347, rtol=2e-5)

    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + rng.normal(0, 0.25, n)
    df2 = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m2 = gam("y ~ s(x1) + s(x2)", df2, method="REML", edge_correct=True)
    np.testing.assert_allclose(
        np.diag(m2.Vc)[[1, 10, 11]],
        [0.025712048, 0.0069904269, 0.013000829], rtol=5e-4,
    )
    m0 = gam("y ~ s(x1) + s(x2)", df2, method="REML")
    np.testing.assert_allclose(
        np.diag(m0.Vc)[[1, 10, 11]],
        [0.025189013, 0.0027884243, 0.0046002536], rtol=5e-4,
    )

    with pytest.raises(ValueError, match="edge_correct"):
        gam("y ~ s(x1)", df2, method="REML", edge_correct=-1.0)


# ---------------------------------------------------------------------------
# 2.2 gam.reparam / get_stableS: log|Sλ|+ and its ρ-derivatives via mgcv's
# similarity-transform reparameterization (gdi.c:550-792) — immune to
# λ-ratio "machine zero leakage" between penalty components.
# ---------------------------------------------------------------------------

def test_get_stable_s_matches_mgcv_oracle():
    # Synthetic penalty roots; oracle values from mgcv:::gam.reparam on
    # the same matrices (R 4.x / mgcv 1.9-4). The disjoint case (3+3 in
    # 6-dim) shows the leakage: at lsp=(0,40) the assembled-eigen
    # determinant is off by 11.2; get_stableS is exact.
    from hea.models.gam import _gam_reparam
    rng = np.random.default_rng(404)
    R1 = rng.normal(size=(3, 6)).round(6)
    rng.normal(size=(3, 6))  # R2 drawn but unused here (keeps stream)
    rng2 = np.random.default_rng(405)
    R3 = rng2.normal(size=(4, 6)).round(6)

    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 0.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 5.6213367839, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rp["det1"], [2.5973053960, 3.4026946040],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        rp["det2"], [[0.2405316599, -0.2405316599],
                     [-0.2405316599, 0.2405316599]], rtol=0, atol=1e-9,
    )
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 20.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 84.7117599761, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rp["det1"], [2.0000000031, 3.9999999969],
                               rtol=0, atol=1e-8)
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 40.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 164.7117599730, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rp["det1"], [2.0, 4.0], rtol=0, atol=1e-8)
    # E is a stable square root: E'E = S.
    np.testing.assert_allclose(rp["E"].T @ rp["E"], rp["S"],
                               rtol=0, atol=1e-12 * np.abs(rp["S"]).max())
    # The assembled-eigen determinant fails here (the stress this guards)
    # — same top-r + clip recipe as the legacy _log_det_S_pos.
    S1, S3 = R1.T @ R1, R3.T @ R3
    lam = np.exp(np.array([0.0, 40.0]))
    w = np.linalg.eigvalsh(lam[0] * S1 + lam[1] * S3)
    top = np.clip(np.sort(w)[::-1][:6], 1e-300, None)
    det_naive = float(np.sum(np.log(top)))
    assert abs(det_naive - 164.7117599730) > 1.0


def test_extreme_fixed_sp_tensor_criterion_matches_mgcv():
    # Fit-level leakage stress: te() with fixed sp=(1e-8, 1e8) — λ ratio
    # 1e16 *within one block*. The legacy assembled-eigen log|S|+ path
    # returns a criterion off by ~334 here; gam.reparam lands on mgcv to
    # 4e-8. Free-fit pins confirm the optimizer surface is unchanged on
    # the healthy regime. mgcv 1.9-4 references.
    rng = np.random.default_rng(77)
    n = 3000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x1) * np.cos(np.pi * x2) + (x1 - 0.5) ** 2
         + rng.normal(0, 0.3, n))
    df = pl.DataFrame({"x1": x1[:400], "x2": x2[:400], "y": y[:400]})

    m = gam("y ~ te(x1, x2, k=5)", df, method="REML",
            sp=np.array([1e-8, 1e8]))
    np.testing.assert_allclose(m.REML_criterion / 2, 151.4513862986,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.0905718512,
                               rtol=0, atol=1e-9)

    m2 = gam("y ~ te(x1, x2, k=5)", df, method="REML")
    np.testing.assert_allclose(m2.REML_criterion / 2, 107.5309262948,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m2.sp, [0.061977, 162.243], rtol=1e-4)
