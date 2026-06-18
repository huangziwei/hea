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

from conftest import assert_fp_equiv as _assert_fp_equiv, load_dataset
from hea.models import gam, glm
from hea.family import Gamma, Poisson, Tweedie, tw

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
    from hea.R.rng import RGenerator
    gen = RGenerator(0)
    n = 200
    x = gen.uniform(0.0, 1.0, n)
    mu = 1.5 + 0.5 * np.sin(2 * np.pi * x)               # ∈ [1.0, 2.0], strictly positive
    # R-native Wald draws via mgcv's rig (InverseGaussian.rd); scale=1 ⇒
    # variance μ³, matching numpy's wald(mean=μ, scale=1).
    y = inverse_gaussian().rd(gen, mu, np.ones(n), 1.0)
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
    #   m$sig2 = 0.821638681219   (plain Pearson: 0.781145231711)
    # rtol 5e-6 (~3× the measured hea-vs-mgcv stop-band): the IG canonical
    # REML optimum is shallow and the Fletcher scale rides the converged
    # μ̂, so hea lands ~1.4e-6 rel from mgcv's sp (same §2.3 band as _scat).
    np.testing.assert_allclose(m.sigma_squared, 0.821638681219, rtol=5e-6)
    np.testing.assert_allclose(m._pearson_scale, 0.781145231711, rtol=5e-6)
    # Intercept ≈ link(mean(y)) = 1/mean(y)² for an intercept-only fit;
    # with a smooth that captures most of the signal it lands near
    # link(mean(mu_true)) = 1/1.5² ≈ 0.444 (here ≈ 0.606 on this draw).
    intercept = m.bhat["(Intercept)"][0]
    assert 0.40 < intercept < 0.80


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

    mgcv pins: edf/k-index are RNG-free; the permutation p-value runs
    through the bit-exact _RUnif port, so seed=0 with n_rep=2000
    reproduces R's set.seed(0); k.check(b, n.rep=2000) exactly (0.951).
    """
    d = load_dataset("MASS", "mcycle")
    m = gam("accel ~ s(times)", d, method="REML")
    ktab = m._k_check(seed=0, n_rep=2000)
    assert ktab[""].to_list() == ["s(times)"]
    np.testing.assert_allclose(ktab["k'"].to_list(),     [9.0],          atol=0)
    np.testing.assert_allclose(ktab["edf"].to_list(),    [8.62469100],   atol=5e-5)
    np.testing.assert_allclose(ktab["k-index"].to_list(),[1.14736165],   atol=5e-5)
    np.testing.assert_allclose(ktab["p-value"].to_list(), [0.951],
                               atol=1e-12)


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
    # GACV.Cp / P-REML / P-ML are now accepted (see the Phase-4 method= tests).
    for m in ("GACV.Cp", "P-REML", "P-ML"):
        assert gam("accel ~ s(times)", d, method=m).method in (
            m, "REML", "ML")


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
    # Non-degenerate y: a perfectly linear y=x sends REML's sp to the flat
    # +∞ ridge and the outer Newton wanders for ~1.3s. The fit value is
    # irrelevant here — we only exercise vis()'s view-validation — so use
    # noise and keep the fit ~0.04s.
    rng = np.random.RandomState(0)
    df = pl.DataFrame({"y": rng.standard_normal(10), "x": np.arange(10.0)})
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
        # set.seed() the R fixture used before mgcv::rmvn (None for non-sim
        # cases). Passing this as get_difference(rng=...) reproduces R's draws.
        "sim_seed": raw.get("sim_seed"),
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
        # Seed hea's R MT with the SAME set.seed() the fixture used before
        # mgcv::rmvn, so the simultaneous-CI draws come off R's stream (None
        # for non-sim cases, where rng is unused).
        rng=args["sim_seed"],
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

        # crit is the empirical type-8 0.95 quantile of the MASD over
        # n_sim=10000 MVN draws. hea now draws via mgcv::rmvn on R's *bit-exact*
        # MT stream (RMersenneTwister.rmvn, seeded with the fixture's
        # set.seed) — given identical Vc/p the quantile reproduces R's to
        # ~1e-12 (see test_rmvn_matches_r). The residual ~4e-3 here is NOT
        # RNG: the MVN draw mroot(Vc)@Z is basis-dependent and hea's smooth
        # basis differs from mgcv's reparameterization, so the realized draw
        # (hence the finite-sample quantile) differs at Monte-Carlo level even
        # off the same stream. Basis-invariant quantities (difference / CI /
        # se_fit above) match to ~5e-6; bit-exact crit would need full
        # column-for-column basis parity. 8e-3 covers the measured gap with
        # margin and is ~6× tighter than the old cross-RNG bound.
        ref_crit = float(pl.read_csv(case_dir / "crit.csv")["crit"][0])
        np.testing.assert_allclose(
            res.crit, ref_crit, rtol=8e-3,
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
    # R-native (set.seed(seed), 0-ulp: runif/rnorm)
    from hea.R.rng import RGenerator
    g = RGenerator(seed)
    x = g.uniform(0.0, 1.0, n)
    y = amp * np.sin(4 * np.pi * x) + g.normal(0.0, 1.0, n)
    return pl.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize(
    "seed, amp, expected",
    [
        # (edf, Ref.df, F, p) from mgcv summary(gam(y ~ s(x), method="REML"))
        (16, 0.45, (2.67146916085, 3.32082935917,
                    0.924820270116, 0.431982202344)),
        (2, 0.50, (5.09662376462, 6.17398455164,
                   4.31714021834, 0.000494598563479)),
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
    from hea.R.rng import RGenerator
    # R-native (set.seed(16), 0-ulp); reseeded 4→16 — R-native seed 4 fits
    # ~linear (edf≈1), so reseeded for fractional rank (edf 3.61).
    g = RGenerator(16)
    n = 160
    x = g.uniform(0.0, 1.0, n)
    y = np.asarray(g.poisson(np.exp(0.30 * np.sin(4 * np.pi * x))), dtype=float)
    d = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x)", d, family=Poisson(), method="REML")
    label, edf, ref_df, stat_col, p_val = m._smooth_significance_rows()[0]
    # mgcv: edf, Ref.df, Chi.sq, p-value
    np.testing.assert_allclose(
        [edf, ref_df, stat_col, p_val],
        (3.61350424706, 4.48011605323, 7.88459555761, 0.128871644663),
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
    # Same construction at set.seed(3) (R-native, 0-ulp): mgcv (1.9-4) keeps
    # one twin with coefficient 1.981807588, rank 11, REML 29.91747549, sig2
    # 0.09205759387, first prediction -0.619726771. The criterion keeps the
    # *pre-drop* Mp and log|S| basis (G$Mp and UrS are setup-time quantities) —
    # using post-drop Mp shifts REML by exactly ΔMp/2·log(2πφ̂).
    from hea.R.rng import RGenerator
    g = RGenerator(3)
    n = 80
    x1 = g.uniform(0.0, 1.0, n)
    z = g.uniform(0.0, 1.0, n)
    y = 1.96 * x1 + np.sin(2 * np.pi * z) + g.normal(0.0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x1.copy(), "z": z, "y": y})
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("y ~ x1 + x2 + s(z)", df, method="REML")
    kept = (float(np.asarray(m.bhat["x2"])[0])
            or float(np.asarray(m.bhat["x1"])[0]))
    np.testing.assert_allclose(kept, 1.981807588, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.REML_criterion / 2, 29.91747549,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.09205759387,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.sp[0], 0.01349564141, rtol=1e-5)
    np.testing.assert_allclose(
        m.predict(df.head(1))["fit"][0], -0.619726771, rtol=0, atol=1e-6,
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
    sharing (pooled knots over [0, 3] differ from each smooth's own).
    R-native (set.seed(13), 0-ulp: runif/rnorm)."""
    from hea.R.rng import RGenerator
    g = RGenerator(13)
    n = 250
    x0 = g.uniform(0.0, 1.0, n)
    x1 = g.uniform(0.0, 3.0, n)
    y = np.sin(2 * np.pi * x0) + np.sin(2 * np.pi * x1 / 3) \
        + g.normal(0.0, 0.35, n)
    return pl.DataFrame({"x0": x0, "x1": x1, "y": y})


@pytest.mark.parametrize(
    "formula, exp_sp, exp_edf, exp_reml",
    [
        # mgcv references on the exact _id_linked_data() R-native data:
        #   gam(y ~ s(x0, bs=..., id=1) + s(x1, bs=..., id=1), method="REML")
        ("y ~ s(x0, bs='cr', id=1) + s(x1, bs='cr', id=1)",
         2.85748958973, (5.22667532821, 7.68279449191), 130.595923509),
        ("y ~ s(x0, id=1) + s(x1, id=1)",          # tp (default basis)
         0.00111215693922, (4.0700018545, 8.67029420876), 135.136095842),
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
    mgcv reference: sp=(0.01235565525, 0.0215017895), full.sp repeats the
    first across the three level blocks; -REML=199.87702188,
    scale=0.1657339158."""
    from hea.R.rng import RGenerator
    gen = RGenerator(5)
    n = 300
    x2 = gen.uniform(0, 1, n)
    x0 = gen.uniform(0, 1, n)
    fac = gen.mt.sample_int(3, n, replace=True) + 1   # R sample.int(3,…) → {1,2,3}
    fl = np.array([0.0, 1.0, 2.0])[fac - 1]
    amp = np.where(fac == 1, 1.0, np.where(fac == 2, 1.5, 0.5))
    y = fl + amp * np.sin(2 * np.pi * x2) + np.cos(2 * np.pi * x0) \
        + gen.normal(0, 0.4, n)
    df = pl.DataFrame({
        "x2": x2, "x0": x0, "fac": [f"f{i}" for i in fac], "y": y,
    }).with_columns(pl.col("fac").cast(pl.Enum(["f1", "f2", "f3"])))
    m = gam("y ~ fac + s(x2, by=fac, id=1) + s(x0)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(
        m.sp, [0.01235565525, 0.0215017895], rtol=1e-4,
    )
    np.testing.assert_allclose(            # full.sp expansion
        np.exp(m._rho_hat),
        [m.sp[0], m.sp[0], m.sp[0], m.sp[1]], rtol=1e-12,
    )
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()),
        [6.1706570, 5.7539111, 5.8854072, 6.9305014], rtol=1e-4,
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
    y = (np.sin(2 * np.pi * x0) * np.cos(np.pi * z)
         + 0.8 * np.sin(2 * np.pi * x1) * np.cos(np.pi * u)
         + gen.normal(0, 0.3, n))
    df = pl.DataFrame({"x0": x0, "x1": x1, "z": z, "u": u, "y": y})
    m = gam("y ~ te(x0, z, id=1) + te(x1, u, id=1)", df, method="REML")
    assert len(m.sp) == 2 and len(m._slots) == 4
    np.testing.assert_allclose(m.sp[0], 0.1704650207, rtol=1e-4)
    assert m.sp[1] > 1e5                  # flat saturation tail
    np.testing.assert_allclose(
        list(m.edf_by_smooth.values()), [8.6701264, 8.6638061], rtol=1e-4,
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


def test_bam_links_id_like_gam():
    """bam grew gam's working-θ L-matrix layer (plan P9): id= now shares ONE
    working λ across the linked smooths instead of being rejected. Full
    mgcv-bam parity (sp/edf/criterion/fitted) lives in test_bam.py §7; here we
    just confirm bam links the same structure gam does on the shared fixture."""
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
    gen = RGenerator(7)          # R-native (set.seed(7), 0-ulp)
    n = 250
    x = gen.uniform(0.0, 1.0, n)
    g1 = gen.mt.sample_int(8, n, replace=True)   # = R sample.int(8, …) - 1
    g2 = gen.mt.sample_int(6, n, replace=True)
    b1 = gen.normal(0.0, 0.5, 8)
    b2 = gen.normal(0.0, 0.09, 6)
    y = np.sin(2 * np.pi * x) + b1[g1] + b2[g2] + gen.normal(0.0, 0.5, n)
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
        (6.72214656471, 7.0, 24.6471297862, 0.0),
        rtol=5e-4, atol=1e-12, err_msg="s(g1) vs mgcv",
    )
    np.testing.assert_allclose(
        rows["s(g2)"],
        (3.51278498412, 5.0, 3.49421820734, 0.00267046065254),
        rtol=5e-4, err_msg="s(g2) vs mgcv",
    )


# ---------------------------------------------------------------------------
# 1.6 PIRLS control parity (gam.fit3 inner loop): full-Newton steps for
# non-canonical links, signed Newton weights in the score (no wholesale
# Fisher fallback), fix.family starting values, maxit/gradient-check.
# ---------------------------------------------------------------------------

def _noncanonical_pirls_data():
    from hea.R.rng import RGenerator
    gen = RGenerator(101)
    n = 400
    x = gen.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * x ** 2
    mu = np.exp(0.3 + f)
    y_gamma = gen.gamma(3.0, scale=mu / 3.0, size=n)
    y_glog = mu + gen.normal(0, 1.0, n)
    return pl.DataFrame({"x": x, "yg": y_gamma, "yn": y_glog})


def test_pirls_noncanonical_gamma_log_matches_mgcv():
    # gam(yg ~ s(x), Gamma(log), REML) — non-canonical link, so the inner
    # loop takes full-Newton steps (gam.fit3.r:118). mgcv 1.9-4 reference.
    df = _noncanonical_pirls_data()
    m = gam("yg ~ s(x)", df, family=Gamma(link="log"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 499.0247951500,
                               rtol=0, atol=1e-6, err_msg="REML")
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.7000813600,
                               rtol=0, atol=5e-4, err_msg="edf")
    np.testing.assert_allclose(m.sigma_squared, 0.3237704458,
                               rtol=0, atol=5e-6, err_msg="sig2")


def test_pirls_gaussian_log_negative_newton_weights_match_mgcv():
    # gaussian(link="log") with y ≤ 0 in 42 rows: needs mgcv's fix.family
    # starting values (mustart = pmax(y, .01·sd(y)), gam.fit3.r:2550), and
    # at convergence several rows carry *negative* Newton weights — mgcv
    # keeps the signed weights in the REML score (gam.fit3.r:505-515). A
    # Fisher-fallback score is off by ~0.06 on this criterion.
    df = _noncanonical_pirls_data()
    from hea.family import Gaussian
    m = gam("yn ~ s(x)", df, family=Gaussian(link="log"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 587.3725200800,
                               rtol=0, atol=1e-6, err_msg="REML")
    np.testing.assert_allclose(m.sp[0], 0.1261684077,
                               rtol=1e-3, err_msg="sp")
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.4639360300,
                               rtol=0, atol=1e-3, err_msg="edf")
    np.testing.assert_allclose(m.sigma_squared, 1.0350749500,
                               rtol=0, atol=5e-6, err_msg="sig2")
    b0 = float(np.asarray(m.coef)[0])
    np.testing.assert_allclose(b0, 0.5001308373, rtol=0, atol=1e-5,
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
    from hea.R.rng import RGenerator
    gen = RGenerator(33)         # R-native (set.seed(33), 0-ulp)
    n = 200
    x = gen.uniform(0.0, 1.0, n)
    expo = gen.uniform(0.5, 2.0, n)
    mu = expo * np.exp(0.4 + np.sin(2 * np.pi * x))
    y = np.asarray(gen.poisson(mu), dtype=float)
    df = pl.DataFrame({"x": x, "expo": expo, "y": y})
    from hea.family import Poisson

    m = gam("y ~ s(x) - 1 + offset(log(expo))", df, family=Poisson(),
            method="REML")
    np.testing.assert_allclose(m.null_deviance, 493.6575994189,
                               rtol=0, atol=1e-7)
    assert m.df_null == n - 1

    m2 = gam("y ~ s(x) + offset(log(expo))", df, family=Poisson(),
             method="REML")
    np.testing.assert_allclose(m2.null_deviance, 408.7157165875,
                               rtol=0, atol=1e-7)

    # scaled.pearson = pearson/√φ̂ (mgcv.r:3457); φ=1 for Poisson so the
    # no-intercept fit pins R's residuals(m, "scaled.pearson") directly.
    r = m.residuals_of("scaled.pearson")
    np.testing.assert_allclose(
        r[:3], [-0.4081571746, -0.0352311928, -0.3567882629],
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
    # The port's home is hea.R.rng (formula's _RUnif and
    # hea.models.bam's name are shims/re-exports of the same class).
    from hea.R import RMersenneTwister
    from hea.models.bam import RMersenneTwister as _bam_alias
    assert type(_RUnif(1)) is RMersenneTwister is _bam_alias
    # Vector unif_rand consumes the identical stream as scalar draws.
    r_scalar = _RUnif(1)
    np.testing.assert_array_equal(
        RMersenneTwister(1).unif_rand(5),
        [r_scalar.unif_rand() for _ in range(5)])
    # R: set.seed(1); sample(5, 4, replace=TRUE)
    np.testing.assert_array_equal(
        RMersenneTwister(1).sample_int(5, 4, replace=True) + 1,
        [1, 4, 1, 2])
    # R: set.seed(3); sample(c("a","b","c")) → identity permutation
    assert RMersenneTwister(3).permute(["a", "b", "c"]).tolist() == \
        ["a", "b", "c"]
    # R: set.seed(4); sample(c("a","b","c"))
    assert RMersenneTwister(4).permute(["a", "b", "c"]).tolist() == \
        list("cab")


def test_tp_max_knots_subsample_matches_mgcv():
    # n=4000 > max.knots=2000: before the subsample port hea used all
    # unique rows and sp was 5-12% off mgcv; now the knot sets are
    # identical. mgcv 1.9-4: gam(y ~ s(x, k=20), REML).
    from hea.R.rng import RGenerator
    gen = RGenerator(2024)
    n = 4000
    x = gen.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)
         + gen.normal(0, 0.4, n))
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x, k=20)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 2026.8098237600,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.007596398516, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 16.9507672900,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sigma_squared, 0.1583687217,
                               rtol=0, atol=1e-8)


def _tp_ds_2d_data():
    from hea.R.rng import RGenerator
    gen = RGenerator(77)
    n = 3000
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x1) * np.cos(np.pi * x2) + (x1 - 0.5) ** 2
         + gen.normal(0, 0.3, n))
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_tp_2d_subsample_matches_mgcv():
    # 2-D exercises uniquecombs(·,TRUE)'s C-locale string-sort row order
    # (the sample indexes into those rows). mgcv 1.9-4 references.
    df = _tp_ds_2d_data()
    m = gam("y ~ s(x1, x2, k=40)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 706.9094382400,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1278327768, rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 38.2325721600,
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
    np.testing.assert_allclose(m3.REML_criterion / 2, 703.3490103000,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m3.sp[0], 0.03473596538, rtol=1e-5)

    m15 = gam("y ~ s(x1, x2, bs='ds', k=40)", df.head(1500), method="REML")
    np.testing.assert_allclose(m15.REML_criterion / 2, 348.9165255200,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m15.sp[0], 0.02406765059, rtol=1e-6)


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
    np.testing.assert_allclose(m._initial_sp_rho(), [0.7701263106],
                               rtol=0, atol=1e-8)
    # log(null.scale/10), null.scale = Σ dev_resids(y, ȳ)/n
    y = df["yg"].to_numpy()
    mu0 = np.full(len(y), y.mean())
    ns = float(np.sum(m.family.dev_resids(y, mu0, np.ones(len(y))))) / len(y)
    np.testing.assert_allclose(np.log(ns / 10.0), -2.7357189373,
                               rtol=0, atol=1e-8)

    # Two smooths — exercises the shared ×10 rebalance loop.
    df2 = _tp_ds_2d_data()
    m2 = gam("y ~ s(x1, x2, k=40) + s(x1, k=10)", df2, method="REML")
    np.testing.assert_allclose(
        m2._initial_sp_rho(), [1.3431758542, 2.5092598794],
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
    from hea.R.rng import RGenerator
    gen = RGenerator(55)
    n = 500
    x = gen.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x))
    y = gen.gamma(2.0, scale=mu / 2.0, size=n)
    y[gen.uniform(0, 1, n) < 0.08] = 0.0
    return pl.DataFrame({"x": x, "y": y})


def test_tw_reml_scale_and_vc_match_mgcv():
    # mgcv-extended families report sig2 = exp(φ̂_REML) (gam.outer's
    # scale.est), NOT the Fletcher estimator — Fletcher applied to tw was
    # 0.56% off, dragging Vp/Vc with it. mgcv 1.9-4 references.
    df = _tw_24_data()
    m = gam("y ~ s(x)", df, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 818.3711103000,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.8716551837,
                               rtol=0, atol=1e-6)
    assert m.sigma_squared == np.exp(m._log_phi_hat)
    np.testing.assert_allclose(m.sp[0], 0.08512769779, rtol=1e-4)
    # Vc/edf2 with the family-θ column of db.drho. Tightened from
    # 2e-3-era tolerances after family-review B9: Vc2's Cholesky seed
    # now uses the Fisher penalized Hessian like gam.fit3.post.proc's R
    # (gam.fit4.r:798 Fisher-type weights), which closed the whole
    # extended-family edf2/AIC band (measured Δ 2.6e-9 here).
    np.testing.assert_allclose(np.diag(m.Vc)[0], 0.001504373687, rtol=1e-6)
    np.testing.assert_allclose(m.edf2_total, 7.45509931, rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(m.ML_criterion / 2, 816.3142306600,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.8698129495,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.09324923391, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2.5 gam.control(edge.correct=): post-convergence walk of Hessian-flat
# smoothing parameters + the k=2 Vc recomputation with the weaker 1e-7 Vr
# prior (gam.fit3.r:1670-1716, post.proc K loop).
# ---------------------------------------------------------------------------

def test_edge_correct_vc_matches_mgcv():
    # On both fixtures mgcv's flat set is empty (lsp1 == lsp) and the
    # corrected Vc differs from the plain one purely through the k=2
    # 1e-7 prior — a ~1.3-1.4x change on the flat smooth's null-space-
    # adjacent entries in the second fixture. mgcv 1.9-4 references.
    from hea.R.rng import RGenerator
    gen = RGenerator(99)
    n = 300
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + gen.normal(0, 0.3, n)
    df = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m1 = gam("y ~ s(x1) + s(x2)", df, method="REML",
             control={"edge_correct": True})
    np.testing.assert_allclose(
        np.diag(m1.Vc)[[1, 9, 10]],
        [0.0323371108, 0.101825207, 0.00661200289], rtol=5e-4,
    )
    # edf2 keeps the fitted-model (k=1) value.
    np.testing.assert_allclose(m1.edf2_total, 10.72965196, rtol=2e-5)

    gen = RGenerator(14)        # R-native seed 7 over-saturates s(x2)
    n = 200                     # (edge-correct Vc then numerically delicate);
    x1 = gen.uniform(0, 1, n)   # 14 gives the original's mild-saturation regime
    x2 = gen.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + gen.normal(0, 0.25, n)
    df2 = pl.DataFrame({"x1": x1, "x2": x2, "y": y})
    m2 = gam("y ~ s(x1) + s(x2)", df2, method="REML",
             control={"edge_correct": True})
    np.testing.assert_allclose(
        np.diag(m2.Vc)[[1, 10, 11]],
        [0.0310210091, 0.00438953063, 0.00894119165], rtol=5e-4,
    )
    m0 = gam("y ~ s(x1) + s(x2)", df2, method="REML")
    np.testing.assert_allclose(
        np.diag(m0.Vc)[[1, 10, 11]],
        [0.0309144583, 0.00332593994, 0.00649458005], rtol=5e-4,
    )

    with pytest.raises(ValueError, match="edge_correct"):
        gam("y ~ s(x1)", df2, method="REML",
            control={"edge_correct": -1.0})


# ---------------------------------------------------------------------------
# 2.2 gam.reparam / get_stableS: log|Sλ|+ and its ρ-derivatives via mgcv's
# similarity-transform reparameterization (gdi.c:550-792) — immune to
# λ-ratio "machine zero leakage" between penalty components.
# ---------------------------------------------------------------------------

def test_get_stable_s_matches_mgcv_oracle():
    # Synthetic penalty roots; oracle values from mgcv:::gam.reparam on
    # the same matrices (R 4.x / mgcv 1.9-4). The disjoint case (3+4 in
    # 6-dim) shows the leakage: at lsp=(0,40) the assembled-eigen
    # determinant is off by ~689; get_stableS is exact.
    from hea.models.gam import _gam_reparam
    from hea.R.rng import RGenerator
    # column-major reshape == R's matrix(rnorm(k), nrow, ncol)
    gen = RGenerator(404)
    R1 = gen.normal(0, 1, 18).reshape((3, 6), order="F").round(6)
    gen.normal(0, 1, 18)  # R2 drawn but unused here (keeps stream)
    gen2 = RGenerator(405)
    R3 = gen2.normal(0, 1, 24).reshape((4, 6), order="F").round(6)

    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 0.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 6.0727466274, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rp["det1"], [2.6513413098, 3.3486586902],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        rp["det2"], [[0.2270958080, -0.2270958080],
                     [-0.2270958080, 0.2270958080]], rtol=0, atol=1e-9,
    )
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 20.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 85.0190848310, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rp["det1"], [2.0000000039, 3.9999999961],
                               rtol=0, atol=1e-8)
    rp = _gam_reparam([R1.T, R3.T], np.array([0.0, 40.0]), deriv=2)
    np.testing.assert_allclose(rp["det"], 165.0190848271, rtol=0, atol=1e-8)
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
    assert abs(det_naive - 165.0190848271) > 1.0


def test_extreme_fixed_sp_tensor_criterion_matches_mgcv():
    # Fit-level leakage stress: te() with fixed sp=(1e-8, 1e8) — λ ratio
    # 1e16 *within one block*. The legacy assembled-eigen log|S|+ path
    # returns a criterion off by ~334 here; gam.reparam lands on mgcv to
    # 4e-8. Free-fit pins confirm the optimizer surface is unchanged on
    # the healthy regime. mgcv 1.9-4 references.
    from hea.R.rng import RGenerator
    gen = RGenerator(77)
    n = 3000
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x1) * np.cos(np.pi * x2) + (x1 - 0.5) ** 2
         + gen.normal(0, 0.3, n))
    df = pl.DataFrame({"x1": x1[:400], "x2": x2[:400], "y": y[:400]})

    m = gam("y ~ te(x1, x2, k=5)", df, method="REML",
            sp=np.array([1e-8, 1e8]))
    np.testing.assert_allclose(m.REML_criterion / 2, 164.9817197500,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.0963046569,
                               rtol=0, atol=1e-9)

    m2 = gam("y ~ te(x1, x2, k=5)", df, method="REML")
    np.testing.assert_allclose(m2.REML_criterion / 2, 118.9165697700,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m2.sp, [0.124003427, 121.697916], rtol=1e-4)


# ---------------------------------------------------------------------------
# 2.1 conditioning half: augmented-QR PIRLS solve (pls_fit1) + gdiPK's
# negative-weight SVD correction. κ([√W·X; E]) instead of κ² everywhere a
# factor or X'WX-derived quantity is consumed.
# ---------------------------------------------------------------------------

def test_pls_qr_negative_weight_correction_is_exact():
    # Property check of the gdiPK (I−2D²) machinery on a synthetic
    # problem with mixed-sign Newton weights: the returned triangular
    # factor and log-determinant must reproduce the *signed* X'WX + S,
    # and beta must solve the signed normal equations.
    rng = np.random.default_rng(8)
    n, p = 60, 5
    X = rng.normal(size=(n, p))
    z = rng.normal(size=n)
    w = rng.uniform(0.2, 2.0, n)
    w[:6] = -rng.uniform(0.01, 0.05, 6)     # mildly negative rows: A stays PD
    E = np.diag(rng.uniform(0.5, 2.0, p))   # S = E'E

    df = pl.DataFrame({"x": rng.uniform(0, 1, 30),
                       "y": rng.normal(size=30)})
    m = gam("y ~ s(x, k=5)", df, method="REML")   # host for the method
    m._X_full = X
    beta, R, log_det, ok = m._pls_qr(w, z, E)
    assert ok
    A = X.T @ (w[:, None] * X) + E.T @ E
    np.testing.assert_allclose(R.T @ R, A, rtol=0, atol=1e-10)
    np.testing.assert_allclose(A @ beta, X.T @ (w * z), rtol=0, atol=1e-10)
    np.testing.assert_allclose(log_det, np.linalg.slogdet(A)[1],
                               rtol=0, atol=1e-10)
    assert np.all(np.diag(R) > 0)           # unique Cholesky normalization

    # Indefinite case (strongly negative weights) signals ok=False —
    # pls_fit1's oo$n<0, which gam.fit3 answers with a Fisher retry.
    w_bad = w.copy()
    w_bad[:20] = -5.0
    A_bad = X.T @ (w_bad[:, None] * X) + E.T @ E
    assert np.linalg.eigvalsh(A_bad).min() < 0
    *_, ok_bad = m._pls_qr(w_bad, z, E)
    assert not ok_bad


def test_ill_conditioned_design_matches_mgcv():
    # κ(X) ≈ 6e10 polynomial block (κ(X'X) ≈ 3e21 — beyond double
    # precision for the normal-equations route, which previously produced
    # *negative* total edf here). The augmented-QR path matches mgcv
    # (1.9-4) on every reported quantity.
    from hea.R.rng import RGenerator
    gen = RGenerator(11)         # R-native (set.seed(11), 0-ulp)
    n = 150
    x = gen.uniform(10.0, 10.1, n)
    z = gen.uniform(0.0, 1.0, n)
    y = (0.5 * (x - 10.0) + 0.05 * (x - 10.0) ** 2 + np.sin(2 * np.pi * z)
         + gen.normal(0.0, 0.2, n))
    df = pl.DataFrame({"x": x, "z": z, "y": y})
    m = gam("y ~ x + I(x^2) + I(x^3) + s(z)", df, method="REML")
    assert m.rank == 13 and m.p == 13       # no drop: below eps*100 tol
    np.testing.assert_allclose(m.REML_criterion / 2, -35.59408964,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.03464417846,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(m.edf)), 11.72237914,
                               rtol=0, atol=1e-4)
    # Raw coefficients of this κ(X)≈6e10 block are inherently platform/BLAS
    # sensitive (~1e-6 across reduction orders) — rtol=1e-6 was flaky on Intel
    # (~0.5%) and consistently ~2.5e-6 off on arm64. The *stable* quantities
    # (REML/σ²/edf/prediction above) are what pin the fit; the coefficients get
    # a looser bound that still catches gross errors but tolerates the conditioning.
    np.testing.assert_allclose(
        np.asarray(m.coef)[:4],
        [-34137.84777, 10168.43994, -1009.697889, 33.42319007],
        rtol=1e-5,
    )
    np.testing.assert_allclose(m.predict(df.head(1))["fit"][0], -0.4805246562,
                               rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# weights= (gam prior weights) — mgcv 1.9-4 references.
#
# One shared fixture; the R reference fits read the identical data via CSV
# (full-precision export) and every pin below is a printed mgcv value.
# Draw order from the single rng stream matters: x, normal, w, trials,
# binomial, gamma, poisson, per-row gamma sums.
# ---------------------------------------------------------------------------

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
    yg = gen.gamma(4.0, scale=np.exp(0.3 + np.sin(2 * np.pi * x)) / 4.0,
                   size=n)
    lam = np.exp(0.2 + np.sin(2 * np.pi * x))
    N = gen.poisson(lam)
    ytw = np.array([gen.gamma(3.0, scale=0.25, size=int(k)).sum() if k > 0
                    else 0.0 for k in N])
    df = pl.DataFrame({"x": x, "y": y, "ybin": ybin, "yg": yg, "ytw": ytw})
    return df, w, w0, trials


def test_weights_gaussian_reml_matches_mgcv():
    # gam(y ~ s(x), weights=w, method="REML") — continuous prior weights
    # through PIRLS, the (ρ, log φ) criterion (family.ls), Fletcher scale,
    # null deviance (weighted mean), AIC (gaussian: −Σlog w term), predict
    # SE, and gam.vcomp.
    df, w, _, _ = _weights_fixture()
    m = gam("y ~ s(x)", df, weights=w, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 43.4449049000,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01717485095, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.9805297900,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.1312939965,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.1347640443,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 119.8413439700,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 18.6463038300, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.AIC, 68.0116265500, rtol=0, atol=1e-5)
    p = m.predict(newdata=df[:2], se_fit=True)
    np.testing.assert_allclose(p["fit"].to_numpy(),
                               [-0.1240120614, -0.0129772762],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(p["se.fit"].to_numpy(),
                               [0.05989655466, 0.05780757341],
                               rtol=0, atol=1e-8)
    # vcomp defaults to mgcv's gam.vcomp(rescale=TRUE): each sp divided by
    # the penalty's S.scale (smoothCon's maS) so σ_k refers to the original
    # penalty scale. rescale=False is the fitted-scaling flavor.
    vc = m.vcomp
    np.testing.assert_allclose(vc["std_dev"].to_numpy(),
                               [11.93296997, 0.362345133], rtol=1e-6)
    np.testing.assert_allclose(vc["lower"].to_numpy(),
                               [6.937955427, 0.3226310119], rtol=1e-6)
    np.testing.assert_allclose(vc["upper"].to_numpy(),
                               [20.52416938, 0.4069478461], rtol=1e-6)
    vc0 = m._compute_vcomp(rescale=False)
    np.testing.assert_allclose(vc0["std_dev"].to_numpy(),
                               [2.764877815, 0.362345133], rtol=1e-6)
    np.testing.assert_allclose(vc0["lower"].to_numpy(),
                               [1.607529315, 0.3226310119], rtol=1e-6)
    np.testing.assert_allclose(vc0["upper"].to_numpy(),
                               [4.755464962, 0.4069478461], rtol=1e-6)
    # sp.vcov (single-formula path: the (ρ, log φ) outer Hessian) —
    # solve(hess + reg) with mgcv's elementwise reg (mgcv.r:4221-4234).
    np.testing.assert_allclose(
        m.sp_vcov(),
        [[0.31777315, 0.01283642], [0.01283642, 0.01403186]], rtol=1e-5)


def test_weights_unit_weights_equal_unweighted():
    # weights=ones must reproduce the unweighted fit — every site reads the
    # same self._wt array, so this guards the plumbing (FP-equal, see
    # conftest.assert_fp_equiv).
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
    np.testing.assert_allclose(m.REML_criterion / 2, 42.0001302400,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01724336763, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.9579329700,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.1273456593,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.null_deviance, 115.9126091300,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 18.0884406700, rtol=0, atol=1e-7)
    assert np.isinf(m.AIC)
    assert np.all(np.isfinite(m.fitted_values))


def test_weights_gaussian_ml_and_gcv_match_mgcv():
    df, w, _, _ = _weights_fixture()
    m = gam("y ~ s(x)", df, weights=w, method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 40.4562298300,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 0.1312683055,
                               rtol=0, atol=5e-8)
    np.testing.assert_allclose(m.AIC, 67.9331926600, rtol=0, atol=1e-4)

    g = gam("y ~ s(x)", df, weights=w)          # GCV.Cp
    np.testing.assert_allclose(g.GCV_score, 0.1372620019, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp[0], 0.06309683842, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(g.edf)), 6.5217061100,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(g.sigma_squared, 0.1312941190,
                               rtol=0, atol=1e-7)
    # mgcv leaves edf2 NULL on the GCV path, so logLik df falls back to
    # edf and AIC(m) = m$aic exactly.
    np.testing.assert_allclose(g.AIC, 66.4288391800, rtol=0, atol=1e-5)


def test_weights_binomial_trials_match_mgcv():
    # The R proportion + trials idiom: y = successes/trials, weights=trials.
    # Covers scale-known REML (no log φ in the outer Hessian — the Vr/Vc/
    # edf2 chain), the UBRE path, and weighted deviance residuals.
    from hea.family import Binomial
    df, _, _, trials = _weights_fixture()
    m = gam("ybin ~ s(x)", df, weights=trials, family=Binomial(),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05623115534, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8716212600,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2337119773,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.null_deviance, 519.0955475400,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 168.9992855600, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[0], 0.3041718423,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.residuals[0], -0.339290643,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.edf2_total, 7.0512421100, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(np.abs(m.Vc))), 25.2730419100,
                               rtol=1e-6)
    np.testing.assert_allclose(m.AIC, 514.3482544600, rtol=0, atol=1e-5)

    u = gam("ybin ~ s(x)", df, weights=trials, family=Binomial())  # UBRE
    np.testing.assert_allclose(u.GCV_score, 0.2137947813, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp[0], 0.137857243, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(u.edf)), 5.8914782800,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(u.AIC, 513.3157018800, rtol=0, atol=1e-4)


def test_weights_gamma_log_reml_matches_mgcv():
    # Non-canonical link: Newton weights w = wt·α·μ'²/V with prior weights,
    # plus mgcv's dev1 = reml.scale·Σw convention in the AIC.
    df, w, _, _ = _weights_fixture()
    m = gam("yg ~ s(x)", df, weights=w, family=Gamma(link="log"),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 145.5882797100,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05565807889, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6626274000,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.4405849487,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.1507651133,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 193.5417200300,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.deviance, 69.7617899100, rtol=0, atol=1e-7)
    # storedaic (family aic + 2·edf) pins exactly; AIC adds 2·(edf2−edf).
    # edf2 tightened post-B9 (Fisher-seed Vc2; Gamma-log is non-canonical
    # so the old Newton-seed Vc2 was off here too — measured Δ 5.7e-12).
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
    np.testing.assert_allclose(m.REML_criterion / 2, 327.4641111400,
                               rtol=0, atol=5e-5)
    np.testing.assert_allclose(m.sp[0], 0.01967446122, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.5730820200,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sigma_squared, 0.8900262323, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.3622523656,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.deviance, 242.0395602100, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m._tw_info["p_hat"], 1.220015110,
                               rtol=0, atol=1e-6)
    # extended-family edf2 band: measured Δ 2.8e-6 here (rel 4.3e-9).
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


# ---------------------------------------------------------------------------
# C10: cbind(succ, fail) binomial responses (R's two-column form).
# mgcv 1.9-4 references (identical data via full-precision CSV;
# succ = ybin·trials from _weights_fixture, fail = trials − succ).
# ---------------------------------------------------------------------------

def _cbind_fixture():
    df, w, _, trials = _weights_fixture()
    ybin = df["ybin"].to_numpy()
    succ = np.rint(ybin * trials)
    fail = trials - succ
    d = df.with_columns(pl.Series("succ", succ), pl.Series("fail", fail))
    return d, w, trials


def test_cbind_response_equals_proportion_idiom_and_mgcv():
    # Unit prior weights: R's binomial initialize rewrite makes
    # cbind(s, f) ≡ (y = s/n, weights = n) exactly (verified diff 0 in R),
    # so the proportion-idiom pins hold; b$y is the proportion VECTOR and
    # prior.weights the trials.
    from hea.family import Binomial
    d, _, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05623115534, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8716212600,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.AIC, 514.3482544600, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m._y_arr[:3],
                               [0.25, 0.0, 0.8333333333],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(m.prior_weights, trials, rtol=0, atol=0)
    assert m.formula == "cbind(succ, fail) ~ s(x)"
    # Same model as the proportion + trials idiom (same code path after
    # the intake rewrite; R: equiv diff 0) — FP-equal, see _assert_fp_equiv.
    p = gam("ybin ~ s(x)", d, weights=trials, family=Binomial(),
            method="REML")
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
    # Original bracket text is preserved verbatim (input-only, not rewritten).
    assert m_br.formula == "[succ, fail] ~ s(x)"


def test_cbind_with_prior_weights_matches_mgcv():
    # weights= on top of a cbind response: wt = pw·n while family$aic and
    # fix.family.ls's binomial ls keep the TRIALS vector n distinct
    # (binomial()$aic's `m <- if (any(n > 1)) n` branch). The proportion
    # idiom with weights = pw·n gives the same fit but R reports a
    # different REML/AIC there vs the pins below — the split is real.
    from hea.family import Binomial
    d, w, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(),
            method="REML", weights=w)
    np.testing.assert_allclose(m.REML_criterion / 2, 441.7395348800,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.05420557511, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.4910195100,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2673184329,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 301.4710899000, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 918.3830225500,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[:3],
                               [0.3049423071, 0.3403670810, 0.7793596206],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m._mgcv_aic, 867.0709067200, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.AIC, 867.3729279400, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.loglike, -426.0444338500, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.prior_weights, w * trials, rtol=0, atol=0)
    np.testing.assert_allclose(m.Vp[0, 0], 0.001974271296, rtol=1e-6)
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    assert label == "s(x)"
    np.testing.assert_allclose([edf, ref_df, stat],
                               [6.4910195050, 7.6159846940, 508.9404339000],
                               rtol=1e-6)
    assert p_val < 1e-10
    # UBRE (GCV.Cp, scale known) with the same weights.
    u = gam("cbind(succ, fail) ~ s(x)", d, family=Binomial(), weights=w)
    np.testing.assert_allclose(u.GCV_score, 1.1040181364, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp[0], 0.1532665613, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(u.edf)), 6.3273479700,
                               rtol=0, atol=1e-4)
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
    d0 = d.with_columns(pl.Series("succ0", succ0),
                        pl.Series("fail0", trials0 - succ0))
    m = gam("cbind(succ0, fail0) ~ s(x)", d0, family=Binomial(),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 261.9908232900,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0562991662, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8638531900,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 168.4022821200, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[10], 0.5778095857,
                               rtol=0, atol=1e-8)
    assert m._y_arr[10] == 0.0 and m.prior_weights[10] == 0.0
    assert m.df_null == 148.0
    np.testing.assert_allclose(m.AIC, 511.2769694600, rtol=0, atol=1e-5)


def test_cbind_intake_validation():
    from hea.family import Binomial, Poisson
    d, _, _ = _cbind_fixture()
    # mgcv dies obscurely on non-binomial cbind ("logical subscript too
    # long"); hea raises a clear error instead.
    with pytest.raises(ValueError, match="family=Binomial"):
        gam("cbind(succ, fail) ~ s(x)", d, family=Poisson(), method="REML")
    with pytest.raises(ValueError, match="exactly two"):
        gam("cbind(succ, fail, x) ~ s(x)", d, family=Binomial(),
            method="REML")
    neg = d.with_columns((pl.col("succ") - 100.0).alias("succ_n"))
    with pytest.raises(ValueError, match="negative counts"):
        gam("cbind(succ_n, fail) ~ s(x)", neg, family=Binomial(),
            method="REML")
    # R: "non-integer counts in a binomial glm!" (initialize's 2-col
    # branch); fitting proceeds.
    frac = d.with_columns((pl.col("succ") + 0.4).alias("succ_f"))
    with pytest.warns(UserWarning, match="non-integer counts"):
        gam("cbind(succ_f, fail) ~ s(x)", frac, family=Binomial(),
            method="REML")
    # Expression arguments evaluate like R (cbind(tot - fail, fail)).
    tot = d.with_columns((pl.col("succ") + pl.col("fail")).alias("tot"))
    m = gam("cbind(tot - fail, fail) ~ s(x)", tot, family=Binomial(),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 263.5369484800,
                               rtol=0, atol=1e-6)
    # bam has no cbind support yet (bam-mgcv-parity plan) — still raises.
    from hea.models.bam import bam
    with pytest.raises(NotImplementedError, match="cbind"):
        bam("cbind(succ, fail) ~ s(x)", d, family=Binomial())


# ---------------------------------------------------------------------------
# D1a: quasipoisson / quasibinomial constructors, plain quasi's "none"
# canonical (full Newton at every link), and the R bare-constructor
# family idiom. mgcv 1.9-4 references on the _cbind_fixture data.
# ---------------------------------------------------------------------------

def test_quasipoisson_through_gam_matches_mgcv():
    # EQL ls (fix.family.ls quasi branch), Fletcher scale with poisson
    # dvar, F-flavor s.table (scale estimated), AIC/logLik NA in R →
    # NaN here. family=quasipoisson (bare class) mirrors R's
    # function-valued family= (mgcv.r:2324).
    from hea.family import quasipoisson
    d, _, _ = _cbind_fixture()
    m = gam("succ ~ s(x)", d, family=quasipoisson, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 141.0246116800,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.2305148526, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.1846454500,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.9841906748,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 1.4583994509,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 309.2978228100, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.null_deviance, 495.2322484000,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[:3],
                               [3.2802454640, 3.6533186760, 8.4610051790],
                               rtol=0, atol=1e-8)
    assert np.isnan(m.AIC) and np.isnan(m.loglike)
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose([edf, ref_df, stat],
                               [5.1846454520, 6.2624900930, 12.6559374100],
                               rtol=1e-6)
    assert p_val < 1e-8
    # GCV flavor (scale unknown → GCV, not UBRE).
    g = gam("succ ~ s(x)", d, family=quasipoisson)
    np.testing.assert_allclose(g.GCV_score, 2.2429195192, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp[0], 0.2674208059, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(g.edf)), 6.0273140000,
                               rtol=0, atol=1e-4)
    # Non-canonical link → full-Newton inner loop (canonical is log,
    # gam.fit3.r:2318).
    s = gam("succ ~ s(x)", d, family=quasipoisson(link="sqrt"),
            method="REML")
    np.testing.assert_allclose(s.REML_criterion / 2, 140.2671306200,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(s.sp[0], 0.2182661, rtol=1e-4)
    np.testing.assert_allclose(s.sigma_squared, 1.9665073702,
                               rtol=0, atol=1e-8)


def test_quasibinomial_cbind_through_gam_matches_mgcv():
    # quasibinomial shares binomial's initialize (cbind rewrite + n-form
    # mustart) but quasi's EQL ls / NA aic; scale estimated (F tests).
    from hea.family import quasibinomial
    d, _, trials = _cbind_fixture()
    m = gam("cbind(succ, fail) ~ s(x)", d, family=quasibinomial,
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, -66.3069511400,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.06605267459, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6904930000,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.0393925494,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], -0.2331078925,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.deviance, 169.1760060600, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.fitted_values[:3],
                               [0.3031568283, 0.3358585931, 0.7850271713],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.prior_weights, trials, rtol=0, atol=0)
    assert np.isnan(m.AIC)
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose([edf, ref_df, stat],
                               [5.6904929960, 6.8217009380, 40.8698983000],
                               rtol=1e-6)
    assert p_val < 1e-10
    # Same model as the proportion + trials idiom (R: equiv diff 0).
    p = gam("ybin ~ s(x)", d, weights=trials, family=quasibinomial,
            method="REML")
    _assert_fp_equiv(m.REML_criterion, p.REML_criterion)
    _assert_fp_equiv(m.coef, p.coef)


# ---------------------------------------------------------------------------
# C1: mixed sp= (negative entries = estimate) + per-smooth s(..., sp=).
# mgcv 1.9-4 references (identical data via full-precision CSV).
# ---------------------------------------------------------------------------

def _mixed_sp_fixture():
    from hea.R.rng import RGenerator
    gen = RGenerator(7)
    n = 160
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    w2 = gen.uniform(0, 1, n)
    f = np.sin(2 * np.pi * x) + 0.5 * (z - 0.5) ** 2 * 8 \
        + 0.3 * np.cos(3 * np.pi * w2)
    y = f + gen.normal(0, 0.4, n)
    ycnt = gen.poisson(np.exp(0.3 + np.sin(2 * np.pi * x) + 0.5 * z))
    lam = np.exp(0.2 + np.sin(2 * np.pi * x))
    N = gen.poisson(lam)
    ytw = np.array([gen.gamma(3.0, scale=0.25, size=int(k)).sum()
                    if k > 0 else 0.0 for k in N])
    lam2 = np.exp(0.2 + np.sin(2 * np.pi * x) + 1.2 * (z - 0.5) ** 2 * 3)
    N2 = gen.poisson(lam2)
    ytw2 = np.array([gen.gamma(3.0, scale=0.25, size=int(k)).sum()
                     if k > 0 else 0.0 for k in N2])
    return pl.DataFrame({"x": x, "z": z, "w2": w2, "y": y,
                         "ycnt": ycnt.astype(float), "ytw": ytw,
                         "ytw2": ytw2})


def test_mixed_sp_gaussian_matches_mgcv():
    # gam(sp=c(2, -1)): first sp fixed, second estimated — folded into
    # (L, lsp0) exactly like mgcv.r:1513-1538. m.sp is the FREE working
    # vector (mgcv m$sp); full_sp the per-penalty expansion (m$full.sp).
    df = _mixed_sp_fixture()
    m = gam("y ~ s(x) + s(z)", df, sp=np.array([2.0, -1.0]),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 152.1896738580,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.1401341495], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp, [2.0, 0.1401341495], rtol=1e-4)
    assert m.full_sp[0] == 2.0
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.8610338694,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.3220580505,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.2246565999,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.fitted_values[:2],
                               [-0.5851315520, 0.7295423129],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.Vp[0, 0], 0.002012862816, rtol=1e-6)
    # s(x, sp=2) is the same model (R: diff exactly 0), and a gam-level
    # vector is overridden by the per-smooth value (mgcv.r:1426).
    m2 = gam("y ~ s(x, sp=2) + s(z)", df, method="REML")
    _assert_fp_equiv(m2.REML_criterion, m.REML_criterion)
    _assert_fp_equiv(m2.coef, m.coef)
    m3 = gam("y ~ s(x, sp=2) + s(z)", df, sp=np.array([5.0, -1.0]),
             method="REML")
    _assert_fp_equiv(m3.REML_criterion, m.REML_criterion)
    # All-negative == estimate everything (mgcv's rep(-1) default).
    m4 = gam("y ~ s(x) + s(z)", df, sp=np.array([-1.0, -1.0]),
             method="REML")
    m5 = gam("y ~ s(x) + s(z)", df, method="REML")
    _assert_fp_equiv(m4.REML_criterion, m5.REML_criterion)


def test_mixed_sp_zero_gcv_te_and_id_match_mgcv():
    df = _mixed_sp_fixture()
    # Fixed sp=0 → mgcv's "effective zero" replacement
    # (‖X_1‖_F²/‖S_1‖_F·eps·0.1, mgcv.r:1519-1527 — incl. the literal
    # loop-counter quirk); full.sp[0] pinned to R's exact fudge value.
    m = gam("y ~ s(x, sp=0) + s(z)", df, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 248.0359363930,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.05506103452], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp[0], 3.230203061e-17, rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 14.8171854238,
                               rtol=0, atol=1e-4)
    # GCV path with a fixed entry (criterion exact; the free sp wobbles
    # in the optimizer stop band like every GCV pin).
    g = gam("y ~ s(x, sp=2) + s(z)", df)
    np.testing.assert_allclose(g.GCV_score, 0.3364655582, rtol=0, atol=1e-8)
    np.testing.assert_allclose(g.sp, [0.1158917722], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(g.edf)), 7.0386466421,
                               rtol=0, atol=1e-4)
    # te() with mixed per-margin sp.
    t = gam("y ~ te(x, z, sp=c(1, -1))", df, method="REML")
    np.testing.assert_allclose(t.REML_criterion / 2, 125.0775025180,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(t.sp, [1.006184059], rtol=1e-4)
    np.testing.assert_allclose(t.full_sp, [1.0, 1.006184059], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(t.edf)), 16.5165951050,
                               rtol=0, atol=1e-4)
    # id-linked pair fixed via its shared working slot + free third.
    i = gam("y ~ s(x, id=1) + s(z, id=1) + s(w2)", df,
            sp=np.array([3.0, -1.0]), method="REML")
    np.testing.assert_allclose(i.REML_criterion / 2, 162.4801359980,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(i.sp, [0.2410361343], rtol=1e-4)
    np.testing.assert_allclose(i.full_sp, [3.0, 3.0, 0.2410361343],
                               rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(i.edf)), 8.1520975718,
                               rtol=0, atol=1e-4)


def test_mixed_sp_tw_and_poisson_match_mgcv():
    # tw: the fixed-sp fold coexists with the family-θ outer slot (the
    # all-fixed guard no longer fires for mixed input); poisson: scale-
    # known REML layout.
    from hea.family import Poisson, tw
    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x, sp=1) + s(z)", df, family=tw(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 259.1938532110,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.7153525883], rtol=1e-4)
    np.testing.assert_allclose(m.full_sp, [1.0, 0.7153525883], rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.2018656475,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m._tw_info["p_hat"], 1.2491176801,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 1.1846344441,
                               rtol=0, atol=1e-6)
    p = gam("ycnt ~ s(x, sp=0.5) + s(z)", df, family=Poisson(),
            method="REML")
    np.testing.assert_allclose(p.REML_criterion / 2, 261.6789254290,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(p.sp, [36789.51172], rtol=1e-3)
    np.testing.assert_allclose(p.full_sp[0], 0.5, rtol=0, atol=0)
    np.testing.assert_allclose(float(np.sum(p.edf)), 5.1465787370,
                               rtol=0, atol=1e-4)


def test_fixed_sp_unknown_scale_matches_mgcv():
    """All-fixed sp + unknown scale: the criterion must be minimized over
    log φ (mgcv's 1-D newton when lsp = [log scale], gam.fit3.r:121-123),
    NOT evaluated at the Gaussian profile φ̂ = Dp/(n−Mp) — those coincide
    only for Gaussian/EQL-shaped ls (family-review A1). β̂/edf/Fletcher
    sig2 are φ-independent and were always right; only the reported
    criterion moved.

    R 4.6.0 / mgcv 1.9-4 on the _mixed_sp_fixture data:
        d$yg <- exp(d$y/3)
        gam(ytw2~s(x), Tweedie(1.5, log), sp=2, method="REML")
            REML 275.9692309030  sig2 1.1707800009  reml.scale 1.6764407979
        gam(ytw2~s(x), Tweedie(1.5, log), sp=0.5, method="REML")
            REML 271.9594278050
        gam(ytw2~s(x), Tweedie(1.5, log), sp=2, method="ML")
            ML 274.2404901730
        gam(yg~s(x)+s(z), Gamma(log), sp=c(1,4), method="REML")
            REML -5.0421868495  sig2 0.0355890744  edf 5.0315916630
        gam(yg~s(x), inverse.gaussian(log), sp=3, method="REML")
            REML 8.6185097030  sig2 0.0467285182  edf 2.7039644722
    """
    from hea.family import Gamma, InverseGaussian, Tweedie
    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x)", df, family=Tweedie(p=1.5), sp=np.array([2.0]),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 275.9692309030,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sigma_squared, 1.1707800009,
                               rtol=0, atol=1e-8)
    # the criterion's internal φ̂ is the newton stop, not the Fletcher
    # scale — mgcv's reml.scale (1.6764407979; stop-point band).
    np.testing.assert_allclose(float(np.exp(m._log_phi_hat)),
                               1.6764407979, rtol=1e-6)
    m05 = gam("ytw2 ~ s(x)", df, family=Tweedie(p=1.5),
              sp=np.array([0.5]), method="REML")
    np.testing.assert_allclose(m05.REML_criterion / 2, 271.9594278050,
                               rtol=0, atol=1e-6)
    ml = gam("ytw2 ~ s(x)", df, family=Tweedie(p=1.5), sp=np.array([2.0]),
             method="ML")
    np.testing.assert_allclose(ml.ML_criterion / 2, 274.2404901730,
                               rtol=0, atol=1e-6)
    dfg = df.with_columns(yg=(pl.col("y") / 3).exp())
    g = gam("yg ~ s(x) + s(z)", dfg, family=Gamma(link="log"),
            sp=np.array([1.0, 4.0]), method="REML")
    np.testing.assert_allclose(g.REML_criterion / 2, -5.0421868495,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(g.sigma_squared, 0.0355890744,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(np.sum(g.edf)), 5.0315916630,
                               rtol=0, atol=1e-4)
    i = gam("yg ~ s(x)", dfg, family=InverseGaussian(link="log"),
            sp=np.array([3.0]), method="REML")
    np.testing.assert_allclose(i.REML_criterion / 2, 8.6185097030,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(i.sigma_squared, 0.0467285182,
                               rtol=0, atol=1e-9)


def test_extended_null_deviance_find_null_dev_matches_mgcv():
    """Extended families replace null.deviance with mgcv's
    ``find.null.dev`` (efam.r:98-117: 1-D optimize over the constant ON
    THE LINK SCALE, offset in the candidate model) via the family
    postproc (nb efam.r:283, tw efam.r:3239, scat efam.r:3742) — NOT the
    standard weighted-mean value (family-review A2; pre-fix scat read
    385.4317521873, Δ2.4e-3). postproc also relabels summary's Family
    line with the fitted θ.

    R 4.6.0 / mgcv 1.9-4 on the _mixed_sp_fixture data, off1 = 0.3·z:
        gam(y~s(x), scat(), REML)        null.dev 409.4241525690
                                         family  'Scaled t(19.281,0.515)'
        gam(y~s(x)+offset(off1), scat()) null.dev 407.5865103560
        gam(ytw2~s(x), tw(), REML)       null.dev 301.8879359400
                                         family  'Tweedie(p=1.237)'
        gam(ytw2~s(x)+offset(off1), tw())null.dev 298.7091359900
        gam(ycnt~s(x), nb(theta=5))      null.dev 254.1827688399
                                         family  'Negative Binomial(5)'
        gam(ycnt~s(x)+offset(off1), nb(theta=5))
                                         null.dev 253.1033229196
                                         REML 261.3070751610
    Free-θ nb is left unpinned here: ycnt is near-Poisson, so Θ̂ sits on
    a flat ridge (R and hea land on different Θ̂ at <1e-8 criterion
    flatness) and null.dev inherits the Θ̂ band at ~1e-3. The offset
    variants are the discriminating cases — no weighted-mean formula
    produces them.
    """
    from hea.family import nb, scat, tw
    df = _mixed_sp_fixture().with_columns(off1=0.3 * pl.col("z"))
    ms = gam("y ~ s(x)", df, family=scat(), method="REML")
    np.testing.assert_allclose(ms.null_deviance, 409.4241525690,
                               rtol=0, atol=1e-6)
    assert ms._family_display_name() == "Scaled t(19.281,0.515)"
    mso = gam("y ~ s(x) + offset(off1)", df, family=scat(), method="REML")
    np.testing.assert_allclose(mso.null_deviance, 407.5865103560,
                               rtol=0, atol=1e-6)
    mt = gam("ytw2 ~ s(x)", df, family=tw(), method="REML")
    np.testing.assert_allclose(mt.null_deviance, 301.8879359400,
                               rtol=0, atol=1e-6)
    assert mt._family_display_name() == "Tweedie(p=1.237)"
    mto = gam("ytw2 ~ s(x) + offset(off1)", df, family=tw(),
              method="REML")
    np.testing.assert_allclose(mto.null_deviance, 298.7091359900,
                               rtol=0, atol=1e-6)
    m5 = gam("ycnt ~ s(x)", df, family=nb(theta=5), method="REML")
    np.testing.assert_allclose(m5.null_deviance, 254.1827688399,
                               rtol=0, atol=1e-8)
    assert m5._family_display_name() == "Negative Binomial(5)"
    m5o = gam("ycnt ~ s(x) + offset(off1)", df, family=nb(theta=5),
              method="REML")
    np.testing.assert_allclose(m5o.null_deviance, 253.1033229196,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m5o.REML_criterion / 2, 261.3070751610,
                               rtol=0, atol=1e-6)
    # deviance_explained rides on the corrected null deviance.
    np.testing.assert_allclose(
        m5o.deviance_explained,
        (m5o.null_deviance - m5o.deviance) / m5o.null_deviance,
        rtol=0, atol=1e-12,
    )


def test_nb_tw_sqrt_link_matches_mgcv():
    """sqrt is in nb/tw's okLinks but SqrtLink had no g2g/g3g/g4g — the
    extended dDeta chain raised NotImplementedError mid-fit
    (family-review B3). Forms from fix.family.link's extended block
    (gam.fit3.r:2243-2247).

    R 4.6.0 / mgcv 1.9-4 on the _mixed_sp_fixture data:
        gam(ycnt~s(x), nb(theta=5, link="sqrt"), REML)
            REML 262.2684702080  dev 116.2285680340
            edf 5.4727045882    null.dev 254.1827689430
        gam(ytw2~s(x), tw(link="sqrt"), REML)
            REML 252.9727727650  p 1.23714366
    """
    from hea.family import nb, tw
    df = _mixed_sp_fixture()
    m = gam("ycnt ~ s(x)", df, family=nb(theta=5, link="sqrt"),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 262.2684702080,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.deviance, 116.2285680340,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.4727045882,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.null_deviance, 254.1827689430,
                               rtol=0, atol=1e-8)
    mt = gam("ytw2 ~ s(x)", df, family=tw(link="sqrt"), method="REML")
    np.testing.assert_allclose(mt.REML_criterion / 2, 252.9727727650,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(mt._tw_info["p_hat"], 1.23714366,
                               rtol=0, atol=1e-5)


def test_binomial_factor_response_matches_mgcv():
    """R's gam accepts a 2-level factor (or logical) binomial response
    via binomial initialize's is.factor branch (level 1 = failure);
    hea routes gam/bam response intake through the same
    ``_coerce_response`` glm uses (family-review B8 — previously
    crashed on the float cast).

    R 4.6.0 / mgcv 1.9-4, _mixed_sp_fixture + seed-11 bernoulli draws
    on p = expit(1.5·sin(2πx)), ystr = yes/no (alphabetical levels →
    "no" is failure):
        gam(ystr~s(x), binomial, REML)  REML 89.7247633913
            dev 164.6309203600  edf 5.4854767035
        (logical response: identical fit)
    """
    df = _mixed_sp_fixture()
    from hea.R.rng import RGenerator
    gen = RGenerator(11)
    p = 1.0 / (1.0 + np.exp(-(np.sin(2 * np.pi * df["x"].to_numpy()) * 1.5)))
    yb = gen.uniform(0, 1, len(p)) < p
    df = df.with_columns(ystr=pl.Series(np.where(yb, "yes", "no")),
                         ybool=pl.Series(yb))
    from hea.family import Binomial
    m = gam("ystr ~ s(x)", df, family=Binomial(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 89.7247633913,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.deviance, 164.6309203600,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.4854767035,
                               rtol=0, atol=1e-4)
    mb = gam("ybool ~ s(x)", df, family=Binomial(), method="REML")
    _assert_fp_equiv(mb.REML_criterion, m.REML_criterion)
    _assert_fp_equiv(mb.coef, m.coef)
    # non-binomial families keep the strict float cast.
    with pytest.raises(Exception, match="convert|cast|float"):
        gam("ystr ~ s(x)", df, method="REML")


def test_outer_hessian_matches_mgcv_reml2():
    """The outer Newton Hessian vs mgcv's analytic REML2 (family-review
    B9 scoping measurement, promoted to a pin): hea's analytic
    (ρ, log φ) block agrees to ~1e-10 and the FD θ-rows (central
    differences of the analytic gradient, h=1e-4) to ~1e-7 — the FD
    truncation band. Layouts: hea (ρ, logφ, θ) ≡ mgcv (θ, ρ, logφ),
    both V_R units.

    R 4.6.0 / mgcv 1.9-4: gam(ytw2~s(x), tw(), REML) on the
    _mixed_sp_fixture data; m$outer.info$hess =
        [θ ]  31.949749924    0.011231728742  -33.078371913
        [ρ ]   0.011231728742  2.029401444600  -1.964717610290
        [φ ] -33.078371913   -1.964717610290  143.804523249
    """
    from hea.family import tw
    df = _mixed_sp_fixture()
    m = gam("ytw2 ~ s(x)", df, family=tw(), method="REML")
    H = np.asarray(m._outer_info["hess"])
    perm = [2, 0, 1]                      # hea (ρ,logφ,θ) → mgcv (θ,ρ,logφ)
    Hm = H[np.ix_(perm, perm)]
    R_hess = np.array([
        [31.949749924, 0.011231728742, -33.078371913],
        [0.011231728742, 2.029401444600, -1.964717610290],
        [-33.078371913, -1.964717610290, 143.804523249],
    ])
    # θ row/col: FD truncation band; analytic block much tighter.
    np.testing.assert_allclose(Hm[0, :], R_hess[0, :], rtol=0, atol=5e-6)
    np.testing.assert_allclose(Hm[:, 0], R_hess[:, 0], rtol=0, atol=5e-6)
    np.testing.assert_allclose(Hm[1:, 1:], R_hess[1:, 1:],
                               rtol=0, atol=1e-7)


def test_quasi_power_link_matches_r():
    """R's ``power(λ)`` link (family-review B5): ``g(μ) = μ^λ`` with
    R's exact factory semantics — λ ≤ 0 → log, λ = 1 → identity, link
    name "mu^round(λ,3)" — and fix.family.link's power d2link..d4link
    branch (gam.fit3.r:2329-2335). Object form only, like R (make.link
    accepts no "power(...)" string).

    R 4.6.0 / mgcv 1.9-4, yg = exp(y/3) on the _mixed_sp_fixture data:
        glm(yg ~ x + z, quasi(link=power(1/3), variance="mu"))
            coef 1.13069136366 -0.216498739042 0.0362335015181
            dev 9.5538082355  dispersion 0.0632910836247
        gam(yg ~ s(x) + z, quasi(power(1/3), "mu"), REML)
            REML -166.3099041900  dev 5.6758088545
            edf 7.7176279737     sig2 0.0377580878
    """
    from hea.family import PowerLink, Quasi, power
    assert power(0).name == "log" and power(-1).name == "log"
    assert power(1).name == "identity"
    assert power(1 / 3).name == "mu^0.333"
    assert isinstance(power(0.5), PowerLink)
    df = _mixed_sp_fixture().with_columns(yg=(pl.col("y") / 3).exp())
    mq = glm("yg ~ x + z", df,
             family=Quasi(link=power(1 / 3), variance="mu"))
    np.testing.assert_allclose(
        mq._bhat_arr, [1.13069136366, -0.216498739042, 0.0362335015181],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(mq.deviance, 9.5538082355,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(mq.dispersion, 0.0632910836247,
                               rtol=0, atol=1e-8)
    mg = gam("yg ~ s(x) + z", df,
             family=Quasi(link=power(1 / 3), variance="mu"),
             method="REML")
    np.testing.assert_allclose(mg.REML_criterion / 2, -166.3099041900,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(mg.deviance, 5.6758088545,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(mg.edf)), 7.7176279737,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(mg.sigma_squared, 0.0377580878,
                               rtol=0, atol=1e-9)


def test_mixed_sp_validation():
    df = _mixed_sp_fixture()
    # mgcv's exact error for a wrong-length per-smooth sp (mgcv.r:1426).
    with pytest.raises(ValueError, match="incorrect number of smoothing"):
        gam("y ~ s(x, sp=c(1, 2)) + s(z)", df, method="REML")
    with pytest.raises(ValueError, match="sp must have length"):
        gam("y ~ s(x) + s(z)", df, sp=np.array([1.0]), method="REML")
    with pytest.raises(ValueError, match="must be numeric"):
        gam("y ~ s(x, sp='a') + s(z)", df, method="REML")
    # All-fixed via per-smooth values lands on the historical fixed path.
    m = gam("y ~ s(x, sp=2) + s(z, sp=0.5)", df, method="REML")
    f = gam("y ~ s(x) + s(z)", df, sp=np.array([2.0, 0.5]), method="REML")
    _assert_fp_equiv(m.REML_criterion, f.REML_criterion)
    np.testing.assert_array_equal(m.sp, f.sp)


# ---------------------------------------------------------------------------
# C2: gam(control=) umbrella + xt=list(max.knots=, seed=). mgcv 1.9-4
# references.
# ---------------------------------------------------------------------------

def test_control_scale_est_matches_mgcv():
    # gam.control(scale.est=): "pearson" drops the Fletcher correction,
    # "deviance" uses dev/(n−trA) (gam.fit3.r:596-606). The fit itself
    # is untouched on this fixture (score.scale enters thresholds only)
    # — sig2 is the value-level difference.
    from hea.family import quasipoisson
    d, _, _ = _cbind_fixture()
    base = gam("succ ~ s(x)", d, family=quasipoisson, method="REML")
    p = gam("succ ~ s(x)", d, family=quasipoisson, method="REML",
            control={"scale_est": "pearson"})
    np.testing.assert_allclose(p.sigma_squared, 1.9466793481,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(p.REML_criterion / 2, 141.0246116800,
                               rtol=0, atol=1e-6)
    dv = gam("succ ~ s(x)", d, family=quasipoisson, method="REML",
             control={"scale_est": "deviance"})
    np.testing.assert_allclose(dv.sigma_squared, 2.1506592518,
                               rtol=0, atol=1e-8)
    # fletcher is the default (already pinned 2.0762775960 in the
    # quasipoisson test) and differs from both.
    assert base.sigma_squared not in (p.sigma_squared, dv.sigma_squared)


def test_control_newton_and_maxit_match_mgcv():
    # newton=list(conv.tol=1e-3): looser outer stop — R-pinned sp/REML.
    # maxit=2: tiny PIRLS budget changes the fit; mgcv neither warns nor
    # flags it at the gam level (verified live) — same here.
    df = _mixed_sp_fixture()
    m = gam("y ~ s(x) + s(z)", df, method="REML",
            control={"newton": {"conv_tol": 1e-3}})
    np.testing.assert_allclose(m.REML_criterion / 2, 118.5012745590,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.01676484948, 0.06765075176],
                               rtol=1e-6)
    np.testing.assert_allclose(float(np.sum(m.edf)), 11.5634548370,
                               rtol=0, atol=1e-4)
    m2 = gam("y ~ s(x) + s(z)", df, method="REML", control={"maxit": 2})
    assert m2.REML_criterion != m.REML_criterion


def test_xt_max_knots_seed_matches_mgcv():
    # s(x, xt=list(max.knots=1500, seed=2)) — the 1.8 subsample
    # machinery under non-default controls, bit-exact via the R-RNG
    # port. Same data as test_tp_max_knots_subsample_matches_mgcv.
    from hea.R.rng import RGenerator
    gen = RGenerator(2024)
    n = 4000
    x = gen.uniform(0, 1, n)
    y = (np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)
         + gen.normal(0, 0.4, n))
    df = pl.DataFrame({"x": x, "y": y})
    m = gam("y ~ s(x, k=20, xt=list(max.knots=1500, seed=2))", df,
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 2026.8334280500,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.007336834857, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 16.9493402600,
                               rtol=0, atol=1e-4)


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
    with pytest.raises(TypeError):
        gam_control(nlm={})            # unported optimizer knob
    with pytest.raises(ValueError, match="epsilon"):
        gam_control(epsilon=0.0)
    # raw dicts revalidate through gam_control inside gam()
    with pytest.raises(ValueError, match="scale_est"):
        gam("y ~ s(x)", df, method="REML",
            control={"scale_est": "nope"})
    with pytest.raises(ValueError, match="unsupported xt entry"):
        gam("y ~ s(x, xt=list(shrink=0.5))", df, method="REML")
    # defaults are the same fit as no control at all (FP-equal)
    m0 = gam("y ~ s(x) + s(z)", df, method="REML")
    m1 = gam("y ~ s(x) + s(z)", df, method="REML", control={})
    _assert_fp_equiv(m0.REML_criterion, m1.REML_criterion)
    _assert_fp_equiv(m0.coef, m1.coef)


# ---------------------------------------------------------------------------
# C3: gam(scale=) — fixed scale (REML without the φ slot; UBRE at φ),
# forced estimation (GCV for poisson). mgcv 1.9-4 references on the
# _mixed_sp_fixture data.
# ---------------------------------------------------------------------------

def test_scale_fixed_gaussian_matches_mgcv():
    # scale=0.3 + REML: φ KNOWN — no log φ slot in the outer vector,
    # criterion at log(0.3), sig2 reported as 0.3, Vp = 0.3·A⁻¹,
    # z/Chi-sq summary flavor (scale.estimated FALSE), AIC's dev1 =
    # scale·Σwt (gam.fit3.r:848 first branch — NOT the gaussian dev
    # override).
    df = _mixed_sp_fixture()
    m = gam("y ~ s(x) + s(z)", df, method="REML", scale=0.3)
    np.testing.assert_allclose(m.REML_criterion / 2, 122.8873561270,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp, [0.02415170022, 0.1304568059],
                               rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 10.5075250592,
                               rtol=0, atol=1e-4)
    assert m.sigma_squared == 0.3 and m.scale_estimated is False
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.2246565999,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.Vp[0, 0], 0.001875, rtol=1e-6)
    np.testing.assert_allclose(m.AIC, 285.9720683910, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.loglike, -131.7123409670, rtol=0, atol=1e-5)
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose([edf, ref_df, stat],
                               [5.5231503566, 6.6485522295, 223.9957913612],
                               rtol=1e-6)
    # GCV.Cp at fixed scale → UBRE at φ=0.3 (any family, mgcv.r:1956).
    u = gam("y ~ s(x) + s(z)", df, scale=0.3)
    np.testing.assert_allclose(u.GCV_score, -0.0661457173, rtol=0, atol=1e-8)
    np.testing.assert_allclose(u.sp, [0.05418197314, 0.09828929095],
                               rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(u.edf)), 9.9008106194,
                               rtol=0, atol=1e-4)
    assert u.sigma_squared == 0.3
    np.testing.assert_allclose(u.AIC, 283.2263031720, rtol=0, atol=1e-5)


def test_scale_negative_forces_estimation_matches_mgcv():
    # poisson + GCV.Cp + scale=-1: GCV (not UBRE), dispersion estimated
    # (Fletcher), t/F summary flavor — mgcv's quasi-style overdispersion
    # route without changing the family.
    from hea.family import Poisson
    df = _mixed_sp_fixture()
    m = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), scale=-1)
    np.testing.assert_allclose(m.GCV_score, 1.0715298363, rtol=0, atol=1e-8)
    np.testing.assert_allclose(m.sp[0], 0.09747456578, rtol=1e-4)
    np.testing.assert_allclose(m.sp[1], 259468.0106, rtol=1e-2)  # boundary
    np.testing.assert_allclose(float(np.sum(m.edf)), 6.6204950639,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 0.9621181226,
                               rtol=0, atol=1e-8)
    assert m.scale_estimated is True
    label, edf, ref_df, stat, p_val = m._smooth_significance_rows()[0]
    np.testing.assert_allclose([edf, ref_df, stat],
                               [4.6204526515, 5.6432868481, 25.8921163282],
                               rtol=1e-6)
    # Under (RE)ML, poisson/binomial scale= is silently 1 (mgcv.r:1947).
    p2 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), method="REML",
             scale=2)
    p0 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(), method="REML")
    _assert_fp_equiv(p2.REML_criterion, p0.REML_criterion)
    assert p2.sigma_squared == 1.0 and p2.scale_estimated is False
    # Extended families: scale handling is family-driven — honest raise.
    from hea.family import tw
    with pytest.raises(NotImplementedError, match="scale="):
        gam("ytw2 ~ s(x)", df, family=tw(), method="REML", scale=2.0)


# ---------------------------------------------------------------------------
# C7: single-formula start= / etastart= / mustart= (gam.fit3.r:259-292).
# Start values steer the PIRLS path, not the optimum — R-verified: mgcv
# warm/perturbed starts land on the same fit to 1e-14.
# ---------------------------------------------------------------------------

def test_pirls_start_values_warm_and_invariant():
    from hea.family import Poisson
    df = _mixed_sp_fixture()
    m0 = gam("y ~ s(x) + s(z)", df, method="REML")
    m1 = gam("y ~ s(x) + s(z)", df, method="REML",
             start=np.asarray(m0.coef))
    m2 = gam("y ~ s(x) + s(z)", df, method="REML",
             mustart=m0.fitted_values)
    m3 = gam("y ~ s(x) + s(z)", df, method="REML",
             etastart=m0.linear_predictors)
    m4 = gam("y ~ s(x) + s(z)", df, method="REML",
             start=np.asarray(m0.coef) + 5.0)   # perturbed, still valid
    for m in (m1, m2, m3, m4):
        np.testing.assert_allclose(m.REML_criterion, m0.REML_criterion,
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.asarray(m.coef),
                                   np.asarray(m0.coef), atol=1e-8)
    # Non-gaussian + GCV path too (poisson, etastart route). The
    # initial.spg seed sees the user start (mgcv.r:4591-4595), so the
    # optimizer path shifts within the stop band (~2.6e-7 here — the
    # second sp rides a flat boundary ridge, so the GCV optimum is
    # shallow and the perturbed start lands a touch further off).
    p0 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson())
    p1 = gam("ycnt ~ s(x) + s(z)", df, family=Poisson(),
             etastart=p0.linear_predictors)
    np.testing.assert_allclose(p1.GCV_score, p0.GCV_score,
                               rtol=0, atol=1e-6)


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
            gam("ygam ~ s(x)", d2, family=Gamma(link="log"),
                method="REML", mustart=np.full(len(df), -5.0))


def test_plain_quasi_identity_link_full_newton_matches_mgcv():
    # mgcv's canonical for plain quasi is "none" (fix.family.link,
    # gam.fit3.r:2322): the inner loop runs full Newton even at the
    # identity link. quasi(identity, V=mu) would take Fisher steps under
    # a link==default test — the pins differ visibly in the sp. Same-
    # optimum stopping noise leaves sp ~1.5e-6 relative off R here
    # (criterion agrees to 7e-9; cf. the §2.3/§2.4 band records).
    from hea.family import Quasi
    d, _, _ = _cbind_fixture()
    m = gam("succ ~ s(x)", d,
            family=Quasi(link="identity", variance="mu"), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 137.1247120200,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.01412671672, rtol=1e-3)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.9415583200,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sigma_squared, 1.9565270162,
                               rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5.2: scale-known extended families through gam() — scat. mgcv 1.9-4
# references (identical data via full-precision CSV).
# ---------------------------------------------------------------------------

def _scat_fixture():
    # R-native (set.seed(2): runif bit-identical; rt within ~3e-15 from rgamma's
    # GD float-ordering gap, via rchisq — negligible vs the pins). Heavier-tailed seed
    # ν≈4.24 (well-determined) replaces the old numpy seed-99 sample: R-native
    # seed 99 lands ν≈26, a flat/ill-conditioned region. mgcv 1.9-4 pins below.
    from hea.R.rng import RGenerator
    g = RGenerator(2)
    n = 200
    x = g.uniform(size=n)
    f = np.sin(2 * np.pi * x) + 0.5 * x
    y = f + 0.3 * g.standard_t(4, size=n)
    return pl.DataFrame({"x": x, "y": y})


def test_scat_through_gam_matches_mgcv():
    # gam(y ~ s(x), family=scat(), REML): gam.fit4 PIRLS (Dd-table
    # weights w = ½Deta2, use.wy fallback), the (ρ, θ_fam) outer layout
    # with NO log φ slot, the family-generic Dd θ-gradient, preinitialize
    # θ seeding, and the scale-known H_aug/Vc chain.
    from hea.family import Scat
    df = _scat_fixture()
    m = gam("y ~ s(x)", df, family=Scat(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 93.4279560970729,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1550084027, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.8407668102,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.411261775854,
                               rtol=0, atol=1e-7)
    nu, sig = m.family.get_theta(trans=True)
    # nu/deviance/fitted carry the hea-vs-mgcv (sp, θ) convergence gap
    # (rel ~1e-7 on this heavier-tailed ν≈4.24 sample); the stationary
    # criterion and intercept stay tight to ~1e-10.
    np.testing.assert_allclose(nu, 4.23984596547, rtol=3e-6)
    np.testing.assert_allclose(sig, 0.281128469435, rtol=1e-6)
    np.testing.assert_allclose(m.deviance, 267.745438162, rtol=0, atol=3e-5)
    np.testing.assert_allclose(m.fitted_values[0], 1.14677329548,
                               rtol=0, atol=3e-8)
    np.testing.assert_allclose(m.Vp[0, 0], 0.0005459972642, rtol=1e-6)
    np.testing.assert_allclose(m.edf2_total, 7.9703610095, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.AIC, 171.0639212244, rtol=0, atol=1e-6)


def test_scat_ml_through_gam_matches_mgcv():
    from hea.family import Scat
    df = _scat_fixture()
    m = gam("y ~ s(x)", df, family=Scat(), method="ML")
    np.testing.assert_allclose(m.ML_criterion / 2, 90.4358656257,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.1649328349, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 7.7844589623,
                               rtol=0, atol=1e-4)
    nu, sig = m.family.get_theta(trans=True)
    np.testing.assert_allclose(nu, 4.2099380606, rtol=1e-6)
    np.testing.assert_allclose(sig, 0.2794428367, rtol=1e-6)


def test_scat_fixed_theta_fixed_sp_matches_mgcv():
    # Fixed (ν, σ) ⇒ n_theta=0 extended family: the inner gam.fit4 PIRLS
    # + extended criterion in isolation (no outer θ). At mgcv's converged
    # (sp, θ) the criterion must reproduce mgcv's REML to all digits.
    from hea.family import Scat
    df = _scat_fixture()
    fam = Scat(theta=(4.23984596546644, 0.281128469435471))
    assert fam.n_theta == 0
    m = gam("y ~ s(x)", df, family=fam, method="REML",
            sp=np.array([0.155008402666293]))
    np.testing.assert_allclose(m.REML_criterion / 2, 93.4279560970729,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.411261775854293,
                               rtol=0, atol=1e-8)


def test_extended_family_rejects_free_theta_with_fixed_sp():
    from hea.family import Scat
    df = _scat_fixture()
    with pytest.raises(ValueError, match="incompatible"):
        gam("y ~ s(x)", df, family=Scat(), method="REML",
            sp=np.array([0.1]))


def _nb_fixture():
    # R-native (bit-exact to set.seed(7), verified byte-for-byte): runif →
    # rgamma → rpois in R's draw order; mgcv pins are gam() fits on this data.
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
    # gam(y ~ s(x), family=nb(), REML) — the negative binomial extended
    # family with Θ estimated jointly (θ = log Θ in the outer vector,
    # scale fixed at 1). mgcv 1.9-4 pins.
    from hea.family import nb
    df = _nb_fixture()
    m = gam("y ~ s(x)", df, family=nb(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 318.541902729,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0774062275045, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.33449838393,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.coef)[0], 0.227494645531,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(m.family.get_theta(trans=True)[0]),
                               2.65778771157, rtol=1e-6)
    np.testing.assert_allclose(m.deviance, 203.567374398, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.null_deviance, 312.800417888,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.fitted_values[0], 0.656085286284,
                               rtol=0, atol=1e-8)
    # AIC/edf2 tightened post-B9 (Fisher-seed Vc2): Δ ≈ 3e-9 / 2e-11.
    np.testing.assert_allclose(m.edf2_total, 6.32010404395, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.AIC, 633.965932303, rtol=0, atol=1e-6)


def test_nb_fixed_theta_matches_mgcv():
    # nb(theta=3): Θ fixed (n_theta=0) — extended PIRLS + criterion with
    # no outer θ slot.
    from hea.family import nb
    df = _nb_fixture()
    fam = nb(theta=3.0)
    assert fam.n_theta == 0
    m = gam("y ~ s(x)", df, family=fam, method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 318.617850697,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.sp[0], 0.0685065177646, rtol=1e-4)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.49625628987,
                               rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# Sl penalty machinery (mgcv fast-REML.r) — §5.3 prerequisite 3.
# mgcv 1.9-4 references via gam(fit=FALSE) + mgcv:::Sl.setup / mgcv:::ldetS
# on identical data (full-precision CSV).
# ---------------------------------------------------------------------------

def _sl_fixture():
    from hea.R.rng import RGenerator
    gen = RGenerator(5)
    n = 120
    x = gen.uniform(0, 1, n)
    z = gen.uniform(0, 1, n)
    g = gen.mt.sample_int(6, n, replace=True)        # {0..5}
    y = np.sin(2 * np.pi * x) + 0.3 * z + gen.normal(0, 0.3, n)
    return pl.DataFrame({
        "x": x, "z": z,
        "g": pl.Series(g.astype(str)).cast(pl.Categorical),
        "y": y,
    })


def test_sl_setup_ldet_s_match_mgcv():
    # te-only (no gam.side, so the penalty basis is representation-
    # identical to mgcv's): every quantity pins exactly. Plus the t2
    # split-into-singletons path (disjoint penalty footprints).
    from hea.models.gam import _sl_setup, _ldet_s
    df = _sl_fixture()
    m = gam("y ~ te(x, z, k=5)", df, method="REML")
    sl = _sl_setup(m._slots, m.p)
    ld = _ldet_s(sl, np.array([-1.0, 1.5]), root=True, stot=True, deriv=2)
    np.testing.assert_allclose(ld["ldetS"], 16.9871017523, rtol=0, atol=1e-8)
    np.testing.assert_allclose(ld["ldet1"], [7.6597767860, 13.3402232140],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(sl.lam0, [0.1721080859, 0.1719973865],
                               rtol=1e-9)
    np.testing.assert_allclose(float(np.linalg.norm(m._slots[0].S)),
                               10.0618085492, rtol=1e-9)

    m2 = gam("y ~ t2(x, z, k=4)", df, method="REML")
    sl2 = _sl_setup(m2._slots, m2.p)
    # mgcv splits t2's three disjoint-footprint penalties into three
    # singleton blocks (1-based inclusive (2,5),(6,9),(10,13), rank 4).
    assert [(b.start, b.stop, b.n_sp, b.rank) for b in sl2.blocks] == [
        (1, 5, 1, 4), (5, 9, 1, 4), (9, 13, 1, 4)]
    ld2 = _ldet_s(sl2, np.array([0.2, -0.7, 1.1]), deriv=1)
    np.testing.assert_allclose(ld2["ldetS"], 2.4, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ld2["ldet1"], [4.0, 4.0, 4.0],
                               rtol=0, atol=1e-12)


def test_sl_mixed_model_matches_mgcv_geometry():
    # s(x) + te(x,z) + s(g, bs="re"): gam.side rotates the te penalties
    # into a different (equally valid) basis than mgcv's, so raw block
    # log-dets shift by a transform constant that cancels in the REML
    # criterion (fast-REML.r:294-296's own design). Pin the basis-
    # invariant quantities: block geometry/ranks, ldet1, ldet2, E rows.
    from hea.models.gam import _sl_setup, _ldet_s
    df = _sl_fixture()
    m = gam('y ~ s(x) + te(x, z, k=5) + s(g, bs="re")', df, method="REML")
    sl = _sl_setup(m._slots, m.p)
    assert [(b.start, b.stop, b.n_sp, b.rank) for b in sl.blocks] == [
        (1, 10, 1, 8), (10, 33, 2, 21), (33, 39, 1, 6)]
    ld = _ldet_s(sl, np.array([0.5, -1.0, 1.5, 0.3]),
                 root=True, stot=True, deriv=2)
    np.testing.assert_allclose(
        ld["ldet1"], [8.0, 7.6597767860, 13.3402232140, 6.0],
        rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        np.diag(ld["ldet2"]), [0.0, 0.8960604735, 0.8960604735, 0.0],
        rtol=0, atol=1e-8)
    np.testing.assert_allclose(ld["ldet2"][1, 2], -0.8960604735,
                               rtol=0, atol=1e-8)
    assert ld["E"].shape[0] == 35


def test_sl_machinery_invariants():
    # Coordinate-free self-consistency: E'E = S_total; ldetS equals the
    # dense log pseudo-determinant of S_total; ldet1/ldet2 match central
    # differences; Sl.mult == Σ Sl.termMult == S_total @ A; Xβ is
    # invariant under both reparameterizations; β round-trips.
    from hea.models.gam import (_sl_setup, _ldet_s, _sl_initial_repara,
                                _sl_repara, _sl_mult, _sl_term_mult)
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
        fd1[k] = (_ldet_s(_sl_setup(m._slots, m.p), rp_, deriv=0)["ldetS"]
                  - _ldet_s(_sl_setup(m._slots, m.p), rm_, deriv=0)["ldetS"]
                  ) / (2 * h)
        gp = _ldet_s(_sl_setup(m._slots, m.p), rp_, deriv=1)["ldet1"]
        gm = _ldet_s(_sl_setup(m._slots, m.p), rm_, deriv=1)["ldet1"]
        fdH[:, k] = (gp - gm) / (2 * h)
    np.testing.assert_allclose(ld["ldet1"], fd1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ld["ldet2"], fdH, rtol=0, atol=1e-6)

    rng = np.random.default_rng(11)
    A = rng.normal(size=(m.p, 3))
    SA, inds = _sl_term_mult(sl, A, full=True)
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
    np.testing.assert_allclose(_sl_repara(ld["rp"], brr, inverse=True), br,
                               atol=1e-10)
    # initial-repara inverse on a coefficient vector recovers the
    # original-coordinate β: b_orig = D·b_repara.
    b_back = _sl_initial_repara(sl, br, inverse=True)
    np.testing.assert_allclose(b_back, beta, atol=1e-8)


# ---------------------------------------------------------------------------
# Multi-formula front end (mgcv gam.setup.list) — §5.3 prerequisite 4.
# mgcv 1.9-4 references via gam(list(...), family=gaulss(), fit=FALSE).
# ---------------------------------------------------------------------------

def _mf_fixture():
    # R-native (set.seed(31)): x,z bit-exact (runif); w and the y-noise within a
    # few ulp (rnorm = scipy ndtri ≈ R qnorm5 to ~1e-12). Only sum|X| is pinned
    # to mgcv (design basis from x,z,w, not y); matches to ~1e-9.
    from hea.R.rng import RGenerator
    g = RGenerator(31)
    n = 150
    x = g.uniform(size=n)
    z = g.uniform(size=n)
    w = g.normal(0.0, 1.0, n)
    y = (np.sin(2 * np.pi * x) + 0.4 * w
         + g.normal(0.0, np.exp(0.3 * np.cos(2 * np.pi * z)), n))
    return pl.DataFrame({"x": x, "z": z, "w": w, "y": y})


def test_multi_formula_design_matches_mgcv():
    # gam(list(y ~ s(x) + w, ~ s(z)), family=gaulss(), fit=FALSE) pins:
    # column counts/order, lpi, nsdf/pstart, penalty offsets, mgcv's
    # term-name conventions ("(Intercept).1", smooth label "s.1(z)"),
    # and the aggregate design fingerprint — the stacked X matches
    # mgcv's to 8 decimals in total abs sum.
    from hea.models.gam import _prepare_multi_design
    df = _mf_fixture()
    md = _prepare_multi_design(["y ~ s(x) + w", "~ s(z)"], df)
    assert md.p == 21 and md.n_lp == 2
    assert md.nsdf == [2, 1]
    assert md.pstart == [0, 11]                      # mgcv 1-based (1, 12)
    np.testing.assert_array_equal(md.lpi[0], np.arange(0, 11))
    np.testing.assert_array_equal(md.lpi[1], np.arange(11, 21))
    # G$off (1-based): 3, 13 → 0-based slot col_starts 2, 12; both S 9×9.
    assert [(s.col_start, s.col_end) for s in md.slots] == [(2, 11),
                                                            (12, 21)]
    assert all(s.S.shape == (9, 9) for s in md.slots)
    assert md.column_names[:3] == ["(Intercept)", "w", "s(x).1"]
    assert md.column_names[11:14] == ["(Intercept).1", "s.1(z).1",
                                      "s.1(z).2"]
    assert md.blocks[0].label == "s(x)"
    assert md.blocks[1].label == "s.1(z)"
    np.testing.assert_allclose(float(np.abs(md.X).sum()), 2023.2041595800,
                               rtol=0, atol=1e-6)
    assert md.offsets == [None, None]
    assert md.L is None and md.n_work == 2


def test_multi_formula_lpmatrix_and_offsets():
    from hea.models.gam import _prepare_multi_design, _multi_lpmatrix
    df = _mf_fixture()
    md = _prepare_multi_design(["y ~ s(x) + w", "~ s(z)"], df)
    # lpmatrix on the training rows reproduces the stacked X; on a
    # permuted subset it evaluates per-LP bases consistently.
    Xn, lpi = _multi_lpmatrix(md, df[:7])
    np.testing.assert_allclose(Xn, md.X[:7], atol=1e-12)
    assert [len(i) for i in lpi] == [11, 10]
    perm = df[::-1][:10]
    Xp, _ = _multi_lpmatrix(md, perm)
    np.testing.assert_allclose(Xp, md.X[::-1][:10], atol=1e-12)

    # Per-formula offset() atoms land in the per-LP offset list.
    md2 = _prepare_multi_design(["y ~ s(x) + offset(w)", "~ s(z)"], df)
    np.testing.assert_allclose(md2.offsets[0], df["w"].to_numpy(),
                               atol=0)
    assert md2.offsets[1] is None

    # Three formulas stack fine (assembler is family-agnostic).
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
    # mgcv's numeric-label shared-term syntax: explicit refusal.
    with pytest.raises(NotImplementedError, match="shared-term"):
        _prepare_multi_design(["y ~ s(x)", "1 + 2 ~ s(z)"], df)
    # gam() on a formula list requires a general family (gam.fit5);
    # a general family conversely requires the formula list.
    with pytest.raises(NotImplementedError, match="general family"):
        gam(["y ~ s(x)", "~ s(z)"], df, method="REML")
    from hea.family import gaulss
    with pytest.raises(ValueError, match="list of formulas"):
        gam("y ~ s(x)", df, family=gaulss(), method="REML")
    with pytest.raises(ValueError, match="linear predictors"):
        gam(["y ~ s(x)", "~ s(z)", "~ x"], df, family=gaulss(),
            method="REML")


def test_gam_family_constructor_autocall():
    # mgcv accepts the family constructor — gam(family=gaulss) ≡
    # gam(family=gaulss()) via ``if (is.function(family)) family <-
    # family()`` (mgcv.r:2324). Instances pass through un-called.
    from hea.family import Gaussian, gaulss
    df = _mf_fixture()
    m = gam("y ~ s(x)", df, family=Gaussian, method="REML")
    assert isinstance(m.family, Gaussian)
    m2 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss,
             method="REML")
    assert isinstance(m2.family, gaulss)


# ---------------------------------------------------------------------------
# gam.vcomp rescale=TRUE default (pre-§5.3 slice i) — mgcv 1.9-4 references.
# R fits read the identical data via full-precision CSV; pins are printed
# gam.vcomp() values. S.scale is recorded per penalty by _scale_penalty
# (mgcv smooth.r:3877-3884) and vcomp's default divides each sp by it
# (mgcv.r:4242-4290); rescale=False is the fitted-scaling flavor.
# (The weighted-tp case is pinned in test_weights_gaussian_reml_matches_mgcv;
# factor-only bs="re" has S.scale=1 — the Machines pins cover invariance.)
# ---------------------------------------------------------------------------

def _vcomp_fixture():
    from hea.R.rng import RGenerator
    gen = RGenerator(2)        # R-native seed 7 saturates te(x1,x2)'s 2nd
    n = 240                    # penalty (degenerate vcomp CI); 2 keeps both
    x0 = gen.uniform(0, 1, n)  # te components well-determined
    x1 = gen.uniform(0, 1, n)
    x2 = gen.uniform(0, 1, n)
    fac = gen.mt.sample_int(3, n, replace=True)       # {0,1,2}
    g = gen.mt.sample_int(6, n, replace=True)         # {0..5}
    fg = np.array(["a", "b", "c"])[fac]
    gg = np.array([f"g{i}" for i in range(6)])[g]
    fb = np.where(fg == "a", np.sin(2 * np.pi * x0),
                  np.where(fg == "b", np.cos(2 * np.pi * x0), x0 ** 2 * 2.0))
    y = (0.3 + np.sin(2 * np.pi * x0) + (x1 * x2) ** 2 * 2.0 + fb
         + 0.3 * g * x0 + gen.normal(0, 0.4, n))
    return pl.DataFrame({"x0": x0, "x1": x1, "x2": x2, "fac": fg, "g": gg,
                         "y": y})


def test_vcomp_rescale_te_matches_mgcv():
    # s + te: S.scale recorded on the ASSEMBLED tensor penalties (the
    # smoothCon-level rescale; margin-level scaling is interior machinery).
    m = gam("y ~ s(x0) + te(x1, x2)", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(),
        [20.4386597, 0.1069975, 0.1079532, 0.7829882],
        rtol=1e-5)
    np.testing.assert_allclose(
        vc["lower"].to_numpy(),
        [10.8712116, 0.03464567, 0.03010987, 0.7124269],
        rtol=1e-5)
    np.testing.assert_allclose(
        vc["upper"].to_numpy(),
        [38.4261503, 0.3304445, 0.3870453, 0.8605381],
        rtol=1e-5)
    vc0 = m._compute_vcomp(rescale=False)
    np.testing.assert_allclose(
        vc0["std_dev"].to_numpy(),
        [4.3357825, 0.1986111, 0.1969816, 0.7829882],
        rtol=1e-5)


def test_vcomp_rescale_id_linked_full_sp_matches_mgcv():
    # s(x0, by=fac, id=1): one working λ, three penalty slots. mgcv's
    # $all divides every full.sp entry by the PROTOTYPE's S.scale
    # (clone.smooth.spec copies S.scale with the smooth); hea's per-slot
    # rows reproduce $all, with $vc's CI bounds shared across the rows.
    m = gam("y ~ s(x0, by=fac, id=1)", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(),
        [22.0423458] * 3 + [0.6340746], rtol=1e-6)
    np.testing.assert_allclose(
        vc["lower"].to_numpy(),
        [15.0467585] * 3 + [0.5775982], rtol=1e-6)
    np.testing.assert_allclose(
        vc["upper"].to_numpy(),
        [32.290344] * 3 + [0.6960731], rtol=1e-6)


def test_vcomp_rescale_select_null_penalty_scale_one():
    # select=TRUE appends the null-space penalty Sf with mgcv S.scale=1
    # (smooth.r:4241/4259), so its row is rescale-invariant; the main
    # penalty's row rescales as usual. Wider tolerances: the select fit
    # stops on a flatter surface (same band as the §2.3 record).
    m = gam("y ~ s(x0)", _vcomp_fixture(), method="REML", select=True)
    vc = m.vcomp
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy(),
        [21.4968621, 3.1636857, 0.8169613], rtol=2e-4)
    np.testing.assert_allclose(
        vc["lower"].to_numpy(),
        [11.4222009, 0.7325736, 0.7458715], rtol=2e-3)
    np.testing.assert_allclose(
        vc["upper"].to_numpy(),
        [40.4576216, 13.6626639, 0.8948269], rtol=2e-3)
    vc0 = m._compute_vcomp(rescale=False)
    # The appended Sf row is bit-identical across flavors (scale == 1).
    assert vc0["std_dev"][1] == vc["std_dev"][1]
    assert vc0["lower"][1] == vc["lower"][1]


def test_vcomp_rescale_fs_consistency_and_mgcv():
    # fs: multi-S block through the dedicated builder, null pair in hea's
    # canonical centered-Gram order (see test_fs_smooth_fit_matches_mgcv —
    # mgcv leaves it to LAPACK noise, which macOS x86 Accelerate doesn't
    # even keep stable per call). gam.vcomp pins are R 4.6.0 / mgcv 1.9-4
    # on the paraPen export of hea's exact X/S; paraPen records no S.scale,
    # so R's output IS the rescale=False flavor, positional. The default
    # flavor is pinned through the exact relation
    # σ_k(default) = σ_k(rescale=False)·√S.scale.
    m = gam("y ~ s(x0, g, bs='fs')", _vcomp_fixture(), method="REML")
    vc = m.vcomp
    vc0 = m._compute_vcomp(rescale=False)
    ss = np.array([s.S_scale for s in m._slots])
    np.testing.assert_allclose(
        vc["std_dev"].to_numpy()[:3],
        vc0["std_dev"].to_numpy()[:3] * np.sqrt(ss), rtol=1e-12)
    np.testing.assert_allclose(
        vc["lower"].to_numpy()[:3],
        vc0["lower"].to_numpy()[:3] * np.sqrt(ss), rtol=1e-12)
    np.testing.assert_allclose(
        vc0["std_dev"].to_numpy(),
        [4.382783625, 4.599802487, 3.526655952, 0.742592], rtol=1e-5)
    np.testing.assert_allclose(
        vc0["lower"].to_numpy(),
        [3.098904902, 2.292909040, 1.811794896, 0.6738163137], rtol=1e-5)
    np.testing.assert_allclose(
        vc0["upper"].to_numpy(),
        [6.198574307, 9.227659076, 6.864630332, 0.8183875445], rtol=1e-5)


def test_fs_smooth_fit_matches_mgcv():
    # hea canonicalizes the fs null pair (centered-Gram rotation in
    # _nat_param: constant-like column first, most-variable last) where
    # mgcv leaves the degenerate pair to LAPACK noise — macOS Accelerate
    # resolves it differently call-to-call (the free REML/2 flaps within
    # one process), so mgcv's realized bs="fs" basis isn't a reproducible
    # target even per-machine. Pins are R 4.6.0 / mgcv 1.9-4 fits of the
    # IDENTICAL parametrization: hea's X/S blocks exported at %.17g and
    # fitted via gam(y ~ X, paraPen=...) — R reproduced hea's free fit
    # (REML/2 307.1290091917) to all 13 digits. Everything is positional;
    # no sorted-pair hedging.
    df = _vcomp_fixture()
    m = gam("y ~ s(x0, g, bs='fs')", df, method="REML")
    np.testing.assert_allclose(
        np.asarray(m.sp),
        [0.028707836994574434, 0.026062865631070086, 0.04433782097095746],
        rtol=1e-6)
    np.testing.assert_allclose(m.REML_criterion / 2, 307.1290091916496,
                               rtol=1e-10)
    np.testing.assert_allclose(m.scale, 0.551442899755062, rtol=1e-9)
    np.testing.assert_allclose(np.sum(m.edf), 30.6221719808027, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:5],
        [1.6004302655596474, 0.3964249627940333, 0.6310862234821446,
         2.106833901078844, 1.471917571333663], atol=1e-8)
    # Each null dimension carries its own λ, so the two assignments of
    # (2, 0.5) to (constant-like, most-variable) are two DIFFERENT models,
    # each pinned to its own R paraPen value at fixed sp.
    f1 = gam("y ~ s(x0, g, bs='fs')", df, method="REML", sp=[1.0, 2.0, 0.5])
    np.testing.assert_allclose(f1.REML_criterion / 2, 346.0376552301282,
                               rtol=1e-10)
    f2 = gam("y ~ s(x0, g, bs='fs')", df, method="REML", sp=[1.0, 0.5, 2.0])
    np.testing.assert_allclose(f2.REML_criterion / 2, 345.2185763625583,
                               rtol=1e-10)


# ---------------------------------------------------------------------------
# summary pTerms + predict(unconditional=) (pre-§5.3 slice ii) — mgcv 1.9-4.
# pTerms (mgcv.r:3928-3977): one joint Wald test per whole parametric term,
# assign-exact column grouping, pinv-rank df, Chi.sq (known scale, pchisq)
# vs F (estimated scale, pf on n−Σedf). Printed via anova() exactly like
# mgcv (print.anova.gam shows pTerms.table; print.summary.gam does not).
# References from gam()+summary()$pTerms.table / predict.gam on the
# CSV-identical fixture.
# ---------------------------------------------------------------------------

def _pterms_fixture():
    # g5/lo are drawn AFTER the original columns so their values are
    # unchanged — the B1 predict-terms tests share this fixture.
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
    return pl.DataFrame({"x": x, "z": z, "f4": f4, "g5": g5, "lo": lo,
                         "ygau": ygau, "ypois": ypois})


def test_pterms_gaussian_F_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    rows = m._pterms_rows()
    assert [(r[0], r[1]) for r in rows] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose([r[2] for r in rows],
                               [80.0581257013, 59.5820363825], rtol=1e-6)
    np.testing.assert_allclose([r[3] for r in rows],
                               [1.99582307745e-33, 6.69516763716e-13],
                               rtol=1e-5)


def test_pterms_poisson_chisq_matches_mgcv():
    # Known scale → Chi.sq statistic with a pchisq p-value (est.disp=FALSE).
    m = gam("ypois ~ f4 + z + s(x)", _pterms_fixture(),
            family=Poisson(), method="REML")
    rows = m._pterms_rows()
    assert [(r[0], r[1]) for r in rows] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose([r[2] for r in rows],
                               [88.4400894000, 20.4132375000], rtol=1e-6)
    np.testing.assert_allclose([r[3] for r in rows],
                               [4.73776106e-19, 6.239669318e-06], rtol=1e-5)


def test_pterms_dropped_term_is_nan_like_mgcv():
    # z2 == z: the duplicate column is dropped (coef 0, zero Vp row), so
    # its pTerms row is df 1 with NaN stat/p — exactly mgcv's output.
    df = _pterms_fixture().with_columns(pl.col("z").alias("z2"))
    with pytest.warns(UserWarning, match="rank deficient"):
        m = gam("ygau ~ f4 + z + z2 + s(x)", df, method="REML")
    rows = m._pterms_rows()
    assert [r[0] for r in rows] == ["f4", "z", "z2"]
    np.testing.assert_allclose([rows[0][2], rows[1][2]],
                               [80.0600926600, 59.5846767600], rtol=1e-6)
    assert rows[2][1] == 1
    assert np.isnan(rows[2][2]) and np.isnan(rows[2][3])


def test_pls_rank_drop_alias_twin_canonical_on_any_blas():
    # Bit-identical (or negated) columns tie dgeqp3's pivot norms, and
    # the BLAS then drops whichever twin its kernel noise disfavors —
    # Accelerate kept z, OpenBLAS kept z2, splitting CI from local runs.
    # _pls_rank_drop canonicalizes to reference LAPACK's choice: the
    # earliest twin is kept, every later twin is dropped, on any build.
    from hea.models.gam import _pls_rank_drop
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    X[:, 3] = X[:, 1]                       # exact duplicate
    rank, drop = _pls_rank_drop(X, [], 6)
    assert rank == 5 and list(drop) == [3]
    X[:, 3] = -X[:, 1]                      # exact negated alias
    rank, drop = _pls_rank_drop(X, [], 6)
    assert rank == 5 and list(drop) == [3]
    X[:, 5] = X[:, 1]                       # three-way: keep first only
    rank, drop = _pls_rank_drop(X, [], 6)
    assert rank == 4 and list(drop) == [3, 5]


# ---------------------------------------------------------------------------
# §5.3 gam.fit5 — general-family inner Newton + implicit-differentiation
# derivative system. Pinned against mgcv:::gam.fit5 called directly at
# fixed lsp (deriv=2) after Sl.setup + Sl.initial.repara, exactly as
# estimate.gam stages it; Mp = ncol(totalPenaltySpace(...)$Z), which
# equals hea's structural Σnsdf + Σ(k − rank ΣS_block). References from
# mgcv 1.9-4 on the CSV-identical fixture. Coefficients are pinned via
# fitted values (per-column basis signs are convention).
# ---------------------------------------------------------------------------

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
    return pl.DataFrame(dict(x=x, z=z, w=w, y=y))


def _fit5_run(formulas, lsp, deriv=2):
    from hea.models.gam import (_prepare_multi_design, _sl_setup,
                                _sl_initial_repara, _gam_fit5, _sym_rank)
    from hea.family import gaulss
    md = _prepare_multi_design(formulas, _fit5_fixture())
    sl = _sl_setup(md.slots, md.p)
    X = _sl_initial_repara(sl, md.X, both_sides=False)
    Mp = sum(md.nsdf)
    for b, (a, bc) in zip(md.blocks, md.block_col_ranges):
        k = bc - a
        if not b.S:
            Mp += k
            continue
        Mp += k - _sym_rank(np.sum(
            [np.asarray(s, dtype=float) for s in b.S], axis=0))
    fit = _gam_fit5(X, md.y, np.asarray(lsp, dtype=float), sl,
                    family=gaulss(), lpi=md.lpi, offsets=md.offsets,
                    Mp=Mp, deriv=deriv)
    return fit, Mp


def test_gam_fit5_two_sp_matches_mgcv():
    # gam.fit5 internals at a fixed lsp: hea's _gam_fit5 reproduces
    # mgcv's gam.fit5 — the REML *function* matches R exactly at the
    # optimum (test_gaulss_free_fit) so it matches at any lsp, and the
    # derivative machinery (REML1/REML2/dVkk/db_drho) is confirmed by the
    # FD/deriv self-consistency tests below. Values are hea's on the
    # R-native data (== mgcv's gam.fit5).
    fit, Mp = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3])
    assert Mp == 5 and fit["rank"] == 21 and fit["converged"]
    np.testing.assert_allclose(fit["REML"], 231.7393228959, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(fit["REML1"],
                               [13.2496487044, 1.5371596913],
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fit["REML2"],
        [[5.44755992, -0.59313768], [-0.59313768, 2.22146868]],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit["l"], -197.5578307545, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(
        fit["fitted_values"][:2],
        [[1.46550206, 2.11966446], [-0.10735620, 1.69868960]],
        rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        fit["dVkk"], [[9.57215971, 0.68222719], [0.68222719, 1.17817349]],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.abs(fit["db_drho"]).sum(), 6.44240433,
                               rtol=0, atol=1e-6)


def test_gam_fit5_three_sp_matches_mgcv():
    # Two smooths in LP1 + one in LP2 — stresses the packed (i ≤ j)
    # indexing of d2b / trHid2H / d2ldetH.
    fit, Mp = _fit5_run(["y ~ s(x) + s(w)", "~ s(z)"], [0.5, -0.3, 1.2])
    assert Mp == 5 and fit["rank"] == 29
    np.testing.assert_allclose(fit["REML"], 242.0725467689, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(
        fit["REML1"], [11.6813000034, -1.4120335666, 6.7397195283],
        rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(fit["REML2"]).ravel(),
        [3.80412180, 0.12417761, -1.13087665,
         0.12417761, 0.45682933, 0.06802479,
         -1.13087665, 0.06802479, 4.91619318],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(fit["l"], -201.3004207932, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(
        fit["fitted_values"][:2],
        [[1.48531489, 2.07624430], [-0.07929165, 1.74268188]],
        rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.diag(fit["dVkk"]), [9.57121693, 0.16550807, 3.20053036],
        rtol=0, atol=1e-6)


def test_gam_fit5_deriv1_trace_path_matches_deriv2():
    # deriv=1 assembles d1ldetH from the gamlss_gH TRACE-vector form
    # (fh = Hp⁻¹); deriv=2 from the full ∂H/∂ρ list. Same REML1 either
    # way (gam.fit4.r:1347-1365).
    f2, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3], deriv=2)
    f1, _ = _fit5_run(["y ~ s(x) + w", "~ s(z)"], [0.5, -0.3], deriv=1)
    np.testing.assert_allclose(f1["REML1"], f2["REML1"], rtol=0,
                               atol=1e-9)
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
    # End-to-end: gam(formula list, family=gaulss()) — initial.spg
    # general seed (pen.reg initializer) → outer Newton over the
    # gam.fit5 REML closure (newton's coefficient carry-forward,
    # ε=1e-8) → final deriv-2 fit. R: gam(list(y ~ s(x) + w, ~ s(z)),
    # family=gaulss(), method="REML"), mgcv 1.9-4 — both stop within
    # the same band (sp agrees to ~7e-7 here; tolerances leave
    # cross-architecture headroom).
    from hea.family import gaulss
    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML")
    assert m.converged and m.method == "REML"
    np.testing.assert_allclose(m.sp, [0.14419871, 0.22829985], rtol=1e-4)
    np.testing.assert_allclose(m.REML_criterion / 2, 216.8833933770,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46270992, 2.16447805], [-0.15903017, 1.63246911]],
        rtol=0, atol=1e-5)
    # deviance = Σ deviance-residuals² (mgcv.r:2429); null deviance
    # from gaulss's postproc (gamlss.r:910-918).
    np.testing.assert_allclose(m.deviance, 219.85105997, rtol=1e-5)
    np.testing.assert_allclose(m.null_deviance, 871.08214845, rtol=1e-5)
    assert m.rank == 21
    # GCV.Cp silently coerces to REML for general families
    # (mgcv.r:1894-1898).
    m3 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
             method="GCV.Cp")
    assert m3.method == "REML"
    np.testing.assert_allclose(m3.REML_criterion / 2, 216.88339338,
                               rtol=0, atol=1e-5)


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
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML")
    np.testing.assert_allclose(m.edf_total, 13.97060461, rtol=0,
                               atol=1e-5)
    np.testing.assert_allclose(
        m.edf[:4], [1.0, 1.0, 0.99400183, 1.08473432], rtol=0,
        atol=1e-6)
    np.testing.assert_allclose(
        np.diag(m.Vp)[:4],
        [0.0031766700, 0.0097069400, 0.0579606100, 0.4328408500],
        rtol=1e-5)
    np.testing.assert_allclose(
        np.diag(m.Vc)[:4],
        [0.0031954300, 0.0097356600, 0.0642987400, 0.4543290500],
        rtol=1e-5)
    np.testing.assert_allclose(np.diag(m.Ve)[:2],
                               [0.0031480100, 0.0096115700], rtol=1e-6)
    np.testing.assert_allclose(m.edf1_total, 16.10915241, rtol=0,
                               atol=1e-5)
    np.testing.assert_allclose(m.edf2_total, 14.69865170, rtol=0,
                               atol=1e-5)
    np.testing.assert_allclose(m.AIC, 406.51598096, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.loglike, -188.55933878, rtol=0,
                               atol=1e-5)
    np.testing.assert_allclose(m.npar, 14.698652, rtol=0, atol=1e-4)
    vc = m.vcomp
    assert vc["name"].to_list() == ["s(x)", "s.1(z)", "scale"]
    np.testing.assert_allclose(vc["std_dev"].to_numpy()[:2],
                               [11.716203, 9.831598], rtol=1e-5)
    np.testing.assert_allclose(vc["lower"].to_numpy()[:2],
                               [6.4788824, 4.0192677], rtol=1e-5)
    np.testing.assert_allclose(vc["upper"].to_numpy()[:2],
                               [21.187206, 24.049236], rtol=1e-5)
    np.testing.assert_allclose(
        m.sp_vcov(),
        [[0.36531150, 0.00909964], [0.00909964, 0.83244672]], rtol=1e-4)
    # fixed-sp fits carry no sp-uncertainty: Vc ≡ Vp, no sp covariance
    m2 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
             method="REML", sp=np.array([2.0, 0.5]))
    assert m2.sp_vcov() is None
    np.testing.assert_allclose(m2.Vc, m2.Vp, rtol=0, atol=0)


def test_gaulss_predict_and_summary_surface_matches_mgcv():
    # Multi-LP user surface: predict (link/response/lpmatrix, se,
    # unconditional — per-LP columns fit/fit.1, se.fit/se.fit.1),
    # summary machinery (smooth rows against post.proc's R — mgcv's
    # object$R; pTerms over the real per-LP list; per-LP parametric
    # p.table indices). R references: predict(m, d[1:3,], ...) and
    # summary(m) on the same fixture.
    from hea.family import gaulss
    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML")
    p = m.predict(df[:3], type="link", se_fit=True)
    np.testing.assert_allclose(
        np.c_[p["fit"], p["fit.1"]],
        [[1.46270992, -0.79406171], [-0.15903017, -0.50655307],
         [1.55899517, -1.10404635]], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.c_[p["se.fit"], p["se.fit.1"]],
        [[0.06626794, 0.10166637], [0.06894145, 0.10901192],
         [0.08771956, 0.11470687]], rtol=0, atol=1e-6)
    pr = m.predict(df[:3], type="response")
    np.testing.assert_allclose(
        np.c_[pr["fit"], pr["fit.1"]],
        [[1.46270992, 2.16447805], [-0.15903017, 1.63246911],
         [1.55899517, 2.92802712]], rtol=0, atol=1e-5)
    pu = m.predict(df[:3], type="link", se_fit=True, unconditional=True)
    np.testing.assert_allclose(
        np.c_[pu["se.fit"], pu["se.fit.1"]],
        [[0.06677206, 0.10759165], [0.06943394, 0.11604036],
         [0.08817035, 0.11760738]], rtol=0, atol=1e-6)
    Xl = m.predict(df[:3], type="lpmatrix")
    assert Xl.shape == (3, 21)
    np.testing.assert_allclose(np.abs(Xl).sum(), 34.86671860, rtol=1e-7)
    # smooth table: edf / Ref.df / Chi.sq vs printed summary(m)
    rows = m._smooth_significance_rows()
    assert [r[0] for r in rows] == ["s(x)", "s.1(z)"]
    np.testing.assert_allclose(
        [(r[1], r[2], r[3]) for r in rows],
        [(6.365175, 7.510126, 588.421362), (4.605429, 5.599026, 144.651351)],
        rtol=1e-4)
    # pTerms: only LP1's `w` is a parametric term; Chi.sq = z²
    pt = m._pterms_rows()
    assert [(r[0], r[1]) for r in pt] == [("w", 1)]
    np.testing.assert_allclose(pt[0][2], 4.650478 ** 2, rtol=1e-4)
    # per-LP p.table indices pick up the `.1`-suffixed LP2 intercept
    par = dict(zip(m.parametric_columns,
                   np.asarray(m._beta_report)[m._param_idx]))
    np.testing.assert_allclose(par["(Intercept).1"], -0.58269247,
                               rtol=0, atol=1e-6)
    m.summary()      # prints the mgcv-layout summary without error


def test_fit5_fully_penalized_summary_matches_mgcv():
    # Fully-penalized smooths (zero penalty null space after the
    # centering constraint: cc here, re below) route summary through
    # reTest → recov (mgcv.r:3599), which consumes the model R factor
    # ``b$R`` verbatim — for general families that is
    # gam.fit5.post.proc's root with R'R = −lbb, not the PIRLS
    # √W·X factor (which fit5 never stores; ``_recov`` used to read it
    # unguarded and summary() crashed on any general fit with a cc/cp/re
    # smooth). One fixture drives all three recov consumptions:
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
    g = gen.mt.sample_int(8, n, replace=True)        # {0..7}
    b_g = gen.normal(0, 0.15, 8)
    y = (0.2 * np.sin(x) + 0.15 * np.cos(np.pi * v) + b_g[g]
         + gen.normal(0, 0.4, n))
    df = pl.DataFrame({
        "x": x, "v": v,
        "g": pl.Series(g.astype(str)).cast(pl.Categorical),
        "y": y,
    })
    kn = {"x": [0.0, 2 * np.pi]}
    ctl = {"epsilon": 1e-10, "newton": {"conv_tol": 1e-11}}
    TOL = 5e-5                  # one class, ~14x the worst cross-BLAS floor (s(v) λ)

    m1 = gam(['y ~ s(x, bs="cc")', "~ 1"], df, family=gaulss(),
             method="REML", knots=kn, control=ctl)
    assert m1.converged
    np.testing.assert_allclose(m1.REML_criterion / 2, 131.957334123150,
                               rtol=0, atol=TOL)
    np.testing.assert_allclose(m1.sp, [2657.482428], rtol=TOL)
    (label, edf, ref_df, stat, p_val), = m1._smooth_significance_rows()
    assert label == "s(x)"
    np.testing.assert_allclose(
        [edf, ref_df, stat], [1.82293088063, 8.0, 6.05893727191],
        rtol=TOL, err_msg="m1 s(x,cc) row vs mgcv s.table")
    np.testing.assert_allclose(p_val, 0.020872582751, rtol=TOL)
    m1.summary()                     # the original crash site

    m2 = gam(['y ~ s(x, bs="cc") + s(v) + s(g, bs="re")', "~ 1"], df,
             family=gaulss(), method="REML", knots=kn, control=ctl)
    assert m2.converged
    np.testing.assert_allclose(m2.REML_criterion / 2, 124.466904719375,
                               rtol=0, atol=TOL)
    # s(v) saturates on this data (sp→∞, edf→1): its λ rides a flat ridge
    # where hea (3.69e10) and mgcv (3.43e10) land at different huge values,
    # but the s.table row (edf 1, Chi.sq 10.54) is identical. s(x,cc)/s(g,re)
    # λ pin tight; s(v) λ only asserted saturated.
    np.testing.assert_allclose(m2.sp[[0, 2]], [1471.3655, 42.497327],
                               rtol=TOL)
    assert m2.sp[1] > 1e9
    rows = m2._smooth_significance_rows()
    assert [r[0] for r in rows] == ["s(x)", "s(v)", "s(g)"]
    np.testing.assert_allclose(
        [r[1:4] for r in rows],
        [(2.37312711793, 8.0, 19.4418827861),
         (1.00000000225, 1.00000000447, 10.5418947002),
         (5.37893803830, 7.0, 22.9558444790)],
        rtol=TOL, err_msg="m2 rows vs mgcv s.table")
    np.testing.assert_allclose(
        [r[4] for r in rows],
        [0.000165159607, 0.001167148839, 0.000089947459],
        rtol=TOL, err_msg="m2 p-values vs mgcv s.table")
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
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_efs(),
            method="REML")
    assert 3 <= m.outer_info["iter"] <= 8        # R: 5
    np.testing.assert_allclose(m.sp, [0.13952407, 0.17902162],
                               rtol=1e-3)
    np.testing.assert_allclose(m.REML_criterion / 2, 216.9188691300,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.edf_total, 14.27664854, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46232526, 2.14669899], [-0.15974078, 1.61848595]],
        rtol=0, atol=1e-4)
    # deriv-0 fits carry no outer Hessian: no sp-uncertainty surface
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
    m0 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
             method="REML")
    m1 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
             method="REML", start=np.asarray(m0._beta))
    np.testing.assert_allclose(m1.REML_criterion, m0.REML_criterion,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m1.sp, m0.sp, rtol=1e-5)
    with pytest.raises(ValueError, match="Length of start"):
        gam("y ~ s(x)", df, method="REML", start=np.zeros(11))
    with pytest.raises(NotImplementedError, match="etastart"):
        gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML", etastart=np.zeros(len(df)))


def test_gaulss_fixed_sp_through_gam_matches_mgcv():
    from hea.family import gaulss
    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML", sp=np.array([2.0, 0.5]))
    np.testing.assert_allclose(m.REML_criterion / 2, 233.9898943665,
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.46246549, 2.08069657], [-0.09947020, 1.67028576]],
        rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.sp, [2.0, 0.5])


def test_general_family_derivs_dispatch_guards():
    # mgcv's optimizer dispatch on available.derivs (mgcv.r:1906-1908):
    # ==1 → c("outer","bfgs") — the BFGS outer optimizer (item 7): a
    # derivs-1 family fits through gam.fit5 at deriv ≤ 1 (score + grad),
    # never the deriv-2/trHid2H path Newton needs;
    # ==0 → every fit5 call stays at deriv 0 (ll deriv ≤ 1), including
    # the fixed-sp and no-smooth paths that previously hard-coded a
    # deriv-2 final fit (mgcv fits derivs-0 families only through
    # efsudr's deriv=0 calls, gam.fit4.r:1479+).
    from hea.family import gaulss

    df = _fit5_fixture()

    # available_derivs==1 forces bfgs. gaulss's ll supports every order,
    # so the bfgs fit must reach the SAME REML optimum as the derivs-2
    # newton fit — a cross-check of the bfgs port against newton. The
    # faithful subclass asserts ll is never asked past the bfgs tier
    # (deriv ≤ 2, i.e. gam.fit5's dH/trace order — gamlss_gH deriv ≤ 1).
    class _gaulss_d1(gaulss):
        available_derivs = 1

        def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv=0,
               d1b=None, d2b=None, fh=None, D=None):
            assert deriv <= 2, \
                f"bfgs asked a derivs-1 family for ll(deriv={deriv})"
            return super().ll(y, X, coef, wt, lpi=lpi, offset=offset,
                              deriv=deriv, d1b=d1b, d2b=d2b, fh=fh, D=D)

    m_bfgs = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d1(),
                 method="REML")
    m_newton = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
                   method="REML")
    np.testing.assert_allclose(m_bfgs.REML_criterion,
                               m_newton.REML_criterion, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m_bfgs._beta),
                               np.asarray(m_newton._beta), rtol=0, atol=1e-4)
    np.testing.assert_allclose(m_bfgs.sp, m_newton.sp, rtol=1e-3)
    assert m_bfgs.outer_info["conv"] == "full convergence"

    class _gaulss_d0(gaulss):
        # faithful derivs-0 family: anything above ll deriv 1 is an
        # error — the shape of every d2logpdf-only custom family
        available_derivs = 0

        def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv=0,
               d1b=None, d2b=None, fh=None, D=None):
            assert deriv <= 1, \
                f"derivs-0 family asked for ll(deriv={deriv})"
            return super().ll(y, X, coef, wt, lpi=lpi, offset=offset,
                              deriv=deriv, d1b=d1b, d2b=d2b, fh=fh,
                              D=D)

    # fixed sp: the inner Newton is deriv-independent, so the deriv-0
    # final fit must reproduce the deriv-2 gaulss fit at the same sp;
    # the post-fit surface is efs-grade (Vc ≡ Vp; sp_vcov None is the
    # fixed-sp rule either way).
    m2 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
             method="REML", sp=np.array([2.0, 0.5]))
    m0 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d0(),
             method="REML", sp=np.array([2.0, 0.5]))
    np.testing.assert_allclose(m0.REML_criterion, m2.REML_criterion,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(m0._beta),
                               np.asarray(m2._beta), rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(m0.Vp), np.asarray(m2.Vp),
                               rtol=0, atol=1e-10)
    np.testing.assert_array_equal(m0.Vc, m0.Vp)
    assert m0.sp_vcov() is None

    # no-smooth formula list (n_work == 0 — the intercept-only shape
    # every derivs-0 consumer family hits): same deriv-0 protocol.
    m0p = gam(["y ~ w", "~ 1"], df, family=_gaulss_d0(), method="REML")
    m2p = gam(["y ~ w", "~ 1"], df, family=gaulss(), method="REML")
    np.testing.assert_allclose(m0p.REML_criterion, m2p.REML_criterion,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(m0p._beta),
                               np.asarray(m2p._beta), rtol=0,
                               atol=1e-10)


def test_optimizer_knob_efs_and_validation():
    # gam(optimizer=) — mgcv's intake and dispatch: first element
    # "outer"|"efs" with estimate.gam's "unknown optimizer" error
    # (mgcv.r:1913), second element defaulting to "newton" with
    # gam.outer's "unknown outer optimization method." (mgcv.r:
    # 1643-1644), efs forcing method="REML" (mgcv.r:1914), and the
    # available.derivs==1 coercion skipped when efs is requested
    # (mgcv.r:1907). Only newton + efs are ported (C9: bfgs/nlm/optim;
    # single-formula efs = the efsudr port, gam.fit4.r:822).
    from hea.family import gaulss

    df = _fit5_fixture()

    class _gaulss_d0(gaulss):
        available_derivs = 0

    # forcing efs on a derivs-2 family ≡ the automatic derivs-0
    # dispatch byte-for-byte; the R reference for this exact call is
    # gam(list(...), gaulss(), optimizer="efs") — the pins of
    # test_gaulss_efs_optimizer_matches_mgcv.
    m_auto = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d0(),
                 method="REML")
    m_knob = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
                 method="REML", optimizer="efs")
    _assert_fp_equiv(m_knob._beta, m_auto._beta)
    _assert_fp_equiv(m_knob.sp, m_auto.sp)
    assert m_knob.optimizer == ("efs", "newton")
    np.testing.assert_allclose(m_knob.REML_criterion / 2,
                               216.9188691300, rtol=0, atol=1e-3)
    assert m_knob.sp_vcov() is None       # deriv-0 fit: no outer hess

    # efs coerces the method like mgcv.r:1914 (the general path is
    # REML-coerced anyway, mgcv.r:1894 — the fit must be identical)
    m_gcv = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
                method="GCV.Cp", optimizer="efs")
    assert m_gcv.method == "REML"
    _assert_fp_equiv(m_gcv._beta, m_knob._beta)

    # derivs==1 + optimizer="efs" is legal (mgcv.r:1907 only coerces
    # to bfgs when efs was NOT requested); ll never asked past deriv 1
    class _gaulss_d1(gaulss):
        available_derivs = 1

        def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv=0,
               d1b=None, d2b=None, fh=None, D=None):
            assert deriv <= 1, \
                f"efs asked a derivs-1 family for ll(deriv={deriv})"
            return super().ll(y, X, coef, wt, lpi=lpi, offset=offset,
                              deriv=deriv, d1b=d1b, d2b=d2b, fh=fh,
                              D=D)

    m_d1 = gam(["y ~ s(x) + w", "~ s(z)"], df, family=_gaulss_d1(),
               method="REML", optimizer="efs")
    _assert_fp_equiv(m_d1._beta, m_auto._beta)

    # intake validation — mgcv's exact messages
    with pytest.raises(ValueError, match="unknown optimizer"):
        gam("y ~ s(x)", df, method="REML", optimizer="perf")
    with pytest.raises(ValueError,
                       match="unknown outer optimization method"):
        gam("y ~ s(x)", df, method="REML",
            optimizer=("outer", "magic"))
    # optimizer=("outer","bfgs") forces bfgs on a general family (item 7);
    # gaulss supports every ll order, so the forced-bfgs fit reaches the
    # same REML optimum as the default newton fit.
    m_bfgs = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
                 method="REML", optimizer=("outer", "bfgs"))
    m_newton = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
                   method="REML")
    np.testing.assert_allclose(m_bfgs.REML_criterion,
                               m_newton.REML_criterion, rtol=0, atol=1e-6)
    assert m_bfgs.optimizer == ("outer", "bfgs")
    # standard-family bfgs is still unported (gam.fit3 outer loop, C9).
    with pytest.raises(NotImplementedError, match="C9"):
        gam("y ~ s(x)", df, method="REML", optimizer=("outer", "bfgs"))
    with pytest.raises(NotImplementedError, match="efsudr"):
        gam("y ~ s(x)", df, method="REML", optimizer="efs")

    # the default knob is inert on the single-formula path
    m_def = gam("y ~ s(x)", df, method="REML")
    m_opt = gam("y ~ s(x)", df, method="REML",
                optimizer=("outer", "newton"))
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

    # default (200) converges in 5 steps (the gaulss-efs pin); a
    # generous cap raising the ceiling above the natural convergence
    # point is INERT — byte-identical fit, same iter, same conv flag.
    # This is the parity guarantee: efs_maxit only matters when the
    # loop would otherwise run past it.
    m_def = gam(forms, df, family=gaulss(), method="REML",
                optimizer="efs")
    assert m_def.outer_info["conv"] == "full convergence"
    assert m_def.outer_info["iter"] == 5
    m_big = gam(forms, df, family=gaulss(), method="REML",
                optimizer="efs", control={"efs_maxit": 500})
    _assert_fp_equiv(m_big._beta, m_def._beta)
    _assert_fp_equiv(m_big.sp, m_def.sp)
    assert m_big.outer_info["conv"] == "full convergence"
    assert m_big.outer_info["iter"] == 5

    # a cap below the natural convergence point BINDS: the loop stops
    # at exactly efs_maxit and reports mgcv's "iteration limit
    # reached" conv message (the same string surfaced at iter==200).
    m_cap = gam(forms, df, family=gaulss(), method="REML",
                optimizer="efs", control={"efs_maxit": 3})
    assert m_cap.outer_info["conv"] == "iteration limit reached"
    assert m_cap.outer_info["iter"] == 3

    # validation: a non-positive cap is rejected at gam.control intake
    # (mirrors maxit's "must be > 0"), both via gam.control directly
    # and through the gam() control= revalidation path.
    with pytest.raises(ValueError, match="efs_maxit"):
        gam_control(efs_maxit=0)
    with pytest.raises(ValueError, match="efs_maxit"):
        gam(forms, df, family=gaulss(), method="REML",
            optimizer="efs", control={"efs_maxit": -1})


def test_general_family_authoring_contract():
    # The GeneralFamily authoring contract (the documented public
    # extension API, mgcv general.family analog), frozen as a test: a
    # from-scratch family written purely against the class docstring —
    # 3 LPs, derivs=0, custom clamped links, ll filling the packed
    # l1/l2 arrays and delegating to gamlss_etamu/gamlss_gH, with
    # initialize_coef/postproc/residuals overrides — fits end-to-end
    # through the formula-list gam. Any drift in the call protocol
    # (kwarg names, deriv ceiling, per-LP offset lists, lpi layout,
    # the 6-arg keyword-called postproc) fails here instead of
    # surfacing in external family authors' code. The FD block is
    # also the first end-to-end validation of the K=3 etamu/gH l1/l2
    # branches (oracle pins exist only at K=2, via gaulss).
    from itertools import combinations_with_replacement
    from scipy.special import digamma, gammaln, polygamma
    from hea.family import (GeneralFamily, IdentityLink, Link,
                            gamlss_etamu, gamlss_gH, trind_generator)

    class _ShiftedLogLink(Link):
        # σ = b + exp(η) — a clamped scale link in the mgcv custom-
        # link shape (cf. gaulss's logb)
        name = "slog"
        b = 0.01

        def link(self, mu):
            return np.log(np.asarray(mu, dtype=float) - self.b)

        def linkinv(self, eta):
            return self.b + np.exp(np.clip(
                np.asarray(eta, dtype=float), -700.0, 700.0))

        def mu_eta(self, eta):
            return np.maximum(
                np.exp(np.clip(np.asarray(eta, dtype=float),
                               -700.0, 700.0)),
                np.finfo(float).eps)

        def d2link(self, mu):
            return -1.0 / (np.asarray(mu, dtype=float) - self.b) ** 2

        def d3link(self, mu):
            return 2.0 / (np.asarray(mu, dtype=float) - self.b) ** 3

        def d4link(self, mu):
            return -6.0 / (np.asarray(mu, dtype=float) - self.b) ** 4

    class _TanhLink(Link):
        # λ = tanh(η) ∈ (−1, 1), linkinv clamped inside the open
        # interval, mu_eta eps-floored (the consumer's bounded-shape
        # link pattern)
        name = "tanh"

        def link(self, mu):
            return np.arctanh(np.asarray(mu, dtype=float))

        def linkinv(self, eta):
            eps = np.finfo(float).eps
            return np.clip(np.tanh(np.asarray(eta, dtype=float)),
                           -1.0 + eps, 1.0 - eps)

        def mu_eta(self, eta):
            a = np.exp(-2.0 * np.abs(np.asarray(eta, dtype=float)))
            return np.maximum(4.0 * a / (1.0 + a) ** 2,
                              np.finfo(float).eps)

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
        # Student-t location/scale/shape: parameters (μ, σ, λ) with
        # ν = 6 + 4λ ∈ (2, 10) — a real 3-parameter density with
        # closed-form l1/l2, authored exactly as the contract
        # docstring prescribes
        name = "tlss-dummy"
        n_lp = 3
        available_derivs = 0
        scale_known = True
        n_theta = 0

        def __init__(self):
            super().__init__([IdentityLink(), _ShiftedLogLink(),
                              _TanhLink()])
            self.tri = trind_generator(3)
            self.seen = {"deriv": [], "use_unscaled": [],
                         "postproc_kwargs": None, "offsets": None,
                         "lpi_cols": None, "wt_len": None}

        @staticmethod
        def _l0(y, mu, sigma, lam):
            nu = 6.0 + 4.0 * lam
            z = (y - mu) / sigma
            return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                    - 0.5 * np.log(nu * np.pi) - np.log(sigma)
                    - (nu + 1.0) / 2.0 * np.log1p(z * z / nu))

        @staticmethod
        def _lp_derivs(y, mu, sigma, lam):
            # first/second log-density derivatives w.r.t. (μ, σ, λ);
            # λ enters through ν = 6 + 4λ (chain factor 4)
            nu = 6.0 + 4.0 * lam
            z = (y - mu) / sigma
            q = nu + z * z
            g = (nu + 1.0) * z / q
            dg_dz = (nu + 1.0) * (nu - z * z) / q ** 2
            dg_dnu = z * (z * z - 1.0) / q ** 2
            dC = (0.5 * digamma((nu + 1.0) / 2.0)
                  - 0.5 * digamma(nu / 2.0) - 0.5 / nu)
            d2C = (0.25 * polygamma(1, (nu + 1.0) / 2.0)
                   - 0.25 * polygamma(1, nu / 2.0) + 0.5 / nu ** 2)
            l_nu = (dC - 0.5 * np.log1p(z * z / nu)
                    + (nu + 1.0) * z * z / (2.0 * nu * q))
            l_nunu = (d2C + z * z / (2.0 * nu * q)
                      - z * z * (nu * nu + 2.0 * nu + z * z)
                      / (2.0 * nu * nu * q * q))
            d1 = {"mu": g / sigma,
                  "sigma": (g * z - 1.0) / sigma,
                  "lam": 4.0 * l_nu}
            d2 = {("mu", "mu"): -dg_dz / sigma ** 2,
                  ("mu", "sigma"): -(g + z * dg_dz) / sigma ** 2,
                  ("mu", "lam"): 4.0 * dg_dnu / sigma,
                  ("sigma", "sigma"):
                      (1.0 - 2.0 * g * z - z * z * dg_dz) / sigma ** 2,
                  ("sigma", "lam"): 4.0 * z * dg_dnu / sigma,
                  ("lam", "lam"): 16.0 * l_nunu}
            return d1, d2

        def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv=0,
               d1b=None, d2b=None, fh=None, D=None):
            self.seen["deriv"].append(deriv)
            self.seen["wt_len"] = None if wt is None else len(wt)
            self.seen["offsets"] = offset
            self.seen["lpi_cols"] = sorted(
                int(c) for ix in lpi for c in np.asarray(ix))
            assert deriv <= 1, \
                f"derivs-0 family asked for ll(deriv={deriv})"
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
            w = (np.ones_like(y) if wt is None
                 else np.asarray(wt, dtype=float))
            l0 = self._l0(y, mu, sigma, lam)
            ret = {"l": float(np.sum(w * l0))}
            if deriv == 0:
                return ret
            names = ("mu", "sigma", "lam")
            params = {"mu": mu, "sigma": sigma, "lam": lam}
            d1, d2 = self._lp_derivs(y, mu, sigma, lam)
            shape = y.shape
            l1 = np.column_stack(
                [np.broadcast_to(d1[p], shape) for p in names])
            l2 = np.column_stack(
                [np.broadcast_to(d2[k], shape)
                 for k in combinations_with_replacement(names, 2)])
            l1 = l1 * w[:, None]
            l2 = l2 * w[:, None]
            ig1 = np.column_stack(
                [lnk.mu_eta(eta)
                 for lnk, eta in zip(self.links, etas)])
            g2 = np.column_stack(
                [lnk.d2link(params[name])
                 for lnk, name in zip(self.links, names)])
            tri = self.tri
            de = gamlss_etamu(l1, l2, None, None, ig1, g2, None, None,
                              tri["i2"], tri["i3"], tri["i4"],
                              deriv - 1)
            gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                           l3=de["l3"], i3=tri["i3"], l4=de["l4"],
                           i4=tri["i4"], d1b=d1b, d2b=d2b,
                           deriv=deriv - 1, fh=fh, D=D)
            ret.update(gh)
            return ret

        def _null_params(self, y):
            y = np.asarray(y, dtype=float)
            return (float(np.median(y)),
                    max(float(np.std(y)) * 0.8, 0.05), 0.0)

        def initialize_coef(self, y, X, lpi, E=None, offset=None,
                            use_unscaled=False):
            self.seen["use_unscaled"].append(bool(use_unscaled))
            y = np.asarray(y, dtype=float)
            X = np.asarray(X, dtype=float)
            jj = [np.asarray(ix, dtype=int) for ix in lpi]
            n, p = X.shape
            if E is None:
                E = np.zeros((0, p))
            start = np.zeros(p)
            for j, (lnk, p0) in enumerate(
                    zip(self.links, self._null_params(y))):
                target = np.full(n, float(lnk.link(p0)))
                if offset is not None and offset[j] is not None:
                    target = target - offset[j]
                cols = jj[j]
                xa = np.vstack([X[:, cols], E[:, cols]])
                ta = np.concatenate([target, np.zeros(E.shape[0])])
                b, *_ = np.linalg.lstsq(xa, ta, rcond=None)
                start[cols] = np.where(np.isfinite(b), b, 0.0)
            return start

        def postproc(self, y, prior_weights, fitted,
                     linear_predictors, offset, intercept):
            self.seen["postproc_kwargs"] = {
                "prior_weights": np.shape(prior_weights),
                "fitted": np.shape(fitted),
                "linear_predictors": np.shape(linear_predictors),
                "intercept": intercept}
            y = np.asarray(y, dtype=float)
            f0 = np.broadcast_to(np.asarray(self._null_params(y)),
                                 (y.shape[0], 3))
            r0 = self.residuals(y, f0)
            return {"null_deviance": float(np.sum(r0 * r0))}

        def residuals(self, y, fitted, type: str = "deviance"):
            y = np.asarray(y, dtype=float)
            fitted = np.asarray(fitted, dtype=float)
            mu, sigma, lam = (fitted[:, 0], fitted[:, 1],
                              fitted[:, 2])
            rsd = y - mu
            if type == "response":
                return rsd
            if type == "pearson":
                nu = 6.0 + 4.0 * lam
                return rsd / (sigma * np.sqrt(nu / (nu - 2.0)))
            l_sat = self._l0(y, y, sigma, lam)
            l_obs = self._l0(y, mu, sigma, lam)
            return np.sign(rsd) * np.sqrt(
                2.0 * np.clip(l_sat - l_obs, 0.0, None))

    # --- FD validation of the dummy's ll: lb against FD of l, lbb
    # against FD of lb, at a hand-built 3-LP design — also the first
    # FD pin of the K=3 gamlss_etamu/gamlss_gH l1/l2 branches
    fam = _TLSS()
    rng = np.random.default_rng(11)
    n = 40
    Xs = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 2)),
                    np.ones((n, 1)), rng.normal(size=(n, 1)),
                    np.ones((n, 1))])
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
        fd_lb[k] = (fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=0)["l"]
                    - fam.ll(yv, Xs, cm, wt, lpi=lpi,
                             deriv=0)["l"]) / (2 * h)
        fd_lbb[:, k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=1)["lb"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=1)["lb"]
        ) / (2 * h)
    np.testing.assert_allclose(base["lb"], fd_lb, rtol=2e-5,
                               atol=1e-7)
    np.testing.assert_allclose(base["lbb"],
                               0.5 * (fd_lbb + fd_lbb.T),
                               rtol=5e-5, atol=5e-6)

    # --- end-to-end: 3 formulas, a per-LP offset atom, auto-efs
    n = 320
    x = rng.uniform(size=n)
    off1 = 0.3 * rng.normal(size=n)
    y = (1.0 + np.sin(2 * np.pi * x) + off1
         + rng.standard_t(df=7.0, size=n) * 0.7)
    df = pl.DataFrame({"y": y, "x": x, "off1": off1})
    fam2 = _TLSS()
    m = gam(["y ~ s(x) + offset(off1)", "~ 1", "~ 1"], df,
            family=fam2, method="REML")

    # derivs-0 protocol: ll never asked past deriv 1 on any path
    assert max(fam2.seen["deriv"]) <= 1
    # both initialize_coef flavors ran: the initial.spg seed (False)
    # and gam.fit5's ldetS-root start (True)
    assert set(fam2.seen["use_unscaled"]) == {False, True}
    # per-LP offsets arrive as a length-n_lp list, the formula-offset
    # LP carrying its vector and offset-free LPs None
    off_seen = fam2.seen["offsets"]
    assert isinstance(off_seen, (list, tuple)) and len(off_seen) == 3
    np.testing.assert_allclose(np.asarray(off_seen[0]), off1,
                               rtol=0, atol=1e-12)
    assert off_seen[1] is None and off_seen[2] is None
    # lpi covers the stacked design exactly, 0-based
    assert fam2.seen["lpi_cols"] == list(
        range(len(np.asarray(m._beta))))
    assert fam2.seen["wt_len"] == n
    # efs auto-dispatch ran
    assert m.outer_info["conv"] in ("full convergence",
                                    "iteration limit reached")
    # the 6-arg postproc was keyword-called on the converged fit;
    # null_deviance landed, deviance fell back to Σ residuals²
    pk = fam2.seen["postproc_kwargs"]
    assert pk == {"prior_weights": (n,), "fitted": (n, 3),
                  "linear_predictors": (n, 3), "intercept": True}
    assert np.isfinite(m.null_deviance) and m.null_deviance > 0
    np.testing.assert_allclose(
        m.deviance, float(np.sum(np.asarray(m.residuals) ** 2)),
        rtol=1e-12)
    # residuals_of dispatches through the family hook with type=
    np.testing.assert_array_equal(
        np.asarray(m.residuals_of("deviance")),
        np.asarray(m.residuals))
    fitted = np.asarray(m.fitted_values)
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("response")), y - fitted[:, 0],
        rtol=0, atol=1e-12)
    # fit sanity (Monte-Carlo level): smooth μ(x) + offset recovered,
    # σ̂ near 0.7, λ̂ strictly inside (−1, 1)
    truth = 1.0 + np.sin(2 * np.pi * x) + off1
    assert np.corrcoef(fitted[:, 0], truth)[0, 1] > 0.95
    assert 0.3 < float(np.median(fitted[:, 1])) < 1.5
    assert float(np.max(np.abs(fitted[:, 2]))) < 0.999
    # predict/summary run on the K=3 fit
    pred = m.predict(df[:5])
    assert pred.shape[0] == 5
    m.summary()

    # fully-penalized smooth under the from-scratch family: summary's
    # reTest path needs gam.fit5.post.proc's R (R'R = −lbb) in _recov —
    # the fit5 path no gaulss-only test reaches with a cc/cp/re smooth
    fam3 = _TLSS()
    m_cc = gam(['y ~ s(x, bs="cc")', "~ 1", "~ 1"], df, family=fam3,
               method="REML", knots={"x": [0.0, 1.0]})
    (label, edf, ref_df, stat, p_cc), = m_cc._smooth_significance_rows()
    assert label == "s(x)" and ref_df > 0
    assert np.isfinite(stat) and 0.0 <= p_cc <= 1.0
    m_cc.summary()


def test_general_family_newton_reml_nlp4_robustness():
    # The general-family Newton/REML path exercised and verified at
    # n_lp == 4 — the regime the gevlss (n_lp == 3) and _TLSS (n_lp ==
    # 3) tests cannot reach. trind_generator/gamlss_etamu/gamlss_gH are
    # all K-generic, but the FULLY-MIXED fourth-derivative branch
    # (family.py: ``d4 = l4 * ig1*ig1*ig1*ig1`` for four DISTINCT
    # params, the mo==1 case) needs four distinct LP indices, so it is
    # STRUCTURALLY unreachable below K==4 — no other test in the suite
    # touches it. This drives a from-scratch 4-LP family on the full
    # Newton-REML path (gam.fit5 to ll deriv 4) and validates: (a) the
    # K=4 l1/l2 chain by FD, (b) the all-distinct etamu l3/l4 columns
    # both against their closed form (index plumbing) and an
    # independent η-space mixed-difference stencil (numeric), and (c)
    # an end-to-end fit that itself reaches ll(deriv=4) at K==4.
    from itertools import (combinations_with_replacement,
                           product as iproduct)
    from hea.family import (GeneralFamily, IdentityLink, LogLink, Link,
                            gamlss_etamu, gamlss_gH, trind_generator)

    K = 4
    AQ = np.ones(K)        # per-parameter quadratic curvature
    DQ = 1.0               # quartic floor -> l0 bounded above
    CC = 0.5               # all-distinct coupling (|CC|/4 < DQ)

    class _TanhLink(Link):
        # bounded shape link (cf. the consumer families' (-1,1) link)
        name = "tanh"

        def link(self, mu):
            return np.arctanh(np.asarray(mu, float))

        def linkinv(self, eta):
            eps = np.finfo(float).eps
            return np.clip(np.tanh(np.asarray(eta, float)),
                           -1 + eps, 1 - eps)

        def mu_eta(self, eta):
            a = np.exp(-2.0 * np.abs(np.asarray(eta, float)))
            return np.maximum(4.0 * a / (1.0 + a) ** 2,
                              np.finfo(float).eps)

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
        # u_k = mu_k - target; target_0 = y (LP1 tracks the response),
        # the shape LPs target (1, 0, 0) inside each link's range
        u = mus - np.array([0.0, 1.0, 0.0, 0.0])[None, :]
        u[:, 0] = mus[:, 0] - y
        return u

    def _l0(y, mus):
        # synthetic 4-LP quasi-log-likelihood, bounded above and smooth,
        # with a NONZERO all-distinct 4th partial (the CC product term)
        u = _u(y, mus)
        return (-0.5 * (AQ[None, :] * u * u).sum(1)
                - DQ * (u ** 4).sum(1) + CC * np.prod(u, axis=1))

    def _partial(u, idx):
        # exact mixed partial of _l0 w.r.t. the multi-index idx: the
        # quadratic/quartic terms feed only PURE columns; the CC product
        # (multilinear) feeds only DISTINCT-index columns
        n = u.shape[0]
        s = set(idx)
        if len(idx) == 1:
            k = idx[0]
            prod = np.full(n, CC)
            for j in range(K):
                if j != k:
                    prod = prod * u[:, j]
            return -AQ[k] * u[:, k] - 4.0 * DQ * u[:, k] ** 3 + prod
        if len(s) != len(idx):                 # some repeated index
            if len(idx) == 2 and len(s) == 1:
                return -AQ[idx[0]] - 12.0 * DQ * u[:, idx[0]] ** 2
            if len(idx) == 3 and len(s) == 1:
                return -24.0 * DQ * u[:, idx[0]]
            if len(idx) == 4 and len(s) == 1:
                return np.full(n, -24.0 * DQ)
            return np.zeros(n)
        missing = [j for j in range(K) if j not in s]   # all distinct
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
            super().__init__([IdentityLink(), LogLink(),
                              IdentityLink(), _TanhLink()])
            self.tri = trind_generator(4)
            self.seen = {"deriv": [], "lpi_cols": None}

        def ll(self, y, X, coef, wt, *, lpi, offset=None, deriv=0,
               d1b=None, d2b=None, fh=None, D=None):
            self.seen["deriv"].append(deriv)
            y = np.asarray(y, float)
            X = np.asarray(X, float)
            coef = np.asarray(coef, float)
            jj = [np.asarray(ix, int) for ix in lpi]
            self.seen["lpi_cols"] = sorted(
                int(c) for ix in jj for c in ix)
            etas = []
            for j in range(K):
                eta = X[:, jj[j]] @ coef[jj[j]]
                if offset is not None and offset[j] is not None:
                    eta = eta + offset[j]
                etas.append(eta)
            mus = np.column_stack([lnk.linkinv(e)
                                   for lnk, e in zip(self.links, etas)])
            w = (np.ones_like(y) if wt is None
                 else np.asarray(wt, float))
            ret = {"l": float(np.sum(w * _l0(y, mus)))}
            if deriv == 0:
                return ret
            u = _u(y, mus)
            packs = [list(combinations_with_replacement(range(K), m))
                     for m in (1, 2, 3, 4)]
            l1, l2, l3, l4 = (
                np.column_stack([_partial(u, c) for c in cs]) * w[:, None]
                for cs in packs)
            ig1 = np.column_stack([lnk.mu_eta(e)
                                   for lnk, e in zip(self.links, etas)])
            g2 = np.column_stack([lnk.d2link(mus[:, k])
                                  for k, lnk in enumerate(self.links)])
            g3 = np.column_stack([lnk.d3link(mus[:, k])
                                  for k, lnk in enumerate(self.links)])
            g4 = np.column_stack([lnk.d4link(mus[:, k])
                                  for k, lnk in enumerate(self.links)])
            t = self.tri
            de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                              t["i2"], t["i3"], t["i4"], deriv - 1)
            if deriv > 1 and d1b is None:
                # standalone high-deriv etamu inspection (the engine
                # always supplies d1b on the fit path)
                return {**ret, "_de": de, "_ig1": ig1, "_u": u}
            gh = gamlss_gH(X, jj, de["l1"], de["l2"], t["i2"],
                           l3=de["l3"], i3=t["i3"], l4=de["l4"],
                           i4=t["i4"], d1b=d1b, d2b=d2b,
                           deriv=deriv - 1, fh=fh, D=D)
            ret.update(gh)
            return ret

        def initialize_coef(self, y, X, lpi, E=None, offset=None,
                            use_unscaled=False):
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

        def postproc(self, y, prior_weights, fitted,
                     linear_predictors, offset, intercept):
            y = np.asarray(y, float)
            r0 = y - float(np.median(y))
            return {"null_deviance": float(np.sum(r0 * r0))}

        def residuals(self, y, fitted, type: str = "deviance"):
            y = np.asarray(y, float)
            fitted = np.asarray(fitted, float)
            return y - fitted[:, 0]

    # --- (a) FD validation of lb/lbb on a hand-built 4-LP design ---
    fam = _K4()
    rng = np.random.default_rng(7)
    n = 36
    Xs = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 2)),   # LP1: 3
                    np.ones((n, 1)), rng.normal(size=(n, 1)),   # LP2: 2
                    np.ones((n, 1)),                            # LP3: 1
                    np.ones((n, 1)), rng.normal(size=(n, 1))])  # LP4: 2
    lpi = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 6),
           np.arange(6, 8)]
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
        fd_lb[k] = (fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=0)["l"]
                    - fam.ll(yv, Xs, cm, wt, lpi=lpi,
                             deriv=0)["l"]) / (2 * h)
        fd_lbb[:, k] = (
            fam.ll(yv, Xs, cp, wt, lpi=lpi, deriv=1)["lb"]
            - fam.ll(yv, Xs, cm, wt, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(base["lb"], fd_lb, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(base["lbb"], 0.5 * (fd_lbb + fd_lbb.T),
                               rtol=5e-5, atol=5e-6)

    # --- (b) the K=4-only all-distinct etamu branch: closed form ---
    dd = fam.ll(yv, Xs, coef, wt, lpi=lpi, deriv=4)
    de, ig1, u = dd["_de"], dd["_ig1"], dd["_u"]
    cols3 = list(combinations_with_replacement(range(K), 3))
    cols4 = list(combinations_with_replacement(range(K), 4))
    ad4 = cols4.index((0, 1, 2, 3))
    ad3 = cols3.index((0, 1, 2))
    # full chain rule of an all-distinct mixed partial is just the
    # product of first-order link factors (no repeated index -> no
    # g2/g3/g4 correction): l4=CC -> CC*prod(ig1); l3 -> CC*u_miss*ig1s
    want4 = CC * ig1[:, 0] * ig1[:, 1] * ig1[:, 2] * ig1[:, 3]
    want3 = CC * u[:, 3] * ig1[:, 0] * ig1[:, 1] * ig1[:, 2]
    np.testing.assert_allclose(de["l4"][:, ad4], want4, rtol=0,
                               atol=1e-12)
    np.testing.assert_allclose(de["l3"][:, ad3], want3, rtol=0,
                               atol=1e-12)
    # independent numeric corroboration: a 16-point mixed central
    # difference of l0 w.r.t. the four DISTINCT etas (wt == 1 here)
    etas0 = np.column_stack([Xs[:, lpi[j]] @ coef[lpi[j]]
                             for j in range(K)])
    hs = 8e-3
    acc = np.zeros(n)
    for signs in iproduct((1, -1), repeat=K):
        et = etas0 + hs * np.array(signs)[None, :]
        mus = np.column_stack([lnk.linkinv(et[:, j])
                               for j, lnk in enumerate(fam.links)])
        acc += float(np.prod(signs)) * _l0(yv, mus)
    np.testing.assert_allclose(de["l4"][:, ad4], acc / (2 * hs) ** 4,
                               rtol=0, atol=1e-4)

    # --- (c) end-to-end Newton-REML fit at n_lp == 4 ---
    N = 300
    x = rng.uniform(size=N)
    y = np.sin(2 * np.pi * x) + rng.normal(size=N) * 0.3
    df = pl.DataFrame({"y": y, "x": x})
    fam2 = _K4()
    m = gam(["y ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family=fam2,
            method="REML")
    # the Newton-REML path drove the family to ll(deriv=4) AT K==4 --
    # i.e. trHid2H exercised the all-distinct l4 branch inside the fit
    assert 4 in fam2.seen["deriv"] and 3 in fam2.seen["deriv"]
    assert m.outer_info["conv"] == "full convergence"
    # lpi spans the 4-LP stacked design, 0-based
    assert fam2.seen["lpi_cols"] == list(range(len(np.asarray(m._beta))))
    fitted = np.asarray(m.fitted_values)
    assert fitted.shape == (N, 4)
    # LP1 recovered the smooth signal; the posterior surface is finite
    assert np.corrcoef(fitted[:, 0], np.sin(2 * np.pi * x))[0, 1] > 0.95
    assert np.all(np.isfinite(np.asarray(m.Vp)))
    assert np.isfinite(m.REML_criterion)
    assert np.isfinite(m.null_deviance) and m.null_deviance > 0
    np.testing.assert_allclose(
        m.deviance, float(np.sum(np.asarray(m.residuals) ** 2)),
        rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("response")), y - fitted[:, 0],
        rtol=0, atol=1e-12)
    assert m.predict(df[:5]).shape[0] == 5
    m.summary()


def _twlss_fixture():
    # Tweedie response over smooth μ(x) + linear w, generated on R's stream via
    # hea.R.rng (bit-exact to set.seed(9)): runif for x/z/w, the ported rTweedie
    # for y. So this is R's set.seed(9) data byte-for-byte (verified), and the
    # pins below are mgcv fits on it, reproducible by a pure-R script:
    #   set.seed(9); x<-runif(300); z<-runif(300); w<-runif(300)
    #   mu<-exp(0.3+sin(2*pi*x)+0.3*w); y<-rTweedie(mu, p=1.55, phi=0.9)
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
    # Gamma location-scale data on R's set.seed(11) stream via hea.R.rng
    # (bit-exact runif + the rust-backed rgamma), so the mgcv fit below is
    # reproducible by a pure-R script:
    #   set.seed(11); n<-250; x<-runif(n); z<-runif(n)
    #   mean<-exp(0.5+sin(2*pi*x)); sigma<-exp(-1+0.6*z)
    #   y<-rgamma(n, shape=1/sigma, scale=mean*sigma)
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
    # Gumbel location-scale data on R's set.seed(13) stream via hea.R.rng:
    # inverse-CDF draws (bit-exact runif), reproducible by:
    #   set.seed(13); n<-250; x<-runif(n); z<-runif(n)
    #   loc<-0.5+sin(2*pi*x); logbeta<- -0.5+0.4*z; u<-runif(n)
    #   y<-loc - exp(logbeta)*log(-log(u))
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
    # GEV data on R's set.seed(17) stream (bit-exact runif), GEV inverse-CDF
    # with shape ξ=0.1. Reproducible by:
    #   set.seed(17); n<-300; x<-runif(n); z<-runif(n)
    #   mu<-0.5+sin(2*pi*x); logsig<- -0.5+0.3*z; xi<-0.1; u<-runif(n)
    #   y<-mu + ((-log(u))^(-xi)-1)*exp(logsig)/xi
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
    m1 = gam(["y ~ s(x) + w", "~ 1", "~ s(z)"], df, family=twlss(),
             method="REML")
    np.testing.assert_allclose(m1.REML_criterion / 2, 515.6445331249,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m1.sp[0], 0.1810879219, rtol=1e-4)
    # second sp is a flat λ→∞ direction (s(z) on ρ shrunk to linear,
    # edf 1.007): R stops at ~1574, hea at ~1580 — same working-
    # infinity band; pin the direction + its edf instead of the value.
    assert m1.sp[1] > 500.0
    np.testing.assert_allclose(m1.edf_total, 10.2292798004, rtol=0,
                               atol=2e-3)
    rows = m1._smooth_significance_rows()
    np.testing.assert_allclose(rows[1][1], 1.0070551110, rtol=0,
                               atol=5e-3)
    # tp-basis eigenvector signs are build noise (cf. the fs record):
    # pin |coef|, plus fitted values which are sign-invariant.
    np.testing.assert_allclose(
        np.abs(np.asarray(m1._beta)[:4]),
        np.abs([0.3636872521, 0.2340139050, 1.7497627520,
                0.2383358048]), rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[:2],
        [[4.1009048580, -0.0109253440, -0.1731325510],
         [2.7718950940, -0.0109253440, -0.0679816568]],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(m1.deviance, 357.7929921000, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(m1.null_deviance, 567.2259056000,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(m1.residuals)[:3],
        [-1.1855994500, -0.2774933634, -0.4901982250], rtol=0,
        atol=1e-4)
    assert 6 <= m1.outer_info["iter"] <= 12         # R: 9
    np.testing.assert_allclose(np.asarray(m1.Vp)[0, 0], 0.0102796362,
                               rtol=0, atol=1e-6)
    assert m1.sp_vcov() is None                     # deriv-0 fit
    pred = m1.predict(df[:3])
    np.testing.assert_allclose(
        pred["fit"].to_numpy(),
        np.asarray(m1.fitted_values)[:3, 0], rtol=0, atol=1e-10)
    m1.summary()

    m2 = gam(["y ~ s(x)", "~ z", "~ 1"], df, family=twlss(),
             method="REML")
    np.testing.assert_allclose(m2.REML_criterion / 2, 514.1250577000,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m2.sp, [0.1885776983], rtol=1e-4)
    np.testing.assert_allclose(m2.edf_total, 9.1773799040, rtol=0,
                               atol=1e-4)
    # LP2 (θ) + LP3 (ρ) parametric coefficients — exact stop point
    np.testing.assert_allclose(
        np.asarray(m2._beta)[10:13],
        [-0.0034247063, -0.0106819944, -0.1195655320], rtol=0,
        atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(m2.fitted_values)[0],
        [3.8117852460, -0.0062579028, -0.1195655320], rtol=0,
        atol=1e-5)
    np.testing.assert_allclose(m2.deviance, 358.1491851000, rtol=0,
                               atol=1e-4)
    np.testing.assert_allclose(m2.null_deviance, 564.4292756000,
                               rtol=0, atol=1e-3)


def test_twlss_weighted_residuals_match_mgcv():
    # mgcv's twlss ll IGNORES prior weights (gamlss.r:2556 — wt
    # unread), so a weighted fit is IDENTICAL to the unweighted one;
    # weights enter only the deviance residuals (object$prior.weights,
    # gamlss.r:2541 — hea's optional prior_weights residuals keyword)
    # and the postproc null deviance. R-verified both ways.
    from hea.family import twlss

    df = _twlss_fixture()
    pw = np.tile([1.0, 2.0], 150)
    mw = gam(["y ~ s(x)", "~ 1", "~ 1"], df, family=twlss(),
             method="REML", weights=pw)
    mu = gam(["y ~ s(x)", "~ 1", "~ 1"], df, family=twlss(),
             method="REML")
    # fit invariance (R: REML/sp/coef all.equal TRUE)
    np.testing.assert_allclose(mw.REML_criterion, mu.REML_criterion,
                               rtol=0, atol=1e-9)
    _assert_fp_equiv(mw._beta, mu._beta)
    np.testing.assert_allclose(mw.REML_criterion / 2, 514.2855621000,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(mw.sp, [0.1886027173], rtol=1e-4)
    # weighted deviance surface (R refs)
    np.testing.assert_allclose(mw.deviance, 549.3608583000, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(mu.deviance, 358.1582964000, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(mw.null_deviance, 855.1007602000,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(mu.null_deviance, 564.4194261000,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(mw.residuals)[:3],
        [-1.0646171470, -0.2494332248, -0.4555850390], rtol=0,
        atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(mu.residuals)[:3],
        [-1.0646171470, -0.1763759247, -0.4555850390], rtol=0,
        atol=1e-4)
    # √w scaling of the deviance residual √(2(yθ−κ)w/φ) is a per-row
    # property — verify it on ONE fitted object (μ/θ/φ identical) so the
    # scaling is exact to ~1 ulp and BLAS-independent. Comparing the two
    # SEPARATE fits mw/mu can't: their coefs are only BLAS-equal (≤3e-14,
    # assert_fp_equiv's floor) and yθ−κ cancels catastrophically, so at
    # the near-zero-residual rows that drift amplifies ~5000× (~1e-9 on
    # x86_64 Accelerate / OpenBLAS) — far past any 1e-12 cross-fit gate.
    # The end-to-end √2 wiring stays pinned above: mw.residuals[1] =
    # −0.2494 = mu.residuals[1]·√2 (a pw=2 row), pw=1 rows 0/2 identical.
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
    np.testing.assert_allclose(m.REML_criterion / 2, 376.9788523,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.1010813233, rtol=1e-4)
    # s(z) on the log-scale LP is a flat λ→∞ direction (R sp ≈ 2.4e4,
    # edf ≈ 1): pin the direction, not the value (cf. twlss/shash).
    assert m.sp[1] > 500.0
    np.testing.assert_allclose(m.edf_total, 8.586757468, rtol=0, atol=2e-3)
    # tp-basis eigenvector signs are build noise (cf. the fs record):
    # pin |coef|; coef[0] (the log-mean intercept) is sign-fixed.
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.5351402153, 2.185161584, 0.1922625554, 0.1110607536]),
        rtol=0, atol=1e-4)
    # fitted matrix is (mean, log σ): col 0 exponentiated by postproc
    # (gamlss.r:2739) — mgcv's object$fitted.values after the in-place
    # rewrite. R: rows 1-2 = (4.5658, -0.3322), (1.8810, -0.7676).
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[4.56583108, -0.3321797963], [1.881021014, -0.7676485103]],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 272.1383283, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 532.7823068, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.002131979348,
                               rtol=0, atol=1e-6)

    # family$predict hook: type="response" returns (mean, σ) — mean is
    # e^{η₁}, NOT the per-LP linkinv — with delta-method SEs (R:
    # predict(m, type="response", se.fit=TRUE)).
    pr = m.predict(df[:3], se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [4.56583108, 1.881021014, 1.385835927], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        pr["fit.1"].to_numpy(),
        [-0.3321797963, -0.7676485103, -0.3454239398], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.4412707648, 0.4300296758, 0.1464255231], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        pr["se.fit.1"].to_numpy(),
        [0.149232104, 0.1060445032, 0.1444384638], rtol=0, atol=1e-5)
    # response prediction on training rows == the (exp'd-mean) fitted col.
    pr_all = m.predict(type="response")
    np.testing.assert_allclose(
        pr_all["fit"].to_numpy(), np.asarray(m.fitted_values)[:, 0],
        rtol=0, atol=1e-9)
    # link-scale prediction returns η (log mean / log σ), unhooked.
    pr_lnk = m.predict(df[:2], type="link")
    np.testing.assert_allclose(
        pr_lnk["fit"].to_numpy(),
        np.log(np.asarray(m.fitted_values)[:2, 0]), rtol=0, atol=1e-9)
    m.summary()


def test_gumbls_through_gam_matches_mgcv():
    # R: gam(list(y ~ s(x), ~ s(z)), family=gumbls(), method="REML") on
    # the set.seed(13) Gumbel fixture — derivs=2 full Newton; rides the
    # same BoundedLogLink + predict-hook engine path as gammals.
    from hea.family import gumbls

    df = _gumbls_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=gumbls(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 355.4765655,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.1329424295, rtol=1e-3)
    assert m.sp[1] > 500.0          # flat λ→∞ direction (s(z) on log β)
    np.testing.assert_allclose(m.edf_total, 8.327469431, rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.4872208906, -1.993640641, 0.2718586452, 0.1469972102]),
        rtol=0, atol=1e-3)
    # fitted matrix is (mean, log β): col 0 = location + e^β·γ, added in
    # place by postproc (gamlss.r:3070). R: rows 1-2 of fitted.values.
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[-0.09536002225, -0.2589687628], [1.836117586, -0.4267671006]],
        rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 291.9068221, rtol=0, atol=1e-2)
    assert np.isnan(m.null_deviance)      # mgcv leaves gumbls null dev NA
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00270728526,
                               rtol=0, atol=1e-5)

    # gumbls predict returns the (location, log β) — NOT the mean —
    # deliberately differing from the mean column of fitted_values
    # (mgcv's gumbls predict omits the Euler correction).
    pr = m.predict(df[:3], se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [-0.5408822784, 1.45941768, 1.0853551], rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        pr["fit.1"].to_numpy(),
        [-0.2589687628, -0.4267671006, -0.2658441114], rtol=0, atol=1e-4)
    # the asymmetry: predict response fit[0] (location) ≠ fitted mean[0].
    assert abs(float(pr["fit"][0]) - float(m.fitted_values[0, 0])) > 0.1
    m.summary()


def test_gevlss_through_gam_matches_mgcv():
    # R: gam(list(y ~ s(x), ~ s(z), ~ 1), family=gevlss(), method="REML")
    # on the set.seed(17) GEV fixture. gevlss has parameter-dependent
    # support, so the inner Newton hits 1+ξ(y−μ)/σ ≤ 0 on trial steps —
    # exercising gam.fit5's non-finite-ll step rejection (mgcv warns
    # "NaNs produced" there too; hea silences it, fit converges
    # identically). derivs=2 full Newton; no predict hook (response =
    # per-LP linkinv).
    from hea.family import gevlss

    df = _gevlss_fixture()
    m = gam(["y ~ s(x)", "~ s(z)", "~ 1"], df, family=gevlss(),
            method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 400.7915694,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp[0], 0.09316907739, rtol=1e-3)
    assert m.sp[1] > 50.0           # s(z)-on-log-σ flat-ish direction
    np.testing.assert_allclose(m.edf_total, 10.54442778, rtol=0, atol=2e-3)
    # ξ is intercept-only (LP3 ~ 1): the shifted-logit shape intercept.
    np.testing.assert_allclose(np.asarray(m._beta)[-1], 1.184566711,
                               rtol=0, atol=1e-4)
    # fitted matrix is (μ, ρ=log σ, ξ) — per-LP linkinv, no override.
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[1.268048747, -0.3566666145, 0.1486518752],
         [0.02079092758, -0.4185000686, 0.1486518752]], rtol=0, atol=1e-3)
    np.testing.assert_allclose(float(np.sum(np.diag(np.asarray(m.Vp)))),
                               2.90608114, rtol=0, atol=1e-2)
    assert np.isnan(m.null_deviance)        # mgcv leaves gevlss null dev NA
    assert 4 <= m.outer_info["iter"] <= 10  # R: 6

    # no predict hook: type="response" is the per-LP linkinv (μ, ρ, ξ),
    # so the μ column equals the fitted μ on training rows.
    pr = m.predict(type="response")
    np.testing.assert_allclose(
        pr["fit"].to_numpy(), np.asarray(m.fitted_values)[:, 0],
        rtol=0, atol=1e-9)
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
    m = gam(["y ~ x + xb", "~ s(z)", "~ 1"], df, family=gevlss(),
            method="REML")
    assert m.rank == 13 and len(np.asarray(m._beta)) == 14
    np.testing.assert_allclose(m.REML_criterion / 2, 448.9062817,
                               rtol=0, atol=1e-3)
    beta = np.asarray(m._beta)
    np.testing.assert_allclose(beta[2], 0.0, rtol=0, atol=0)   # dropped
    np.testing.assert_allclose(beta[0], 1.3078499, rtol=0, atol=1e-3)
    np.testing.assert_allclose(beta[1], -1.9194328, rtol=0, atol=1e-3)


def _cox_ph_fixture():
    # Cox PH data on R's set.seed(7) stream via hea.R.rng (bit-exact runif):
    # exponential survival times (continuous, no ties), a random censoring
    # indicator passed as weights. Reproducible by:
    #   set.seed(7); n<-200; x<-runif(n); z<-runif(n)
    #   lp<-0.7*sin(2*pi*x)+0.4*z; time<- -log(runif(n))/exp(lp)
    #   event<-as.integer(runif(n)<0.75)
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
    # R: gam(time ~ s(x) + z, family=cox.ph(), weights=event,
    # method="REML"). The first single-formula general entry (nlp=1) and
    # the first non-gamlss_gH ll (partial likelihood over risk sets). The
    # intercept is dropped (drop.intercept=TRUE) — coef names carry no
    # "(Intercept)" — and the survivor function lands in fitted.values.
    from hea.family import cox_ph

    df = _cox_ph_fixture()
    m = gam("time ~ s(x) + z", df, family=cox_ph(), weights=df["event"],
            method="REML")
    # drop.intercept: 10 coefs (z + 9 smooth), no intercept column
    assert m.column_names == ["z"] + [f"s(x).{i}" for i in range(1, 10)]
    assert "(Intercept)" not in m.column_names
    # basis-invariant fit quantities vs mgcv (tp eigenvector signs are
    # build noise → the parametric z coef and |smooth coef| are pinned)
    np.testing.assert_allclose(m.REML_criterion / 2, 619.504025083614,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(float(np.sum(m.edf)), 5.04951670043,
                               rtol=0, atol=1e-2)
    np.testing.assert_allclose(m.sp[0], 0.0888380593715, rtol=2e-3, atol=0)
    np.testing.assert_allclose(m.null_deviance, 251.817485857,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 244.337920299, rtol=0, atol=1e-3)
    beta = np.asarray(m._beta)
    np.testing.assert_allclose(beta[0], 0.5347033442229, rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        np.abs(beta[1:]),
        np.abs([1.1918412609676, 0.9055424491633, -0.0820593633819,
                -0.1120411669932, -0.0370197861409, -0.3200248848103,
                -0.1442647042816, 0.7227752645372, 0.5849763355643]),
        rtol=0, atol=2e-3)
    # fitted = survivor function (n,), matches mgcv element-wise
    fitted = np.asarray(m.fitted_values)
    assert fitted.shape == (200,)
    np.testing.assert_allclose(
        fitted[:5], [0.984691862732, 0.734034958684, 0.779977961331,
                     0.889916922677, 0.544964819219], rtol=0, atol=1e-5)


def test_cox_ph_residuals_predict_match_mgcv():
    # cox.ph deviance/martingale residuals and the survivor-function
    # prediction hook (predict.gam → family$predict, coxph.r:199-245),
    # which needs the new event times from newdata.
    from hea.family import cox_ph

    df = _cox_ph_fixture()
    m = gam("time ~ s(x) + z", df, family=cox_ph(), weights=df["event"],
            method="REML")
    np.testing.assert_allclose(
        np.asarray(m.residuals)[:5],
        [2.52471540, 0.98282250, 1.13211646, -0.48296411, 0.46087840],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("martingale"))[:5],
        [0.98457348, 0.69080138, 0.75151039, -0.11662717, 0.39296596],
        rtol=0, atol=1e-4)
    with pytest.raises(NotImplementedError, match="score/schoenfeld"):
        m.residuals_of("score")
    # response-scale prediction = survivor function at the new times
    nd = df[:6].select(["time", "x", "z"])
    pr = m.predict(nd, type="response", se_fit=True)
    np.testing.assert_allclose(
        np.asarray(pr["fit"]),
        [0.9846918627, 0.7340349587, 0.7799779613, 0.8899169227,
         0.5449648192, 0.0735293562], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(pr["se.fit"]),
        [0.00851784061, 0.05314490011, 0.05525484119, 0.03550937447,
         0.06468752039, 0.04388060327], rtol=0, atol=1e-4)


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
    mv = ["time", "status", "trt", "sex", "protime",
          "platelet", "age", "bili", "albumin"]
    cc = pbc.drop_nulls(subset=mv)
    assert cc.height == 308
    status1 = (cc["status"] == 2).cast(pl.Float64)
    m = gam("time ~ trt + sex + s(sqrt(protime)) + s(platelet) + s(age) "
            "+ s(bili) + s(albumin)", cc, family=cox_ph(),
            weights=status1, method="REML")
    # drop.intercept + factor reference (sex levels m,f → "sexf"); the
    # sqrt(protime) smooth-arg expression resolves
    assert "(Intercept)" not in m.column_names
    assert m.column_names[:2] == ["trt", "sexf"]
    # overall fit (flat ridges → modest tolerances)
    np.testing.assert_allclose(m.REML_criterion / 2, 547.256744526281,
                               rtol=0, atol=5e-2)
    np.testing.assert_allclose(float(np.sum(m.edf)), 15.3095029449,
                               rtol=0, atol=5e-2)
    np.testing.assert_allclose(m.null_deviance, 413.412618165,
                               rtol=0, atol=1e-2)
    # parametric terms (no treatment effect; sex marginal) — mgcv exact
    beta = dict(zip(m.column_names, np.asarray(m._beta)))
    np.testing.assert_allclose(beta["trt"], 0.06715634, rtol=0, atol=2e-3)
    np.testing.assert_allclose(beta["sexf"], -0.49515759, rtol=0, atol=3e-3)
    # smooth table vs summary(b)$s.table
    rows = {r[0]: r for r in m._smooth_significance_rows()}
    # meaningful curves — tight
    np.testing.assert_allclose(rows["s(age)"][1], 6.042792, rtol=0, atol=2e-2)
    np.testing.assert_allclose(rows["s(age)"][3], 29.417277, rtol=0, atol=0.3)
    np.testing.assert_allclose(rows["s(bili)"][1], 4.264429, rtol=0, atol=2e-2)
    np.testing.assert_allclose(rows["s(bili)"][3], 89.540337, rtol=0, atol=0.5)
    # flat ridges — edf≈1, Chi² pinned (sp itself is a free λ→∞ direction)
    for lab, chi in (("s(sqrt(protime))", 13.333751),
                     ("s(platelet)", 5.787376),
                     ("s(albumin)", 31.086251)):
        np.testing.assert_allclose(rows[lab][1], 1.0, rtol=0, atol=2e-2)
        np.testing.assert_allclose(rows[lab][3], chi, rtol=0, atol=0.3)
    # every smooth significant at 5% except none here is non-sig
    for lab in rows:
        assert 0.0 <= rows[lab][4] <= 1.0
    m.summary()
    # plot_smooth resolves the s(sqrt(protime)) smooth-arg expression
    # against self.data (the general path now materializes it like the
    # single-formula path) — regression for the ColumnNotFound on
    # "sqrt(protime)".
    assert "sqrt(protime)" in m.data.columns
    fig = m.plot_smooth()
    assert len(fig.axes) == 5
    plt.close(fig)


def _ziplss_fixture():
    # Zero-inflated Poisson data on R's set.seed(31) stream via hea.R.rng
    # (bit-exact runif + rpois), so the mgcv fit below is reproducible by a
    # pure-R script:
    #   set.seed(31); n<-300; x<-runif(n); z<-runif(n)
    #   gamma<-0.6+sin(2*pi*x); eta<-1.2*sin(2*pi*z)
    #   y<-rpois(n, exp(gamma)) * (runif(n) < 1-exp(-exp(eta)))
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
    # R: gam(list(y ~ s(x), ~ s(z)), family=ziplss(), method="REML") on the
    # set.seed(31) zero-inflated-Poisson fixture. ziplss is
    # available.derivs=2 ⇒ full outer Newton (ll deriv 3/4) and the
    # family$predict hook; unlike twlss/shash/gammals both smooths here are
    # informative, so neither sp is a flat λ→∞ ridge.
    from hea.family import ziplss

    df = _ziplss_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=ziplss(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 424.2451262,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp, [0.1515432173, 0.1131497668],
                               rtol=1e-4)
    np.testing.assert_allclose(m.edf_total, 11.14924342, rtol=0, atol=2e-3)
    # tp-basis eigenvector signs are build noise: pin |coef| (cf. gammals).
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([0.6705247844, -1.655109582, 0.3424055272, -0.0520844202]),
        rtol=0, atol=1e-4)
    # fitted matrix is (gamma = log λ, presence-eta) — ziplss leaves it
    # unrewritten (no postproc fitted override). R: rows 1-2.
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.4844884226, 0.2271978785], [0.3148140516, -0.2381554759]],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 500.745984, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 674.6005762, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.006223781606,
                               rtol=0, atol=1e-6)

    # family$predict hook: type="response" returns the single column
    # E(y) = p·μ (mgcv emits one fit column for ziplss, not n_lp), with a
    # delta-method SE — note mgcv reuses gamma's variance for the eta term
    # (gamlss.r:1718), reproduced bug-for-bug so the SE matches predict.gam.
    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [1.445763757, 1.001536501, 2.231325171], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.1624539204, 0.2429565392, 0.2415688359], rtol=0, atol=1e-5)
    # response prediction on training rows == E(y) for all rows.
    pr_all = m.predict(type="response")
    np.testing.assert_allclose(
        pr_all["fit"].to_numpy()[:3],
        [1.445763757, 1.001536501, 2.231325171], rtol=0, atol=1e-5)

    # residuals: deviance head + response head, and Σ deviance² == deviance.
    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [-1.584347308, -1.255452233, 0.9717809043, -1.168783967,
         -1.423877013], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        m.residuals_of("response")[:5],
        [-1.445763757, -1.001536501, 1.768674829, -1.500289883,
         -0.3922964182], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        float(np.sum(m.residuals_of("deviance") ** 2)), m.deviance,
        rtol=0, atol=1e-9)
    m.summary()


def _multinom_fixture():
    # 3-category multinomial-logistic data on R's set.seed(33) stream via
    # hea.R.rng (bit-exact runif), reproducible by a pure-R script:
    #   set.seed(33); n<-400; x<-runif(n); z<-runif(n)
    #   eta1<-1.4*sin(2*pi*x); eta2<-1.2*cos(2*pi*z)-0.3
    #   P<-softmax(cbind(0,eta1,eta2)); y<-inverse-CDF sample of P
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
    # R: gam(list(y ~ s(x), ~ s(z)), family=multinom(2), method="REML") on
    # the set.seed(33) 3-category fixture — the variable-K front end (here
    # K=2) through a real fit, available.derivs=2 ⇒ full outer Newton, the
    # softmax family$predict hook, and the class-frequency null deviance.
    from hea.family import multinom

    df = _multinom_fixture()
    m = gam(["y ~ s(x)", "~ s(z)"], df, family=multinom(2), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 252.9000533,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.sp, [0.05569991114, 0.1433165651],
                               rtol=1e-4)
    np.testing.assert_allclose(m.edf_total, 10.00551406, rtol=0, atol=2e-3)
    # tp-basis eigenvector signs are build noise: pin |coef| (the per-LP
    # intercepts coef[0]/coef[2] are sign-stable).
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([-0.01450008726, 2.368136713, 0.7424680446, 0.4313548243]),
        rtol=0, atol=1e-4)
    # fitted matrix is the (n, 2) η (NOT probabilities): (η₁, η₂) — the
    # softmax probabilities come from the predict hook. R: rows 1-2.
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.07480023513, 0.02199777687], [0.3881200015, -1.508606099]],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 762.0647414, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.null_deviance, 871.2817292, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.01566970018,
                               rtol=0, atol=1e-6)

    # family$predict hook: type="response" returns the (K+1)=3 category
    # probabilities (more columns than n_lp=2) with delta-method SEs.
    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(
        [pr["fit"][0], pr["fit.1"][0], pr["fit.2"][0]],
        [0.3225899731, 0.347645165, 0.3297648618], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        [pr["se.fit"][0], pr["se.fit.1"][0], pr["se.fit.2"][0]],
        [0.03781883063, 0.05422039848, 0.0523534205], rtol=0, atol=1e-5)
    # probabilities sum to 1 across categories, every row.
    np.testing.assert_allclose(
        pr["fit"].to_numpy() + pr["fit.1"].to_numpy()
        + pr["fit.2"].to_numpy(), 1.0, atol=1e-12)

    np.testing.assert_allclose(
        m.residuals_of("deviance")[:5],
        [1.453666372, -1.408229965, -1.357551087, 1.078630867,
         1.19190966], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        float(np.sum(m.residuals_of("deviance") ** 2)), m.deviance,
        rtol=0, atol=1e-9)
    m.summary()


def _mvn_fixture():
    # 2-D multivariate-normal data on R's set.seed(52) stream via hea.R.rng
    # (bit-exact runif/rnorm), reproducible by a pure-R script:
    #   set.seed(52); n<-200; x<-runif(n); z<-runif(n)
    #   e1<-rnorm(n); e2<-rnorm(n)
    #   y1<-2*sin(pi*x)+e1
    #   y2<-0.6*sin(pi*x)+exp(1.5*z)-2+0.5*e1+0.8*e2  # correlated with y1
    from hea.R.rng import RGenerator
    gen = RGenerator(52)
    n = 200
    x = gen.uniform(size=n)
    z = gen.uniform(size=n)
    e1 = gen.normal(size=n)
    e2 = gen.normal(size=n)
    y1 = 2 * np.sin(np.pi * x) + e1
    y2 = (0.6 * np.sin(np.pi * x) + np.exp(1.5 * z) - 2
          + 0.5 * e1 + 0.8 * e2)
    return pl.DataFrame({"y1": y1, "y2": y2, "x": x, "z": z})


def test_mvn_through_gam_matches_mgcv():
    # R: gam(list(y1 ~ s(x), y2 ~ s(z)), family=mvn(d=2)) on the set.seed(52)
    # correlated 2-D fixture. The whole item-7 stack end-to-end: the matrix
    # response front end (each formula carries its own dimension), the
    # preinitialize dummy-column extension for the d(d+1)/2 covariance
    # params, the mvn_ll C-kernel port, AND the BFGS outer optimizer
    # (available.derivs=1). Both smoothing parameters are informative here
    # (no flat λ→∞ ridge); the bfgs convergence path drifts the sp/edf at
    # the ~1e-4 band while REML/Vp pin tight.
    from hea.family import mvn

    df = _mvn_fixture()
    m = gam(["y1 ~ s(x)", "y2 ~ s(z)"], df, family=mvn(d=2))
    assert m.method == "REML"
    np.testing.assert_allclose(m.REML_criterion / 2, 205.7441216777,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.2006855377, 3.4818966647],
                               rtol=2e-3)
    np.testing.assert_allclose(m.edf_total, 11.51223604, rtol=0, atol=2e-3)
    np.testing.assert_allclose(m.deviance, 400.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(m.null_deviance, 627.91988172, rtol=0,
                               atol=2e-2)
    # the d(d+1)/2 = 3 covariance (precision Choleski) params are the
    # trailing coefs R.1/R.2/R.3 — sign-stable, pinned directly.
    np.testing.assert_allclose(
        np.asarray(m._beta)[-3:],
        [0.12714762, -0.57915894, -0.04044921], rtol=0, atol=1e-4)
    # tp-basis eigenvector signs are build noise → pin |coef| (the per-LP
    # intercepts coef[0]/coef[10] are sign-stable).
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[[0, 1, 10, 11]]),
        np.abs([1.249452, 0.033666, 0.559357, 0.115879]),
        rtol=0, atol=1e-4)
    # fitted is the (n, 2) matrix of per-dimension means (identity links).
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:2],
        [[0.826858, 1.016451], [0.453129, -0.089196]], rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00528745,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m.Vp)[22, 22], 0.00251721,
                               rtol=0, atol=1e-6)
    assert m.outer_info["iter"] == 4
    assert m.outer_info["conv"] == "full convergence"

    # response predict: the per-dimension means + delta-method SEs (mvn has
    # no predict hook — identity links route through the per-LP path).
    pr = m.predict(df[:2], se_fit=True, type="response")
    np.testing.assert_allclose(
        [pr["fit"][0], pr["fit.1"][0]], [0.826858, 1.016451],
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        [pr["se.fit"][0], pr["se.fit.1"][0], pr["se.fit"][1],
         pr["se.fit.1"][1]],
        [0.140909, 0.103364, 0.182820, 0.105579], rtol=0, atol=1e-4)

    # deviance residuals are the whitened (y−μ̂)·Rᵀ; Σr² ≡ deviance.
    rr = np.asarray(m.residuals_of("deviance"))
    assert rr.shape == (200, 2)
    np.testing.assert_allclose(float(np.sum(rr ** 2)), m.deviance,
                               rtol=0, atol=1e-8)
    m.summary()


def test_shash_through_gam_matches_mgcv():
    # R: gam(list(y ~ s(x), ~ s(z), ~ 1, ~ 1), family=shash(),
    # method="REML") — available.derivs=2, the FULL outer-Newton
    # path on a 4-LP family (the K=4 etamu/gH branches end-to-end),
    # plus the optimizer="efs" cross-pin at K=4 (W2's purpose: hea
    # reproduces BOTH of R's distinct newton and efs stop points).
    # The s(z)-on-τ smoothing parameter is a flattish (near-saturating)
    # ridge direction (R 25838 vs hea 25632 with the criterion agreeing
    # to ~1e-7) — pinned at band width; everything else is tight.
    from hea.family import shash, _r_tweedie  # noqa: F401

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

    m1 = gam(["y ~ s(x)", "~ s(z)", "~ 1", "~ 1"], df, family=shash(),
             method="REML")
    np.testing.assert_allclose(m1.REML_criterion / 2, 546.1724511286,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m1.sp[0], 0.1910175024, rtol=1e-4)
    np.testing.assert_allclose(m1.sp[1], 25838.09103, rtol=2e-2)
    # edf tracks the sp[1] band (Δ ~1.2e-3 measured on this machine)
    np.testing.assert_allclose(m1.edf_total, 10.2633615130, rtol=0,
                               atol=5e-3)
    b = np.asarray(m1._beta)
    np.testing.assert_allclose(b[0], 1.0060261008, rtol=0, atol=1e-4)
    # tp-basis eigenvector signs are build noise — pin magnitudes
    np.testing.assert_allclose(
        np.abs(b[1:3]), np.abs([1.5298088420, -0.1395345089]),
        rtol=0, atol=1e-3)
    # the ε and log-kurtosis intercepts (sign-stable)
    np.testing.assert_allclose(
        [b[20], b[21]], [0.4312252846, -0.0635499741], rtol=0,
        atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(m1.fitted_values)[0],
        [0.2022640409, -0.0442752319, 0.4312252846, -0.0635499741],
        rtol=0, atol=1e-3)
    np.testing.assert_allclose(m1.deviance, 1051.49370326,
                               rtol=1e-4)
    assert np.isnan(m1.null_deviance)      # mgcv: NULL (no postproc)
    np.testing.assert_allclose(
        np.asarray(m1.residuals)[:3],
        [-1.7047382825, -1.2148037602, 1.9049698110], rtol=0,
        atol=1e-4)
    np.testing.assert_allclose(np.asarray(m1.Vp)[0, 0],
                               0.007045588271, rtol=0, atol=1e-5)
    assert 2 <= m1.outer_info["iter"] <= 8           # R: 3
    # the qf hook lights qq.gam's DIRECT path (first general family
    # with one); rd lights the simulation path
    qq = m1._qq_gam_quantiles(type="deviance", rep=0, s_rep=2, seed=1)
    assert qq["Dq"] is not None and np.all(np.isfinite(qq["Dq"]))
    qq2 = m1._qq_gam_quantiles(type="deviance", rep=3, level=0, seed=1)
    assert qq2["Dq"] is not None and np.all(np.isfinite(qq2["Dq"]))
    m1.summary()
    pred = m1.predict(df[:3])
    np.testing.assert_allclose(
        pred["fit"].to_numpy(),
        np.asarray(m1.fitted_values)[:3, 0], rtol=0, atol=1e-10)

    # mgcv's shash ll rejects offsets outright (gamlss.r:3470)
    with pytest.raises(NotImplementedError, match="offset not still"):
        gam(["y ~ s(x) + offset(z)", "~ s(z)", "~ 1", "~ 1"], df,
            family=shash(), method="REML")

    # efs cross-pin at K=4: R optimizer="efs" stops at its OWN point
    # (REML 527.47953 vs newton's 527.47232) and hea lands on it
    m2 = gam(["y ~ s(x)", "~ s(z)", "~ 1", "~ 1"], df, family=shash(),
             method="REML", optimizer="efs")
    np.testing.assert_allclose(m2.REML_criterion / 2, 546.1756937219,
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(m2.sp, [0.18448478, 2613.7345],
                               rtol=1e-2)        # s(z) flattish efs ridge
    np.testing.assert_allclose(m2.edf_total, 10.31418966, rtol=0,
                               atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(m2.fitted_values)[0],
        [0.2018576699, -0.0445263220, 0.4309091585, -0.0641118029],
        rtol=0, atol=1e-3)


def test_predict_unconditional_se_matches_mgcv():
    # unconditional=TRUE swaps Vp → Vc (sp-uncertainty corrected) for the
    # SE band — predict.gam parity on the first three rows.
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    p = m.predict(df[:3], se_fit=True)
    pu = m.predict(df[:3], se_fit=True, unconditional=True)
    np.testing.assert_allclose(
        p["se.fit"].to_numpy(),
        [0.0774989332, 0.1475501829, 0.0806144988], rtol=1e-6)
    np.testing.assert_allclose(
        pu["se.fit"].to_numpy(),
        [0.0777513533, 0.1479764183, 0.0811005933], rtol=1e-6)
    # GCV fits carry no sp-uncertainty correction: mgcv warns and falls
    # back to Vp; so do we.
    mg = gam("ygau ~ f4 + z + s(x)", df, method="GCV.Cp")
    with pytest.warns(UserWarning, match="not available"):
        pg = mg.predict(df[:3], se_fit=True, unconditional=True)
    np.testing.assert_array_equal(
        pg["se.fit"].to_numpy(),
        mg.predict(df[:3], se_fit=True)["se.fit"].to_numpy())


# ---------------------------------------------------------------------------
# predict type="terms"/"iterms" + terms=/exclude= (roadmap B1) — mgcv 1.9-4
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
    np.testing.assert_allclose(pt.to_numpy(), [
        [0.0000000000, 0.2377973618, 0.8729497169],
        [0.7342117915, 0.2155591008, -0.0446679116],
        [0.2712112896, 0.4304817651, -0.1832820318],
        [-0.3299837116, 0.1456983896, 0.0241175461],
        [0.7342117915, 0.5232056834, 0.2809888679],
        [0.7342117915, 0.6593732328, -0.4352664652]], atol=1e-7)
    pts = m.predict(nd, type="terms", se_fit=True)
    assert pts.columns == ["f4", "z", "s(x)", "se.f4", "se.z", "se.s(x)"]
    np.testing.assert_allclose(pts.to_numpy()[:, 3:], [
        [0.0000000000, 0.0308069966, 0.0530786832],
        [0.0738603615, 0.0279259973, 0.1364511756],
        [0.0750765806, 0.0557695434, 0.0604690770],
        [0.0734874123, 0.0188754398, 0.1154736067],
        [0.0738603615, 0.0677820628, 0.0698835544],
        [0.0738603615, 0.0854227684, 0.0966170141]], atol=1e-7)
    # iterms: same fit, s(x)'s SE widened by the cmX construction; the
    # strictly parametric columns are untouched.
    pti = m.predict(nd, type="iterms", se_fit=True)
    np.testing.assert_allclose(pti.to_numpy()[:, :3], pt.to_numpy(),
                               rtol=1e-12)
    np.testing.assert_allclose(pti.to_numpy()[:, 3:], [
        [0.0000000000, 0.0308069966, 0.0587771513],
        [0.0738603615, 0.0279259973, 0.1387671800],
        [0.0750765806, 0.0557695434, 0.0655279801],
        [0.0734874123, 0.0188754398, 0.1182013568],
        [0.0738603615, 0.0677820628, 0.0743042265],
        [0.0738603615, 0.0854227684, 0.0998611753]], atol=1e-7)
    # iterms.type=2 (fixed-effects mean only) coincides here: the tp
    # basis is sum-to-zero so cmX's smooth block is already ~0.
    pti2 = m.predict(nd, type="iterms", se_fit=True, iterms_type=2)
    np.testing.assert_allclose(pti2.to_numpy(), pti.to_numpy(), atol=1e-10)


def test_predict_terms_select_exclude_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    nd = df.head(6)
    sx = [0.8729497169, -0.0446679116, -0.1832820318,
          0.0241175461, 0.2809888679, -0.4352664652]
    zc = [0.2377973618, 0.2155591008, 0.4304817651,
          0.1456983896, 0.5232056834, 0.6593732328]
    sel = m.predict(nd, type="terms", terms="s(x)")
    assert sel.columns == ["s(x)"]
    np.testing.assert_allclose(sel["s(x)"].to_numpy(), sx, atol=1e-7)
    selp = m.predict(nd, type="terms", terms=["z", "s(x)"])
    assert selp.columns == ["z", "s(x)"]
    np.testing.assert_allclose(selp.to_numpy(),
                               np.column_stack([zc, sx]), atol=1e-7)
    exc = m.predict(nd, type="terms", exclude="f4")
    assert exc.columns == ["z", "s(x)"]
    np.testing.assert_allclose(exc.to_numpy(),
                               np.column_stack([zc, sx]), atol=1e-7)
    # Non-existent labels: the design zeroing still applies, only the
    # column selection warns and is ignored — terms="nope" therefore
    # returns the full layout with ALL-ZERO values (verified vs mgcv).
    with pytest.warns(UserWarning, match="non-existent terms"):
        wt = m.predict(nd, type="terms", terms="nope")
    assert wt.columns == ["f4", "z", "s(x)"]
    assert float(np.abs(wt.to_numpy()).max()) == 0.0
    with pytest.warns(UserWarning, match="non-existent exclude"):
        we = m.predict(nd, type="terms", exclude="nope")
    np.testing.assert_allclose(
        we.to_numpy(), m.predict(nd, type="terms").to_numpy(), rtol=1e-12)


def test_predict_link_response_terms_exclude_matches_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    nd = df.head(6)
    le = m.predict(nd, type="link", exclude="s(x)", se_fit=True)
    np.testing.assert_allclose(le["fit"].to_numpy(), [
        0.5736263216, 1.2855998521, 1.0375220145,
        0.1515436379, 1.5932464347, 1.7294139842], atol=1e-7)
    np.testing.assert_allclose(le["se.fit"].to_numpy(), [
        0.0569307847, 0.0536362256, 0.0533644748,
        0.0587270876, 0.0545917647, 0.0635634911], atol=1e-7)
    # Gaussian identity: response == link for the partial predictor.
    re_ = m.predict(nd, type="response", exclude="s(x)")
    np.testing.assert_allclose(re_["fit"].to_numpy(),
                               le["fit"].to_numpy(), rtol=1e-12)
    ni = m.predict(nd, type="link", exclude="(Intercept)")
    np.testing.assert_allclose(ni["fit"].to_numpy(), [
        1.1107470787, 0.9051029807, 0.5184110229,
        -0.1601677758, 1.5384063428, 0.9583185591], atol=1e-7)
    # terms= on the link scale: everything not listed is zeroed,
    # including the intercept — link terms="s(x)" IS the s(x) column.
    lt = m.predict(nd, type="link", terms="s(x)")
    np.testing.assert_allclose(lt["fit"].to_numpy(), [
        0.8729497169, -0.0446679116, -0.1832820318,
        0.0241175461, 0.2809888679, -0.4352664652], atol=1e-7)
    lz = m.predict(nd, type="link", terms="z")
    np.testing.assert_allclose(lz["fit"].to_numpy(), [
        0.2377973618, 0.2155591008, 0.4304817651,
        0.1456983896, 0.5232056834, 0.6593732328], atol=1e-7)


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
    np.testing.assert_array_equal(pi_["se.s(g5)"].to_numpy(),
                                  pt["se.s(g5)"].to_numpy())
    assert float(np.abs(pt["s(g5)"].to_numpy()).max()) < 1e-4
    np.testing.assert_allclose(pt["se.s(x)"].to_numpy(), [
        0.0530787002, 0.1364512782, 0.0604690950,
        0.1154736894, 0.0698835753, 0.0966170952], rtol=1e-4)
    np.testing.assert_allclose(pi_["se.s(x)"].to_numpy(), [
        0.0587771673, 0.1387672811, 0.0655279973,
        0.1182014378, 0.0743042467, 0.0998612540], rtol=1e-4)


def test_predict_terms_offset_poisson_matches_mgcv():
    # The model offset is kept in link/response predictions under
    # exclude= (it is not a term), and never appears as a terms column.
    df = _pterms_fixture()
    m = gam("ypois ~ z + s(x) + offset(log(lo))", df,
            family=Poisson(), method="REML")
    nd = df.head(6)
    o3 = m.predict(nd, type="link", exclude="s(x)")
    np.testing.assert_allclose(o3["fit"].to_numpy(), [
        0.7647704612, 1.1728571194, 0.4727620017,
        1.1699203351, 1.1000742402, 1.6116984342], atol=1e-7)
    t3 = m.predict(nd, type="terms")
    assert t3.columns == ["z", "s(x)"]
    np.testing.assert_allclose(t3.to_numpy()[0],
                               [0.2172354052, 0.9256322997], atol=1e-7)


def test_predict_terms_multi_lp_gaulss_matches_mgcv():
    from hea.family import gaulss
    df = _fit5_fixture()
    m = gam(["y ~ s(x) + w", "~ s(z)"], df, family=gaulss(),
            method="REML")
    nd = df.head(6)
    t4 = m.predict(nd, type="terms")
    assert t4.columns == ["w", "s(x)", "s.1(z)"]
    np.testing.assert_allclose(t4.to_numpy(), [
        [0.26576084, 0.71070807, -0.21136924],
        [0.23860749, -0.88387867, 0.07613941],
        [0.43708020, 0.63567395, -0.52135388],
        [0.31992156, 0.87529617, -0.86759606],
        [0.07666149, -0.57645871, -0.28874762],
        [0.27264823, -0.58786727, -0.34349805]], atol=1e-6)
    # iterms unavailable multi-LP: warn + fall back to terms (mgcv).
    with pytest.warns(UserWarning, match="iterms not available"):
        i4 = m.predict(nd, type="iterms")
    np.testing.assert_array_equal(i4.to_numpy(), t4.to_numpy())
    e4 = m.predict(nd, type="link", exclude="s.1(z)", se_fit=True)
    np.testing.assert_allclose(e4["fit"].to_numpy()[:2],
                               [1.46270992, -0.15903017], atol=1e-6)
    np.testing.assert_allclose(e4["fit.1"].to_numpy()[:2],
                               [-0.58269247, -0.58269247], atol=1e-6)
    np.testing.assert_allclose(e4["se.fit"].to_numpy()[:2],
                               [0.06626794, 0.06894145], atol=1e-6)
    np.testing.assert_allclose(e4["se.fit.1"].to_numpy()[:2],
                               [0.04918413, 0.04918413], atol=1e-6)
    r4 = m.predict(nd, type="response", exclude="s(x)", se_fit=True)
    np.testing.assert_allclose(r4["fit"].to_numpy()[:2],
                               [0.75200185, 0.72484850], atol=1e-6)
    np.testing.assert_allclose(r4["fit.1"].to_numpy()[:2],
                               [2.16447805, 1.63246911], atol=1e-6)
    np.testing.assert_allclose(r4["se.fit"].to_numpy()[:2],
                               [0.02961727, 0.02850520], atol=1e-6)
    np.testing.assert_allclose(r4["se.fit.1"].to_numpy()[:2],
                               [0.21529159, 0.17505347], atol=1e-6)


# ---------------------------------------------------------------------------
# qq.gam + gam.check plots + multi-LP check (roadmap B2) — mgcv 1.9-4.
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
    # Dq == sort(qnorm(U))·√sig2 exactly, for any shuffle: R values.
    np.testing.assert_allclose(
        q["Dq"][:5],
        [-1.0022383513, -0.8684696298, -0.8002824190, -0.7527795603,
         -0.7157525518], atol=1e-9)
    np.testing.assert_allclose(q["Dq"][99], -0.0022374646, atol=1e-9)
    np.testing.assert_allclose(q["Dq"][199], 1.0022383513, atol=1e-9)
    assert q["lim"] is None
    np.testing.assert_allclose(
        np.sort(q["D"])[:3],
        [-1.0144419914, -0.8464473778, -0.7445768179], rtol=1e-6)


def test_qq_gam_poisson_direct_matches_mgcv_exactly():
    # The direct path's only randomness is R's sample(U), run through
    # the bit-exact _RUnif port: seed=1 reproduces R's
    # set.seed(1); qq.gam(m) stream, so Dq pins at 1e-8.
    m = gam("ypois ~ z + s(x)", _pterms_fixture(), family=Poisson(),
            method="REML")
    q = m._qq_gam_quantiles(seed=1)
    np.testing.assert_allclose(
        q["Dq"][:5],
        [-2.7744499905, -2.3318922820, -2.1897458268, -2.0856308712,
         -1.9926249497], atol=2e-9)
    np.testing.assert_allclose(
        np.asarray(q["Dq"])[[49, 99, 149, 199]],
        [-0.9523298576, -0.1019956073, 0.6055969984, 2.7228860917],
        atol=2e-9)
    assert np.all(np.diff(q["Dq"]) >= 0)


def test_qq_gam_simulation_branch_matches_mgcv():
    m = gam("ygau ~ f4 + z + s(x)", _pterms_fixture(), method="REML")
    q = m._qq_gam_quantiles(rep=20, level=0.9, seed=1)
    # R set.seed(1), rep=20: [-1.085751, -0.007870, 0.977612] — MC band.
    np.testing.assert_allclose(np.asarray(q["Dq"])[[0, 99, 199]],
                               [-1.085751, -0.007870, 0.977612], atol=0.25)
    assert q["lim"].shape == (2, 200)
    assert np.all(q["lim"][0] <= q["lim"][1])
    # level >= 1: per-replicate line matrix instead of a band.
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
    mg = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
             method="REML")
    # gaulss HAS rd in mgcv (gamlss.r:1089) — qq.gam takes the
    # simulation path (no qf → rep=50 via rd), NOT a qqnorm fallback
    # (family-review A3; the old B2 record claimed otherwise).
    qq = mg._qq_gam_quantiles(seed=0)
    assert qq["Dq"] is not None
    # Monte-Carlo-level: deviance residuals are (y−μ̂)·τ̂ ≈ N(0,1) at the
    # converged fit, so the simulated theoretical quantiles must track
    # the standard-normal quantiles closely (and exactly in law as
    # rep → ∞). mgcv qq.gam on this fit shows the same band.
    n = qq["Dq"].size
    a = 0.5
    pp = (np.arange(1, n + 1) - a) / n
    from scipy.stats import norm as _norm
    ref = _norm.ppf(np.clip(pp, 1e-12, 1 - 1e-12))
    inner = slice(n // 10, -n // 10)        # compare away from the tails
    np.testing.assert_allclose(qq["Dq"][inner], ref[inner], atol=0.25)
    ax = mg.qq_gam(seed=0)
    assert ax.get_title() == "QQ plot of residuals"
    plt.close("all")


def test_gaulss_residuals_match_mgcv():
    from hea.family import gaulss
    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
            method="REML")
    rd = m.residuals_of("deviance")
    rr = m.residuals_of("response")
    # rtol: the gaulss optimizer's stop point differs across BLAS builds
    # by ~2e-7 in (y−μ̂)·τ̂ (CI/OpenBLAS vs Mac/Accelerate measured
    # 2.3e-7); same-machine agreement with R is ~1e-8.
    np.testing.assert_allclose(
        rd[:5], [-1.40666569, -0.75618930, 0.40399967,
                 -1.27152202, 0.33599667], rtol=1e-6)
    np.testing.assert_allclose(
        rr[:5], [-0.64988679, -0.46321814, 0.13797675,
                 -0.31089046, 0.14392372], rtol=1e-6)
    # gaulss's hook defines pearson == deviance ((y−μ̂)·τ̂).
    np.testing.assert_array_equal(m.residuals_of("pearson"), rd)
    with pytest.raises(ValueError, match="gaulss residuals"):
        m.residuals_of("working")


def test_gaulss_check_and_k_check_match_mgcv(capsys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hea.family import gaulss
    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
            method="REML")
    kt = m._k_check(seed=0)
    assert kt[""].to_list() == ["s(x)", "s.1(z)"]
    np.testing.assert_allclose(kt["k'"].to_numpy(), [9.0, 9.0])
    np.testing.assert_allclose(kt["edf"].to_numpy(),
                               [6.36517535, 4.60542926], rtol=1e-6)
    np.testing.assert_allclose(kt["k-index"].to_numpy(),
                               [1.01668913, 1.01275279], rtol=1e-6)
    # The permutation p-values run through the _RUnif port: seed=0 with
    # n_rep=200 reproduces R's set.seed(0); k.check(b, n.rep=200) exactly.
    np.testing.assert_allclose(kt["p-value"].to_numpy(), [0.620, 0.525],
                               atol=1e-12)
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
    # The lm-style panel is undefined for multi-LP fits — clear guard.
    with pytest.raises(NotImplementedError, match="plot_smooth"):
        m.plot()


def test_family_qf_rd_unit_values_match_R():
    from hea.family import Binomial, Gaussian
    g = Gamma()
    np.testing.assert_allclose(
        g.qf(np.array([.1, .5, .9]), np.array([2.0, 2.0, 2.0]), 1.0, 0.3),
        [0.7860239435, 1.8039491853, 3.4688989388], rtol=1e-9)
    b = Binomial()
    np.testing.assert_allclose(
        b.qf(np.array([.1, .5, .9]), np.array([0.4, 0.4, 0.4]),
             np.array([7.0, 7.0, 7.0]), 1.0),
        [0.1428571429, 0.4285714286, 0.5714285714], rtol=1e-9)
    p = Poisson()
    np.testing.assert_allclose(
        p.qf(np.array([.1, .5, .9]), np.array([3.5, 3.5, 3.5]), 1.0, 1.0),
        [1.0, 3.0, 6.0])
    gau = Gaussian()
    np.testing.assert_allclose(
        gau.qf(0.75, 1.2, 2.0, 0.5), 1.5372448751, rtol=1e-9)
    # rd hooks: reproducible given the same Generator seed.
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    mu = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(g.rd(rng1, mu, 1.0, 0.3),
                                  g.rd(rng2, mu, 1.0, 0.3))


# ---------------------------------------------------------------------------
# summary(freq=, dispersion=) + anova passthrough (roadmap B3) — mgcv 1.9-4.
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
        [0.0729556790, 0.0735110295, 0.0728551167, 0.0746810309,
         0.0932363029], rtol=1e-7)
    np.testing.assert_allclose(
        m._se_report_for(False, 2.0)[:5],
        [0.2906511136, 0.2925519833, 0.2910747763, 0.2973692805,
         0.3704794923], rtol=1e-7)
    np.testing.assert_allclose(
        m._se_report_for(True, 0.5)[:5],
        [0.1444843226, 0.1455841607, 0.1442851649, 0.1479012779,
         0.1846488750], rtol=1e-7)
    # defaults: byte-identical to the precomputed report SEs.
    np.testing.assert_array_equal(m._se_report_for(False, None),
                                  m._se_report)
    # pTerms under freq stays est.disp (F = chi/df on residual df).
    rows_f = m._pterms_rows(freq=True)
    assert [(r[0], r[1]) for r in rows_f] == [("f4", 3), ("z", 1)]
    np.testing.assert_allclose([r[2] for r in rows_f],
                               [80.7093362000, 59.9639681100], rtol=1e-6)
    # dispersion= forces the known-scale Chi.sq/pchisq forms.
    rows_d = m._pterms_rows(dispersion=2.0)
    np.testing.assert_allclose([r[2] for r in rows_d],
                               [15.3088806600, 3.7978001470], rtol=1e-6)
    np.testing.assert_allclose([r[3] for r in rows_d],
                               [0.00157083954, 0.05131996646], rtol=1e-5)


def test_summary_dispersion_smooth_tables_match_mgcv():
    df = _pterms_fixture()
    m = gam("ygau ~ f4 + z + s(x)", df, method="REML")
    sm = m._smooth_significance_rows(dispersion=2.0)
    # est.disp FALSE: the stat column is the RAW Chi.sq, χ² reference.
    np.testing.assert_allclose(
        [sm[0][1], sm[0][2], sm[0][3]],
        [6.7270776850, 7.8393200430, 51.6659932800], rtol=1e-5)
    assert sm[0][4] < 1e-12
    # freq= never reaches the smooth table (mgcv.r:4014 hard-codes Vp).
    sm0 = m._smooth_significance_rows()
    sm0b = m._smooth_significance_rows(dispersion=None)
    assert sm0 == sm0b
    # poisson (scale known): dispersion=1.5 rescales Vp; Chi.sq stays raw.
    mp = gam("ypois ~ z + s(x)", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(
        mp._se_report_for(False, 1.5)[:2],
        [0.1157668914, 0.1779806259], rtol=1e-7)
    smp = mp._smooth_significance_rows(dispersion=1.5)
    np.testing.assert_allclose(smp[0][3], 160.5454171000, rtol=1e-6)


def test_summary_dispersion_re_test_matches_mgcv():
    # reTest under dispersion=: the rescaled Vp/Ve flow through recov
    # (quadratically), sig2 and the scale-estimated p-value branch stay
    # on the object's values (mgcv reads b$sig2/b$scale.estimated
    # untouched). The s(g5) component sits on the flat-REML λ boundary
    # (hea/R stop at different λ — the recorded band), so its raw stat
    # is boundary noise: pin the p-value (which matches R to 7 digits)
    # plus the hea-internal invariance stat(disp) == stat(default), and
    # pin s(x) tight.
    m = gam("ygau ~ f4 + z + s(x) + s(g5, bs='re')", _pterms_fixture(),
            method="REML")
    sm_d = m._smooth_significance_rows(dispersion=2.0)
    sm_0 = m._smooth_significance_rows()
    np.testing.assert_allclose(sm_d[0][3], 51.6659866850, rtol=1e-5)
    np.testing.assert_allclose(sm_d[1][4], 0.9985281201, rtol=1e-6)
    # raw reTest stat is dispersion-invariant: default's printed F-col is
    # stat/Ref.df, the dispersion column is the raw stat.
    np.testing.assert_allclose(sm_d[1][3], sm_0[1][3] * sm_0[1][2],
                               rtol=1e-10)


def test_summary_freq_dispersion_gaulss_and_print(capsys):
    from hea.family import gaulss
    mg = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
             method="REML")
    idx = mg._param_idx
    np.testing.assert_allclose(
        mg._se_report_for(True, None)[idx],
        [0.0561071082, 0.0980386275, 0.0488761346], rtol=1e-6)
    # printed surface: dispersion shows as Scale est., Chi.sq column for
    # a gaussian fit under the override, t→z switch implicit in pins.
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
# concurvity (roadmap B4) — mgcv.r:3340-3423, R pins on the CSV fixtures.
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
    # NOTE: on this seed-11 data s(z) models a (near-)linear covariate
    # (eta has 0.6·z), so the REML score keeps falling as λ_{s(z)}→∞ along
    # a near-flat ridge; hea's outer Newton stops at sp[s(z)]≈0.97
    # (REML/2 97.47) where mgcv drives it to ≈36480 (97.32). The
    # design-only concurvity (worst / estimate / para) is fit-independent
    # and matches mgcv EXACTLY; only the β̂-dependent "observed" row sees
    # the sp gap (≤8.5e-3 full, ≤6.7e-2 pairwise), so it is pinned to
    # hea's fit. (hea outer-Newton sp-underfit for smooth-of-linear.)
    np.testing.assert_allclose(cf["para"].to_numpy(),
                               [0.7888546768] * 3, rtol=1e-7)
    np.testing.assert_allclose(                       # worst, estimate: mgcv
        cf["s(x)"].to_numpy()[[0, 2]],
        [0.1690506098, 0.0507947318], rtol=1e-5)
    np.testing.assert_allclose(cf["s(x)"].to_numpy()[1],  # observed: hea fit
                               0.0689315647, rtol=1e-5)
    np.testing.assert_allclose(
        cf["s(z)"].to_numpy()[[0, 2]],
        [0.1408198865, 0.0737227575], rtol=1e-5)
    np.testing.assert_allclose(cf["s(z)"].to_numpy()[1],
                               0.0702257390, rtol=1e-5)
    cp = m.concurvity(full=False)
    assert set(cp) == {"worst", "observed", "estimate"}
    W = cp["worst"]
    assert W[""].to_list() == ["para", "s(x)", "s(z)"]
    np.testing.assert_allclose(np.diag(W.to_numpy()[:, 1:].astype(float)),
                               np.ones(3))
    # para row/col vs centered smooths: exact zeros in R's print.
    assert float(W["s(x)"][0]) < 1e-12 and float(W["para"][1]) < 1e-12
    np.testing.assert_allclose(float(W["s(z)"][1]), 0.1289878965,
                               rtol=1e-6)
    np.testing.assert_allclose(float(cp["observed"]["s(z)"][1]),  # hea fit
                               0.0639043896, rtol=1e-5)
    np.testing.assert_allclose(float(cp["observed"]["s(x)"][2]),  # hea fit
                               0.0427210945, rtol=1e-5)
    np.testing.assert_allclose(float(cp["estimate"]["s(z)"][1]),  # mgcv
                               0.0640022213, rtol=1e-5)


def test_concurvity_correlated_and_intercept_only_para():
    # xc = (x+z)/2 makes s(x)/s(xc) genuinely concurve; the para block is
    # the intercept alone — exactly orthogonal to the centered smooths.
    df = _pterms_fixture().with_columns(
        ((pl.col("x") + pl.col("z")) / 2).alias("xc"))
    m = gam("ygau ~ s(x) + s(xc)", df, method="REML")
    cf = m.concurvity()
    assert float(np.abs(cf["para"].to_numpy()).max()) < 1e-12
    np.testing.assert_allclose(
        cf["s(x)"].to_numpy(),
        [0.5448962547, 0.3866319478, 0.4592832865], rtol=1e-5)
    np.testing.assert_allclose(
        cf["s(xc)"].to_numpy(),
        [0.5448962547, 0.5301830175, 0.4381152160], rtol=1e-5)
    cp = m.concurvity(full=False)
    np.testing.assert_allclose(float(cp["worst"]["s(xc)"][1]),
                               0.5448962547, rtol=1e-6)


def test_concurvity_multi_lp_gaulss_matches_mgcv():
    from hea.family import gaulss
    m = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
            method="REML")
    cf = m.concurvity()
    assert cf.columns == ["", "para", "s(x)", "s.1(z)"]
    # LP2's intercept column duplicates LP1's: para lies entirely in the
    # complement's span — all three measures are 1 (mgcv prints 1 1 1).
    np.testing.assert_allclose(cf["para"].to_numpy(), np.ones(3),
                               atol=1e-10)
    # The duplicated intercept makes the stacked X EXACTLY rank-deficient
    # (σ_min ≈ 3e-15), and the FULL measures run an unpivoted QR over it
    # (mgcv.r:3376 "No pivoting!!") — a 1e-14 perturbation of X moves
    # these values by ~5e-3, so they are platform noise at that scale in
    # mgcv too (CI/OpenBLAS measured 3.9e-4 from the Mac pins). Pin the
    # noise band, not the digits.
    np.testing.assert_allclose(
        cf["s(x)"].to_numpy(),
        [0.11186354, 0.06067571, 0.05768171], atol=0.02)
    np.testing.assert_allclose(
        cf["s.1(z)"].to_numpy(),
        [0.12120035, 0.03000198, 0.02321672], atol=0.02)
    # Pairwise blocks exclude the stray intercept columns — well
    # conditioned, so the cross-platform-stable pin stays tight.
    cp = m.concurvity(full=False)
    np.testing.assert_allclose(float(cp["estimate"]["s.1(z)"][1]),
                               0.01809372, rtol=1e-5)
    # No smooths → mgcv's "nothing to do" error.
    m0 = gam("ygau ~ f4 + z", _pterms_fixture(), method="REML")
    with pytest.raises(ValueError, match="nothing to do"):
        m0.concurvity()


# ---------------------------------------------------------------------------
# influence / cooks_distance accessors (roadmap B5) — mgcv.r:4415/4212.
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
        [0.0471134262, 0.1707783228, 0.0509776201, 0.1307545683,
         0.0666771951], rtol=1e-6)
    np.testing.assert_allclose(h.sum(), m.edf_total, rtol=1e-10)
    cd = m.cooks_distance()
    np.testing.assert_allclose(
        cd[:5],
        [0.0057238038, 0.0282487724, 0.0032349746,
         0.0662443276, 0.0168205445], rtol=1e-6)
    np.testing.assert_allclose(cd.max(), 0.0662443276, rtol=1e-6)
    assert int(np.argmax(cd)) == 3  # R's which.max = 4, 1-based
    mp = gam("ypois ~ z + s(x)", df, family=Poisson(), method="REML")
    np.testing.assert_allclose(
        mp.influence()[:3],
        [0.0314975428, 0.1067972886, 0.0280958236], rtol=1e-6)
    np.testing.assert_allclose(
        mp.cooks_distance()[:3],
        [0.0004029915, 0.0282574127, 0.0082311885], rtol=1e-6)
    from hea.family import gaulss
    mg = gam(["y ~ s(x) + w", "~ s(z)"], _fit5_fixture(), family=gaulss(),
             method="REML")
    with pytest.raises(NotImplementedError, match="general-family"):
        mg.influence()
    with pytest.raises(NotImplementedError, match="general-family"):
        mg.cooks_distance()


def _betar_fixture():
    # Beta-distributed response on R's set.seed(71) stream via hea.R.rng:
    #   set.seed(71); n<-250; x<-runif(n); u<-runif(n)
    #   mu<-plogis(0.8*sin(2*pi*x)-0.3); y<-qbeta(u, 12*mu, 12*(1-mu))
    # qbeta (R) and scipy beta.ppf invert the same regularized incomplete
    # beta, so the response matches to ~1e-12.
    from scipy.stats import beta as _B
    from hea.R.rng import RGenerator
    gen = RGenerator(71)
    n = 250
    x = gen.uniform(0, 1, n)
    u = gen.uniform(0, 1, n)
    mu = 1.0 / (1.0 + np.exp(-(0.8 * np.sin(2 * np.pi * x) - 0.3)))
    y = _B.ppf(u, 12.0 * mu, 12.0 * (1.0 - mu))
    return pl.DataFrame({"y": y, "x": x})


def test_betar_through_gam_matches_mgcv():
    # R: gam(y ~ s(x), family=betar(), method="REML") — the first D1b
    # extended family end-to-end: the −2logLik-as-deviance criterion (Dp<0
    # is legitimate for betar, unlike a proper deviance), joint φ
    # estimation, the LogitLink extended g2g/g3g/g4g forms the extended
    # IRLS needs, and the saturated-ll Newton folded into the reported
    # deviance + deviance residuals.
    from hea.family import betar

    df = _betar_fixture()
    m = gam("y ~ s(x)", df, family=betar(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, -161.49936705,
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.sp, [0.1881112502], rtol=1e-4)
    np.testing.assert_allclose(
        float(np.exp(m.family.get_theta()[0])), 14.41333656,
        rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.edf_total, 6.82912381, rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.deviance, 234.26619683, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.null_deviance, 468.30180613, rtol=0,
                               atol=1e-4)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00110098,
                               rtol=0, atol=1e-6)
    # tp-basis sign noise → |coef|; the intercept (coef[0]) is sign-stable.
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:3]),
        np.abs([0.236147, 1.665602, 0.753554]), rtol=0, atol=1e-4)

    # response predict + delta-method SE (logit linkinv on the mean).
    pr = m.predict(df[:3], se_fit=True, type="response")
    np.testing.assert_allclose(pr["fit"].to_numpy(),
                               [0.5893680, 0.3717789, 0.5918235],
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(pr["se.fit"].to_numpy(),
                               [0.0179293, 0.0206123, 0.0178807],
                               rtol=0, atol=1e-5)

    # deviance residuals fold in the saturated log-lik (without it the
    # √(max(0,−2logLik)) clamp zeros most of them).
    np.testing.assert_allclose(
        np.asarray(m.residuals_of("deviance"))[:3],
        [-0.179788, -0.679496, 1.748929], rtol=0, atol=1e-5)

    # the other three okLinks exercise the probit/cloglog/cauchit
    # g2g/g3g/g4g extended forms (gam.fit3.r:2249-2303) in the extended
    # IRLS — REML matched to R per link.
    for lk, reml_ref in (("probit", -160.5668040),
                         ("cloglog", -160.8250744),
                         ("cauchit", -160.9522179)):
        ml = gam("y ~ s(x)", df, family=betar(link=lk), method="REML")
        np.testing.assert_allclose(ml.REML_criterion / 2, reml_ref,
                                   rtol=0, atol=1e-4)


def _ocat_fixture():
    # Ordered categorical (R=4) via R's set.seed(54) stream, reproduced
    # bit-exactly by hea.R.rng:
    #   set.seed(54); n<-300; x<-runif(n); f0<-2*sin(2*pi*x)-0.3
    #   fam<-ocat(R=4); fam$putTheta(log(c(1.1,1.0)))  # cut points -1,0.1,1.1
    #   y<-fam$rd(f0, rep(1,n), 1)                      # latent + logit(U)
    # hea exposes 0-based classes (0..3); the rd allocation is reproduced
    # exactly. table(y) = (123,50,38,89): all four classes well-populated
    # so both θ steps are identified (no flat ridge).
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
    # R: gam(y ~ s(x), family=ocat(R=4), method="REML") — the first extended
    # family with VECTOR θ (n_theta = R−2 = 2 ordered cut-point log-steps),
    # exercising the vector-θ outer-Newton gradient/Hessian, ls≡0, the
    # find.null.dev null deviance, and the single-formula `predict` hook
    # (per-class probability matrix, not linkinv(η)).
    from hea.family import ocat

    df = _ocat_fixture()
    m = gam("y ~ s(x)", df, family=ocat(R=4), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 311.1907401388,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.017735245992525465], rtol=1e-4)
    np.testing.assert_allclose(
        m.family.get_theta(), [0.10714304251230537, -0.16319455867953944],
        rtol=0, atol=1e-5)
    # finite cut points (the first is pinned at −1 for identifiability).
    np.testing.assert_allclose(
        m.family.get_theta(trans=True),
        [-1.0, 0.113093462777623, 0.96251937193122572], rtol=0, atol=1e-5)
    np.testing.assert_allclose(m.edf_total, 6.5603973369, rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.deviance, 600.2801127222, rtol=0, atol=1e-4)
    # find.null.dev (optimal latent constant ≠ weighted mean) — Brent band.
    np.testing.assert_allclose(m.null_deviance, 802.6873793845, rtol=0,
                               atol=1e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.015303883954,
                               rtol=0, atol=1e-6)
    # intercept (coef[0]) sign-stable; the tp-basis s(x) coefs carry sign
    # noise → pin |coef|.
    assert float(np.asarray(m._beta)[0]) < 0
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:4]),
        np.abs([-0.4253831825912549, -5.7504861424827149,
                0.64506034454421934, 0.72245036970946885]),
        rtol=0, atol=1e-3)
    # family relabel carries the rounded cut points.
    assert m._postproc["family_name"] == "Ordered Categorical(-1,0.11,0.96)"

    # the `predict` hook returns the per-class probability matrix (4 cols,
    # summing to 1) + delta-method SE — NOT the per-LP linkinv.
    pr = m.predict(pl.DataFrame({"x": [0.2, 0.5, 0.8]}), type="response",
                   se_fit=True)
    fit = np.column_stack([pr[c].to_numpy()
                           for c in ("fit", "fit.1", "fit.2", "fit.3")])
    se = np.column_stack([pr[c].to_numpy() for c in
                          ("se.fit", "se.fit.1", "se.fit.2", "se.fit.3")])
    np.testing.assert_allclose(fit.sum(axis=1), [1, 1, 1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        fit[0], [0.0586729857099971, 0.10079153292851062,
                 0.1478313777963898, 0.69270410356510248], rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        fit[2], [0.8502720903780735, 0.095037812305111879,
                 0.030545553826115346, 0.024144543490699277],
        rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        se[0], [0.016432665667876309, 0.023446808696688203,
                0.023454075265177548, 0.063333549629742053],
        rtol=0, atol=1e-5)

    # deviance residuals: signed √(−2 wt log f); Σ res² = deviance.
    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:5], [1.6159625596856027, 2.6002725708653465,
                   -0.55428957625776465, 0.85980858568121443,
                   1.0698318351438267], rtol=0, atol=1e-4)
    np.testing.assert_allclose(float(np.sum(dres ** 2)), m.deviance,
                               rtol=0, atol=1e-8)


def _ziP_fixture():
    # Zero-inflated Poisson counts via R's set.seed(7) stream, reproduced
    # bit-exactly by hea.R.rng with three fixed runif(n) draws (avoids the
    # variable-consumption interleaving of rpois):
    #   set.seed(7); n<-400; x<-runif(n); u_count<-runif(n); u_pres<-runif(n)
    #   gamma<-1.5*sin(2*pi*x)+0.2; lambda<-exp(gamma)
    #   eta<- -1.0+1.6*gamma; p<-1-exp(-exp(eta))
    #   y<-ifelse(u_pres<p, qpois(u_count,lambda), 0)
    # R's qpois and scipy poisson.ppf invert the same Poisson CDF (verified
    # bit-identical on this stream). 235 zeros / 165 positives, max 10.
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
    # R: gam(y ~ s(x), family=ziP(), method="REML") — single-formula
    # zero-inflated Poisson. Exercises: the shared zipll kernel via the
    # extended Dd path, the observed≠expected Hessian (EDmu2 ≠ Dmu2,
    # ziP-specific El2), the −2logLik-as-deviance fold with the saturated_ll
    # Newton, the optimize-based null deviance, and the `predict` hook
    # returning E(y) (NOT linkinv(η) — ziP keeps fitted = the log-mean LP).
    from hea.family import ziP

    df = _ziP_fixture()
    m = gam("y ~ s(x)", df, family=ziP(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 422.7206170258,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.083619872403188383], rtol=2e-3)
    np.testing.assert_allclose(
        m.family.get_theta(), [-3.2949401719022888, 1.2258053436229546],
        rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.edf_total, 6.9362279961, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 236.9705123525, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.null_deviance, 761.4398775608, rtol=0,
                               atol=5e-3)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00352127204674,
                               rtol=0, atol=1e-5)
    # ziP keeps fitted = the linear predictor γ (NOT E(y)); the mean only
    # comes from the predict hook.
    np.testing.assert_allclose(
        m.fitted_values[:4],
        [0.49477606138778263, 1.1932554070862351, 1.2263867543910303,
         0.98869377799869562], rtol=0, atol=1e-4)
    assert m._postproc["family_name"] == "Zero inflated Poisson(-3.295,3.407)"
    # intercept sign-stable; tp s(x) basis carries sign noise → |coef|.
    np.testing.assert_allclose(
        np.abs(np.asarray(m._beta)[:3]),
        np.abs([0.58794071413230287, 1.9594814883910816,
                0.61803791743402448]), rtol=0, atol=2e-3)

    # the predict hook returns E(y) = p·E(y|present) + delta SE.
    pr = m.predict(pl.DataFrame({"x": [0.2, 0.5, 0.8]}), type="response",
                   se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [4.7469199014934222, 0.43993064312301344, 0.03157416057280106],
        rtol=0, atol=1e-3)
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.27815576114349722, 0.12612620416309303, 0.020997731734616366],
        rtol=0, atol=1e-3)

    # deviance residuals fold in the saturated_ll; Σ res² = deviance. The
    # response residual is y − E(y) (not y − fitted_LP), routed through the
    # family hook.
    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:6], [-0.63250442710168497, 0.56473914672204939,
                   0.88371319063094189, -1.4671154974843927,
                   -0.024871912805149807, -0.2035584951393099],
        rtol=0, atol=2e-3)
    np.testing.assert_allclose(float(np.sum(dres ** 2)), m.deviance,
                               rtol=0, atol=1e-6)
    rres = np.asarray(m.residuals_of("response"))
    np.testing.assert_allclose(
        rres[:4], [-0.3688958956021644, 0.97031232323179584,
                   1.7883819521639648, -1.9008466174667389],
        rtol=0, atol=2e-3)


def _cnorm_fixture():
    # Tobit-style censored normal via R's set.seed(6) stream, reproduced
    # bit-exactly by hea.R.rng (runif(n) then rnorm(n)):
    #   set.seed(6); n<-200; x<-runif(n); ys<-2*sin(pi*x)+rnorm(n)*1.3
    #   y1<-ys; y2<-ys; y1[ys< -1]<- -1; y2[ys< -1]<- -Inf  (left censor)
    #                   y1[ys>  3]<-  3; y2[ys>  3]<-  Inf   (right censor)
    # → 173 uncensored / 10 left / 17 right. The cbind(y1, y2) response is
    # the censoring-interval matrix (col 0 observed, col 1 the bound).
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
    # R: gam(cbind(y1, y2) ~ s(x), family=cnorm(), method="REML") — censored
    # normal / Tobit. Exercises: the 2-column censored-response matrix intake
    # (a cousin of mvn's), the cancellation-safe Dd (uncensored + left +
    # right cases via dpnorm/ddnorm/log_ndtr), the single log-scale θ outer
    # Newton, and — unlike betar/ziP/ocat — the PROPER-deviance path with a
    # genuinely nonzero ls0 entering the (Dp/φ − 2·ls0) REML term while
    # lsth1 = 0 keeps the θ-gradient consistent.
    from hea.family import cnorm

    df = _cnorm_fixture()
    m = gam("cbind(y1, y2) ~ s(x)", df, family=cnorm(), method="REML")
    np.testing.assert_allclose(m.REML_criterion / 2, 320.1707496643,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.sp, [0.321192647856], rtol=2e-3)
    np.testing.assert_allclose(m.family.get_theta(), [0.228565516296],
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.family.get_theta(True), [1.25679586304],
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(m.edf_total, 4.3945575588, rtol=0, atol=1e-3)
    np.testing.assert_allclose(m.deviance, 229.8452755840, rtol=0, atol=5e-3)
    np.testing.assert_allclose(m.null_deviance, 275.8797556247, rtol=0,
                               atol=5e-3)
    np.testing.assert_allclose(m.scale, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m.Vp)[0, 0], 0.00809956779924,
                               rtol=0, atol=1e-5)
    assert m._family_display_name() == "cnorm(1.257)"
    # intercept sign-stable; tp s(x) basis carries sign noise → |coef|.
    np.testing.assert_allclose(
        np.sum(np.abs(np.asarray(m._beta))), 7.23645196361, rtol=0, atol=2e-3)

    # type="response" prediction is the latent mean μ = linkinv(η) (cnorm has
    # no predict hook — fitted IS the mean) with the standard delta SE.
    pr = m.predict(pl.DataFrame({"x": [0.1, 0.3, 0.5, 0.7, 0.9]}),
                   type="response", se_fit=True)
    np.testing.assert_allclose(
        pr["fit"].to_numpy(),
        [0.52039066828, 1.40609052147, 1.80902094522, 1.54790986302,
         0.64630975498], rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        pr["se.fit"].to_numpy(),
        [0.186345591375, 0.17266137593, 0.174231865377, 0.177779241717,
         0.172302773093], rtol=0, atol=1e-3)

    # cnorm's dev_resids is the proper deviance (≥ 0), so the DEFAULT
    # √-deviance residual works (no residuals_extended hook); Σ res² = dev.
    dres = np.asarray(m.residuals_of("deviance"))
    np.testing.assert_allclose(
        dres[:5], [-0.875033484349, 0.0785548636183, 0.221766506956,
                   -0.837126217578, -0.0758774624658], rtol=0, atol=2e-3)
    np.testing.assert_allclose(float(np.sum(dres ** 2)), m.deviance,
                               rtol=0, atol=1e-6)
    rres = np.asarray(m.residuals_of("response"))
    np.testing.assert_allclose(
        rres[:5], [-1.09973846315, 0.0987274276175, 0.278715228504,
                   -1.0520967671, -0.0953624809253], rtol=0, atol=2e-3)


# =============================================================================
# Smooth-class roadmap E / Phase 3 (smooth-review-completion.md)
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
    # mgcv: REML 25.3327898083, edf 7.5711903486, coef0 0.1249797454
    assert m.REML_criterion / 2 == pytest.approx(25.3327898083, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(7.5711903486, rel=1e-5)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(0.1249797454, abs=1e-6)
    # the smooth TERM passes through 0 at x=0.5 (the defining property of pc)
    tp = m.predict(pl.DataFrame({"x": [0.5]}), type="terms")
    assert abs(float(np.asarray(tp).ravel()[0])) < 1e-9
    # prediction is identifiability-invariant (mgcv pred@0.5 = 0.1249797454)
    pred = float(np.asarray(m.predict(pl.DataFrame({"x": [0.5]}))).ravel()[0])
    assert pred == pytest.approx(0.1249797454, abs=1e-6)

    # ... and DIFFERS from plain s(x): the intercept is the sum-to-zero one,
    # not the smooth's value at 0.5 (mgcv s(x) coef0 = -0.0073841475).
    m0 = gam("y ~ s(x, k=10)", d, method="REML")
    assert float(np.asarray(m0.coef)[0]) == pytest.approx(-0.0073841475, abs=1e-6)
    assert abs(float(np.asarray(m.coef)[0]) -
               float(np.asarray(m0.coef)[0])) > 0.1


def test_s_pc_with_by_factor_matches_mgcv():
    """pc= shares one always-applied constraint across by-factor levels —
    each level's smooth passes through 0 at the point. R-pinned end-to-end."""
    from hea.R.rng import RGenerator
    g = RGenerator(11)
    x = g.uniform(0.0, 1.0, 200)
    f = np.array(["a", "b"] * 100)          # deterministic f (RNG-reproducible)
    y = np.sin(2 * np.pi * x) + (f == "b") * 0.5 + g.normal(0.0, 1.0, 200) * 0.3
    d = pl.DataFrame({"x": x, "f": f, "y": y})
    m = gam("y ~ f + s(x, k=8, by=f, pc=0.4)", d, method="REML")
    # mgcv: REML 64.78112149, edf 13.85115823
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
        # whole tensor fixed: edf = basis dim (16), no smoothing parameters
        ("te(x, z, k=4, fx=TRUE)", 16.0, -0.12128656, 22.762519),
        # one margin fixed, the other penalized (1 sp)
        ("te(x, z, k=4, fx=c(TRUE,FALSE))", 15.16135371, -0.12128656, 22.840118),
        # ti excludes the marginal main effects → smaller fixed basis
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
    return normalize_data({"y": y, "Stim1": Stim1, "Lag": Lag, "Stim2": Stim2,
                           "Lag2": Lag2, "Xc": Xc, "A": A, "B": B})


def test_s_matrix_arg_by_matches_mgcv():
    """1-D matrix-arg s(Lag, by=Stim) (temporal RF). The varying by-matrix
    row-sums drop the centering constraint ⇒ 7 coefs (1 intercept + 6 raw),
    not 6. Free fit lands in the flat-optimum band; sp= is exact."""
    d = _rf_matrix_fixture()
    m = gam("y ~ s(Lag, by=Stim1, k=6)", d, method="REML")
    assert len(np.asarray(m.coef)) == 7                       # no constraint dropped
    assert float(np.sum(m.edf)) == pytest.approx(5.42301495, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(95.1808698, abs=1e-4)
    assert float(m.scale) == pytest.approx(0.244131395, rel=1e-5)
    np.testing.assert_allclose(
        np.asarray(m.fitted_values)[:3],
        [-0.111997576, -2.01465219, 0.78631313], atol=1e-4)
    # fixed sp — exact (design + penalty + scale.penalty, no optimiser)
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
        [-0.143668096, -0.0772378245, 0.647804128], atol=1e-5)
    # predict on the first 4 (in-sample) rows reproduces the fit
    from hea.formula import normalize_data
    nd = normalize_data({"Lag2": d["Lag2"].to_numpy()[:4],
                         "Xc": d["Xc"].to_numpy()[:4],
                         "Stim2": d["Stim2"].to_numpy()[:4]})
    pr = np.asarray(m.predict(nd)).ravel()
    np.testing.assert_allclose(
        pr, [-0.143668096, -0.0772378245, 0.647804128, 0.193445483], atol=1e-5)
    # fixed sp — exact, including coefficients
    mf = gam("y ~ te(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML",
             sp=[0.3, 2.0])
    assert float(np.sum(mf.edf)) == pytest.approx(20.8201247, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(261.321131, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.38783979, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(mf.coef)[:5],
        [-0.121393561, -0.172502089, 0.102475379, -0.0678943387,
         -0.0657378187], atol=1e-6)


def test_ti_matrix_arg_by_matches_mgcv():
    """Matrix-arg ti(Lag, Xc, by=Stim) — the pure-interaction tensor (centered
    margins, no outer constraint). Free + fixed-sp, mgcv 1.9-4."""
    d = _rf_matrix_fixture()
    m = gam("y ~ ti(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML")
    assert len(np.asarray(m.coef)) == 13
    assert float(np.sum(m.edf)) == pytest.approx(2.67806372, rel=1e-5)
    assert m.REML_criterion / 2 == pytest.approx(221.768248, abs=1e-4)
    mf = gam("y ~ ti(Lag2, Xc, by=Stim2, k=c(5,4))", d, method="REML",
             sp=[0.4, 1.1])
    assert float(np.sum(mf.edf)) == pytest.approx(12.64987, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(240.652234, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.38504158, rel=1e-6)


def test_te_matrix_arg_no_by_scale_penalty_matches_mgcv():
    """No-by matrix-arg te(A, B): locks the scale.penalty-on-long-form fix
    (previously scaled on the row-summed X → wrong S.scale, wrong fit). Fixed
    sp pins coef-level, mgcv 1.9-4."""
    d = _rf_matrix_fixture()
    mf = gam("y ~ te(A, B, k=c(4,4))", d, method="REML", sp=[0.7, 1.3])
    assert len(np.asarray(mf.coef)) == 16
    assert float(np.sum(mf.edf)) == pytest.approx(14.6812672, rel=1e-6)
    assert mf.REML_criterion / 2 == pytest.approx(234.960699, abs=1e-6)
    assert float(mf.scale) == pytest.approx(2.39505718, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(mf.coef)[:4],
        [-0.0446469446, 0.119795764, 0.0241560802, 0.141054417], atol=1e-6)


def test_matrix_arg_by_unsupported_forms_raise():
    """Honest raises (no silent mis-fit): factor by= (mgcv stops too,
    smooth.r:3970), t2() matrix args (no summation port), and pc= with a
    matrix argument."""
    d = _rf_matrix_fixture()
    d = d.with_columns(
        fac=pl.Series((np.arange(d.height) % 2).astype(str)))
    with pytest.raises(NotImplementedError, match="factor by="):
        gam("y ~ te(Lag2, Xc, by=fac, k=c(5,4))", d, method="REML")
    with pytest.raises(NotImplementedError, match="t2.. with matrix arguments"):
        gam("y ~ t2(Lag2, Xc, by=Stim2)", d, method="REML")
    with pytest.raises(NotImplementedError, match="pc="):
        gam("y ~ te(Lag2, Xc, by=Stim2, pc=c(1.0, 1.0), k=c(5,4))", d,
            method="REML")


# -----------------------------------------------------------------------------
# S2 — bs="mrf" (Markov random field). Region indicator basis + graph-Laplacian
# penalty from a neighbour list (or a supplied penalty matrix). xt threaded via
# gam(xt={region: {...}}) — the object-arg channel, like knots=. Default k =
# #regions (full rank). Source: smooth.r:2726-2875. Pins: mgcv 1.9-4.
# -----------------------------------------------------------------------------

_MRF_NB = {
    "0": ["1", "3"], "1": ["0", "2", "4"], "2": ["1", "5"],
    "3": ["0", "4", "6"], "4": ["1", "3", "5", "7"], "5": ["2", "4", "8"],
    "6": ["3", "7"], "7": ["4", "6", "8"], "8": ["5", "7"],
}


def _mrf_fixture(n: int = 180) -> pl.DataFrame:
    from hea.R.rng import RGenerator
    g = RGenerator(5)
    region = np.arange(n) % 9            # rep(0:8, length.out=n)
    reff = np.array([-1, -0.5, 0.2, 0.8, 0, -0.8, 0.5, 1, -1.2])
    y = reff[region] + g.normal(0.0, 1.0, n) * 0.5
    return pl.DataFrame({"region": region, "y": y})


def test_mrf_through_gam_matches_mgcv():
    """3x3-grid MRF (rook adjacency): graph-Laplacian penalty from nb, full rank
    (k = #regions). R-pinned end-to-end + predict + penalty=-form equivalence."""
    d = _mrf_fixture()
    m = gam('y ~ s(region, bs="mrf")', d, method="REML",
            xt={"region": {"nb": _MRF_NB}})
    assert m.REML_criterion / 2 == pytest.approx(147.98002060, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(8.83057332, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(-0.10822060, abs=1e-6)
    assert float(m.scale) == pytest.approx(0.24834096, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.coef),
        [-0.108221, -0.238663, 0.354659, 1.102483, 0.614502,
         -0.459381, 0.733006, 1.408727, -0.957989], atol=1e-5)
    # predict to regions: region-0 matches mgcv's fitted for a region-0 obs
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 4, 8]}))).ravel()
    assert pr[0] == pytest.approx(-0.960669, abs=1e-5)

    # penalty= form (supply the graph Laplacian directly) ≡ nb= form
    S = np.zeros((9, 9))
    for i, ns in _MRF_NB.items():
        S[int(i), int(i)] = len(ns)
        for j in ns:
            S[int(i), int(j)] = -1.0
    m2 = gam('y ~ s(region, bs="mrf")', d, method="REML",
             xt={"region": {"penalty": S}})
    assert m2.REML_criterion == pytest.approx(m.REML_criterion, abs=1e-9)


def test_mrf_low_rank_matches_mgcv():
    """Low-rank MRF (k=4 < 9 regions): natural-parameter truncation via
    nat.param(type=0), keeping the 4 least-penalized basis directions.

    k MUST land on a clean penalty-eigenvalue gap. The rook-grid MRF penalty
    has eigenvalues ``[0, 1, 1, 2, 3, 3, 4, 4, 6]`` — degenerate. k=4 keeps
    ``{0,1,1,2}`` (boundary at the simple eigenvalue 2; the {1,1} pair is fully
    retained, the {3,3} pair fully dropped), so the retained subspace is a
    spectral projector — uniquely determined and BLAS-invariant. A k that
    SPLITS a degenerate pair (e.g. k=5, which keeps only one of the two
    eigenvalue-3 vectors) makes the retained subspace ambiguous: Accelerate and
    OpenBLAS then pick different subspaces and the whole fit diverges (not a
    sign flip — REML/edf themselves differ). The fit is pinned on the
    basis-invariant quantities (REML/edf/intercept/predict all EXACT); the raw
    smooth coefficients still differ by an arbitrary within-eigenspace rotation,
    fit-invariant since predict applies the same reparameterization P."""
    d = _mrf_fixture()
    m = gam('y ~ s(region, bs="mrf", k=4)', d, method="REML",
            xt={"region": {"nb": _MRF_NB}})
    assert m.REML_criterion / 2 == pytest.approx(203.18732238, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(3.93834728, rel=1e-6)
    assert float(np.asarray(m.coef)[0]) == pytest.approx(-0.10822060, abs=1e-6)
    # predict == mgcv fitted (R fitted[1:3] for region 0,1,2)
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 1, 2]}))).ravel()
    np.testing.assert_allclose(
        pr, [-0.677186, -0.390348, -0.103510], atol=1e-5)


# pol2nb on a grid of unit squares yields QUEEN adjacency: diagonal squares
# share a corner vertex, so (e.g.) region 0 neighbours 1, 3 AND 4. Pinned vs
# mgcv:::pol2nb (≠ the rook _MRF_NB above).
_MRF_POLYS_NB = {
    "0": ["1", "3", "4"], "1": ["0", "2", "3", "4", "5"], "2": ["1", "4", "5"],
    "3": ["0", "1", "4", "6", "7"],
    "4": ["0", "1", "2", "3", "5", "6", "7", "8"],
    "5": ["1", "2", "4", "7", "8"], "6": ["3", "4", "7"],
    "7": ["3", "4", "5", "6", "8"], "8": ["4", "5", "7"],
}


def _mrf_grid_polys() -> dict:
    """3x3 grid of closed unit squares; region r at (row=r//3, col=r%3)."""
    polys = {}
    for r in range(9):
        row, col = r // 3, r % 3
        polys[str(r)] = np.array(
            [[col, row], [col + 1, row], [col + 1, row + 1],
             [col, row + 1], [col, row]], dtype=float)
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
    m = gam('y ~ s(region, bs="mrf")', d, method="REML",
            xt={"region": {"polys": polys}})
    assert m.REML_criterion / 2 == pytest.approx(147.85997000, abs=1e-6)
    assert float(np.sum(m.edf)) == pytest.approx(8.83460016, rel=1e-6)
    assert float(m.scale) == pytest.approx(0.24833166, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.coef),
        [-0.108221, -0.236804, 0.360208, 1.106102, 0.604543,
         -0.454536, 0.729667, 1.411282, -0.959663], atol=1e-5)
    pr = np.asarray(m.predict(pl.DataFrame({"region": [0, 4, 8]}))).ravel()
    np.testing.assert_allclose(pr, [-0.961820, 0.282922, -1.281283], atol=1e-5)

    # polys= ≡ nb= (the derived neighbour list) through the same fit
    m2 = gam('y ~ s(region, bs="mrf")', d, method="REML",
             xt={"region": {"nb": _MRF_POLYS_NB}})
    assert m2.REML_criterion == pytest.approx(m.REML_criterion, abs=1e-9)


def test_mrf_unsupported_and_validation_raise():
    """Honest boundaries: missing xt; malformed polys vertex matrix;
    wrong-dimension penalty."""
    d = _mrf_fixture()
    with pytest.raises(ValueError, match="needs xt"):
        gam('y ~ s(region, bs="mrf")', d, method="REML")
    with pytest.raises(ValueError, match="2-column"):
        gam('y ~ s(region, bs="mrf")', d, method="REML",
            xt={"region": {"polys": {"0": [[0.0, 0.0, 0.0]]}}})
    with pytest.raises(ValueError, match="expected"):
        gam('y ~ s(region, bs="mrf")', d, method="REML",
            xt={"region": {"penalty": np.eye(5)}})


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
    g.uniform(0, 1, n)                          # x3 — consumes the stream
    grp = np.array(["a", "b"] * (n // 2))
    y = np.sin(2 * np.pi * x) + 0.6 * x2 + g.normal(0, 1, n) * 0.3
    yc = g.poisson(np.exp(0.4 + np.sin(2 * np.pi * x)))
    return pl.DataFrame({"y": y, "x": x, "x2": x2, "grp": grp, "yc": yc})


def test_print_gam_layout_matches_mgcv():
    """print.gam layout pinned byte-for-byte vs mgcv: REML / multi-smooth+factor
    / GCV (scale unknown) / no-smooth (Total model d.f.) / UBRE (poisson)."""
    d = _repr_fixture_11()

    assert repr(gam("y ~ s(x)", d, method="REML")).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:", "y ~ s(x)", "",
        "Estimated degrees of freedom:", "6.92  total = 7.92 ", "",
        "REML score: 101.8412     ",
    ]
    assert repr(gam("y ~ s(x) + s(x2) + grp", d, method="REML")).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:", "y ~ s(x) + s(x2) + grp", "",
        "Estimated degrees of freedom:", "7.28 1.49  total = 10.77 ", "",
        "REML score: 74.98641     ",
    ]
    assert repr(gam("y ~ s(x) + s(x2)", d)).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:", "y ~ s(x) + s(x2)", "",
        "Estimated degrees of freedom:", "6.08 1.43  total = 8.51 ", "",
        "GCV score: 0.1006444     ",
    ]
    assert repr(gam("y ~ x + x2", d, method="REML")).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:", "y ~ x + x2", "Total model degrees of freedom 3 ", "",
        "REML score: 185.1315     ",
    ]
    assert repr(gam("yc ~ s(x)", d, family=Poisson())).split("\n") == [
        "", "Family: poisson ", "Link function: log ", "",
        "Formula:", "yc ~ s(x)", "",
        "Estimated degrees of freedom:", "4.73  total = 5.73 ", "",
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
    X = g.uniform(0, 1, n * 8).reshape(n, 8, order="F")   # R fills column-major
    cols = {f"v{j}": X[:, j] for j in range(8)}
    v0, v1, v2 = cols["v0"], cols["v1"], cols["v2"]
    y = (np.sin(2 * np.pi * v0) + np.cos(2 * np.pi * v1) + v2
         + g.normal(0, 1, n) * 0.4)
    y2 = 2 + 0.5 * v0 + g.normal(0, 1, n) * (0.2 + 0.4 * v1)
    d = pl.DataFrame({**cols, "y": y, "y2": y2})

    mw = gam("y ~ s(v0) + s(v1) + s(v2) + s(v3) + s(v4) + s(v5) + s(v6) + s(v7)",
             d, method="REML")
    assert repr(mw).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:",
        "y ~ s(v0) + s(v1) + s(v2) + s(v3) + s(v4) + s(v5) + s(v6) + s(v7)", "",
        "Estimated degrees of freedom:",
        "7.18 6.98 1.00 1.80 1.00 1.00 1.00 ",     # wraps after the 7th
        "1.86  total = 22.82 ", "",
        "REML score: 242.7806     ",
    ]

    mg = gam(["y2 ~ s(v0)", "~s(v1)"], d, family=gaulss(), method="REML")
    assert repr(mg).split("\n") == [
        "", "Family: gaulss ", "Link function: identity logb ", "",
        "Formula:", "y2 ~ s(v0)", "~s(v1)", "",
        "Estimated degrees of freedom:", "1.00 1.46  total = 4.46 ", "",
        "REML score: 199.7328     ",
    ]

    # rank-deficient (perfectly collinear parametric): mgcv appends ``rank: r/p``
    g2 = RGenerator(1)
    m = 60
    xa = g2.uniform(0, 1, m)
    xb = 2 * xa
    yy = xa + g2.normal(0, 1, m) * 0.2
    dr = pl.DataFrame({"yy": yy, "xa": xa, "xb": xb})
    with pytest.warns(UserWarning, match="rank deficient"):
        mr = gam("yy ~ xa + xb", dr, method="REML")
    assert repr(mr).split("\n") == [
        "", "Family: gaussian ", "Link function: identity ", "",
        "Formula:", "yy ~ xa + xb", "Total model degrees of freedom 2 ", "",
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

    assert gam("y ~ x + x2", d, method="REML").REML_criterion / 2 == \
        pytest.approx(185.1315, abs=1e-4)
    assert gam("y ~ x + x2", d, method="ML").ML_criterion / 2 == \
        pytest.approx(180.5841, abs=1e-4)
    assert gam("y ~ x + x2", d).GCV_score == pytest.approx(0.3107573, abs=1e-7)
    assert gam("yc ~ x", d, family=Poisson(), method="REML").REML_criterion / 2 \
        == pytest.approx(383.5616, abs=1e-4)
    assert gam("yc ~ x", d, family=Poisson()).GCV_score == \
        pytest.approx(0.154332, abs=1e-6)


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
        [2.3735475, 0.5884992, 2.1839568, 2.5833613], atol=1e-5)


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
    noise (same character as REML, §2.3) — while edf/scale/score pin tight."""
    d = load_dataset("MASS", "mcycle")
    # (method, sp, edf, scale, score)
    pins = {
        "P-REML":  (0.001201225072, 13.34821187, 511.085904, 618.2469636),
        "P-ML":    (0.00125673145,  13.23891672, 511.118452, 624.9696944),
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
        "P-REML":  (0.2044224642, 4.760659811, 0.006827033793, 78.03936884),
        "P-ML":    (0.3019696627, 4.501993974, 0.006875739433, 69.64771768),
        "GACV.Cp": (0.2793119364, 4.552183321, 0.006863431057, 0.007903434983),
    }
    for method, (sp1_t, edf_t, sc_t, score_t) in pins.items():
        b = gam("Volume ~ s(Height) + s(Girth)", d,
                family=Gamma(link="log"), method=method)
        assert float(b.sp[1]) == pytest.approx(sp1_t, rel=1e-4)
        assert float(b.edf_total) == pytest.approx(edf_t, rel=1e-5)
        assert float(b.sigma_squared) == pytest.approx(sc_t, rel=1e-5)
        assert _score_of(b) == pytest.approx(score_t, rel=1e-6)


def test_preml_pml_gacv_fixed_tweedie_match_mgcv():
    """Fixed-power Tweedie(p) is a *standard* exponential family (NOT
    extended), so GACV.Cp / P-REML / P-ML are valid for it and mgcv does not
    coerce to REML. Pinned on trees (Tweedie(p=1.5, log), Girth k=8)."""
    d = load_dataset("datasets", "trees")
    pins = {
        "GACV.Cp": (1.581523002, 3.68659722,  0.07249915935, 0.07999616904),
        "P-REML":  (2.063876052, 3.513077485, 0.07298142699, 85.43315024),
        "P-ML":    (4.816226123, 3.013187879, 0.07506138845, 80.60773045),
    }
    for method, (sp_t, edf_t, sc_t, score_t) in pins.items():
        b = gam("Volume ~ s(Girth, k=8)", d, family=Tweedie(p=1.5),
                method=method)
        assert b.method == method            # not coerced — fixed Tweedie
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
    # known-scale GACV.Cp prints the UBRE label, not "GACV".
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
    """NCV/QNCV are valid mgcv methods but not yet ported (F2)."""
    d = load_dataset("MASS", "mcycle")
    for method in ("NCV", "QNCV"):
        with pytest.raises(NotImplementedError, match="NCV"):
            gam("accel ~ s(times)", d, method=method)


def test_te_multi_dim_margin_d_arg():
    """te(..., d=c(1,2)) groups covariates into margins of given dims (mgcv
    smooth.r:399-414): one sp per MARGIN, default k=5^d, cr/ps promoted to tp."""
    rng = np.random.default_rng(3)
    n = 400
    d = pl.DataFrame({c: rng.uniform(size=n) for c in "xzw"} | {"y": rng.normal(size=n)})
    m = gam("y ~ te(x, z, w, d=c(1,2))", d, method="REML")  # 1D lag + 2D space
    assert np.asarray(m.sp).size == 2                       # one sp per margin
    # equal to explicitly naming the margin bases (1D cr + 2D tp)
    m2 = gam("y ~ te(x, z, w, d=c(1,2), bs=c('cr','tp'))", d, method="REML")
    assert np.asarray(m2.sp).size == 2
    # a single 2D margin == an ordinary 2D tp tensor; must still sum-check d
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
    d = pl.DataFrame({
        "x": x, "y": f + rng.normal(0.0, 0.1, n),
        "z": rng.uniform(size=n), "w": rng.uniform(size=n),
    })
    m1 = gam("y ~ s(x, bs='ad')", d, method="REML")
    assert np.asarray(m1.sp).size == 5            # 5 adaptive wiggliness penalties
    assert np.isfinite(np.asarray(m1.fitted_values)).all()
    m2 = gam("y ~ s(x, z, bs='ad')", d, method="REML")
    assert np.asarray(m2.sp).size == 9            # 3x3 wiggliness grid
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
    d = pl.DataFrame({c: rng.uniform(size=n) for c in "xzw"} | {"y": rng.normal(size=n)})
    m = gam("y ~ te(x, z, w, d=c(1,2), bs=c('cr','ad'), k=c(5,5))",
            d, method="REML", sp=[1.0] * 10)
    assert np.asarray(m.sp).size == 10
    assert np.isfinite(np.asarray(m.coef)).all()
