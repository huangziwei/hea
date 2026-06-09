"""
hea↔mgcv parity for the ``gam``/``bam`` ``knots=`` named-list override.

mgcv's ``gam(..., knots=list(var=...))`` threads a per-covariate knot override
to every smooth's ``smooth.construct.*.smooth.spec``; each basis consumes it its
own way. This module pins the resolved knot vectors for every per-basis branch
plus an end-to-end cyclic fit against **mgcv 1.9.4** references (generated
locally with the same hardcoded covariate vectors; CI has no R).

Why resolved-knot parity is sufficient for X/S parity: hea's basis builders
(``_cc_basis``/``_cp_basis``/``_cr_basis``/``_ps_basis``/``_bs_design`` and the
penalties) are pure functions of ``(x, knots)`` and are already matched against
mgcv's ``smoothCon`` for the default-knot path in ``test_smooths.py``. The only
thing ``knots=`` changes is the knot vector, so matching the resolved knots here
(+ that corpus) pins X and S under ``knots=`` too. The end-to-end test below
corroborates this: the full REML fit reproduces mgcv to ~1e-11.

References: mgcv 1.9.4 ``smoothCon`` / ``gam``, generated 2026-06-09.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hea.formula import expand, materialize_smooths, parse
from hea.models.bam import bam
from hea.models.gam import gam

# Covariate vectors mirrored verbatim in the R generator so that data-adaptive
# placement (cc length-2 -> place.knots) is reproducible between mgcv and hea.
XA = np.array(  # cyclic covariate strictly inside (0, 2*pi)
    [0.3, 0.7, 1.1, 1.6, 2.0, 2.4, 2.9, 3.3, 3.8, 4.2, 4.7, 5.1, 5.6, 6.0, 6.2]
)
XB = np.array(  # non-uniform covariate on [1, 11]
    [1.0, 1.3, 1.7, 2.2, 2.6, 3.1, 3.5, 4.0, 4.6, 5.2,
     5.9, 6.5, 7.2, 8.0, 8.7, 9.4, 10.1, 11.0]
)
TWO_PI = 2.0 * np.pi

DA = pl.DataFrame({"xa": XA})
DB = pl.DataFrame({"xb": XB})

KNOT_ATOL = 1e-8   # resolved-knot arithmetic: observed <= 4.3e-11
FIT_ATOL = 1e-4    # end-to-end REML fit vs mgcv: observed ~1.5e-11, kept loose
#                    against documented cross-machine REML/devfun FP drift.

# --- mgcv 1.9.4 resolved knot vectors, one per resolver branch ---------------
MGCV_KNOTS = {
    # cc length-2: endpoints pinned, interior ADAPTIVE (place.knots(c(lo,hi,x))).
    "cc_len2": [0.0, 0.966666666667, 2.13333333333, 3.3,
                4.53333333333, 5.73333333333, 6.28318530718],
    # cc full length-nk: verbatim.
    "cc_verb": [0.0, 1.0471975512, 2.09439510239, 3.14159265359,
                4.18879020479, 5.23598775598, 6.28318530718],
    # cp length-2: EVEN over [lo,hi] (seq), nk = bs_dim + 1 = 8.
    "cp_len2": [0.0, 0.897597901026, 1.79519580205, 2.69279370308,
                3.5903916041, 4.48798950513, 5.38558740615, 6.28318530718],
    # cp full length-nk: verbatim.
    "cp_verb": [0.0, 0.897597901026, 1.79519580205, 2.69279370308,
                3.5903916041, 4.48798950513, 5.38558740615, 6.28318530718],
    # cr: verbatim (no length-2 form), length must equal k.
    "cr_verb": [1.0, 4.0, 6.5, 9.0, 11.0],
    # ps length-2: [lo,hi] range fed into the PADDED even-knot build (note the
    # -0.012 / 12.012 from the 0.1% pad mgcv applies to the supplied range too).
    "ps_len2": [-5.16514285714, -3.44742857143, -1.72971428571, -0.012,
                1.70571428571, 3.42342857143, 5.14114285714, 6.85885714286,
                8.57657142857, 10.2942857143, 12.012, 13.7297142857,
                15.4474285714, 17.1651428571],
    # bs length-2: identical construction to ps for this (k, m).
    "bs_len2": [-5.16514285714, -3.44742857143, -1.72971428571, -0.012,
                1.70571428571, 3.42342857143, 5.14114285714, 6.85885714286,
                8.57657142857, 10.2942857143, 12.012, 13.7297142857,
                15.4474285714, 17.1651428571],
    # bs length-4: sorted boundary+interior, NO pad (boundary lands at 0 and 12).
    "bs_len4": [-5.14285714286, -3.42857142857, -1.71428571429, 0.0, 2.0,
                3.6, 5.2, 6.8, 8.4, 10.0, 12.0, 13.7142857143, 15.4285714286,
                17.1428571429],
}

# (case, formula, data, knots dict) for each resolved-knot reference above.
KNOT_CASES = [
    ("cc_len2", 'y ~ s(xa, bs="cc", k=7)', DA, {"xa": [0.0, TWO_PI]}),
    ("cc_verb", 'y ~ s(xa, bs="cc", k=7)', DA, {"xa": list(np.linspace(0, TWO_PI, 7))}),
    ("cp_len2", 'y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, TWO_PI]}),
    ("cp_verb", 'y ~ s(xa, bs="cp", k=7)', DA, {"xa": list(np.linspace(0, TWO_PI, 8))}),
    ("cr_verb", 'y ~ s(xb, bs="cr", k=5)', DB, {"xb": [1.0, 4.0, 6.5, 9.0, 11.0]}),
    ("ps_len2", 'y ~ s(xb, bs="ps", k=10)', DB, {"xb": [0.0, 12.0]}),
    ("bs_len2", 'y ~ s(xb, bs="bs", k=10)', DB, {"xb": [0.0, 12.0]}),
    ("bs_len4", 'y ~ s(xb, bs="bs", k=10)', DB, {"xb": [0.0, 2.0, 10.0, 12.0]}),
]


def _resolved_knots(formula: str, data: pl.DataFrame, knots) -> np.ndarray:
    """Build the (single) smooth and read back the knot vector the basis used."""
    blocks = materialize_smooths(expand(parse(formula)), data, knots=knots)
    return np.asarray(blocks[0][0].spec.raw.knots, dtype=float)


@pytest.mark.parametrize("case, formula, data, knots", KNOT_CASES)
def test_resolved_knots_match_mgcv(case, formula, data, knots):
    got = _resolved_knots(formula, data, knots)
    want = np.asarray(MGCV_KNOTS[case], dtype=float)
    assert got.shape == want.shape, f"{case}: got {got.shape} want {want.shape}"
    assert np.allclose(got, want, atol=KNOT_ATOL, rtol=0), (
        f"{case}: resolved knots diverge from mgcv "
        f"(max|diff|={float(np.max(np.abs(got - want))):.2e})"
    )


def test_cc_len2_is_adaptive_not_linspace():
    """Guards the origin-spec bug: cc length-2 interior is place.knots-adaptive,
    NOT an even linspace (that rule is cp's). On non-uniform data the two differ."""
    got = _resolved_knots('y ~ s(xa, bs="cc", k=7)', DA, {"xa": [0.0, TWO_PI]})
    even = np.linspace(0.0, TWO_PI, 7)
    # endpoints pinned to the period either way ...
    assert got[0] == pytest.approx(0.0)
    assert got[-1] == pytest.approx(TWO_PI)
    # ... but the interior must be adaptive, i.e. clearly NOT the even grid.
    assert not np.allclose(got, even, atol=1e-3)


def test_cp_len2_is_even_not_adaptive():
    """cp length-2 is the complementary rule: an even linspace over [lo,hi]."""
    got = _resolved_knots('y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, TWO_PI]})
    spacing = np.diff(got)
    assert np.allclose(spacing, spacing[0], atol=KNOT_ATOL)


# --- back-compat: the knots param must not perturb the default path ----------

@pytest.mark.parametrize("formula, data", [
    ('y ~ s(xa, bs="cc", k=7)', DA),
    ('y ~ s(xa, bs="cp", k=7)', DA),
    ('y ~ s(xb, bs="cr", k=5)', DB),
    ('y ~ s(xb, bs="ps", k=10)', DB),
    ('y ~ s(xb, bs="bs", k=10)', DB),
])
def test_no_knots_equals_empty_and_ignores_irrelevant_keys(formula, data):
    base = _resolved_knots(formula, data, None)
    assert np.array_equal(base, _resolved_knots(formula, data, {}))
    # a dict that names a different covariate leaves this smooth on its default
    assert np.array_equal(base, _resolved_knots(formula, data, {"zzz": [0.0, 1.0]}))


# --- error paths (match mgcv's stop()s) --------------------------------------

@pytest.mark.parametrize("formula, data, knots", [
    # range does not cover the data -> error (cp / ps / bs)
    ('y ~ s(xb, bs="cp", k=7)', DB, {"xb": [2.0, 10.0]}),
    ('y ~ s(xb, bs="ps", k=10)', DB, {"xb": [2.0, 10.0]}),
    ('y ~ s(xb, bs="bs", k=10)', DB, {"xb": [2.0, 10.0]}),
    # wrong-length verbatim vector -> error
    ('y ~ s(xa, bs="cc", k=7)', DA, {"xa": [0.0, 1.0, 2.0]}),
    ('y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, 1.0, 2.0]}),
    ('y ~ s(xb, bs="cr", k=5)', DB, {"xb": [1.0, 5.0, 11.0]}),
])
def test_bad_knots_raise(formula, data, knots):
    with pytest.raises(ValueError):
        _resolved_knots(formula, data, knots)


@pytest.mark.parametrize("model", [gam, bam])
def test_non_dict_knots_rejected(model):
    with pytest.raises(TypeError):
        model('y ~ s(xa, bs="cc", k=7)', DA.with_columns(y=pl.Series(XA)),
              knots=[0.0, TWO_PI])


# --- end-to-end: cyclic period correctness (the acceptance case) -------------
#
# theta = (pi/6)*month, a clean harmonic signal. With the TRUE period
# knots=[0,2*pi], month 1 (theta=pi/6) and month 12 (theta=2*pi) are distinct
# points, so f(Jan) != f(Dec). Without knots the default period is the data
# range [pi/6, 2*pi], which identifies those endpoints and forces f(Jan)==f(Dec)
# EXACTLY -- the silently-wrong behavior this feature fixes.

_MONTH = np.arange(1, 13)
_THETA = (np.pi / 6.0) * _MONTH
_Y = 2.0 * np.sin(_THETA) + 1.5 * np.cos(_THETA)
_DF = pl.DataFrame({"theta": _THETA, "y": _Y})

# mgcv gam(y ~ s(theta, bs="cc", k=7), knots=list(theta=c(0,2*pi)), method="REML")
MGCV_FIT_KNOTS = np.array([
    2.29433162013, 2.48709409373, 1.99590569157, 0.984046239426,
    -0.298425928558, -1.50304785431, -2.29433162013, -2.48709409373,
    -1.99590569157, -0.984046239426, 0.298425928558, 1.50304785431,
])
MGCV_GAP_KNOTS = 0.791283765816          # fit[Jan] - fit[Dec]
MGCV_PRED_AT_1 = 2.49811578643           # predict at theta = 1.0
MGCV_PRED_AT_4 = -2.49588668767          # predict at theta = 4.0


def test_cc_period_correctness_matches_mgcv():
    gk = gam('y ~ s(theta, bs="cc", k=7)', _DF,
             knots={"theta": [0.0, TWO_PI]}, method="REML")
    g0 = gam('y ~ s(theta, bs="cc", k=7)', _DF, method="REML")

    fk = np.asarray(gk.fitted_values, dtype=float).ravel()
    f0 = np.asarray(g0.fitted_values, dtype=float).ravel()

    # structural: default period collapses Jan==Dec; true period does not.
    assert abs(f0[0] - f0[11]) < 1e-6
    assert abs(fk[0] - fk[11]) > 0.5

    # parity: fit and Jan-Dec gap reproduce mgcv.
    assert np.allclose(fk, MGCV_FIT_KNOTS, atol=FIT_ATOL, rtol=0)
    assert (fk[0] - fk[11]) == pytest.approx(MGCV_GAP_KNOTS, abs=FIT_ATOL)


def test_cc_predict_reuses_supplied_period():
    gk = gam('y ~ s(theta, bs="cc", k=7)', _DF,
             knots={"theta": [0.0, TWO_PI]}, method="REML")
    nd = pl.DataFrame({"theta": [1.0, 1.0 + TWO_PI, 4.0, 4.0 + TWO_PI]})
    p = np.asarray(gk.predict(nd), dtype=float).ravel()
    # predicting one full period apart is identical (period reused at predict).
    assert p[0] == pytest.approx(p[1], abs=1e-8)
    assert p[2] == pytest.approx(p[3], abs=1e-8)
    # and matches mgcv's predictions.
    assert p[0] == pytest.approx(MGCV_PRED_AT_1, abs=FIT_ATOL)
    assert p[2] == pytest.approx(MGCV_PRED_AT_4, abs=FIT_ATOL)


# --- te/ti/t2 marginal knots (slice 3) ---------------------------------------
#
# mgcv passes the same knots= list to every marginal smooth.construct. hea's
# tensor cr margins consume it (verbatim, length k) exactly like a standalone
# cr; tp margins ignore it (matches mgcv); cc/cp/ps/bs margins are unsupported
# in hea's te and raise regardless of knots (loud, never silent).

# deterministic 8x8 grid in [1,11] x [0,10]
_I = np.arange(64)
_GX = 1.0 + 10.0 * (_I % 8) / 7.0
_GZ = 0.0 + 10.0 * (_I // 8) / 7.0
_GY = np.sin(_GX / 2) + np.cos(_GZ / 3) + 0.5 * np.sin(_GX * _GZ / 20)
_DGRID = pl.DataFrame({"x": _GX, "z": _GZ, "y": _GY})
_CR_MARGIN_KNOTS = [0.0, 3.0, 6.0, 9.0, 12.0]   # length k=5 (te default), brackets [1,11]

# mgcv gam(y ~ {te,ti,t2}(x, z), knots=list(x=c(0,3,6,9,12)), method="REML"):
# (first 6 fitted values, sum of fitted values).
_TENSOR_REF = {
    "te": ([1.488167379, 1.950615789, 1.943195215, 1.482116663, 0.804951035,
            0.2042411712], 9.67081174),
    "ti": ([-0.2017852234, -0.1009608832, -0.0001362798873, 0.1006891471,
            0.2015162364, 0.3023452372], 9.67081174),
    "t2": ([1.487610899, 1.950943544, 1.943587776, 1.481921192, 0.8041563889,
            0.203241685], 9.67081174),
}


@pytest.mark.parametrize("fn", ["te", "ti", "t2"])
def test_tensor_cr_margin_knots_match_mgcv(fn):
    g = gam(f"y ~ {fn}(x, z)", _DGRID, knots={"x": _CR_MARGIN_KNOTS}, method="REML")
    g0 = gam(f"y ~ {fn}(x, z)", _DGRID, method="REML")
    fit = np.asarray(g.fitted_values, dtype=float).ravel()
    fit0 = np.asarray(g0.fitted_values, dtype=float).ravel()
    ref6, refsum = _TENSOR_REF[fn]
    # the cr-margin knots actually changed the fit (else they were ignored)
    assert np.sum(np.abs(fit - fit0)) > 1e-5
    # and the fit reproduces mgcv
    assert np.allclose(fit[:6], np.array(ref6), atol=FIT_ATOL, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-3)


def test_tensor_tp_margin_ignores_knots():
    # tp margins ignore knots in mgcv -> hea fit must be unchanged by knots=.
    a = gam('y ~ te(x, z, bs="tp")', _DGRID, knots={"x": [0.0, 12.0]}, method="REML")
    b = gam('y ~ te(x, z, bs="tp")', _DGRID, method="REML")
    assert np.allclose(np.asarray(a.fitted_values, dtype=float),
                       np.asarray(b.fitted_values, dtype=float), atol=1e-8)


def test_tensor_unsupported_margin_raises_not_silent():
    # cc margins are unsupported in hea's te -> raise (loud), regardless of knots.
    with pytest.raises(NotImplementedError):
        materialize_smooths(
            expand(parse('y ~ te(x, z, bs="cc")')), _DGRID, knots={"x": [0.0, 12.0]})


def test_tensor_unrelated_knots_ignored():
    blocks = materialize_smooths(
        expand(parse('y ~ te(x, z)')), _DGRID, knots={"zzz": [0.0, 1.0]})
    assert blocks


@pytest.mark.parametrize("discrete", [False, True])
def test_bam_knots_passthrough_both_paths(discrete):
    """knots= must reach the builder on BOTH bam materialize_smooths sites."""
    bk = bam('y ~ s(theta, bs="cc", k=7)', _DF,
             knots={"theta": [0.0, TWO_PI]}, method="fREML", discrete=discrete)
    b0 = bam('y ~ s(theta, bs="cc", k=7)', _DF, method="fREML", discrete=discrete)
    fk = np.asarray(bk.fitted_values, dtype=float).ravel()
    f0 = np.asarray(b0.fitted_values, dtype=float).ravel()
    assert abs(f0[0] - f0[11]) < 1e-6        # default period: collapsed
    assert abs(fk[0] - fk[11]) > 0.5         # supplied period: distinct
    assert (fk[0] - fk[11]) == pytest.approx(MGCV_GAP_KNOTS, abs=1e-3)
