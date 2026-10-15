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

XA = np.array(  # cyclic covariate strictly inside (0, 2*pi)
    [0.3, 0.7, 1.1, 1.6, 2.0, 2.4, 2.9, 3.3, 3.8, 4.2, 4.7, 5.1, 5.6, 6.0, 6.2]
)
XB = np.array(  # non-uniform covariate on [1, 11]
    [
        1.0,
        1.3,
        1.7,
        2.2,
        2.6,
        3.1,
        3.5,
        4.0,
        4.6,
        5.2,
        5.9,
        6.5,
        7.2,
        8.0,
        8.7,
        9.4,
        10.1,
        11.0,
    ]
)
TWO_PI = 2.0 * np.pi

DA = pl.DataFrame({"xa": XA})
DB = pl.DataFrame({"xb": XB})

KNOT_ATOL = 1e-8  # resolved-knot arithmetic: observed <= 4.3e-11
FIT_ATOL = 1e-4  # end-to-end REML fit vs mgcv: observed ~1.5e-11, kept loose

MGCV_KNOTS = {
    "cc_len2": [
        0.0,
        0.966666666667,
        2.13333333333,
        3.3,
        4.53333333333,
        5.73333333333,
        6.28318530718,
    ],
    "cc_verb": [
        0.0,
        1.0471975512,
        2.09439510239,
        3.14159265359,
        4.18879020479,
        5.23598775598,
        6.28318530718,
    ],
    "cp_len2": [
        0.0,
        0.897597901026,
        1.79519580205,
        2.69279370308,
        3.5903916041,
        4.48798950513,
        5.38558740615,
        6.28318530718,
    ],
    "cp_verb": [
        0.0,
        0.897597901026,
        1.79519580205,
        2.69279370308,
        3.5903916041,
        4.48798950513,
        5.38558740615,
        6.28318530718,
    ],
    "cr_verb": [1.0, 4.0, 6.5, 9.0, 11.0],
    "cs_verb": [1.0, 4.0, 6.5, 9.0, 11.0],
    "ps_len2": [
        -5.16514285714,
        -3.44742857143,
        -1.72971428571,
        -0.012,
        1.70571428571,
        3.42342857143,
        5.14114285714,
        6.85885714286,
        8.57657142857,
        10.2942857143,
        12.012,
        13.7297142857,
        15.4474285714,
        17.1651428571,
    ],
    "bs_len2": [
        -5.16514285714,
        -3.44742857143,
        -1.72971428571,
        -0.012,
        1.70571428571,
        3.42342857143,
        5.14114285714,
        6.85885714286,
        8.57657142857,
        10.2942857143,
        12.012,
        13.7297142857,
        15.4474285714,
        17.1651428571,
    ],
    "bs_len4": [
        -5.14285714286,
        -3.42857142857,
        -1.71428571429,
        0.0,
        2.0,
        3.6,
        5.2,
        6.8,
        8.4,
        10.0,
        12.0,
        13.7142857143,
        15.4285714286,
        17.1428571429,
    ],
}

KNOT_CASES = [
    ("cc_len2", 'y ~ s(xa, bs="cc", k=7)', DA, {"xa": [0.0, TWO_PI]}),
    ("cc_verb", 'y ~ s(xa, bs="cc", k=7)', DA, {"xa": list(np.linspace(0, TWO_PI, 7))}),
    ("cp_len2", 'y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, TWO_PI]}),
    ("cp_verb", 'y ~ s(xa, bs="cp", k=7)', DA, {"xa": list(np.linspace(0, TWO_PI, 8))}),
    ("cr_verb", 'y ~ s(xb, bs="cr", k=5)', DB, {"xb": [1.0, 4.0, 6.5, 9.0, 11.0]}),
    ("cs_verb", 'y ~ s(xb, bs="cs", k=5)', DB, {"xb": [1.0, 4.0, 6.5, 9.0, 11.0]}),
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
    assert got[0] == pytest.approx(0.0)
    assert got[-1] == pytest.approx(TWO_PI)
    assert not np.allclose(got, even, atol=1e-3)


def test_cp_len2_is_even_not_adaptive():
    """cp length-2 is the complementary rule: an even linspace over [lo,hi]."""
    got = _resolved_knots('y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, TWO_PI]})
    spacing = np.diff(got)
    assert np.allclose(spacing, spacing[0], atol=KNOT_ATOL)


@pytest.mark.parametrize(
    "formula, data",
    [
        ('y ~ s(xa, bs="cc", k=7)', DA),
        ('y ~ s(xa, bs="cp", k=7)', DA),
        ('y ~ s(xb, bs="cr", k=5)', DB),
        ('y ~ s(xb, bs="ps", k=10)', DB),
        ('y ~ s(xb, bs="bs", k=10)', DB),
    ],
)
def test_no_knots_equals_empty_and_ignores_irrelevant_keys(formula, data):
    base = _resolved_knots(formula, data, None)
    assert np.array_equal(base, _resolved_knots(formula, data, {}))
    assert np.array_equal(base, _resolved_knots(formula, data, {"zzz": [0.0, 1.0]}))


@pytest.mark.parametrize(
    "formula, data, knots",
    [
        ('y ~ s(xb, bs="cp", k=7)', DB, {"xb": [2.0, 10.0]}),
        ('y ~ s(xb, bs="ps", k=10)', DB, {"xb": [2.0, 10.0]}),
        ('y ~ s(xb, bs="bs", k=10)', DB, {"xb": [2.0, 10.0]}),
        ('y ~ s(xa, bs="cc", k=7)', DA, {"xa": [0.0, 1.0, 2.0]}),
        ('y ~ s(xa, bs="cp", k=7)', DA, {"xa": [0.0, 1.0, 2.0]}),
        ('y ~ s(xb, bs="cr", k=5)', DB, {"xb": [1.0, 5.0, 11.0]}),
    ],
)
def test_bad_knots_raise(formula, data, knots):
    with pytest.raises(ValueError):
        _resolved_knots(formula, data, knots)


@pytest.mark.parametrize("model", [gam, bam])
def test_non_dict_knots_rejected(model):
    with pytest.raises(TypeError):
        model(
            'y ~ s(xa, bs="cc", k=7)',
            DA.with_columns(y=pl.Series(XA)),
            knots=[0.0, TWO_PI],
        )


_MONTH = np.arange(1, 13)
_THETA = (np.pi / 6.0) * _MONTH
_Y = 2.0 * np.sin(_THETA) + 1.5 * np.cos(_THETA)
_DF = pl.DataFrame({"theta": _THETA, "y": _Y})

MGCV_FIT_KNOTS = np.array(
    [
        2.29433162013,
        2.48709409373,
        1.99590569157,
        0.984046239426,
        -0.298425928558,
        -1.50304785431,
        -2.29433162013,
        -2.48709409373,
        -1.99590569157,
        -0.984046239426,
        0.298425928558,
        1.50304785431,
    ]
)
MGCV_GAP_KNOTS = 0.791283765816  # fit[Jan] - fit[Dec]
MGCV_PRED_AT_1 = 2.49811578643  # predict at theta = 1.0
MGCV_PRED_AT_4 = -2.49588668767  # predict at theta = 4.0


def test_cc_period_correctness_matches_mgcv():
    gk = gam(
        'y ~ s(theta, bs="cc", k=7)', _DF, knots={"theta": [0.0, TWO_PI]}, method="REML"
    )
    g0 = gam('y ~ s(theta, bs="cc", k=7)', _DF, method="REML")

    fk = np.asarray(gk.fitted_values, dtype=float).ravel()
    f0 = np.asarray(g0.fitted_values, dtype=float).ravel()

    assert abs(f0[0] - f0[11]) < 1e-6
    assert abs(fk[0] - fk[11]) > 0.5

    assert np.allclose(fk, MGCV_FIT_KNOTS, atol=FIT_ATOL, rtol=0)
    assert (fk[0] - fk[11]) == pytest.approx(MGCV_GAP_KNOTS, abs=FIT_ATOL)


def test_cc_predict_reuses_supplied_period():
    gk = gam(
        'y ~ s(theta, bs="cc", k=7)', _DF, knots={"theta": [0.0, TWO_PI]}, method="REML"
    )
    nd = pl.DataFrame({"theta": [1.0, 1.0 + TWO_PI, 4.0, 4.0 + TWO_PI]})
    p = np.asarray(gk.predict(nd), dtype=float).ravel()
    assert p[0] == pytest.approx(p[1], abs=1e-8)
    assert p[2] == pytest.approx(p[3], abs=1e-8)
    assert p[0] == pytest.approx(MGCV_PRED_AT_1, abs=FIT_ATOL)
    assert p[2] == pytest.approx(MGCV_PRED_AT_4, abs=FIT_ATOL)


_I = np.arange(64)
_GX = 1.0 + 10.0 * (_I % 8) / 7.0
_GZ = 0.0 + 10.0 * (_I // 8) / 7.0
_GY = np.sin(_GX / 2) + np.cos(_GZ / 3) + 0.5 * np.sin(_GX * _GZ / 20)
_DGRID = pl.DataFrame({"x": _GX, "z": _GZ, "y": _GY})
_CR_MARGIN_KNOTS = [
    0.0,
    3.0,
    6.0,
    9.0,
    12.0,
]  # length k=5 (te default), brackets [1,11]

_TENSOR_REF = {
    "te": (
        [1.488167379, 1.950615789, 1.943195215, 1.482116663, 0.804951035, 0.2042411712],
        9.67081174,
    ),
    "ti": (
        [
            -0.2017852234,
            -0.1009608832,
            -0.0001362798873,
            0.1006891471,
            0.2015162364,
            0.3023452372,
        ],
        9.67081174,
    ),
    "t2": (
        [1.487610899, 1.950943544, 1.943587776, 1.481921192, 0.8041563889, 0.203241685],
        9.67081174,
    ),
}


@pytest.mark.parametrize("fn", ["te", "ti", "t2"])
def test_tensor_cr_margin_knots_match_mgcv(fn):
    g = gam(f"y ~ {fn}(x, z)", _DGRID, knots={"x": _CR_MARGIN_KNOTS}, method="REML")
    g0 = gam(f"y ~ {fn}(x, z)", _DGRID, method="REML")
    fit = np.asarray(g.fitted_values, dtype=float).ravel()
    fit0 = np.asarray(g0.fitted_values, dtype=float).ravel()
    ref6, refsum = _TENSOR_REF[fn]
    assert np.sum(np.abs(fit - fit0)) > 1e-5
    assert np.allclose(fit[:6], np.array(ref6), atol=FIT_ATOL, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-3)


def test_tensor_tp_margin_ignores_knots():
    a = gam('y ~ te(x, z, bs="tp")', _DGRID, knots={"x": [0.0, 12.0]}, method="REML")
    b = gam('y ~ te(x, z, bs="tp")', _DGRID, method="REML")
    assert np.allclose(
        np.asarray(a.fitted_values, dtype=float),
        np.asarray(b.fitted_values, dtype=float),
        atol=1e-8,
    )


def test_tensor_unsupported_margin_raises_not_silent():
    with pytest.raises(NotImplementedError):
        materialize_smooths(expand(parse('y ~ te(x, z, bs="gp")')), _DGRID)


def test_tensor_unrelated_knots_ignored():
    blocks = materialize_smooths(
        expand(parse("y ~ te(x, z)")), _DGRID, knots={"zzz": [0.0, 1.0]}
    )
    assert blocks


_J = np.arange(64)
_TGX = 0.15 + (2 * np.pi - 0.3) * (_J % 8) / 7.0  # x in (0, 2pi)
_TGZ = (_J // 8) / 7.0 * 10.0  # z in [0, 10]
_TGY = np.sin(_TGX) + np.cos(_TGZ / 3) + 0.4 * np.sin(_TGX) * _TGZ / 10
_DTENS = pl.DataFrame({"x": _TGX, "z": _TGZ, "y": _TGY})

_TE_MARGIN_REF = {
    "cc": (
        [0.0, 2 * np.pi],
        [
            1.170485472,
            1.897275812,
            1.946279596,
            1.390922313,
            0.6339380525,
            0.07987115742,
        ],
        -3.067510149,
    ),
    "cp": (
        [0.0, 2 * np.pi],
        [
            1.158396002,
            1.857008723,
            1.965038429,
            1.428696211,
            0.5922861809,
            0.05574713466,
        ],
        -3.067510149,
    ),
    "ps": (
        [0.0, 7.0],
        [1.15036304, 1.963036709, 1.920773823, 1.371115039, 0.6615983155, 0.133281978],
        -3.067510149,
    ),
    "bs": (
        [0.0, 7.0],
        [
            1.150526902,
            1.963300792,
            1.921058023,
            1.371374024,
            0.6618215159,
            0.1334866525,
        ],
        -3.067510149,
    ),
}


@pytest.mark.parametrize("margin", ["cc", "cp", "ps", "bs"])
def test_te_new_margin_knots_match_mgcv(margin):
    kn, ref6, refsum = _TE_MARGIN_REF[margin]
    f = f'y ~ te(x, z, bs=c("{margin}","tp"))'
    g = gam(f, _DTENS, knots={"x": kn}, method="REML")
    g0 = gam(f, _DTENS, method="REML")
    fit = np.asarray(g.fitted_values, dtype=float).ravel()
    fit0 = np.asarray(g0.fitted_values, dtype=float).ravel()
    assert np.sum(np.abs(fit - fit0)) > 1e-3  # margin knots took effect
    assert np.allclose(fit[:6], np.array(ref6), atol=FIT_ATOL, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-3)


_SX = np.linspace(0.0, 1.0, 40)
_SY = np.sin(4 * _SX) + 0.5 * _SX + 0.2 * np.cos(8 * _SX)
_DSHRINK = pl.DataFrame({"x": _SX, "y": _SY})

_SHRINK_REF = {
    "cs": (
        [
            0.2050605341,
            0.3080972801,
            0.4081741552,
            0.5023312884,
            0.5876088086,
            0.6612796069,
        ],
        26.77717736,
        "cr",
    ),
    "ts": (
        [0.2046326916, 0.308104904, 0.408655978, 0.5028161393, 0.587536063, 0.66108276],
        26.77717736,
        "tp",
    ),
}


@pytest.mark.parametrize("bs", ["cs", "ts"])
def test_shrinkage_basis_matches_mgcv(bs):
    ref6, refsum, base = _SHRINK_REF[bs]
    g = gam(f'y ~ s(x, bs="{bs}", k=10)', _DSHRINK, method="REML")
    gbase = gam(f'y ~ s(x, bs="{base}", k=10)', _DSHRINK, method="REML")
    fit = np.asarray(g.fitted_values, dtype=float).ravel()
    fitb = np.asarray(gbase.fitted_values, dtype=float).ravel()
    assert np.sum(np.abs(fit - fitb)) > 1e-5
    assert np.allclose(fit[:6], np.array(ref6), atol=1e-5, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-4)


# --- ds (Duchon spline, Tier 2) --------------------------------------
# Reproducible NON-GRID points (√2,√3,√5 mod 1). A regular grid's symmetry makes
# the kernel eigenvalues degenerate at the rank-k truncation boundary, where
# eigh and mgcv's slanczos pick different (equally valid) bases of the degenerate
# eigenspace — an arbitrary truncation, not a bug; generic data has no such
# degeneracy and matches mgcv to ~1e-9. Covers odd kernel exponents (1-D default
# ke=3, 2-D s=0.5 ke=3, 3-D ke=1 — the spherical R³-embedding case).
_DI = np.arange(120)
_DDX = ((_DI + 1) * np.sqrt(2)) % 1
_DDY = ((_DI + 1) * np.sqrt(3)) % 1
_DDZ = ((_DI + 1) * np.sqrt(5)) % 1
_DDRESP = np.sin(3 * _DDX) + np.cos(3 * _DDY) + 0.5 * _DDX * _DDY
_DDUCHON = pl.DataFrame({"x": _DDX, "y": _DDY, "z": _DDZ, "resp": _DDRESP})

_DS_REF = {
    'resp~s(x,bs="ds")': (
        [
            1.018977856,
            0.8944856975,
            0.7266364385,
            1.082800301,
            0.2868604633,
            1.085883881,
        ],
        99.1190289,
    ),
    'resp~s(x,y,bs="ds",m=c(2,0.5),k=15)': (
        [
            0.513294058,
            0.9854715318,
            1.519510961,
            0.2893410526,
            -0.1647833207,
            1.481446076,
        ],
        99.1190289,
    ),
    'resp~s(x,y,z,bs="ds",m=c(2,0),k=15)': (
        [
            0.5989270326,
            0.8611405181,
            1.459403579,
            0.1248205412,
            -0.08803478233,
            1.53621942,
        ],
        99.1190289,
    ),
}


@pytest.mark.parametrize("formula", list(_DS_REF))
def test_duchon_spline_matches_mgcv(formula):
    ref6, refsum = _DS_REF[formula]
    fit = np.asarray(
        gam(formula, _DDUCHON, method="REML").fitted_values, dtype=float
    ).ravel()
    assert np.allclose(fit[:6], np.array(ref6), atol=1e-5, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-4)


_SI = np.arange(100)
_SLA = -90 + 180 * (((_SI + 1) * np.sqrt(2)) % 1)
_SLO = -180 + 360 * (((_SI + 1) * np.sqrt(3)) % 1)
_SOSRESP = (
    np.sin(_SLA * np.pi / 180)
    + np.cos(_SLO * np.pi / 180)
    + 0.3 * np.sin(2 * _SLA * np.pi / 180)
)
_DSOS = pl.DataFrame({"la": _SLA, "lo": _SLO, "resp": _SOSRESP})

_SOS_REF = {
    0: (
        [
            -0.305692179,
            2.180044305,
            -1.419174683,
            -0.1965701097,
            -0.5825114232,
            0.7050912758,
        ],
        0.1032882204,
    ),
    1: (
        [
            -0.2999685876,
            2.200354645,
            -1.419732277,
            -0.1657718073,
            -0.5563202624,
            0.6917979824,
        ],
        0.1032882204,
    ),
    2: (
        [
            -0.3071683682,
            2.206072393,
            -1.426163737,
            -0.1890717838,
            -0.5646068575,
            0.7029232916,
        ],
        0.1032882204,
    ),
}


@pytest.mark.parametrize("m", [0, 1, 2])
def test_sos_matches_mgcv(m):
    """sos incl. the DEFAULT order m=0 (dilogarithm kernel) and closed forms."""
    ref6, refsum = _SOS_REF[m]
    fit = np.asarray(
        gam(f'resp~s(la,lo,bs="sos",m={m},k=30)', _DSOS, method="REML").fitted_values,
        dtype=float,
    ).ravel()
    assert np.allclose(fit[:6], np.array(ref6), atol=1e-5, rtol=0)
    assert fit.sum() == pytest.approx(refsum, abs=1e-4)


@pytest.mark.parametrize("discrete", [False, True])
def test_bam_knots_passthrough_both_paths(discrete):
    """knots= must reach the builder on BOTH bam materialize_smooths sites."""
    bk = bam(
        'y ~ s(theta, bs="cc", k=7)',
        _DF,
        knots={"theta": [0.0, TWO_PI]},
        method="fREML",
        discrete=discrete,
    )
    b0 = bam('y ~ s(theta, bs="cc", k=7)', _DF, method="fREML", discrete=discrete)
    fk = np.asarray(bk.fitted_values, dtype=float).ravel()
    f0 = np.asarray(b0.fitted_values, dtype=float).ravel()
    assert abs(f0[0] - f0[11]) < 1e-6  # default period: collapsed
    assert abs(fk[0] - fk[11]) > 0.5  # supplied period: distinct
    assert (fk[0] - fk[11]) == pytest.approx(MGCV_GAP_KNOTS, abs=1e-3)
