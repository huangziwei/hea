"""mgcv's ``magic`` and its callers against mgcv itself: the C engine on fixed
inputs, the R wrapper ``magic()`` with ``magic.post.proc``, gam's Gaussian
additive GCV/UBRE fits (``am.fit``), bam's GCV.Cp fits (``bam.fit`` and the
``bgam.fit`` PIRLS rail), the null deviance gam refits by ``glm`` when there
is an offset, and bam's id-linked bases. The fixtures in
``tests/fixtures/magic`` are written by ``tests/scripts/make_magic_fixtures.R``.

With the compiled port under the arithmetic the fixtures were written with
(``PINNED_ARITHMETIC``) every value must be mgcv's to the bit; elsewhere the
BLAS and LAPACK round differently, the comparisons take a tolerance and the
singular vectors in ``rV`` may change sign.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import (
    FIXTURE_ROOT,
    PINNED_ARITHMETIC,
    assert_allclose_up_to_signs,
    load_hex_fixture,
)

import hea
from hea._dispatch import HAVE_RS, rs_fn
from hea.family import Poisson
from hea.models._magic import magic, magic_post_proc
from hea.models.gam import gam_control

_MAGIC = FIXTURE_ROOT / "magic"


def _load(name: str) -> dict:
    return load_hex_fixture(_MAGIC / f"{name}.txt")


def _same(got, want, name, rtol=1e-8, atol=1e-10):
    """Exact under PINNED_ARITHMETIC, else to the tolerance. Off the pinned
    arithmetic ``rV``'s columns, singular vectors, may each be mgcv's
    negated; and ``R``, the pivoted QR factor with its columns unpivoted,
    is compared through ``R'R = X'X``: when columns of X tie in norm (two of
    norm 20 in the ``big`` case), the pivot order and with it the rows of
    ``R`` follow the arithmetic."""
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(got, want, err_msg=name)
    elif name == "rV":
        assert_allclose_up_to_signs(got, want, rtol=rtol, atol=atol, err_msg=name)
    elif name == "R":
        np.testing.assert_allclose(
            got.T @ got, want.T @ want, rtol=rtol, atol=atol, err_msg=name
        )
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=name)


class _Slot:
    def __init__(self, S, col_start):
        self.S = S
        self.col_start = col_start
        self.col_end = col_start + S.shape[0]


_ENGINE_CASES = [
    "gcv_auto",
    "ubre_auto",
    "gcv_given",
    "gcv_flat",
    "gcv_gamma",
    "gcv_L",
    "ubre_L",
    "gcv_H",
    "small",
    "L0",
    "big",
]


@pytest.mark.skipif(not HAVE_RS, reason="the C engine's port lives in hea._rs")
@pytest.mark.parametrize("case", _ENGINE_CASES)
def test_magic_engine_matches_mgcv(case):
    """``_rs.magic_c`` is mgcv's C ``magic``: the pivoted QR, the
    ``fit_magic`` SVDs and scores, ``magic_gH``'s derivatives, the
    Newton/steepest-descent search with step halving and the
    infinite-smoothing-parameter checks give mgcv's coefficients,
    smoothing parameters, score, scale, rms gradient, ``rV``, ``R`` and
    convergence record (autoinitialized, given and flat-reset starting
    values; an ``L`` map, one with no columns; a fixed penalty)."""
    d = _load(f"c_{case}")
    q = d["X"].shape[1]
    S = d["S"] if d["S"].ndim == 2 else d["S"].reshape(q, -1, order="F")
    L = d["L"]
    if L is not None and L.ndim == 1:
        L = L.reshape(-1, 0)
    b, sp, score, scale, info, rms, rV, R = rs_fn("magic_c")(
        d["y"],
        d["X"],
        np.atleast_1d(d["sp0"]),
        np.atleast_1d(d["def_sp"]),
        S,
        d["H"],
        L,
        np.atleast_1d(d["lsp0"]),
        float(d["gamma"][0]),
        float(d["scale_in"][0]),
        bool(d["gcv"][0]),
        int(d["max_half"][0]),
        [int(c) for c in np.atleast_1d(d["cS"])] if d["cS"] is not None else [],
        float(d["rank_tol"][0]),
        float(d["tol"][0]),
        float(d["norm_const"][0]),
        int(d["n_score"][0]),
    )
    assert tuple(int(v) for v in info) == tuple(int(v) for v in d["o_info"])
    for name, got in [
        ("b", b),
        ("sp", sp),
        ("score", score),
        ("scale", scale),
        ("rms", rms),
        ("rV", rV),
        ("R", R),
    ]:
        _same(got, d[f"o_{name}"], name, rtol=1e-6, atol=1e-9)


_WRAPPER_CASES = [
    "chol",
    "svd",
    "w",
    "ubre",
    "given",
    "control",
    "L",
    "C",
    "H",
    "pad",
    "C0",
]


@pytest.mark.parametrize("case", _WRAPPER_CASES)
def test_magic_wrapper_matches_mgcv(case):
    """``magic()`` is mgcv's R ``magic``: ``initial.sp`` starting values,
    penalty roots by ``mroot`` (pivoted Cholesky with the ranks, the
    eigen route without), ``L`` mapped by ``lm``, constraints absorbed by
    ``qr(t(C))``, weights, the zero padding of a short design, the ridge
    added to ``H``; ``magic_post_proc`` is ``magic.post.proc``."""
    d = _load(f"r_{case}")
    slots = [_Slot(d[f"S{i + 1}"], int(d["off"][i])) for i in range(int(d["nS"][0]))]
    kw: dict = {}
    for key, v in d.items():
        if not key.startswith("a_"):
            continue
        arg = key[2:]
        if arg == "rank":
            kw["rank"] = [int(x) for x in v]
        elif arg == "control":
            kw["control"] = {"tol": v[0], "step_half": int(v[1]), "rank_tol": v[2]}
        elif arg == "C":
            kw["C"] = v if v.ndim == 2 else v.reshape(-1, d["X"].shape[1], order="F")
        elif arg in ("L", "lsp0", "H"):
            kw[arg] = v
        elif arg == "gcv":
            kw["gcv"] = bool(v[0])
        elif arg == "n_score":
            kw["n_score"] = int(v[0])
        else:
            kw[arg] = float(v[0])
    m = magic(d["y"], d["X"], d["sp_in"], slots, w=d["w"], **kw)
    gi = m["gcv_info"]
    assert (
        gi["full_rank"],
        gi["rank"],
        int(gi["fully_converged"]),
        int(gi["hess_pos_def"]),
        gi["iter"],
        gi["score_calls"],
    ) == tuple(int(v) for v in d["o_info"])
    for name in ("b", "scale", "score", "sp", "sp_full", "R", "rV"):
        _same(m[name], d[f"o_{name}"], name, rtol=1e-6, atol=1e-9)
    if "p_Ve" in d:
        w = d["w"]
        mv = magic_post_proc(d["X"], m, w=None if w is None else w**2)
        for name in ("Ve", "Vb", "hat", "edf", "edf1", "F"):
            _same(mv[name], d[f"p_{name}"], name, rtol=1e-6, atol=1e-9)


def _data() -> pl.DataFrame:
    D = _load("data")
    fac = np.array(["a", "b", "c"])[D["fac"].astype(int) - 1]
    return pl.DataFrame(
        {
            "x1": D["x1"],
            "x2": D["x2"],
            "x3": D["x3"],
            "fac": fac,
            "y": D["y"],
            "w": D["w"],
            "off": D["off"],
            "yp": D["yp"],
        }
    )


def _fit_values(m, hat: bool) -> dict:
    got = {
        "coef": np.asarray(m.coef, dtype=float),
        "sp": np.asarray(m.sp, dtype=float),
        "full_sp": np.asarray(m.full_sp, dtype=float),
        "sig2": m.scale,
        "edf": np.asarray(m.edf, dtype=float),
        "edf1": np.asarray(m.edf1, dtype=float),
        "Vp": np.asarray(m.Vp, dtype=float),
        "Ve": np.asarray(m.Ve, dtype=float),
        "gcv": m.GCV_score,
        "aic": m._mgcv_aic,
        "deviance": m.deviance,
        "null_deviance": m.null_deviance,
        "fitted": np.asarray(m.fitted_values, dtype=float),
        "rank": float(m.rank),
    }
    if hat:
        got["hat"] = np.asarray(m.leverage, dtype=float)
    return got


_GAM_CASES = {
    "s2": ("y ~ s(x1) + s(x2)", {}),
    "w": ("y ~ s(x1) + s(x2)", {"weights": "w"}),
    "off": ("y ~ s(x1) + s(x2) + offset(off)", {}),
    "ubre": ("y ~ s(x1) + s(x2)", {"scale": 0.1}),
    "id": ("y ~ s(x1, id = 1) + s(x2, id = 1)", {}),
    "pfix": ("y ~ s(x1) + s(x2) + s(x3)", {"sp": [-1, 0.5, -1]}),
    "afix": ("y ~ s(x1) + s(x2)", {"sp": [0.1, 2]}),
    "zsp": ("y ~ s(x1) + s(x2)", {"sp": [0, 2]}),
    "param": ("y ~ x1 + x2 + fac", {}),
    "te": ("y ~ te(x1, x2)", {}),
    "gamma": ("y ~ s(x1) + s(x2)", {"gamma": 1.4}),
    "ridge": ("y ~ s(x1) + s(x2)", {"control": gam_control(irls_reg=0.1)}),
    "by": ("y ~ fac + s(x1, by = fac)", {}),
    "cr3": ('y ~ s(x1, bs = "cr") + s(x2, bs = "cr") + s(x3, bs = "cr")', {}),
}


@pytest.mark.skipif(not HAVE_RS, reason="the bases' reference LAPACK lives in hea._rs")
@pytest.mark.parametrize("case", list(_GAM_CASES))
def test_gam_am_fit_matches_mgcv(case):
    """A Gaussian identity gam selected by GCV.Cp is mgcv's ``am.fit``:
    magic on the rows of positive weight, then ``magic.post.proc`` — with
    weights (some zero), an offset, a known scale (UBRE), id-linked,
    partly and wholly fixed smoothing parameters (one of them zero), no
    smooths, a tensor product, gamma, the ``irls.reg`` ridge, a factor
    ``by`` and several cr smooths."""
    df = _data()
    formula, kw = _GAM_CASES[case]
    kw = dict(kw)
    if "weights" in kw:
        kw["weights"] = df[kw["weights"]].to_numpy()
    m = hea.models.gam(formula, df, **kw)
    want = _load(f"gam_{case}")
    got = _fit_values(m, hat=True)
    if "weights" in kw:
        got["hat"] = got["hat"][kw["weights"] > 0]
    for name, value in got.items():
        if want.get(name) is not None:
            _same(value, want[name], name)
    conv = want.get("conv")
    if conv is not None:
        mc = m._mgcv_conv
        assert (mc["iter"], int(mc["fully_converged"]), int(mc["hess_pos_def"])) == (
            int(conv[0]),
            int(conv[1]),
            int(conv[2]),
        )
        _same(mc["rms_grad"], conv[3], "rms_grad", rtol=1e-4, atol=1e-12)


_BAM_CASES = {
    "s2": ("y ~ s(x1) + s(x2)", {}),
    "w": ("y ~ s(x1) + s(x2)", {"weights": "w"}),
    "ubre": ("y ~ s(x1) + s(x2)", {"scale": 0.1}),
    "id": ("y ~ s(x1, id = 1) + s(x2, id = 1)", {}),
    "pfix": ("y ~ s(x1) + s(x2) + s(x3)", {"sp": [-1, 0.5, -1]}),
    "afix": ("y ~ s(x1) + s(x2)", {"sp": [0.1, 2]}),
    "pois": ("yp ~ s(x1) + s(x2)", {"family": Poisson()}),
    "pois_off": ("yp ~ s(x1) + s(x2) + offset(off)", {"family": Poisson()}),
    "pois_gcv": ("yp ~ s(x1) + s(x2)", {"family": Poisson(), "scale": -1}),
}


@pytest.mark.skipif(not HAVE_RS, reason="the bases' reference LAPACK lives in hea._rs")
@pytest.mark.parametrize("case", list(_BAM_CASES))
def test_bam_gcv_matches_mgcv(case):
    """bam(method = "GCV.Cp") runs magic on the accumulated QR and reports
    magic's coefficients, smoothing parameters, score and scale with
    ``magic.post.proc``'s covariances and edf: Gaussian (``bam.fit``) and
    Poisson (each ``bgam.fit`` PIRLS iteration), with weights, a known
    scale, id linkage, fixed smoothing parameters and an offset."""
    df = _data()
    formula, kw = _BAM_CASES[case]
    kw = dict(kw)
    if "weights" in kw:
        kw["weights"] = df[kw["weights"]].to_numpy()
    m = hea.models.bam(formula, df, method="GCV.Cp", **kw)
    want = _load(f"bam_{case}")
    for name, value in _fit_values(m, hat=False).items():
        if want.get(name) is not None:
            _same(value, want[name], name)


def test_offset_null_deviance_matches_glm():
    """With an intercept and an offset, gam's null deviance is that of
    ``glm(y ~ offset(off), family, weights)`` (estimate.gam), R's glm.fit
    iterations included."""
    df = _data()
    want = _load("null_deviance")["null_deviance"]
    m_p = hea.models.gam(
        "yp ~ s(x1) + offset(off)", df, family=Poisson(), method="REML"
    )
    m_g = hea.models.gam("y ~ s(x1) + offset(off)", df, method="REML")
    _same([m_p.null_deviance, m_g.null_deviance], want, "null.deviance")


@pytest.mark.skipif(not HAVE_RS, reason="the bases' reference LAPACK lives in hea._rs")
def test_bam_id_linked_basis_matches_mgcv():
    """bam's sweep-and-drop constraint on id-linked smooths takes the
    constraint and the dropped column from the pooled basis construction
    (smoothCon, before the model matrix is replaced by each smooth's own
    data), so the intercept is mgcv's as well as the smooth coefficients."""
    m = hea.models.bam("y ~ s(x1, id = 1) + s(x2, id = 1)", _data(), method="fREML")
    _same(np.asarray(m.coef, dtype=float), _load("bam_id_freml")["coef"], "coef")


@pytest.mark.skipif(not HAVE_RS, reason="the bases' reference LAPACK lives in hea._rs")
def test_check_reports_magic_convergence(capsys):
    """gam.check's convergence lines for a magic fit come from
    ``mgcv.conv`` (plots.r:316-326)."""
    m = hea.models.gam("y ~ s(x1) + s(x2)", _data())
    mc = m._mgcv_conv
    m.check(plots=False, seed=1)
    text = capsys.readouterr().out
    assert "Method: GCV   Optimizer: magic" in text
    assert (
        f"Smoothing parameter selection converged after {mc['iter']} iterations."
        in text
    )
    assert (
        f"The RMS GCV score gradient at convergence was {mc['rms_grad']:.7g} ." in text
    )
    assert "The Hessian was positive definite." in text
