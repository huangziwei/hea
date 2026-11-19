"""mgcv's ``gam.fit4`` against mgcv itself: ``hea.models._fit4.gam_fit4`` on
the arguments mgcv's own fits passed to ``gam.fit4`` (the first call of each
fit and the last), compared with every numeric value gam.fit4 returned. The
fixtures in ``tests/fixtures/gam_fit4`` are written by
``tests/scripts/make_gdi2_fixtures.R`` from fits of nb (REML, ML, a fixed
theta, zero prior weights), tw, scat, betar, ocat and ziP.

Under ``PINNED_ARITHMETIC`` every value must be mgcv's to the bit.
Elsewhere ``K`` and ``rV`` are compared through ``K K'`` and ``rV rV'`` and
each array to ``1e-10`` of the larger of 1 and its largest entry. hea's
ocat numbers its classes from 0, mgcv's from 1, which moves only gam.fit4's
provisional null deviance (taken at the labels' weighted mean; the family's
postproc replaces it).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC, load_hex_fixture

_FIT4 = FIXTURE_ROOT / "gam_fit4"
_CASES = sorted(p.name for p in _FIT4.glob("*_[12].txt.gz"))


def _family(case: str):
    from hea import family as F

    kind = case.rsplit("_", 1)[0]
    if kind == "nb_fixed_reml":
        return F.nb(theta=3.0)
    if kind.startswith("nb"):
        return F.nb()
    if kind.startswith("tw"):
        return F.tw()
    if kind.startswith("scat"):
        return F.scat()
    if kind.startswith("betar"):
        return F.betar()
    if kind.startswith("ocat"):
        return F.ocat(R=3)
    if kind.startswith("zip"):
        return F.ziP()
    raise KeyError(case)


def _num(a: dict, key: str) -> float:
    return float(np.ravel(a[key])[0])


@pytest.mark.parametrize("name", _CASES)
def test_gam_fit4_matches_mgcv(name):
    """The coefficients, fit, deviance, the REML/ML score and its
    derivatives, ``rV``, ``db.drho``, ``dw.drho``, ``K``, the scale and the
    AIC are mgcv's."""
    from hea.models._fit4 import gam_fit4

    fx = load_hex_fixture(_FIT4 / name)
    a = {k[2:]: v for k, v in fx.items() if k.startswith("a_")}
    want = {k[2:]: v for k, v in fx.items() if k.startswith("o_")}
    fam = _family(name)
    if fam.n_theta == 0:
        fam.set_theta(np.ravel(a["theta"]))
    UrS = [a[f"UrS{i + 1}"] for i in range(int(_num(a, "nUrS")))]
    y = np.ravel(a["y"])
    if name.startswith("ocat"):
        y = y - 1
    got = gam_fit4(
        a["x"],
        y,
        np.ravel(a["sp"]),
        a["Eb"],
        UrS,
        family=fam,
        weights=a["weights"],
        start=a["start"],
        etastart=a["etastart"],
        mustart=a["mustart"],
        offset=a["offset"],
        U1=a["U1"],
        Mp=_num(a, "Mp"),
        epsilon=_num(a, "epsilon"),
        maxit=int(_num(a, "maxit")),
        deriv=int(_num(a, "deriv")),
        gamma=_num(a, "gamma"),
        scale=_num(a, "scale"),
        score_type=a["scoreType"],
        null_coef=a["null.coef"],
    )
    for key, w in want.items():
        if key == "null.deviance" and name.startswith("ocat"):
            continue
        g = got[key]
        if w is None:
            assert g is None, key
            continue
        w = np.asarray(w, dtype=float)
        g = np.asarray(g, dtype=float).reshape(w.shape)
        if PINNED_ARITHMETIC:
            np.testing.assert_array_equal(g, w, err_msg=key)
            continue
        if key in ("K", "rV"):
            g, w = g @ g.T, w @ w.T
        finite = np.isfinite(w)
        scale = max(1.0, float(np.max(np.abs(w[finite])))) if finite.any() else 1.0
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-10 * scale, err_msg=key)


# fits whose L-BFGS-B has three or more parameters: off the pinned
# arithmetic its 2 col-term dot products round otherwise, and the optimizer
# stops elsewhere within its tolerance (factr 1e7) - (fitted values, report,
# smoothing and family parameters)
_OFF_OPTIM = {"tw_optim": (1e-5, 1e-5, 1e-3)}

# The fits the fixtures record, as hea calls them (ocat's classes 0-based).
_SETUP = {
    "nb_reml": ("ynb ~ s(x1) + s(x2)", "REML", {}),
    "nb_ml": ("ynb ~ s(x1) + s(x2)", "ML", {}),
    "nb_fixed_reml": ("ynb ~ s(x1) + s(x2)", "REML", {}),
    "tw_reml": ("ytw ~ s(x1) + s(x2)", "REML", {}),
    "scat_reml": ("yt ~ s(x1) + s(x2)", "REML", {}),
    "betar_reml": ("ybr ~ s(x1) + s(x2)", "REML", {}),
    "ocat_reml": ("yoc ~ s(x1) + s(x2)", "REML", {}),
    "zip_reml": ("yzp ~ s(x1) + s(x2)", "REML", {}),
    "nb_reml_w0": ("ynb ~ s(x1) + s(x2)", "REML", {"weights": True}),
    "nb_efs": ("ynb ~ s(x1) + s(x2)", "REML", {"optimizer": "efs"}),
    "nb_nlm": ("ynb ~ s(x1) + s(x2)", "REML", {"optimizer": ("outer", "nlm")}),
    "nb_ml_optim": ("ynb ~ s(x1) + s(x2)", "ML", {"optimizer": ("outer", "optim")}),
    "tw_optim": ("ytw ~ s(x1) + s(x2)", "REML", {"optimizer": ("outer", "optim")}),
    "tw_efs": ("ytw ~ s(x1) + s(x2)", "REML", {"optimizer": "efs"}),
}


def _fit_spied(case, monkeypatch):
    """The fixture fit by hea, with every gam.fit4 call it makes."""
    import importlib
    import warnings

    import polars as pl

    import hea

    fit4_mod = importlib.import_module("hea.models._fit4")
    data = load_hex_fixture(_FIT4 / "data.txt.gz")
    cols = ("x1", "x2", "ynb", "ytw", "yt", "ybr", "yoc", "yzp", "w0")
    df = pl.DataFrame({k: np.ravel(data[k]) for k in cols})
    df = df.with_columns(pl.col("yoc") - 1)
    form, method, extra = _SETUP[case]
    kw = {"weights": df["w0"].to_numpy()} if extra.get("weights") else {}
    if "optimizer" in extra:
        kw["optimizer"] = extra["optimizer"]
    calls: list = []
    real = fit4_mod.gam_fit4

    def spy(*a, **k):
        calls.append(
            (tuple(np.array(v) if isinstance(v, np.ndarray) else v for v in a), k)
        )
        return real(*a, **k)

    monkeypatch.setattr(fit4_mod, "gam_fit4", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = hea.models.gam(form, df, family=_family(case + "_1"), method=method, **kw)
    return m, calls


@pytest.mark.parametrize("case", sorted(_SETUP))
def test_extended_gam_follows_mgcv_newton_path(case, monkeypatch):
    """hea's gam calls gam.fit4 as mgcv's newton does: the first call gets
    mgcv's arguments (model matrix, the family's starting parameters, log
    smoothing parameters and log scale, penalty roots, null coefficients,
    ``U1``, ``Eb``) and the last, after the same Newton steps, mgcv's, so
    the fit ends on mgcv's coefficients. Under other arithmetic the fitted
    values are compared."""
    m, calls = _fit_spied(case, monkeypatch)
    first = load_hex_fixture(_FIT4 / f"{case}_1.txt.gz")
    last = load_hex_fixture(_FIT4 / f"{case}_2.txt.gz")
    if not PINNED_ARITHMETIC:
        np.testing.assert_allclose(
            np.asarray(m.fitted_values, dtype=float),
            np.ravel(last["o_fitted.values"]),
            rtol=_OFF_OPTIM.get(case, (1e-10,))[0],
            atol=0,
        )
        return
    a, k = calls[0]
    for key, got in (
        ("x", a[0]),
        ("sp", a[2]),
        ("Eb", a[3]),
        ("null.coef", k["null_coef"]),
        ("U1", k["U1"]),
        ("weights", k["weights"]),
    ):
        np.testing.assert_array_equal(
            np.ravel(got), np.ravel(first["a_" + key]), err_msg=key
        )
    for i in range(int(_num(first, "a_nUrS"))):
        np.testing.assert_array_equal(a[4][i], first[f"a_UrS{i + 1}"], err_msg="UrS")
    a, k = calls[-1]
    np.testing.assert_array_equal(np.ravel(a[2]), np.ravel(last["a_sp"]), err_msg="sp")
    for key in ("start", "mustart"):
        w = last["a_" + key]
        if w is None:
            assert k[key] is None, key
        else:
            np.testing.assert_array_equal(np.ravel(k[key]), np.ravel(w), err_msg=key)
    np.testing.assert_array_equal(m._beta, np.ravel(last["o_coefficients"]))


@pytest.mark.parametrize("case", sorted(_SETUP))
def test_extended_gam_post_fit_matches_mgcv(case, monkeypatch):
    """What the fitted gam reports is mgcv's: coefficients, ``Vp``, ``Ve``,
    ``Vc``, edfs, leverages, scale, AIC and logLik, the score, the outer
    Hessian, the deviances, rank, ``R`` and the family's parameters."""
    m, _ = _fit_spied(case, monkeypatch)
    want = load_hex_fixture(_FIT4 / f"{case}_gam.txt.gz")
    score = (m.REML_criterion if m._reml_ind == 1.0 else m.ML_criterion) / 2
    got = {
        "coefficients": m._beta,
        "Vp": m.Vp,
        "Ve": m.Ve,
        "Vc": m.Vc,
        "edf": m.edf,
        "edf1": m.edf1,
        "edf2": m.edf2,
        "hat": m.leverage,
        "scale": m.scale,
        "aic": m._mgcv_aic,
        "gcv.ubre": score,
        "deviance": m.deviance,
        "rank": m.rank,
        "sp": m.sp,
        "hess": (m._outer_info or {}).get("hess"),
        "R": m._object_R(),
        "df.residual": m.df_residuals,
        "fitted.values": m.fitted_values,
        "linear.predictors": m.linear_predictors,
        "weights": m._fisher_w,
        "logLik": m.logLik,
        "AIC": m.AIC,
        "theta": m.family.get_theta(),
    }
    for key, g in got.items():
        w = want[key]
        if w is None:
            continue
        w = np.asarray(w, dtype=float)
        g = np.asarray(g, dtype=float).reshape(w.shape)
        if PINNED_ARITHMETIC:
            np.testing.assert_array_equal(g, w, err_msg=key)
            continue
        if key == "R":
            g, w = g.T @ g, w.T @ w
        if key in ("coefficients", "Vp", "Ve", "Vc", "R"):
            g, w = np.abs(g), np.abs(w)
        off, off_sp = _OFF_OPTIM.get(case, (0, 1e-10, 1e-7))[1:]
        tol = off_sp if key in ("sp", "theta") else off
        scale = max(1.0, float(np.max(np.abs(w)))) if w.size else 1.0
        np.testing.assert_allclose(g, w, rtol=0, atol=tol * scale, err_msg=key)
