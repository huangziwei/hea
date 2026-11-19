"""mgcv's ``gam.fit3`` against mgcv itself: ``hea.models._fit3.gam_fit3``
on the arguments mgcv's own fits passed to ``gam.fit3`` (the first call of
each fit, from scratch, and the last, from the previous fit's coefficients
and means), compared with every numeric value gam.fit3 returned. The
fixtures in ``tests/fixtures/gam_fit3`` are written by
``tests/scripts/make_gdi1_fixtures.R``, over REML, ML, UBRE and GCV, Fisher
and Newton weights (with negative weights), known and unknown scale, a fixed
penalty and ranks 8 to 28.

Under the arithmetic the fixtures were written with (``PINNED_ARITHMETIC``)
every value must be mgcv's to the bit. Elsewhere the BLAS, the C library's
``log`` and the fused multiply-adds round differently: ``K`` and ``rV`` are
then fixed only up to an orthogonal change of their columns (a pivot tie can
break the other way), so ``K K'`` and ``rV rV'`` are compared, and each array
to ``1e-10`` of the larger of 1 and its largest entry, the score's
derivatives at an optimum being near zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC, load_hex_fixture

from hea._dispatch import HAVE_RS
from hea.family import Binomial, Gamma, Gaussian, Poisson

_FIT3 = FIXTURE_ROOT / "gam_fit3"
_CASES = sorted(p.name for p in _FIT3.glob("*_[12].txt.gz"))
_FAMILY = {
    "gaussian": Gaussian,
    "poisson": Poisson,
    "binomial": Binomial,
    "Gamma": Gamma,
}

pytestmark = pytest.mark.skipif(not HAVE_RS, reason="gam_fit3 uses the compiled gdi1")

# off the pinned arithmetic: the fitted values of a whole fit, relative
_OFF_MU = 1e-10
# off the pinned arithmetic: what the fitted gam reports, to this fraction of
# the larger of 1 and each array's largest entry; the smoothing parameters,
# which a flat direction lets move further, to the second
_OFF_POST = 1e-10
_OFF_POST_SP = 1e-7
# fits whose L-BFGS-B has three or more parameters: off the pinned
# arithmetic its 2 col-term dot products round otherwise, and the optimizer
# stops elsewhere within its tolerance (factr 1e7) - (fitted values, report,
# smoothing parameters)
_OFF_OPTIM = {"gauss_reml_optim": (1e-8, 1e-8, 1e-5)}


def _num(a: dict, key: str) -> float:
    return float(np.ravel(a[key])[0])


@pytest.mark.parametrize("name", _CASES)
def test_gam_fit3_matches_mgcv(name):
    """The coefficients, fit, deviance, score and its derivatives (REML or
    GCV/UBRE), ``rV``, ``db.drho``, ``dw.drho``, ``K``, the scale estimates
    and the AIC are mgcv's; where mgcv returns NULL, so does the port."""
    from hea.models._fit3 import gam_fit3

    fx = load_hex_fixture(_FIT3 / name)
    a = {k[2:]: v for k, v in fx.items() if k.startswith("a_")}
    want = {k[2:]: v for k, v in fx.items() if k.startswith("o_")}
    UrS = [a[f"UrS{i + 1}"] for i in range(int(_num(a, "nUrS")))]
    got = gam_fit3(
        a["x"],
        a["y"],
        np.ravel(a["sp"]),
        a["Eb"],
        UrS,
        family=_FAMILY[a["family"]](link=a["link"]),
        weights=a["weights"],
        start=a["start"],
        etastart=a["etastart"],
        mustart=a["mustart"],
        offset=a["offset"],
        U1=a["U1"],
        Mp=_num(a, "Mp"),
        epsilon=_num(a, "epsilon"),
        maxit=int(_num(a, "maxit")),
        intercept=bool(_num(a, "intercept")),
        deriv=int(_num(a, "deriv")),
        gamma=_num(a, "gamma"),
        scale=_num(a, "scale"),
        score_type=a["scoreType"],
        null_coef=a["null.coef"],
        pearson_extra=_num(a, "pearson.extra"),
        dev_extra=_num(a, "dev.extra"),
        n_true=_num(a, "n.true"),
    )
    for key, w in want.items():
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


# The fits the fixtures record, as hea calls them (yp and yb are R integers).
_SETUP = {
    "gauss_reml": ("y ~ s(x1) + s(x2)", Gaussian, {}, "REML", {}),
    "pois_reml": ("yp ~ s(x1) + s(x2)", Poisson, {}, "REML", {}),
    "binom_ml": ("yb ~ s(x1) + s(x2)", Binomial, {}, "ML", {}),
    "gauss_ml_cr": ("y ~ s(x1, bs = 'cr') + s(x2, bs = 'cr')", Gaussian, {}, "ML", {}),
    "pois_flat": ("yp ~ s(x1) + s(x2) + s(x3)", Poisson, {}, "REML", {}),
    "gauss_parapen1_sp": (
        "y ~ s(x1) + x3",
        Gaussian,
        {},
        "REML",
        {"paraPen": {"x3": {"S": np.array([[1.0]]), "sp": 0.5}}},
    ),
    "gauss_rankdef": ("y ~ x1 + xc + s(x2)", Gaussian, {}, "REML", {}),
    "gauss_reml_id": ("y ~ s(x1, id = 1) + s(x2, id = 1)", Gaussian, {}, "REML", {}),
    "gauss_reml_nlm": (
        "y ~ s(x1) + s(x2)",
        Gaussian,
        {},
        "REML",
        {"optimizer": ("outer", "nlm")},
    ),
    "pois_reml_optim": (
        "yp ~ s(x1) + s(x2)",
        Poisson,
        {},
        "REML",
        {"optimizer": ("outer", "optim")},
    ),
    "gauss_reml_optim": (
        "y ~ s(x1) + s(x2)",
        Gaussian,
        {},
        "REML",
        {"optimizer": ("outer", "optim")},
    ),
    "gamma_log_gcv_nlm": (
        "yg ~ s(x1) + s(x2)",
        Gamma,
        {"link": "log"},
        "GCV.Cp",
        {"optimizer": ("outer", "nlm")},
    ),
    "gauss_reml_efs": ("y ~ s(x1) + s(x2)", Gaussian, {}, "REML", {"optimizer": "efs"}),
    "pois_reml_efs": ("yp ~ s(x1) + s(x2)", Poisson, {}, "REML", {"optimizer": "efs"}),
    "gauss_reml_nosmooth": ("y ~ x1 + x2", Gaussian, {}, "REML", {}),
    "pois_reml_nosmooth": ("yp ~ x1 + x2", Poisson, {}, "REML", {}),
    "gamma_log_gcv_nosmooth": ("yg ~ x1 + x2", Gamma, {"link": "log"}, "GCV.Cp", {}),
    "gauss_reml_fixsp": (
        "y ~ s(x1) + s(x2)",
        Gaussian,
        {},
        "REML",
        {"sp": [0.5, 20.0]},
    ),
    "pois_reml_fixsp": ("yp ~ s(x1) + s(x2)", Poisson, {}, "REML", {"sp": [0.5, 20.0]}),
    "gamma_log_gcv_fixsp": (
        "yg ~ s(x1) + s(x2)",
        Gamma,
        {"link": "log"},
        "GCV.Cp",
        {"sp": [0.5, 20.0]},
    ),
    "gauss_min_sp": (
        "y ~ s(x1) + s(x2)",
        Gaussian,
        {},
        "REML",
        {"min_sp": [0.01, 0.1]},
    ),
    "pois_sqrt_reml": ("yp ~ s(x1) + s(x2)", Poisson, {"link": "sqrt"}, "REML", {}),
    "binom_cloglog_ml": ("yb ~ s(x1) + s(x2)", Binomial, {"link": "cloglog"}, "ML", {}),
    "gamma_id_reml": ("yg ~ s(x1) + s(x2)", Gamma, {"link": "identity"}, "REML", {}),
    "gamma_id_ml": ("yg ~ s(x1) + s(x2)", Gamma, {"link": "identity"}, "ML", {}),
    "pois_ubre": ("yp ~ s(x1) + s(x2)", Poisson, {}, "GCV.Cp", {}),
    "gamma_log_gcv": ("yg ~ s(x1) + s(x2)", Gamma, {"link": "log"}, "GCV.Cp", {}),
    "gauss_reml_q9": ("y ~ s(x1, k = 5) + s(x2, k = 5)", Gaussian, {}, "REML", {}),
    "pois_reml_q10": ("yp ~ s(x1, k = 6) + s(x2, k = 5)", Poisson, {}, "REML", {}),
    "binom_ml_q11": ("yb ~ s(x1, k = 6) + s(x2, k = 6)", Binomial, {}, "ML", {}),
    "gamma_id_ml_q8": (
        "yg ~ s(x1, k = 4) + s(x2, k = 5)",
        Gamma,
        {"link": "identity"},
        "ML",
        {},
    ),
    "pois_ubre_q9": ("yp ~ s(x1, k = 5) + s(x2, k = 5)", Poisson, {}, "GCV.Cp", {}),
    "gamma_log_gcv_q10": (
        "yg ~ s(x1, k = 6) + s(x2, k = 5)",
        Gamma,
        {"link": "log"},
        "GCV.Cp",
        {},
    ),
    "gauss_reml_n30": (
        "y ~ s(x1, k = 5) + s(x2, k = 5)",
        Gaussian,
        {},
        "REML",
        {"n": 30},
    ),
    "pois_ubre_n30": (
        "yp ~ s(x1, k = 5) + s(x2, k = 5)",
        Poisson,
        {},
        "GCV.Cp",
        {"n": 30},
    ),
    "gamma_id_reml_n30": (
        "yg ~ s(x1, k = 5) + s(x2, k = 5)",
        Gamma,
        {"link": "identity"},
        "REML",
        {"n": 30},
    ),
}


def _fit_spied(case, monkeypatch):
    """The fixture fit by hea, with every gam.fit3 call it makes."""
    import importlib
    import warnings

    import polars as pl

    import hea

    fit3_mod = importlib.import_module("hea.models._fit3")
    data = load_hex_fixture(_FIT3 / "data.txt.gz")
    df = pl.DataFrame(
        {
            "x1": data["x1"],
            "x2": data["x2"],
            "x3": data["x3"],
            "y": data["y"],
            "yg": data["yg"],
            "yp": data["yp"].astype(np.int64),
            "yb": data["yb"].astype(np.int64),
            "xc": data["xc"],
        }
    )
    form, fam, fam_kw, method, extra = _SETUP[case]
    if "n" in extra:
        df = df.head(extra["n"])
    kw = {k: extra[k] for k in ("min_sp", "paraPen", "sp", "optimizer") if k in extra}
    calls: list = []
    real = fit3_mod.gam_fit3

    def spy(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(fit3_mod, "gam_fit3", spy)
    with warnings.catch_warnings():
        # the rank-deficient design's report
        warnings.filterwarnings("ignore", "model is rank deficient", UserWarning)
        m = hea.models.gam(form, df, family=fam(**fam_kw), method=method, **kw)
    return m, calls


def _same_args(a, k, want, keys):
    got = {
        "x": a[0],
        "y": a[1],
        "sp": a[2],
        "Eb": a[3],
        "null.coef": k["null_coef"],
        "U1": k["U1"],
        "Mp": k["Mp"],
        "weights": k["weights"],
        "offset": k["offset"],
        "start": k["start"],
        "mustart": k["mustart"],
        "epsilon": k["epsilon"],
        "maxit": k["maxit"],
        "scale": k["scale"],
        "gamma": k["gamma"],
        "deriv": k["deriv"],
    }
    for key in keys:
        w = want["a_" + key]
        if w is None:
            assert got[key] is None, key
            continue
        np.testing.assert_array_equal(
            np.ravel(np.asarray(got[key], dtype=float)), np.ravel(w), err_msg=key
        )
    assert k["score_type"] == want["a_scoreType"]
    for i in range(int(_num(want, "a_nUrS"))):
        np.testing.assert_array_equal(
            a[4][i], want[f"a_UrS{i + 1}"], err_msg=f"UrS{i + 1}"
        )


@pytest.mark.parametrize("case", sorted(_SETUP))
def test_gam_follows_mgcv_newton_path(case, monkeypatch):
    """hea's gam calls gam.fit3 as mgcv's newton does. The first call gets
    mgcv's arguments (the model matrix, the initial log smoothing parameters
    with the log scale when it is estimated, the penalty roots ``UrS``, the
    null coefficients, where an integer response keeps R's integer
    ``mean``, ``U1`` and ``Eb``), and the last, after the same Newton
    steps, mgcv's smoothing parameters and starting values, so the fit ends
    on mgcv's coefficients. Under other arithmetic the path cannot be the
    same to the bit, so the fitted values and the score are compared."""
    m, calls = _fit_spied(case, monkeypatch)
    first = load_hex_fixture(_FIT3 / f"{case}_1.txt.gz")
    # a fit with nothing to optimize makes a single gam.fit3 call
    last_file = _FIT3 / f"{case}_2.txt.gz"
    last = load_hex_fixture(last_file) if last_file.exists() else first
    if not PINNED_ARITHMETIC:
        mu = np.asarray(m.fitted_values, dtype=float)
        np.testing.assert_allclose(
            mu,
            np.ravel(last["o_fitted.values"]),
            rtol=_OFF_OPTIM.get(case, (_OFF_MU,))[0],
            atol=0,
        )
        return
    keys = ("x", "y", "sp", "Eb", "null.coef", "U1", "Mp", "weights", "offset")
    keys += ("start", "mustart", "epsilon", "maxit", "scale", "gamma", "deriv")
    _same_args(*calls[0], first, keys)
    _same_args(*calls[-1], last, ("sp", "start", "mustart", "deriv"))
    np.testing.assert_array_equal(m._beta, np.ravel(last["o_coefficients"]))


@pytest.mark.parametrize("case", sorted(_SETUP))
def test_gam_post_fit_matches_mgcv(case, monkeypatch):
    """What the fitted gam reports is what mgcv's does: coefficients, the
    gam.fit3.post.proc covariances (``Vp``, ``Ve``, ``Vc``), edfs and
    leverages, scale, AIC and logLik, the score, the outer Newton Hessian,
    the deviances, rank and the QR factor ``R``. Under other arithmetic a
    smooth's basis vectors can come out with the other sign, so the
    coefficients and covariances are compared in absolute value and ``R``
    through ``R'R``."""
    m, _ = _fit_spied(case, monkeypatch)
    want = load_hex_fixture(_FIT3 / f"{case}_gam.txt.gz")
    score = (
        m.GCV_score
        if not m._is_laplace
        else ((m.REML_criterion if m._reml_ind == 1.0 else m.ML_criterion) / 2)
    )
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
        "null.deviance": m.null_deviance,
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
        off, off_sp = _OFF_OPTIM.get(case, (0, _OFF_POST, _OFF_POST_SP))[1:]
        tol = off_sp if key == "sp" else off
        scale = max(1.0, float(np.max(np.abs(w)))) if w.size else 1.0
        np.testing.assert_allclose(g, w, rtol=0, atol=tol * scale, err_msg=key)
