"""mgcv-oracle regression tests for hea.family.

Pins per-link and per-family numerics against the canonical R/mgcv values
generated with stats::make.link / stats::<family>() / mgcv:::fix.family.*
at fixed (μ, η, scale) inputs. Test points cover the boundary of each
link's domain so that overflow/underflow paths get exercised too.

File sections:

1. Standard link / family numerics (Logit, Probit, …, Gaussian, Poisson,
   Binomial, Gamma, InverseGaussian, Tweedie, tw).
2. ``Scat`` family unit tests — ``Dd`` levels 0/1/2, ``ls_extended``,
   ``dev_resids``, ``aic``, ``preinitialize``, link validation,
   ``_estimate_theta`` vs mgcv ``estimate.theta``.
3. ``Scat`` end-to-end ``hea.models.bam(family=Scat(...), discrete=True)``
   parity (simple and ``by=factor`` oracles). These exercise bam's
   PIRLS-with-θ-Newton cadence; they live here because Scat is the
   exotic-family vehicle, and the value the assertions guard is θ/sp
   convergence, not bam-internal mechanics tested in ``test_bam.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import hea
from hea.family import (
    Binomial,
    CauchitLink,
    CloglogLink,
    Gamma,
    Gaussian,
    InverseGaussian,
    InverseSquareLink,
    LogitLink,
    Poisson,
    ProbitLink,
    Scat,
    SqrtLink,
    Tweedie,
    tw,
)


MUS = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
ETAS = np.array([-2.5, -0.5, 0.0, 0.7, 2.0])


# ---------------------------------------------------------------------------
# Links — values pinned to R::stats::make.link + mgcv:::fix.family.link.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,link_oracle,linkinv_oracle,mu_eta_oracle,d2_oracle,d3_oracle,d4_oracle",
    [
        (
            LogitLink,
            [-2.9444389791664403, -1.3862943611198906, 0.0, 1.3862943611198908, 2.9444389791664394],
            [0.07585818002124356, 0.37754066879814546, 0.5, 0.66818777216816616, 0.88079707797788243],
            [0.070103716545108177, 0.23500371220159449, 0.25, 0.22171287329310907, 0.10499358540350652],
            [-398.89196675900268, -23.437499999999996, 0.0, 23.437500000000014, 398.89196675900212],
            [16002.33270155999, 253.90624999999994, 32.0, 253.90625000000017, 16002.332701559952],
            [-959992.63357402082, -3735.3515624999991, 0.0, 3735.3515625000032, 959992.63357401767],
        ),
        (
            ProbitLink,
            [-1.6448536269514726, -0.84162123357291418, 0.0, 0.84162123357291441, 1.6448536269514715],
            [0.0062096653257761349, 0.30853753872598694, 0.5, 0.75803634777692697, 0.97724986805182079],
            [0.01752830049356854, 0.35206532676429952, 0.3989422804014327, 0.31225393336676127, 0.053990966513188063],
            [-154.6356833289386, -10.737885188829354, 0.0, 10.737885188829367, 154.63568332893792],
            [5843.9347164857427, 110.1329652673685, 15.749609945722417, 110.13296526736866, 5843.9347164857027],
            [-337755.43402393488, -1541.2451459794413, 0.0, 1541.2451459794445, 337755.4340239318],
        ),
        (
            CauchitLink,
            [-6.3137515146750438, -1.3763819204711736, 0.0, 1.3763819204711742, 6.3137515146750376],
            [0.12111894159084341, 0.35241638234956674, 0.5, 0.69440011221421472, 0.85241638234956674],
            [0.043904811887419404, 0.25464790894703254, 0.31830988618379069, 0.21363079609650382, 0.063661977236758135],
            [-5092.749842851993, -78.637795426380578, 0.0, 78.637795426380677, 5092.7498428519775],
            [305581.72289978294, 1199.5876940407713, 62.012553360599632, 1199.5876940407732, 305581.72289978171],
            [-24446195.30522709, -23852.714815240906, 0.0, 23852.71481524095, 24446195.30522697],
        ),
        (
            CloglogLink,
            [-2.9701952490421637, -1.4999399867595158, -0.36651292058166435, 0.4758849953271107, 1.0971887003649483],
            [0.07880634482448419, 0.45476078810739495, 0.63212055882855767, 0.86651320334191617, 0.99938202101066886],
            [0.075616179917426515, 0.33070429889041808, 0.36787944117144233, 0.26880939818177735, 0.0045662814201279153],
            [-399.54304335262657, -24.377665865346501, -2.5546957604665774, 5.8819458413853711, 88.952114337550896],
            [16000.865329326627, 251.39709691607473, 21.17475642459932, 70.530011682474793, 3261.7900829111441],
            [-959997.46017832356, -3745.1331057436109, -67.169617126383642, 915.99384298698556, 183838.66891563043],
        ),
        (
            SqrtLink,
            [0.22360679774997896, 0.44721359549995793, 0.70710678118654757, 0.89442719099991586, 0.97467943448089633],
            [6.25, 0.25, 0.0, 0.48999999999999994, 4.0],
            [-5.0, -1.0, 0.0, 1.3999999999999999, 4.0],
            [-22.360679774997894, -2.7950849718747368, -0.70710678118654757, -0.3493856214843421, -0.26999430318030371],
            [670.82039324993684, 20.963137289060526, 2.1213203435596428, 0.65509804028314145, 0.42630679449521647],
            [-33541.019662496838, -262.03921611325654, -10.606601717798213, -2.0471813758848167, -1.1218599855137275],
        ),
        (
            InverseSquareLink,
            [400.0, 25.0, 4.0, 1.5624999999999998, 1.10803324099723],
            # linkinv at η<0 is NaN in R; we only check the η>0 entries.
            [np.nan, np.nan, np.nan, 1.1952286093343936, 0.70710678118654746],
            [np.nan, np.nan, np.nan, -0.8537347209531384, -0.17677669529663687],
            [959999.99999999977, 3749.9999999999991, 96.0, 14.648437499999996, 7.3664259789289535],
            [-76799999.99999997, -74999.999999999971, -768.0, -73.242187499999972, -31.016530437595598],
            [7679999999.9999971, 1874999.9999999993, 7680.0, 457.76367187499983, 163.24489703997685],
        ),
    ],
)
def test_link_values_match_mgcv(
    cls, link_oracle, linkinv_oracle, mu_eta_oracle, d2_oracle, d3_oracle, d4_oracle,
):
    lk = cls()
    np.testing.assert_allclose(lk.link(MUS), link_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.link")
    np.testing.assert_allclose(lk.d2link(MUS), d2_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d2link")
    np.testing.assert_allclose(lk.d3link(MUS), d3_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d3link")
    np.testing.assert_allclose(lk.d4link(MUS), d4_oracle, rtol=1e-12, atol=0,
                               err_msg=f"{lk.name}.d4link")
    # InverseSquare's linkinv/mu_eta are only defined for η>0; mask the rest.
    linkinv_oracle = np.asarray(linkinv_oracle, dtype=float)
    mu_eta_oracle = np.asarray(mu_eta_oracle, dtype=float)
    mask = ~np.isnan(linkinv_oracle)
    np.testing.assert_allclose(lk.linkinv(ETAS[mask]), linkinv_oracle[mask],
                               rtol=1e-12, atol=0, err_msg=f"{lk.name}.linkinv")
    np.testing.assert_allclose(lk.mu_eta(ETAS[mask]), mu_eta_oracle[mask],
                               rtol=1e-12, atol=0, err_msg=f"{lk.name}.mu_eta")


@pytest.mark.parametrize("cls", [LogitLink, ProbitLink, CauchitLink, CloglogLink, SqrtLink])
def test_link_round_trip(cls):
    """linkinv(link(μ)) ≈ μ on the link's natural domain."""
    lk = cls()
    mu = MUS if cls is not SqrtLink else np.array([0.1, 0.4, 1.0, 2.5, 9.0])
    np.testing.assert_allclose(lk.linkinv(lk.link(mu)), mu, rtol=1e-12, atol=0)


def test_inverse_square_round_trip():
    lk = InverseSquareLink()
    mu = np.array([0.1, 0.5, 1.0, 2.5, 9.0])
    np.testing.assert_allclose(lk.linkinv(lk.link(mu)), mu, rtol=1e-12, atol=0)


def test_link_valideta():
    # sqrt and 1/μ²  reject η ≤ 0 (matches R make.link).
    assert SqrtLink().valideta(np.array([0.1, 1.0])) is True
    assert SqrtLink().valideta(np.array([0.1, 0.0])) is False
    assert InverseSquareLink().valideta(np.array([0.1, 1.0])) is True
    assert InverseSquareLink().valideta(np.array([0.1, -1.0])) is False
    # Bernoulli-type links accept any finite η.
    assert LogitLink().valideta(np.array([-1e3, 0.0, 1e3])) is True


# ---------------------------------------------------------------------------
# Poisson family — pinned against R::stats::poisson + mgcv::fix.family.{var,ls}.
# ---------------------------------------------------------------------------


def test_poisson_static_fields():
    f = Poisson()
    assert f.name == "poisson"
    assert f.canonical_link_name == "log"
    assert f.scale_known is True
    assert f.is_canonical
    # variance/dvar/d2var
    mu = np.array([0.5, 1.2, 2.1])
    np.testing.assert_array_equal(f.variance(mu), mu)
    np.testing.assert_array_equal(f.dvar(mu), np.ones_like(mu))
    np.testing.assert_array_equal(f.d2var(mu), np.zeros_like(mu))


def test_poisson_oracle():
    f = Poisson()
    y = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    mu = np.array([0.5, 1.2, 2.1, 2.8, 4.5])
    wt = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [1.0, 0.07071377282418145, 0.00483934332227196,
         0.01395722892170814, 0.16081546973479077],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               21.297689743799772, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 1.0),
                               [-10.0236819644984, 0.0, 0.0],
                               rtol=1e-12, atol=0)


def test_poisson_initialize():
    f = Poisson()
    y = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(f.initialize(y, np.ones(3)), y + 0.1)
    with pytest.raises(ValueError, match="negative values"):
        f.initialize(np.array([-1.0, 0.0]), np.ones(2))


def test_poisson_validmu():
    assert Poisson().validmu(np.array([0.1, 1.0, 100.0]))
    assert not Poisson().validmu(np.array([0.0, 1.0]))
    assert not Poisson().validmu(np.array([np.inf, 1.0]))


# ---------------------------------------------------------------------------
# Binomial family — pinned against R::stats::binomial + mgcv.
# ---------------------------------------------------------------------------


def test_binomial_static_fields():
    f = Binomial()
    assert f.name == "binomial"
    assert f.canonical_link_name == "logit"
    assert f.scale_known is True
    mu = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(f.variance(mu), mu * (1 - mu), rtol=1e-12)
    np.testing.assert_allclose(f.dvar(mu), 1 - 2 * mu, rtol=1e-12)
    np.testing.assert_array_equal(f.d2var(mu), -2.0 * np.ones_like(mu))


def test_binomial_bernoulli_oracle():
    f = Binomial()
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    mu = np.array([0.3, 0.7, 0.6, 0.4, 0.85])
    wt = np.ones(5)
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [0.713349887877465, 0.713349887877465, 1.021651247531981,
         1.021651247531981, 0.325037858995550],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               3.79504012981444, rtol=1e-12, atol=0)
    # mgcv ls is identically zero in the Bernoulli case (saturated dbinom = 1).
    np.testing.assert_array_equal(f.ls(y, wt, 1.0), np.zeros(3))


def test_binomial_proportion_oracle():
    """y is the success proportion in [0,1]; wt is the binomial size m."""
    f = Binomial()
    y = np.array([0.2, 0.5, 0.7, 0.0])
    mu = np.array([0.3, 0.4, 0.6, 0.1])
    wt = np.array([5.0, 4.0, 10.0, 3.0])
    np.testing.assert_allclose(
        f.dev_resids(y, mu, wt),
        [0.257320924779854, 0.163287978081021,
         0.432017082870932, 0.632163093946958],
        rtol=1e-12, atol=0,
    )
    np.testing.assert_allclose(f.aic(y, mu, None, wt, len(y)),
                               7.87389855174967, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 1.0),
                               [-3.19455473603545, 0.0, 0.0],
                               rtol=1e-11, atol=0)


def test_binomial_initialize_and_validmu():
    f = Binomial()
    y = np.array([0.0, 0.5, 1.0])
    wt = np.array([1.0, 3.0, 1.0])
    expected = (wt * y + 0.5) / (wt + 1.0)
    # m = wt·y = [0, 1.5, 1] isn't integral → R's NCOL=1 branch warns
    # "non-integer #successes in a binomial glm!" (family-review B6).
    with pytest.warns(UserWarning, match="non-integer #successes"):
        out = f.initialize(y, wt)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    # integral counts (0/1 at unit weights) stay silent...
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f.initialize(np.array([0.0, 1.0]), np.ones(2))
    # ...and quasibinomial never warns (R's template guard
    # "quasibinomial" == "binomial" is false).
    from hea.family import QuasiBinomial
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QuasiBinomial().initialize(y, wt)
    with pytest.raises(ValueError, match="0 <= y <= 1"):
        f.initialize(np.array([-0.1, 0.5]), np.ones(2))
    assert f.validmu(np.array([0.01, 0.5, 0.99]))
    assert not f.validmu(np.array([0.0, 0.5]))
    assert not f.validmu(np.array([0.5, 1.0]))


def test_binomial_dev_resids_no_warning_on_boundary():
    """y=0,μ=0 and y=1,μ=1 must yield 0 contribution without numpy warnings."""
    f = Binomial()
    with np.errstate(divide="raise", invalid="raise"):
        d = f.dev_resids(
            np.array([0.0, 1.0, 0.5]),
            np.array([1e-15, 1.0 - 1e-15, 0.5]),
            np.ones(3),
        )
    assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# InverseGaussian family — pinned against R::stats::inverse.gaussian + mgcv.
# ---------------------------------------------------------------------------


def test_inverse_gaussian_static_fields():
    f = InverseGaussian()
    assert f.name == "inverse.gaussian"
    assert f.canonical_link_name == "1/mu^2"
    assert f.scale_known is False
    mu = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(f.variance(mu), mu ** 3, rtol=1e-12)
    np.testing.assert_allclose(f.dvar(mu), 3 * mu ** 2, rtol=1e-12)
    np.testing.assert_allclose(f.d2var(mu), 6 * mu, rtol=1e-12)


def test_inverse_gaussian_oracle():
    f = InverseGaussian()
    y = np.array([1.0, 2.0, 0.5, 3.0])
    mu = np.array([0.9, 2.1, 0.6, 2.5])
    wt = np.array([1.0, 1.0, 2.0, 1.0])
    dev_v = f.dev_resids(y, mu, wt)
    np.testing.assert_allclose(
        dev_v,
        [0.01234567901234567, 0.00113378684807256,
         0.11111111111111106, 0.01333333333333333],
        rtol=1e-12, atol=0,
    )
    dev = float(dev_v.sum())
    np.testing.assert_allclose(f.aic(y, mu, dev, wt, len(y)),
                               -0.546674508250539, rtol=1e-12, atol=0)
    # log-scale derivatives: d1 = -nobs/2 = -2; d2 = 0 (algebraic
    # cancellation since the log(2π φ y³) term is linear in log φ).
    np.testing.assert_allclose(f.ls(y, wt, 0.5),
                               [-3.59080461442099, -2.0, 0.0],
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.ls(y, wt, 2.0),
                               [-6.36339333666077, -2.0, 0.0],
                               rtol=1e-12, atol=0)


def test_inverse_gaussian_initialize():
    f = InverseGaussian()
    y = np.array([0.5, 1.0, 2.5])
    np.testing.assert_array_equal(f.initialize(y, np.ones(3)), y)
    with pytest.raises(ValueError, match="positive values"):
        f.initialize(np.array([0.0, 1.0]), np.ones(2))


# ---------------------------------------------------------------------------
# Existing families should not regress.  Gamma.ls is now log-scale; pin the
# converted values so a future refactor doesn't silently drift back.
# ---------------------------------------------------------------------------


def test_gaussian_unchanged_oracle():
    # ls(y=μ=[1..4], wt=1, scale): log-scale convention (d1 = -n/2, d2 = 0).
    f = Gaussian()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = f.ls(y, np.ones(4), 1.0)
    np.testing.assert_allclose(out, [-0.5 * 4 * np.log(2 * np.pi), -2.0, 0.0],
                               rtol=1e-12, atol=0)


def test_gaussian_ls_with_weights_oracle():
    """Pins mgcv's gaussian()$ls form: ls0 = -nobs/2·log(2πφ) + ½·Σ log w[w>0],
    d/d log φ = -nobs/2, d²/d log φ² = 0.  ``nobs`` is the count of w>0
    observations, NOT Σw — weights act as precision multipliers, not as
    sample-size multipliers."""
    f = Gaussian()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    wt = np.array([0.0, 1.5, 2.0, 0.5])              # one zero-weight row
    nobs = 3
    log_w_sum = float(np.sum(np.log(wt[wt > 0])))    # log(1.5)+log(2)+log(0.5)
    # scale=0.5
    expected_ls0 = -0.5 * nobs * np.log(2.0 * np.pi * 0.5) + 0.5 * log_w_sum
    np.testing.assert_allclose(
        f.ls(y, wt, 0.5),
        [expected_ls0, -0.5 * nobs, 0.0],
        rtol=1e-12, atol=0,
    )
    # scale=2.0
    expected_ls0 = -0.5 * nobs * np.log(2.0 * np.pi * 2.0) + 0.5 * log_w_sum
    np.testing.assert_allclose(
        f.ls(y, wt, 2.0),
        [expected_ls0, -0.5 * nobs, 0.0],
        rtol=1e-12, atol=0,
    )


def test_gamma_ls_log_scale_conversion():
    """mgcv returns d/dφ; we apply the chain rule to log φ. Pin both scales."""
    f = Gamma()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    wt = np.ones(4)
    # mgcv raw form at scale=0.5: [-5.63287638586838, -4.32580552738365, 8.02744183124810]
    # convert: d1_log = 0.5·(-4.32580552738365) = -2.162902763691825
    #          d2_log = 0.5·(-4.32580552738365) + 0.25·(8.02744183124810)
    #                 = -0.156042305879800
    np.testing.assert_allclose(
        f.ls(y, wt, 0.5),
        [-5.63287638586838, -2.162902763691825, -0.156042305879800],
        rtol=1e-10, atol=0,
    )
    # mgcv raw at scale=2: [-8.853807963166636, -1.270362845461478, 0.536662295325308]
    # d1_log = 2·(-1.270362845461478) = -2.540725690922956
    # d2_log = 2·(-1.270362845461478) + 4·(0.536662295325308) = -0.394076509621724
    np.testing.assert_allclose(
        f.ls(y, wt, 2.0),
        [-8.853807963166636, -2.540725690922956, -0.394076509621724],
        rtol=1e-10, atol=0,
    )


# ---------------------------------------------------------------------------
# Family/link composition behaviour
# ---------------------------------------------------------------------------


def test_family_link_resolution():
    # default link = canonical
    assert Poisson().link.name == "log"
    assert Binomial().link.name == "logit"
    assert InverseGaussian().link.name == "1/mu^2"
    # explicit string
    assert Poisson(link="sqrt").link.name == "sqrt"
    assert Binomial(link="probit").link.name == "probit"
    # is_canonical reports correctly
    assert Poisson().is_canonical
    assert not Poisson(link="sqrt").is_canonical
    assert Binomial().is_canonical
    assert not Binomial(link="cloglog").is_canonical


def test_family_link_unknown_raises():
    with pytest.raises(ValueError, match="unknown link"):
        Poisson(link="banana")


# ---------------------------------------------------------------------------
# Tweedie / tw — oracles pinned to mgcv 1.9-4.
#
# All numeric oracles in this section are produced by R/mgcv:
#   Tweedie(p, link='log')$variance / dev.resids / dvar / d2var / d3var
#   ldTweedie(y, mu, rho=log(phi), theta, a, b)              # log f + derivs
# The (rho, theta) parametrisation matches hea's tw() exactly:
#   p(theta) = (a + b·exp(theta))/(1 + exp(theta));  rho = log(phi).
# ---------------------------------------------------------------------------


def test_tweedie_static_fields():
    f = Tweedie(p=1.5)
    assert f.name == "Tweedie"
    assert f.canonical_link_name == "log"
    assert f.scale_known is False
    assert f.link.name == "log"
    assert f.p == 1.5
    assert f.n_theta == 0  # fixed-p Tweedie isn't "estimable"


def test_tweedie_p_out_of_range_raises():
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=1.0)
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=2.0)
    with pytest.raises(ValueError, match="1 < p < 2"):
        Tweedie(p=0.5)


def test_tweedie_validmu_and_initialize():
    f = Tweedie(p=1.5)
    assert f.validmu(np.array([0.1, 1.0, 5.0]))
    assert not f.validmu(np.array([0.0, 1.0]))
    assert not f.validmu(np.array([-0.1, 1.0]))
    y = np.array([0.0, 1.0, 2.5])
    # mgcv bumps only the zeros: mustart = y + 0.1·(y==0)
    # (Tweedie gam.fit3.r:3078, tw efam.r:3234).
    np.testing.assert_allclose(f.initialize(y, np.ones(3)),
                               np.array([0.1, 1.0, 2.5]))
    with pytest.raises(ValueError, match="negative values"):
        f.initialize(np.array([-1.0, 1.0]), np.ones(2))


_TW_MUS = np.array([0.5, 1.0, 2.0, 5.0])


@pytest.mark.parametrize(
    "p, V, dV, d2V, d3V",
    [
        (
            1.1,
            [0.466516495768403705, 1.0, 2.143546925072586262, 5.873094715440095648],
            [1.02633629069048826, 1.10000000000000009, 1.17895080878992253, 1.29208083739682111],
            [0.2052672581380978467, 0.1100000000000001116, 0.0589475404394961822, 0.0258416167479364467],
            [-0.3694810646485760519, -0.0990000000000000879, -0.0265263931977732792, -0.0046514910146285603],
        ),
        (
            1.5,
            [0.353553390593273786, 1.0, 2.828427124746190291, 11.180339887498949025],
            [1.06066017177982141, 1.50000000000000000, 2.12132034355964283, 3.35410196624968471],
            [1.060660171779821415, 0.750000000000000000, 0.530330085889910707, 0.335410196624968460],
            [-1.0606601717798214146, -0.3750000000000000000, -0.1325825214724776768, -0.0335410196624968474],
        ),
        (
            1.9,
            [0.267943365634073283, 1.0, 3.732131966147229640, 21.283498063019610669],
            [1.01818478940947843, 1.89999999999999991, 3.54552536783986794, 8.08772926394745184],
            [1.83273262093706091, 1.70999999999999974, 1.59548641552794046, 1.45579126751054133],
            [-0.3665465241874125146, -0.1710000000000001241, -0.0797743207763970952, -0.0291158253502108513],
        ),
    ],
)
def test_tweedie_variance_oracle(p, V, dV, d2V, d3V):
    """mgcv:::fix.family.var(Tweedie(p, link='log'))$dvar/d2var/d3var at mu=(0.5,1,2,5)."""
    f = Tweedie(p=p)
    np.testing.assert_allclose(f.variance(_TW_MUS), V, rtol=1e-13)
    np.testing.assert_allclose(f.dvar(_TW_MUS), dV, rtol=1e-13)
    np.testing.assert_allclose(f.d2var(_TW_MUS), d2V, rtol=1e-13)
    np.testing.assert_allclose(f.d3var(_TW_MUS), d3V, rtol=1e-13)


_TW_DEV_Y = np.array([0.0, 0.5, 1.0, 2.5, 4.0])
_TW_DEV_MU = np.array([0.6, 0.7, 1.2, 2.0, 3.5])


@pytest.mark.parametrize(
    "p, dev_oracle",
    [
        (
            1.1,
            [1.4032130388652341857, 0.0665577307825964692,
             0.0349267878477985683, 0.1071552861439123427, 0.0599447148388876361],
        ),
        (
            1.5,
            [3.0983866769659336171, 0.0802430753127092444,
             0.0332641767424362023, 0.0788114206843384402, 0.0356745147454633482],
        ),
        (
            1.9,
            [19.0040043301135099796, 0.0968397017685943551,
             0.0316900713608997964, 0.0579905047662896411, 0.0212341107387551409],
        ),
    ],
)
def test_tweedie_dev_resids_oracle(p, dev_oracle):
    """Tweedie(p)$dev.resids(y, mu, wt=1)."""
    f = Tweedie(p=p)
    np.testing.assert_allclose(
        f.dev_resids(_TW_DEV_Y, _TW_DEV_MU, np.ones(5)),
        dev_oracle, rtol=1e-13,
    )


def test_tweedie_dev_resids_weighted_oracle():
    """Tweedie(p=1.5)$dev.resids with non-unit prior weights."""
    f = Tweedie(p=1.5)
    wt = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
    oracle = [1.5491933384829668086, 0.0802430753127092444,
              0.0665283534848724045, 0.0788114206843384402, 0.0178372573727316741]
    np.testing.assert_allclose(
        f.dev_resids(_TW_DEV_Y, _TW_DEV_MU, wt), oracle, rtol=1e-13,
    )


def test_tweedie_dev_resids_zero_at_y_equals_mu():
    f = Tweedie(p=1.5)
    y = np.array([0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(f.dev_resids(y, y, np.ones(4)), 0.0, atol=1e-13)


def test_tweedie_dev_limit_to_poisson():
    """Tweedie deviance → Poisson deviance as p → 1."""
    y = np.array([1.0, 2.0, 5.0, 10.0])
    mu = np.array([1.5, 1.5, 4.0, 8.0])
    pois = Poisson().dev_resids(y, mu, np.ones(4))
    np.testing.assert_allclose(
        Tweedie(p=1.001).dev_resids(y, mu, np.ones(4)), pois, rtol=1e-2,
    )


def test_tweedie_dev_limit_to_gamma():
    """Tweedie deviance → Gamma deviance as p → 2."""
    y = np.array([0.5, 1.0, 2.0, 5.0])
    mu = np.array([0.6, 1.5, 2.5, 4.0])
    gam_dev = Gamma().dev_resids(y, mu, np.ones(4))
    np.testing.assert_allclose(
        Tweedie(p=1.999).dev_resids(y, mu, np.ones(4)), gam_dev, rtol=1e-2,
    )


def test_tweedie_log_density_oracle_p15():
    """ldTweedie(y, mu, rho=0, theta=0, a=1.01, b=1.99)[, 1] at p=1.5, phi=1."""
    f = Tweedie(p=1.5)
    log_f = f._log_density(_TW_DEV_Y, _TW_DEV_MU, phi=1.0)
    oracle = [-1.549193338482966809, -0.608940759999017533,
              -1.045247308713201040, -1.710387087424116714,
              -2.026689918613598707]
    np.testing.assert_allclose(log_f, oracle, rtol=1e-12)


def test_tweedie_log_density_oracle_p17():
    """Same at p=1.7, phi=2.0 — exercises the Dunn-Smyth series at off-default p."""
    f = Tweedie(p=1.7)
    log_f = f._log_density(_TW_DEV_Y, _TW_DEV_MU, phi=2.0)
    oracle = [-1.429862000740157901, -0.970520782376560809,
              -1.491788879150033331, -2.222992852580873091,
              -2.589272021845887117]
    np.testing.assert_allclose(log_f, oracle, rtol=1e-12)


def test_tweedie_ls_saturated_oracle_p15():
    """tw() at default theta=0 ⇒ p=1.5, scale=1; sums of ldTweedie(y, y, ...)[, 1:3]."""
    f = tw()
    assert f.p == pytest.approx(1.5, abs=1e-12)
    ls = f.ls(_TW_DEV_Y, np.ones(5), scale=1.0)
    np.testing.assert_allclose(ls[0], -5.2772684810074608, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -2.4805433077476438, rtol=1e-12)
    np.testing.assert_allclose(ls[2], -0.6812533145450566, rtol=1e-9)


def test_tweedie_ls_saturated_oracle_p17():
    """tw().set_theta(...) → p=1.7, scale=2."""
    f = tw()
    f.set_theta(np.log((1.7 - 1.01) / (1.99 - 1.7)))
    assert f.p == pytest.approx(1.7, abs=1e-12)
    ls = f.ls(_TW_DEV_Y, np.ones(5), scale=2.0)
    np.testing.assert_allclose(ls[0], -7.22064210632427717, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -2.8497276172663244, rtol=1e-12)
    np.testing.assert_allclose(ls[2], -0.837451847912255021, rtol=1e-9)


def test_tweedie_ls_weighted_oracle():
    """Weighted saturated ls — mgcv's convention is the weight OUTSIDE the
    density at unmodified φ: colSums(wt·ldTweedie(y, y, rho=log(φ)))
    (tw()$ls efam.r:3224, fix.family.ls gam.fit3.r:3083). Oracle from
    mgcv 1.9-4 ldTweedie at p=1.5, φ=1.5."""
    f = Tweedie(p=1.5)
    wt = np.array([0.5, 1.0, 2.0, 1.0, 0.5])
    ls = f.ls(_TW_DEV_Y, wt, scale=1.5)
    np.testing.assert_allclose(ls[0], -6.5366223458531341, rtol=1e-12)
    np.testing.assert_allclose(ls[1], -3.2991695964476975, rtol=1e-12)
    np.testing.assert_allclose(ls[2], -1.4641591053552576, rtol=1e-9)


def test_tweedie_ls_zero_weights_dropped():
    """Rows with wt=0 should drop out of ls (mgcv's good-subset convention)."""
    f = Tweedie(p=1.5)
    y = np.array([0.0, 1.0, 2.5])
    wt_drop = np.array([1.0, 0.0, 1.0])
    ls_drop = f.ls(y, wt_drop, 1.0)
    ls_two = f.ls(np.array([0.0, 2.5]), np.array([1.0, 1.0]), 1.0)
    np.testing.assert_allclose(ls_drop, ls_two, rtol=1e-12)


def test_tw_dls_dp_chain_to_theta_oracle_p15():
    """dls/dp · dp/dθ_tw must equal Σ ldTweedie[, 'th'] at default (a=1.01, b=1.99)."""
    f = tw()
    dls_dp = f.dls_dp(_TW_DEV_Y, np.ones(5), scale=1.0)
    np.testing.assert_allclose(dls_dp * f.dp_dtheta(),
                               -0.1740833687231026, rtol=1e-9)


def test_tw_dls_dp_chain_to_theta_oracle_p17():
    f = tw()
    f.set_theta(np.log((1.7 - 1.01) / (1.99 - 1.7)))
    dls_dp = f.dls_dp(_TW_DEV_Y, np.ones(5), scale=2.0)
    np.testing.assert_allclose(dls_dp * f.dp_dtheta(),
                               -0.0302015497694411161, rtol=1e-9)


def test_tw_default_theta_zero_gives_p_15():
    """θ=0 with default (a,b)=(1.01, 1.99) → p = (1.01+1.99)/2 = 1.5."""
    f = tw()
    assert f.theta == 0.0
    assert f.p == pytest.approx(1.50, abs=1e-12)
    assert f.n_theta == 1
    np.testing.assert_array_equal(f.get_theta(), np.array([0.0]))


def test_tw_set_theta_array_or_scalar():
    """set_theta accepts both scalar and length-1 array (Family-base contract)."""
    f = tw()
    f.set_theta(0.5)
    p_scalar = f.p
    f.set_theta(np.array([0.5]))
    np.testing.assert_allclose(f.p, p_scalar, rtol=1e-13)


def test_tw_dp_dtheta_default_a_b():
    f = tw()
    np.testing.assert_allclose(f.dp_dtheta(), 0.245, rtol=1e-13)
    f.set_theta(2.0)
    s = 1.0 / (1.0 + np.exp(-2.0))
    np.testing.assert_allclose(f.dp_dtheta(), 0.98 * s * (1.0 - s), rtol=1e-13)


def test_tw_invalid_a_b_raises():
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=2.0, b=1.5)
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=0.9, b=1.5)
    with pytest.raises(ValueError, match="1 ≤ a < b ≤ 2"):
        tw(a=1.5, b=2.5)


# =============================================================================
# 2. Scat family unit tests
# =============================================================================
#
# Pins ``Scat.Dd`` levels 0/1/2, ``Scat.ls_extended``, ``Scat.dev_resids``,
# ``Scat.aic``, and ``Scat.preinitialize`` against a frozen mgcv oracle
# generated by ``tests/r_oracle/dump_scat_unit.R``.
#
# The pin is per-element to ≤ 5e-13 — well inside double-precision
# roundoff for the cumulative formulas in ``Dd`` level 2, which exercises
# products of up to four ν-, σ- and (y-μ)-dependent quantities.

_SCAT_UNIT = Path(__file__).parent / "fixtures" / "scat_unit"


def _scat_have_unit_fixtures() -> bool:
    return all((_SCAT_UNIT / f).exists() for f in (
        "inputs.csv", "theta.csv", "dd_lvl0.csv", "dd_lvl1.csv",
        "dd_lvl2.csv", "ls_summary.csv", "LSTH1.csv", "dev.csv",
        "aic.csv", "preinit.csv",
    ))


_scat_unit_skip = pytest.mark.skipif(
    not _scat_have_unit_fixtures(),
    reason="scat oracle missing — run tests/r_oracle/dump_scat_unit.R",
)


def _scat_load_inputs():
    df = pl.read_csv(str(_SCAT_UNIT / "inputs.csv"))
    y = df["y"].to_numpy().astype(float)
    mu = df["mu"].to_numpy().astype(float)
    wt = df["wt"].to_numpy().astype(float)
    theta = np.atleast_1d(np.loadtxt(_SCAT_UNIT / "theta.csv"))
    return y, mu, wt, theta


def _scat_unit():
    s = Scat(min_df=5)
    _, _, _, theta = _scat_load_inputs()
    s.set_theta(theta)
    return s


@_scat_unit_skip
def test_scat_dd_level0():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    dd = s.Dd(y, mu, theta, wt, level=0)
    oracle = pl.read_csv(str(_SCAT_UNIT / "dd_lvl0.csv")).to_numpy()
    assert np.max(np.abs(dd["Dmu"] - oracle[:, 0])) < 5e-13
    assert np.max(np.abs(dd["Dmu2"] - oracle[:, 1])) < 5e-13
    assert np.max(np.abs(dd["EDmu2"] - oracle[:, 2])) < 5e-13


@_scat_unit_skip
def test_scat_dd_level1():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    dd = s.Dd(y, mu, theta, wt, level=1)
    oracle = pl.read_csv(str(_SCAT_UNIT / "dd_lvl1.csv")).to_numpy()
    pairs = [
        (dd["Dth"][:, 0],     oracle[:, 0]),
        (dd["Dth"][:, 1],     oracle[:, 1]),
        (dd["Dmuth"][:, 0],   oracle[:, 2]),
        (dd["Dmuth"][:, 1],   oracle[:, 3]),
        (dd["Dmu2th"][:, 0],  oracle[:, 4]),
        (dd["Dmu2th"][:, 1],  oracle[:, 5]),
        (dd["EDmu2th"][:, 0], oracle[:, 6]),
        (dd["EDmu2th"][:, 1], oracle[:, 7]),
        (dd["Dmu3"],          oracle[:, 8]),
        (dd["EDmu3"],         oracle[:, 9]),
    ]
    err = max(float(np.max(np.abs(h - m))) for h, m in pairs)
    assert err < 5e-13, f"Dd level 1 max abs err {err:.3e}"


@_scat_unit_skip
def test_scat_dd_level2():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    dd = s.Dd(y, mu, theta, wt, level=2)
    oracle = pl.read_csv(str(_SCAT_UNIT / "dd_lvl2.csv")).to_numpy()
    pairs = [
        (dd["Dmu4"],          oracle[:, 0]),
        (dd["Dmu3th"][:, 0],  oracle[:, 1]),
        (dd["Dmu3th"][:, 1],  oracle[:, 2]),
        (dd["Dmu2th2"][:, 0], oracle[:, 3]),
        (dd["Dmu2th2"][:, 1], oracle[:, 4]),
        (dd["Dmu2th2"][:, 2], oracle[:, 5]),
        (dd["Dmuth2"][:, 0],  oracle[:, 6]),
        (dd["Dmuth2"][:, 1],  oracle[:, 7]),
        (dd["Dmuth2"][:, 2],  oracle[:, 8]),
        (dd["Dth2"][:, 0],    oracle[:, 9]),
        (dd["Dth2"][:, 1],    oracle[:, 10]),
        (dd["Dth2"][:, 2],    oracle[:, 11]),
    ]
    err = max(float(np.max(np.abs(h - m))) for h, m in pairs)
    assert err < 5e-13, f"Dd level 2 max abs err {err:.3e}"


@_scat_unit_skip
def test_scat_ls_extended():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    ls = s.ls_extended(y, wt, theta=theta, scale=1.0)
    summary = np.atleast_1d(np.loadtxt(_SCAT_UNIT / "ls_summary.csv"))
    assert abs(ls["ls"] - summary[0]) < 5e-13
    assert abs(ls["lsth1"][0] - summary[1]) < 5e-13
    assert abs(ls["lsth1"][1] - summary[2]) < 5e-13
    assert abs(ls["lsth2"][0, 0] - summary[3]) < 5e-13
    LSTH1 = pl.read_csv(str(_SCAT_UNIT / "LSTH1.csv")).to_numpy()
    assert np.max(np.abs(ls["LSTH1"] - LSTH1)) < 5e-13


@_scat_unit_skip
def test_scat_dev_resids():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    dev_h = s.dev_resids(y, mu, wt, theta=theta)
    dev_o = pl.read_csv(str(_SCAT_UNIT / "dev.csv")).to_numpy().ravel()
    assert np.max(np.abs(dev_h - dev_o)) < 5e-13


@_scat_unit_skip
def test_scat_aic():
    y, mu, wt, theta = _scat_load_inputs()
    s = _scat_unit()
    aic_h = s.aic(y, mu, dev=None, wt=wt, n=len(y), theta=theta)
    aic_o = float(np.atleast_1d(np.loadtxt(_SCAT_UNIT / "aic.csv"))[0])
    assert abs(aic_h - aic_o) < 5e-13


@_scat_unit_skip
def test_scat_preinitialize():
    y, _, _, _ = _scat_load_inputs()
    s = Scat(min_df=5)   # n_theta=2 → preinit returns Theta
    pini = s.preinitialize(y)
    assert pini is not None and "Theta" in pini
    pini_o = np.atleast_1d(np.loadtxt(_SCAT_UNIT / "preinit.csv"))
    assert np.max(np.abs(pini["Theta"] - pini_o)) < 5e-13


def test_scat_preinitialize_locked():
    """When user supplies both θ as positive (n_theta=0), preinit
    returns ``None`` so the constructor's iniTheta is used unchanged."""
    s = Scat(theta=(7.0, 0.5), min_df=5)
    assert s.n_theta == 0
    assert s.preinitialize(np.array([1.0, 2.0, 3.0])) is None


def test_scat_get_set_theta_roundtrip():
    s = Scat(min_df=5)
    s.set_theta([0.4, -0.3])
    assert np.allclose(s.get_theta(trans=False), [0.4, -0.3])
    nu, sig = s.get_theta(trans=True)
    assert nu == pytest.approx(np.exp(0.4) + 5.0)
    assert sig == pytest.approx(np.exp(-0.3))


def test_scat_link_validation():
    Scat(link="identity")
    Scat(link="log")
    Scat(link="inverse")
    with pytest.raises(ValueError, match="not available for scat"):
        Scat(link="probit")


def test_tw_ls_extended_lsth2_matches_mgcv():
    """tw's analytic ``lsth2`` (family-review B4 — previously
    NaN-poisoned): the (θ,θ)/(θ,logφ) saturated-likelihood second
    derivatives via ldTweedie's column-5/6 forms (density closed forms
    gam.fit3.r:2802-2806 + Dunn-Smyth series moments) chained through
    p(θ).

    Oracle: R 4.6.0 / mgcv 1.9-4, y = the _mixed_sp_fixture ytw2
    column, w = 1, theta = log(0.49/0.49) for p = 1.5 (a=1.01, b=1.99):
        fam <- tw(); fam$ls(y, w, theta, scale)
        scale=0.8: ls -133.8139693830
                   lsth1 (-4.6929474731, -66.9089188162)
                   lsth2 [-0.7111684232, 2.4000778201;
                          2.4000778201, -13.8069254790]
        scale=1.3: ls -168.3807909970
                   lsth1 (-2.3840557958, -76.5635517288)
                   lsth2 [-3.3107988091, 8.0792314245;
                          8.0792314245, -27.1024409298]
    hea matches all printed digits; pinned at 1e-8.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_gam import _mixed_sp_fixture
    from hea.family import tw
    y = _mixed_sp_fixture()["ytw2"].to_numpy()
    w = np.ones_like(y)
    f = tw()
    expected = {
        0.8: (-133.8139693830, [-4.6929474731, -66.9089188162],
              [[-0.7111684232, 2.4000778201],
               [2.4000778201, -13.8069254790]]),
        1.3: (-168.3807909970, [-2.3840557958, -76.5635517288],
              [[-3.3107988091, 8.0792314245],
               [8.0792314245, -27.1024409298]]),
    }
    for sc, (ls_r, lsth1_r, lsth2_r) in expected.items():
        out = f.ls_extended(y, w, scale=sc)
        np.testing.assert_allclose(out["ls"], ls_r, rtol=0, atol=1e-8)
        np.testing.assert_allclose(out["lsth1"], lsth1_r,
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(out["lsth2"], lsth2_r,
                                   rtol=0, atol=1e-8)
        assert np.all(np.isfinite(out["lsth2"]))


def test_rtweedie_moments_and_rd_hook():
    """mgcv ``rTweedie`` construction (gam.fit3.r:3112-3146) at
    Monte-Carlo level: compound Poisson-Gamma with E[Y] = μ,
    Var[Y] = φ·μ^p, P(Y=0) = exp(−μ^(2−p)/((2−p)φ)). The family rd
    hooks (Tweedie gam.fit3.r:3097, tw efam.r:3245 — tw inherits)
    drive qq.gam's simulation path; mgcv's rd ignores wt, bug-for-bug.
    """
    from hea.family import Tweedie, _r_tweedie, tw
    rng = np.random.default_rng(0)
    n = 200_000
    mu = np.full(n, 2.0)
    p, phi = 1.5, 1.3
    y = _r_tweedie(rng, mu, p, phi)
    lam = mu[0] ** (2 - p) / ((2 - p) * phi)
    assert abs(y.mean() - mu[0]) < 0.02
    assert abs(y.var() - phi * mu[0] ** p) < 0.06
    assert abs(np.mean(y == 0) - np.exp(-lam)) < 0.005
    # hooks present and shape-correct; wt unread (mgcv signature quirk).
    f = Tweedie(p=1.5)
    d = f.rd(np.random.default_rng(1), mu[:100], None, 1.0)
    assert d.shape == (100,) and np.all(d >= 0)
    d2 = tw().rd(np.random.default_rng(1), mu[:100], None, 1.0)
    np.testing.assert_array_equal(d, d2)   # same seed, same p=1.5 start
    with pytest.raises(ValueError, match="must be positive"):
        _r_tweedie(rng, mu[:5], 1.5, 0.0)


def test_link_validation_matches_r_acceptance():
    """Construction-time link validation ≡ R (probed live, R 4.6.0 /
    mgcv 1.9-4, 2026-06-11). The extended families (tw here; scat/nb
    above) enforce okLinks strictly: ``tw(link="logit")`` errs in R
    with 'link "logit" not available for tw family; available links
    are log, identity, sqrt, inverse' (efam.r:3098-3101). The standard
    constructors do NOT: their is.character fallback routes any
    make.link-known name through — ``poisson(link="logit")``,
    ``binomial(link="inverse")``, ``gaussian(link="logit")`` and
    ``Tweedie(1.5, link="logit")`` (gam.fit3.r:3042-3045) all
    construct fine in R (the okLinks message there fires only for
    non-character link objects), and misfits surface downstream as
    link-domain errors, not at construction. hea mirrors both
    behaviors; unknown names error in ``_resolve_link`` like R's
    make.link ('bogus' link not recognised)."""
    from hea.family import (Binomial, Gaussian, Poisson, Tweedie, tw)
    tw(link="log"), tw(link="identity"), tw(link="sqrt"), tw(link="inverse")
    with pytest.raises(ValueError,
                       match='link "logit" not available for tw family'):
        tw(link="logit")
    # R-permissive standard constructors — construction must succeed.
    Poisson(link="logit")
    Binomial(link="inverse")
    Gaussian(link="logit")
    Tweedie(p=1.5, link="logit")
    with pytest.raises(ValueError, match="unknown link"):
        Poisson(link="bogus")


_SCAT_ESTTH = Path(__file__).parent / "fixtures" / "scat_estth"


@pytest.mark.skipif(
    not (_SCAT_ESTTH / "inputs.csv").exists(),
    reason="scat estimate.theta oracle missing — run "
           "tests/r_oracle/dump_scat_estth.R",
)
@pytest.mark.parametrize("init_name", ["near", "far"])
def test_scat_estimate_theta(init_name: str):
    """mgcv ``estimate.theta`` parity for ``_estimate_theta`` on a
    Scat family with heavy-tailed residuals (so ν stays finite and
    both impls converge to a well-defined optimum)."""
    from hea.models.bam import _estimate_theta
    df = pl.read_csv(str(_SCAT_ESTTH / "inputs.csv"))
    y = df["y"].to_numpy().astype(float)
    mu = df["mu"].to_numpy().astype(float)
    th0 = np.atleast_1d(np.loadtxt(_SCAT_ESTTH / f"{init_name}_init.csv"))
    th_oracle = np.atleast_1d(np.loadtxt(_SCAT_ESTTH / f"{init_name}_out.csv"))

    s = Scat(min_df=4)
    s.set_theta(th0)
    th_h = _estimate_theta(s, y, mu, scale=1.0, wt=np.ones(len(y)),
                           tol=1e-7)
    err = float(np.max(np.abs(th_h - th_oracle)))
    assert err < 1e-12, (
        f"{init_name}-init: hea={th_h}, mgcv={th_oracle}, err={err:.3e}"
    )


# =============================================================================
# 3. Scat × bam end-to-end parity
# =============================================================================
#
# Two oracles from ``tests/r_oracle/dump_bam_scat.R``:
#
# * ``simple`` — ``y ~ s(x, k=10)`` on heavy-tailed data; baseline scat
#   PIRLS path.
# * ``factor`` — ``y ~ g + s(x, by=g, k=10)`` on factor-level-shifted
#   heavy-tailed data; exercises the by=factor discrete-path fix
#   together with the extended-family θ-Newton.
#
# Each oracle is checked at two operating points:
#
# * **force-θ-and-sp** — feed mgcv's converged ``(ν, σ)`` and ``sp`` to
#   hea, refit; assert fitted matches mgcv to ≤ 1e-9 (predictive
#   equivalence; gauge-invariant).
# * **auto-fit** — let hea estimate θ and sp from scratch; assert
#   fitted ≤ 1e-9 / ≤ 1e-7 with θ and sp matching to 1e-6.

_SCAT_BAM = Path(__file__).parent / "fixtures" / "scat_bam"


def _scat_bam_have(name: str) -> bool:
    sub = _SCAT_BAM / name
    return all((sub / f).exists() for f in
               ("data.csv", "sp.csv", "theta.csv", "fitted.csv"))


def _scat_bam_load(name: str):
    sub = _SCAT_BAM / name
    df = pl.read_csv(str(sub / "data.csv"))
    sp = np.atleast_1d(np.loadtxt(sub / "sp.csv"))
    theta = np.atleast_1d(np.loadtxt(sub / "theta.csv"))
    fitted = np.loadtxt(sub / "fitted.csv")
    return df, sp, theta, fitted


@pytest.mark.skipif(
    not _scat_bam_have("simple"),
    reason="scat_bam/simple oracle missing — run dump_bam_scat.R",
)
def test_scat_bam_simple_force_theta_sp():
    df, sp_mgcv, theta_mgcv, fit_mgcv = _scat_bam_load("simple")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0   # both θ supplied positive ⇒ locked
    m = hea.models.bam("y ~ s(x, k=10)", dat, family=fam, discrete=True,
                sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"force-(θ,sp) fitted rel diff {rel:.3e}"


@pytest.mark.skipif(
    not _scat_bam_have("simple"),
    reason="scat_bam/simple oracle missing",
)
def test_scat_bam_simple_auto_fit():
    """Auto-fit on the simple oracle. The dev0-under-new-θ recompute
    (mgcv bgam.fitd:567-569) makes hea's PIRLS cadence bit-identical
    to mgcv's, so we pin tightly."""
    df, sp_mgcv, theta_mgcv, fit_mgcv = _scat_bam_load("simple")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    m = hea.models.bam("y ~ s(x, k=10)", dat, family=Scat(min_df=5),
                discrete=True)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"auto-fit fitted rel diff {rel:.3e}"

    theta_h = np.asarray(m.family.get_theta(trans=True))
    # ν and σ are on different absolute scales; rel tol handles both.
    assert np.allclose(theta_h, theta_mgcv, rtol=1e-6, atol=0), (
        f"auto-fit θ mismatch: hea={theta_h} mgcv={theta_mgcv}"
    )

    sp_h = np.asarray(m.sp)
    assert np.allclose(sp_h, sp_mgcv, rtol=1e-6, atol=0), (
        f"auto-fit sp mismatch: hea={sp_h} mgcv={sp_mgcv}"
    )


@pytest.mark.skipif(
    not _scat_bam_have("factor"),
    reason="scat_bam/factor oracle missing",
)
def test_scat_bam_factor_force_theta_sp():
    df, sp_mgcv, theta_mgcv, fit_mgcv = _scat_bam_load("factor")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float),
           "g": df["g"].to_numpy()}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0
    m = hea.models.bam("y ~ g + s(x, by=g, k=10)", dat, family=fam,
                discrete=True, sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"factor force-(θ,sp) fitted rel diff {rel:.3e}"


@pytest.mark.skipif(
    not _scat_bam_have("factor"),
    reason="scat_bam/factor oracle missing",
)
def test_scat_bam_factor_auto_fit():
    """Auto-fit on the factor-by oracle (3 levels × 1 smooth per level).
    Combines extended-family θ-Newton with the by=factor discrete-path
    fix; PIRLS-step-halving with dev0-under-new-θ keeps iterates bit-
    identical to mgcv."""
    df, sp_mgcv, theta_mgcv, fit_mgcv = _scat_bam_load("factor")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float),
           "g": df["g"].to_numpy()}
    m = hea.models.bam("y ~ g + s(x, by=g, k=10)", dat, family=Scat(min_df=5),
                discrete=True)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-7, f"factor auto-fit fitted rel diff {rel:.3e}"

    theta_h = np.asarray(m.family.get_theta(trans=True))
    assert np.allclose(theta_h, theta_mgcv, rtol=1e-6, atol=0), (
        f"factor auto-fit θ mismatch: hea={theta_h} mgcv={theta_mgcv}"
    )

    sp_h = np.asarray(m.sp)
    assert np.allclose(sp_h, sp_mgcv, rtol=1e-5, atol=0), (
        f"factor auto-fit sp mismatch: hea={sp_h} mgcv={sp_mgcv}"
    )


# ---------------------------------------------------------------------------
# General-family seam (mgcv gamlss.r authoring kit) + gaulss — §5.3
# prerequisite 5. mgcv 1.9-4 oracle references: gaulss()$ll evaluated in R
# at identical (y, X, lpi, coef, d1b, d2b, fh, D) for every deriv level.
# ---------------------------------------------------------------------------

def test_trind_generator_matches_mgcv():
    from hea.family import trind_generator
    tri = trind_generator(3)
    # R's 1-based packed indices, flattened column-major (mgcv K=3).
    np.testing.assert_array_equal(
        (tri["i2"] + 1).flatten(order="F"), [1, 2, 3, 2, 4, 5, 3, 5, 6])
    assert tri["i3"][0, 1, 2] + 1 == 5
    assert tri["i3"][2, 2, 2] + 1 == 10
    assert tri["i3"][1, 0, 1] + 1 == 4
    assert tri["i4"][0, 1, 2, 2] + 1 == 9
    assert tri["i4"][2, 2, 2, 2] + 1 == 15
    assert tri["i4"][1, 0, 2, 0] + 1 == 5
    # symmetry: any permutation hits the same packed column
    assert tri["i4"][0, 2, 1, 2] == tri["i4"][2, 2, 1, 0]


def _gaulss_oracle_inputs():
    from hea.R.rng import RGenerator
    gen = RGenerator(17)        # column-major reshape == R matrix(rnorm())
    n = 40
    c2 = gen.normal(0, 1, 2 * n).reshape((n, 2), order="F")
    c1 = gen.normal(0, 1, n).reshape((n, 1), order="F")
    X = np.hstack([np.ones((n, 1)), c2, np.ones((n, 1)), c1])
    y = 1.0 + X[:, 1] * 0.5 + gen.normal(0, 0.7, n)
    coef = np.array([0.8, 0.4, -0.2, 0.3, 0.1])
    d1b = gen.normal(0, 1, 5 * 2).reshape((5, 2), order="F") * 0.3
    d2b = gen.normal(0, 1, 5 * 3).reshape((5, 3), order="F") * 0.2
    lpi = [np.arange(0, 3), np.arange(3, 5)]
    return X, y, coef, d1b, d2b, lpi


def test_gaulss_ll_matches_mgcv_oracle():
    # Every output of gaulss()$ll at deriv 1/2/3/4, pinned to all printed
    # digits: l, lb, lbb, the tr(Hp⁻¹∂H/∂ρ) vector (fh = Hp⁻¹), the full
    # ∂H/∂ρ list, and trHid2H through the preconditioned-Cholesky fh/D
    # convention gam.fit5 uses.
    from scipy.linalg import cholesky
    from hea.family import gaulss
    X, y, coef, d1b, d2b, lpi = _gaulss_oracle_inputs()
    fam = gaulss()

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -57.9372871859, rtol=0,
                               atol=1e-10)
    np.testing.assert_allclose(
        r1["lb"],
        [4.0145452690, -1.1669739590, 6.2728855900, -22.5854933110,
         -7.3560025880], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r1["lbb"][0, 0], -21.7376940450,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               269.7531350500, rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][0, 4], 0.6777580010,
                               rtol=0, atol=1e-9)

    Hp = -r1["lbb"] + np.eye(5) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(r2["d1H"], [-3.0610301460, -0.2929096240],
                               rtol=0, atol=1e-9)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               406.9875674700, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               116.4523491000, rtol=0, atol=1e-8)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], -13.7169040930,
                               rtol=0, atol=1e-9)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(5)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"], [-12.8681383050, 0.3818310880, 2.5707114050],
        rtol=0, atol=1e-9)
    # The eigendecomposition fh variant must agree with the Cholesky one.
    w, V = np.linalg.eigh(D[:, None] * Hp * D[None, :])
    r4e = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                 fh={"values": w, "vectors": V}, D=D)
    np.testing.assert_allclose(r4e["trHid2H"], r4["trHid2H"], atol=1e-9)


def test_gaulss_ll_derivatives_match_fd():
    from hea.family import gaulss
    X, y, coef, d1b, _, lpi = _gaulss_oracle_inputs()
    fam = gaulss()
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-6)
    # d1H along the d1b directions: H(β + h·d1b_j) FD.
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-5)


def test_gaulss_initialize_and_residuals():
    from hea.family import gaulss
    X, y, coef, _, _, lpi = _gaulss_oracle_inputs()
    fam = gaulss()
    start = fam.initialize_coef(y, X, lpi)
    assert start.shape == (5,) and np.all(np.isfinite(start))
    # identity μ-link: LP1 start is the plain LS fit of y on X₁.
    b1, *_ = np.linalg.lstsq(X[:, :3], y, rcond=None)
    np.testing.assert_allclose(start[:3], b1, atol=1e-10)
    mu = X[:, :3] @ coef[:3]
    tau = fam.links[1].linkinv(X[:, 3:] @ coef[3:])
    fitted = np.column_stack([mu, tau])
    np.testing.assert_allclose(fam.residuals(y, fitted, type="response"),
                               y - mu, atol=0)
    np.testing.assert_allclose(fam.residuals(y, fitted),
                               (y - mu) * tau, atol=0)
    # logb link: μ = 1/(e^η + b) round-trips and stays below 1/b.
    lk = fam.links[1]
    eta = np.linspace(-3, 3, 9)
    np.testing.assert_allclose(lk.link(lk.linkinv(eta)), eta, atol=1e-10)
    assert np.all(lk.linkinv(eta) < 1.0 / fam.b)


# ---------------------------------------------------------------------------
# gammals (Gamma location-scale, gamlss.r:2664-2980) — mgcv 1.9-4 oracle.
# Inputs reproduce R's set.seed(21) stream via hea.R.rng (bit-exact rnorm),
# so gammals()$ll evaluated in R at the identical (X, y, coef, d1b, d2b)
# matches hea below; the bounded "log" scale link is SoftplusLink.
# ---------------------------------------------------------------------------

def _gammals_oracle_inputs():
    from hea.R.rng import RGenerator
    gen = RGenerator(21)            # == R set.seed(21)
    n = 40
    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    e = gen.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, np.ones(n), x2])
    y = np.exp(0.6 + 0.4 * x1 + 0.25 * e)          # positive gamma response
    coef = np.array([0.5, 0.3, -0.8, 0.2])
    d1b = gen.normal(0, 1, 4 * 2).reshape((4, 2), order="F") * 0.3
    d2b = gen.normal(0, 1, 4 * 3).reshape((4, 3), order="F") * 0.2
    lpi = [np.arange(0, 2), np.arange(2, 4)]
    return X, y, coef, d1b, d2b, lpi


def test_gammals_ll_matches_mgcv_oracle():
    # Every gammals()$ll output at deriv 1/2/3/4, pinned to the live
    # R values (Rscript: gammals()$ll on the set.seed(21) inputs above).
    from scipy.linalg import cholesky
    from hea.family import gammals
    X, y, coef, d1b, d2b, lpi = _gammals_oracle_inputs()
    fam = gammals()
    p = 4

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -50.1287641294, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        r1["lb"], [10.4292062387, 11.368308003, -18.2209272134,
                   -1.5743414661], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][0, 0], -99.9024955545,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               311.139546477, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1["lbb"][0, 2], -10.4058942597,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][2, 2], -4.74349105844,
                               rtol=0, atol=1e-8)

    Hp = -r1["lbb"] + np.eye(p) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(
        r2["d1H"], [-0.0843074527955, -0.640211644011], rtol=0, atol=1e-9)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               136.804494605, rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               193.697974752, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], -12.4965794289,
                               rtol=0, atol=1e-8)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"],
        [-2.07694884277, 3.65876948038, -9.30036542487], rtol=0, atol=1e-8)
    # eigen fh variant agrees with the Cholesky one.
    w, V = np.linalg.eigh(D[:, None] * Hp * D[None, :])
    r4e = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                 fh={"values": w, "vectors": V}, D=D)
    np.testing.assert_allclose(r4e["trHid2H"], r4["trHid2H"], atol=1e-9)


def test_gammals_ll_derivatives_match_fd():
    from hea.family import gammals
    X, y, coef, d1b, _, lpi = _gammals_oracle_inputs()
    fam = gammals()
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-5)
    # d1H along d1b directions: H(β + h·d1b_j) FD.
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-4)


def test_gammals_initialize_residuals_and_link():
    from hea.family import gammals, SoftplusLink
    X, y, coef, _, _, lpi = _gammals_oracle_inputs()
    fam = gammals()
    start = fam.initialize_coef(y, X, lpi)
    assert start.shape == (4,) and np.all(np.isfinite(start))
    # residuals on the (mean, log σ) fitted matrix (cols as postproc
    # leaves them): deviance/pearson/response forms (gamlss.r:2721-2735).
    mu = np.array([2.0, 3.0, 1.5])
    rho = np.array([-0.5, 0.2, -1.0])
    yy = np.array([2.2, 2.5, 1.7])
    fitted = np.column_stack([mu, rho])
    np.testing.assert_allclose(fam.residuals(yy, fitted, type="response"),
                               yy - mu, atol=0)
    np.testing.assert_allclose(
        fam.residuals(yy, fitted, type="pearson"),
        (yy - mu) / (np.exp(rho * 0.5) * mu), atol=0)
    dexp = (np.sqrt(np.maximum(0.0, 2.0 * ((yy - mu) / mu
            - np.log(yy / mu)) * np.exp(-rho))) * np.sign(yy - mu))
    np.testing.assert_allclose(fam.residuals(yy, fitted), dexp, atol=0)
    # SoftplusLink round-trips and floors at b.
    lk = SoftplusLink(b=-7.0)
    eta = np.linspace(-4, 6, 11)
    np.testing.assert_allclose(lk.link(lk.linkinv(eta)), eta, atol=1e-9)
    assert np.all(lk.linkinv(np.array([-50.0, 0.0, 50.0])) >= -7.0)
    # identity is the only allowed mean link; scale link must be valid.
    with pytest.raises(ValueError, match="mean parameter of gammals"):
        gammals(link=("log", "log"))
    with pytest.raises(ValueError, match="scale"):
        gammals(link=("identity", "sqrt"))


# ---------------------------------------------------------------------------
# gumbls (Gumbel location-scale, gamlss.r:2985-3329) — mgcv 1.9-4 oracle,
# set.seed(23) stream (bit-exact rnorm); shares SoftplusLink with gammals.
# ---------------------------------------------------------------------------

def _gumbls_oracle_inputs():
    from hea.R.rng import RGenerator
    gen = RGenerator(23)
    n = 40
    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    e = gen.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, np.ones(n), x2])
    y = 1.0 + 0.5 * x1 + 0.8 * e            # Gumbel support is all of R
    coef = np.array([0.5, 0.3, -0.4, 0.2])
    d1b = gen.normal(0, 1, 4 * 2).reshape((4, 2), order="F") * 0.3
    d2b = gen.normal(0, 1, 4 * 3).reshape((4, 3), order="F") * 0.2
    lpi = [np.arange(0, 2), np.arange(2, 4)]
    return X, y, coef, d1b, d2b, lpi


def test_gumbls_ll_matches_mgcv_oracle():
    from scipy.linalg import cholesky
    from hea.family import gumbls
    X, y, coef, d1b, d2b, lpi = _gumbls_oracle_inputs()
    fam = gumbls()
    p = 4

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -57.8884083473, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        r1["lb"], [5.76161239291, 2.53486659051, 24.5358131205,
                   -24.1421493465], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][0, 0], -97.2333719429,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               963.95750617, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"][0, 2], 38.9766385199,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][2, 2], -133.911330844,
                               rtol=0, atol=1e-7)

    Hp = -r1["lbb"] + np.eye(p) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(
        r2["d1H"], [3.07537910463, 5.94416984003], rtol=0, atol=1e-8)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               1488.93342144, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               3114.42700999, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], 81.6327066253,
                               rtol=0, atol=1e-7)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"],
        [-9.32912949141, -9.22534206361, -22.1211490539], rtol=0,
        atol=1e-7)


def test_gumbls_ll_derivatives_match_fd():
    from hea.family import gumbls
    X, y, coef, d1b, _, lpi = _gumbls_oracle_inputs()
    fam = gumbls()
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-5)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-4)
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-3)


def test_gumbls_residuals_and_validation():
    from hea.family import gumbls
    fam = gumbls()
    euler = 0.5772156649015328606065121
    # fitted matrix is (mean, log β); location = mean − β·γ.
    mean = np.array([1.0, 2.0, 0.5])
    lb = np.array([-0.3, 0.1, -0.6])
    yy = np.array([1.2, 1.7, 0.8])
    fitted = np.column_stack([mean, lb])
    beta = np.exp(lb)
    mu = mean - beta * euler
    np.testing.assert_allclose(fam.residuals(yy, fitted, type="response"),
                               yy - mean, atol=0)
    np.testing.assert_allclose(
        fam.residuals(yy, fitted, type="pearson"),
        (yy - mean) / (np.pi * beta / np.sqrt(6.0)), atol=0)
    z = (yy - mu) / beta
    dexp = np.sqrt(np.maximum(0.0, 2.0 * (z + np.exp(-z) - 1.0))) \
        * np.sign(yy - mu)
    np.testing.assert_allclose(fam.residuals(yy, fitted), dexp, atol=0)
    with pytest.raises(ValueError, match="location parameter of gumbls"):
        gumbls(link=("log", "log"))


# ---------------------------------------------------------------------------
# gevlss (GEV location-scale-shape, gamlss.r:1945-2446) — mgcv 1.9-4 oracle,
# set.seed(31) stream. 3-LP; auto-generated Maxima l1..l4 (3/6/10/15 cols)
# transcribed verbatim; ShiftedLogitLink confines ξ to (−1, 0.5).
# ---------------------------------------------------------------------------

def _gevlss_oracle_inputs():
    from hea.R.rng import RGenerator
    gen = RGenerator(31)
    n = 40
    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    x3 = gen.normal(0, 1, n)
    e = gen.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, np.ones(n), x2, np.ones(n), x3])
    y = 0.5 + 0.7 * e                       # inside the (wide) GEV support
    coef = np.array([0.3, 0.2, -0.1, 0.15, 1.0, 0.1])
    d1b = gen.normal(0, 1, 6 * 2).reshape((6, 2), order="F") * 0.2
    d2b = gen.normal(0, 1, 6 * 3).reshape((6, 3), order="F") * 0.15
    lpi = [np.arange(0, 2), np.arange(2, 4), np.arange(4, 6)]
    return X, y, coef, d1b, d2b, lpi


def test_gevlss_ll_matches_mgcv_oracle():
    # gevlss()$ll at deriv 1/2/3/4 vs live R (set.seed(31) inputs;
    # Hp = −lbb + 5·I keeps the off-optimum penalized Hessian PD).
    from scipy.linalg import cholesky
    from hea.family import gevlss
    X, y, coef, d1b, d2b, lpi = _gevlss_oracle_inputs()
    fam = gevlss()
    p = 6

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -43.80220977, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        r1["lb"], [2.871313816, -1.72187301, -25.60284638, -0.92031962,
                   -1.577838505, -0.7972557566], rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1["lbb"][0, 0], -53.22895994, rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               279.1671661, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"][0, 4], 5.895272599, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1["lbb"][4, 4], 1.330649383, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1["lbb"][2, 2], -27.58408247, rtol=0, atol=1e-7)

    Hp = -r1["lbb"] + np.eye(p) * 5.0
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b, fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(
        r2["d1H"], [2.288605619, -3.991901518], rtol=0, atol=1e-8)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], 43.54938912, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r3["d1H"][1][0, 0], -59.38174154, rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               301.8381504, rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               457.9403905, rtol=0, atol=1e-6)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"],
        [-2.058358974, 6.146709207, -10.58300064], rtol=0, atol=1e-7)


def test_gevlss_ll_derivatives_match_fd():
    from hea.family import gevlss
    X, y, coef, d1b, _, lpi = _gevlss_oracle_inputs()
    fam = gevlss()
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-5)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-4)
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-3)


def test_gevlss_link_residuals_and_validation():
    from hea.family import gevlss, ShiftedLogitLink
    # shifted-logit confines ξ to (−1, 0.5) and round-trips.
    lk = ShiftedLogitLink()
    eta = np.linspace(-6, 6, 13)
    xi = lk.linkinv(eta)
    assert np.all(xi > -1.0) and np.all(xi < 0.5)
    np.testing.assert_allclose(lk.link(xi), eta, atol=1e-7)
    # mu.eta finite-diff
    fd = (lk.linkinv(eta + 1e-6) - lk.linkinv(eta - 1e-6)) / 2e-6
    np.testing.assert_allclose(lk.mu_eta(eta), fd, atol=1e-7)
    # response residual = y − GEV mean (μ + e^ρ(Γ(1−ξ)−1)/ξ).
    from scipy.special import gamma as _g
    fam = gevlss()
    mu = np.array([0.5, 1.0]); rho = np.array([-0.2, 0.1]); xi3 = np.array([0.1, 0.2])
    yy = np.array([0.7, 1.3])
    fitted = np.column_stack([mu, rho, xi3])
    fv = mu + np.exp(rho) * (_g(1.0 - xi3) - 1.0) / xi3
    np.testing.assert_allclose(fam.residuals(yy, fitted, type="response"),
                               yy - fv, atol=1e-12)
    with pytest.raises(ValueError, match="shape parameter of gevlss"):
        gevlss(link=("identity", "identity", "log"))
    with pytest.raises(ValueError, match="log-scale parameter of gevlss"):
        gevlss(link=("identity", "log", "logit"))


# ---------------------------------------------------------------------------
# cox_ph (Cox proportional hazards, coxph.r + src/coxph.c) — mgcv 1.9-4
# oracle references generated live (R --vanilla): cox.ph()$ll / $hazard at
# fixed inputs (set.seed(5), n=12 with tied integer times so Peto's tie
# correction is exercised; the synthetic d1b/d2b/Hp from set.seed(99)).
# ---------------------------------------------------------------------------

def _cox_oracle_inputs():
    # 12 obs in descending-time order (cox.ph's internal sort handles any
    # order; pinned here as R produced them). Two covariates, integer times
    # WITH TIES, censoring indicator d (1 = event).
    time = np.array([11, 10, 10, 9, 7, 7, 5, 5, 4, 4, 3, 2], float)
    d = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0], int)
    X = np.array([
        -0.60290798145, 1.71144087270, -0.47216638517, -0.84085548079,
        1.38435934348, 0.07014276643, -1.25549186263, -0.63537131252,
        -0.28577363487, 0.13810822480, 1.22763034385, -0.80177945465,
        -2.18396676009, -0.59731309471, 0.24081725594, -1.08039260003,
        -0.15753435611, -0.13898614055, -1.07176003988, -0.25935540673,
        0.90051194533, 0.94186939387, 1.46796190342, 0.70676108956,
    ]).reshape(2, 12).T
    beta = np.array([0.3, -0.2])
    d1b = np.array([0.02139625021848798, 0.04796581345708748,
                    0.00878287049737438, 0.04438585074947837]
                   ).reshape(2, 2).T
    d2b = np.array([-0.0181418960259587, 0.0061337014762436,
                    -0.0431922594072422, 0.0244812133346184,
                    -0.0182058456273331, -0.0647121003340961]
                   ).reshape(3, 2).T
    return time, d, X, beta, d1b, d2b


def test_cox_ph_ll_matches_mgcv_oracle():
    # cox.ph()$ll at every engine deriv level (= C deriv 0..3) vs live R.
    from scipy.linalg import cholesky
    from hea.family import cox_ph
    time, d, X, beta, d1b, d2b = _cox_oracle_inputs()
    fam = cox_ph()
    lpi = [np.arange(2)]
    p = 2

    r1 = fam.ll(time, X, beta, d, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -5.58629239323487, rtol=0, atol=1e-9)
    np.testing.assert_allclose(r1["lb"], [-1.001195162311, 0.829726360424],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        r1["lbb"], [[-3.607277244680, -0.991185222861],
                    [-0.991185222861, -1.796449359815]], rtol=0, atol=1e-9)

    Hp = -r1["lbb"] + 0.5 * np.eye(p)
    # deriv 2 → d1H as the trace vector tr(Hp⁻¹ ∂H/∂ρ) (fh = Hp⁻¹)
    r2 = fam.ll(time, X, beta, d, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(r2["d1H"], [0.0149517979284, 0.0110099642008],
                               rtol=0, atol=1e-9)
    # deriv 3 → d1H as the per-ρ matrix list
    r3 = fam.ll(time, X, beta, d, lpi=lpi, deriv=3, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(
        r3["d1H"][0], [[-0.0241339294506, 0.0590998865919],
                       [0.0590998865919, 0.0727777783225]], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        r3["d1H"][1], [[-0.0090719771863, 0.0543165991505],
                       [0.0543165991505, 0.0539383891138]], rtol=0, atol=1e-9)
    # deriv 4 → trHid2H; cox rebuilds eigen(Hp) from the preconditioned
    # Cholesky pieces (D = ones, L = chol(Hp)) the engine passes
    L = cholesky(Hp, lower=False)
    r4 = fam.ll(time, X, beta, d, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(L, np.arange(p)), D=np.ones(p))
    np.testing.assert_allclose(
        r4["trHid2H"],
        [-0.000159634648342, -0.004026984816385, -0.015872013597495],
        rtol=0, atol=1e-9)


def test_cox_ph_ll_derivatives_match_fd():
    # internal self-consistency: lb vs FD of l, lbb vs FD of lb, and the
    # ∂H/∂ρ matrices vs FD of lbb along the d1b directions.
    from hea.family import cox_ph
    time, d, X, beta, d1b, _ = _cox_oracle_inputs()
    fam = cox_ph()
    lpi = [np.arange(2)]
    p = 2
    r1 = fam.ll(time, X, beta, d, lpi=lpi, deriv=1)
    h = 1e-6
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = beta.copy(); cp[k] += h
        cm = beta.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(time, X, cp, d, lpi=lpi, deriv=0)["l"]
                    - fam.ll(time, X, cm, d, lpi=lpi, deriv=0)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(time, X, cp, d, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(time, X, cm, d, lpi=lpi,
                                 deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], 0.5 * (fd_lbb + fd_lbb.T),
                               rtol=0, atol=1e-6)
    r3 = fam.ll(time, X, beta, d, lpi=lpi, deriv=3, d1b=d1b,
                fh=np.linalg.inv(-r1["lbb"] + 0.5 * np.eye(p)))
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(time, X, beta + h * d1b[:, j], d, lpi=lpi,
                      deriv=1)["lbb"]
               - fam.ll(time, X, beta - h * d1b[:, j], d, lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-4)


def test_cox_ph_hazard_matches_mgcv_oracle():
    # cox.ph()$hazard (the coxpp kernel): baseline cumulative hazard h,
    # its variance q, Kaplan-Meier hazard km, and the `a` vectors — and
    # the internal sort handles arbitrary row order.
    from hea.family import _coxpp
    time, d, X, beta, _, _ = _cox_oracle_inputs()
    rng = np.random.default_rng(2)
    perm = rng.permutation(time.size)
    hz = _coxpp((X @ beta)[perm], X[perm], d[perm], time[perm])
    np.testing.assert_allclose(
        hz["h"], [0.440813001016, 0.440813001016, 0.440813001016,
                  0.239455306944, 0.107523327814, 0, 0, 0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        hz["q"], [0.0695122341035, 0.0695122341035, 0.0695122341035,
                  0.0289673131413, 0.0115612660241, 0, 0, 0],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        hz["km"], [0.541666666667, 0.541666666667, 0.541666666667,
                   0.291666666667, 0.125, 0, 0, 0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        hz["a"][:, 0], [0.1324572533078, 0.1324572533078, 0.1324572533078,
                        0.0820811959820, 0.0215023786239, 0, 0, 0],
        rtol=0, atol=1e-9)


def test_cox_ph_validation():
    from hea.family import cox_ph
    with pytest.raises(ValueError, match="link not available"):
        cox_ph(link="log")
    fam = cox_ph()
    assert fam.n_lp == 1 and fam.drop_intercept is True
    assert fam.available_derivs == 2 and fam.is_general


# ---------------------------------------------------------------------------
# ziplss (zero-inflated Poisson location-scale, gamlss.r:1455-1939) — mgcv
# 1.9-4 oracle. Inputs reproduce R's set.seed(27) stream via hea.R.rng
# (bit-exact rnorm/runif/rpois), so ziplss()$ll evaluated in R at the
# identical (X, y, coef, d1b, d2b) matches hea below.
# ---------------------------------------------------------------------------

def _ziplss_oracle_inputs():
    from hea.R.rng import RGenerator
    gen = RGenerator(27)            # == R set.seed(27)
    n = 40
    x1 = gen.normal(0, 1, n)
    x2 = gen.normal(0, 1, n)
    d1b = gen.normal(0, 1, 4 * 2).reshape((4, 2), order="F") * 0.3
    d2b = gen.normal(0, 1, 4 * 3).reshape((4, 3), order="F") * 0.2
    lam = np.exp(0.5 + 0.6 * x1)                     # Poisson mean
    pu = gen.uniform(0, 1, n)
    y = (gen.poisson(lam) * (pu < 0.7)).astype(float)   # zero-inflated counts
    X = np.column_stack([np.ones(n), x1, np.ones(n), x2])
    coef = np.array([0.4, 0.3, 0.2, -0.5])
    lpi = [np.arange(0, 2), np.arange(2, 4)]
    return X, y, coef, d1b, d2b, lpi


def test_ziplss_ll_matches_mgcv_oracle():
    # Every ziplss()$ll output at deriv 1/2/3/4, pinned to the live R
    # values (Rscript: ziplss()$ll on the set.seed(27) inputs above).
    from scipy.linalg import cholesky
    from hea.family import ziplss
    X, y, coef, d1b, d2b, lpi = _ziplss_oracle_inputs()
    fam = ziplss()
    p = 4

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -67.9945015131, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        r1["lb"], [10.3739285482, 7.0452932554, -14.9712013386,
                   17.7409397677], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][0, 0], -25.7869884817,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               163.3090407273, rtol=0, atol=1e-7)
    # gamma and eta never share a term (the log-lik is separable in the two
    # LPs), so the whole cross-block of the Hessian is exactly zero.
    np.testing.assert_allclose(r1["lbb"][0, 2], 0.0, rtol=0, atol=0)
    np.testing.assert_array_equal(r1["lbb"][lpi[0]][:, lpi[1]], 0.0)
    np.testing.assert_allclose(r1["lbb"][2, 2], -32.8245236943,
                               rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][3, 3], -31.0262755407,
                               rtol=0, atol=1e-8)

    Hp = -r1["lbb"] + np.eye(p) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(
        r2["d1H"], [-1.6738041395, 0.2376725511], rtol=0, atol=1e-9)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               104.8260123501, rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               35.1235191591, rtol=0, atol=1e-7)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], -25.5797222722,
                               rtol=0, atol=1e-8)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"],
        [-0.6446428047, -0.5142583615, -0.7976950165], rtol=0, atol=1e-8)
    # eigen fh variant agrees with the Cholesky one.
    w, V = np.linalg.eigh(D[:, None] * Hp * D[None, :])
    r4e = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                 fh={"values": w, "vectors": V}, D=D)
    np.testing.assert_allclose(r4e["trHid2H"], r4["trHid2H"], atol=1e-9)


def test_ziplss_ll_derivatives_match_fd():
    from hea.family import ziplss
    X, y, coef, d1b, _, lpi = _ziplss_oracle_inputs()
    fam = ziplss()
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-5)
    # d1H along d1b directions: H(β + h·d1b_j) FD.
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-4)


def test_ziplss_helpers_initialize_and_validation():
    from hea.family import (ziplss, _l1ee, _lee1, _ldg, _lde, _zipll,
                            _ziplss_ls)
    # robustified scalar helpers, incl. the tail branches the moderate-eta
    # oracle never reaches (R: mgcv:::l1ee/lee1/ldg/lde).
    np.testing.assert_allclose(
        _l1ee(np.array([-30., -1, 0.5, 3, 8])),
        [-30, -1.17830709642, -0.213559185373, -1.89217874881e-09, 0],
        rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        _lee1(np.array([-30., -1, 0.5, 3, 8])),
        [-30, -0.810427655249, 1.43516208533, 20.0855369213,
         2980.95798704], rtol=1e-11, atol=0)
    lg = _ldg(np.array([-30., -1, 0.5, 2]), 4)
    np.testing.assert_allclose(
        lg[0], [-1, -1.19519230416, -2.04124350898, -7.39362520396],
        rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        lg[3], [-4.67881148442e-14, -0.272552750274, -2.10286784068,
                -6.80541955362], rtol=1e-9, atol=1e-13)
    le = _lde(np.array([-30., -1, 0.5, 2]), 4)
    np.testing.assert_allclose(
        le[0], [1, 0.82731286299, 0.39252223828, 0.00456910503104],
        rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        le[3], [-4.67881148442e-14, -0.0953266908979, 0.45414656998,
                -0.583636545308], rtol=1e-9, atol=1e-13)

    fam = ziplss()
    X, y, coef, _, _, lpi = _ziplss_oracle_inputs()
    start = fam.initialize_coef(y, X, lpi)
    assert start.shape == (4,) and np.all(np.isfinite(start))

    # residuals: response = y − E(y), E(y) = p·λ/(1−e^{−λ}). fitted is
    # (gamma, presence-eta) — ziplss leaves it unrewritten.
    gamma = np.array([0.5, -3.0, 1.0])      # log Poisson mean
    eta = np.array([0.3, 1.2, -0.5])        # presence LP
    yy = np.array([2.0, 0.0, 3.0])
    fitted = np.column_stack([gamma, eta])
    lam = np.exp(gamma)
    p_pres = 1.0 - np.exp(-np.exp(eta))
    ey = np.where(lam > np.sqrt(np.finfo(float).eps),
                  p_pres * lam / (1.0 - np.exp(-lam)), p_pres)
    np.testing.assert_allclose(fam.residuals(yy, fitted, type="response"),
                               yy - ey, rtol=0, atol=1e-12)
    dev = (np.sqrt(np.maximum(0.0, 2.0 * (_ziplss_ls(yy)
           - _zipll(yy, gamma, eta)["l"]))) * np.sign(yy - ey))
    np.testing.assert_allclose(fam.residuals(yy, fitted), dev,
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="deviance"):
        fam.residuals(yy, fitted, type="pearson")

    # rd: structural absence → exact zeros; present rows are zero-truncated
    # Poisson (never 0), so P(y=0) == 1−p; draws are integer-valued.
    rng = np.random.default_rng(0)
    big = 200000
    mu = np.column_stack([np.full(big, np.log(3.0)), np.full(big, 0.4)])
    draws = fam.rd(rng, mu, np.ones(big), 1.0)
    assert np.allclose(draws, np.round(draws)) and draws.min() == 0.0
    p04 = 1.0 - np.exp(-np.exp(0.4))
    np.testing.assert_allclose((draws == 0).mean(), 1.0 - p04,
                               rtol=0, atol=5e-3)
    ey3 = p04 * 3.0 / (1.0 - np.exp(-3.0))
    np.testing.assert_allclose(draws.mean(), ey3, rtol=0, atol=2e-2)

    # count intake: non-integer and binary responses are rejected.
    with pytest.raises(ValueError, match="non-integer"):
        fam.initialize_coef(y + 0.5, X, lpi)
    with pytest.raises(ValueError, match="binary"):
        fam.initialize_coef((y > 0).astype(float), X, lpi)
    with pytest.raises(ValueError, match="identity"):
        ziplss(link=("log", "identity"))


# ---------------------------------------------------------------------------
# multinom (multinomial logistic, gamlss.r:1107-1411) — the variable-K
# family. Inputs reproduce R's set.seed stream via hea.R.rng (bit-exact
# rnorm/runif), so multinom(K)$ll at the identical (X, y, coef, d1b, d2b)
# matches hea. K=2 and K=4 together cover every l3/l4 packing branch (l3
# all-different needs K≥3, l4 all-unique needs K≥4).
# ---------------------------------------------------------------------------

def _multinom_oracle_inputs(K, seed):
    from hea.R.rng import RGenerator
    gen = RGenerator(seed)
    n = 30 + 10 * K
    xs = [gen.normal(0, 1, n) for _ in range(K)]
    p = 2 * K
    d1b = gen.normal(0, 1, p * 2).reshape((p, 2), order="F") * 0.3
    d2b = gen.normal(0, 1, p * 3).reshape((p, 3), order="F") * 0.2
    a = np.array([0.4, -0.2, 0.3, -0.1, 0.2, -0.3])
    b = np.array([0.7, 0.5, -0.6, 0.4, -0.5, 0.45])
    etas = [a[i] + b[i] * xs[i] for i in range(K)]
    ee = np.exp(np.column_stack([np.zeros(n)] + etas))
    P = ee / ee.sum(axis=1, keepdims=True)
    u = gen.uniform(0, 1, n)
    y = (np.cumsum(P, axis=1) > u[:, None]).argmax(axis=1).astype(float)
    cols = []
    for i in range(K):
        cols += [np.ones(n), xs[i]]
    X = np.column_stack(cols)
    coef = np.array([0.3, 0.5, -0.2, 0.4, 0.1, -0.3, 0.25, -0.15,
                     0.2, -0.1, 0.35, -0.05])[:p]
    lpi = [np.arange(2 * i, 2 * i + 2) for i in range(K)]
    return X, y, coef, d1b, d2b, lpi


# live multinom(K)$ll references (Rscript, mgcv 1.9-4) at the inputs above.
_MULTINOM_ORACLE = {
    (2, 41): dict(
        l=-38.0695935472, lb_sumabs=9.2002046302, lbb_sumabs=75.4481988693,
        lbb00=-11.5076064746, d1H_sumabs=0.8726944587,
        d1H0mat_sumabs=15.8743078433,
        trHid2H=[0.0018660042, -0.3379211462, 0.0702141514],
        lb=[-1.7610060440, 1.3832675257, 3.4277307608, 2.6282002998],
        lbb02=5.4228156488, lbb22=-9.4456506089),
    (4, 43): dict(
        l=-100.8787924997, lb_sumabs=43.6795492702,
        lbb_sumabs=167.9531090628, lbb00=-12.3369275107,
        d1H_sumabs=0.5315310582, d1H0mat_sumabs=50.9607930529,
        trHid2H=[0.4880585105, -0.0145410370, 0.8359873276]),
}


@pytest.mark.parametrize("K,seed", [(2, 41), (4, 43)])
def test_multinom_ll_matches_mgcv_oracle(K, seed):
    from scipy.linalg import cholesky
    from hea.family import multinom
    X, y, coef, d1b, d2b, lpi = _multinom_oracle_inputs(K, seed)
    fam = multinom(K)
    p = 2 * K
    ref = _MULTINOM_ORACLE[(K, seed)]

    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], ref["l"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lb"]))),
                               ref["lb_sumabs"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               ref["lbb_sumabs"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(r1["lbb"][0, 0], ref["lbb00"],
                               rtol=0, atol=1e-8)
    if "lb" in ref:
        np.testing.assert_allclose(r1["lb"], ref["lb"], rtol=0, atol=1e-8)
        # the two LPs are coupled — the cross-block is nonzero (unlike
        # ziplss, whose log-lik is separable in its two LPs).
        np.testing.assert_allclose(r1["lbb"][0, 2], ref["lbb02"],
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(r1["lbb"][2, 2], ref["lbb22"],
                                   rtol=0, atol=1e-8)

    Hp = -r1["lbb"] + np.eye(p) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(float(np.sum(np.abs(r2["d1H"]))),
                               ref["d1H_sumabs"], rtol=0, atol=1e-8)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               ref["d1H0mat_sumabs"], rtol=0, atol=1e-7)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    np.testing.assert_allclose(r4["trHid2H"], ref["trHid2H"],
                               rtol=0, atol=1e-8)


@pytest.mark.parametrize("K,seed", [(2, 41), (4, 43)])
def test_multinom_ll_derivatives_match_fd(K, seed):
    # FD self-checks at K=2 and K=4 — the K=4 d1H validates the l4
    # all-unique branch against the (R-pinned) lower-order derivatives.
    from hea.family import multinom
    X, y, coef, d1b, _, lpi = _multinom_oracle_inputs(K, seed)
    fam = multinom(K)
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-5)
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-4)


def test_multinom_components_and_validation():
    from hea.family import multinom
    X, y, coef, _, _, lpi = _multinom_oracle_inputs(2, 41)
    fam = multinom(2)
    assert fam.n_lp == 2 and fam.is_general and fam.available_derivs == 2
    start = fam.initialize_coef(y, X, lpi)
    assert start.shape == (4,) and np.all(np.isfinite(start))

    # residuals: sign +ve when the most-probable category equals y, mag
    # √(−2 log P̂(y)); fitted is the (n, 2) η matrix.
    eta = np.array([[2.0, -1.0], [-0.5, 1.5], [0.2, 0.3]])
    yy = np.array([1.0, 0.0, 2.0])
    p = fam.predict(eta=eta)["fit"]
    assert p.shape == (3, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    pc = np.argmax(p, axis=1)
    exp = np.where(pc == yy.astype(int), 1.0, -1.0) * np.sqrt(
        -2.0 * np.log(p[np.arange(3), yy.astype(int)]))
    np.testing.assert_allclose(fam.residuals(yy, eta), exp, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="deviance"):
        fam.residuals(yy, eta, type="response")

    # rd: categories land in 0..K with frequencies tracking the softmax
    # probabilities at a fixed η.
    rng = np.random.default_rng(0)
    big = 200000
    mu = np.tile(np.array([0.5, -0.3]), (big, 1))
    draws = fam.rd(rng, mu, np.ones(big), 1.0)
    assert set(np.unique(draws).astype(int)).issubset({0, 1, 2})
    probs = np.exp(np.array([0.0, 0.5, -0.3]))
    probs = probs / probs.sum()
    for c in range(3):
        np.testing.assert_allclose((draws == c).mean(), probs[c],
                                   rtol=0, atol=5e-3)

    # variable-K validation: K<1 rejected; response outside 0..K rejected.
    with pytest.raises(ValueError, match="at least 2"):
        multinom(0)
    with pytest.raises(ValueError, match="0\\.\\.2"):
        fam.initialize_coef(y + 5, X, lpi)
    assert multinom(4).n_lp == 4


# ---------------------------------------------------------------------------
# twlss (Tweedie location-scale-shape, gamlss.r:2493-2662) — mgcv 1.9-4
# oracle references generated live (R --vanilla): ldTweedie in the working
# (rho, theta) parameterization with all.derivs=TRUE, tw.null.fit, and
# twlss()$ll at fixed inputs.
# ---------------------------------------------------------------------------

def test_ld_tweedie_work_matches_mgcv():
    # R: ldTweedie(y, mu, p=NA, phi=NA, rho=rho, theta=theta, a=, b=,
    # all.derivs=TRUE) — columns [l, ρ, ρρ, θ, θθ, θρ, μ, μμ, μθ, μρ].
    # First block: vector θ/ρ (mgcv's C_tweedious2 path), a=1.01/b=1.99,
    # y including zeros; second block: constant θ/ρ (the buffered
    # C_tweedious path), a=1.001/b=1.999. hea runs one vectorized series
    # (same Dunn-Smyth eps gate) for both.
    from hea.family import _ld_tweedie_work

    y = np.array([0.0, 0.0, 7.291920, 0.510023, 3.460132, 6.182369,
                  1.789924, 3.323070, 1.768233, 6.144270, 6.137256,
                  2.765712])
    mu = np.array([0.964286, 0.962858, 1.448242, 2.617561, 1.745190,
                   2.476464, 2.596160, 1.642260, 2.498690, 1.454858,
                   2.399253, 1.591939])
    theta = np.array([0.970123, -0.433116, -1.001834, 0.759094,
                      0.956343, 1.119591, 0.175365, 0.528191, 0.657741,
                      0.306626, 0.535174, -0.271605])
    rho = np.array([-0.539535, -0.500435, -0.174001, -0.361758,
                    -0.492932, 0.007027, 0.422145, 0.309870, 0.070487,
                    0.254540, -0.050035, -0.029111])
    ref = np.array([
        [-6.07774368217828798, 6.07774368217828798, -6.07774368217828798,
         -4.29264953384012138, -4.07010384939407821, 4.29264953384012138,
         -1.76075519799844527, 1.31586751382190803, -0.01250737445020930,
         1.76075519799844527],
        [-2.66694553685743596, 2.66694553685743596, -2.66694553685743596,
         -1.05538028855574351, -1.04185298247669467, 1.05538028855574351,
         -1.67431641885951432, 0.68776186762230362, -0.01482012360987546,
         1.67431641885951432],
        [-7.56812018644871287, 4.93397293143535265, -5.47250478445514688,
         0.77800471109246061, 0.15147403859011277, -0.98603991725351747,
         4.33979792225844019, -4.55794399448242427, -0.30942109699124137,
         -4.33979792225844019],
        [-1.60808230290744536, 0.74989206256702534, -1.45857918029695632,
         0.20922394151316182, -0.09604868289867108, -0.08155635723105270,
         -0.60234993317785701, 0.10022464910862995, 0.12336268116127148,
         0.60234993317785701],
        [-2.38749735096340387, 0.06443397190057354, -0.65417482960264195,
         -0.03692999319651746, 0.00287371993390728, -0.10033583517840627,
         1.07858781636451839, -1.69068258557892737, -0.11804240939558083,
         -1.07858781636451861],
        [-3.35616740164874017, 0.21440167727280013, -0.84978971263152481,
         -0.01222126991580463, -0.03123472986535525, -0.18447851506165436,
         0.75352113586700065, -0.73545300736029551, -0.12423910530784306,
         -0.75352113586700087],
        [-1.76919047681530506, -0.60122779270616222, -0.28216547480977727,
         -0.07096374827319610, 0.03231977006462894, -0.00311461756014708,
         -0.12130361090784719, -0.07836798765475869, 0.02813641261244896,
         0.12130361090784721],
        [-2.44720418176075372, -0.29536468667040516, -0.42962875743214646,
         -0.10697045172956915, 0.02804274659285744, -0.06735703151094063,
         0.55021427533224998, -0.87227805011580684, -0.06241644545934319,
         -0.55021427533225031],
        [-1.58252896535058873, -0.53506678082128367, -0.18484744971538714,
         -0.05552151948434525, 0.02588742447302428, -0.00800067267036964,
         -0.14946475540104281, -0.10558604462229911, 0.03015374572297890,
         0.14946475540104270],
        [-4.49098892819283346, 1.38935757398455184, -2.03323450529134231,
         0.14861507806937357, -0.11862696527510774, -0.42197856370548714,
         2.01467720382923687, -2.61003492989971297, -0.18077081522394509,
         -2.01467720382923687],
        [-3.42429659036231371, 0.45462315068891002, -1.06302033227015968,
         0.04487081508984825, -0.09006261112099878, -0.29073216112123368,
         0.94531632201480986, -0.89436228699683151, -0.18884126199460971,
         -0.94531632201480964],
        [-1.96307613706432882, -0.28507735563739622, -0.34423808958002056,
         -0.09559903538486081, -0.02000319210936929, -0.06317624785322451,
         0.62042896728882224, -1.08739831509373963, -0.06938763593958744,
         -0.62042896728882235],
    ])
    ld = _ld_tweedie_work(y, mu, theta, rho, a=1.01, b=1.99)
    np.testing.assert_allclose(ld, ref, rtol=0, atol=1e-9)

    ref_const = np.array([
        [-1.21116109521257287, 1.21116109521257287, -1.21116109521257287,
         -0.49580329144675028, -0.49523047682818921, 0.49580329144675034,
         -0.75171489454688600, 0.31299928986520725, -0.00655508229496946,
         0.75171489454688600],
        [-1.21008732699060784, 1.21008732699060784, -1.21008732699060784,
         -0.49579373725034942, -0.49522850604279584, 0.49579373725034942,
         -0.75216232191462673, 0.31365007058357408, -0.00682626553159174,
         0.75216232191462673],
        [-5.51755802584939481, 2.47326612289146830, -3.06353887974802319,
         0.39958761454983538, -0.10040866722653874, -0.68443126125093379,
         2.57618777493490425, -2.93390901353341471, -0.22877243954650908,
         -2.57618777493490425],
        [-1.68334800887195923, -0.21714268635201872, -1.57067657979921371,
         0.32851200765334265, -0.15711748994350128, 0.45673492656099213,
         -0.40532242126467699, 0.02469974682476480, 0.09351875928772453,
         0.40532242126467694],
        [-2.35618698177088426, -0.21915517766960413, -0.43786533874262101,
         -0.10693058685162926, -0.03922065645825956, -0.10020604966777169,
         0.58212489995790528, -0.80692994562309539, -0.07772813334755921,
         -0.58212489995790517],
        [-3.27424505592075121, 0.33616366186271662, -0.93765313183847887,
         0.01645934094090329, -0.08290451707926660, -0.28363030815688539,
         0.77027630175638839, -0.64377491480577276, -0.16748912329783522,
         -0.77027630175638839],
        [-1.65352006685852659, -0.54805135067817101, -0.27639480629174695,
         -0.08167760548363368, 0.00641612141837111, -0.06352562253752003,
         -0.15684991340137053, -0.10987212765279498, 0.03588072736244631,
         0.15684991340137047],
        [-2.34169221937344840, -0.20919033753980365, -0.45326151230521727,
         -0.10464319340366135, -0.03708180712771458, -0.09871066944946616,
         0.62127852436154707, -0.89983142681279726, -0.07390019650786997,
         -0.62127852436154729],
        [-1.63518007422290701, -0.56023496595131128, -0.26971786596123160,
         -0.08279554240922748, 0.00805790639565629, -0.06064202998341850,
         -0.14993690714205718, -0.12116524143336850, 0.03292355380853813,
         0.14993690714205718],
        [-4.54984393460295955, 1.61571801658653946, -2.21765725348626042,
         0.20512335572403106, -0.07446136943338999, -0.47163298632771689,
         2.05416527824867989, -2.41688420610359556, -0.18466041314788254,
         -2.05416527824868034],
        [-3.31230619349391020, 0.37891096619140541, -0.98093355615571554,
         0.02460243733357270, -0.08214519851698032, -0.29098987648577079,
         0.81221504400718636, -0.69173652323214097, -0.17043962893315948,
         -0.81221504400718625],
        [-2.05907281339363823, -0.38255857036408258, -0.30973534524907542,
         -0.11582005320502997, -0.02921512840035989, -0.07143971417858497,
         0.45320420212415546, -0.78510037679750022, -0.05052614098104411,
         -0.45320420212415541],
    ])
    ld2 = _ld_tweedie_work(y, mu, np.full(12, -0.4), np.full(12, 0.3),
                           a=1.001, b=1.999)
    np.testing.assert_allclose(ld2, ref_const, rtol=0, atol=1e-9)

    with pytest.raises(ValueError, match="1<a<b<2"):
        _ld_tweedie_work(y, mu, theta, rho, a=1.0, b=1.99)


def _twlss_oracle_y():
    # R: set.seed(42); mu <- exp(0.3 + 0.8*runif(60));
    #    yt <- rTweedie(mu, p=1.6, phi=0.9)
    return np.array([
        4.04832308895089, 4.17936043923019, 3.18587833770696,
        1.44074695069965, 4.96682704573932, 0.0149315304839502,
        1.31627343712032, 1.19929694965169, 4.98639984327987,
        1.35357831643216, 0.198097759483503, 0.313528574866513,
        0.777866064977357, 3.28558602605701, 1.26797018732695,
        0.82937768794381, 0, 2.92638123008549, 2.0162054963584, 0,
        4.34056314942428, 1.85939302159064, 0.225282722969358,
        1.49126091505468, 2.28072824296011, 0.859291105463246,
        1.14598294283332, 1.73778438895884, 0.775647292698183,
        0.245179071165799, 1.11855663081434, 0, 1.39221568742179,
        3.13850404746383, 2.79303322538822, 4.29183163087591,
        0.472418069551852, 3.69910714314767, 0.971713018042964,
        2.18919413113105, 0.689386047811524, 1.24381420521243,
        0.48420624007319, 2.63209085648186, 1.55645192039097,
        3.99772753051548, 0.760473374388309, 2.24332902793916,
        3.75950204398062, 0, 2.00239166059865, 3.12175565523763,
        2.70646186227391, 2.11588414331005, 1.45415683039719,
        0.606670224754281, 0, 1.5079915994605, 1.70426119447661,
        2.30116003594764])


def test_tw_null_fit_matches_mgcv():
    # mgcv:::tw.null.fit(yt) — stabilized Newton on (log mu, theta,
    # rho); deterministic, so the stop point reproduces to ~1e-12.
    from hea.family import _tw_null_fit
    mu0, p0, phi0 = _tw_null_fit(_twlss_oracle_y())
    np.testing.assert_allclose(
        [mu0, p0, phi0],
        [1.803702629136, 1.420825162004, 0.926231178339],
        rtol=0, atol=1e-9)


def test_twlss_ll_oracle_matches_mgcv():
    # twlss()$ll evaluated in R at identical (y, X, lpi, coef, wt) with
    # deriv=1 — l/lb/lbb to all printed digits. The family has no
    # l3/l4 (available.derivs=0), so deriv=1 is the whole surface.
    from hea.family import twlss

    yt = _twlss_oracle_y()
    n = 60
    i = np.arange(1, n + 1, dtype=float)
    X = np.hstack([
        np.column_stack([np.ones(n), np.sin(i / 7), np.cos(i / 5)]),
        np.column_stack([np.ones(n), (i % 5) / 10]),
        np.column_stack([np.ones(n), i / n]),
    ])
    lpi = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 7)]
    coef = np.array([0.4, 0.3, -0.2, 0.1, -0.3, -0.5, 0.6])
    fam = twlss()
    ll = fam.ll(yt, X, coef, np.ones(n), lpi=lpi, deriv=1)
    np.testing.assert_allclose(ll["l"], -113.4552049409, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(
        ll["lb"],
        [14.0826693067, -13.6588037265, 22.6003907425, -3.9719270265,
         -0.6427082058, 15.6769401965, 2.7477361989],
        rtol=0, atol=1e-8)
    lbb_ref = np.array([
        [-101.3354356225977, -21.3611818930165, -0.6473717887347,
         0.7296901371795, -0.1370969366965, -14.0826693066860,
         -3.5523998943407],
        [-21.3611818930165, -50.8040979992333, 22.7233919247163,
         1.3163140036167, -0.0430450825389, 13.6588037265190,
         8.3363037834457],
        [-0.6473717887347, 22.7233919247163, -49.6565854115001,
         -1.9895117246515, -0.0821326181627, -22.6003907424747,
         -4.9701863439456],
        [0.7296901371795, 1.3163140036167, -1.9895117246515,
         -9.6031255217363, -1.1036910632160, 5.0993398162256,
         2.6919079819855],
        [-0.1370969366965, -0.0430450825389, -0.0821326181627,
         -1.1036910632160, -0.2004962088083, 0.3896344813640,
         0.2466017773393],
        [-14.0826693066860, 13.6588037265190, -22.6003907424747,
         5.0993398162256, 0.3896344813640, -56.0219829589650,
         -23.0331127321579],
        [-3.5523998943407, 8.3363037834457, -4.9701863439456,
         2.6919079819855, 0.2466017773393, -23.0331127321579,
         -14.0254838354779],
    ])
    np.testing.assert_allclose(ll["lbb"], lbb_ref, rtol=0, atol=1e-8)

    # FD self-check of lb (the family's whole derivative surface)
    h = 1e-6
    fd = np.empty(7)
    for k in range(7):
        cp = coef.copy()
        cm = coef.copy()
        cp[k] += h
        cm[k] -= h
        fd[k] = (fam.ll(yt, X, cp, np.ones(n), lpi=lpi, deriv=0)["l"]
                 - fam.ll(yt, X, cm, np.ones(n), lpi=lpi,
                          deriv=0)["l"]) / (2 * h)
    np.testing.assert_allclose(ll["lb"], fd, rtol=5e-6, atol=1e-6)

    # constructor surface: okLinks + the (a, b) bounds
    with pytest.raises(ValueError, match="not available for the mu"):
        twlss(link=("inverse", "identity", "identity"))
    with pytest.raises(ValueError, match='only the "identity"'):
        twlss(link=("log", "log", "identity"))
    with pytest.raises(ValueError, match="1<a<b<2"):
        twlss(a=1.5, b=1.2)
    assert repr(twlss()) == \
        "twlss(link=('log', 'identity', 'identity'), a=1.01, b=1.99)"


# ---------------------------------------------------------------------------
# shash (sinh-arcsinh location-scale-shape, gamlss.r:3334-4080) — mgcv
# 1.9-4 oracle references generated live (R --vanilla): shash()$ll at
# every deriv level (the K=4 etamu/gH branches through all orders), the
# logeb link, and the residuals/rd/qf/cdf hooks.
# ---------------------------------------------------------------------------

def _shash_oracle_inputs():
    n = 40
    i = np.arange(1, n + 1, dtype=float)
    y = 0.7 + np.sin(i / 4) + 0.4 * np.cos(i * 1.7)
    X = np.column_stack([np.ones(n), np.sin(i / 7), np.cos(i / 5),
                         np.ones(n), i / n, np.ones(n), (i % 5) / 10,
                         np.ones(n), np.cos(i / 9)])
    lpi = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 7),
           np.arange(7, 9)]
    coef = np.array([0.5, 0.4, -0.3, -0.8, 0.2, 0.15, -0.25, 0.1, 0.05])
    # R: matrix(sin(1:18)/5, 9, 2) / matrix(cos(1:27)/5, 9, 3) —
    # column-major fill
    d1b = (np.sin(np.arange(1, 19)) / 5).reshape(2, 9).T
    d2b = (np.cos(np.arange(1, 28)) / 5).reshape(3, 9).T
    return y, X, lpi, coef, d1b, d2b


def test_shash_ll_matches_mgcv_oracle():
    # Every output of shash()$ll at deriv 1/2/3/4 — l, lb, lbb, the
    # tr(Hp⁻¹∂H/∂ρ) vector, the full ∂H/∂ρ list, trHid2H — pinned to
    # all printed digits. This is the K=4 stress pin: the packed L1
    # (4), L2 (10), L3 (20) and L4 (35 auto-generated columns) all
    # flow through gamlss_etamu/gamlss_gH's highest-order branches.
    from hea.family import shash

    y, X, lpi, coef, d1b, d2b = _shash_oracle_inputs()
    fam = shash()
    r1 = fam.ll(y, X, coef, None, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], -100.9497210553, rtol=0,
                               atol=1e-8)
    np.testing.assert_allclose(
        r1["lb"],
        [37.6260962202, -62.0123901747, 113.0759290090, 154.0254235235,
         85.7420345932, 54.8783816066, 17.2037035469, -112.6281058568,
         60.2205948365], rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.abs(r1["lbb"]).sum()),
                               10088.3601306536, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        [r1["lbb"][0, 0], r1["lbb"][0, 8], r1["lbb"][3, 8]],
        [-193.4295889712, -40.3834376029, -202.9264742803],
        rtol=0, atol=1e-8)

    Hp = -r1["lbb"] + np.eye(9) * 0.5
    r2 = fam.ll(y, X, coef, None, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(r2["d1H"],
                               [-6.7252978570, 7.5418328552],
                               rtol=0, atol=1e-8)

    r3 = fam.ll(y, X, coef, None, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(
        [float(np.abs(r3["d1H"][0]).sum()), r3["d1H"][0][0, 0]],
        [11283.4240316976, -202.6392545658], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        [float(np.abs(r3["d1H"][1]).sum()), r3["d1H"][1][0, 0]],
        [9755.5096903696, 167.1180992130], rtol=0, atol=1e-7)

    D = 1.0 / np.sqrt(np.diag(Hp))
    w, V = np.linalg.eigh(D[:, None] * Hp * D[None, :])
    r4 = fam.ll(y, X, coef, None, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh={"values": w, "vectors": V}, D=D)
    np.testing.assert_allclose(
        r4["trHid2H"], [-4.8268188590, 5.1230900806, -5.4504659452],
        rtol=0, atol=1e-8)

    # FD self-checks: lb against FD of l, lbb against FD of lb
    h = 1e-6
    fd_lb = np.empty(9)
    fd_lbb = np.empty((9, 9))
    for k in range(9):
        cp = coef.copy()
        cm = coef.copy()
        cp[k] += h
        cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, None, lpi=lpi, deriv=0)["l"]
                    - fam.ll(y, X, cm, None, lpi=lpi,
                             deriv=0)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, None, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, None, lpi=lpi,
                                 deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1["lbb"], 0.5 * (fd_lbb + fd_lbb.T),
                               rtol=1e-4, atol=1e-3)

    # constructor surface
    with pytest.raises(ValueError, match='only the "identity"'):
        shash(link=("log", "logeb", "identity", "identity"))
    with pytest.raises(ValueError, match='only the "logeb"'):
        shash(link=("identity", "log", "identity", "identity"))
    assert repr(shash()) == ("shash(link=('identity', 'logeb', "
                             "'identity', 'identity'), b=0.01, "
                             "phiPen=0.001)")
    # logeb link round-trip: τ = log(e^η + b) keeps σ = e^τ > b
    lk = shash().links[1]
    eta = np.linspace(-3.0, 3.0, 9)
    np.testing.assert_allclose(lk.link(lk.linkinv(eta)), eta,
                               rtol=0, atol=1e-12)
    assert np.all(np.exp(lk.linkinv(eta)) > shash().b)


def test_shash_hooks_match_mgcv():
    # residuals (deviance vs the ls=0 reference; raw mean via Bessel
    # K), rd-shape, qf and cdf (incl. log.p) — R values at the fitted
    # parameter matrix implied by the oracle coefficients.
    from hea.family import shash

    y, X, lpi, coef, _, _ = _shash_oracle_inputs()
    fam = shash()
    F = np.column_stack([
        X[:, 0:3] @ coef[0:3],
        np.log(np.exp(X[:, 3:5] @ coef[3:5]) + 0.01),
        X[:, 5:7] @ coef[5:7],
        X[:, 7:9] @ coef[7:9]])
    np.testing.assert_allclose(
        fam.residuals(y, F, "deviance")[:3],
        [-1.3300547156, -1.0212378918, 2.4582366281], rtol=0,
        atol=1e-8)
    np.testing.assert_allclose(
        fam.residuals(y, F, "response")[:3],
        [-0.0399933645, -0.2195953901, 0.4355807611], rtol=0,
        atol=1e-8)
    with pytest.raises(ValueError, match="deviance"):
        fam.residuals(y, F, "pearson")
    n = y.shape[0]
    p = (np.arange(1, n + 1) - 0.5) / n
    np.testing.assert_allclose(
        fam.qf(p, F, None, None)[[0, 19, 39]],
        [-0.5698615266, 0.8685826873, 1.7077315470], rtol=0,
        atol=1e-8)
    np.testing.assert_allclose(
        fam.cdf(y, F, None, None, False)[[0, 19, 39]],
        [0.8922773135, 0.0002822583, 0.4429213918], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fam.cdf(y, F, None, None, True)[[0, 19, 39]],
        [-0.1139783051, -8.1726878430, -0.8143629698], rtol=0,
        atol=1e-8)
    # rd: the quantile transform of uniforms — cdf(draws) must be
    # uniform at Monte-Carlo level
    rng = np.random.default_rng(4)
    Fc = np.broadcast_to([0.5, np.log(np.exp(0.2) + 0.01), 0.3, 0.0],
                         (200000, 4))
    draws = fam.rd(rng, Fc, None, None)
    u = fam.cdf(draws, Fc, None, None, False)
    np.testing.assert_allclose(u.mean(), 0.5, rtol=0, atol=5e-3)
    np.testing.assert_allclose(u.var(), 1.0 / 12.0, rtol=0, atol=5e-3)


# ---------------------------------------------------------------------------
# mvn (multivariate normal, mvam.r) — the matrix-response general family and
# the only available_derivs=1 family (fit by the bfgs outer optimizer). The
# ll is a numpy port of mgcv's C kernel mvn_ll (src/mvn.c). Inputs reproduce
# R's set.seed stream via hea.R.rng; the dH derivative blocks are exercised
# at m=2 AND m=3 (m=3 lights the off-diagonal × off-diagonal theta cross
# terms a 2-D covariance never reaches).
# ---------------------------------------------------------------------------

def _mvn_oracle_inputs(m, n, pv, seed, beta):
    from hea.R.rng import RGenerator
    g = RGenerator(seed)               # X/Y stream == R set.seed(seed)
    Xlps = []
    for p in pv:
        inner = g.normal(size=n * (p - 1)).reshape(n, p - 1, order="F")
        Xlps.append(np.column_stack([np.ones(n), inner]))
    Xm = np.concatenate(Xlps, axis=1)
    Y = g.normal(size=n * m).reshape(n, m, order="F")
    ncoef = sum(pv)
    ntheta = m * (m + 1) // 2
    nb = ncoef + ntheta
    X = np.concatenate([Xm, np.zeros((n, ntheta))], axis=1)
    cs = np.cumsum([0] + list(pv))
    lpi = [np.arange(cs[k], cs[k + 1]) for k in range(m)]
    from hea.R.rng import RGenerator as _RG
    g2 = _RG(seed + 1000)              # d1b stream == R set.seed(seed+1000)
    d1b = g2.normal(size=nb * 2).reshape(nb, 2, order="F")
    return X, Y, np.asarray(beta, dtype=float), lpi, d1b


# live mvn(d)$ll references (Rscript, mgcv 1.9-4) at the inputs above.
_MVN_ORACLE = {
    (2, 101): dict(
        beta=[0.4, -0.3, 0.2, 0.5, 0.15, 0.1, -0.2, -0.05],
        n=12, pv=(3, 2),
        l=-15.7256094975,
        lb=[-4.9632116233, 2.4997313471, 1.3787612378, -6.7585550687,
            -0.4513484407, -6.2541223309, -0.2294738282, -2.4429914298],
        lbb_sum=-191.5095845216, lbb_fro=67.3135835142,
        lbb_diag=[-14.6568330979, -5.5978210354, -15.4202009864,
                  -11.3380490164, -10.7476175503, -37.1926183914,
                  -15.9619740982, -28.8859828595],
        d1Htr=[-26.7424444768, -3.3268032864],
        d1H0_fro=139.9866331557, d1H1_fro=118.8887931462,
        d1H0_sum=238.6527021845, d1H1_sum=187.9967632078),
    (3, 103): dict(
        beta=[0.3, -0.2, 0.4, 0.1, -0.15, 0.25, 0.2,
              0.05, -0.1, 0.03, 0.0, 0.08, -0.04],
        n=14, pv=(2, 3, 2),
        l=-21.8485684831,
        lb=[-2.7005117456, -5.4481415192, -1.7222714051, 1.0765211381,
            -1.4765460177, -6.8279173238, -3.5559462890, 2.6549350058,
            5.1388436379, -4.3797131544, -0.5040107190, -1.5259774175,
            -3.3607073011],
        lbb_sum=-314.4691803182, lbb_fro=77.6586394144,
        lbb_diag=[-15.4723928531, -18.2212267690, -14.1400000000,
                  -8.6564737967, -11.1090117136, -13.0258288494,
                  -5.9106796852, -22.2066744641, -14.5022949557,
                  -18.8066297050, -29.0063056747, -18.8066297050,
                  -34.7214146023],
        d1Htr=[2.4098222737, -4.4527135016],
        d1H0_fro=174.1301225815, d1H1_fro=105.8376694333,
        d1H0_sum=151.3467988839, d1H1_sum=-153.5943517839),
}


@pytest.mark.parametrize("m,seed", [(2, 101), (3, 103)])
def test_mvn_ll_matches_mgcv_oracle(m, seed):
    from hea.family import mvn, _mvn_ll
    ref = _MVN_ORACLE[(m, seed)]
    X, Y, beta, lpi, d1b = _mvn_oracle_inputs(
        m, ref["n"], ref["pv"], seed, ref["beta"])
    nb = beta.size

    # the family ll dispatches to _mvn_ll with mgcv's deriv codes intact.
    fam = mvn(d=m)
    r1 = fam.ll(Y, X, beta, lpi=lpi, deriv=1)
    np.testing.assert_allclose(r1["l"], ref["l"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lb"], ref["lb"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(r1["lbb"])), ref["lbb_sum"],
                               rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sqrt(np.sum(r1["lbb"] ** 2))),
                               ref["lbb_fro"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.diag(r1["lbb"]), ref["lbb_diag"],
                               rtol=0, atol=1e-8)
    # the precision factor couples the dimensions: the off-diagonal
    # mean×mean blocks are nonzero (full d×d precision, not block-diagonal).
    assert abs(r1["lbb"][0, lpi[1][0]]) > 1e-6

    # deriv 2 — the per-rho traces tr(fh·dH/drho), fh = Hp^{-1}.
    fh = np.linalg.inv(-r1["lbb"] + np.eye(nb))
    r2 = fam.ll(Y, X, beta, lpi=lpi, deriv=2, d1b=d1b, fh=fh)
    np.testing.assert_allclose(r2["d1H"], ref["d1Htr"], rtol=0, atol=1e-7)

    # deriv 3 — the dH/drho matrices themselves.
    r3 = fam.ll(Y, X, beta, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sqrt(np.sum(r3["d1H"][0] ** 2))),
                               ref["d1H0_fro"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sqrt(np.sum(r3["d1H"][1] ** 2))),
                               ref["d1H1_fro"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(float(np.sum(r3["d1H"][0])), ref["d1H0_sum"],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(float(np.sum(r3["d1H"][1])), ref["d1H1_sum"],
                               rtol=0, atol=1e-6)
    # the trace at deriv 2 must equal sum(fh * dH) from the deriv-3 matrices.
    np.testing.assert_allclose(
        [float(np.sum(fh * dH)) for dH in r3["d1H"]], r2["d1H"],
        rtol=0, atol=1e-10)
    # deriv 4 is genuinely unavailable (available_derivs=1 ⇒ bfgs path).
    with pytest.raises(NotImplementedError, match="deriv 3"):
        fam.ll(Y, X, beta, lpi=lpi, deriv=4, d1b=d1b)


@pytest.mark.parametrize("m,seed", [(2, 101), (3, 103)])
def test_mvn_ll_derivatives_match_fd(m, seed):
    from hea.family import mvn
    ref = _MVN_ORACLE[(m, seed)]
    X, Y, beta, lpi, d1b = _mvn_oracle_inputs(
        m, ref["n"], ref["pv"], seed, ref["beta"])
    fam = mvn(d=m)
    r1 = fam.ll(Y, X, beta, lpi=lpi, deriv=1)
    h = 1e-6
    nb = beta.size
    fd_lb = np.zeros(nb)
    fd_lbb = np.zeros((nb, nb))
    for k in range(nb):
        cp = beta.copy(); cp[k] += h
        cm = beta.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(Y, X, cp, lpi=lpi)["l"]
                    - fam.ll(Y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(Y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(Y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=1e-5, atol=1e-5)


def test_mvn_components_and_validation():
    from hea.family import mvn, _mvn_ll
    fam = mvn(d=2)
    assert fam.n_lp == 2 and fam.n_extra_coef == 3
    assert fam.available_derivs == 1 and fam.matrix_response is True

    # R-factor rebuild from theta: diag = exp(theta), off-diag = theta.
    coef = np.array([1.0, 2.0, 3.0, 0.05, -0.3, -0.52])   # 3 mean + 3 theta
    R = fam._R_from_coef(coef)
    np.testing.assert_allclose(np.diag(R), [np.exp(0.05), np.exp(-0.52)],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose([R[0, 1], R[1, 0]], [-0.3, 0.0],
                               rtol=0, atol=1e-12)

    # postproc deviance ≡ Σ‖R(y−μ̂)‖²; residuals deviance ≡ (y−μ̂)·Rᵀ.
    y = np.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.5]])
    fitted = np.array([[0.9, 1.8], [0.6, -0.8], [1.7, 0.7]])
    fam.set_fit_context(coef=coef)
    pp = fam.postproc(y, np.ones(3), fitted, None, [None, None], True)
    rsd = (y - fitted) @ R.T
    np.testing.assert_allclose(pp["deviance"], float(np.sum(rsd ** 2)),
                               rtol=0, atol=1e-12)
    rsd0 = (y - y.mean(axis=0)) @ R.T
    np.testing.assert_allclose(pp["null_deviance"],
                               float(np.sum(rsd0 ** 2)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(fam.residuals(y, fitted, type="deviance"),
                               rsd, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fam.residuals(y, fitted, type="response"),
                               y - fitted, rtol=0, atol=1e-12)

    # initialize_coef returns the full (mean + theta) vector; the diagonal
    # theta seeds are −½ log(residual scale), off-diagonals zero.
    X, Y, beta, lpi, _ = _mvn_oracle_inputs(
        2, 12, (3, 2), 101, _MVN_ORACLE[(2, 101)]["beta"])
    E = np.zeros((0, X.shape[1]))
    start = fam.initialize_coef(Y, X, lpi, E=E, offset=[None, None],
                                use_unscaled=True)
    assert start.shape == (8,) and np.all(np.isfinite(start))
    assert start[6] == 0.0          # the single off-diagonal theta

    # validation: d<2 rejected; offsets rejected.
    with pytest.raises(ValueError, match="2 or more"):
        mvn(d=1)
    with pytest.raises(NotImplementedError, match="offset"):
        fam.ll(Y, X, beta, lpi=lpi, deriv=1,
               offset=[np.ones(12), None])


# ---------------------------------------------------------------------------
# betar (Beta regression, efam.r:3269-3546) — the first D1b extended family
# and mgcv's prototype for "-2logLik as deviance" (dev_resids omit the
# saturated reference; ls≡0; the saturated log-lik is folded in by a Newton
# solver only for the reported deviance/residuals). Inputs reproduce R's
# set.seed(202) runif stream via hea.R.rng.
# ---------------------------------------------------------------------------

def _betar_dd_inputs():
    from hea.R.rng import RGenerator
    g = RGenerator(202)
    n = 9
    y = g.uniform(0.05, 0.95, n)
    mu = g.uniform(0.1, 0.9, n)
    return y, mu, np.ones(n)


def test_betar_components_match_mgcv():
    from hea.family import betar
    fam = betar(theta=8, link="logit")
    th = fam.get_theta()                 # log φ
    y, mu, wt = _betar_dd_inputs()
    D = fam.Dd(y, mu, th, wt, level=2)
    # live betar(theta=8)$Dd references (Rscript, mgcv 1.9-4).
    ref = dict(
        Dmu=[34.2887061086, -42.2888136009, -7.2280831059, 1.0291405749,
             22.0623768581, -58.0012103192, 0.0717342770, 25.0824892008,
             12.7764533599],
        Dmu2=[77.7453285973, 167.5795893924, 95.5390226276, 97.4278924302,
              87.8326134591, 239.6591074798, 101.7925754531, 76.3881169600,
              79.0007145291],
        Dth=[7.0207781840, 7.1218338511, -0.9256218382, -1.0076881530,
             1.9454945428, 13.1964579256, -1.0391933060, 3.3601482647,
             -0.0420342682],
        Dmu3=[95.4725129451, -1230.3954193622, -273.1248071677,
              -292.3136124966, 196.9375226494, -2647.7474545772,
              -337.6647022406, 79.6729211303, 109.1267089537],
        Dmu2th2=[68.2749537031, 123.1224623633, 80.5672546417,
                 81.8239313358, 75.3467745148, 157.2427064925,
                 84.6944775667, 67.3017228061, 69.1704603378],
        Dmu3th=[60.5529734155, -470.2984365066, -154.3149409500,
                -163.2230554834, 116.8599251673, -749.4336166052,
                -183.5497409113, 50.9977946726, 68.6314414742],
    )
    for nm, val in ref.items():
        np.testing.assert_allclose(D[nm], val, rtol=0, atol=1e-8)
    # EDmu2 ≡ Dmu2 (observed = expected here); Dmu2th ≡ EDmu2th.
    np.testing.assert_allclose(D["EDmu2"], D["Dmu2"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(D["Dmu2th"], D["EDmu2th"], rtol=0, atol=1e-12)

    # dev_resids (the −2logLik), variance, aic.
    np.testing.assert_allclose(
        fam.dev_resids(y, mu, wt),
        [5.3971578279, 7.1706034046, -1.3766068861, -1.9131058921,
         1.4996938245, 13.1550170991, -1.9336677014, 2.3120105027,
         -0.4907305916], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fam.variance(mu),
        [0.0263382964, 0.0150105684, 0.0224799726, 0.0221506673,
         0.0239649202, 0.0118411627, 0.0214348104, 0.0267040863,
         0.0260108531], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fam.aic(y, mu, 0, wt, 9), 23.8203715876,
                               rtol=0, atol=1e-7)

    # saturated_ll: the per-datum Newton matches mgcv's saturated.ll.
    sl = fam.saturated_ll(y, wt, np.exp(th[0]))
    np.testing.assert_allclose(sl["f"], 8.1716871009, rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        sl["term"],
        [1.0966594358, 0.8344161099, 0.8344537876, 0.9592418787,
         0.8130724065, 0.9401460009, 0.9668464788, 0.9383451518,
         0.7885058510], rtol=0, atol=1e-8)

    # ls ≡ 0 (the saturated reference lives in saturated_ll, not ls).
    le = fam.ls_extended(y, wt)
    assert le["ls"] == 0.0 and float(np.sum(np.abs(le["LSTH1"]))) == 0.0


def test_betar_Dd_matches_fd():
    # FD-check the μ/θ derivatives of dev_resids (the −2logLik).
    from hea.family import betar
    fam = betar(theta=5, link="logit")
    th = fam.get_theta()
    y, mu, wt = _betar_dd_inputs()
    D = fam.Dd(y, mu, th, wt, level=1)
    h = 1e-6
    fd_mu = (fam.dev_resids(y, mu + h, wt)
             - fam.dev_resids(y, mu - h, wt)) / (2 * h)
    fd_mu2 = (fam.dev_resids(y, mu + h, wt) - 2 * fam.dev_resids(y, mu, wt)
              + fam.dev_resids(y, mu - h, wt)) / h ** 2
    fd_th = (fam.dev_resids(y, mu, wt, theta=th + h)
             - fam.dev_resids(y, mu, wt, theta=th - h)) / (2 * h)
    np.testing.assert_allclose(D["Dmu"], fd_mu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(D["Dmu2"], fd_mu2, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(D["Dth"], fd_th, rtol=1e-5, atol=1e-5)


def test_betar_construction_and_validation():
    from hea.family import betar
    # theta sign convention: fixed (>0, n_theta=0) vs free start (<0).
    assert betar(theta=4).n_theta == 0
    assert betar(theta=-4).n_theta == 1
    assert betar().n_theta == 1
    np.testing.assert_allclose(betar(theta=4).get_theta(trans=True)[0], 4.0)
    # okLinks
    for lk in ("logit", "probit", "cloglog", "cauchit"):
        betar(link=lk)
    with pytest.raises(ValueError, match="not available"):
        betar(link="log")
    # preinitialize clamps y into (eps, 1-eps).
    fam = betar()
    pre = fam.preinitialize(np.array([0.0, 0.5, 1.0]))
    assert pre["y"][0] > 0 and pre["y"][2] < 1


# ---------------------------------------------------------------------------
# ocat (ordered categorical, efam.r:2618-3081) — the first extended family
# with VECTOR θ (n_theta = R−2 ordered log-step params). Classes are 0-based
# in hea (mgcv: 1..R); the verbatim Dd/dev/aic helpers work 1-based and the
# class converts at the boundary. Oracle: live ocat(R=4)$Dd / dev.resids /
# aic / preinitialize (Rscript, mgcv 1.9-4) on a fixed (y, μ, θ) table whose
# classes 0..3 light every branch (y==1, mid, y==R) of the θ-chain.
# ---------------------------------------------------------------------------

def _ocat_dd_inputs():
    # Fixed probe table; y0 0-based classes spanning 0..3 (mgcv 1..4).
    y0 = np.array([0, 1, 2, 3, 1, 2, 0, 3])
    mu = np.array([-1.5, -0.5, 0.5, 1.5, -0.3, 0.8, -2.0, 2.0])
    th = np.array([-0.3, 0.4])
    wt = np.array([1.0, 1, 1, 2, 1, 1, 1, 2])
    return y0, mu, th, wt


def test_ocat_components_match_mgcv():
    from hea.family import ocat
    y0, mu, th, wt = _ocat_dd_inputs()
    fam = ocat(R=4)
    fam.set_theta(th)        # so residuals_extended uses the same θ as Dd
    D = fam.Dd(y0, mu, th, wt, level=2)
    # live ocat(R=4)$Dd(y, mu, theta, wt, level=2) references.
    np.testing.assert_allclose(
        D["D"], [0.94815396836021326, 3.4033848405830938, 2.0624996810431218,
                 2.2735085228913441, 3.4475560178681786, 2.1051446787034322,
                 0.62652337503644562, 1.5253409548574155], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu"], [0.75508133759629081, 0.12508810834514919,
                   0.01158217642058367, -1.7342241692960858,
                   0.31596926720447793, 0.27205911041460462,
                   0.53788284273999032, -1.268204398689236], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu2"], [0.47000742440318893, 0.96282774356019796,
                    0.8728306898144822, 0.98234080195341111,
                    0.94321753851302692, 0.85966791304002421,
                    0.3932238664829637, 0.86611879947555437], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu3"], [0.11511358970464147, -0.056058657808741363,
                    -0.0035821305595196231, -0.13054122133675861,
                    -0.13895868706639994, -0.0837958137273902,
                    0.18171549534589687, -0.31691096383438538],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu4"], [-0.19271351257916913, -0.42850879434420946,
                    -0.26994501209635102, -0.46514937482029561,
                    -0.39603714552583974, -0.26240133428201973,
                    -0.070651161032471238, -0.25912386273190896],
        rtol=0, atol=1e-9)
    # vector-θ blocks (n_theta = 2): Dth/Dmuth/Dmu2th (n×2), Dmu3th (n×2).
    np.testing.assert_allclose(D["Dth"], np.array([
        [0, 0], [-2.0018700821058992, 0],
        [-0.0085802873275185494, -1.8345684322093283],
        [1.2847448633611565, 2.5871584470023166],
        [-2.0755253980313082, 0], [-0.20154634609759842, -2.0400833323027197],
        [0, 0], [0.93950892609768777, 1.8919386436218986]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth"], np.array([
        [0, 0], [-0.3650902719536912, 0],
        [-0.64660887858476102, -0.65413562360419775],
        [-0.72773596500617788, -1.4654802698548306],
        [-0.37025486600654539, 0],
        [-0.63685765371547653, -0.71206813055814888],
        [0, 0], [-0.64163658792646583, -1.2920974161490391]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmu2th"], np.array([
        [0, 0], [-0.043748969569600636, 0],
        [0.0026537075873529679, -0.22945111953838168],
        [0.096707315316315809, 0.19474461805043206],
        [-0.0075555234051465454, 0],
        [0.062077465626101752, -0.15167705715753937],
        [0, 0], [0.23477341634231763, 0.47277560280143555]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmu3th"], np.array([
        [0, 0], [0.17468144108232198, 0],
        [0.19998018354312383, 0.206340975445134],
        [0.34459113220558429, 0.69392132544931229],
        [0.18489616284655114, 0], [0.19439168956731451, 0.30757115418809855],
        [0, 0], [0.19196367892522651, 0.38656737817166814]]),
        rtol=0, atol=1e-9)
    # second-θ-deriv blocks (n×3, packed (j,k≥j)): Dth2/Dmuth2/Dmu2th2.
    np.testing.assert_allclose(D["Dth2"], np.array([
        [0, 0, 0], [0.17958315855029516, 0, 0],
        [0.47043935158264499, 0.48459558876298758, 0.8082585471108592],
        [1.8238649260831261, 1.0856544859580193, 4.7733981074777461],
        [0.10975386800166609, 0, 0],
        [0.2702494077554346, 0.52751304548424494, 0.68916879168764833],
        [0, 0, 0],
        [1.4148450014896607, 0.95720930877897625, 3.8195214807915052]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth2"], np.array([
        [0, 0, 0], [-0.3326802381604822, 0, 0],
        [-0.64857479351783343, 0.1699815701098521, -0.31183477657540037],
        [-0.79937850626571694, -0.14427036143146205, -1.7560051007951816],
        [-0.36465759660122665, 0, 0],
        [-0.68284577134503588, 0.1123651276016875, -0.48579255062498522],
        [0, 0, 0],
        [-0.81556101248454971, -0.35024078084908605, -1.99739573685046]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmu2th2"], np.array([
        [0, 0, 0], [-0.173156163938325, 0, 0],
        [-0.14549525615666759, -0.15286115428299446, -0.53727568284282401],
        [-0.15857207410692392, -0.5140695616124592, -0.84046435347481852],
        [-0.14452996977600635, 0, 0],
        [-0.081931439954469026, -0.22785431517864946, -0.61051930125737597],
        [0, 0, 0],
        [0.092563225285414746, -0.28637615727073196, -0.1039151592574919]]),
        rtol=0, atol=1e-9)
    # EDmu2 ≡ Dmu2, EDmu2th ≡ Dmu2th (observed = expected, mgcv ocat).
    np.testing.assert_allclose(D["EDmu2"], D["Dmu2"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(D["EDmu2th"], D["Dmu2th"], rtol=0, atol=1e-12)

    # dev_resids(theta=th) ≡ Dd$D; the latent-midpoint sign; aic ≡ Σ Dd$D.
    np.testing.assert_allclose(fam.dev_resids(y0, mu, wt, theta=th), D["D"],
                               rtol=0, atol=1e-12)
    res = fam.residuals_extended(y0, mu, wt, "deviance")
    # sign(res) reproduces mgcv's attr(.,"sign"); |res| = √(Dd$D).
    np.testing.assert_array_equal(np.sign(res), [-1, -1, -1, 1, -1, -1, -1, 1])
    np.testing.assert_allclose(np.abs(res), np.sqrt(D["D"]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(fam.aic(y0, mu, 0, wt, 0, theta=th),
                               16.392112039343242, rtol=0, atol=1e-9)

    # ls ≡ 0 (vector θ); lsth1/LSTH1/lsth2 all zero.
    le = fam.ls_extended(y0, wt)
    assert le["ls"] == 0.0
    assert float(np.sum(np.abs(le["lsth1"]))) == 0.0
    assert le["lsth2"].shape == (2, 2) and float(np.sum(np.abs(le["lsth2"]))) == 0.0

    # preinitialize seeds θ from empirical cumulative class proportions.
    yc = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 2, 0, 3, 1, 2, 3, 0])
    pre = ocat(R=4).preinitialize(yc)
    np.testing.assert_allclose(
        pre["Theta"], [-0.070896606721916375, 0.052931367693468845],
        rtol=0, atol=1e-9)


def test_ocat_Dd_matches_fd():
    # FD-check the μ/θ derivatives of the ocat deviance.
    from hea.family import ocat
    fam = ocat(R=4)
    y0, mu, th, wt = _ocat_dd_inputs()
    D = fam.Dd(y0, mu, th, wt, level=1)
    h = 1e-6
    fd_mu = (fam.dev_resids(y0, mu + h, wt, theta=th)
             - fam.dev_resids(y0, mu - h, wt, theta=th)) / (2 * h)
    h2 = 1e-4   # second difference: coarser step keeps cancellation in check
    fd_mu2 = (fam.dev_resids(y0, mu + h2, wt, theta=th)
              - 2 * fam.dev_resids(y0, mu, wt, theta=th)
              + fam.dev_resids(y0, mu - h2, wt, theta=th)) / h2 ** 2
    np.testing.assert_allclose(D["Dmu"], fd_mu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(D["Dmu2"], fd_mu2, rtol=1e-4, atol=1e-4)
    # θ-gradient column k via central diff in θ_k.
    for k in range(2):
        thp = th.copy(); thp[k] += h
        thm = th.copy(); thm[k] -= h
        fd_thk = (fam.dev_resids(y0, mu, wt, theta=thp)
                  - fam.dev_resids(y0, mu, wt, theta=thm)) / (2 * h)
        np.testing.assert_allclose(D["Dth"][:, k], fd_thk, rtol=1e-5, atol=1e-5)


def test_ocat_construction_and_validation():
    from hea.family import ocat
    # R derived from theta length; n_theta = R−2; sign convention.
    assert ocat(R=4).n_theta == 2
    assert ocat(theta=[0.5, 0.5]).n_theta == 0          # fixed (all >0)
    assert ocat(theta=[-0.5, -0.5]).n_theta == 2        # free start (<0)
    assert ocat(theta=[0.2, 0.3, 0.4])._R == 5
    # negative theta = "initial supplied" → ini = log|θ| ([−1,−1] → [0,0]).
    np.testing.assert_allclose(ocat(theta=[-1.0, -1.0]).get_theta(), [0.0, 0.0])
    # get_theta(trans) = finite cut points [−1, −1+cumsum(e^θ)]; the default
    # ocat(R=4) seeds θ = [−1,−1].
    np.testing.assert_allclose(
        ocat(R=4).get_theta(trans=True),
        [-1.0, -0.6321205588, -0.2642411177], rtol=0, atol=1e-9)
    # set_theta requires the right length.
    fam = ocat(R=4)
    fam.set_theta([0.1, -0.2])
    np.testing.assert_allclose(fam.get_theta(), [0.1, -0.2])
    with pytest.raises(ValueError, match="log-step"):
        fam.set_theta([0.1])
    # okLinks: identity only.
    ocat(R=3, link="identity")
    with pytest.raises(ValueError, match="not available"):
        ocat(R=3, link="logit")
    # must supply theta or R.
    with pytest.raises(ValueError, match="theta or R"):
        ocat()
    # initialize rejects out-of-range classes (0..R−1).
    with pytest.raises(ValueError, match="out of range"):
        ocat(R=4).initialize(np.array([0, 1, 4]), np.ones(3))
