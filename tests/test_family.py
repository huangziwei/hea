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
        scale=0.8: ls -159.1465003977
                   lsth1 (-7.3948798555, -74.8238393909)
                   lsth2 [0.1209236332, 1.4618616825;
                          1.4618616825, -14.3754131012]
        scale=1.3: ls -197.6649193951
                   lsth1 (-5.6102775745, -85.0364843846)
                   lsth2 [-1.7094624821, 6.9296834083;
                          6.9296834083, -29.0369350068]
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
        0.8: (-159.1465003977, [-7.3948798555, -74.8238393909],
              [[0.1209236332, 1.4618616825],
               [1.4618616825, -14.3754131012]]),
        1.3: (-197.6649193951, [-5.6102775745, -85.0364843846],
              [[-1.7094624821, 6.9296834083],
               [6.9296834083, -29.0369350068]]),
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
    rng = np.random.default_rng(17)
    n = 40
    X = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 2)),
                   np.ones((n, 1)), rng.normal(size=(n, 1))])
    y = 1.0 + X[:, 1] * 0.5 + rng.normal(0, 0.7, n)
    coef = np.array([0.8, 0.4, -0.2, 0.3, 0.1])
    d1b = rng.normal(size=(5, 2)) * 0.3
    d2b = rng.normal(size=(5, 3)) * 0.2
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
    np.testing.assert_allclose(r1["l"], -54.793169613785, rtol=0,
                               atol=1e-10)
    np.testing.assert_allclose(
        r1["lb"],
        [4.7999681426, 1.3391588398, -1.6129245631, -29.6772211001,
         -5.2891939141], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r1["lbb"][0, 0], -21.2901328670,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(float(np.sum(np.abs(r1["lbb"]))),
                               255.3526070152, rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["lbb"][0, 4], -7.5504145515,
                               rtol=0, atol=1e-9)

    Hp = -r1["lbb"] + np.eye(5) * 0.5
    r2 = fam.ll(y, X, coef, lpi=lpi, deriv=2, d1b=d1b,
                fh=np.linalg.inv(Hp))
    np.testing.assert_allclose(r2["d1H"], [2.4746058080, -0.7768359917],
                               rtol=0, atol=1e-9)

    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][0]))),
                               345.3335722086, rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.sum(np.abs(r3["d1H"][1]))),
                               246.6055510218, rtol=0, atol=1e-8)
    np.testing.assert_allclose(r3["d1H"][0][0, 0], 10.8408987310,
                               rtol=0, atol=1e-9)

    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(5)), D=D)
    np.testing.assert_allclose(
        r4["trHid2H"], [-6.7777512659, 2.3794298623, -3.2244531536],
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
        cp = coef.copy(); cp[k] += h
        cm = coef.copy(); cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-6)
    # d1H along the d1b directions: H(β + h·d1b_l) FD.
    r3 = fam.ll(y, X, coef, lpi=lpi, deriv=3, d1b=d1b)
    for l in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, l], lpi=lpi, deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, l], lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][l], fdH, rtol=0, atol=1e-5)


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
