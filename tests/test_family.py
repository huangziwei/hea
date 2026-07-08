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


def _scat_bam_have_force(name: str) -> bool:
    return (_scat_bam_have(name)
            and (_SCAT_BAM / name / "fitted_force.csv").exists())


def _scat_bam_load(name: str):
    sub = _SCAT_BAM / name
    df = pl.read_csv(str(sub / "data.csv"))
    sp = np.atleast_1d(np.loadtxt(sub / "sp.csv"))
    theta = np.atleast_1d(np.loadtxt(sub / "theta.csv"))
    fitted = np.loadtxt(sub / "fitted.csv")
    return df, sp, theta, fitted


def _scat_bam_force_fitted(name: str):
    """mgcv's OWN force-fit fitted (bam re-run at the converged (θ, sp), θ
    locked) — generated by ``dump_bam_scat_force.R``. This is NOT the auto
    fit's ``fitted.csv``: mgcv's penalised-deviance convergence test
    (bgam.fitd:678 reads ``dev + sum(rSb²)``) stops the force fit in fewer
    iters at a slightly different β (mgcv force-vs-auto ~3.6e-8 on ``factor``).
    The force tests pin hea's force fit to THIS, the faithful target."""
    return np.loadtxt(_SCAT_BAM / name / "fitted_force.csv")


@pytest.mark.skipif(
    not _scat_bam_have_force("simple"),
    reason="scat_bam/simple force oracle missing — run dump_bam_scat_force.R",
)
def test_scat_bam_simple_force_theta_sp():
    df, sp_mgcv, theta_mgcv, _ = _scat_bam_load("simple")
    fit_force = _scat_bam_force_fitted("simple")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0   # both θ supplied positive ⇒ locked
    m = hea.models.bam("y ~ s(x, k=10)", dat, family=fam, discrete=True,
                sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    # Pin to mgcv's OWN force-fit (same penalised-deviance convergence), NOT
    # the auto fit — see :func:`_scat_bam_force_fitted`.
    rel = float(np.linalg.norm(fit_h - fit_force) / np.linalg.norm(fit_force))
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
    not _scat_bam_have_force("factor"),
    reason="scat_bam/factor force oracle missing — run dump_bam_scat_force.R",
)
def test_scat_bam_factor_force_theta_sp():
    df, sp_mgcv, theta_mgcv, _ = _scat_bam_load("factor")
    fit_force = _scat_bam_force_fitted("factor")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float),
           "g": df["g"].to_numpy()}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0
    m = hea.models.bam("y ~ g + s(x, by=g, k=10)", dat, family=fam,
                discrete=True, sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    # Pin to mgcv's OWN force-fit, NOT the auto fit: mgcv's force-vs-auto gap is
    # ~3.6e-8 here (the penalised-deviance convergence stops it early), and hea
    # reproduces mgcv's force fit to ~7e-14. Pinning to the auto fit would
    # require hea to over-converge past where mgcv itself stops.
    rel = float(np.linalg.norm(fit_h - fit_force) / np.linalg.norm(fit_force))
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


_SCAT_BAM_ND = Path(__file__).parent / "fixtures" / "scat_bam_nondiscrete"


@pytest.mark.skipif(
    not (_SCAT_BAM_ND / "simple" / "fitted.csv").exists(),
    reason="scat_bam_nondiscrete oracle missing — "
           "run tests/r_oracle/dump_bam_scat_nondiscrete.R",
)
@pytest.mark.parametrize("sub,formula,has_g", [
    ("simple", "y ~ s(x, k=10)", False),
    ("factor", "y ~ g + s(x, by=g, k=10)", True),
])
def test_scat_bam_nondiscrete_matches_mgcv(sub, formula, has_g):
    """``bam(family=scat, discrete=FALSE)`` routes through mgcv ``bgam.fit``
    (bam.r:909-1353), whose PIRLS cadence differs from the discrete
    ``bgam.fitd``: it estimates the family θ at the END of each iteration
    (bam.r:1204), so each working-model build uses the PREVIOUS iteration's θ.
    hea's shared loop was bgam.fitd-shaped (θ estimated mid-iter, build uses
    this-iter θ), diverging ~3e-6 on the fitted values (simple: hea iter 10 vs
    mgcv 12). This pins the faithful bgam.fit θ-cadence for both a single smooth
    and a 3-level factor-by smooth; both land at the reduced-(R,f) floor."""
    d = _SCAT_BAM_ND / sub
    df = pl.read_csv(str(d / "data.csv"))
    sp_mgcv = np.atleast_1d(np.loadtxt(d / "sp.csv"))
    theta_mgcv = np.atleast_1d(np.loadtxt(d / "theta.csv"))
    edf_mgcv = float(np.loadtxt(d / "edf.csv"))
    fit_mgcv = np.loadtxt(d / "fitted.csv")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    if has_g:
        dat["g"] = df["g"].to_numpy()
    m = hea.models.bam(formula, dat, family=Scat(min_df=5),
                       method="fREML", discrete=False)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-8, f"scat non-discrete {sub} fitted rel diff {rel:.3e}"
    np.testing.assert_allclose(np.asarray(m.family.get_theta(trans=True)),
                               theta_mgcv, rtol=1e-6, atol=0)
    np.testing.assert_allclose(np.sort(np.asarray(m.sp)), np.sort(sp_mgcv),
                               rtol=1e-6, atol=0)
    np.testing.assert_allclose(float(m.edf_total), edf_mgcv, rtol=1e-7, atol=0)


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


def test_trind_generator_ifunc_reverse_match_mgcv():
    # trind.generator(K, ifunc=, reverse=) formals (gamlss.r:20-112).
    # Pins: mgcv 1.9-4 trind.generator(4) — i2r/i3r/i4r (1-based), and
    # ifunc spot values i4(3,1,4,2)=15, i3(2,4,1)=7, i2(4,2)=7.
    from hea.family import trind_generator
    ta = trind_generator(4)                    # reverse defaults TRUE
    tf = trind_generator(4, ifunc=True)        # reverse defaults FALSE
    assert tf["i2r"] is None and tf["i3r"] is None and tf["i4r"] is None
    np.testing.assert_array_equal(
        ta["i2r"] + 1, [1, 2, 3, 4, 6, 7, 8, 11, 12, 16])
    np.testing.assert_array_equal(
        ta["i3r"] + 1,
        [1, 2, 3, 4, 6, 7, 8, 11, 12, 16, 22, 23, 24, 27, 28, 32, 43, 44,
         48, 64])
    np.testing.assert_array_equal(
        ta["i4r"] + 1,
        [1, 2, 3, 4, 6, 7, 8, 11, 12, 16, 22, 23, 24, 27, 28, 32, 43, 44,
         48, 64, 86, 87, 88, 91, 92, 96, 107, 108, 112, 128, 171, 172,
         176, 192, 256])
    # 0-based hea args ≡ mgcv's 1-based spot pins
    assert tf["i4"](2, 0, 3, 1) + 1 == 15
    assert tf["i3"](1, 3, 0) + 1 == 7
    assert tf["i2"](3, 1) + 1 == 7
    # closures ≡ arrays over every index tuple, K = 2..5
    for K in range(2, 6):
        a = trind_generator(K)
        f = trind_generator(K, ifunc=True, reverse=True)
        np.testing.assert_array_equal(f["i4r"], a["i4r"])
        for i in range(K):
            for j in range(K):
                assert f["i2"](i, j) == a["i2"][i, j]
                for k in range(K):
                    assert f["i3"](i, j, k) == a["i3"][i, j, k]
                    for l_ in range(K):
                        assert f["i4"](i, j, k, l_) == a["i4"][i, j, k, l_]
        # reverse extraction: ravel()[ixr] recovers packing order from a
        # symmetric array (same cells R reads column-major — digit sum
        # of the reversed tuple commutes).
        packed = np.arange(a["i4"].max() + 1, dtype=float)
        np.testing.assert_array_equal(
            packed[a["i4"]].ravel()[a["i4r"]], packed)
        packed3 = np.arange(a["i3"].max() + 1, dtype=float)
        np.testing.assert_array_equal(
            packed3[a["i3"]].ravel()[a["i3r"]], packed3)
        packed2 = np.arange(a["i2"].max() + 1, dtype=float)
        np.testing.assert_array_equal(
            packed2[a["i2"]].ravel()[a["i2r"]], packed2)


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
# matches hea below; the bounded "log" scale link is BoundedLogLink.
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
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
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
    from hea.family import gammals, BoundedLogLink
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
    # BoundedLogLink round-trips and floors at b.
    lk = BoundedLogLink(b=-7.0)
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
# set.seed(23) stream (bit-exact rnorm); shares BoundedLogLink with gammals.
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
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
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
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
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
    mu = np.array([0.5, 1.0])
    rho = np.array([-0.2, 0.1])
    xi3 = np.array([0.1, 0.2])
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
        cp = beta.copy()
        cp[k] += h
        cm = beta.copy()
        cm[k] -= h
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


def test_cox_ph_predict_se_matches_mgcv():
    # End-to-end hazard-GAUGE pin (mgcv 1.9-4, deterministic data — no RNG
    # stream): the predict-with-se survivor pieces consume the X-weighted
    # baseline-hazard `a` vectors, which mgcv builds from the ORIGINAL-gauge
    # X (coxph.r:52 un-reparas G$X before family$hazard). hea's engine sets
    # cox_ph's fit context with the never-repara'd md.X + un-repara'd coefs
    # (gam.py `set_fit_context` call) — same gauge by construction. A wrong
    # (initial-repara'd) X here would corrupt the `a`-vector quadratic
    # forms at O(1); the observed agreement is ~1e-5 (the sp endpoint).
    from hea.family import cox_ph
    n = 150
    i = np.arange(n)
    x = (i + 0.5) / n
    z = ((7 * i) % 11) / 10
    u = (((3 * i + 1) % 17) + 0.5) / 17.5
    h0 = np.exp(0.7 * np.sin(2 * np.pi * x) + 0.3 * z)
    t0 = -np.log(1 - u) / h0
    d = ((i % 4) != 0).astype(float)
    time = np.where(d > 0, t0, t0 * 0.7)
    df = pl.DataFrame({"time": time, "x": x, "z": z, "d": d})
    m = hea.models.gam("time ~ s(x) + z", df, family=cox_ph(), weights=d,
                       method="REML")
    np.testing.assert_allclose(m.sp, [0.2031654074], rtol=2e-4)
    np.testing.assert_allclose(float(m.edf_total), 4.4506640787, rtol=1e-4)
    np.testing.assert_allclose(m.deviance, 175.0787132143, rtol=1e-5)
    pr = m.predict(se_fit=True, type="response")
    fit = pr["fit"].to_numpy()
    se = pr["se.fit"].to_numpy()
    ix = [0, 1, 2, 73, 148]
    np.testing.assert_allclose(
        fit[ix], [0.9419122191, 0.7060079479, 0.5640457392,
                  0.07442545795, 0.9253112995], rtol=1e-4)
    np.testing.assert_allclose(
        se[ix], [0.02707868482, 0.08956395605, 0.1063837158,
                 0.04154851196, 0.03308801276], rtol=1e-4)


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
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
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
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
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


def test_tweedious_work_census_matches_mgcv():
    """Census gate for the faithful tweedious/tweedious2 port (per-term θ-chain,
    up-then-down sweep from the located peak, ``eps = double.eps²``) vs live
    mgcv ldTweedie(all.derivs=TRUE). Block A: constant θ/ρ (buffer=TRUE scalar
    C_tweedious). Block B: per-row θ/ρ (vector C_tweedious2 with the lgamma(j+1)
    recursion). ``y[0]=0`` exercises the closed-form point mass.

    Pins: 1e-13 on the density + FIRST-derivative columns (l, ρ, θ — LSTH1's
    columns — and the μ columns, which are exact closed forms) and 1e-12 on the
    SECOND derivatives (ρρ, θθ, θρ). The 2nd-derivative floor is intrinsic: the
    ``w2i/wi − (w1i/wi)²`` cancellation (mgcv misc.c:499-500) amplifies the
    ~1-ulp exp/accumulation floor ~150×; an FMA audit (accumulation- and
    base-site fusion) and R-oracle special functions were both shown NOT to
    reduce it, so the residual is the reference's own last-ulp imprecision, not
    a porting gap. 17-digit refs from mgcv 1.9-4; the pins carry cross-platform
    headroom over the measured drift (≤2.1e-14 / ≤1.5e-13).
    """
    from hea.family import _ld_tweedie_work
    a, b = 1.001, 1.999
    y = np.array([0.0, 0.260876, 5.494473, 1.292911, 7.045937, 6.365127,
                  9.920183, 4.274371, 8.754516, 7.918497, 1.394391,
                  0.273828, 5.444585, 12.572887, 0.550657])
    mu = np.array([3.092199, 2.486576, 2.42595, 0.914386, 2.04399, 2.405801,
                   2.945419, 0.515313, 0.685186, 3.115468, 2.629075,
                   1.404792, 0.777984, 2.211337, 2.981747])
    # column tolerances: 1st derivs + μ closed forms 1e-13; 2nd derivs 1e-12
    tol = np.array([1e-13, 1e-13, 1e-12, 1e-13, 1e-12, 1e-12,
                    1e-13, 1e-13, 1e-13, 1e-13])

    # ---- Block A: constant theta=0.42, rho=-0.18 (buffer path) ----
    ref_a = np.array([
        [-4.7226478375437413, 4.7226478375437413, -4.7226478375437413, -1.5696696786364239, -1.9081340122507715, 1.5696696786364239, -0.60590744724664036, 0.11821024223800798, 0.16334743766531182, 0.60590744724664036],
        [-1.6083521475249245, 1.0031510966457988, -1.9332577242326785, 0.31263051728590119, -0.15288898813956719, 0.081796435719003302, -0.61855845699358936, 0.12091323045959126, 0.13455874500358009, 0.61855845699358925],
        [-3.0805284683816794, 0.2997152403503307, -0.89344764873526117, 0.012569889185853445, -0.07109672986904858, -0.25009791535954307, 0.88721822553969831, -0.87548533439914056, -0.18777201620892311, -0.8872182255396982],
        [-1.1943640066601118, -0.50475471006910233, -0.18716727544502287, -0.030795792745489936, 0.016107960352173212, 0.003979707263191079, 0.52310348883908286, -2.2991572922796197, 0.011180982609209232, -0.52310348883908275],
        [-4.6700608899589611, 1.6986055417553931, -2.2820969568054235, 0.34492569663210126, -0.24493061597152632, -0.61362686556897073, 1.9033946717838528, -1.8735260908604032, -0.32496287050031308, -1.9033946717838524],
        [-3.6404986874353042, 0.74705717105487324, -1.3345699452067112, 0.12957589824902094, -0.13930618447627818, -0.38563525544368238, 1.1601913736793128, -1.0662033769393515, -0.24323357527346059, -1.1601913736793126],
        [-5.1624028963438509, 1.9271793916362547, -2.4985954695511765, 0.57347897344356991, -0.44101105227858817, -0.88394963508852786, 1.4775089859163844, -1.0160870527228745, -0.38116423234311447, -1.4775089859163841],
        [-8.1125799672282142, 5.5231492262063639, -6.1287129375144076, -0.20016137880103146, -0.040811505725868003, -0.0049773355173690081, 13.028115860440705, -43.999738452679857, 2.0627180103847307, -13.028115860440701],
        [-15.194696655116093, 12.056035436848612, -12.631618739961366, 0.90318089736445373, -0.54482151494454101, -1.1985171736598259, 17.711479961790225, -43.638252458918515, 1.5991108208270199, -17.711479961790232],
        [-3.8095365766900695, 0.74825480714088322, -1.3273831921118973, 0.17779526120444133, -0.19789010229188886, -0.46087789609765695, 0.92989523374557403, -0.6721469367190952, -0.2523565995824707, -0.92989523374557392],
        [-1.4359069613920497, -0.31814283393871179, -0.36640220138995261, 0.005789360321502901, -0.00076539691664390119, -0.045754563728800823, -0.31381136846395641, -0.062793214982665616, 0.072441456503184301, 0.31381136846395652],
        [-0.85239951038377426, 0.21653302534779506, -1.1372796873809765, 0.19177051882674068, -0.10136484039091109, 0.18429203604440225, -0.7851613776536438, 0.2018566889552042, 0.063731428927854633, 0.78516137765364424],
        [-7.7347354612309367, 4.9608941833769951, -5.5550286694949964, 0.25015209973035502, -0.15736221196821631, -0.486521137656029, 8.3556034950366627, -19.009812976550318, 0.50094976011764902, -8.3556034950366627],
        [-8.8244727672649521, 5.4056522822948594, -5.9698564201380417, 1.5823365279373149, -1.0001507218475414, -1.9211672351134155, 3.4755269016836201, -2.8552719097331773, -0.65868555768432835, -3.4755269016836197],
        [-1.737228249377277, 0.64047107760248334, -1.43974297841309, 0.26110063952984586, -0.084559943229046031, -0.10526036319805643, -0.50497081351605611, 0.063807521808889489, 0.13174943081423771, 0.50497081351605611],
    ])
    ha = _ld_tweedie_work(y, mu, np.full(y.size, 0.42),
                          np.full(y.size, -0.18), a=a, b=b)
    da = np.abs(ha - ref_a)
    assert np.all(da <= tol), (
        "block A over tol at cols "
        f"{np.where((da > tol).any(axis=0))[0]}, max/col {da.max(axis=0)}")

    # ---- Block B: per-row theta/rho (tweedious2 path) ----
    th_b = np.array([-0.619448, 0.572273, -0.6759, 0.564558, -0.778581,
                     1.180143, -0.987644, -0.277671, 0.374675, 0.328789,
                     -0.70749, -0.87141, -0.401739, -0.593529, 0.595368])
    rho_b = np.array([-0.225026, 0.15186, 0.470401, 0.161344, -0.210919,
                      0.16348, 0.378983, -0.081667, 0.033346, -0.286301,
                      0.136659, -0.374453, 0.198765, 0.441105, 0.208678])
    ref_b = np.array([
        [-4.0135151190716618, 4.0135151190716618, -4.01351511907166, -0.37362948718044275, -0.63682363573120992, 0.37362948718044275, -0.84339761469370589, 0.095519049727411029, 0.21614230048805883, 0.84339761469370589],
        [-1.3218367597016942, 0.45912279248925847, -1.4798146257634057, 0.32504032145587169, -0.213769814212454, 0.15708628769562827, -0.42965501973369941, 0.090161408870314361, 0.090070322911922643, 0.42965501973369924],
        [-2.9436597498570594, 0.045020071116603333, -0.65963055362297851, -0.058145984291012809, -0.069244179728089605, -0.18321781352929989, 0.58594430812455112, -0.51400275838100984, -0.11582051842018701, -0.58594430812455112],
        [-1.3818582804398387, -0.57124891011751711, -0.21496671026132308, -0.024118068459787256, 0.020153990710539027, 0.021942680870590259, 0.37296120126773802, -1.6530979336228049, 0.0076986412837158945, -0.37296120126773802],
        [-5.3108844809731011, 2.6730170573613208, -3.2151185728770777, 0.56254497224675859, 0.00035884233298233426, -0.79240495247722009, 2.4124678750880859, -2.0343617504834608, -0.37116050210226303, -2.4124678750880859],
        [-3.4616369204151454, 0.17679190184085236, -0.83808098594142155, -0.024163778105698919, -0.020590476926250822, -0.17585840444955902, 0.71435086150006677, -0.70433590658080458, -0.11252315767455587, -0.71435086150006688],
        [-4.8959740898809159, 1.7759618522856471, -2.33051005318816, 0.44125270133862671, -0.013593173962206606, -0.69571212909791313, 1.2085417112207109, -0.69512303476895543, -0.25762825655984345, -1.2085417112207111],
        [-7.4734646538016687, 4.9740798887143356, -5.563078231917558, -0.14792978785575528, -0.10664279003415178, -0.064024711562353787, 10.534639727549195, -32.059996002483103, 1.7094121524984058, -10.534639727549195],
        [-12.921708124840512, 9.6719502054395079, -10.266577324515245, 0.69147447286055019, -0.42880500162274782, -0.99860216365577914, 14.249989831796757, -34.883559417337068, 1.2980700485288366, -14.249989831796755],
        [-3.9186892428581377, 0.94411884783507993, -1.5109813872349864, 0.25010592696137479, -0.23109001414197117, -0.53306701743110807, 1.0603393945235318, -0.75895563760381268, -0.29265384411973938, -1.060339394523532],
        [-1.5394029791190369, -0.36431170636571952, -0.39848942525833841, -0.022184841297629987, 0.0085420708319730032, -0.12530324298298434, -0.29761888650271917, -0.090432933969829529, 0.063495589333968275, 0.29761888650271912],
        [-1.1724235641680947, -0.0099735379975003369, -2.4047859812912211, 0.25559082276264228, -0.2574656548431713, 0.9282376959228793, -1.0589106224895881, 0.040140509529614565, 0.074697317856199333, 1.0589106224895881],
        [-6.3175419733565654, 3.5220775128515918, -4.120467455282494, 0.16507344746249863, -0.061081495066402702, -0.41235831397188516, 5.4379892380685977, -10.958724371078796, 0.32723622596535146, -5.4379892380685977],
        [-7.3522537632620502, 3.9364813409109285, -4.4996225711672331, 1.1047418931119566, -0.17962916067186896, -1.4338262124096488, 2.272290829990594, -1.6127944310930862, -0.41251074723493852, -2.2722908299905935],
        [-1.5468639531825241, 0.1494990857190186, -1.068244375687124, 0.22575174229707606, -0.10292788676574682, 0.0035353533352247002, -0.32733907104087095, 0.045866538242877397, 0.081764267342503374, 0.32733907104087118],
    ])
    hb = _ld_tweedie_work(y, mu, th_b, rho_b, a=a, b=b)
    db = np.abs(hb - ref_b)
    assert np.all(db <= tol), (
        "block B over tol at cols "
        f"{np.where((db > tol).any(axis=0))[0]}, max/col {db.max(axis=0)}")


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
    from hea.family import mvn
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
        cp = beta.copy()
        cp[k] += h
        cm = beta.copy()
        cm[k] -= h
        fd_lb[k] = (fam.ll(Y, X, cp, lpi=lpi)["l"]
                    - fam.ll(Y, X, cm, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(Y, X, cp, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(Y, X, cm, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=1e-5, atol=1e-5)


def test_mvn_components_and_validation():
    from hea.family import mvn
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
# Prior weights in general-family ll() — the weighted-likelihood contract.
#
# hea's general families honour gam(weights=) as a weighted log-likelihood:
# l = Σ wᵢ·l0ᵢ and every per-observation derivative row scales by wᵢ. This is
# a defined extension beyond mgcv, whose gamlss families drop prior weights.
# The gate is the *duplication identity* — weighting row i by integer wᵢ is
# exactly the unweighted fit on the design with row i repeated wᵢ times — which
# holds to machine precision for l/lb/lbb because gamlss_etamu/gamlss_gH are
# linear, row by row, in (l1..l4). At unit weights the scaling is the identity,
# so every oracle pin above is bit-for-bit unchanged (cox_ph, whose prior
# weights ARE the censoring indicator, and mvn, fit by bfgs, are excluded).
# ---------------------------------------------------------------------------

_WEIGHTED_LL_FAMILIES = ["gaulss", "gammals", "gumbls", "gevlss", "ziplss",
                         "shash", "twlss", "multinom"]


def _twlss_weighted_design():
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
    return X, yt, coef, lpi


def _weighted_ll_build(name):
    """One (family, X, y, coef, lpi) row per weightable general family,
    reusing the per-family oracle designs above."""
    if name == "gaulss":
        from hea.family import gaulss
        X, y, coef, _, _, lpi = _gaulss_oracle_inputs()
        return gaulss(), X, y, coef, lpi
    if name == "gammals":
        from hea.family import gammals
        X, y, coef, _, _, lpi = _gammals_oracle_inputs()
        return gammals(), X, y, coef, lpi
    if name == "gumbls":
        from hea.family import gumbls
        X, y, coef, _, _, lpi = _gumbls_oracle_inputs()
        return gumbls(), X, y, coef, lpi
    if name == "gevlss":
        from hea.family import gevlss
        X, y, coef, _, _, lpi = _gevlss_oracle_inputs()
        return gevlss(), X, y, coef, lpi
    if name == "ziplss":
        from hea.family import ziplss
        X, y, coef, _, _, lpi = _ziplss_oracle_inputs()
        return ziplss(), X, y, coef, lpi
    if name == "shash":
        from hea.family import shash
        y, X, lpi, coef, _, _ = _shash_oracle_inputs()
        return shash(), X, y, coef, lpi
    if name == "twlss":
        from hea.family import twlss
        X, y, coef, lpi = _twlss_weighted_design()
        return twlss(), X, y, coef, lpi
    if name == "multinom":
        from hea.family import multinom
        X, y, coef, _, _, lpi = _multinom_oracle_inputs(3, 41)
        return multinom(3), X, y, coef, lpi
    raise ValueError(name)


@pytest.mark.parametrize("name", _WEIGHTED_LL_FAMILIES)
def test_general_family_weighted_ll_duplication_identity(name):
    # The hard gate: integer weights ≡ row duplication, exact for l/lb/lbb.
    fam, X, y, coef, lpi = _weighted_ll_build(name)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    rng = np.random.default_rng(7)
    w = rng.integers(1, 5, size=n).astype(float)
    reps = w.astype(int)
    Xd = np.repeat(X, reps, axis=0)
    yd = np.repeat(y, reps, axis=0)

    rw = fam.ll(y, X, coef, w, lpi=lpi, deriv=1)
    rd = fam.ll(yd, Xd, coef, np.ones(yd.shape[0]), lpi=lpi, deriv=1)
    for k in ("l", "lb", "lbb"):
        np.testing.assert_allclose(
            np.asarray(rw[k]), np.asarray(rd[k]), rtol=1e-8, atol=1e-8,
            err_msg=f"{name}: {k} breaks the weight=duplication identity")

    # l0 is the raw per-observation log-density — never scaled by wt.
    l0_w = fam.ll(y, X, coef, w, lpi=lpi, deriv=0)["l0"]
    l0_1 = np.asarray(fam.ll(y, X, coef, lpi=lpi, deriv=0)["l0"])
    np.testing.assert_array_equal(np.asarray(l0_w), l0_1)
    # the weighted objective is exactly Σ wt·l0.
    np.testing.assert_allclose(rw["l"], float(np.sum(w * l0_1)),
                               rtol=0, atol=1e-9)
    # wt=None is the unit-weight path, bit-for-bit (so the oracle pins hold).
    r_none = fam.ll(y, X, coef, lpi=lpi, deriv=1)
    r_ones = fam.ll(y, X, coef, np.ones(n), lpi=lpi, deriv=1)
    for k in ("l", "lb", "lbb"):
        np.testing.assert_array_equal(np.asarray(r_none[k]),
                                      np.asarray(r_ones[k]))


def test_gaulss_weighted_ll_matches_fd():
    # Non-integer precision weights: lb/lbb are the gradient/Hessian of the
    # weighted objective Σ wt·l0, and the sp-derivative blocks inherit it.
    from hea.family import gaulss
    X, y, coef, d1b, d2b, lpi = _gaulss_oracle_inputs()
    fam = gaulss()
    n = y.shape[0]
    rng = np.random.default_rng(3)
    w = rng.uniform(0.3, 2.5, size=n)
    r1 = fam.ll(y, X, coef, w, lpi=lpi, deriv=1)
    h = 1e-6
    p = coef.size
    fd_lb = np.zeros(p)
    fd_lbb = np.zeros((p, p))
    for k in range(p):
        cp = coef.copy()
        cp[k] += h
        cm = coef.copy()
        cm[k] -= h
        fd_lb[k] = (fam.ll(y, X, cp, w, lpi=lpi)["l"]
                    - fam.ll(y, X, cm, w, lpi=lpi)["l"]) / (2 * h)
        fd_lbb[:, k] = (fam.ll(y, X, cp, w, lpi=lpi, deriv=1)["lb"]
                        - fam.ll(y, X, cm, w, lpi=lpi, deriv=1)["lb"]) / (2 * h)
    np.testing.assert_allclose(r1["lb"], fd_lb, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r1["lbb"], fd_lbb, rtol=0, atol=1e-6)
    # deriv=3: each ∂H/∂ρ matrix is the FD of the weighted Hessian along d1b.
    r3 = fam.ll(y, X, coef, w, lpi=lpi, deriv=3, d1b=d1b)
    for j in range(d1b.shape[1]):
        fdH = (fam.ll(y, X, coef + h * d1b[:, j], w, lpi=lpi,
                      deriv=1)["lbb"]
               - fam.ll(y, X, coef - h * d1b[:, j], w, lpi=lpi,
                        deriv=1)["lbb"]) / (2 * h)
        np.testing.assert_allclose(r3["d1H"][j], fdH, rtol=0, atol=1e-5)
    # deriv=4: trHid2H stays finite under non-unit weights (the gam.fit5
    # preconditioned-Cholesky fh/D convention).
    from scipy.linalg import cholesky
    Hp = -r1["lbb"] + np.eye(p) * 0.5
    D = 1.0 / np.sqrt(np.diag(Hp))
    R = cholesky(D[:, None] * Hp * D[None, :], lower=False)
    r4 = fam.ll(y, X, coef, w, lpi=lpi, deriv=4, d1b=d1b, d2b=d2b,
                fh=(R, np.arange(p)), D=D)
    assert np.all(np.isfinite(r4["trHid2H"]))


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
        thp = th.copy()
        thp[k] += h
        thm = th.copy()
        thm[k] -= h
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


# ---------------------------------------------------------------------------
# ziP (single-formula zero-inflated Poisson, efam.r:3848-4147). The 1 LP μ is
# the log Poisson mean γ; presence p = 1−exp(−exp(θ₁+(b+e^θ₂)·γ)). Its Dd
# chains the shared `zipll` kernel (also used by the 2-LP ziplss general
# family) through the affine `lind` map. Oracle: live ziP()$Dd / dev.resids /
# aic / saturated.ll (Rscript, mgcv 1.9-4) on a fixed (y, γ, θ) table with a
# y=0/y>0 mix lighting both zipll branches.
# ---------------------------------------------------------------------------

def _zip_dd_inputs():
    y = np.array([0, 1, 2, 0, 3, 0, 5, 1.0])
    mu = np.array([-0.5, 0.3, 0.8, 0.1, 1.2, -0.2, 1.5, 0.4])  # log Pois mean
    th = np.array([-0.3, 0.5])
    wt = np.array([1.0, 1, 1, 2, 1, 1, 1, 2])
    return y, mu, th, wt


def test_ziP_components_match_mgcv():
    from hea.family import ziP
    y, mu, th, wt = _zip_dd_inputs()
    fam = ziP()
    fam.set_theta(th)
    D = fam.Dd(y, mu, th, wt, level=2)
    np.testing.assert_allclose(
        D["Dmu"], [1.071207530364356, -0.045702953296894222,
                   0.37959551008497128, 5.7613192687163126,
                   0.80561117203009136, 1.7566400578209209,
                   -0.93847893199421539, 0.73725952565309738],
        rtol=0, atol=1e-9)
    # observed (Dmu2) ≠ expected (EDmu2) Hessian — ziP is the first extended
    # family where they differ (zipll's El2 term, gamlss.r:1620-1621).
    np.testing.assert_allclose(
        D["Dmu2"], [1.7661226406458672, 3.9501198151604182,
                    5.6134546907312206, 9.4988096256270929,
                    6.6359440750126044, 2.8962098282932556,
                    8.6578802094348006, 8.6715916500204102], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["EDmu2"], [1.6965919520630381, 4.7378473526773135,
                     6.2052588566875286, 7.6897751651358615,
                     6.7419479391751942, 2.6213385128948734,
                     8.663861663445525, 10.319785479013458], rtol=0, atol=1e-9)
    assert not np.allclose(D["Dmu2"], D["EDmu2"])
    np.testing.assert_allclose(
        D["Dmu3"], [2.9118439642979204, 3.8670656254252247,
                    2.3243406650630134, 15.660889476102508,
                    4.2024063395585829, 4.7750427483178566,
                    9.1778599926392399, 7.644548876750628], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu4"], [4.8008190808977655, 0.25290016932971637,
                    -4.10270399115561, 25.820441597333993,
                    15.434163778299446, 7.8727145476540485,
                    14.094571640120012, -2.3911089052853205],
        rtol=0, atol=1e-8)
    # θ-blocks (n×2): Dth, Dmuth, Dmu2th, Dmu3th.
    np.testing.assert_allclose(D["Dth"], np.column_stack([
        [0.64972021008103364, -1.0252845008523144, -0.37026076261421353,
         3.4944167768696115, -0.050742490399748402, 1.0654560531477617,
         -0.0026880592208130411, -1.7966052442293883],
        [-0.535603765182178, -0.5071225095223123, -0.48836543602216376,
         0.57613192687163128, -0.10039226790043468, -0.35132801156418425,
         -0.0066477906213841106, -1.1848405124849566]]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth"], np.column_stack([
        [1.071207530364356, 1.2297502965642129, 1.1937419527052584,
         5.7613192687163126, 0.36665767864736087, 1.7566400578209209,
         0.034509398702116112, 2.6118262198238336],
        [0.1881462100414224, -1.0821537235958119, 0.96406140429410825,
         6.7112002312790224, 0.64175935337136103, 1.1773980921622698,
         0.080912709104619271, -1.2396319036140457]]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmu2th"], np.column_stack([
        [1.7661226406458672, 0.87140601345902002, -1.4247644203663541,
         9.4988096256270929, -1.925801924952341, 2.8962098282932556,
         -0.37891588167269141, 1.2652595228228209],
        [2.0763232991427745, 4.4860426321321833, 2.0570639738287344,
         20.563708198864436, -2.6011000884861391, 4.8374111069229393,
         -0.82329725152128841, 9.4467710032852779]]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmu3th"], np.column_stack([
        [2.9118439642979204, -1.8281291989265558, -6.0042497164072683,
         15.660889476102508, 4.7570126547021756, 4.7750427483178566,
         3.324939807886468, -5.9597820298435131],
        [6.3351223524448779, 3.405894240660206, -14.966585594412944,
         49.564712588040926, -0.11374625191593601, 12.750585335422757,
         6.348668455830448, 2.3277731041658822]]), rtol=0, atol=1e-8)
    # θ²-blocks (n×3, ordered th1th1, th1th2, th2th2) — third col pins the
    # th2th2 term (the only nonzero p.th2 entry).
    np.testing.assert_allclose(D["Dth2"][:, 2], [
        -0.094073105020711201, -0.32464611707874358, 0.77124912343528684,
        0.6711200231279022, 0.77011122404563315, -0.23547961843245402,
        0.12136906365692891, -0.49585276144561841], rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth2"][:, 2], [
        -0.85001543952996506, 0.26365906604384293, 2.6097125833570955,
        8.767571051165465, -2.4795607528120067, 0.20991587077768203,
        -1.1540331681773133, 2.5390764977000653], rtol=0, atol=1e-8)
    np.testing.assert_allclose(D["Dmu2th2"][:, 2], [
        0.98508542206310956, 9.9938535364624279, -7.8591405278728868,
        46.083887656532966, -5.3386956792714049, 7.124705146761328,
        7.8764081807030975, 19.824651248236911], rtol=0, atol=1e-8)

    # dev_resids (the −2logLik), aic (≡ Σ dev), saturated_ll (−2 sat ll).
    np.testing.assert_allclose(
        fam.dev_resids(y, mu, wt, theta=th),
        [0.64972021008103364, 2.2035637592855939, 2.5381524505282114,
         1.7472083884348057, 2.9595657277001139, 1.0654560531477617,
         3.5159100149128975, 2.2193885014463297], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fam.aic(y, mu, 0, wt, 0, theta=th),
                               20.865561995417881, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        fam.saturated_ll(y, wt),
        [0, 2.203300357324264, 2.5251914898914745, 0, 2.9094837035556234,
         0, 3.4669063924766128, 2.203300357324264], rtol=0, atol=1e-8)
    np.testing.assert_allclose(fam.get_theta(trans=True),
                               [-0.3, 1.6487212707001282], rtol=0, atol=1e-12)
    # ls ≡ 0.
    le = fam.ls_extended(y, wt)
    assert le["ls"] == 0.0 and float(np.sum(np.abs(le["LSTH1"]))) == 0.0


def test_ziP_b_nonzero_components_match_mgcv():
    # ziP(b=0.3): the b>0 presence-slope floor (slope = b + e^θ₂ > b,
    # efam.r:3869 `.b`, threaded into `lind(k=b)` at efam.r:3900 and the
    # presence LP of dev.resids/aic). Oracle: live ziP(b=0.3)$Dd /
    # dev.resids / aic / getTheta(TRUE) (Rscript, mgcv 1.9-4) on the same
    # component table as the b=0 test — every output shifts with b, so
    # these pins are b-differentiating. (mgcv's `logid` alternative map
    # is dead code — zero call sites in the package — so `lind` is the
    # whole b surface; see the divergence-audit plan.)
    from hea.family import ziP
    y, mu, th, wt = _zip_dd_inputs()
    fam = ziP(b=0.3)
    fam.set_theta(th)
    D = fam.Dd(y, mu, th, wt, level=2)
    np.testing.assert_allclose(
        D["Dmu"], [1.08976267596958154, -0.22004623489546793,
                   0.57215865967827506, 7.01702884537704197,
                   0.87542275833633465, 1.95536392161329253,
                   -0.93410281687360253, 0.57445371903023101],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu2"], [2.1236437066770151, 4.8577163632621687,
                    5.7862868392389384, 13.6742333681026036,
                    6.2117657841435481, 3.8104592660074412,
                    8.6023719446642435, 10.5532768670240706],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["EDmu2"], [2.0180607160596753, 6.2440825914745579,
                     6.4056243610448726, 10.2069022340631523,
                     6.2358673592575826, 3.3434058534384645,
                     8.6024716464923259, 13.3335630486959023],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu3"], [4.1383896625899634, 4.4868711592001551,
                    -1.1538055877588214, 26.6472694249389974,
                    5.4317245891259009, 7.4255230228050992,
                    9.7709378844557797, 7.6392877628959575],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        D["Dmu4"], [8.0645679619345891, -4.5612557172249000,
                    -4.9240133146708649, 51.9281007344558034,
                    23.2449206230913035, 14.4702746606138106,
                    9.2611868607395618, -17.7249681840395397],
        rtol=0, atol=1e-8)
    np.testing.assert_allclose(D["Dth"], np.column_stack([
        [0.55921936726130994, -0.95691039796727217, -0.21444505775023529,
         3.6008376112485259, -0.0071065252647155115, 1.0034087229472164,
         -2.8606088762550079e-05, -1.6035680088371769],
        [-0.46099843289559428, -0.47330355818482989, -0.28284810248746428,
         0.59367775620024843, -0.014060015237645187, -0.33086826094582555,
         -7.0745200521528338e-05, -1.0575346740736420]]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth"], np.column_stack([
        [1.08976267596958154, 1.50617333726872316, 1.09865886407181312,
         7.01702884537704197, 0.09254172245864013, 1.95536392161329253,
         0.00071232090990356114, 3.17534311039586292],
        [0.0236394138481182958, -0.8327005217278636851, 1.0955456626414211,
         7.09369003345143323, 0.1713739281290409533, 1.0095712868244369,
         0.0017144644865693542, -0.5497343940317986899]]),
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dth2"][:, 2], [
        -0.0809694718583982564, -0.2842163189121490463,
        0.6979680159088302860, 0.6915586706591358990,
        0.1718251907663267686, -0.2217663530216369416,
        0.0021649004419334612, -0.3488461525583514966], rtol=0, atol=1e-9)
    np.testing.assert_allclose(D["Dmuth2"][:, 2], [
        -0.755905910445411555, 0.560366653477998300, 0.878061772576305133,
        9.242050942630795518, -1.524462157900505987, 0.131161416228383843,
        -0.046277012289079662, 3.232019812313918106], rtol=0, atol=1e-8)
    np.testing.assert_allclose(D["Dmu2th2"][:, 2], [
        0.40161458642977266, 10.95259405800349484, -15.51518510456715916,
        52.97034126309804947, 8.03760319065614226, 6.80845044097395746,
        0.87271760716416480, 19.53488770370943328], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        fam.dev_resids(y, mu, wt, theta=th),
        [0.55921936726130994, 2.11434673657913752, 2.46875973990692454,
         1.80041880562426293, 2.95104173975606976, 1.00340872294721639,
         3.51560614274561978, 2.11735606584551572], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fam.aic(y, mu, 0, wt, 0, theta=th),
                               20.447932192135838, rtol=0, atol=1e-9)
    # getTheta(trans): θ₂ ↦ b + e^θ₂ (the slope floor).
    np.testing.assert_allclose(fam.get_theta(trans=True),
                               [-0.3, 1.94872127070012824],
                               rtol=0, atol=1e-12)


def test_ziP_Dd_matches_fd():
    from hea.family import ziP
    y, mu, th, wt = _zip_dd_inputs()
    fam = ziP()
    fam.set_theta(th)
    D = fam.Dd(y, mu, th, wt, level=1)
    h = 1e-6
    # ziP.dev_resids does NOT carry wt (mgcv: −2·zipll$l), but Dd does
    # (−2·wt·…), so Dd = wt · FD(dev_resids).
    fd_mu = (fam.dev_resids(y, mu + h, wt, theta=th)
             - fam.dev_resids(y, mu - h, wt, theta=th)) / (2 * h)
    np.testing.assert_allclose(D["Dmu"], wt * fd_mu, rtol=1e-5, atol=1e-5)
    for k in range(2):
        thp = th.copy()
        thp[k] += h
        thm = th.copy()
        thm[k] -= h
        fd_thk = (fam.dev_resids(y, mu, wt, theta=thp)
                  - fam.dev_resids(y, mu, wt, theta=thm)) / (2 * h)
        np.testing.assert_allclose(D["Dth"][:, k], wt * fd_thk,
                                   rtol=1e-5, atol=1e-5)


def test_ziP_construction_and_validation():
    from hea.family import ziP
    assert ziP().n_theta == 2
    assert ziP(theta=[-1.0, 0.5]).n_theta == 0       # fixed θ supplied
    np.testing.assert_allclose(ziP().get_theta(), [0.0, 0.0])  # start Poisson
    # getTheta(trans): θ₂ → b + e^θ₂ (the presence slope).
    np.testing.assert_allclose(ziP(theta=[-1.0, 0.5], b=0.2).get_theta(True),
                               [-1.0, 0.2 + np.exp(0.5)], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="2 params"):
        ziP().set_theta([0.1])
    with pytest.raises(ValueError, match="not available"):
        ziP(link="log")
    # initialize validation: negatives, non-integer, binary-only all rejected.
    with pytest.raises(ValueError, match="negative"):
        ziP().initialize(np.array([0.0, 1, -1]), np.ones(3))
    with pytest.raises(ValueError, match="Non-integer"):
        ziP().initialize(np.array([0.0, 1.5, 2]), np.ones(3))
    with pytest.raises(ValueError, match="binary"):
        ziP().initialize(np.array([0.0, 1, 0, 1]), np.ones(4))
    # mustart = log(y + (y==0)/5).
    np.testing.assert_allclose(
        ziP().initialize(np.array([0.0, 2, 5]), np.ones(3)),
        np.log(np.array([0.2, 2.0, 5.0])), rtol=0, atol=1e-12)


def _cnorm_dd_inputs():
    # 5 obs covering all four censoring cases (mgcv efam.r:836-843):
    #   i0,i2 uncensored (yat==y); i1 interval [-0.5,0.8]; i3 left (-∞);
    #   i4 right (+∞).
    y = np.array([1.2, -0.5, 2.0, 0.7, -1.0])
    yat = np.array([1.2, 0.8, 2.0, -np.inf, np.inf])
    mu = np.array([0.5, 0.1, 1.7, 0.3, -0.4])
    wt = np.array([1.0, 1.0, 1.0, 2.0, 1.0])
    return y, yat, mu, wt


def test_cnorm_components_match_mgcv():
    from hea.family import cnorm
    y, yat, mu, wt = _cnorm_dd_inputs()
    th = np.array([0.3])
    fam = cnorm()
    fam.set_theta(th)
    fam.set_censor(yat)
    D = fam.Dd(y, mu, th, wt, level=2)
    # cnorm has a single log-scale θ, so every θ-block is a length-n vector
    # (no n×2 packing like ziP). Full level-2 table vs mgcv 1.9-4.
    np.testing.assert_allclose(
        D["Dmu"], [-0.768336290531637, -0.0507690995168156,
                   -0.329286981656416, 1.1558448812426, -0.797264561425975],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        D["Dmu2"], [1.09762327218805, 1.01538528180947, 1.09762327218805,
                    1.17546159102389, 0.580344231466461], rtol=0, atol=1e-12)
    # observed == expected Hessian for cnorm (mgcv sets EDmu2 = Dmu2,
    # EDmu2th = Dmu2th, efam.r:1056-1059).
    np.testing.assert_array_equal(D["EDmu2"], D["Dmu2"])
    np.testing.assert_array_equal(D["EDmu2th"], D["Dmu2th"])
    np.testing.assert_allclose(
        D["Dth"], [1.46216459662785, 1.84780435468546, 1.90121390550308,
                   0.462337952497041, 0.478358736855585], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        D["Dmuth"], [1.53667258106327, 0.0935707485602448,
                     0.658573963312832, -0.685660244833047,
                     0.449058022546098], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        D["Dmu3"], [0, -0.000197480812203978, 0, 0.60605462153934,
                    -0.21623962112153], rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        D["Dmu2th"], [-2.19524654437611, -1.87144071001021,
                      -2.19524654437611, -2.10850133343205, -1.03094469026],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        D["Dmu4"], [0, 0.00394885852463045, 0, -0.232125048982469,
                    -0.0565948284912688], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        D["Dth2"], [1.07567080674429, 0.294277388026857, 0.19757218899385,
                    -0.274264097933219, -0.269434813527659],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        D["Dmuth2"], [-3.07334516212655, -0.156286864452314,
                      -1.31714792662566, -0.157740288539771,
                      0.169508791609905], rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        D["Dmu2th2"], [4.39049308875221, 3.12593735397926, 4.39049308875221,
                       3.45259711317969, 1.6522839242444], rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        D["Dmu3th"], [0, 0.00154423954643431, 0, -1.91101388421101,
                      0.682675760459353], rtol=0, atol=1e-10)

    # dev_resids is the PROPER deviance (≥ 0; uncensored → z²), distinct
    # from the −2logLik that Dd differentiates.
    dr = fam.dev_resids(y, mu, wt)
    np.testing.assert_allclose(
        dr, [0.268917701686073, 0.00126922645931482, 0.0493930472484624,
             0.823718399219946, 0.796017476899608], rtol=0, atol=1e-12)
    assert np.all(dr >= 0.0)
    # aic = Σ(−2logLik); ls is a genuinely NONZERO saturated log-lik whose
    # θ-derivatives are forced to zero (Dd already carries them).
    np.testing.assert_allclose(fam.aic(y, mu, 0, wt, 0), 7.60432360893484,
                               rtol=0, atol=1e-11)
    le = fam.ls_extended(y, wt)
    np.testing.assert_allclose(le["ls"], -3.43250387871072, rtol=0, atol=1e-11)
    assert float(np.sum(np.abs(le["lsth1"]))) == 0.0
    assert float(np.sum(np.abs(le["LSTH1"]))) == 0.0
    np.testing.assert_allclose(fam.get_theta(trans=True), [np.exp(0.3)],
                               rtol=0, atol=1e-12)


def test_cnorm_Dd_matches_fd():
    from hea.family import cnorm, _cnorm_dpnorm
    from scipy.special import log_ndtr
    y, yat, mu, wt = _cnorm_dd_inputs()
    th0 = 0.3
    fam = cnorm()
    fam.set_theta([th0])
    fam.set_censor(yat)
    D = fam.Dd(y, mu, np.array([th0]), wt, level=1)
    log2pi = float(np.log(2.0 * np.pi))

    def m2ll(mu_, th_):
        # per-datum −2logLik — what cnorm's Dd differentiates (NOT the
        # proper deviance dev_resids, which folds in the θ-dependent
        # saturated reference).
        thw = th_ - np.log(wt) / 2.0
        eth = np.exp(-thw)
        out = np.zeros(y.shape[0])
        iu = [0, 2]
        z = (y[iu] - mu_[iu]) * eth[iu]
        out[iu] = z ** 2 + log2pi + 2.0 * thw[iu]            # density: 1/σ Jac
        i = 1
        y0 = min(y[i], yat[i])
        y1 = max(y[i], yat[i])
        z0 = (y0 - mu_[i]) * eth[i]
        z1 = (y1 - mu_[i]) * eth[i]
        out[i] = -2.0 * _cnorm_dpnorm(np.array([z0]), np.array([z1]))[0]
        out[3] = -2.0 * log_ndtr((y[3] - mu_[3]) * eth[3])  # left
        out[4] = -2.0 * log_ndtr(-(y[4] - mu_[4]) * eth[4])  # right
        return out

    h = 1e-4
    fd_mu = (m2ll(mu + h, th0) - m2ll(mu - h, th0)) / (2 * h)
    fd_th = (m2ll(mu, th0 + h) - m2ll(mu, th0 - h)) / (2 * h)
    np.testing.assert_allclose(D["Dmu"], fd_mu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D["Dth"], fd_th, rtol=1e-5, atol=1e-6)
    fd_mu2 = (m2ll(mu + h, th0) - 2 * m2ll(mu, th0)
              + m2ll(mu - h, th0)) / h ** 2
    np.testing.assert_allclose(D["Dmu2"], fd_mu2, rtol=1e-4, atol=1e-5)


def test_cnorm_construction_and_validation():
    from hea.family import cnorm
    # θ intake (mgcv efam.r:743-753): None → working 0; θ>0 fixed (store
    # log θ, n_theta=0); θ<0 an initial value (store log|θ|).
    assert cnorm().n_theta == 1
    np.testing.assert_allclose(cnorm().get_theta(), [0.0])
    assert cnorm(theta=2.0).n_theta == 0
    np.testing.assert_allclose(cnorm(theta=2.0).get_theta(), [np.log(2.0)])
    np.testing.assert_allclose(cnorm(theta=2.0).get_theta(True), [2.0])
    assert cnorm(theta=-0.5).n_theta == 1
    np.testing.assert_allclose(cnorm(theta=-0.5).get_theta(), [np.log(0.5)])
    for lk in ("identity", "log", "sqrt"):
        assert cnorm(link=lk).link.name == lk
    with pytest.raises(ValueError, match="not available"):
        cnorm(link="logit")
    with pytest.raises(ValueError, match="1 param"):
        cnorm().set_theta([0.1, 0.2])
    # No censor set ⇒ all uncensored: dev_resids reduces to z² (= the
    # Gaussian deviance with σ = e^θ).
    fam = cnorm()
    fam.set_theta([0.0])
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.5, 1.5, 2.0])
    np.testing.assert_allclose(fam.dev_resids(y, mu, np.ones(3)),
                               (y - mu) ** 2, rtol=0, atol=1e-12)
    # identity link mustart = y; non-identity validmu requires μ > 0.
    np.testing.assert_array_equal(fam.initialize(y, np.ones(3)), y)
    assert cnorm(link="log").validmu(np.array([0.1, 1.0]))
    assert not cnorm(link="log").validmu(np.array([-0.1, 1.0]))
    # mustart floor is min(y>0) — the LOGICAL min (efam.r:1123): 1 when
    # every y is positive, 0 as soon as any y ≤ 0.
    np.testing.assert_array_equal(
        cnorm(link="log").initialize(np.array([0.3, 2.0]), np.ones(2)),
        [1.0, 2.0])
    np.testing.assert_array_equal(
        cnorm(link="log").initialize(np.array([0.0, 0.3, 2.0]), np.ones(3)),
        [0.0, 0.3, 2.0])


# ---------------------------------------------------------------------------
# cpois (censored Poisson) — mgcv ``cpois()`` (efam.r:344-537) + ``dppois``
# (efam.r:312-339). Slot oracle pinned to live mgcv 1.9-4 via hex-float
# transfer (values reproduced bit-identically on arm64 at pin time; the
# randomized 200-row × level-2 census was 0 DIFF / 1002 values).
# ---------------------------------------------------------------------------


def _cpois_dd_inputs():
    # 6 obs covering all cases: i0 uncensored, i1 interval [2,6], i2 left
    # (−∞), i3 right (+∞), i4 uncensored ZERO count (the mustart quirk
    # row), i5 interval given as yat < y (pmin/pmax swap).
    y = np.array([3.0, 2.0, 4.0, 1.0, 0.0, 5.0])
    yat = np.array([3.0, 6.0, -np.inf, np.inf, 0.0, 2.0])
    mu = np.array([2.5, 3.1, 1.7, 2.2, 0.9, 4.4])
    wt = np.ones(6)
    return y, yat, mu, wt


def test_cpois_components_match_mgcv():
    from hea.family import cpois
    y, yat, mu, wt = _cpois_dd_inputs()
    fam = cpois()
    fam.set_censor(yat)

    # dev.resids: the PROPER deviance (≥ 0), saturated reference included
    # (interval rows maximize over μ via the analytic lgamma mean).
    dr = fam.dev_resids(y, mu, wt)
    np.testing.assert_allclose(
        dr, [0.093929340763727609, 0.32427618396314406,
             0.060124358816485542, 0.87567736610300229, 1.8,
             0.046856172595730383], rtol=1e-14, atol=0)
    assert np.all(dr >= 0.0)

    # Dd level 2: μ-derivatives ONLY — no θ keys at any level (mgcv's
    # returned names are exactly these five; getTheta() is NULL).
    D = fam.Dd(y, mu, np.zeros(0), wt, level=2)
    assert set(D.keys()) == {"Dmu", "Dmu2", "EDmu2", "Dmu3", "Dmu4"}
    np.testing.assert_allclose(
        D["Dmu"], [-0.39999999999999991, -0.57472288165500895,
                   0.1310296735043821, -0.75536305631350764, 2.0,
                   0.1865592035083572], rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        D["Dmu2"], [0.95999999999999974, 0.62496911183805759,
                    0.18585982829290848, 0.6973028859562822, 0.0,
                    0.34589032104013429], rtol=1e-14, atol=1e-300)
    assert D["EDmu2"] is D["Dmu2"]     # mgcv: r$EDmu2 = r$Dmu2 (alias)
    np.testing.assert_allclose(
        D["Dmu3"], [-0.76799999999999979, -0.37803766150915585,
                    0.094454617880994629, -0.75099712064338142, 0.0,
                    -0.14239116315360656], rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(
        D["Dmu4"], [0.92159999999999442, 0.37863924185178632,
                    -0.12641915824024608, 1.0331210386956542, 0.0,
                    0.093556511921981667], rtol=1e-13, atol=1e-300)

    # dDeta must survive the θ-free Dd (R's NULL list reads): θ entries
    # come back as None, the μ-chain as usual.
    dd = fam.dDeta(y, mu, wt, np.zeros(0), level=2, dd=D)
    assert dd["Dth"] is None and dd["Detath"] is None
    assert dd["Dth2"] is None and dd["Deta3th"] is None
    assert np.all(np.isfinite(dd["Deta3"])) and np.all(np.isfinite(dd["Deta4"]))

    # aic = −2·Σ logLik; ls the genuinely NONZERO saturated log-lik with
    # all derivatives zero (left/right rows contribute exactly 0).
    np.testing.assert_allclose(fam.aic(y, mu, 0.0, wt, 0),
                               8.2329365948746229, rtol=1e-14)
    le = fam.ls_extended(y, wt)
    np.testing.assert_allclose(le["ls"], -2.5160365863162668, rtol=1e-14)
    assert float(np.sum(np.abs(le["lsth1"]))) == 0.0
    assert float(np.sum(np.abs(le["LSTH1"]))) == 0.0
    np.testing.assert_allclose(fam.ls(y, wt, 1.0),
                               [-2.5160365863162668, 0.0, 0.0], rtol=1e-14)

    # dppois on probe triples: negative bounds (the Dd shift probes),
    # y1 < 0 (both probs 0 → −Inf), opposite tails, and a far tail.
    from hea.family import _dppois
    y0p = np.array([1.0, -1.0, 0.0, -2.0, 3.0, 40.0])
    y1p = np.array([4.0, 2.0, 9.0, -1.0, 5.0, 60.0])
    mup = np.array([3.0, 5.0, 2.0, 3.0, 4.0, 50.0])
    np.testing.assert_allclose(
        _dppois(y0p, y1p, mup),
        [-0.48432169154524585, -2.0822292679157206, -0.14546723515894291,
         -np.inf, -1.0450897209662637, -0.17224867495290261],
        rtol=1e-14)
    np.testing.assert_allclose(
        _dppois(y0p, y1p, mup, log_p=False),
        [0.61611497105231638, 0.12465201948308118, 0.86461821868837008,
         0.0, 0.35166026666369643, 0.8417698200687822], rtol=1e-14)

    # mustart (log link): pmax(y, min(y>0)) with a zero present keeps the
    # exact zero (mgcv-verified: mustart = 3 2 4 1 0 5).
    np.testing.assert_array_equal(fam.initialize(y, wt), y)


def test_cpois_Dd_matches_fd():
    from hea.family import cpois, _cpois_dev_resids
    y, yat, mu, wt = _cpois_dd_inputs()
    fam = cpois()
    fam.set_censor(yat)
    D = fam.Dd(y, mu, np.zeros(0), wt, level=2)

    def m2ll(mu_):
        # Dd differentiates −2logLik; dev_resids = −2logLik + 2·l_sat and
        # the saturated part is μ-free, so FD in μ sees the same thing.
        return _cpois_dev_resids(y, mu_, yat)

    h = 1e-5
    fd_mu = (m2ll(mu + h) - m2ll(mu - h)) / (2 * h)
    np.testing.assert_allclose(D["Dmu"], fd_mu, rtol=2e-5, atol=1e-6)
    fd_mu2 = (m2ll(mu + h) - 2 * m2ll(mu) + m2ll(mu - h)) / h ** 2
    np.testing.assert_allclose(D["Dmu2"], fd_mu2, rtol=1e-4, atol=1e-4)
    # Dmu3/Dmu4 via FD of Dmu (analytic first derivative — steadier).
    def dmu_at(mu_):
        return fam.Dd(y, mu_, np.zeros(0), wt, level=0)["Dmu"]
    fd_mu3 = ((dmu_at(mu + h) - 2 * dmu_at(mu) + dmu_at(mu - h)) / h ** 2)
    np.testing.assert_allclose(D["Dmu3"], fd_mu3, rtol=1e-4, atol=1e-4)


def test_cpois_construction_and_validation():
    from hea.family import cpois
    fam = cpois()
    assert fam.n_theta == 0
    assert fam.is_extended and fam.scale_known
    assert fam.get_theta().size == 0
    assert fam.get_theta(trans=True).size == 0
    fam.set_theta([1.0])           # mgcv putTheta: silently ignored
    assert fam.get_theta().size == 0
    for lk in ("log", "identity", "sqrt"):
        assert cpois(link=lk).link.name == lk
    with pytest.raises(ValueError, match="not available"):
        cpois(link="logit")
    # validmu (efam.r:359-360): identity → finite; log → μ>0; sqrt → μ≥0.
    assert cpois(link="identity").validmu(np.array([-1.0, 0.0]))
    assert not cpois(link="log").validmu(np.array([0.0, 1.0]))
    assert cpois(link="sqrt").validmu(np.array([0.0, 1.0]))
    assert not cpois(link="sqrt").validmu(np.array([-0.1, 1.0]))
    # identity mustart = y; log/sqrt floor = min(y>0) as a LOGICAL min
    # (1 when all y positive, 0 otherwise — efam.r:500).
    y = np.array([0.5, 2.0])
    np.testing.assert_array_equal(
        cpois(link="identity").initialize(y, np.ones(2)), y)
    np.testing.assert_array_equal(
        cpois(link="log").initialize(y, np.ones(2)), [1.0, 2.0])
    # No censor ⇒ all uncensored: dev reduces to the plain Poisson
    # deviance 2·(dpois(y,y) − dpois(y,μ)).
    fam = cpois()
    y = np.array([2.0, 0.0, 5.0])
    mu = np.array([1.5, 0.8, 4.0])
    dev = fam.dev_resids(y, mu, np.ones(3))
    ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    np.testing.assert_allclose(dev, 2.0 * (ylogy - (y - mu)),
                               rtol=1e-12, atol=1e-12)
    assert repr(fam) == "cpois(link=log)"


# ---------------------------------------------------------------------------
# clog (censored logistic) — mgcv ``clog()`` (efam.r:2192-2612). Slot oracle
# pinned to live mgcv 1.9-4 via hex-float transfer (bit-identical on arm64
# at pin time; randomized 200-row level-2 census incl. mixed weights and
# every log1pexp band was 0 DIFF / 2804 values).
# ---------------------------------------------------------------------------


def _clog_dd_inputs():
    # 6 obs: i0/i2 uncensored, i1 interval [-0.5, 0.8], i3 left (−∞, wt 2),
    # i4 right (+∞), i5 an uncensored row placed in log1pexp's QUIRK band
    # (−(y−μ)/s = 34.5 ∈ (33.3, 37] → mgcv's exp(x) branch — dev blows up
    # to 3.8e15, faithfully).
    y = np.array([1.2, -0.5, 2.0, 0.7, -1.0,
                  float.fromhex("-0x1.6c8f9fb870caap+5")])
    yat = np.array([1.2, 0.8, 2.0, -np.inf, np.inf,
                    float.fromhex("-0x1.6c8f9fb870caap+5")])
    mu = np.array([0.5, 0.1, 1.7, 0.3, -0.4, 1.0])
    wt = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
    return y, yat, mu, wt


def test_clog_components_match_mgcv():
    from hea.family import clog
    y, yat, mu, wt = _clog_dd_inputs()
    fam = clog()
    fam.set_theta([0.3])
    fam.set_censor(yat)

    # rtol 1e-13, not 1e-14: the pins are arm64 R values, and the
    # uncensored deviance 2·(z + 2·log1pexp(−z) − 2log2) cancels
    # ~1.4-magnitude terms down to 0.025 (row 2), amplifying the 1-2 ulp
    # glibc↔Apple libm exp/log scatter ~57× (measured 1.8e-14 rel on
    # linux CI, where hea ≡ linux R holds separately via the live-R
    # oracle gates). Same-platform parity stays bit-level (census).
    np.testing.assert_allclose(
        fam.dev_resids(y, mu, wt),
        [0.13297872286444656, 0.00064770448452167173, 0.024645863841215476,
         1.010811660242168, 0.9907951404171742, 3847863142179033.5],
        rtol=1e-13, atol=0)

    D = fam.Dd(y, mu, np.array([0.3]), wt, level=2)
    exp = {
        "Dmu": [-0.37578439269408498, -0.025906946306693591,
                -0.16396913450701819, 0.83130782189432806,
                -0.57883297064136885, 1.4816364413634331],
        "Dmu2": [0.51350815864591082, 0.51804028675465141,
                 0.54209016682628131, 0.52540422036177381,
                 0.261286207431696, 1.9497706066361287e-15],
        "Dth": [-0.26304907488585949, -0.0012216848009996006,
                -0.049190740352105466, 0.33252312875773121,
                0.34729978238482129, -68.999999999999886],
        "Dmuth": [0.73524010374622251, 0.04886307737374479,
                  0.32659618455490258, -0.62114613374961858,
                  0.42206124618235125, -1.4816364413635239],
        "Dmu3": [0.096484175770105721, 0.0079878924099485715,
                 0.044443027739635224, 0.11368032043932709,
                 -0.04232451164291777, -1.4444255915456878e-15],
        "Dmu2th": [-0.95947739425274758, -0.97691640500401522,
                   -1.0708474253306721, -1.0053363125478167,
                   -0.49717770787764137, 6.3367544715674075e-14],
        "Dmu4": [-0.11371669079977768, -0.16172200627308253,
                 -0.14328723208820304, -0.25145288474698707,
                 -0.061414535592725583, 1.0700567966360104e-15],
        "Dth2": [0.5146680726223557, 0.0021596824071976606,
                 0.097978855366470788, -0.24845845349984741,
                 -0.25323674770941074, 69.000000000004107],
        "Dmuth2": [-1.4068742797231457, -0.086373540635621679,
                   -0.64785041215410422, 0.21901160873049191,
                   -0.12375462145576643, 1.4816364413605729],
        "Dmu2th2": [1.6606168408963822, 1.7263704824328405,
                    2.0888002748077343, 1.8340237790089229,
                    0.89606206198464955, 1.9921781173304531e-12],
        "Dmu3th": [-0.36905421087016155, -0.019960730575645763,
                   -0.17631525284536659, -0.44162211521677608,
                   0.16382225628438868, -4.5499406133689004e-14],
    }
    for k, v in exp.items():
        np.testing.assert_allclose(D[k], v, rtol=1e-13, atol=0,
                                   err_msg=k)
    # EDmu2 zeroes NEGATIVE Dmu2 rows (none here — all Dmu2 ≥ 0); EDmu2th
    # is Dmu2th itself (mgcv aliases them).
    np.testing.assert_array_equal(D["EDmu2"], D["Dmu2"])
    assert D["EDmu2th"] is D["Dmu2th"]

    # aic (SATURATED pieces only — the slot never sees μ; gam.fit4.r:794
    # reports it verbatim), and ls with NONZERO θ-derivatives.
    np.testing.assert_allclose(fam.aic(y, mu, 0.0, wt, 0),
                               15.910610320785761, rtol=1e-14)
    le = fam.ls_extended(y, wt)
    np.testing.assert_allclose(le["ls"], -6.5018787506728541, rtol=1e-14)
    np.testing.assert_allclose(le["lsth1"], [-3.9623749785560696],
                               rtol=1e-14)
    np.testing.assert_allclose(le["lsth2"], [[-0.073257888763456291]],
                               rtol=1e-13)
    np.testing.assert_allclose(
        le["LSTH1"].ravel(),
        [-1.0, -0.96237497855606957, -1.0, 0.0, 0.0, -1.0], rtol=1e-14)


def test_clog_construction_and_validation():
    from hea.family import clog
    # θ intake (efam.r:2213-2221): None/0 → working 0; θ>0 fixed (log θ,
    # n_theta=0); θ<0 an initial value (log|θ|, still estimated).
    assert clog().n_theta == 1
    np.testing.assert_allclose(clog().get_theta(), [0.0])
    assert clog(theta=2.0).n_theta == 0
    np.testing.assert_allclose(clog(theta=2.0).get_theta(), [np.log(2.0)])
    np.testing.assert_allclose(clog(theta=2.0).get_theta(True), [2.0])
    assert clog(theta=-0.5).n_theta == 1
    np.testing.assert_allclose(clog(theta=-0.5).get_theta(), [np.log(0.5)])
    assert clog(theta=0.0).n_theta == 1
    np.testing.assert_allclose(clog(theta=0.0).get_theta(), [0.0])
    for lk in ("identity", "log", "sqrt"):
        assert clog(link=lk).link.name == lk
    with pytest.raises(ValueError, match="not available"):
        clog(link="logit")
    with pytest.raises(ValueError, match="1 param"):
        clog().set_theta([0.1, 0.2])
    # identity validmu = finite; log/sqrt require μ > 0 (2-way, unlike
    # cpois' 3-way — efam.r:2229).
    assert clog(link="identity").validmu(np.array([-1.0, 0.0]))
    assert not clog(link="log").validmu(np.array([0.0, 1.0]))
    assert not clog(link="sqrt").validmu(np.array([0.0, 1.0]))
    # log1pexp quirk band receipt: 33.3 < x ≤ 37 returns exp(x) — mgcv's
    # own typo (first mask is x<=37, efam.r:2237), replicated bit-for-bit.
    from hea.family import _clog_log1pexp
    np.testing.assert_array_equal(
        _clog_log1pexp(np.array([35.0])), np.exp(35.0))
    np.testing.assert_allclose(
        _clog_log1pexp(np.array([-40.0, 0.0, 20.0, 38.0])),
        [np.exp(-40.0), np.log(2.0), 20.0 + np.exp(-20.0), 38.0],
        rtol=1e-15)


# ---------------------------------------------------------------------------
# bcg (censored Box-Cox Gaussian) — mgcv ``bcg()`` (efam.r:1477-2170). Slot
# oracle pinned to live mgcv 1.9-4 via hex transfer (bit-identical on arm64
# at pin time; the randomized 200-row level-2 census — 4 cases, mixed wt,
# y∈{0,1} edges, full (λ,t) derivative matrices — was 0 DIFF / 4602 values
# after the dnorm.c FMA-contraction fix it exposed in nmath).
# ---------------------------------------------------------------------------


def _bcg_dd_inputs():
    # 6 obs: i0 uncensored, i1 interval [1.5, 2.6], i2 LEFT censored
    # (bcg: yat ≤ 0, wt 2), i3 right (+∞), i4 left at y=1 (bc λ-deriv
    # bly = 0 → sign(0) rows), i5 uncensored ZERO (in iu AND il — the
    # later left block overwrites, and ls → +Inf via (λ−1)·log 0).
    y = np.array([3.0, 1.5, 0.9, 2.2, 1.0, 0.0])
    yat = np.array([3.0, 2.6, 0.0, np.inf, 0.0, 0.0])
    mu = np.array([1.1, 0.4, 0.8, 1.9, 0.2, 0.5])
    wt = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    return y, yat, mu, wt


def test_bcg_components_match_mgcv():
    from hea.family import bcg
    y, yat, mu, wt = _bcg_dd_inputs()
    th = np.array([0.4, -0.35])
    fam = bcg()
    fam.set_theta(th)
    fam.set_censor(yat)

    d = fam.dev_resids(y, mu, wt)
    np.testing.assert_allclose(
        d, [0.15744314608370377, 0.29788622636774864, 6.7076037221149409,
            0.17475492756973865, 1.8920743538362341, 22.957041212134278],
        rtol=1e-14, atol=0)
    # mgcv's attr(d,"sign") = sign(bc(y,λ) − μ), stashed for residuals.
    np.testing.assert_array_equal(fam._dev_sign, [1, 1, -1, -1, -1, -1])

    D = fam.Dd(y, mu, th, wt, level=2)
    np.testing.assert_allclose(
        D["Dmu"], [-1.1261466364532253, -1.4824267405144576,
                   8.8633898309156631, -0.47628893494170177,
                   2.8009839610885905, 12.69035263562167],
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        D["Dmu2"], [4.0275054149409524, 3.691333997409322,
                    7.0390203241607736, 1.0466952667846496,
                    2.7946577680930598, 3.8568290720002807],
        rtol=1e-13, atol=0)
    assert D["EDmu2"] is D["Dmu2"] and D["EDmu2th"] is D["Dmu2th"]
    # θ blocks: (n,2) matrices, columns [λ, log σ]; θ² blocks (n,3)
    # [λλ, λt, tt] — mgcv's own packing.
    np.testing.assert_allclose(
        D["Dth"].ravel(order="F"),
        [-1.2814908867536796, -0.79432629876290561, -0.047834925246138082,
         0.18318606765722237, 0.0, -79.314703972635471,
         1.6851137078325924, 1.287417003734145, -8.005158529054766,
         0.46344801352644838, -0.56019679221771812, -38.071057906864986],
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        D["Dmuth"].ravel(order="F"),
        [-3.2749930409424346, -1.0795509097014206, -0.037988965557830073,
         -0.40257074202485166, 0.0, -24.105181700001765,
         2.2522932729064506, 2.7041718209509069, -15.220830342851738,
         -0.5421870500573136, -3.3599155147072026, -24.260839851622222],
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        D["Dmu3"], [0.0, -0.026598900024447758, 1.0876045350424164,
                    -1.590360013833727, 1.0617526205466099,
                    0.08918904228937663], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        D["Dmu2th"].ravel(order="F"),
        [0.0, -0.43185598108667644, -0.0058697047770180366,
         0.61167030287853608, 0.0, -0.55743151430601756,
         -8.0550108298819048, -6.754840935851564, -15.060333753558169,
         -0.5459071991236828, -5.8016660602954424, -7.9812252708657638],
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        D["Dmu4"], [0.0, 0.062798377990909593, -1.4679056433719779,
                    0.75371975840918415, -0.89905663520180124,
                    -0.065243109622599604], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        D["Dth2"].ravel(order="F"),
        [3.3575912323843515, 0.67538398720085868, 0.0035531314471942497,
         0.25359899816612091, 0.0, 547.23090548819118,
         -1.8314673811650801, -1.0091661750898306, 0.082145465281791286,
         0.2085312220130115, 0.0, 151.63024907263957,
         0.62977258433481509, 1.2111589607056146, 13.747015776444481,
         0.52756949169852851, 0.67198310294144048, 72.782519554865985],
        rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        D["Dmuth2"].ravel(order="F"),
        [-2.4838128187057316, -0.22899659577570419, 0.0026906387185074328,
         -0.45230348721151326, 0.0, 124.00985546440461,
         6.5499860818848692, 1.6161446669546113, 0.081279279488592815,
         0.20996203307867312, 0.0, 49.882657942915102,
         -4.5045865458129013, -4.4195013256452702, 28.822890220348171,
         1.0733763942620562, 4.5202487267662903, 48.204515664210703],
        rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        D["Dmu2th2"].ravel(order="F"),
        [0.0, -0.50356411446106319, 0.00036808288773070713,
         0.44127999364757825, 0.0, 0.23859860125821797,
         0.0, 1.5768937572121571, 0.010454051122515517,
         -1.5529373478733481, 0.0, 0.44898623742756172,
         16.11002165976381, 11.207258313087056, 31.870149419673965,
         -2.8370091100428398, 12.204421427510779, 16.177963935588195],
        rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        D["Dmu3th"].ravel(order="F"),
        [0.0, -0.090272421757199517, 0.0079221559762974181,
         -0.28988907473869796, 0.0, 0.40776943518903863,
         0.0, 0.20470879674238507, -1.9370434872535611,
         4.037680839537507, -3.0054465345994732, -0.071837797964690253],
        rtol=1e-12, atol=0)

    np.testing.assert_allclose(fam.aic(y, mu, 0.0, wt, 0),
                               35.895971253414217, rtol=1e-14)
    # ls: the uncensored y=0 row's (λ−1)·log(0) term makes it +Inf —
    # exactly mgcv's value on this frame.
    assert fam.ls(y, wt, 1.0)[0] == np.inf
    le = fam.ls_extended(y, wt)
    assert float(np.sum(np.abs(le["lsth1"]))) == 0.0
    assert le["LSTH1"].shape == (6, 2) and le["lsth2"].shape == (2, 2)


def test_bcg_construction_and_validation():
    from hea.family import bcg
    # θ intake quirks (efam.r:1489-1497, live-R verified 2026-07-06):
    # default (1, 0); fixed θ₂>0 stores LENGTH-3 c(λ, log λ, log σ) with
    # n_theta=0 (the working "log σ" is log λ!); θ₂<0 keeps the RAW pair
    # (the log(-θ₂) line runs after iniTheta was taken — dead).
    assert bcg().n_theta == 2
    np.testing.assert_allclose(bcg().get_theta(), [1.0, 0.0])
    f = bcg(theta=(2.0, 0.5))
    assert f.n_theta == 0
    np.testing.assert_allclose(f.get_theta(),
                               [2.0, np.log(2.0), np.log(0.5)])
    np.testing.assert_allclose(f.get_theta(True),
                               [2.0, 2.0, np.log(0.5)])
    f = bcg(theta=(1.3, -0.5))
    assert f.n_theta == 2
    np.testing.assert_allclose(f.get_theta(), [1.3, -0.5])
    np.testing.assert_allclose(f.get_theta(True), [1.3, np.exp(-0.5)])
    f = bcg(theta=(1.3, 0.0))
    assert f.n_theta == 2
    np.testing.assert_allclose(f.get_theta(), [1.3, 0.0])
    with pytest.raises(ValueError, match="length 2"):
        bcg(theta=[1.0])
    with pytest.raises(ValueError, match="not available"):
        bcg(link="logit")
    # non-negative response required (efam.r:2132).
    with pytest.raises(ValueError, match="non-negative"):
        bcg().initialize(np.array([-0.1, 1.0]), np.ones(2))
    # identity validmu = finite; log/sqrt μ>0 (2-way).
    assert bcg(link="identity").validmu(np.array([-1.0, 0.0]))
    assert not bcg(link="log").validmu(np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# gfam (grouped families) — mgcv ``gfam()`` (gfam.r:3-604). Assembly pinned
# to live mgcv 1.9-4 via hex transfer: a randomized 120-row 5-member
# (binomial, Gamma, nb, tw, gaussian) level-2 census of 10,597 values had
# gfam's own arithmetic bit-identical — every binomial/Gamma/gaussian slot
# 0 DIFF incl. the free-scale chain, the θ² row-major upper-triangle
# packing (filth/filsc) and the raw fix.family.ls → log-scale chain rule.
# Residual last-ulp drift is confined to the nb (≤1e-15) and tw (≤4.3e-13)
# MEMBER internals (receipted standalone: R tw()$dev.resids vs hea tw
# shows the identical drift — numpy-pow vs libm-pow class, pre-existing) —
# hence the 1e-11 pin tolerances below, member-inherited not gfam-born.
# ---------------------------------------------------------------------------


def _gfam_slot_inputs():
    # 8 obs over binomial (rows 0,1,7), tw (2,3,4), gaussian (5,6);
    # θ = (twθ, tw log φ, gauss log σ²) = (0.1, −0.2, 0.3).
    y = np.array([1.0, 0.0, 2.2, 0.4, 1.5, -0.2, 0.7, 1.0])
    fi = np.array([1.0, 1, 2, 2, 2, 3, 3, 1])
    mu = np.array([0.6, 0.3, 1.8, 0.5, 1.2, 0.1, 0.4, 0.7])
    wt = np.ones(8)
    th = np.array([0.1, -0.2, 0.3])
    return y, fi, mu, wt, th


def test_gfam_components_match_mgcv():
    from hea.family import Binomial, Gaussian, gfam, tw
    y, fi, mu, wt, th = _gfam_slot_inputs()
    g = gfam([Binomial(), tw(), Gaussian()])
    assert g.name == "gfam{binomial,Tweedie,gaussian}"
    assert g.link.name == "{logit,log,identity}"
    assert g.n_theta == 3
    g.set_fi(fi)
    pre = g.preinitialize(y)
    assert pre.get("Theta") is None      # no member preinitialize ⇒ no mod
    np.testing.assert_array_equal(pre["y"], y)

    # deviance: member deviances, tw's divided by exp(tw log φ)
    # (gfam.r:103-106).
    np.testing.assert_allclose(
        g.dev_resids(y, mu, wt, th),
        [1.0216512475319814, 0.7133498878774648, 0.07185266607650018,
         0.03923448388024454, 0.07409608592118926, 0.06667363986135462,
         0.06667363986135456, 0.7133498878774648],
        rtol=1e-11, atol=0)

    D = g.Dd(y, mu, th, wt, level=2)
    # θ matrices are (8, 3): off-member columns EXACT zeros; the tw
    # log-scale column is −(scaled member table) (gfam.r:172-178);
    # pins column-major raveled like R writes them.
    exp = {
        "Dmu": [-3.3333333333333335, 2.857142857142857, -0.39883306491867987,
                0.7027534406074493, -0.5550089060549714, 0.44449093240903076,
                -0.4444909324090306, -2.857142857142857],
        "Dmu2": [5.555555555555556, 4.081632653061225, 1.334867591608585,
                 4.884867832761118, 2.5551128178240297, 1.4816364413634358,
                 1.4816364413634358, 4.081632653061225],
        "Dmu3": [-18.518518518518526, 11.661807580174923, -2.1626641823899977,
                 -32.03509534242298, -6.183860859059008, -0.0, 0.0,
                 -11.661807580174932],
        "Dmu4": [92.59259259259262, 49.97917534360683, 4.480649480682058,
                 248.2897785817015, 19.18963656151585, 0.0, 0.0,
                 49.9791753436068],
        "Dth": [0.0, 0.0, -0.011494749284079071, 0.007359766645563358,
                -0.004646648785357787, 0.0, 0.0, 0.0,
                0.0, 0.0, -0.07185266607650018, -0.03923448388024454,
                -0.07409608592118926, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, -0.06667363986135462,
                -0.06667363986135456, 0.0],
        "Dmuth": [0.0, 0.0, 0.057291696845296086, 0.11904447438855675,
                  0.02472969573097741, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.39883306491867987, -0.7027534406074493,
                  0.5550089060549714, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, -0.44449093240903076,
                  0.4444909324090306, 0.0],
        "Dmu2th": [0.0, 0.0, -0.13760135382703625, 0.4839932519102857,
                   -0.0008174123947630714, 0.0, 0.0, 0.0,
                   0.0, 0.0, -1.334867591608585, -4.884867832761118,
                   -2.5551128178240297, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, -1.4816364413634358,
                   -1.4816364413634358, 0.0],
    }
    for k, v in exp.items():
        got = np.asarray(D[k], dtype=float)
        if got.ndim == 2:
            got = got.ravel(order="F")
        np.testing.assert_allclose(got, v, rtol=1e-11, atol=0, err_msg=k)
    # Dth2 (8, 6) row-major-upper packed pairs (θθ, θρ_tw, ρρ_tw, …):
    # in-family blocks via filth, the tw scale pairs = −Dth columns
    # (gfam.r:189-196).
    np.testing.assert_allclose(
        np.asarray(D["Dth2"]).ravel(order="F"),
        [0.0, 0.0, 0.002422743278417013, 0.0010193790896375246,
         0.0005357661343759946, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.011494749284079071, -0.007359766645563358,
         0.004646648785357787, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.07185266607650018, 0.03923448388024454,
         0.07409608592118926, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.06667363986135462,
         0.06667363986135456, 0.0],
        rtol=1e-11, atol=0)

    # ls: member saturated lls; exponential free-scale entries via the
    # raw fix.family.ls values chain-ruled in gfam (gfam.r:341-351) —
    # gaussian lsth1 = −nobs/(2φ)·φ = −1 here, LSTH1 rows −0.5.
    le = g.ls_extended(y, wt, theta=th, scale=1.0)
    np.testing.assert_allclose(le["ls"], -5.078841267754759, rtol=1e-12)
    np.testing.assert_allclose(
        le["lsth1"],
        [0.0016670861446814733, -1.8222480844147504, -1.0],
        rtol=1e-11, atol=0)
    np.testing.assert_allclose(
        np.asarray(le["lsth2"]).ravel(order="F"),
        [0.0009962090903217202, 0.0938659100200604, 0.0,
         0.0938659100200604, -0.4354457278967834, 0.0,
         0.0, 0.0, 0.0],
        rtol=5e-11, atol=0)
    np.testing.assert_allclose(
        np.asarray(le["LSTH1"]).ravel(order="F"),
        [0.0, 0.0, -0.10764741499602315, 0.1655207146761607,
         -0.05620621353545607, 0.0, 0.0, 0.0,
         0.0, 0.0, -0.5619928544092048, -0.6828579090037183,
         -0.5773973210018273, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, 0.0],
        rtol=1e-11, atol=0)

    # aic recomputes member deviances internally (the dev argument is
    # dead, gfam.r:359) — tw's power comes from the θ ARGUMENT
    # (efam.r:3213, the tw.aic override this exposed).
    np.testing.assert_allclose(g.aic(y, mu, 0.0, wt, None, theta=th),
                               7.319208537355566, rtol=1e-11)

    # per-row link dispatch (logit rows 0,1,7 now stats-C exact).
    np.testing.assert_allclose(
        g.link.link(mu),
        [0.4054651081081642, -0.8472978603872036, 0.5877866649021191,
         -0.6931471805599453, 0.1823215567939546, 0.1, 0.4,
         0.8472978603872034],
        rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        g.link.g2g(mu),
        [0.19999999999999996, -0.3999999999999999, -1.0, -1.0, -1.0,
         0.0, 0.0, 0.3999999999999999],
        rtol=1e-14, atol=0)

    # member mustarts on their subsets — binomial (w·y+.5)/(w+1),
    # tw y+(y==0)·0.1, gaussian y (stock forms; fix.family's gam patches
    # never match "gfam{…}", mgcv.r:1916).
    np.testing.assert_allclose(
        g.initialize(y, wt),
        [0.75, 0.25, 2.2, 0.4, 1.5, -0.2, 0.7, 0.75],
        rtol=0, atol=0)


def test_gfam_construction_and_validation():
    from hea.family import (
        Binomial, Gamma, Gaussian, Poisson, Tweedie, gaulss, gfam, nb,
        negbin, tw,
    )
    # n_theta accounting (gfam.r:36-49): exponential poisson/binomial
    # scale fixed 1 (no slot); other exponentials a free log-scale slot;
    # extended members their n_theta, +1 for tw (scale = -1, efam.r:3263).
    assert gfam([Poisson(), Gaussian()]).n_theta == 1
    assert gfam([Binomial(), tw(), Gaussian()]).n_theta == 3
    assert gfam([Gamma(), nb()]).n_theta == 2
    np.testing.assert_array_equal(
        gfam([Binomial(), tw(), Gaussian()]).get_theta(), np.zeros(3))
    # R-style string / constructor-class intake (gfam.r:23-24).
    g = gfam(["poisson", Gaussian])
    assert g.name == "gfam{poisson,gaussian}"
    with pytest.raises(ValueError, match="family not recognized"):
        gfam(["nope"])
    with pytest.raises(ValueError, match="family not recognized"):
        gfam([])
    # fix.family.ls has no Tweedie/negbin rows — a fixed-p Tweedie()
    # or negbin() member dies exactly like mgcv (gam.fit3.r:2546).
    with pytest.raises(ValueError, match="family not recognised"):
        gfam([Tweedie(p=1.5), Gaussian()])
    with pytest.raises(ValueError, match="family not recognised"):
        gfam([negbin(theta=2.0), Gaussian()])
    # general (multi-LP) members: mgcv's stop, message verbatim
    # (gfam.r:55).
    with pytest.raises(NotImplementedError, match="general familes"):
        gfam([gaulss(), Gaussian()])
    # fixed-θ extended members leave mgcv's .Theta walk inconsistent
    # (getTheta() entries with no n.theta slots, gfam.r:36-50) — refused.
    with pytest.raises(NotImplementedError, match="fixed-theta"):
        gfam([nb(theta=2.0), Gaussian()])
    # the family index must cover 1..nf exactly (gfam.r:389-390).
    g = gfam([Poisson(), Gaussian()])
    g.set_fi(np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="does not match family list"):
        g.preinitialize(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="expects 1 params"):
        g.set_theta([0.1, 0.2])
    # no fi set → any consumer refuses (the two-column response is the
    # only intake).
    g2 = gfam([Poisson(), Gaussian()])
    with pytest.raises(ValueError, match="two-column response"):
        g2.initialize(np.array([1.0, 2.0]), np.ones(2))


# ---------------------------------------------------------------------------
# SoftplusLink (Thread A) — the genuine softplus *mean* link μ = log(1+e^η)
# for Poisson RF/point-process GLMs (comp-neuro soft-rectifier; Paninski 2004).
# NOT an mgcv built-in, so the link math is checked against closed forms +
# finite differences; the fit is R-pinned via a custom-link glm (exact). Must
# stay distinct from BoundedLogLink (mgcv's bounded "log" gamlss-scale link).
# ---------------------------------------------------------------------------


def test_softplus_link_math():
    from hea.family import SoftplusLink, BoundedLogLink, _resolve_link
    L = SoftplusLink()
    assert L.name == "softplus"
    assert isinstance(_resolve_link("softplus", "log"), SoftplusLink)
    # distinct from the bounded-log gamlss link (which displays as "log")
    assert BoundedLogLink().name == "log"
    assert not isinstance(BoundedLogLink(), SoftplusLink)

    # round-trips both ways
    eta = np.linspace(-9, 9, 19)
    np.testing.assert_allclose(L.link(L.linkinv(eta)), eta, atol=1e-9)
    mu = np.linspace(0.05, 7, 25)
    np.testing.assert_allclose(L.linkinv(L.link(mu)), mu, atol=1e-9)

    # μ = softplus(η), dμ/dη = σ(η) (vs finite diff of linkinv)
    np.testing.assert_allclose(L.linkinv(eta), np.logaddexp(0.0, eta), atol=1e-12)
    h = 1e-6
    fd = (L.linkinv(eta + h) - L.linkinv(eta - h)) / (2 * h)
    np.testing.assert_allclose(L.mu_eta(eta), fd, atol=1e-7)

    # link derivatives vs closed forms (u = e^{-μ}, s = 1-u)
    u = np.exp(-mu)
    s = 1.0 - u
    np.testing.assert_allclose(L.d2link(mu), -u / s ** 2, rtol=1e-12)
    np.testing.assert_allclose(L.d3link(mu), u * (1 + u) / s ** 3, rtol=1e-12)
    np.testing.assert_allclose(
        L.d4link(mu), -u * (1 + 4 * u + u * u) / s ** 4, rtol=1e-12)
    # extended-family ratios g_kg = d^k link / g'^k, g'(μ)=1/s
    np.testing.assert_allclose(L.g2g(mu), -u, rtol=1e-12)
    np.testing.assert_allclose(L.g3g(mu), u * (1 + u), rtol=1e-12)
    np.testing.assert_allclose(L.g4g(mu), -u * (1 + 4 * u + u * u), rtol=1e-12)
    # the ratio identity itself: d2link == g2g * g'^2
    gp = 1.0 / s
    np.testing.assert_allclose(L.d2link(mu), L.g2g(mu) * gp ** 2, rtol=1e-12)
    np.testing.assert_allclose(L.d3link(mu), L.g3g(mu) * gp ** 3, rtol=1e-12)
    np.testing.assert_allclose(L.d4link(mu), L.g4g(mu) * gp ** 4, rtol=1e-12)

    # numerically safe at extreme η (no overflow; μ>0, μ_η finite)
    big = np.array([-700.0, -50.0, 0.0, 50.0, 700.0])
    assert np.all(np.isfinite(L.linkinv(big))) and np.all(L.linkinv(big) > 0)
    assert np.all(np.isfinite(L.mu_eta(big)))


def test_softplus_poisson_glm_matches_mgcv():
    """Poisson GLM with the softplus link, R-pinned EXACT against a custom
    link-glm fit (``glm(y~x+z, poisson(link=softplus))``), mgcv-independent —
    R 4.x glm IRLS. The reusable R link lives in
    ``tests/r_oracle/softplus_link.R``. Data: softplus is the true link."""
    from hea.R.rng import RGenerator
    from hea.family import SoftplusLink
    from hea.models import glm
    g = RGenerator(7)
    n = 300
    x = g.uniform(-1.0, 2.0, n)
    z = g.uniform(-1.0, 1.0, n)
    mu = np.log1p(np.exp(0.6 + 1.1 * x - 0.5 * z))
    y = g.poisson(mu).astype(float)
    d = pl.DataFrame({"y": y, "x": x, "z": z})
    m = glm("y ~ x + z", d, family=Poisson(link=SoftplusLink()))
    np.testing.assert_allclose(
        np.asarray(m.bhat.to_numpy()).ravel(),
        [0.595080256, 1.21651587, -0.704083015], atol=1e-6)
    assert float(m.deviance) == pytest.approx(328.072958, abs=1e-4)
    # string form resolves to the same link
    m2 = glm("y ~ x + z", d, family=Poisson(link="softplus"))
    np.testing.assert_allclose(np.asarray(m2.bhat.to_numpy()).ravel(),
                               np.asarray(m.bhat.to_numpy()).ravel(), atol=0)


# ===========================================================================
# negbin — fixed-θ negative binomial (mgcv negbin(), gam.fit3.r:2564-2642)
# ===========================================================================


def test_negbin_components_match_mgcv():
    """Slot-level bit-pins vs mgcv 1.9-4 ``negbin(2)`` at fixed (y, μ, w):
    dev.resids / aic / ls / variance / initialize, plus the constructor
    surface (famname θ-format, natural-scale getTheta, link intake,
    validation errors)."""
    from hea.family import negbin

    f = negbin(2)
    y = np.array([0.0, 1.0, 3.0, 7.0])
    mu = np.array([1.5, 2.0, 2.5, 6.0])
    w = np.ones(4)
    # mgcv: fam$dev.resids(y, mu, w) — bit-identical.
    np.testing.assert_array_equal(
        f.dev_resids(y, mu, w),
        [2.2384631517416911, 0.33979807359079484,
         0.040324184185464018, 0.038014875766715139])
    # mgcv: fam$aic(y, 1, mu, w, 0) (dev unused — Θ-form direct).
    assert f.aic(y, mu, None, w, None) == 14.422747381464536
    # mgcv: fam$ls(y, w, 4, 1) = c(-sum(term·w), 0, 0).
    np.testing.assert_array_equal(
        f.ls(y, w, 1.0), [-5.8830735480899392, 0.0, 0.0])
    np.testing.assert_array_equal(f.variance(mu), [2.625, 4.0, 5.625, 24.0])
    np.testing.assert_array_equal(f.dvar(mu), [2.5, 3.0, 3.5, 7.0])
    np.testing.assert_array_equal(f.d2var(mu), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(f.d3var(mu), [0.0, 0.0, 0.0, 0.0])
    # initialize: mustart <- y + (y == 0)/6.
    np.testing.assert_allclose(
        f.initialize(y, w), [1.0 / 6.0, 1.0, 3.0, 7.0], rtol=0)
    with pytest.raises(ValueError, match="negative values not allowed"):
        f.initialize(np.array([-1.0, 2.0]), np.ones(2))
    # qf = qnbinom(p, size=Θ, mu) — R oracle.
    np.testing.assert_array_equal(
        f.qf(np.array([0.1, 0.5, 0.9]), np.full(3, 7.3), None, None),
        [1.0, 6.0, 15.0])
    # famname: paste("Negative Binomial(", format(round(theta, 3)), ")").
    assert f.name == "Negative Binomial(2)"
    assert negbin(2.3456).name == "Negative Binomial(2.346)"
    assert negbin(3.7).name == "Negative Binomial(3.7)"
    # getTheta: the θ vector on the NATURAL scale (mgcv negbin$getTheta()).
    np.testing.assert_array_equal(negbin([2, 9]).get_theta(), [2.0, 9.0])
    # canonical="" (gam.fit3.r:2641) → never the Fisher shortcut.
    assert not f.is_canonical
    assert f.scale_known and not f.is_extended and f.n_theta == 0
    # link intake: mgcv falls through to make.link(link) for ANY character
    # link (gam.fit3.r:2577-2579) — "inverse" is accepted despite the
    # nominal okLinks (verified live); unknown names error like make.link.
    assert negbin(2, link="inverse").link.name == "inverse"
    assert negbin(2, link="sqrt").link.name == "sqrt"
    with pytest.raises(ValueError, match="banana"):
        negbin(2, link="banana")
    # theta = stop("'theta' must be specified") — the lazy default fires
    # on first access; hea validates eagerly with the same message.
    with pytest.raises(ValueError, match="'theta' must be specified"):
        negbin()
    with pytest.raises(ValueError, match="positive and finite"):
        negbin(-1)


def test_qnbinom_pnbinom_mu_match_r():
    """``qnbinom_mu``/``pnbinom_mu`` (nmath qnbinom_mu.c / pnbinom.c)
    bit-exact vs R across parametrizations, tails and the log scale, and
    the rust kernels 0-ulp against the Python port."""
    from hea.R import nmath as nm

    ps = [1e-4, .001, .01, .1, .3, .5, .77, .9, .99, .9999]
    grids = {
        (2.5, 7.3): [0, 0, 0, 2, 4, 6, 10, 14, 24, 42],
        (2.0, 0.03): [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        (37.5, 4200.0): [2103, 2386, 2764, 3345, 3813, 4163,
                         4690, 5103, 5966, 7254],
        (0.07, 123456.7): [0, 0, 0, 0, 0, 52, 25362, 266391,
                           2326395, 8729299],
    }
    for (size, mu), expect in grids.items():
        got = [nm.qnbinom_mu(p, size, mu) for p in ps]
        np.testing.assert_array_equal(got, expect, err_msg=f"{size},{mu}")
    # upper tail + log scale (size=2.5, mu=7.3).
    np.testing.assert_array_equal(
        [nm.qnbinom_mu(p, 2.5, 7.3, False) for p in ps],
        [42, 34, 24, 14, 9, 6, 3, 2, 0, 0])
    np.testing.assert_array_equal(
        [nm.qnbinom_mu(np.log(p), 2.5, 7.3, True, True) for p in ps],
        [0, 0, 0, 2, 4, 6, 10, 14, 24, 42])
    # boundaries: R_Q_P01_boundaries + the size/mu special cases.
    assert nm.qnbinom_mu(0.0, 2.5, 7.3) == 0.0
    assert nm.qnbinom_mu(1.0, 2.5, 7.3) == np.inf
    assert nm.qnbinom_mu(0.5, 0.0, 3.0) == 0.0
    assert nm.qnbinom_mu(0.5, 2.0, 0.0) == 0.0
    assert nm.qnbinom_mu(0.5, np.inf, 3.0) == 3.0     # Poisson limit
    # pnbinom_mu: bratio on BOTH tail ratios (pnbinom.c:83).
    np.testing.assert_array_equal(
        [nm.pnbinom_mu(x, 2.5, 7.3) for x in [0, 1, 3, 7, 20, -1, 1e18]],
        [0.032868874445290082, 0.094078768182692535, 0.26302499000982504,
         0.59957811374520131, 0.9746158998054657, 0.0, 1.0])
    np.testing.assert_array_equal(
        [nm.pnbinom_mu(x, 2.5, 7.3, True, True) for x in [0, 1, 3, 7, 20]],
        [-3.4152291345059278, -2.3636228882137487, -1.3355062322696321,
         -0.51152901484550251, -0.025711834520021577])
    np.testing.assert_array_equal(
        [nm.pnbinom_mu(x, 2.5, 7.3, False) for x in [0, 1, 3, 7, 20]],
        [0.96713112555470993, 0.90592123181730744, 0.73697500999017496,
         0.40042188625479869, 0.025384100194534378])
    np.testing.assert_array_equal(
        [nm.pnbinom_mu(x, 0.0, 0.0) for x in [-1, 0, 3]], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        [nm.pnbinom_mu(x, np.inf, 7.3) for x in [0, 3, 7]],
        [0.00067553877519384439, 0.067406047117412868, 0.55410661183907739])
    # x + 1e-7 left-fuzz (pnbinom.c floor(x + 1e-7)).
    assert nm.pnbinom_mu(2.9999999, 2.5, 7.3) == 0.26302499000982504
    # rust kernels 0-ulp vs the Python port (skips cleanly if the
    # extension is absent — _disp then runs the Python scalars anyway).
    rs_q = nm.rs_fn("qnbinom_mu")
    rs_p = nm.rs_fn("pnbinom_mu")
    if rs_q is not None:
        for (size, mu) in grids:
            for lt in (True, False):
                for lg in (True, False):
                    p_in = np.log(np.asarray(ps)) if lg else np.asarray(ps)
                    got_rs = rs_q(p_in, np.full(10, size), np.full(10, mu),
                                  lt, lg)
                    got_py = [nm.qnbinom_mu(p, size, mu, lt, lg)
                              for p in p_in]
                    np.testing.assert_array_equal(got_rs, got_py)
        xs = np.array([0.0, 1, 3, 7, 20, 1e3])
        for (size, mu) in grids:
            for lt in (True, False):
                for lg in (True, False):
                    got_rs = rs_p(xs, np.full(6, size), np.full(6, mu),
                                  lt, lg)
                    got_py = [nm.pnbinom_mu(x, size, mu, lt, lg) for x in xs]
                    np.testing.assert_array_equal(got_rs, got_py)


# ===========================================================================
# tw/nb θ-chain + R_pow/lgamma/digamma parity census pins (audit-2 B14
# follow-up closure). Values: live mgcv 1.9-4 hex-float census on the
# default_rng(42) frame below (bit-exact on arm64; rel tolerances carry
# ~100x headroom for glibc↔Apple libm exp/log last-ulp scatter, the
# 2026-07-06(f) CI lesson). The fixes these pin: p(θ) as mgcv's literal
# branch expressions (not expit algebra), R_pow sequential ^2/^3
# scalars + mgcv's Dth2 parenthesization in tw$Dd, _rpow_int for nb's
# mu^3/mu^4, nmath lgammafn/dpsifn (not scipy) + _rsum in nb$ls/aic,
# and tw$ls as the mechanical colSums(w·ldTweedie(y,y,…)) port.
# ===========================================================================


def _census_frame_42():
    rng = np.random.default_rng(42)
    n = 200
    mu = np.abs(rng.normal(2.0, 1.5, n)) + 0.05
    y = np.abs(mu * np.exp(rng.normal(0.0, 0.5, n)))
    y[:10] = 0.0
    wt = np.where(rng.uniform(size=n) < 0.3,
                  rng.integers(1, 5, n).astype(float), 1.0)
    ynb = np.floor(y * 3).astype(float)
    return y, mu, wt, ynb


def test_tw_nb_theta_chain_census_matches_mgcv():
    from hea.family import nb, tw

    y, mu, wt, ynb = _census_frame_42()

    fam = tw(a=1.01, b=1.99)
    dev = fam.dev_resids(y, mu, wt, theta=0.9)
    assert float(dev[0]) == pytest.approx(35.71796731366923, rel=1e-13)
    assert float(dev[10]) == pytest.approx(1.4661145459181908, rel=1e-13)
    fam.set_theta(0.9)
    assert float(fam.variance(mu)[0]) == pytest.approx(
        4.8003315958745265, rel=1e-13)
    dd = fam.Dd(y, mu, 0.9, wt, level=2)
    assert float(dd["Dth"][10]) == pytest.approx(
        -0.40292464474908973, rel=1e-13)
    assert float(dd["Dth2"][0]) == pytest.approx(
        18.27147175320773, rel=1e-13)
    assert float(dd["Dmuth2"][1]) == pytest.approx(
        -0.13234424141561646, rel=1e-13)
    assert float(dd["Dmu2th2"][1]) == pytest.approx(
        0.3740245563340534, rel=1e-13)
    dd_neg = fam.Dd(y, mu, -0.7, wt, level=2)
    assert float(dd_neg["Dth2"][10]) == pytest.approx(
        -0.026858465349980126, rel=1e-13)

    fam_nb = nb()
    th = float(np.log(1.7))
    ddn = fam_nb.Dd(ynb, mu, th, wt, level=2)
    assert float(ddn["Dmu3"][5]) == pytest.approx(
        1.172357101442247, rel=1e-13)
    assert float(ddn["Dmu4"][3]) == pytest.approx(
        -0.028757174064230596, rel=1e-13)
    assert float(ddn["Dmu3th"][5]) == pytest.approx(
        -2.1553653475720496, rel=1e-13)
    ls3 = fam_nb.ls_extended(ynb, wt, theta=th, scale=1.0)
    assert float(ls3["LSTH1"][11, 0]) == pytest.approx(
        0.3940667463254954, rel=1e-13)
    assert float(ls3["ls"]) == pytest.approx(-603.812776094261, rel=1e-12)
    assert float(np.asarray(ls3["lsth1"]).ravel()[0]) == pytest.approx(
        96.4119849078045, rel=1e-12)
    assert float(np.asarray(ls3["lsth2"]).ravel()[0]) == pytest.approx(
        -32.24760362160137, rel=1e-12)
