"""Tests for ``hea._loess`` — local polynomial regression port.

R-oracle parity tests against ``stats::loess`` come later (X.1). These are
hand-checked sanity tests covering the algorithmic core: fit recovery on
linear/constant data, outlier suppression by the M-step, prediction
consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from hea._loess import loess


def test_loess_recovers_linear_data():
    """Pure linear data → near-perfect linear fit, low residual sd."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 2 * x + 1 + 0.3 * rng.standard_normal(80)

    fit = loess(x, y, span=0.75, degree=1)
    # Residual sd should be close to the noise sd (0.3); inflated by the
    # df approximation but still in the right order of magnitude.
    assert 0.2 < fit.sigma < 0.6
    # Prediction at x=5 should be ≈11.
    assert abs(float(fit.predict([5.0])[0]) - 11.0) < 0.5


def test_loess_constant_data_gives_constant_fit():
    x = np.linspace(0, 10, 30)
    y = np.full(30, 7.0)
    fit = loess(x, y, span=0.75, degree=1)
    assert np.allclose(fit.fitted, 7.0)
    assert fit.sigma == pytest.approx(0.0, abs=1e-10)


def test_loess_predict_at_training_x_matches_fitted():
    """Predicting at the training x's must reproduce ``fitted`` exactly."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 40)
    y = np.sin(x) + 0.1 * rng.standard_normal(40)

    fit = loess(x, y, span=0.5, degree=2)
    pred = fit.predict(x)
    np.testing.assert_allclose(pred, fit.fitted, atol=1e-10)


def test_loess_predict_se_positive_in_data_range():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 50)
    y = x + rng.standard_normal(50)

    fit = loess(x, y, span=0.5, degree=1)
    grid = np.linspace(1, 9, 9)
    pred, se = fit.predict(grid, se=True)
    assert (se > 0).all()
    assert (se < fit.sigma).any(), "interior SE should be smaller than residual sd"


def test_loess_symmetric_family_suppresses_outlier():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + 0.3 * rng.standard_normal(50)
    y_dirty = y.copy()
    y_dirty[10] = 100  # gross outlier

    fit_g = loess(x, y_dirty, span=0.4, degree=1, family="gaussian")
    fit_s = loess(x, y_dirty, span=0.4, degree=1, family="symmetric", iterations=4)

    # Gaussian fit gets pulled toward the outlier; symmetric should not.
    pred_g = float(fit_g.predict([x[10]])[0])
    pred_s = float(fit_s.predict([x[10]])[0])
    truth = 2 * x[10] + 1
    assert abs(pred_s - truth) < abs(pred_g - truth)
    assert abs(pred_s - truth) < 5  # close to truth


def test_loess_degree_2_fits_curvature_better_than_degree_1():
    """On strongly curved data, quadratic local fit beats linear at the same span."""
    x = np.linspace(0, 10, 50)
    y = (x - 5) ** 2  # parabola

    fit1 = loess(x, y, span=0.3, degree=1)
    fit2 = loess(x, y, span=0.3, degree=2)
    assert fit2.sigma < fit1.sigma


def test_loess_span_out_of_range_errors():
    x = np.arange(10).astype(float)
    y = x.copy()
    with pytest.raises(ValueError, match="span"):
        loess(x, y, span=1.5)
    with pytest.raises(ValueError, match="span"):
        loess(x, y, span=0)


def test_loess_degree_out_of_range_errors():
    x = np.arange(10).astype(float)
    y = x.copy()
    with pytest.raises(ValueError, match="degree"):
        loess(x, y, degree=3)


def test_loess_length_mismatch_errors():
    with pytest.raises(ValueError, match="length"):
        loess(np.arange(10), np.arange(8))
