"""
hea ``harmonic()`` formula term — R-parity + behaviour tests.

``harmonic(x, K, period=p)`` is a hea-native periodic basis (the trig sibling
of ``poly``/``bs``/``ns``): ``K`` raw cos/sin harmonic pairs ``cos(2πj x/p)``,
``sin(2πj x/p)``, cos-first interleaved with recoverable ``cos{j}``/``sin{j}``
column suffixes. Its name is aligned with ``TSA::harmonic`` but it is NOT a
port — it always emits ``2K`` columns (no integer-``ts`` Nyquist drop) and
takes an explicit ``period`` (hea has no ``ts`` frequency to infer one from).
See ``.claude/plans/harmonic-formula-term.md``.

R references generated with R 4.6.0 (base ``lm``) on the deterministic data
below — no RNG: an exact two-harmonic signal at period 12 plus a ``0.3*x``
linear trend the ``K=2`` basis cannot represent, so the least-squares fit is
non-trivial (a genuine cross-check, not a tautology).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hea.formula import _harmonic_basis
from hea.models import gam, lm
from hea.R import fitted

# --- shared deterministic dataset (mirrored exactly in the R reference) ------
_X = np.arange(36, dtype=float)
_P = 12.0
_Y = (5.0 + 3.0 * np.cos(2 * np.pi * _X / _P)
      - 2.0 * np.sin(2 * np.pi * _X / _P)
      + 1.5 * np.cos(2 * np.pi * 2 * _X / _P)
      + 0.5 * np.sin(2 * np.pi * 2 * _X / _P)
      + 0.3 * _X)
_DF = pl.DataFrame({"x": _X, "y": _Y})

# lm(y ~ cos1 + sin1 + cos2 + sin2), R 4.6.0.
# coef order: (Intercept), cos1, sin1, cos2, sin2
_R_COEF = np.array([10.25, 2.7, -3.11961524227, 1.2, -0.01961524227])
_R_SIGMA = 3.207784437
_R_PRED_X = [36.0, 42.5, 50.0]            # fresh x, incl. non-integer
_R_PRED = [14.15, 9.478838971, 8.281346652]


def test_harmonic_basis_helper_values_and_order():
    # cos-first interleaved, exact trig values, 2K columns.
    x = np.array([0.0, 1.0, 3.0, 7.5])
    cols, suffixes = _harmonic_basis(x, 3, 12.0)
    assert suffixes == ["cos1", "sin1", "cos2", "sin2", "cos3", "sin3"]
    assert cols.shape == (4, 6)
    for j in (1, 2, 3):
        w = 2 * np.pi * j * x / 12.0
        np.testing.assert_allclose(cols[:, 2 * (j - 1)], np.cos(w))
        np.testing.assert_allclose(cols[:, 2 * (j - 1) + 1], np.sin(w))


def test_harmonic_lm_coef_parity_R():
    m = lm("y ~ harmonic(x, 2, period=12)", _DF)
    np.testing.assert_allclose(np.asarray(m.coef), _R_COEF, atol=1e-8)
    np.testing.assert_allclose(m.sigma, _R_SIGMA, atol=1e-6)


def test_harmonic_coef_labels_carry_suffixes():
    # downstream (pycircstat2) reads (harmonic index, cos|sin) from the suffix,
    # so no coefficient-name regex is needed.
    m = lm("y ~ harmonic(x, 2, period=12)", _DF)
    names = list(m.bhat.columns)
    assert names[0] == "(Intercept)"
    assert [n.rsplit(")", 1)[-1] for n in names[1:]] == \
        ["cos1", "sin1", "cos2", "sin2"]


def test_harmonic_predict_is_stateless_R():
    # period is a formula literal -> the basis is a pure function of
    # (x, K, period); re-evaluating on fresh x reproduces R's predictions.
    m = lm("y ~ harmonic(x, 2, period=12)", _DF)
    pred = m.predict(newdata=pl.DataFrame({"x": _R_PRED_X}))
    np.testing.assert_allclose(np.asarray(pred["fit"]), _R_PRED, atol=1e-6)


def test_harmonic_equals_explicit_cos_sin_terms():
    # spans the same column space as the hand-written cos/sin formula ->
    # identical fitted values (the baseline pycircstat2 uses today).
    m_h = lm("y ~ harmonic(x, 2, period=12)", _DF)
    m_e = lm(
        "y ~ cos(2*pi*x/12) + sin(2*pi*x/12)"
        " + cos(2*pi*2*x/12) + sin(2*pi*2*x/12)",
        _DF,
    )
    np.testing.assert_allclose(
        np.asarray(fitted(m_h)), np.asarray(fitted(m_e)), atol=1e-9)


def test_harmonic_positional_and_keyword_forms_agree():
    a = lm("y ~ harmonic(x, 2, 12)", _DF)           # all positional
    b = lm("y ~ harmonic(x, 2, period=12)", _DF)    # period keyword
    c = lm("y ~ harmonic(x, K=2, period=12)", _DF)  # all keyword
    np.testing.assert_allclose(np.asarray(a.coef), np.asarray(b.coef))
    np.testing.assert_allclose(np.asarray(a.coef), np.asarray(c.coef))


def test_harmonic_period_2pi_angular():
    # the pycircstat2 path: angular predictor with period=2*pi (parses `pi`),
    # no `ts` involved. Exact 2-harmonic signal -> recovered coefficients.
    th = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
    yy = 1.0 + 0.8 * np.cos(th) - 0.5 * np.sin(th) + 0.3 * np.cos(2 * th)
    m = lm("y ~ harmonic(theta, 2, period=2*pi)",
           pl.DataFrame({"theta": th, "y": yy}))
    np.testing.assert_allclose(
        np.asarray(m.coef), [1.0, 0.8, -0.5, 0.3, 0.0], atol=1e-9)


def test_harmonic_composes_with_smooth():
    x = np.arange(60, dtype=float)
    d = pl.DataFrame({
        "x": x,
        "z": np.sin(x / 7.0),
        "y": np.cos(2 * np.pi * x / 12) + 0.5 * x / 60,
    })
    m = gam("y ~ harmonic(x, 2, period=12) + s(z)", d, method="REML")
    assert np.isfinite(m.edf_total)
    assert any("cos1" in c for c in m.X.columns)


@pytest.mark.parametrize("formula, match", [
    ("y ~ harmonic(x, 2)", "period"),                          # missing period
    ("y ~ harmonic(x, 2, period=0)", "must be positive"),      # period <= 0
    ("y ~ harmonic(x, 2, period=-3)", "must be positive"),
    ("y ~ harmonic(x, 0, period=12)", "positive integer"),     # K < 1
    ("y ~ harmonic(x, 2.5, period=12)", "positive integer"),   # non-integer K
])
def test_harmonic_errors(formula, match):
    with pytest.raises(ValueError, match=match):
        lm(formula, _DF)
