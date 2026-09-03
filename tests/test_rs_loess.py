"""Rust LOESS kernel parity gate.

``hea._rs.loess_eval`` must match the pure-Python ``_loess_local_fit`` loop it
replaces (the reference, itself tolerance-tested vs ``stats::loess``). The Rust
kernel forms the weighted normal equations in a scaled basis where the Python
path uses ``lstsq``; they agree to ~1e-13 (the intercept and var00 are invariant
under the basis scaling). Covers degree 1/2, several spans, robustness weights,
and BOTH the serial (n < 64) and parallel (n >= 64) branches.

Platform-independent (Rust vs pure-Python hea, not vs R), so it runs in CI —
unlike the macOS-only Rust==R d/p/q gate.
"""

import numpy as np
import pytest

rs = pytest.importorskip("hea._rs")

from hea.ggplot.stats.smooth import _loess_local_fit


def _pyref(xq, x, y, span, degree, w, want_var=True):
    f = np.empty(len(xq))
    v = np.empty(len(xq))
    for i, q in enumerate(xq):
        beta, var00 = _loess_local_fit(q, x, y, span, degree, w, want_var=want_var)
        f[i] = beta[0]
        v[i] = var00
    return f, v


def _rust(xq, x, y, span, degree, w, want_var=True):
    f, v = rs.loess_eval(
        np.ascontiguousarray(xq),
        np.ascontiguousarray(x),
        np.ascontiguousarray(y),
        float(span),
        int(degree),
        np.ascontiguousarray(w),
        bool(want_var),
    )
    return np.asarray(f), np.asarray(v)


@pytest.mark.parametrize("n", [40, 400])  # 40 -> serial branch, 400 -> parallel
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("span", [0.3, 0.5, 0.75])
def test_loess_eval_matches_python(n, degree, span):
    g = np.random.default_rng(n * 100 + degree * 10 + int(span * 4))
    x = np.sort(g.uniform(0, 10, n))
    y = np.sin(x) + g.normal(0, 0.3, n)
    w = np.ones(n)
    rf, rv = _rust(x, x, y, span, degree, w)
    pf, pv = _pyref(x, x, y, span, degree, w)
    np.testing.assert_allclose(rf, pf, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rv, pv, rtol=1e-7, atol=1e-12)


def test_loess_eval_matches_python_with_robust_weights():
    n = 300
    g = np.random.default_rng(7)
    x = np.sort(g.uniform(0, 10, n))
    y = np.sin(x) + g.normal(0, 0.3, n)
    w = g.uniform(0.0, 1.0, n)
    rf, rv = _rust(x, x, y, 0.4, 2, w)
    pf, pv = _pyref(x, x, y, 0.4, 2, w)
    np.testing.assert_allclose(rf, pf, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rv, pv, rtol=1e-7, atol=1e-12)


def test_loess_eval_predict_grid_matches_python():
    g = np.random.default_rng(11)
    x = np.sort(g.uniform(0, 10, 500))
    y = np.cos(x) + g.normal(0, 0.2, 500)
    xq = np.linspace(0.5, 9.5, 200)
    w = np.ones(500)
    rf, rv = _rust(xq, x, y, 0.5, 2, w)
    pf, pv = _pyref(xq, x, y, 0.5, 2, w)
    np.testing.assert_allclose(rf, pf, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rv, pv, rtol=1e-7, atol=1e-12)
