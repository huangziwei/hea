"""Bit-exact parity test for hea.bam(discrete=True) on the small_data
Poisson te(pm10, lag, k=c(5,3)) oracle. This is the rank-deficient
canonical example used to develop Phase 1 (RNG fix) and Phase 2′
(chol2qr identity-bias fix + pivoted Chol with rank.tol + POI optimizer).

With the POI port (Phase 2′.7) the fitted match mgcv at ≤ 1e-9 when
sp is forced; auto-sp lands in a different specific point of the flat
REML basin (rank-deficient null direction), so we don't pin auto-sp
fitted as tightly there.

The coef gauge is *not* asserted element-wise: hea and mgcv pivoted
Cholesky pick different positions to drop from the rank-deficient null
space, so β values differ by a null(A)-direction. Fitted is gauge-
invariant and is the right thing to check.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hea.family import Poisson
import hea


_FIX = Path(__file__).parent / "fixtures" / "small_data"


@pytest.mark.skipif(
    not (_FIX / "data.csv").exists(),
    reason="small_data oracle missing — see tests/r_oracle/",
)
def test_small_data_force_sp():
    """With mgcv's sp forced, hea's β must give fitted matching mgcv
    to ≤ 1e-9 (gauge-invariant predictive equivalence)."""
    df = pl.read_csv(str(_FIX / "data.csv"))
    pm10 = df.select(["X1", "X2", "X3", "X4"]).to_numpy().astype(float)
    lag = df.select(["X1.1", "X2.1", "X3.1", "X4.1"]).to_numpy().astype(float)
    dat = {"y": df["y"].to_numpy().astype(float),
           "pm10": pm10, "lag": lag}

    sp_mgcv = np.atleast_1d(np.loadtxt(_FIX / "sp.csv"))
    m = hea.bam("y ~ te(pm10, lag, k=c(5, 3))", dat,
                family=Poisson(), discrete=True, sp=sp_mgcv)

    fit_mgcv = np.loadtxt(_FIX / "fitted.csv")
    fit_hea = np.asarray(m.fitted_values)
    rel_fit = float(
        np.linalg.norm(fit_hea - fit_mgcv) / np.linalg.norm(fit_mgcv)
    )
    assert rel_fit < 1e-9, (
        f"small_data force-sp fitted rel diff {rel_fit:.3e} > 1e-9"
    )


@pytest.mark.skipif(
    not (_FIX / "data.csv").exists(),
    reason="small_data oracle missing",
)
def test_small_data_auto_sp():
    """Auto-sp on the rank-deficient small_data model lands in a
    different specific basin point than mgcv, but fitted should still
    be within 1e-5 because the basin is flat and both impls minimise
    REML to machine precision."""
    df = pl.read_csv(str(_FIX / "data.csv"))
    pm10 = df.select(["X1", "X2", "X3", "X4"]).to_numpy().astype(float)
    lag = df.select(["X1.1", "X2.1", "X3.1", "X4.1"]).to_numpy().astype(float)
    dat = {"y": df["y"].to_numpy().astype(float),
           "pm10": pm10, "lag": lag}

    m = hea.bam("y ~ te(pm10, lag, k=c(5, 3))", dat,
                family=Poisson(), discrete=True)

    fit_mgcv = np.loadtxt(_FIX / "fitted.csv")
    fit_hea = np.asarray(m.fitted_values)
    rel_fit = float(
        np.linalg.norm(fit_hea - fit_mgcv) / np.linalg.norm(fit_mgcv)
    )
    assert rel_fit < 1e-5, (
        f"small_data auto-sp fitted rel diff {rel_fit:.3e} > 1e-5"
    )
