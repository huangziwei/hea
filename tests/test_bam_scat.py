"""End-to-end ``bam(family=Scat(...), discrete=True)`` parity tests.

Two oracles from ``tests/r_oracle/dump_bam_scat.R``:

* ``simple`` — ``y ~ s(x, k=10)`` on heavy-tailed data; baseline scat
  PIRLS path.
* ``factor`` — ``y ~ g + s(x, by=g, k=10)`` on factor-level-shifted
  heavy-tailed data; exercises the by=factor discrete-path fix together
  with the extended-family θ-Newton.

Each oracle is checked at two operating points:

* **force-θ-and-sp** — feed mgcv's converged ``(ν, σ)`` and ``sp`` to
  hea, refit; assert fitted matches mgcv to ≤ 1e-9 (predictive
  equivalence; gauge-invariant).
* **auto-fit** — let hea estimate θ and sp from scratch; assert
  fitted ≤ 1e-4, θ_log_diff ≤ 1e-4, sp within 3× (REML basin slack
  consistent with the standing flat-basin tolerance from Phase 2).
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hea.family import Scat
import hea


_FIX = Path(__file__).parent / "fixtures" / "scat_bam"


def _have_fix(name: str) -> bool:
    sub = _FIX / name
    return all((sub / f).exists() for f in
               ("data.csv", "sp.csv", "theta.csv", "fitted.csv"))


def _load(name: str):
    sub = _FIX / name
    df = pl.read_csv(str(sub / "data.csv"))
    sp = np.atleast_1d(np.loadtxt(sub / "sp.csv"))
    theta = np.atleast_1d(np.loadtxt(sub / "theta.csv"))
    fitted = np.loadtxt(sub / "fitted.csv")
    return df, sp, theta, fitted


# ---------- simple --------------------------------------------------------


@pytest.mark.skipif(
    not _have_fix("simple"),
    reason="scat_bam/simple oracle missing — run dump_bam_scat.R",
)
def test_scat_simple_force_theta_sp():
    df, sp_mgcv, theta_mgcv, fit_mgcv = _load("simple")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0   # both θ supplied positive ⇒ locked
    m = hea.bam("y ~ s(x, k=10)", dat, family=fam, discrete=True,
                sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"force-(θ,sp) fitted rel diff {rel:.3e}"


@pytest.mark.skipif(
    not _have_fix("simple"),
    reason="scat_bam/simple oracle missing",
)
def test_scat_simple_auto_fit():
    """End-to-end auto-fit on the simple oracle. The dev0-under-new-θ
    recompute (mgcv bgam.fitd:567-569) makes hea's PIRLS cadence
    bit-identical to mgcv's, so we pin tightly."""
    df, sp_mgcv, theta_mgcv, fit_mgcv = _load("simple")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float)}
    m = hea.bam("y ~ s(x, k=10)", dat, family=Scat(min_df=5),
                discrete=True)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"auto-fit fitted rel diff {rel:.3e}"

    theta_h = np.asarray(m.family.get_theta(trans=True))
    # ν and σ are on different absolute scales; rel tolerance handles both.
    assert np.allclose(theta_h, theta_mgcv, rtol=1e-6, atol=0), (
        f"auto-fit θ mismatch: hea={theta_h} mgcv={theta_mgcv}"
    )

    sp_h = np.asarray(m.sp)
    assert np.allclose(sp_h, sp_mgcv, rtol=1e-6, atol=0), (
        f"auto-fit sp mismatch: hea={sp_h} mgcv={sp_mgcv}"
    )


# ---------- factor --------------------------------------------------------


@pytest.mark.skipif(
    not _have_fix("factor"),
    reason="scat_bam/factor oracle missing",
)
def test_scat_factor_force_theta_sp():
    df, sp_mgcv, theta_mgcv, fit_mgcv = _load("factor")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float),
           "g": df["g"].to_numpy()}
    fam = Scat(theta=tuple(theta_mgcv), min_df=5)
    assert fam.n_theta == 0
    m = hea.bam("y ~ g + s(x, by=g, k=10)", dat, family=fam,
                discrete=True, sp=sp_mgcv)
    fit_h = np.asarray(m.fitted_values)
    rel = float(np.linalg.norm(fit_h - fit_mgcv) / np.linalg.norm(fit_mgcv))
    assert rel < 1e-9, f"factor force-(θ,sp) fitted rel diff {rel:.3e}"


@pytest.mark.skipif(
    not _have_fix("factor"),
    reason="scat_bam/factor oracle missing",
)
def test_scat_factor_auto_fit():
    """End-to-end auto-fit on the factor-by oracle (3 levels × 1 smooth
    per level). Combines extended-family θ-Newton with the by=factor
    discrete-path fix; PIRLS-step-halving with dev0-under-new-θ keeps
    iterates bit-identical to mgcv."""
    df, sp_mgcv, theta_mgcv, fit_mgcv = _load("factor")
    dat = {"y": df["y"].to_numpy().astype(float),
           "x": df["x"].to_numpy().astype(float),
           "g": df["g"].to_numpy()}
    m = hea.bam("y ~ g + s(x, by=g, k=10)", dat, family=Scat(min_df=5),
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
