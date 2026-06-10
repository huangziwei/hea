"""
Compare hea smooth basis machinery against mgcv per-smooth fixtures.

Two parametrised tests, both walking the same mgcv fixture corpus:

1. ``test_mgcv_smooths_match_R`` — fit-time basis: ``materialize_smooths``
   produces ``X`` and ``S`` (penalties) matching R per block. Skips
   fixtures where R's ``smoothCon`` errored.
2. ``test_mgcv_predict_mat_matches_R`` — predict-time basis:
   ``BasisSpec.predict_mat`` produces the design at ``predict_data``
   matching R's ``PredictMat``. Skips fixtures missing ``predict_data.csv``
   or per-block ``Xpred.mtx``.

Sign conventions for basis columns are arbitrary between np.linalg.eigh
and mgcv's Rlanczos, so we match each column up to sign and apply the
same flip to S. Tolerances are normalized against max|ref| because S
(and X, for high-k tp) spans several orders of magnitude.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy.io import mmread

from conftest import (
    FIXTURE_ROOT,
    _apply_schema,
    fixture_meta,
    fixtures_by_kind,
    load_dataset,
)
from hea.formula import (
    expand,
    materialize_smooths,
    parse,
)


def _fs_null_layout(r_meta, ncol):
    """Column layout of an fs.interaction block.

    mgcv's fs null-space basis is resolved from an exactly-degenerate
    eigenspace, so the basis WITHIN the null span is LAPACK-build noise:
    R itself produces an O(1)-rotated null pair when re-run on a different
    machine (these fixtures carry the generating machine's draw). hea's
    nat.param is fp-faithful to mgcv's, making its draw equally legitimate
    — so tests compare the null SPAN per level tightly instead of raw null
    columns. Returns ``(p, rank, null_d, nf, null_cols)`` where ``p`` is
    the per-level block width and ``null_cols`` the global indices of all
    null columns.
    """
    p = r_meta["bs_dim"]
    null_d = r_meta["n_penalties"] - 1
    rank = p - null_d
    nf = ncol // p
    null_cols = np.array(
        [j * p + rank + i for j in range(nf) for i in range(null_d)],
        dtype=int,
    )
    return p, rank, null_d, nf, null_cols


MGCV_FIXTURES = fixtures_by_kind("mgcv")


def _has_error(fx_id: str) -> bool:
    sm_meta = json.loads((FIXTURE_ROOT / fx_id / "smooth_meta.json").read_text())
    return any("error" in s for s in sm_meta.get("smooths", []))


MGCV_OK = [e["id"] for e in MGCV_FIXTURES if not _has_error(e["id"])]


@pytest.mark.parametrize("fx_id", MGCV_OK)
def test_mgcv_smooths_match_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    fx = FIXTURE_ROOT / fx_id
    sm_meta = json.loads((fx / "smooth_meta.json").read_text())

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)

    # R's gam drops rows with NA in ANY formula variable — match that.
    need = set(meta.get("need_vars", [])) & set(data.columns)
    if need:
        data = data.drop_nulls(subset=list(need))

    ours = materialize_smooths(ef, data)

    assert len(ours) == len(sm_meta["smooths"]), (
        f"n smooths: got {len(ours)} want {len(sm_meta['smooths'])}"
    )

    for i, (ours_blocks, r_meta) in enumerate(zip(ours, sm_meta["smooths"]), start=1):
        assert len(ours_blocks) == r_meta["n_blocks"], (
            f"smooth #{i}: got {len(ours_blocks)} blocks want {r_meta['n_blocks']}"
        )

        for k, blk in enumerate(ours_blocks, start=1):
            X_ref = np.asarray(
                mmread(fx / f"smooth_{i}_{k}_X.mtx").todense(), dtype=float
            )
            assert blk.X.shape == X_ref.shape, (
                f"smooth #{i} block {k}: X shape got {blk.X.shape} want {X_ref.shape}"
            )

            # fs.interaction: the null columns carry a machine-noise rotation
            # (see _fs_null_layout) — compare their span per level instead of
            # raw values, and compare S up to the common maXX = ||X||_inf^2
            # factor that scale.penalty derives from rotation-sensitive row
            # sums. Everything else (and every other class) compares raw.
            is_null = np.zeros(blk.X.shape[1], dtype=bool)
            s_scale_ratio = 1.0
            if r_meta["class"] == "fs.smooth.spec":
                p_lev, fs_rank, null_d, nf, null_cols = _fs_null_layout(
                    r_meta, blk.X.shape[1]
                )
                is_null[null_cols] = True
                maXX_ours = float(np.abs(blk.X).sum(axis=1).max()) ** 2
                maXX_ref = float(np.abs(X_ref).sum(axis=1).max()) ** 2
                s_scale_ratio = maXX_ours / maXX_ref

            # mgcv's Lanczos uses an arbitrary per-eigenvector sign convention;
            # hea's np.linalg.eigh uses its own. Match each column up to sign,
            # then apply the same flip to S.
            signs = np.ones(blk.X.shape[1])
            X_got = blk.X.copy()
            for c in range(blk.X.shape[1]):
                plus = float(np.max(np.abs(blk.X[:, c] - X_ref[:, c])))
                minus = float(np.max(np.abs(blk.X[:, c] + X_ref[:, c])))
                if minus < plus:
                    signs[c] = -1.0
                    X_got[:, c] = -blk.X[:, c]

            tol_X = max(1e-6, 1e-5 * float(np.max(np.abs(X_ref))))
            assert np.allclose(
                X_got[:, ~is_null], X_ref[:, ~is_null], atol=tol_X, rtol=0
            ), (
                f"smooth #{i} block {k} ({r_meta['class']}): X values diverge"
            )
            if is_null.any():
                # Null block: same span, orthogonal relative rotation.
                for j in range(nf):
                    cols = np.arange(j * p_lev + fs_rank, (j + 1) * p_lev)
                    A, B = X_got[:, cols], X_ref[:, cols]
                    G, *_ = np.linalg.lstsq(A, B, rcond=None)
                    assert float(np.max(np.abs(A @ G - B))) < tol_X, (
                        f"smooth #{i} block {k}: fs null span diverges "
                        f"(level {j})"
                    )
                    assert np.allclose(G.T @ G, np.eye(null_d), atol=1e-6), (
                        f"smooth #{i} block {k}: fs null relative rotation "
                        f"not orthogonal (level {j})"
                    )

            assert len(blk.S) == r_meta["n_penalties"], (
                f"smooth #{i} block {k}: got {len(blk.S)} penalties want {r_meta['n_penalties']}"
            )
            for j, S_got in enumerate(blk.S, start=1):
                S_ref = np.asarray(
                    mmread(fx / f"smooth_{i}_{k}_S_{j}.mtx").todense(), dtype=float
                ) * s_scale_ratio
                assert S_got.shape == S_ref.shape, (
                    f"smooth #{i} block {k} S_{j}: got {S_got.shape} want {S_ref.shape}"
                )
                S_got_flipped = S_got * signs[:, None] * signs[None, :]
                tol_S = max(1e-6, 1e-5 * float(np.max(np.abs(S_ref))))
                assert np.allclose(S_got_flipped, S_ref, atol=tol_S, rtol=0), (
                    f"smooth #{i} block {k} S_{j} ({r_meta['class']}): penalty values diverge"
                )


# =============================================================================
# Predict-time basis (BasisSpec.predict_mat vs mgcv's PredictMat)
# =============================================================================
#
# The R generator dumps PredictMat(s, predict_data) per smooth block
# (`smooth_<i>_<k>_Xpred.mtx`) and the predict_data subset
# (`predict_data.csv`). Rebuild the smooth at fit-time on `data`, then ask
# each block's BasisSpec to produce the design at predict_data and compare
# against R's output.

def _load_predict_data(fx_id: str, pkg: str, name: str) -> pl.DataFrame:
    """Load `predict_data.csv` and re-apply factor schema (CSV round-trip
    erases R factor types — without this, fs/sz/by=factor smooths fail to
    match levels in our predict closure)."""
    df = pl.read_csv(FIXTURE_ROOT / fx_id / "predict_data.csv", null_values="NA")
    return _apply_schema(df, pkg, name)


def _has_predict_data(fx_id: str) -> bool:
    return (FIXTURE_ROOT / fx_id / "predict_data.csv").exists()


MGCV_OK_PREDICT = [
    e["id"] for e in MGCV_FIXTURES
    if not _has_error(e["id"]) and _has_predict_data(e["id"])
]


@pytest.mark.parametrize("fx_id", MGCV_OK_PREDICT)
def test_mgcv_predict_mat_matches_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    fx = FIXTURE_ROOT / fx_id
    sm_meta = json.loads((fx / "smooth_meta.json").read_text())

    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)

    # NA-omit on every formula variable — same rule R's gam applies and
    # what `make_mgcv_fixture` did when constructing predict_data.
    need = set(meta.get("need_vars", [])) & set(data.columns)
    if need:
        data = data.drop_nulls(subset=list(need))

    ours = materialize_smooths(ef, data)
    new = _load_predict_data(fx_id, pkg, name)

    assert len(ours) == len(sm_meta["smooths"]), (
        f"n smooths: got {len(ours)} want {len(sm_meta['smooths'])}"
    )

    for i, (ours_blocks, r_meta) in enumerate(zip(ours, sm_meta["smooths"]), start=1):
        for k, blk in enumerate(ours_blocks, start=1):
            xpred_path = fx / f"smooth_{i}_{k}_Xpred.mtx"
            if not xpred_path.exists():
                # mgcv's PredictMat refused for this block (rare niche cases) —
                # nothing to compare against.
                continue

            assert blk.spec is not None, (
                f"smooth #{i} block {k} ({r_meta['class']}): missing BasisSpec"
            )

            X_pred_ref = np.asarray(mmread(xpred_path).todense(), dtype=float)
            X_pred_ours = blk.spec.predict_mat(new)

            assert X_pred_ours.shape == X_pred_ref.shape, (
                f"smooth #{i} block {k}: predict shape "
                f"got {X_pred_ours.shape} want {X_pred_ref.shape}"
            )

            # Match column signs against an in-sample anchor that lives in the
            # same column space as the predict basis. For most bases this is
            # `sm$X` (smooth_*_X.mtx) — fit and predict bases coincide. For
            # `t2` they don't: sm$X is the partial absorb (sm$Cp ignored), but
            # PredictMat applies the full absorb via sm$qrc, giving a basis
            # whose per-column signs do not match sm$X's. For those we use
            # `Xpredfit.mtx` (PredictMat at fit data), which carries the
            # predict-side sign convention and is paired with our
            # in-sample `predict_mat(data)` output. We gate on coef_remap —
            # only t2 needs this; for other bases sm$X already matches the
            # predict basis (sometimes Xpredfit doesn't, e.g. random.effect).
            xpredfit_path = fx / f"smooth_{i}_{k}_Xpredfit.mtx"
            use_predfit_anchor = (
                xpredfit_path.exists()
                and blk.spec is not None
                and blk.spec.coef_remap is not None
            )
            if use_predfit_anchor:
                anchor_ref = np.asarray(
                    mmread(xpredfit_path).todense(), dtype=float
                )
                anchor_ours = np.asarray(blk.spec.predict_mat(data), dtype=float)
            else:
                anchor_ref = np.asarray(
                    mmread(fx / f"smooth_{i}_{k}_X.mtx").todense(), dtype=float
                )
                anchor_ours = blk.X

            signs = np.ones(blk.X.shape[1])
            for c in range(blk.X.shape[1]):
                plus = float(np.max(np.abs(anchor_ours[:, c] - anchor_ref[:, c])))
                minus = float(np.max(np.abs(anchor_ours[:, c] + anchor_ref[:, c])))
                if minus < plus:
                    signs[c] = -1.0
            X_pred_aligned = X_pred_ours * signs[None, :]

            # fs.interaction: align the machine-noise null rotation (see
            # _fs_null_layout) by the per-level 2x2 map estimated from the
            # FIT anchors; predict columns are the same fixed P applied to
            # new data, so the fit-time relative rotation carries over
            # exactly.
            if r_meta["class"] == "fs.smooth.spec":
                p_lev, fs_rank, null_d, nf, _ = _fs_null_layout(
                    r_meta, blk.X.shape[1]
                )
                for j in range(nf):
                    cols = np.arange(j * p_lev + fs_rank, (j + 1) * p_lev)
                    G, *_ = np.linalg.lstsq(
                        anchor_ours[:, cols], anchor_ref[:, cols], rcond=None
                    )
                    X_pred_aligned[:, cols] = X_pred_ours[:, cols] @ G

            tol = max(1e-6, 1e-5 * float(np.max(np.abs(X_pred_ref))))
            assert np.allclose(X_pred_aligned, X_pred_ref, atol=tol, rtol=0), (
                f"smooth #{i} block {k} ({r_meta['class']}): predict_mat diverges "
                f"(max abs diff = {float(np.max(np.abs(X_pred_aligned - X_pred_ref))):.2e}, "
                f"tol = {tol:.2e})"
            )
