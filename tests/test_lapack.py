"""R's bundled LAPACK as hea reproduces it: base R's ``eigen(symmetric =
TRUE)`` (``dsyevr``), ``qr(x, LAPACK = TRUE)`` with ``qr.qty``/``qr.qy``
(``dgeqp3``, ``dormqr``) and ``solve`` (``dgesv``), the default LINPACK
``qr`` with ``qr.coef``/``qr.qy``, ``lm.fit``'s ``Cdqrls`` and ``svd``
(``dgesdd``), and mgcv's ``slanczos`` (Lanczos with
``dstedc`` on its tridiagonal), against R's own outputs in
``tests/fixtures/lapack`` (written by ``tests/scripts/make_lapack_fixtures.R``).

The inputs are rebuilt here from R's generator (``set.seed``/``runif``
reproduced by :mod:`hea.R.rng`) or from integer arithmetic, so they hold the
same bits R used. With the compiled port and the arithmetic the fixtures were
written under (``PINNED_ARITHMETIC``) every output must be R's to the bit.
Elsewhere the BLAS rounds differently, so eigenvalues are compared to a
tolerance and eigenvectors through the residual and orthogonality they must
satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FIXTURE_ROOT, PINNED_ARITHMETIC

from hea._dispatch import rs_fn
from hea._rlinalg import (
    r_eigen_sym,
    r_qr,
    r_qr_coef,
    r_qr_lapack,
    r_qr_qy,
    r_qr_qy_lapack,
    r_qr_R,
    r_solve,
    r_svd,
)
from hea.formula import _tp_rlanczos
from hea.R.rng import RGenerator

FIX = FIXTURE_ROOT / "lapack"
EXACT_EIGEN = PINNED_ARITHMETIC and rs_fn("r_dsyevr") is not None
EXACT_LANCZOS = PINNED_ARITHMETIC and rs_fn("r_dstedc") is not None
EXACT_QR = PINNED_ARITHMETIC and rs_fn("r_dgeqp3") is not None
EXACT_SOLVE = PINNED_ARITHMETIC and rs_fn("r_dgesv") is not None
EXACT_SVD = PINNED_ARITHMETIC and rs_fn("r_dgesdd") is not None


def _unif_sym(seed: int, n: int) -> np.ndarray:
    U = RGenerator(seed).uniform(size=n * n).reshape(n, n, order="F")
    return U + U.T


def _diffpen50() -> np.ndarray:
    D = np.diff(np.eye(50), 2, axis=0)
    return D.T @ D


def _clus36() -> np.ndarray:
    u = RGenerator(4).uniform(size=36)
    return np.kron(np.eye(12), np.ones((3, 3))) + np.diag(u) * 1e-10


def _asym45() -> np.ndarray:
    """Not symmetric: ``eigen(symmetric = TRUE)`` reads the lower triangle."""
    return RGenerator(9).uniform(size=45 * 45).reshape(45, 45, order="F")


CASES = {
    "rand6": lambda: _unif_sym(1, 6),
    "rand40": lambda: _unif_sym(2, 40),
    "rand140": lambda: _unif_sym(3, 140),
    "diffpen50": _diffpen50,
    "clus36": _clus36,
    "asym45": _asym45,
}


def _read(name: str, shape=None) -> np.ndarray:
    x = np.fromfile(FIX / name, dtype="<f8")
    return x if shape is None else x.reshape(shape, order="F")


@pytest.mark.parametrize("name", sorted(CASES))
def test_eigen_symmetric_matches_r(name: str):
    A = CASES[name]()
    n = A.shape[0]
    vals_r = _read(f"eigen_{name}_values.bin")
    vecs_r = _read(f"eigen_{name}_vectors.bin", (n, n))
    only_r = _read(f"eigen_{name}_onlyvalues.bin")
    w, V = r_eigen_sym(A)
    w_only = r_eigen_sym(A, only_values=True)
    if EXACT_EIGEN:
        np.testing.assert_array_equal(w, vals_r)
        np.testing.assert_array_equal(V, vecs_r)
        np.testing.assert_array_equal(w_only, only_r)
        return
    scale = max(1.0, float(np.abs(vals_r).max()))
    np.testing.assert_allclose(w, vals_r, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(w_only, only_r, rtol=0, atol=1e-12 * scale)
    lower = np.tril(A) + np.tril(A, -1).T
    np.testing.assert_allclose(lower @ V, V * w, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(V.T @ V, np.eye(n), rtol=0, atol=1e-12)


@pytest.mark.parametrize(("seed", "n", "k"), [(5, 120, 10), (6, 300, 20)])
def test_slanczos_matches_mgcv(seed: int, n: int, k: int):
    A = _unif_sym(seed, n)
    vals_r = _read(f"slanczos_{n}_{k}_values.bin")
    vecs_r = _read(f"slanczos_{n}_{k}_vectors.bin", (n, k))
    vals, vecs = _tp_rlanczos(A, k, -1, tol=float(np.finfo(float).eps) ** 0.5)
    if EXACT_LANCZOS:
        np.testing.assert_array_equal(vals, vals_r)
        np.testing.assert_array_equal(vecs, vecs_r)
        return
    scale = float(np.abs(vals_r).max())
    np.testing.assert_allclose(vals, vals_r, rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(A @ vecs, vecs * vals, rtol=0, atol=1e-6 * scale)


def _unif_mat(g: RGenerator, n: int, p: int) -> np.ndarray:
    return g.uniform(size=n * p).reshape(n, p, order="F")


def _close(got, ref, scale: float = 1.0) -> None:
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * max(1.0, scale))


def test_qr_lapack_matches_r():
    g = RGenerator(41)
    X = _unif_mat(g, 150, 140)
    Y = _unif_mat(g, 150, 3)
    q = r_qr_lapack(X)
    qty = r_qr_qy_lapack(q, Y, transpose=True)
    qy = r_qr_qy_lapack(q, Y)
    R = r_qr_R(q)
    if EXACT_QR:
        np.testing.assert_array_equal(q["qr"], _read("qrl_qr.bin", (150, 140)))
        np.testing.assert_array_equal(q["qraux"], _read("qrl_qraux.bin", None))
        np.testing.assert_array_equal(
            q["pivot"], _read("qrl_pivot.bin", None).astype(int)
        )
        np.testing.assert_array_equal(qty, _read("qrl_qty.bin", (150, 3)))
        np.testing.assert_array_equal(qy, _read("qrl_qy.bin", (150, 3)))
        np.testing.assert_array_equal(R, _read("qrl_R.bin", (140, 140)))
        return
    full = np.vstack([R, np.zeros((10, 140))])
    _close(r_qr_qy_lapack(q, full), X[:, q["pivot"] - 1], 140.0)
    _close(r_qr_qy_lapack(q, qty), Y, 150.0)


def test_qr_linpack_matches_r():
    g = RGenerator(42)
    Z = _unif_mat(g, 40, 9)
    Z[:, 8] = Z[:, 0] + Z[:, 1]
    W = _unif_mat(g, 40, 2)
    q = r_qr(Z)
    assert q["rank"] == int(_read("qr_rank.bin", None)[0]) == 8
    np.testing.assert_array_equal(q["pivot"], _read("qr_pivot.bin", None).astype(int))
    coef = r_qr_coef(q, W)
    coef_r = _read("qr_coef.bin", (9, 2))
    np.testing.assert_array_equal(np.isnan(coef), np.isnan(coef_r))
    if EXACT_QR:
        np.testing.assert_array_equal(q["qr"], _read("qr_qr.bin", (40, 9)))
        np.testing.assert_array_equal(q["qraux"], _read("qr_qraux.bin", None))
        np.testing.assert_array_equal(np.nan_to_num(coef), np.nan_to_num(coef_r))
        np.testing.assert_array_equal(r_qr_qy(q, W), _read("qr_qy.bin", (40, 2)))
        return
    _close(np.nan_to_num(coef), np.nan_to_num(coef_r), 10.0)
    _close(r_qr_qy(q, W), _read("qr_qy.bin", (40, 2)), 10.0)


def test_solve_matches_r():
    g = RGenerator(43)
    A = _unif_mat(g, 70, 70)
    B = _unif_mat(g, 70, 2)
    x, inv = r_solve(A, B), r_solve(A)
    if EXACT_SOLVE:
        np.testing.assert_array_equal(x, _read("solve_ab.bin", (70, 2)))
        np.testing.assert_array_equal(inv, _read("solve_inv.bin", (70, 70)))
        return
    _close(A @ x, B, 70.0)
    _close(A @ inv, np.eye(70), 70.0)


def test_solve_reports_singularity_as_r():
    with pytest.raises(np.linalg.LinAlgError, match=r"exactly singular: U\[2,2\] = 0"):
        r_solve(np.array([[1.0, 2.0], [2.0, 4.0]]))


@pytest.mark.parametrize(
    ("name", "n", "p", "deficient"), [("s", 12, 4, False), ("d", 60, 6, True)]
)
def test_cdqrls_matches_r(name: str, n: int, p: int, deficient: bool):
    from hea.R.linalg import Cdqrls

    g = RGenerator(44)
    Xs, ys = _unif_mat(g, 12, 4), g.uniform(size=12)
    Xd, yd = _unif_mat(g, 60, 6), g.uniform(size=60)
    Xd[:, 5] = Xd[:, 0] - Xd[:, 1]
    X, y = (Xs, ys) if name == "s" else (Xd, yd)
    z = Cdqrls(X, y, 1e-7)
    assert z["rank"] == (p - 1 if deficient else p)
    got = np.concatenate(
        [
            np.asarray(z["coefficients"], float),
            np.asarray(z["residuals"], float),
            np.asarray(z["effects"], float),
            np.asarray(z["qraux"], float),
            np.asarray(z["qr"], float).ravel(order="F"),
        ]
    )
    ref = np.fromfile(FIX / f"cdqrls_{name}.bin", dtype="<f8")
    if PINNED_ARITHMETIC:
        np.testing.assert_array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-11)


@pytest.mark.parametrize(
    "seed, n, p",
    [
        (51, 12, 8),
        (52, 40, 40),
        (53, 30, 70),
        (54, 200, 30),
        (55, 7, 5),
        (56, 3, 3),
        (57, 140, 140),
    ],
)
def test_svd_matches_r(seed: int, n: int, p: int):
    """``svd(X)`` (``jobu = 'S'``), ``svd(X, 0, 0)`` (``'N'``) and
    ``svd(X, nrow(X), ncol(X))`` (``'A'``): tall, wide and square, past the
    divide-and-conquer threshold and the QR/LQ-first ratio."""
    X = _unif_mat(RGenerator(seed), n, p)
    k = min(n, p)
    for job, nu, nv in (("S", None, None), ("N", 0, 0), ("A", n, p)):
        sv = r_svd(X, nu, nv)
        ref = _read(f"svd_{n}x{p}_{job}.bin", None)
        if job == "N":
            if EXACT_SVD:
                np.testing.assert_array_equal(sv["d"], ref)
            else:
                np.testing.assert_allclose(sv["d"], ref, rtol=1e-12, atol=1e-12)
            continue
        uc, vc = (k, k) if job == "S" else (n, p)
        d = ref[:k]
        u = ref[k : k + n * uc].reshape(n, uc, order="F")
        v = ref[k + n * uc :].reshape(p, vc, order="F")
        if EXACT_SVD:
            np.testing.assert_array_equal(sv["d"], d)
            np.testing.assert_array_equal(sv["u"], u)
            np.testing.assert_array_equal(sv["v"], v)
            continue
        np.testing.assert_allclose(sv["d"], d, rtol=1e-12, atol=1e-12)
        recon = sv["u"][:, :k] @ np.diag(sv["d"]) @ sv["v"][:, :k].T
        _close(recon, X, float(d[0]))
        _close(sv["u"].T @ sv["u"], np.eye(uc))
        _close(sv["v"].T @ sv["v"], np.eye(vc))
