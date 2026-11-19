"""R's arithmetic in formulas: the response and the model-matrix columns
hea evaluates from a formula's expressions round as R's do."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from hea.formula import prepare_design


def test_formula_arithmetic_is_r_arithmetic():
    """Formula arithmetic is R's: the response's ``y/3`` is an IEEE division
    (polars would round it through the reciprocal), ``^`` is ``R_pow``
    (libm ``pow``, also at 0.5 where numpy takes ``sqrt``), and ``log(x, 10)``
    and ``log(x, 2)`` are ``log10`` and ``log2`` as R's ``logbase`` makes
    them, not the ratio of logs."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.1, 20.0, 400)
    y = rng.normal(0.0, 3.0, 400)
    df = pl.DataFrame({"x": x, "y": y})
    d = prepare_design("y/3 ~ log(x, 10) + log(x, 2) + I(x^0.5) + I(x^1.7)", df)
    np.testing.assert_array_equal(d.y.to_numpy(), y / 3)
    X = d.X.to_numpy()
    np.testing.assert_array_equal(X[:, 1], np.log10(x))
    np.testing.assert_array_equal(X[:, 2], np.log2(x))
    np.testing.assert_array_equal(X[:, 3], [math.pow(v, 0.5) for v in x])
    np.testing.assert_array_equal(X[:, 4], [math.pow(v, 1.7) for v in x])
    d2 = prepare_design("I(y^0.5 + x) ~ x", df.with_columns(pl.col("y").abs()))
    np.testing.assert_array_equal(
        d2.y.to_numpy(), [math.pow(abs(u), 0.5) + v for u, v in zip(y, x)]
    )
