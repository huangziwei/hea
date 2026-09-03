"""
Compare hea.formula.materialize() against R's X.csv fixture by fixture.

Per WR fixture: parse + expand + materialize, then assert shape and values
match R's stats::model.matrix output. Replaces the legacy
test_wr_fixtures.py, which compared formulaic against R.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from conftest import FIXTURE_ROOT, fixture_meta, fixtures_by_kind, load_dataset

from hea.formula import (
    Call,
    ParseError,
    Subscript,
    deparse,
    expand,
    materialize,
    parse,
    prepare_design,
    referenced_columns,
)

WR_FIXTURES = fixtures_by_kind("wr")
WR_IDS = [e["id"] for e in WR_FIXTURES]


def _load_X_ref(fx_id: str, n_rows: int) -> pl.DataFrame:
    path = FIXTURE_ROOT / fx_id / "X.csv"
    head = path.read_text().splitlines()[:1]
    if not head or not head[0].strip() or "," not in head[0] and '"' not in head[0]:
        return pl.DataFrame()
    return pl.read_csv(path, null_values="NA", infer_schema_length=0).cast(pl.Float64)


@pytest.mark.parametrize("fx_id", WR_IDS)
def test_wr_materialize_matches_R(fx_id: str):
    meta, _ = fixture_meta(fx_id)
    pkg, name = meta["dataset"]["pkg"], meta["dataset"]["name"]
    data = load_dataset(pkg, name)

    X_ref = _load_X_ref(fx_id, len(data))

    f = parse(meta["formula"])
    data_cols = list(data.columns) if "." in meta["formula"] else None
    ef = expand(f, data_columns=data_cols)
    X_got = materialize(ef, data)

    got_shape = X_got.shape if X_got.width > 0 else (len(data), 0)
    ref_shape = X_ref.shape if X_ref.width > 0 else (len(data), 0)
    assert got_shape == ref_shape, (
        f"shape: got {got_shape} want {ref_shape}  formula={meta['formula']!r}"
    )

    if X_got.width > 0:
        np.testing.assert_allclose(
            X_got.to_numpy().astype(float),
            X_ref.to_numpy().astype(float),
            rtol=1e-6,
            atol=1e-8,
            err_msg=(
                f"formula={meta['formula']!r}\n"
                f"  got cols: {list(X_got.columns)}\n"
                f"  ref cols: {list(X_ref.columns)}"
            ),
        )


def test_referenced_columns_includes_smooth_vars():
    ef = expand(parse("y ~ x + s(z, bs='cr') + te(u, v, by=g)"))
    assert referenced_columns(ef) >= {"x", "z", "u", "v", "g"}


def test_prepare_design_drops_na_on_smooth_only_var():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.1, None, 0.3, 0.4, None],
        }
    )
    d = prepare_design("y ~ s(z)", df)
    assert d.data.height == 3
    assert d.X.height == 3
    assert d.y.len() == 3


@pytest.mark.parametrize(
    "formula,want",
    [
        ("y ~ a + b:a", ["a", "a:b"]),
        ("y ~ z + x:z", ["z", "z:x"]),
        ("y ~ c + a:b:c", ["c", "c:a:b"]),
        ("y ~ x:z + a:z", ["x:z", "z:a"]),
        ("y ~ a:c + c:b", ["a:c", "c:b"]),
        ("y ~ (x1 + x2):f2", ["x1:f2", "x2:f2"]),
        ("y ~ x2:f2 + x1:f2", ["x2:f2", "f2:x1"]),
    ],
)
def test_within_term_var_order_matches_R(formula, want):
    assert expand(parse(formula)).term_labels == want


def test_within_term_order_model_matrix_columns():
    d = pl.DataFrame(
        {
            "a": pl.Series(
                [["A1", "A2"][i % 2] for i in range(12)], dtype=pl.Enum(["A1", "A2"])
            ),
            "b": pl.Series(
                [["B1", "B2", "B3"][i % 3] for i in range(12)],
                dtype=pl.Enum(["B1", "B2", "B3"]),
            ),
            "x": np.arange(12, dtype=float),
            "z": np.arange(12, dtype=float) + 0.5,
        }
    )
    assert materialize(expand(parse("~ b + a:b")), d).columns == [
        "(Intercept)",
        "bB2",
        "bB3",
        "bB1:aA2",
        "bB2:aA2",
        "bB3:aA2",
    ]
    assert materialize(expand(parse("~ z + x:z")), d).columns == [
        "(Intercept)",
        "z",
        "z:x",
    ]
    d2 = pl.DataFrame(
        {
            "x1": np.arange(6, dtype=float),
            "x2": np.arange(6, dtype=float) + 0.5,
            "f2": pl.Series(["P", "Q"] * 3, dtype=pl.Enum(["P", "Q"])),
        }
    )
    assert materialize(expand(parse("~ (x1+x2):f2")), d2).columns == [
        "(Intercept)",
        "x1:f2P",
        "x1:f2Q",
        "x2:f2P",
        "x2:f2Q",
    ]


def test_no_intercept_factor_coding_matches_R():
    d = pl.DataFrame(
        {
            "a": pl.Series(
                [["A1", "A2"][i % 2] for i in range(12)], dtype=pl.Enum(["A1", "A2"])
            ),
            "b": pl.Series(
                [["B1", "B2", "B3"][i % 3] for i in range(12)],
                dtype=pl.Enum(["B1", "B2", "B3"]),
            ),
            "c": pl.Series(
                [["C1", "C2"][i % 2] for i in range(12)], dtype=pl.Enum(["C1", "C2"])
            ),
        }
    )
    assert materialize(expand(parse("~ a + b - 1")), d).columns == [
        "aA1",
        "aA2",
        "bB2",
        "bB3",
    ]
    assert materialize(expand(parse("~ 0 + b + a")), d).columns == [
        "bB1",
        "bB2",
        "bB3",
        "aA2",
    ]
    assert materialize(expand(parse("~ 0 + a + b + c")), d).columns == [
        "aA1",
        "aA2",
        "bB2",
        "bB3",
        "cC2",
    ]
    assert materialize(expand(parse("~ a:b + b:c - 1")), d).columns == [
        "aA1:bB1",
        "aA2:bB1",
        "aA1:bB2",
        "aA2:bB2",
        "aA1:bB3",
        "aA2:bB3",
        "bB1:cC2",
        "bB2:cC2",
        "bB3:cC2",
    ]
    X = materialize(expand(parse("~ a + b - 1")), d).to_numpy()
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_logical_is_two_level_factor_matches_R():
    d = pl.DataFrame(
        {"l": [True, False, True, True, False, False], "x": [1.0, 2, 3, 4, 5, 6]}
    )
    assert materialize(expand(parse("~ l")), d).columns == ["(Intercept)", "lTRUE"]
    assert materialize(expand(parse("~ 0 + l")), d).columns == ["lFALSE", "lTRUE"]
    assert materialize(expand(parse("~ l:x")), d).columns == [
        "(Intercept)",
        "lFALSE:x",
        "lTRUE:x",
    ]
    assert materialize(expand(parse("~ l*x")), d).columns == [
        "(Intercept)",
        "lTRUE",
        "x",
        "lTRUE:x",
    ]
    assert materialize(expand(parse("~ l")), d)["lTRUE"].to_list() == [
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]


def test_logical_expression_is_two_level_factor_matches_R():
    d = pl.DataFrame(
        {
            "x1": [-1.0, 2, -3, 4, 0.5, -0.5],
            "f2": pl.Series(["P", "Q", "P", "Q", "P", "Q"], dtype=pl.Enum(["P", "Q"])),
        }
    )
    assert materialize(expand(parse("~ I(x1>0)")), d).columns == [
        "(Intercept)",
        "I(x1 > 0)TRUE",
    ]
    assert materialize(expand(parse("~ I(x1>0) - 1")), d).columns == [
        "I(x1 > 0)FALSE",
        "I(x1 > 0)TRUE",
    ]
    assert materialize(expand(parse("~ x1>0")), d).columns == [
        "(Intercept)",
        "x1 > 0TRUE",
    ]
    assert materialize(expand(parse("~ I(x1>0):f2")), d).columns == [
        "(Intercept)",
        "I(x1 > 0)FALSE:f2P",
        "I(x1 > 0)TRUE:f2P",
        "I(x1 > 0)FALSE:f2Q",
        "I(x1 > 0)TRUE:f2Q",
    ]
    assert materialize(expand(parse("~ I(x1>0)*f2")), d).columns == [
        "(Intercept)",
        "I(x1 > 0)TRUE",
        "f2Q",
        "I(x1 > 0)TRUE:f2Q",
    ]
    assert materialize(expand(parse("~ I(x1>0)")), d)["I(x1 > 0)TRUE"].to_list() == [
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]
    assert materialize(expand(parse("~ I((x1>0) + 1)")), d).columns == [
        "(Intercept)",
        "I((x1 > 0) + 1)",
    ]


def test_cut_breaks_and_diglab_labels_match_R():
    import hea.formula as _F

    d = pl.DataFrame({"x": [-3.0, -1, 0, 1.5, 3, 5.2, 7, 9]})

    def _cut(expr):
        blk = _F._eval_call(parse("~ " + expr).rhs, d)
        codes = [int(c) + 1 for c in blk.codes]  # R's 1-based integer codes
        return blk.levels, codes

    lev, codes = _cut("cut(x,3)")
    assert lev == ["(-3.01,1]", "(1,5]", "(5,9.01]"]
    assert codes == [1, 1, 1, 2, 2, 3, 3, 3]
    lev, _ = _cut("cut(x,4,dig.lab=4)")
    assert lev == ["(-3.012,0]", "(0,3]", "(3,6]", "(6,9.012]"]
    lev, codes = _cut("cut(x,c(-5,0,5,10),right=FALSE)")
    assert lev == ["[-5,0)", "[0,5)", "[5,10)"]
    assert codes == [1, 1, 2, 2, 2, 3, 3, 3]


def test_contrasts_arg_matrix_and_options_default():
    from hea.formula import with_contrasts, with_default_contrasts

    d = pl.DataFrame(
        {"f": pl.Series(["a", "b", "c", "a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))}
    )
    M = np.array([[1.0, 0], [-1, 1], [0, -1]])  # k×(k-1)
    with with_contrasts({"f": M}):
        X = materialize(expand(parse("~ f")), d)
        assert X.columns == ["(Intercept)", "f1", "f2"]
        np.testing.assert_allclose(
            X.select(["f1", "f2"]).head(3).to_numpy(), [[1, 0], [-1, 1], [0, -1]]
        )
        assert materialize(expand(parse("~ 0 + f")), d).columns == ["fa", "fb", "fc"]
    with with_contrasts({"f": (M, ["lo", "hi"])}):
        assert materialize(expand(parse("~ f")), d).columns == [
            "(Intercept)",
            "flo",
            "fhi",
        ]
    with with_default_contrasts("contr.sum", "contr.poly"):
        X = materialize(expand(parse("~ f")), d)
        assert X.columns == ["(Intercept)", "f1", "f2"]
        np.testing.assert_allclose(
            X.select(["f1", "f2"]).head(3).to_numpy(), [[1, 0], [0, 1], [-1, -1]]
        )  # contr.sum
    assert materialize(expand(parse("~ f")), d).columns == ["(Intercept)", "fb", "fc"]


def test_prepare_design_na_action():
    d = pl.DataFrame({"y": [1.0, 2, None, 4, 5], "x": [1.0, 2, 3, None, 5]})
    assert prepare_design("y ~ x", d).X.height == 3
    assert prepare_design("y ~ x", d, na_action="exclude").X.height == 3
    des = prepare_design("y ~ x", d, na_action="pass")
    assert des.X.height == 5 and des.y.len() == 5
    assert np.isnan(des.X.to_numpy()).any()
    with pytest.raises(ValueError, match="missing values"):
        prepare_design("y ~ x", d, na_action="fail")
    with pytest.raises(ValueError, match="na_action must be"):
        prepare_design("y ~ x", d, na_action="bogus")
    clean = pl.DataFrame({"y": [1.0, 2, 3], "x": [1.0, 2, 3]})
    assert prepare_design("y ~ x", clean, na_action="fail").X.height == 3


def test_bs_predict_extrapolates_beyond_boundary_matches_R():
    trn = pl.DataFrame(
        {"x1": [-1.2, -0.5, 0.1, 0.3, 0.7, 0.9, 1.1, 1.4, 1.8, 2.0, 2.3, 2.6]}
    )
    new = pl.DataFrame({"x1": [-2.0, -1.5, 0.5, 3.0, 3.5]})
    state: dict = {}
    materialize(expand(parse("~ bs(x1, df=5)")), trn, basis_state=state)
    Xn = materialize(expand(parse("~ bs(x1, df=5)")), new, basis_state=state)
    cols = [c for c in Xn.columns if c != "(Intercept)"]
    got = Xn.select(cols).to_numpy()
    want = np.array(
        [
            [-2.562925, 0.524316, -0.027902, 0.000000, 0.000000],
            [-0.661962, 0.062595, -0.001471, 0.000000, 0.000000],
            [0.152626, 0.579579, 0.267742, 0.000000, 0.000000],
            [0.000000, -0.007765, 0.271253, -1.863097, 2.599609],
            [0.000000, -0.088452, 1.689254, -6.868472, 6.267670],
        ]
    )
    assert np.allclose(got, want, atol=1e-6)


def test_scale_center_false_uses_rms_matches_R():
    import hea.formula as _F

    d = pl.DataFrame({"x": [2.0, 4, 4, 4, 5, 5, 7, 9]})

    def _col(expr):
        return np.asarray(_F._eval_call(parse("~ " + expr).rhs, d).values).ravel()

    assert np.allclose(
        _col("scale(x, center=FALSE)"),
        [
            0.347404,
            0.694808,
            0.694808,
            0.694808,
            0.868510,
            0.868510,
            1.215915,
            1.563319,
        ],
        atol=1e-6,
    )
    assert np.allclose(
        _col("scale(x)"),
        [-1.403122, -0.467707, -0.467707, -0.467707, 0.0, 0.0, 0.935414, 1.870829],
        atol=1e-6,
    )


def test_log_positional_base_matches_R():
    import hea.formula as _F

    d = pl.DataFrame({"x": [2.0, 4, 5, 8]})

    def _col(expr):
        return np.asarray(_F._eval_call(parse("~ " + expr).rhs, d).values).ravel()

    assert np.allclose(_col("log(x, 2)"), [1.0, 2.0, 2.321928, 3.0], atol=1e-6)
    assert np.allclose(_col("log(x, base=2)"), [1.0, 2.0, 2.321928, 3.0], atol=1e-6)
    assert np.allclose(
        _col("log(x)"), [0.693147, 1.386294, 1.609438, 2.079442], atol=1e-6
    )


def test_factor_labels_rename_matches_R():
    d = pl.DataFrame({"f3": ["A", "B", "C", "A", "B", "C"]})

    def _suffixes(expr):
        cols = materialize(expand(parse("~ " + expr)), d).columns
        pre = expr  # the whole call deparse is the column prefix
        return [c[len(pre) :] for c in cols if c != "(Intercept)"]

    assert _suffixes('factor(f3, labels = c("L1", "L2", "L3"))') == ["L2", "L3"]
    assert _suffixes(
        'factor(f3, levels = c("C", "B", "A"), labels = c("z1", "z2", "z3"))'
    ) == ["z2", "z3"]
    assert _suffixes('factor(f3, labels = "g")') == ["g2", "g3"]
    X = materialize(expand(parse('~ factor(f3, labels = c("L1", "L2", "L3"))')), d)
    assert X.to_numpy()[:, 1].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_bracket_response_is_cbind_alias():
    assert parse("[y1, y2] ~ x") == parse("cbind(y1, y2) ~ x")
    lhs = parse("[y1, y2] ~ x").lhs
    assert isinstance(lhs, Call) and lhs.fn == "cbind" and not lhs.kwargs
    assert parse("[log(a), b] ~ x") == parse("cbind(log(a), b) ~ x")
    assert parse("[succ, fail] ~ s(x)") == parse("cbind(succ, fail) ~ s(x)")


def test_bracket_response_deparses_to_cbind():
    assert deparse(parse("[y1, y2] ~ x").lhs) == "cbind(y1, y2)"
    assert deparse(parse("[log(a), b] ~ x").lhs) == "cbind(log(a), b)"


def test_bracket_single_element_stays_cbind():
    lhs = parse("[y1] ~ x").lhs
    assert isinstance(lhs, Call) and lhs.fn == "cbind"
    assert lhs == parse("cbind(y1) ~ x").lhs


def test_bracket_does_not_disturb_subscript():
    assert isinstance(parse("y[1] ~ x").lhs, Subscript)
    assert isinstance(parse("y ~ x[1]").rhs, Subscript)
    assert isinstance(parse("y ~ a[b]").rhs, Subscript)


@pytest.mark.parametrize("formula", ["[] ~ x", "[a,] ~ x", "[a,,b] ~ x", "[,a] ~ x"])
def test_bracket_response_rejects_empty_and_trailing(formula):
    with pytest.raises(ParseError):
        parse(formula)
