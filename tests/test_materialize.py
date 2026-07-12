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
    # Zero-column X (e.g. `y ~ 0`) lands as either an empty file or a
    # file of bare newlines with no header.
    if not head or not head[0].strip() or "," not in head[0] and '"' not in head[0]:
        return pl.DataFrame()
    # infer_schema_length=0 forces all columns to Float64 — X fixtures are
    # always numeric, and the default-100 inference picks i64 for columns
    # whose first rows happen to be integral and then chokes on later floats.
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

    # polars collapses a 0-column frame to (0, 0) — treat that as the
    # R-equivalent zero-column case for this assertion.
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
    # NA-omit must see smooth-only variables; otherwise prepare_design and
    # materialize_smooths disagree on row count and gam() raises a concat
    # ValueError. Regression: pisa's `s(Income)` had 3 rows with non-NA
    # Overall but NA Income, breaking n=57 vs 54.
    ef = expand(parse("y ~ x + s(z, bs='cr') + te(u, v, by=g)"))
    assert referenced_columns(ef) >= {"x", "z", "u", "v", "g"}


def test_prepare_design_drops_na_on_smooth_only_var():
    df = pl.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "z": [0.1, None, 0.3, 0.4, None],
    })
    d = prepare_design("y ~ s(z)", df)
    assert d.data.height == 3
    assert d.X.height == 3
    assert d.y.len() == 3


# ---------------------------------------------------------------------------
# stats::model.matrix / terms parity regressions. Reference values are from
# live R 4.6.0; the generating R script is in the comment above each block
# (CI has no R — values are hardcoded).
# ---------------------------------------------------------------------------

# Within-interaction variable order follows R's `variables`-attribute (global
# first-appearance) order, not the order written inside the term.
#   R> for (s in c("y~a+b:a","y~z+x:z","y~c+a:b:c","y~x:z+a:z",
#   R>             "y~(x1+x2):f2","y~x2:f2+x1:f2"))
#   R>   print(attr(terms(as.formula(s)),"term.labels"))
#   "a" "a:b" | "z" "z:x" | "c" "c:a:b" | "x:z" "z:a"
#   | "x1:f2" "x2:f2" | "x2:f2" "f2:x1"
# The order comes from the SOURCE RHS (R's `variables` attr), not the expanded
# terms: `(x1+x2):f2` emits `x1:f2` first but x2 precedes f2 in the source, so
# `x2:f2` (not `f2:x2`). This case distinguishes a source-walk from the (wrong)
# expanded-term iteration.
@pytest.mark.parametrize("formula,want", [
    ("y ~ a + b:a", ["a", "a:b"]),
    ("y ~ z + x:z", ["z", "z:x"]),
    ("y ~ c + a:b:c", ["c", "c:a:b"]),
    ("y ~ x:z + a:z", ["x:z", "z:a"]),
    ("y ~ a:c + c:b", ["a:c", "c:b"]),
    ("y ~ (x1 + x2):f2", ["x1:f2", "x2:f2"]),
    ("y ~ x2:f2 + x1:f2", ["x2:f2", "f2:x1"]),
])
def test_within_term_var_order_matches_R(formula, want):
    assert expand(parse(formula)).term_labels == want


# The within-term order propagates to model.matrix column names/order.
#   R> set.seed(1); d <- data.frame(
#   R>   a=factor(rep(c("A1","A2"),length.out=12)),
#   R>   b=factor(rep(c("B1","B2","B3"),length.out=12)),
#   R>   x=rnorm(12), z=rnorm(12))
#   R> colnames(model.matrix(~ b + a:b, d))
#   "(Intercept)" "bB2" "bB3" "bB1:aA2" "bB2:aA2" "bB3:aA2"
#   R> colnames(model.matrix(~ z + x:z, d))  ->  "(Intercept)" "z" "z:x"
def test_within_term_order_model_matrix_columns():
    d = pl.DataFrame({
        "a": pl.Series([["A1", "A2"][i % 2] for i in range(12)],
                       dtype=pl.Enum(["A1", "A2"])),
        "b": pl.Series([["B1", "B2", "B3"][i % 3] for i in range(12)],
                       dtype=pl.Enum(["B1", "B2", "B3"])),
        "x": np.arange(12, dtype=float), "z": np.arange(12, dtype=float) + 0.5,
    })
    assert materialize(expand(parse("~ b + a:b")), d).columns == [
        "(Intercept)", "bB2", "bB3", "bB1:aA2", "bB2:aA2", "bB3:aA2"]
    assert materialize(expand(parse("~ z + x:z")), d).columns == [
        "(Intercept)", "z", "z:x"]
    # Distributive case: source order x1,x2,f2 → x2:f2 keeps x2 first.
    #   R> colnames(model.matrix(~ (x1+x2):f2, d2))
    #   "(Intercept)" "x1:f2P" "x1:f2Q" "x2:f2P" "x2:f2Q"
    d2 = pl.DataFrame({"x1": np.arange(6, dtype=float),
                       "x2": np.arange(6, dtype=float) + 0.5,
                       "f2": pl.Series(["P", "Q"] * 3, dtype=pl.Enum(["P", "Q"]))})
    assert materialize(expand(parse("~ (x1+x2):f2")), d2).columns == [
        "(Intercept)", "x1:f2P", "x1:f2Q", "x2:f2P", "x2:f2Q"]


# No-intercept / covered-margin factor coding. R reduces a factor once an
# earlier full-coded factor covers its margin; the result is full-rank, not the
# both-factors-full (rank-deficient) coding.
#   R> d <- data.frame(
#   R>   a=factor(rep(c("A1","A2"),length.out=12)),
#   R>   b=factor(rep(c("B1","B2","B3"),length.out=12)),
#   R>   c=factor(rep(c("C1","C2"),length.out=12)))
#   R> colnames(model.matrix(~ a + b - 1, d))
#   "aA1" "aA2" "bB2" "bB3"
#   R> colnames(model.matrix(~ 0 + b + a, d))      -> "bB1" "bB2" "bB3" "aA2"
#   R> colnames(model.matrix(~ 0 + a + b + c, d))  -> "aA1" "aA2" "bB2" "bB3" "cC2"
#   R> colnames(model.matrix(~ a:b + b:c - 1, d))  -> 6×(a:b full) + "bB1:cC2" "bB2:cC2" "bB3:cC2"
def test_no_intercept_factor_coding_matches_R():
    d = pl.DataFrame({
        "a": pl.Series([["A1", "A2"][i % 2] for i in range(12)], dtype=pl.Enum(["A1", "A2"])),
        "b": pl.Series([["B1", "B2", "B3"][i % 3] for i in range(12)], dtype=pl.Enum(["B1", "B2", "B3"])),
        "c": pl.Series([["C1", "C2"][i % 2] for i in range(12)], dtype=pl.Enum(["C1", "C2"])),
    })
    assert materialize(expand(parse("~ a + b - 1")), d).columns == ["aA1", "aA2", "bB2", "bB3"]
    assert materialize(expand(parse("~ 0 + b + a")), d).columns == ["bB1", "bB2", "bB3", "aA2"]
    assert materialize(expand(parse("~ 0 + a + b + c")), d).columns == ["aA1", "aA2", "bB2", "bB3", "cC2"]
    assert materialize(expand(parse("~ a:b + b:c - 1")), d).columns == [
        "aA1:bB1", "aA2:bB1", "aA1:bB2", "aA2:bB2", "aA1:bB3", "aA2:bB3",
        "bB1:cC2", "bB2:cC2", "bB3:cC2"]
    # no-intercept multi-factor X must be full rank (was rank-deficient pre-F3)
    X = materialize(expand(parse("~ a + b - 1")), d).to_numpy()
    assert np.linalg.matrix_rank(X) == X.shape[1]


# A logical column is a 2-level factor FALSE < TRUE (not a 0/1 numeric).
#   R> d <- data.frame(l=c(TRUE,FALSE,TRUE,TRUE,FALSE,FALSE), x=c(1,2,3,4,5,6))
#   R> colnames(model.matrix(~ l,     d))  -> "(Intercept)" "lTRUE"
#   R> colnames(model.matrix(~ 0 + l, d))  -> "lFALSE" "lTRUE"
#   R> colnames(model.matrix(~ l:x,   d))  -> "(Intercept)" "lFALSE:x" "lTRUE:x"
#   R> colnames(model.matrix(~ l*x,   d))  -> "(Intercept)" "lTRUE" "x" "lTRUE:x"
#   R> model.matrix(~ l, d)[,2]            -> 1 0 1 1 0 0
def test_logical_is_two_level_factor_matches_R():
    d = pl.DataFrame({"l": [True, False, True, True, False, False],
                      "x": [1.0, 2, 3, 4, 5, 6]})
    assert materialize(expand(parse("~ l")), d).columns == ["(Intercept)", "lTRUE"]
    assert materialize(expand(parse("~ 0 + l")), d).columns == ["lFALSE", "lTRUE"]
    assert materialize(expand(parse("~ l:x")), d).columns == ["(Intercept)", "lFALSE:x", "lTRUE:x"]
    assert materialize(expand(parse("~ l*x")), d).columns == ["(Intercept)", "lTRUE", "x", "lTRUE:x"]
    assert materialize(expand(parse("~ l")), d)["lTRUE"].to_list() == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]


# A *computed* logical expression (`I(x>0)`, bare `x>0`, `!flag`) is also a
# 2-level factor FALSE<TRUE in R, not a 0/1 numeric — so names always carry the
# `TRUE`/`FALSE` suffix and no-intercept/interaction contexts emit two columns.
#   R> d <- data.frame(x1=c(-1,2,-3,4,0.5,-0.5), f2=factor(c("P","Q","P","Q","P","Q")))
#   R> colnames(model.matrix(~ I(x1>0),     d)) -> "(Intercept)" "I(x1 > 0)TRUE"
#   R> colnames(model.matrix(~ I(x1>0) - 1, d)) -> "I(x1 > 0)FALSE" "I(x1 > 0)TRUE"
#   R> colnames(model.matrix(~ x1>0,        d)) -> "(Intercept)" "x1 > 0TRUE"
#   R> colnames(model.matrix(~ I(x1>0):f2,  d)) -> "(Intercept)" + 4 full×full cols
#   R> colnames(model.matrix(~ I(x1>0)*f2,  d)) -> "(Intercept)" "I(x1 > 0)TRUE" "f2Q" "I(x1 > 0)TRUE:f2Q"
#   R> model.matrix(~ I(x1>0), d)[,2]           -> 0 1 0 1 1 0
def test_logical_expression_is_two_level_factor_matches_R():
    d = pl.DataFrame({"x1": [-1.0, 2, -3, 4, 0.5, -0.5],
                      "f2": pl.Series(["P", "Q", "P", "Q", "P", "Q"],
                                      dtype=pl.Enum(["P", "Q"]))})
    assert materialize(expand(parse("~ I(x1>0)")), d).columns == [
        "(Intercept)", "I(x1 > 0)TRUE"]
    assert materialize(expand(parse("~ I(x1>0) - 1")), d).columns == [
        "I(x1 > 0)FALSE", "I(x1 > 0)TRUE"]
    assert materialize(expand(parse("~ x1>0")), d).columns == [
        "(Intercept)", "x1 > 0TRUE"]
    assert materialize(expand(parse("~ I(x1>0):f2")), d).columns == [
        "(Intercept)", "I(x1 > 0)FALSE:f2P", "I(x1 > 0)TRUE:f2P",
        "I(x1 > 0)FALSE:f2Q", "I(x1 > 0)TRUE:f2Q"]
    assert materialize(expand(parse("~ I(x1>0)*f2")), d).columns == [
        "(Intercept)", "I(x1 > 0)TRUE", "f2Q", "I(x1 > 0)TRUE:f2Q"]
    assert materialize(expand(parse("~ I(x1>0)")), d)["I(x1 > 0)TRUE"].to_list() == [
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    # arithmetic on a logical stays numeric (R coerces inside `I((x>0)+1)`)
    assert materialize(expand(parse("~ I((x1>0) + 1)")), d).columns == [
        "(Intercept)", "I((x1 > 0) + 1)"]


# cut(): R seeds interior breaks at seq(min,max) (only the 2 endpoints are
# widened by dx/1000), and labels use formatC(digits=dig.lab=3, format="g").
#   R> x <- c(-3,-1,0,1.5,3,5.2,7,9)
#   R> levels(cut(x,3))                -> "(-3.01,1]" "(1,5]" "(5,9.01]"
#   R> as.integer(cut(x,3))            -> 1 1 1 2 2 3 3 3
#   R> levels(cut(x,4,dig.lab=4))      -> "(-3.012,0]" "(0,3]" "(3,6]" "(6,9.012]"
#   R> levels(cut(x,c(-5,0,5,10),right=FALSE)) -> "[-5,0)" "[0,5)" "[5,10)"
#   R> as.integer(cut(x,c(-5,0,5,10),right=FALSE)) -> 1 1 2 2 2 3 3 3
def test_cut_breaks_and_diglab_labels_match_R():
    import hea.formula as _F
    d = pl.DataFrame({"x": [-3.0, -1, 0, 1.5, 3, 5.2, 7, 9]})

    def _cut(expr):
        blk = _F._eval_call(parse("~ " + expr).rhs, d)
        codes = [int(c) + 1 for c in blk.codes]   # R's 1-based integer codes
        return blk.levels, codes

    lev, codes = _cut("cut(x,3)")
    assert lev == ["(-3.01,1]", "(1,5]", "(5,9.01]"]
    assert codes == [1, 1, 1, 2, 2, 3, 3, 3]
    lev, _ = _cut("cut(x,4,dig.lab=4)")
    assert lev == ["(-3.012,0]", "(0,3]", "(3,6]", "(6,9.012]"]
    lev, codes = _cut("cut(x,c(-5,0,5,10),right=FALSE)")
    assert lev == ["[-5,0)", "[0,5)", "[5,10)"]
    assert codes == [1, 1, 2, 2, 2, 3, 3, 3]


# contrasts.arg as a custom matrix, and options(contrasts) default.
#   R> d <- data.frame(f=factor(c("a","b","c","a","b","c")))
#   R> M <- matrix(c(1,-1,0, 0,1,-1), 3, 2)
#   R> colnames(model.matrix(~ f,     d, contrasts.arg=list(f=M))) -> "(Intercept)" "f1" "f2"
#   R>          model.matrix(~ f,     d, contrasts.arg=list(f=M))[1:3,2:3] -> (1,0)(-1,1)(0,-1)
#   R> colnames(model.matrix(~ 0 + f, d, contrasts.arg=list(f=M))) -> "fa" "fb" "fc"  (full)
#   R> colnames(model.matrix(~ f,     d, contrasts.arg=list(f=Mn))) named lo/hi -> "flo" "fhi"
#   R> options(contrasts=c("contr.sum","contr.poly")); colnames(model.matrix(~ f,d)) -> "f1" "f2"
#   R>   contr.sum rows: a=(1,0) b=(0,1) c=(-1,-1)
def test_contrasts_arg_matrix_and_options_default():
    from hea.formula import with_contrasts, with_default_contrasts
    d = pl.DataFrame({"f": pl.Series(["a", "b", "c", "a", "b", "c"],
                                     dtype=pl.Enum(["a", "b", "c"]))})
    M = np.array([[1.0, 0], [-1, 1], [0, -1]])   # k×(k-1)
    with with_contrasts({"f": M}):
        X = materialize(expand(parse("~ f")), d)
        assert X.columns == ["(Intercept)", "f1", "f2"]
        np.testing.assert_allclose(
            X.select(["f1", "f2"]).head(3).to_numpy(),
            [[1, 0], [-1, 1], [0, -1]])
        # promoted (no intercept) → full indicator coding, custom matrix ignored
        assert materialize(expand(parse("~ 0 + f")), d).columns == ["fa", "fb", "fc"]
    with with_contrasts({"f": (M, ["lo", "hi"])}):
        assert materialize(expand(parse("~ f")), d).columns == ["(Intercept)", "flo", "fhi"]
    with with_default_contrasts("contr.sum", "contr.poly"):
        X = materialize(expand(parse("~ f")), d)
        assert X.columns == ["(Intercept)", "f1", "f2"]
        np.testing.assert_allclose(
            X.select(["f1", "f2"]).head(3).to_numpy(),
            [[1, 0], [0, 1], [-1, -1]])           # contr.sum
    # default restored outside the block
    assert materialize(expand(parse("~ f")), d).columns == ["(Intercept)", "fb", "fc"]


# na_action on prepare_design (R's na.action): omit drops NA rows, pass keeps
# them (NaN flows into X, X and y stay the same length), fail raises.
def test_prepare_design_na_action():
    d = pl.DataFrame({"y": [1.0, 2, None, 4, 5], "x": [1.0, 2, 3, None, 5]})
    # omit (default): drop the two NA rows
    assert prepare_design("y ~ x", d).X.height == 3
    assert prepare_design("y ~ x", d, na_action="exclude").X.height == 3
    # pass: keep all rows, X and y aligned, X carries NaN
    des = prepare_design("y ~ x", d, na_action="pass")
    assert des.X.height == 5 and des.y.len() == 5
    assert np.isnan(des.X.to_numpy()).any()
    # fail: raise when any NA is present
    with pytest.raises(ValueError, match="missing values"):
        prepare_design("y ~ x", d, na_action="fail")
    with pytest.raises(ValueError, match="na_action must be"):
        prepare_design("y ~ x", d, na_action="bogus")
    # fail with no NA present is fine
    clean = pl.DataFrame({"y": [1.0, 2, 3], "x": [1.0, 2, 3]})
    assert prepare_design("y ~ x", clean, na_action="fail").X.height == 3


# bs() predict beyond the training Boundary.knots must extrapolate the boundary
# polynomial piece (R's predict.bs), not clamp to 0. Capture the training basis
# state, then replay on new x straddling both boundaries.
#   R> library(splines)
#   R> trn <- c(-1.2,-0.5,0.1,0.3,0.7,0.9,1.1,1.4,1.8,2.0,2.3,2.6)
#   R> bo <- bs(trn, df=5)                       # knots .5667,1.5333; bnd -1.2,2.6
#   R> round(predict(bo, c(-2,-1.5,0.5,3,3.5)), 6)   # 2 below, 1 inside, 2 above
def test_bs_predict_extrapolates_beyond_boundary_matches_R():
    trn = pl.DataFrame({"x1": [-1.2, -0.5, 0.1, 0.3, 0.7, 0.9,
                               1.1, 1.4, 1.8, 2.0, 2.3, 2.6]})
    new = pl.DataFrame({"x1": [-2.0, -1.5, 0.5, 3.0, 3.5]})
    state: dict = {}
    materialize(expand(parse("~ bs(x1, df=5)")), trn, basis_state=state)
    Xn = materialize(expand(parse("~ bs(x1, df=5)")), new, basis_state=state)
    cols = [c for c in Xn.columns if c != "(Intercept)"]
    got = Xn.select(cols).to_numpy()
    want = np.array([
        [-2.562925,  0.524316, -0.027902,  0.000000,  0.000000],
        [-0.661962,  0.062595, -0.001471,  0.000000,  0.000000],
        [ 0.152626,  0.579579,  0.267742,  0.000000,  0.000000],
        [ 0.000000, -0.007765,  0.271253, -1.863097,  2.599609],
        [ 0.000000, -0.088452,  1.689254, -6.868472,  6.267670],
    ])
    assert np.allclose(got, want, atol=1e-6)


# scale(x, center=FALSE) divides by the root-mean-square sqrt(sum(x^2)/(n-1)),
# NOT the sd about the mean. (center=TRUE already matched, since RMS-of-centered
# == sd.)
#   R> x <- c(2,4,4,4,5,5,7,9)
#   R> as.vector(scale(x, center=FALSE))  # divisor sqrt(sum(x^2)/7) = 5.756983
#   0.347404 0.694808 0.694808 0.694808 0.868510 0.868510 1.215915 1.563319
#   R> as.vector(scale(x))                # default center=TRUE (sd divisor)
#  -1.403122 -0.467707 -0.467707 -0.467707 0.000000 0.000000 0.935414 1.870829
def test_scale_center_false_uses_rms_matches_R():
    import hea.formula as _F
    d = pl.DataFrame({"x": [2.0, 4, 4, 4, 5, 5, 7, 9]})

    def _col(expr):
        return np.asarray(_F._eval_call(parse("~ " + expr).rhs, d).values).ravel()

    assert np.allclose(_col("scale(x, center=FALSE)"), [
        0.347404, 0.694808, 0.694808, 0.694808,
        0.868510, 0.868510, 1.215915, 1.563319], atol=1e-6)
    assert np.allclose(_col("scale(x)"), [
        -1.403122, -0.467707, -0.467707, -0.467707,
        0.0, 0.0, 0.935414, 1.870829], atol=1e-6)


# R's log(x, base): the base may be the 2nd positional arg, not only `base=`.
#   R> x <- c(2,4,5,8); log(x, 2)  ->  1.000000 2.000000 2.321928 3.000000
#   R> log(x)  ->  0.6931472 1.3862944 1.6094379 2.0794415
def test_log_positional_base_matches_R():
    import hea.formula as _F
    d = pl.DataFrame({"x": [2.0, 4, 5, 8]})

    def _col(expr):
        return np.asarray(_F._eval_call(parse("~ " + expr).rhs, d).values).ravel()

    assert np.allclose(_col("log(x, 2)"), [1.0, 2.0, 2.321928, 3.0], atol=1e-6)
    assert np.allclose(_col("log(x, base=2)"), [1.0, 2.0, 2.321928, 3.0], atol=1e-6)
    assert np.allclose(_col("log(x)"),
                       [0.693147, 1.386294, 1.609438, 2.079442], atol=1e-6)


# factor(x, labels=) renames levels (drives column suffixes); a single label is
# used as a prefix. levels= still recodes against the given order.
#   R> d <- data.frame(f3=factor(c("A","B","C","A","B","C")))
#   R> colnames(model.matrix(~ factor(f3, labels=c("L1","L2","L3")), d))
#   "(Intercept)" "...L2" "...L3"
#   R> colnames(model.matrix(~ factor(f3, levels=c("C","B","A"),
#   R>                               labels=c("z1","z2","z3")), d))  -> "...z2" "...z3"
#   R> colnames(model.matrix(~ factor(f3, labels="g"), d))           -> "...g2" "...g3"
#   R> model.matrix(~ factor(f3, labels=c("L1","L2","L3")), d)[,2]   -> 0 1 0 0 1 0
def test_factor_labels_rename_matches_R():
    d = pl.DataFrame({"f3": ["A", "B", "C", "A", "B", "C"]})

    def _suffixes(expr):
        cols = materialize(expand(parse("~ " + expr)), d).columns
        pre = expr  # the whole call deparse is the column prefix
        return [c[len(pre):] for c in cols if c != "(Intercept)"]

    assert _suffixes("factor(f3, labels = c(\"L1\", \"L2\", \"L3\"))") == ["L2", "L3"]
    assert _suffixes("factor(f3, levels = c(\"C\", \"B\", \"A\"), "
                     "labels = c(\"z1\", \"z2\", \"z3\"))") == ["z2", "z3"]
    assert _suffixes("factor(f3, labels = \"g\")") == ["g2", "g3"]
    X = materialize(expand(parse("~ factor(f3, labels = c(\"L1\", \"L2\", \"L3\"))")), d)
    assert X.to_numpy()[:, 1].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# hea-dialect `[y1, y2]` sugar — a pure alias for `cbind(y1, y2)` lowered at
# parse time. `cbind` stays canonical (deparse always emits it); brackets are
# an input convenience only.
# ---------------------------------------------------------------------------

def test_bracket_response_is_cbind_alias():
    # Byte-identical AST: `[a, b]` lowers to Call("cbind", [a, b]).
    assert parse("[y1, y2] ~ x") == parse("cbind(y1, y2) ~ x")
    lhs = parse("[y1, y2] ~ x").lhs
    assert isinstance(lhs, Call) and lhs.fn == "cbind" and not lhs.kwargs
    # Expression elements pass through the same `parse_expr` as a call arg list.
    assert parse("[log(a), b] ~ x") == parse("cbind(log(a), b) ~ x")
    # Binomial two-column form is just cbind too — no carve-out.
    assert parse("[succ, fail] ~ s(x)") == parse("cbind(succ, fail) ~ s(x)")


def test_bracket_response_deparses_to_cbind():
    # cbind is canonical: deparse/round-trip never emit `[...]`.
    assert deparse(parse("[y1, y2] ~ x").lhs) == "cbind(y1, y2)"
    assert deparse(parse("[log(a), b] ~ x").lhs) == "cbind(log(a), b)"


def test_bracket_single_element_stays_cbind():
    # Invariant: `[y1]` is `cbind(y1)`, NOT unwrapped to bare `y1`.
    lhs = parse("[y1] ~ x").lhs
    assert isinstance(lhs, Call) and lhs.fn == "cbind"
    assert lhs == parse("cbind(y1) ~ x").lhs


def test_bracket_does_not_disturb_subscript():
    # Postfix `a[i]` needs a preceding operand → still Subscript, both sides.
    assert isinstance(parse("y[1] ~ x").lhs, Subscript)
    assert isinstance(parse("y ~ x[1]").rhs, Subscript)
    assert isinstance(parse("y ~ a[b]").rhs, Subscript)


@pytest.mark.parametrize("formula", ["[] ~ x", "[a,] ~ x", "[a,,b] ~ x", "[,a] ~ x"])
def test_bracket_response_rejects_empty_and_trailing(formula):
    # Responses are never empty: no `[]`, no trailing comma, no empty slot.
    with pytest.raises(ParseError):
        parse(formula)
