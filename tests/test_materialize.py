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
from hea.formula import expand, materialize, parse, prepare_design, referenced_columns

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
#   R> for (s in c("y~a+b:a","y~z+x:z","y~c+a:b:c","y~x:z+a:z"))
#   R>   print(attr(terms(as.formula(s)),"term.labels"))
#   "a" "a:b"   |   "z" "z:x"   |   "c" "c:a:b"   |   "x:z" "z:a"
@pytest.mark.parametrize("formula,want", [
    ("y ~ a + b:a", ["a", "a:b"]),
    ("y ~ z + x:z", ["z", "z:x"]),
    ("y ~ c + a:b:c", ["c", "c:a:b"]),
    ("y ~ x:z + a:z", ["x:z", "z:a"]),
    ("y ~ a:c + c:b", ["a:c", "c:b"]),
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
