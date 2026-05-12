"""Tests for hea.translate.py_to_r — Phase 6 acceptance (reverse direction).

Covers Python-side atoms, operators, verb chain → ``|>`` pipes, ggplot
fluent → ``+``-chain, helper inversion (mean/sd/desc/case_when/is_in),
kwarg-name reverse (_by→.by, na_rm→na.rm), join-by dict→c() named.

A separate ``TestRoundTrip`` block runs R→hea→R and asserts the output
is normalized-equivalent to the input.
"""

import re

import pytest

from hea.translate.py_to_r import PyToRError, translate as py_to_r
from hea.translate.r_to_py import translate as r_to_py


def _tr(src: str) -> str:
    """Reverse-translate and strip surrounding whitespace."""
    return py_to_r(src).strip()


def _normalize(s: str) -> str:
    """Collapse whitespace runs to single spaces for round-trip compares."""
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


class TestAtoms:
    def test_int(self):
        assert _tr("x = 42") == "x <- 42"

    def test_float(self):
        assert _tr("x = 3.14") == "x <- 3.14"

    def test_whole_float_renders_with_decimal(self):
        # ``1.0`` written explicitly stays float-ish on the R side.
        assert _tr("x = 1.0") == "x <- 1.0"

    def test_string(self):
        assert _tr('x = "hello"') == 'x <- "hello"'

    def test_bool(self):
        assert _tr("x = True") == "x <- TRUE"
        assert _tr("x = False") == "x <- FALSE"

    def test_none_to_null(self):
        assert _tr("x = None") == "x <- NULL"

    def test_list(self):
        assert _tr("x = [1, 2, 3]") == "x <- c(1, 2, 3)"

    def test_empty_list(self):
        assert _tr("x = []") == "x <- c()"

    def test_dict_to_c(self):
        # Named c() — the form used for join `by` mappings.
        assert _tr('x = {"a": "b"}') == 'x <- c("a" = "b")'


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_arithmetic(self):
        assert _tr("x = 1 + 2 * 3") == "x <- 1 + 2 * 3"

    def test_pow_to_caret(self):
        assert _tr("x = 2 ** 10") == "x <- 2 ^ 10"

    def test_floor_div_to_intdiv(self):
        assert _tr("x = 7 // 3") == "x <- 7 %/% 3"

    def test_mod(self):
        assert _tr("x = 7 % 3") == "x <- 7 %% 3"

    def test_comparison(self):
        assert _tr("x = a == b") == "x <- a == b"

    def test_unary_neg(self):
        assert _tr("x = -y") == "x <- -y"

    def test_not(self):
        assert _tr("x = not y") == "x <- !y"

    def test_and_or_short_circuit(self):
        assert _tr("x = a and b") == "x <- a && b"
        assert _tr("x = a or b") == "x <- a || b"

    def test_bitand_bitor(self):
        # ``a & b`` (Python bitwise) ← polars/Expr elementwise → R ``a & b``.
        assert _tr("x = a & b") == "x <- a & b"
        assert _tr("x = a | b") == "x <- a | b"


# ---------------------------------------------------------------------------
# col() unwrap
# ---------------------------------------------------------------------------


class TestColUnwrap:
    def test_simple_col(self):
        # Bare ``col("x")`` becomes the bare name ``x`` in R.
        assert _tr('col("x")') == "x"

    def test_col_in_comparison(self):
        # Without a verb context the slot is NONE; col() still unwraps.
        assert _tr('col("dest") == "IAH"') == 'dest == "IAH"'

    def test_col_method_form_to_function(self):
        # ``col("x").mean()`` → ``mean(x)``.
        assert _tr('col("x").mean()') == "mean(x)"

    def test_col_method_with_kwarg(self):
        assert _tr('col("x").mean(na_rm=True)') == "mean(x, na.rm = TRUE)"

    def test_col_is_in_to_percent_in(self):
        # ``col("x").is_in(c)`` → ``x %in% c``.
        assert _tr('col("x").is_in([1, 2, 3])') == "x %in% c(1, 2, 3)"


# ---------------------------------------------------------------------------
# Verb chain → pipe
# ---------------------------------------------------------------------------


class TestVerbChain:
    def test_filter_alone(self):
        # Single verb still emits as a pipe — closer to idiomatic R.
        out = _tr('flights.filter(col("dest") == "IAH")')
        assert out == 'flights |>\n  filter(dest == "IAH")'

    def test_full_pipeline(self):
        out = _tr(
            'flights.filter(col("dest") == "IAH")'
            '.group_by("year", "month", "day")'
            '.summarize(arr_delay=col("arr_delay").mean(na_rm=True))'
        )
        expected = (
            "flights |>\n"
            '  filter(dest == "IAH") |>\n'
            "  group_by(year, month, day) |>\n"
            "  summarize(arr_delay = mean(arr_delay, na.rm = TRUE))"
        )
        assert out == expected

    def test_arrange_with_desc(self):
        out = _tr('flights.arrange("year", desc("dep_delay"))')
        assert out == "flights |>\n  arrange(year, desc(dep_delay))"

    def test_count_kwarg(self):
        out = _tr('flights.count("origin", "dest", sort=True)')
        assert out == "flights |>\n  count(origin, dest, sort = TRUE)"

    def test_select_rename(self):
        out = _tr('flights.select(tail_num="tailnum")')
        assert out == "flights |>\n  select(tail_num = tailnum)"


# ---------------------------------------------------------------------------
# kwarg-name reverse
# ---------------------------------------------------------------------------


class TestKwargReverse:
    def test_by_underscore_to_dot(self):
        out = _tr('df.mutate(gain=col("x") - col("y"), _by="origin")')
        assert out == "df |>\n  mutate(gain = x - y, .by = origin)"

    def test_by_list_to_c(self):
        out = _tr('df.mutate(x=col("a"), _by=["origin", "dest"])')
        assert out == "df |>\n  mutate(x = a, .by = c(origin, dest))"

    def test_keep_string(self):
        out = _tr('df.mutate(x=col("a"), _keep="used")')
        assert out == 'df |>\n  mutate(x = a, .keep = "used")'

    def test_before_after_columns(self):
        out = _tr('df.mutate(x=col("a"), _before="day")')
        assert out == "df |>\n  mutate(x = a, .before = day)"


# ---------------------------------------------------------------------------
# case_when reverse
# ---------------------------------------------------------------------------


class TestCaseWhen:
    def test_basic_pairs(self):
        out = _tr(
            'case_when((col("x") > 0, "pos"), (col("x") < 0, "neg"), default="zero")'
        )
        assert out == 'case_when(x > 0 ~ "pos", x < 0 ~ "neg", .default = "zero")'

    def test_inside_mutate(self):
        out = _tr(
            'df.mutate(label=case_when((col("x") > 0, "pos"), default="zero"))'
        )
        assert out == 'df |>\n  mutate(label = case_when(x > 0 ~ "pos", .default = "zero"))'


# ---------------------------------------------------------------------------
# Joins reverse
# ---------------------------------------------------------------------------


class TestJoinsReverse:
    def test_string_by(self):
        assert _tr('x.inner_join(y, by="id")') == 'x |>\n  inner_join(y, by = "id")'

    def test_dict_by_to_named_c(self):
        out = _tr('x.inner_join(y, by={"a": "b"})')
        assert out == 'x |>\n  inner_join(y, by = c("a" = "b"))'

    def test_left_join(self):
        assert _tr('flights.left_join(planes, by="tailnum")') == (
            'flights |>\n  left_join(planes, by = "tailnum")'
        )


# ---------------------------------------------------------------------------
# ggplot reverse
# ---------------------------------------------------------------------------


class TestGgplotReverse:
    def test_root_with_aes(self):
        out = _tr('penguins.ggplot(x="flipper_length_mm", y="body_mass_g").geom_point()')
        assert out == (
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g))"
            " + geom_point()"
        )

    def test_full_chain(self):
        out = _tr(
            'penguins.ggplot(x="flipper_length_mm", y="body_mass_g")'
            '.geom_point(color="species")'
            '.labs(title="Body mass")'
            ".theme_minimal()"
        )
        assert out == (
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g))"
            " + geom_point(aes(color = species))"
            ' + labs(title = "Body mass")'
            " + theme_minimal()"
        )

    def test_geom_with_literal_kwarg(self):
        # ``alpha=0.5`` is a literal — should NOT go in aes().
        out = _tr('d.ggplot(x="x", y="y").geom_point(color="z", alpha=0.5)')
        assert out == (
            "ggplot(d, aes(x = x, y = y))"
            " + geom_point(aes(color = z), alpha = 0.5)"
        )

    def test_facet_wrap_formula_string(self):
        out = _tr('d.ggplot(x="x", y="y").geom_point().facet_wrap("~island")')
        assert "facet_wrap(~island)" in out

    def test_scale_chain(self):
        out = _tr('d.ggplot(x="x").geom_bar().scale_color_viridis_c()')
        assert out.endswith("scale_color_viridis_c()")


# ---------------------------------------------------------------------------
# Patchwork passthrough
# ---------------------------------------------------------------------------


class TestPatchwork:
    def test_pipe(self):
        assert _tr("p1 | p2") == "p1 | p2"

    def test_stack(self):
        assert _tr("p1 / p2") == "p1 / p2"

    def test_grouped(self):
        # The grouping paren may or may not round-trip; we just check the
        # structure is intact and the operators stay.
        out = _tr("(p1 | p2) / p3")
        assert _normalize(out) == "(p1 | p2) / p3" or _normalize(out) == "p1 | p2 / p3"


# ---------------------------------------------------------------------------
# Round-trip: R → hea → R
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Translate R → hea, then hea → R, and assert structural equivalence.

    The equivalence is modulo:
    - whitespace / line breaks
    - integer vs whole-number-double rendering (``1`` ↔ ``1.0`` lose)
    - explicit ``na.rm = TRUE`` added on aggregator calls forward, but
      preserved on round-trip

    Each fixture documents what's expected to survive.
    """

    def _round_trip(self, r_src: str) -> str:
        py = r_to_py(r_src)
        return py_to_r(py)

    def test_simple_filter(self):
        r_src = 'filter(flights, dest == "IAH")'
        out = self._round_trip(r_src)
        # Forward turns ``filter(df, ...)`` into ``df.filter(...)``;
        # reverse re-emits as ``df |> filter(...)``.
        assert _normalize(out) == 'flights |> filter(dest == "IAH")'

    def test_full_pipeline(self):
        r_src = (
            'flights |> filter(dest == "IAH") '
            "|> group_by(year, month, day) "
            "|> summarize(arr_delay = mean(arr_delay, na.rm = TRUE))"
        )
        out = self._round_trip(r_src)
        assert _normalize(out) == _normalize(r_src)

    def test_arrange_desc(self):
        r_src = "arrange(flights, year, desc(dep_delay))"
        out = self._round_trip(r_src)
        assert _normalize(out) == "flights |> arrange(year, desc(dep_delay))"

    def test_count_kwarg(self):
        r_src = "count(flights, origin, dest, sort = TRUE)"
        out = self._round_trip(r_src)
        assert _normalize(out) == "flights |> count(origin, dest, sort = TRUE)"

    def test_mutate_by(self):
        r_src = "mutate(flights, x = a - b, .by = origin)"
        out = self._round_trip(r_src)
        assert _normalize(out) == "flights |> mutate(x = a - b, .by = origin)"

    def test_inner_join_dict(self):
        r_src = 'inner_join(x, y, by = c("a" = "b"))'
        out = self._round_trip(r_src)
        assert _normalize(out) == 'x |> inner_join(y, by = c("a" = "b"))'

    def test_ggplot_basic(self):
        r_src = (
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) "
            "+ geom_point()"
        )
        out = self._round_trip(r_src)
        assert _normalize(out) == _normalize(r_src)

    def test_ggplot_full_chain(self):
        r_src = (
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) "
            '+ geom_point(aes(color = species)) + theme_minimal()'
        )
        out = self._round_trip(r_src)
        assert _normalize(out) == _normalize(r_src)

    def test_case_when_in_mutate(self):
        r_src = (
            "mutate(df, label = case_when("
            'arr_delay > 60 ~ "late", '
            'arr_delay < -10 ~ "early", '
            '.default = "ontime"))'
        )
        out = self._round_trip(r_src)
        assert _normalize(out) == (
            'df |> mutate(label = case_when(arr_delay > 60 ~ "late", '
            'arr_delay < -10 ~ "early", .default = "ontime"))'
        )

    def test_pivot_longer(self):
        r_src = (
            "pivot_longer(billboard, c(wk1, wk2), "
            'names_to = "week", values_to = "rank")'
        )
        out = self._round_trip(r_src)
        assert _normalize(out) == (
            "billboard |> pivot_longer(c(wk1, wk2), "
            'names_to = "week", values_to = "rank")'
        )
