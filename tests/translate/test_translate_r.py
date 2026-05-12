"""Tests for hea.translate.r_to_py — Phase 1+2 acceptance.

Each test parses an R snippet, runs the translator, and asserts on the
emitted Python source. We use string equality on the canonical
``ast.unparse`` output (which is stable across runs for a given Python
version). For end-to-end runnability we also ``compile()`` the output
to catch any syntactic regressions.
"""

import ast

import pytest

from hea.translate.r_to_py import translate


def _tr(src: str) -> str:
    """Translate and strip trailing whitespace/newlines for stable compare."""
    return translate(src).strip()


def _compiles(src: str) -> bool:
    """Translated output must be valid Python source."""
    try:
        compile(_tr(src), "<translated>", "exec")
    except SyntaxError:
        return False
    return True


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


class TestAtoms:
    def test_int_literal(self):
        # ``42L`` → Python int 42
        assert _tr("42L") == "42"

    def test_num_literal(self):
        assert _tr("3.14") == "3.14"

    def test_string_literal(self):
        # Wrap with assignment — a bare top-level string would be unparsed
        # as a triple-quoted docstring, which is correct but noisy here.
        assert _tr('x <- "hello"') == "x = 'hello'"

    def test_bool_literal(self):
        assert _tr("TRUE") == "True"
        assert _tr("FALSE") == "False"

    def test_null_literal(self):
        assert _tr("NULL") == "None"

    def test_identifier(self):
        # Bare identifier outside any verb is a Python name.
        assert _tr("x") == "x"

    def test_dotted_identifier(self):
        # ``data.frame`` becomes ``data_frame`` outside verb context.
        assert _tr("data.frame") == "data_frame"


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_arithmetic(self):
        # Whole-number doubles render as ints (matches hand-written hea idiom).
        assert _tr("1 + 2") == "1 + 2"
        assert _tr("2 * 3 + 1") == "2 * 3 + 1"

    def test_fractional_stays_float(self):
        assert _tr("1.5 + 0.5") == "1.5 + 0.5"

    def test_power_to_pow(self):
        # R's ``^`` → Python ``**``.
        assert _tr("2 ^ 10") == "2 ** 10"

    def test_modulo_floor_div(self):
        # R's ``%%`` is just %infix% with value %%; same for %/%.
        assert _tr("7 %% 3") == "7 % 3"
        assert _tr("7 %/% 3") == "7 // 3"

    def test_comparison_emits_ast_compare(self):
        # ``a == b`` → ``a == b`` (single comparison).
        # Outside any slot, ``a`` and ``b`` are Python names.
        assert _tr("a == b") == "a == b"

    def test_unary_neg(self):
        assert _tr("-x") == "-x"

    def test_unary_not_outside_slot(self):
        # ``!x`` outside slot is logical ``not``.
        assert _tr("!x") == "not x"

    def test_sequence_to_hea_seq(self):
        # ``1:5`` → ``hea.seq(1, 5)``.
        assert _tr("1:5") == "hea.seq(1, 5)"


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_left_arrow(self):
        assert _tr("x <- 1") == "x = 1"

    def test_right_arrow_flipped(self):
        # ``1 -> x`` parses as ``x <- 1`` and emits as ``x = 1``.
        assert _tr("1 -> x") == "x = 1"

    def test_eq_assign(self):
        assert _tr("x = 1") == "x = 1"


# ---------------------------------------------------------------------------
# c() to list
# ---------------------------------------------------------------------------


class TestC:
    def test_c_to_list(self):
        assert _tr('c(1, 2, 3)') == "[1, 2, 3]"

    def test_c_of_strings(self):
        assert _tr('c("a", "b")') == "['a', 'b']"


# ---------------------------------------------------------------------------
# Pipe rewriting
# ---------------------------------------------------------------------------


class TestPipes:
    def test_native_pipe_unknown_func(self):
        # ``x |> f(y)`` with unknown f → ``f(x, y)`` (function form).
        assert _tr("x |> f(y)") == "f(x, y)"

    def test_native_pipe_to_verb(self):
        # ``x |> filter(cond)`` becomes ``x.filter(...)`` with NSE on cond.
        # ``cond`` is a bare name inside filter's EXPR slot → col("cond").
        assert _tr('x |> filter(cond)') == "x.filter(col('cond'))"

    def test_magrittr_pipe(self):
        assert _tr('x %>% filter(a == 1)') == "x.filter(col('a') == 1)"

    def test_magrittr_placeholder(self):
        # ``x %>% f(., y)`` → ``f(x, y)`` (unknown f, function form).
        assert _tr("x %>% f(., y)") == "f(x, y)"

    def test_chained_pipe(self):
        # Method chain on the result of a pipe chain.
        out = _tr('x |> filter(a == 1) |> select(b, c)')
        assert out == "x.filter(col('a') == 1).select('b', 'c')"


# ---------------------------------------------------------------------------
# Verb dispatch & NSE
# ---------------------------------------------------------------------------


class TestVerbs:
    def test_filter_expr_slot(self):
        # Inside filter: bare ``dest`` → col("dest"); literal "IAH" stays.
        out = _tr('filter(flights, dest == "IAH")')
        assert out == "flights.filter(col('dest') == 'IAH')"

    def test_filter_and_chain_uses_bitand(self):
        # In EXPR slot, ``&`` becomes bitwise ``&`` (polars-Expr-friendly).
        out = _tr('filter(flights, month == 1 & day == 1)')
        assert out == "flights.filter((col('month') == 1) & (col('day') == 1))"

    def test_select_column_name_slot(self):
        # Bare names in select become strings.
        out = _tr("select(flights, year, month, day)")
        assert out == "flights.select('year', 'month', 'day')"

    def test_select_with_rename(self):
        # Named arg in select: ``new = old`` — value is also a column name.
        out = _tr("select(flights, tail_num = tailnum)")
        assert out == "flights.select(tail_num='tailnum')"

    def test_group_by(self):
        out = _tr("group_by(flights, year, month)")
        assert out == "flights.group_by('year', 'month')"

    def test_arrange_with_desc(self):
        out = _tr("arrange(flights, year, desc(dep_delay))")
        assert out == "flights.arrange('year', desc('dep_delay'))"

    def test_summarize_with_aggregator(self):
        out = _tr("summarize(flights, avg = mean(dep_delay))")
        assert out == "flights.summarize(avg=col('dep_delay').mean())"

    def test_summarize_with_na_rm_kwarg(self):
        out = _tr("summarize(flights, avg = mean(dep_delay, na.rm = TRUE))")
        assert out == "flights.summarize(avg=col('dep_delay').mean(na_rm=True))"

    def test_count(self):
        out = _tr('count(flights, origin, dest, sort = TRUE)')
        assert out == "flights.count('origin', 'dest', sort=True)"

    def test_distinct(self):
        out = _tr("distinct(flights, origin, dest)")
        assert out == "flights.distinct('origin', 'dest')"

    def test_n_helper(self):
        # ``n()`` is a function helper, no NSE on its (empty) args.
        out = _tr("summarize(flights, total = n())")
        assert out == "flights.summarize(total=n())"


# ---------------------------------------------------------------------------
# Helpers — function-form translation
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_starts_with_in_select(self):
        out = _tr('select(flights, starts_with("arr"))')
        assert out == "flights.select(selectors.starts_with('arr'))"

    def test_in_operator(self):
        out = _tr('filter(flights, dest %in% c("IAH", "JFK"))')
        assert out == "flights.filter(col('dest').is_in(['IAH', 'JFK']))"

    def test_unknown_function_passes_through(self):
        out = _tr("filter(flights, my_custom_fn(x) > 0)")
        # Unknown ``my_custom_fn`` is emitted as a function call; the bare
        # ``x`` inside is still NSE-wrapped because filter's slot is active.
        assert out == "flights.filter(my_custom_fn(col('x')) > 0)"


# ---------------------------------------------------------------------------
# End-to-end: canonical r4ds pipeline
# ---------------------------------------------------------------------------


def test_canonical_pipeline_full():
    src = """\
flights |>
  filter(dest == "IAH") |>
  group_by(year, month, day) |>
  summarize(arr_delay = mean(arr_delay, na.rm = TRUE))
"""
    out = _tr(src)
    expected = (
        "flights.filter(col('dest') == 'IAH')"
        ".group_by('year', 'month', 'day')"
        ".summarize(arr_delay=col('arr_delay').mean(na_rm=True))"
    )
    assert out == expected


def test_canonical_pipeline_compiles():
    src = """\
flights |>
  filter(dest == "IAH") |>
  group_by(year, month, day) |>
  summarize(arr_delay = mean(arr_delay, na.rm = TRUE))
"""
    assert _compiles(src)


def test_translate_output_is_valid_python():
    # A grab-bag of inputs — every translated output should parse cleanly.
    inputs = [
        "x <- 1",
        '"hello"',
        "1 + 2 * 3",
        "f(g(h(x)))",
        'flights |> filter(dest == "IAH")',
        "for (i in 1:10) print(i)",
        "if (x > 0) y else z",
        "function(x, y = 1) x + y",
    ]
    for src in inputs:
        assert _compiles(src), f"failed to compile output of: {src!r}"


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


class TestControlFlow:
    def test_if_else_ternary(self):
        out = _tr("if (x > 0) a else b")
        assert out == "a if x > 0 else b"

    def test_for_loop(self):
        out = _tr("for (i in 1:5) print(i)")
        assert "for i in hea.seq(1, 5)" in out

    def test_lambda_shorthand(self):
        out = _tr("f <- \\(x) x + 1")
        # Body bare ``x`` is outside any verb slot (Slot.NONE) — emits as ``x``.
        assert out == "f = lambda x: x + 1"
