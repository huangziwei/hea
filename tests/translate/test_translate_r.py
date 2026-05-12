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


class TestMutateSummarize:
    """Phase 3 — mutate / summarize NSE and dplyr's dot-prefixed kwargs."""

    def test_mutate_sequential_columns(self):
        # Sequential evaluation is hea's runtime concern — the translator
        # just emits both kwargs in order; hea.mutate evaluates them
        # sequentially so the second can see the first.
        out = _tr("mutate(flights, hours = air_time / 60, gain_per_hour = gain / hours)")
        assert out == (
            "flights.mutate(hours=col('air_time') / 60, "
            "gain_per_hour=col('gain') / col('hours'))"
        )

    def test_mutate_dot_by_to_underscore_by_as_string(self):
        # ``.by = origin`` — the column name slot, not the EXPR slot.
        out = _tr("mutate(flights, gain = dep_delay - arr_delay, .by = origin)")
        assert out == "flights.mutate(gain=col('dep_delay') - col('arr_delay'), _by='origin')"

    def test_mutate_dot_by_with_c_list(self):
        out = _tr("mutate(flights, gain = dep_delay - arr_delay, .by = c(origin, dest))")
        assert out == (
            "flights.mutate(gain=col('dep_delay') - col('arr_delay'), "
            "_by=['origin', 'dest'])"
        )

    def test_mutate_dot_before_as_column_name(self):
        out = _tr("mutate(flights, gain = dep_delay - arr_delay, .before = day)")
        assert out == "flights.mutate(gain=col('dep_delay') - col('arr_delay'), _before='day')"

    def test_mutate_dot_after_as_column_name(self):
        out = _tr("mutate(flights, speed = distance / air_time, .after = day)")
        assert out == "flights.mutate(speed=col('distance') / col('air_time'), _after='day')"

    def test_mutate_dot_keep_string_literal(self):
        # ``.keep`` takes a literal string — should NOT become col("used").
        out = _tr('mutate(flights, x = a + b, .keep = "used")')
        assert out == "flights.mutate(x=col('a') + col('b'), _keep='used')"

    def test_transmute_auto_injects_keep_none(self):
        out = _tr("transmute(flights, gain = dep_delay - arr_delay)")
        assert out == "flights.mutate(gain=col('dep_delay') - col('arr_delay'), _keep='none')"

    def test_transmute_user_keep_wins(self):
        # Explicit ``.keep`` from the user overrides the auto-injected default.
        out = _tr('transmute(flights, gain = dep_delay - arr_delay, .keep = "all")')
        assert out == "flights.mutate(gain=col('dep_delay') - col('arr_delay'), _keep='all')"

    def test_summarize_dot_by(self):
        out = _tr("summarize(flights, avg = mean(dep_delay), .by = origin)")
        assert out == "flights.summarize(avg=col('dep_delay').mean(), _by='origin')"

    def test_summarise_british_spelling(self):
        out = _tr("summarise(flights, avg = mean(dep_delay))")
        assert out == "flights.summarize(avg=col('dep_delay').mean())"

    def test_mutate_in_pipe(self):
        out = _tr("flights |> mutate(gain = dep_delay - arr_delay, .before = day)")
        assert out == (
            "flights.mutate(gain=col('dep_delay') - col('arr_delay'), _before='day')"
        )

    def test_mutate_with_nested_aggregator(self):
        # ``mutate(rank = min_rank(desc(arr_delay)))`` — desc() takes
        # COLUMN_NAME slot, min_rank() is method-form on its arg.
        out = _tr("mutate(flights, rank = min_rank(desc(arr_delay)))")
        # desc("arr_delay") is a function call producing a polars Expr;
        # min_rank is method form, so it becomes .min_rank() on whatever
        # desc returns. Since desc returns an Expr, we get desc('...').min_rank().
        assert out == "flights.mutate(rank=desc('arr_delay').min_rank())"


class TestJoins:
    """Phase 4 — every dplyr join, plus the `by = c("a" = "b")` named-vec."""

    def test_string_by(self):
        assert _tr('inner_join(x, y, by = "id")') == "x.inner_join(y, by='id')"

    def test_natural_join_no_by(self):
        # No ``by`` → polars' natural join.
        assert _tr("left_join(flights, planes)") == "flights.left_join(planes)"

    def test_unnamed_vec_by(self):
        out = _tr('inner_join(x, y, by = c("a", "b"))')
        assert out == "x.inner_join(y, by=['a', 'b'])"

    def test_named_vec_by_to_dict(self):
        out = _tr('inner_join(x, y, by = c("a" = "b"))')
        assert out == "x.inner_join(y, by={'a': 'b'})"

    def test_multi_named_vec_by(self):
        out = _tr('inner_join(x, y, by = c("a" = "x", "b" = "y"))')
        assert out == "x.inner_join(y, by={'a': 'x', 'b': 'y'})"

    def test_all_join_kinds(self):
        # Every join kind goes through the same machinery — smoke each.
        for verb, method in [
            ("inner_join", "inner_join"),
            ("left_join", "left_join"),
            ("right_join", "right_join"),
            ("full_join", "full_join"),
            ("semi_join", "semi_join"),
            ("anti_join", "anti_join"),
            ("cross_join", "cross_join"),
        ]:
            out = _tr(f'{verb}(x, y, by = "id")')
            assert out == f"x.{method}(y, by='id')", f"failed for {verb}"

    def test_join_in_pipe(self):
        out = _tr('flights |> left_join(planes, by = "tailnum")')
        assert out == "flights.left_join(planes, by='tailnum')"


class TestCaseWhen:
    """Phase 4 — case_when's tilde syntax → tuple-pair form."""

    def test_basic(self):
        out = _tr(
            'case_when(x > 0 ~ "pos", x < 0 ~ "neg", .default = "zero")'
        )
        assert out == (
            "case_when((col('x') > 0, 'pos'), (col('x') < 0, 'neg'), default='zero')"
        )

    def test_inside_mutate(self):
        out = _tr(
            'mutate(df, label = case_when('
            '  arr_delay > 60 ~ "late",'
            '  arr_delay < -10 ~ "early",'
            '  .default = "ontime"))'
        )
        assert out == (
            "df.mutate(label=case_when("
            "(col('arr_delay') > 60, 'late'), "
            "(col('arr_delay') < -10, 'early'), "
            "default='ontime'))"
        )

    def test_no_default(self):
        out = _tr('case_when(x > 0 ~ "pos")')
        assert out == "case_when((col('x') > 0, 'pos'))"

    def test_unary_tilde_as_default(self):
        # The degenerate ``~ value`` form is treated as the default branch.
        out = _tr('case_when(x > 0 ~ "pos", ~ "fallback")')
        assert out == "case_when((col('x') > 0, 'pos'), default='fallback')"


class TestExpressionHelpers:
    """Phase 4 — if_else, coalesce, na_if, between, near."""

    def test_if_else(self):
        out = _tr('mutate(df, status = if_else(x > 0, "pos", "neg"))')
        assert out == "df.mutate(status=if_else(col('x') > 0, 'pos', 'neg'))"

    def test_ifelse_alias_to_if_else(self):
        # R's base ``ifelse`` aliases to the same hea helper.
        out = _tr('mutate(df, status = ifelse(x > 0, "pos", "neg"))')
        assert out == "df.mutate(status=if_else(col('x') > 0, 'pos', 'neg'))"

    def test_coalesce(self):
        out = _tr("mutate(df, x = coalesce(a, b, c))")
        assert out == "df.mutate(x=coalesce(col('a'), col('b'), col('c')))"

    def test_na_if(self):
        out = _tr("mutate(df, x = na_if(a, -1))")
        assert out == "df.mutate(x=na_if(col('a'), -1))"

    def test_between(self):
        out = _tr("filter(df, between(x, 0, 100))")
        assert out == "df.filter(between(col('x'), 0, 100))"

    def test_near(self):
        out = _tr("filter(df, near(x, 3.14))")
        assert out == "df.filter(near(col('x'), 3.14))"


class TestPivot:
    """Phase 4 — pivot_longer / pivot_wider full kwarg coverage."""

    def test_pivot_longer_simple(self):
        out = _tr(
            'pivot_longer(df, c(wk1, wk2, wk3), '
            'names_to = "week", values_to = "rank")'
        )
        assert out == (
            "df.pivot_longer(['wk1', 'wk2', 'wk3'], "
            "names_to='week', values_to='rank')"
        )

    def test_pivot_longer_with_selector(self):
        out = _tr(
            'pivot_longer(billboard, starts_with("wk"), '
            'names_to = "week", values_to = "rank", values_drop_na = TRUE)'
        )
        assert out == (
            "billboard.pivot_longer(selectors.starts_with('wk'), "
            "names_to='week', values_to='rank', values_drop_na=True)"
        )

    def test_pivot_longer_with_names_sep(self):
        out = _tr(
            'pivot_longer(df, c(x_a, x_b), '
            'names_to = c("prefix", "key"), names_sep = "_")'
        )
        # ``names_to = c("prefix", "key")`` becomes a list.
        assert "names_to=['prefix', 'key']" in out
        assert "names_sep='_'" in out

    def test_pivot_wider_simple(self):
        out = _tr(
            'pivot_wider(fish_encounters, '
            'names_from = station, values_from = seen)'
        )
        assert out == (
            "fish_encounters.pivot_wider("
            "names_from='station', values_from='seen')"
        )

    def test_pivot_in_pipe(self):
        out = _tr(
            'billboard |> pivot_longer(starts_with("wk"), '
            'names_to = "week", values_to = "rank")'
        )
        assert out == (
            "billboard.pivot_longer(selectors.starts_with('wk'), "
            "names_to='week', values_to='rank')"
        )


class TestAcross:
    """Phase 4 — across() expansion at translate time."""

    def test_single_col_single_fn(self):
        out = _tr("mutate(df, across(x, mean))")
        assert out == "df.mutate(x=col('x').mean())"

    def test_multi_col_single_fn(self):
        out = _tr("mutate(df, across(c(a, b), mean))")
        assert out == "df.mutate(a=col('a').mean(), b=col('b').mean())"

    def test_with_lambda(self):
        out = _tr("mutate(df, across(c(a, b), \\(x) mean(x, na.rm = TRUE)))")
        assert out == (
            "df.mutate(a=col('a').mean(na_rm=True), b=col('b').mean(na_rm=True))"
        )

    def test_inside_summarize(self):
        out = _tr("summarize(df, across(c(a, b, c), mean))")
        assert out == (
            "df.summarize(a=col('a').mean(), b=col('b').mean(), c=col('c').mean())"
        )

    def test_mixed_with_explicit_kwargs(self):
        out = _tr(
            "mutate(df, gain = x - y, across(c(a, b), mean), .by = origin)"
        )
        assert out == (
            "df.mutate(gain=col('x') - col('y'), "
            "a=col('a').mean(), b=col('b').mean(), _by='origin')"
        )

    def test_names_kwarg_raises(self):
        # .names glue templating is deferred — translator should fail loudly.
        from hea.translate.r_to_py import RTranslateError
        with pytest.raises(RTranslateError):
            _tr('mutate(df, across(c(a, b), mean, .names = "{col}_avg"))')

    def test_list_fns_raises(self):
        from hea.translate.r_to_py import RTranslateError
        with pytest.raises(RTranslateError):
            _tr("mutate(df, across(c(a, b), list(mean = mean, sd = sd)))")


class TestGgplot:
    """Phase 5 — ``ggplot(df, aes(...)) + geom_*()`` chain detection,
    aes() unwrapping, facet formulas, and patchwork operators."""

    # ----- ggplot root + simplest chains -----

    def test_ggplot_with_named_aes(self):
        out = _tr(
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) "
            "+ geom_point()"
        )
        assert out == (
            "penguins.ggplot(x='flipper_length_mm', y='body_mass_g').geom_point()"
        )

    def test_positional_aes_maps_to_x_y(self):
        # R convention: aes(x, y) — first positional is x, second is y.
        out = _tr("ggplot(d, aes(x, y)) + geom_point()")
        assert out == "d.ggplot(x='x', y='y').geom_point()"

    def test_positional_and_named_mixed(self):
        out = _tr("ggplot(d, aes(x, y, color = z)) + geom_point()")
        assert out == "d.ggplot(x='x', y='y', color='z').geom_point()"

    def test_no_aes(self):
        # ``ggplot(df)`` with no aes — produces empty kwargs.
        out = _tr("ggplot(d) + geom_blank()")
        assert out == "d.ggplot().geom_blank()"

    # ----- aes inside geoms -----

    def test_aes_unwraps_inside_geom(self):
        out = _tr(
            "ggplot(d, aes(x = a)) + geom_point(aes(color = species, shape = species))"
        )
        assert out == (
            "d.ggplot(x='a').geom_point(color='species', shape='species')"
        )

    def test_geom_with_mixed_aes_and_literal_kwarg(self):
        # ``geom_point(aes(color = z), alpha = 0.5)`` — aesthetic kwargs
        # from aes unwrap; literal ``alpha`` is a regular kwarg.
        out = _tr(
            "ggplot(d, aes(x, y)) + geom_point(aes(color = z), alpha = 0.5)"
        )
        assert out == "d.ggplot(x='x', y='y').geom_point(color='z', alpha=0.5)"

    def test_aes_with_expression(self):
        # ``aes(x = log(weight))`` — value is a polars Expr in hea.
        out = _tr("ggplot(d, aes(x = log(weight), y = height)) + geom_point()")
        assert out == "d.ggplot(x=col('weight').log(), y='height').geom_point()"

    # ----- full chain -----

    def test_full_chain(self):
        out = _tr(
            "ggplot(penguins, aes(x = flipper_length_mm, y = body_mass_g)) "
            "+ geom_point(aes(color = species)) "
            "+ labs(title = \"Body mass vs flipper\", x = \"Flipper length (mm)\") "
            "+ theme_minimal()"
        )
        assert out == (
            "penguins.ggplot(x='flipper_length_mm', y='body_mass_g')"
            ".geom_point(color='species')"
            ".labs(title='Body mass vs flipper', x='Flipper length (mm)')"
            ".theme_minimal()"
        )

    def test_chain_with_scales(self):
        out = _tr(
            "ggplot(d, aes(x, y, color = z)) + geom_point() "
            "+ scale_color_viridis_c() + scale_x_log10()"
        )
        assert out == (
            "d.ggplot(x='x', y='y', color='z').geom_point()"
            ".scale_color_viridis_c().scale_x_log10()"
        )

    def test_chain_with_coord_polar(self):
        out = _tr("ggplot(d, aes(x = species)) + geom_bar() + coord_polar()")
        assert out == "d.ggplot(x='species').geom_bar().coord_polar()"

    # ----- facets -----

    def test_facet_wrap_formula(self):
        out = _tr(
            "ggplot(d, aes(x, y)) + geom_point() + facet_wrap(~island)"
        )
        assert out == "d.ggplot(x='x', y='y').geom_point().facet_wrap('~island')"

    def test_facet_grid_binary_formula(self):
        out = _tr("ggplot(d, aes(x, y)) + geom_point() + facet_grid(year ~ month)")
        assert out == (
            "d.ggplot(x='x', y='y').geom_point().facet_grid('year ~ month')"
        )

    # ----- theme + element_* -----

    def test_theme_with_element_text(self):
        out = _tr(
            "ggplot(d, aes(x, y)) + geom_point() "
            "+ theme(axis.text = element_text(size = 10))"
        )
        # ``axis.text`` becomes ``axis_text`` via the dot→underscore rule;
        # element_text is a regular function call (no special unwrap).
        assert "axis_text=element_text(size=10)" in out
        assert ".theme(" in out

    def test_theme_bare_then_named_theme(self):
        out = _tr("ggplot(d) + theme_bw() + theme(legend.position = \"none\")")
        assert out == "d.ggplot().theme_bw().theme(legend_position='none')"

    # ----- patchwork operators -----

    def test_patchwork_pipe(self):
        # ``p1 | p2`` — neither is a chain extension, so ``|`` stays as ``|``.
        assert _tr("p1 | p2") == "p1 | p2"

    def test_patchwork_stack(self):
        assert _tr("p1 / p2") == "p1 / p2"

    def test_patchwork_grouped(self):
        assert _tr("(p1 | p2) / p3") == "(p1 | p2) / p3"

    def test_patchwork_plus_with_bare_name(self):
        # ``p1 + p2`` — RHS isn't a ggplot extension call, so stays as ``+``.
        assert _tr("p1 + p2") == "p1 + p2"

    def test_patchwork_with_plot_annotation(self):
        # ``plot_annotation`` IS a chain extension — gets method-call form.
        out = _tr('(p1 | p2) + plot_annotation(title = "title")')
        assert out == "(p1 | p2).plot_annotation(title='title')"

    # ----- aes with c(name = value) for renamed mappings -----

    def test_aes_with_string_value(self):
        # ``aes(x = "y_column")`` — string literal value.
        out = _tr('ggplot(d, aes(x = "weight")) + geom_point()')
        assert out == "d.ggplot(x='weight').geom_point()"

    # ----- arithmetic on plot expressions remains arithmetic -----

    def test_plus_with_geom_does_chain(self):
        # Sanity: a ``+`` with a geom RHS is a chain — even when LHS is bare.
        out = _tr("p + geom_point()")
        assert out == "p.geom_point()"

    def test_plus_with_non_chain_rhs_stays_arithmetic(self):
        # ``p + 1`` — RHS is a number, not a chain extension.
        assert _tr("p + 1") == "p + 1"


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
