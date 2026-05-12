"""Tests for hea.translate.r_lexer — Phase 0 acceptance.

Covers every token kind the lexer can emit, the maximal-munch resolution of
overlapping operators, and the newline-suppression rules. End-to-end smoke
tests use the canonical r4ds pipeline snippet.
"""

import pytest

from hea.translate.r_lexer import RLexError, Token, tokenize


def _kinds(src: str) -> list[str]:
    return [t.kind for t in tokenize(src)]


def _values(src: str) -> list[str]:
    return [t.value for t in tokenize(src)]


def _cooked(src: str) -> list[object]:
    return [t.cooked for t in tokenize(src)]


# ---------------------------------------------------------------------------
# Numeric literals
# ---------------------------------------------------------------------------


class TestNumbers:
    def test_int(self):
        toks = tokenize("42")
        assert toks[0].kind == "NUM"
        assert toks[0].cooked == 42.0

    def test_double(self):
        assert tokenize("3.14")[0].cooked == 3.14

    def test_integer_suffix(self):
        toks = tokenize("42L")
        assert toks[0].kind == "INT"
        assert toks[0].cooked == 42

    def test_complex(self):
        toks = tokenize("3.5i")
        assert toks[0].kind == "COMPLEX"
        assert toks[0].cooked == 3.5

    def test_hex(self):
        assert tokenize("0xFF")[0].cooked == 255.0
        assert tokenize("0x10L")[0].cooked == 16
        assert tokenize("0x10L")[0].kind == "INT"

    def test_scientific(self):
        assert tokenize("1e3")[0].cooked == 1000.0
        assert tokenize("1.5e-2")[0].cooked == 0.015

    def test_leading_dot(self):
        toks = tokenize(".5")
        assert toks[0].kind == "NUM"
        assert toks[0].cooked == 0.5

    def test_malformed_exponent(self):
        with pytest.raises(RLexError):
            tokenize("1e")


# ---------------------------------------------------------------------------
# String literals
# ---------------------------------------------------------------------------


class TestStrings:
    def test_double_quoted(self):
        toks = tokenize('"hello"')
        assert toks[0].kind == "STR"
        assert toks[0].cooked == "hello"

    def test_single_quoted(self):
        assert tokenize("'world'")[0].cooked == "world"

    def test_basic_escapes(self):
        assert tokenize('"line\\nbreak"')[0].cooked == "line\nbreak"
        assert tokenize('"tab\\there"')[0].cooked == "tab\there"
        assert tokenize('"q\\"uote"')[0].cooked == 'q"uote'

    def test_hex_escape(self):
        assert tokenize('"\\x41"')[0].cooked == "A"

    def test_unicode_escape(self):
        assert tokenize('"\\u00e9"')[0].cooked == "é"
        assert tokenize('"\\u{00e9}"')[0].cooked == "é"

    def test_raw_string_paren(self):
        toks = tokenize('r"(no \\n escapes)"')
        assert toks[0].kind == "STR"
        assert toks[0].cooked == "no \\n escapes"

    def test_raw_string_bracket(self):
        assert tokenize('r"[content]"')[0].cooked == "content"

    def test_raw_string_brace(self):
        assert tokenize('r"{body}"')[0].cooked == "body"

    def test_raw_string_dashed(self):
        # r"--(...)--" — dashes let the body include the closer ``)``
        # alone, or the bare quote character, without ending the string.
        assert tokenize('r"--(a "b" c)--"')[0].cooked == 'a "b" c'

    def test_unterminated_string(self):
        with pytest.raises(RLexError):
            tokenize('"oops')


# ---------------------------------------------------------------------------
# Identifiers and keywords
# ---------------------------------------------------------------------------


class TestIdents:
    def test_plain(self):
        toks = tokenize("flights")
        assert toks[0].kind == "IDENT"
        assert toks[0].value == "flights"

    def test_with_dots(self):
        assert tokenize("data.frame")[0].value == "data.frame"

    def test_backticked(self):
        toks = tokenize("`weird name`")
        assert toks[0].kind == "IDENT"
        assert toks[0].value == "`weird name`"

    def test_dot_leading_is_id(self):
        # ``.foo`` is an identifier (leading dot, next char non-digit).
        assert _kinds(".foo") == ["IDENT", "EOF"]

    def test_keywords(self):
        for kw in ("if", "else", "for", "while", "repeat", "break", "next", "in", "function"):
            assert _kinds(kw) == [kw, "EOF"]


# ---------------------------------------------------------------------------
# Constant literals
# ---------------------------------------------------------------------------


class TestConstants:
    def test_true_false(self):
        assert tokenize("TRUE")[0].kind == "BOOL"
        assert tokenize("TRUE")[0].cooked is True
        assert tokenize("FALSE")[0].cooked is False

    def test_null(self):
        assert tokenize("NULL")[0].kind == "NULL"

    def test_na_variants(self):
        assert tokenize("NA")[0].cooked == "NA"
        assert tokenize("NA_integer_")[0].cooked == "NA_integer_"
        assert tokenize("NA_real_")[0].cooked == "NA_real_"
        assert tokenize("NA_character_")[0].cooked == "NA_character_"

    def test_inf_nan(self):
        assert tokenize("Inf")[0].kind == "INF"
        assert tokenize("NaN")[0].kind == "NAN"


# ---------------------------------------------------------------------------
# Operators — maximal munch
# ---------------------------------------------------------------------------


class TestOperators:
    @pytest.mark.parametrize("src,kind", [
        ("<-", "<-"), ("<<-", "<<-"), ("<=", "<="), ("<", "<"),
        ("->", "->"), ("->>", "->>"), (">=", ">="), (">", ">"),
        ("==", "=="), ("!=", "!="), ("=", "="),
        ("&", "&"), ("&&", "&&"), ("|", "|"), ("||", "||"), ("|>", "|>"),
        (":", ":"), ("::", "::"), (":::", ":::"),
        ("[", "["), ("[[", "[["), ("]", "]"), ("]]", "]]"),
    ])
    def test_op_kind(self, src, kind):
        assert tokenize(src)[0].kind == kind

    def test_user_infix(self):
        toks = tokenize("a %in% b")
        assert [t.kind for t in toks[:3]] == ["IDENT", "%infix%", "IDENT"]
        assert toks[1].value == "%in%"

    def test_modulo_as_infix(self):
        # %% is just a special-case user infix.
        toks = tokenize("a %% b")
        assert toks[1].kind == "%infix%"
        assert toks[1].value == "%%"

    def test_magrittr_pipe_is_infix(self):
        toks = tokenize("a %>% b")
        assert toks[1].kind == "%infix%"
        assert toks[1].value == "%>%"

    def test_native_pipe_separate(self):
        toks = tokenize("a |> b")
        assert toks[1].kind == "|>"

    def test_lambda_backslash(self):
        toks = tokenize("\\(x) x + 1")
        assert toks[0].kind == "\\"


# ---------------------------------------------------------------------------
# Comments and whitespace
# ---------------------------------------------------------------------------


class TestCommentsWhitespace:
    def test_comment_skipped(self):
        toks = tokenize("x # this is a comment\ny")
        # x, TERM, y, EOF — the comment is dropped, newline still ends the stmt
        assert [t.kind for t in toks] == ["IDENT", "TERM", "IDENT", "EOF"]

    def test_blank_lines(self):
        toks = tokenize("\n\nx\n\n")
        # Leading blank lines are absorbed (last_kind is None);
        # trailing newline emits TERM after x.
        assert [t.kind for t in toks] == ["IDENT", "TERM", "EOF"]


# ---------------------------------------------------------------------------
# Newline suppression
# ---------------------------------------------------------------------------


class TestNewlines:
    def test_term_after_complete_expression(self):
        assert _kinds("x\ny") == ["IDENT", "TERM", "IDENT", "EOF"]

    def test_no_term_after_binary_op(self):
        # ``x +\n y`` continues — no TERM in between.
        assert _kinds("x +\ny") == ["IDENT", "+", "IDENT", "EOF"]

    def test_no_term_inside_parens(self):
        assert _kinds("f(\nx\n)") == ["IDENT", "(", "IDENT", ")", "EOF"]

    def test_no_term_inside_brackets(self):
        assert _kinds("df[\nx\n]") == ["IDENT", "[", "IDENT", "]", "EOF"]

    def test_term_inside_brace(self):
        # ``{ }`` does NOT suppress newlines — they separate block statements.
        assert _kinds("{\nx\ny\n}") == ["{", "IDENT", "TERM", "IDENT", "TERM", "}", "EOF"]

    def test_semicolon_always_term(self):
        assert _kinds("x;y") == ["IDENT", "TERM", "IDENT", "EOF"]


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_canonical_pipeline():
    src = 'flights |> filter(dest == "IAH")'
    kinds = _kinds(src)
    assert kinds == [
        "IDENT", "|>", "IDENT", "(", "IDENT", "==", "STR", ")", "EOF",
    ]


def test_multiline_pipeline():
    src = """\
flights |>
  filter(dest == "IAH") |>
  group_by(year, month, day) |>
  summarize(arr_delay = mean(arr_delay, na.rm = TRUE))
"""
    kinds = _kinds(src)
    # Should be one continuous statement (no TERM until the final newline).
    # The lexer doesn't emit a trailing TERM if there's no token after — but
    # there's a newline before EOF that follows ``)`` so we do get one.
    assert "TERM" in kinds  # the final newline
    # The pipeline shouldn't be broken by interior newlines.
    pipe_count = kinds.count("|>")
    assert pipe_count == 3


def test_assignment_then_call():
    toks = tokenize("x <- f(1, 2)\ny")
    assert [t.kind for t in toks] == [
        "IDENT", "<-", "IDENT", "(", "NUM", ",", "NUM", ")", "TERM", "IDENT", "EOF",
    ]
