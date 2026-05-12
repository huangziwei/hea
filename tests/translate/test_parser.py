"""Tests for hea.translate.r_parser — Phase 0 acceptance.

Covers each AST node type, R's full precedence table from ``?Syntax``, the
canonical r4ds pipeline as a golden parse, and out-of-grammar failures.
"""

import pytest

from hea.translate import r_ast as A
from hea.translate.r_parser import RParseError, parse


def _one(src: str):
    """Parse ``src`` and return the single top-level expression."""
    prog = parse(src)
    assert isinstance(prog, A.Program)
    assert len(prog.statements) == 1, f"expected 1 stmt, got {len(prog.statements)}"
    return prog.statements[0]


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


class TestAtoms:
    def test_int(self):
        n = _one("42L")
        assert isinstance(n, A.IntLit) and n.value == 42

    def test_num(self):
        n = _one("3.14")
        assert isinstance(n, A.NumLit) and n.value == 3.14

    def test_string(self):
        n = _one('"hello"')
        assert isinstance(n, A.StrLit) and n.value == "hello"

    def test_bool(self):
        assert _one("TRUE").value is True
        assert _one("FALSE").value is False

    def test_null(self):
        assert isinstance(_one("NULL"), A.NullLit)

    def test_na(self):
        n = _one("NA_integer_")
        assert isinstance(n, A.NaLit) and n.kind == "NA_integer_"

    def test_inf_nan(self):
        assert isinstance(_one("Inf"), A.InfLit)
        assert isinstance(_one("NaN"), A.NanLit)

    def test_identifier(self):
        n = _one("flights")
        assert isinstance(n, A.Identifier) and n.name == "flights"

    def test_backticked_identifier(self):
        n = _one("`weird name`")
        assert isinstance(n, A.Identifier) and n.name == "weird name" and n.backticked


# ---------------------------------------------------------------------------
# Operator precedence (R's ?Syntax table)
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_arith_left_assoc(self):
        # a + b + c -> (a + b) + c
        n = _one("a + b + c")
        assert isinstance(n, A.BinOp) and n.op == "+"
        assert isinstance(n.left, A.BinOp) and n.left.op == "+"

    def test_mul_over_add(self):
        # a + b * c -> a + (b * c)
        n = _one("a + b * c")
        assert n.op == "+" and isinstance(n.right, A.BinOp) and n.right.op == "*"

    def test_caret_right_assoc(self):
        # a ^ b ^ c -> a ^ (b ^ c)
        n = _one("a ^ b ^ c")
        assert n.op == "^" and isinstance(n.right, A.BinOp) and n.right.op == "^"

    def test_unary_minus_below_caret(self):
        # -2 ^ 2 in R parses as -(2^2) = -4 (NOT (-2)^2 = 4).
        n = _one("-2 ^ 2")
        assert isinstance(n, A.UnaryOp) and n.op == "-"
        assert isinstance(n.operand, A.BinOp) and n.operand.op == "^"

    def test_colon_higher_than_arith(self):
        # 1:5 + 1 -> (1:5) + 1
        n = _one("1:5 + 1")
        assert n.op == "+" and isinstance(n.left, A.BinOp) and n.left.op == ":"

    def test_special_infix_between_colon_and_mul(self):
        # a %in% b * c -> (a %in% b) * c
        n = _one("a %in% b * c")
        assert n.op == "*"
        assert isinstance(n.left, A.BinOp) and n.left.op == "%in%"

    def test_comparison_below_arith(self):
        # a + b == c * d -> (a+b) == (c*d)
        n = _one("a + b == c * d")
        assert n.op == "==" and n.left.op == "+" and n.right.op == "*"

    def test_not_above_and(self):
        # !a & b -> (!a) & b
        n = _one("!a & b")
        assert n.op == "&" and isinstance(n.left, A.UnaryOp) and n.left.op == "!"

    def test_and_above_or(self):
        # a | b & c -> a | (b & c)
        n = _one("a | b & c")
        assert n.op == "|" and isinstance(n.right, A.BinOp) and n.right.op == "&"

    def test_assignment_right_assoc(self):
        # a <- b <- 1 -> a <- (b <- 1)
        n = _one("a <- b <- 1")
        assert isinstance(n, A.Assign) and n.op == "<-"
        assert isinstance(n.value, A.Assign)

    def test_right_arrow_flips(self):
        # 1 -> a  is equivalent to  a <- 1
        n = _one("1 -> a")
        assert isinstance(n, A.Assign) and n.op == "<-"
        assert n.target.name == "a"
        assert n.value.value == 1.0  # NumLit

    def test_tilde_low_precedence(self):
        # y ~ x + z -> y ~ (x + z)
        n = _one("y ~ x + z")
        assert isinstance(n, A.Tilde)
        assert isinstance(n.rhs, A.BinOp) and n.rhs.op == "+"

    def test_unary_tilde(self):
        n = _one("~ x")
        assert isinstance(n, A.Tilde) and n.lhs is None


# ---------------------------------------------------------------------------
# Calls, subscripts, dollar / at
# ---------------------------------------------------------------------------


class TestCalls:
    def test_simple_call(self):
        n = _one("f(1, 2)")
        assert isinstance(n, A.Call) and n.func.name == "f"
        assert len(n.args) == 2

    def test_named_arg(self):
        n = _one("f(x = 1, 2)")
        assert isinstance(n.args[0], A.NamedArg) and n.args[0].name == "x"
        assert n.args[0].value.value == 1.0
        assert isinstance(n.args[1], A.NumLit)

    def test_named_arg_via_string(self):
        n = _one('f("x" = 1)')
        assert isinstance(n.args[0], A.NamedArg) and n.args[0].name == "x"

    def test_zero_arg(self):
        n = _one("f()")
        assert isinstance(n, A.Call) and n.args == ()

    def test_chained_calls(self):
        n = _one("f()()")
        # outer Call's func is itself a Call
        assert isinstance(n, A.Call) and isinstance(n.func, A.Call)

    def test_subscript(self):
        n = _one("df[i]")
        assert isinstance(n, A.Subscript)
        assert len(n.args) == 1

    def test_subscript_multi_with_missing(self):
        n = _one('df[, "a"]')
        assert isinstance(n, A.Subscript)
        assert len(n.args) == 2
        assert isinstance(n.args[0], A.MissingArg)
        assert isinstance(n.args[1], A.StrLit)

    def test_double_subscript(self):
        n = _one("x[[1]]")
        assert isinstance(n, A.DoubleSubscript)

    def test_dollar(self):
        n = _one("df$col")
        assert isinstance(n, A.Dollar) and n.name == "col"

    def test_at_slot(self):
        n = _one("obj@slot")
        assert isinstance(n, A.At) and n.name == "slot"


# ---------------------------------------------------------------------------
# Pipes
# ---------------------------------------------------------------------------


class TestPipes:
    def test_native_pipe(self):
        n = _one("x |> f()")
        assert isinstance(n, A.Pipe) and n.op == "|>"
        assert n.lhs.name == "x"
        assert isinstance(n.rhs, A.Call) and n.rhs.func.name == "f"

    def test_magrittr_pipe(self):
        n = _one("x %>% f()")
        assert isinstance(n, A.Pipe) and n.op == "%>%"

    def test_pipe_chain_left_assoc(self):
        # a |> b() |> c() -> (a |> b()) |> c()
        n = _one("a |> b() |> c()")
        assert isinstance(n, A.Pipe)
        assert isinstance(n.lhs, A.Pipe)
        assert n.rhs.func.name == "c"

    def test_pipe_below_special_infix(self):
        # ``a %any% b |> f()`` — special infix and pipe are at the same
        # precedence (both at the special-ops level). Left-to-right.
        n = _one("a %in% b |> f()")
        assert isinstance(n, A.Pipe)
        assert isinstance(n.lhs, A.BinOp) and n.lhs.op == "%in%"


# ---------------------------------------------------------------------------
# Control flow & function defs
# ---------------------------------------------------------------------------


class TestControlFlow:
    def test_if_else(self):
        n = _one("if (x > 0) a else b")
        assert isinstance(n, A.If)
        assert n.otherwise is not None

    def test_if_no_else(self):
        n = _one("if (x) a")
        assert isinstance(n, A.If) and n.otherwise is None

    def test_if_else_across_term(self):
        # Standard idiom: ``if (...) { ... } else { ... }`` with newline
        # between the closing brace and ``else``. We must absorb the TERM.
        n = _one("if (x) {\n  a\n} else {\n  b\n}")
        assert isinstance(n, A.If)
        assert isinstance(n.then, A.Block)
        assert isinstance(n.otherwise, A.Block)

    def test_for_loop(self):
        n = _one("for (i in 1:10) print(i)")
        assert isinstance(n, A.For)
        assert n.var == "i"
        assert isinstance(n.iterable, A.BinOp) and n.iterable.op == ":"

    def test_while_loop(self):
        n = _one("while (x < 10) x <- x + 1")
        assert isinstance(n, A.While)

    def test_repeat(self):
        n = _one("repeat { x <- x + 1; if (x > 10) break }")
        assert isinstance(n, A.Repeat)

    def test_break_next(self):
        prog = parse("for (i in 1:3) { if (i == 2) next; if (i == 3) break }")
        # we only care it parses without error
        assert isinstance(prog, A.Program)

    def test_function_def(self):
        n = _one("function(x, y = 1) x + y")
        assert isinstance(n, A.FunctionDef)
        assert len(n.params) == 2
        assert n.params[0].name == "x" and n.params[0].default is None
        assert n.params[1].name == "y" and n.params[1].default.value == 1.0
        assert not n.shorthand

    def test_lambda_shorthand(self):
        n = _one("\\(x) x + 1")
        assert isinstance(n, A.FunctionDef) and n.shorthand


# ---------------------------------------------------------------------------
# Blocks and multi-statement scripts
# ---------------------------------------------------------------------------


class TestStatements:
    def test_block(self):
        n = _one("{ x <- 1; y <- 2; x + y }")
        assert isinstance(n, A.Block)
        assert len(n.statements) == 3

    def test_multi_statement_program(self):
        prog = parse("x <- 1\ny <- 2\nz <- x + y")
        assert len(prog.statements) == 3
        assert all(isinstance(s, A.Assign) for s in prog.statements)

    def test_semicolon_separator(self):
        prog = parse("a; b; c")
        assert len(prog.statements) == 3


# ---------------------------------------------------------------------------
# Golden: the canonical r4ds pipeline
# ---------------------------------------------------------------------------


def test_canonical_pipeline():
    src = """\
flights |>
  filter(dest == "IAH") |>
  group_by(year, month, day) |>
  summarize(arr_delay = mean(arr_delay, na.rm = TRUE))
"""
    prog = parse(src)
    assert len(prog.statements) == 1
    top = prog.statements[0]

    # Top of the chain: ``... |> summarize(...)``
    assert isinstance(top, A.Pipe) and top.op == "|>"
    summarize = top.rhs
    assert isinstance(summarize, A.Call) and summarize.func.name == "summarize"
    assert isinstance(summarize.args[0], A.NamedArg) and summarize.args[0].name == "arr_delay"

    # The mean(arr_delay, na.rm = TRUE) call.
    mean_call = summarize.args[0].value
    assert isinstance(mean_call, A.Call) and mean_call.func.name == "mean"
    assert isinstance(mean_call.args[1], A.NamedArg) and mean_call.args[1].name == "na.rm"
    assert mean_call.args[1].value.value is True

    # Walk down the pipe chain.
    p1 = top.lhs                            # ... |> group_by(...)
    assert isinstance(p1, A.Pipe)
    p2 = p1.lhs                             # ... |> filter(...)
    assert isinstance(p2, A.Pipe)
    p3 = p2.lhs                             # flights
    assert isinstance(p3, A.Identifier) and p3.name == "flights"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unclosed_paren(self):
        with pytest.raises(RParseError):
            parse("f(1, 2")

    def test_dangling_operator(self):
        with pytest.raises(RParseError):
            parse("1 +")

    def test_unexpected_token(self):
        with pytest.raises(RParseError):
            parse(")")
