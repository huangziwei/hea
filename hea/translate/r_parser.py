"""R parser — token stream → :mod:`hea.translate.r_ast` tree.

Pratt-style precedence climbing, implementing R's full operator precedence
table from ``?Syntax``. Out-of-grammar input raises :class:`RParseError`
with a span pointing at the offending token.

Precedence table (low → high). Right-associative slots have ``rbp < lbp``;
left-associative have ``rbp == lbp``.

==============  =====  =====  =================================
operator        lbp    rbp    notes
==============  =====  =====  =================================
``?``             5      5    help (rare in scripts)
``=``            10      9    right-assoc assignment
``<-``  ``<<-``  15     14    right-assoc assignment
``->``  ``->>``  20     20    left-assoc (the parser flips ``a -> b`` to ``b <- a``)
``~``            25     25    formula (binary); unary handled in nud
``|``  ``||``    30     30
``&``  ``&&``    35     35
``!``             —      —    unary only (prefix, nud)
``<`` ``<=`` ``>`` ``>=`` ``==`` ``!=`` 40 40
``+``  ``-``     45     45
``*``  ``/``     50     50
``%...%`` ``|>`` 55     55    special operators incl. pipe
``:``            60     60    sequence
unary ``+`` ``-`` —      65    nud
``^``            70     69    right-assoc
``$``  ``@``     80     80
``::`` ``:::``   85     85
``(`` ``[`` ``[[`` 90    —    postfix (call / subscript)
==============  =====  =====  =================================
"""

from __future__ import annotations

from typing import Optional

from . import r_ast as A
from .r_ast import Node, Span
from .r_lexer import Token, tokenize


class RParseError(Exception):
    def __init__(self, message: str, span: Span, source: str):
        self.span = span
        self.source = source
        snippet = source[span[0]:span[1]] if span[1] > span[0] else source[span[0]:span[0]+1]
        super().__init__(f"{message} at byte {span[0]}: {snippet!r}")


# Precedence (left binding power). Higher binds tighter.
_LBP = {
    "?": 5,
    "=": 10,
    "<-": 15, "<<-": 15,
    "->": 20, "->>": 20,
    "~": 25,
    "|": 30, "||": 30,
    "&": 35, "&&": 35,
    "<": 40, "<=": 40, ">": 40, ">=": 40, "==": 40, "!=": 40,
    "+": 45, "-": 45,
    "*": 50, "/": 50,
    "%infix%": 55, "|>": 55,
    ":": 60,
    "^": 70,
    "$": 80, "@": 80,
    "::": 85, ":::": 85,
    # Postfix
    "(": 90, "[": 90, "[[": 90,
}

# Right-associative ops use a slightly lower rbp than lbp.
_RIGHT_ASSOC = {"^", "<-", "<<-", "=", "?"}


def parse(src: str) -> A.Program:
    """Parse an R source string into an :class:`r_ast.Program`."""
    tokens = tokenize(src)
    return _Parser(tokens, src).parse_program()


class _Parser:
    __slots__ = ("tokens", "src", "i")

    def __init__(self, tokens: list[Token], src: str):
        self.tokens = tokens
        self.src = src
        self.i = 0

    # -- token helpers ------------------------------------------------------

    def _peek(self, off: int = 0) -> Token:
        idx = self.i + off
        if idx >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[idx]

    def _advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _accept(self, kind: str) -> Token | None:
        if self._peek().kind == kind:
            return self._advance()
        return None

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise RParseError(f"expected {kind!r}, got {tok.kind!r}", tok.span, self.src)
        return self._advance()

    def _skip_terms(self):
        while self._peek().kind == "TERM":
            self._advance()

    def _peek_skip_terms(self) -> Token:
        """Look ahead past any TERMs without consuming them. Used to decide
        e.g. whether an ``else`` follows the end of an if-body."""
        j = self.i
        while j < len(self.tokens) and self.tokens[j].kind == "TERM":
            j += 1
        return self.tokens[j] if j < len(self.tokens) else self.tokens[-1]

    # -- entry --------------------------------------------------------------

    def parse_program(self) -> A.Program:
        stmts: list[Node] = []
        start = self._peek().span[0]
        self._skip_terms()
        while self._peek().kind != "EOF":
            stmts.append(self.parse_expr(0))
            # A statement may or may not be followed by a TERM; skip any.
            self._skip_terms()
        end = self._peek().span[1]
        return A.Program(tuple(stmts), (start, end))

    # -- Pratt core ---------------------------------------------------------

    def parse_expr(self, rbp: int = 0) -> Node:
        left = self._nud(self._advance())
        while True:
            tok = self._peek()
            lbp = _LBP.get(tok.kind, 0)
            # ``%infix%`` precedence is keyed on the token kind, not value.
            if lbp <= rbp:
                break
            # Operator consumed — _led decides its own right-binding.
            self._advance()
            left = self._led(tok, left, lbp)
        return left

    # -- nud: prefix / atom handlers ---------------------------------------

    def _nud(self, tok: Token) -> Node:
        kind = tok.kind

        if kind == "NUM":
            return A.NumLit(float(tok.cooked), tok.span)
        if kind == "INT":
            return A.IntLit(int(tok.cooked), tok.span)
        if kind == "COMPLEX":
            return A.ComplexLit(float(tok.cooked), tok.span)
        if kind == "STR":
            raw = tok.value.startswith(("r", "R"))
            return A.StrLit(str(tok.cooked), tok.span, raw=raw)
        if kind == "BOOL":
            return A.BoolLit(bool(tok.cooked), tok.span)
        if kind == "NULL":
            return A.NullLit(tok.span)
        if kind == "NA":
            return A.NaLit(str(tok.cooked), tok.span)
        if kind == "INF":
            return A.InfLit(tok.span)
        if kind == "NAN":
            return A.NanLit(tok.span)

        if kind == "IDENT":
            name = tok.value
            backticked = name.startswith("`") and name.endswith("`")
            if backticked:
                name = name[1:-1]
            return A.Identifier(name, tok.span, backticked=backticked)

        if kind == "(":
            # Parenthesized expression.
            inner = self.parse_expr(0)
            end = self._expect(")").span[1]
            # R's surface ``( expr )`` doesn't have its own AST node; we
            # just return the inner expression. Span is widened.
            return _with_span(inner, (tok.span[0], end))

        if kind == "{":
            return self._parse_block(tok.span[0])

        if kind == "-" or kind == "+":
            # Unary +/-
            operand = self.parse_expr(65)
            return A.UnaryOp(kind, operand, (tok.span[0], _end(operand)))

        if kind == "!":
            operand = self.parse_expr(45)
            return A.UnaryOp("!", operand, (tok.span[0], _end(operand)))

        if kind == "~":
            # Unary tilde: ``~ rhs``
            rhs = self.parse_expr(25)
            return A.Tilde(None, rhs, (tok.span[0], _end(rhs)))

        if kind == "function":
            return self._parse_function_def(tok.span[0], shorthand=False)

        if kind == "\\":
            return self._parse_function_def(tok.span[0], shorthand=True)

        if kind == "if":
            return self._parse_if(tok.span[0])

        if kind == "for":
            return self._parse_for(tok.span[0])

        if kind == "while":
            return self._parse_while(tok.span[0])

        if kind == "repeat":
            return self._parse_repeat(tok.span[0])

        if kind == "break":
            return A.Break(tok.span)

        if kind == "next":
            return A.Next(tok.span)

        if kind == "?":
            # Unary help. Rare; consume the operand with low rbp.
            rhs = self.parse_expr(5)
            return A.UnaryOp("?", rhs, (tok.span[0], _end(rhs)))

        raise RParseError(f"unexpected token in expression: {kind!r}", tok.span, self.src)

    # -- led: infix / postfix handlers -------------------------------------

    def _led(self, tok: Token, left: Node, lbp: int) -> Node:
        kind = tok.kind
        rbp = lbp - 1 if kind in _RIGHT_ASSOC else lbp

        # Postfix call.
        if kind == "(":
            args = self._parse_arg_list(")")
            end = self._expect(")").span[1]
            return A.Call(left, tuple(args), (_start(left), end))

        # Single-bracket subscript.
        if kind == "[":
            args = self._parse_arg_list("]", allow_missing=True)
            end = self._expect("]").span[1]
            return A.Subscript(left, tuple(args), (_start(left), end))

        # Double-bracket subscript.
        if kind == "[[":
            args = self._parse_arg_list("]]")
            end = self._expect("]]").span[1]
            return A.DoubleSubscript(left, tuple(args), (_start(left), end))

        # Component / slot access — RHS must be an identifier or string,
        # but is NOT a normal expression (no NSE on a `$b` access).
        if kind == "$":
            name_tok = self._advance()
            name = _component_name_from(name_tok, self.src)
            return A.Dollar(left, name, (_start(left), name_tok.span[1]))

        if kind == "@":
            name_tok = self._advance()
            name = _component_name_from(name_tok, self.src)
            return A.At(left, name, (_start(left), name_tok.span[1]))

        # Namespace access — ``pkg::name`` / ``pkg:::name``. Both sides are
        # identifiers; we model as a BinOp to keep one node for both forms.
        if kind == "::" or kind == ":::":
            name_tok = self._advance()
            if name_tok.kind != "IDENT":
                raise RParseError(f"expected identifier after {kind!r}", name_tok.span, self.src)
            rhs = self._nud(name_tok)
            return A.BinOp(kind, left, rhs, (_start(left), _end(rhs)))

        # Pipes — both ``|>`` and ``%>%``-via-infix get the dedicated node.
        if kind == "|>":
            rhs = self.parse_expr(rbp)
            return A.Pipe("|>", left, rhs, (_start(left), _end(rhs)))

        if kind == "%infix%":
            if tok.value == "%>%":
                rhs = self.parse_expr(rbp)
                return A.Pipe("%>%", left, rhs, (_start(left), _end(rhs)))
            rhs = self.parse_expr(rbp)
            return A.BinOp(tok.value, left, rhs, (_start(left), _end(rhs)))

        # Right-assignment: ``a -> b`` / ``a ->> b`` flip to leftward form so
        # downstream consumers don't need a second code path.
        if kind == "->" or kind == "->>":
            rhs = self.parse_expr(rbp)
            flipped_op = "<-" if kind == "->" else "<<-"
            return A.Assign(flipped_op, rhs, left, (_start(left), _end(rhs)))

        # Leftward assignment.
        if kind == "<-" or kind == "<<-" or kind == "=":
            rhs = self.parse_expr(rbp)
            # Named-arg-vs-assignment is decided by the caller (``_parse_arg_list``)
            # for ``=`` inside parens — we never reach here for that case.
            return A.Assign(kind, left, rhs, (_start(left), _end(rhs)))

        # Tilde formula.
        if kind == "~":
            rhs = self.parse_expr(rbp)
            return A.Tilde(left, rhs, (_start(left), _end(rhs)))

        # Sequence ``:``
        if kind == ":":
            rhs = self.parse_expr(rbp)
            return A.BinOp(":", left, rhs, (_start(left), _end(rhs)))

        # Remaining: arithmetic / comparison / logical — all BinOp.
        rhs = self.parse_expr(rbp)
        return A.BinOp(kind, left, rhs, (_start(left), _end(rhs)))

    # -- compound constructs ------------------------------------------------

    def _parse_block(self, start: int) -> A.Block:
        """Parse ``{ stmt; stmt; … }``. Caller already consumed ``{``."""
        stmts: list[Node] = []
        self._skip_terms()
        while self._peek().kind != "}":
            if self._peek().kind == "EOF":
                raise RParseError("unterminated brace block", (start, self.src.__len__() - 1), self.src)
            stmts.append(self.parse_expr(0))
            self._skip_terms()
        end = self._expect("}").span[1]
        return A.Block(tuple(stmts), (start, end))

    def _parse_arg_list(self, closer: str, *, allow_missing: bool = False) -> list[Node]:
        """Parse a comma-separated argument list up to but not consuming
        ``closer``. ``=`` inside the list is a named-arg, not assignment.

        When ``allow_missing`` is True (subscript context), empty positions
        become :class:`r_ast.MissingArg` rather than a syntax error — so
        ``df[, "a"]`` parses cleanly.
        """
        args: list[Node] = []
        self._skip_terms()
        if self._peek().kind == closer:
            return args
        while True:
            self._skip_terms()
            # Empty position (subscript)?
            if allow_missing and self._peek().kind in (",", closer):
                args.append(A.MissingArg(self._peek().span))
            else:
                arg = self._parse_arg()
                args.append(arg)
            self._skip_terms()
            if self._peek().kind == ",":
                self._advance()
                continue
            break
        return args

    def _parse_arg(self) -> Node:
        """Parse one call/subscript argument. Detects named-arg form
        ``IDENT = expr`` (or ``STR = expr``) and turns it into a NamedArg
        without going through the Assign code path."""
        tok0 = self._peek()
        # Named-arg head: IDENT '='  or  STR '='
        if (tok0.kind == "IDENT" or tok0.kind == "STR") and self._peek(1).kind == "=":
            name_tok = self._advance()
            self._advance()  # '='
            value = self.parse_expr(0)
            if name_tok.kind == "IDENT":
                # Strip backticks if present.
                name = name_tok.value
                if name.startswith("`"):
                    name = name[1:-1]
            else:
                name = str(name_tok.cooked)
            return A.NamedArg(name, value, (name_tok.span[0], _end(value)))
        return self.parse_expr(0)

    def _parse_function_def(self, start: int, *, shorthand: bool) -> A.FunctionDef:
        """Parse ``function(params) body`` or ``\\(params) body``."""
        self._expect("(")
        params: list[A.Param] = []
        self._skip_terms()
        if self._peek().kind != ")":
            while True:
                self._skip_terms()
                name_tok = self._expect("IDENT")
                name = name_tok.value
                if name.startswith("`"):
                    name = name[1:-1]
                default: Optional[Node] = None
                if self._accept("="):
                    default = self.parse_expr(0)
                params.append(A.Param(name, default, (name_tok.span[0], _end(default) if default else name_tok.span[1])))
                self._skip_terms()
                if self._peek().kind == ",":
                    self._advance()
                    continue
                break
        self._expect(")")
        body = self.parse_expr(0)
        return A.FunctionDef(tuple(params), body, (start, _end(body)), shorthand=shorthand)

    def _parse_if(self, start: int) -> A.If:
        self._expect("(")
        cond = self.parse_expr(0)
        self._expect(")")
        then = self.parse_expr(0)
        otherwise: Optional[Node] = None
        # ``else`` may legally appear after a TERM when the if-body is a
        # brace block — peek across newlines for it.
        if self._peek_skip_terms().kind == "else":
            self._skip_terms()
            self._advance()  # 'else'
            otherwise = self.parse_expr(0)
        return A.If(cond, then, otherwise, (start, _end(otherwise) if otherwise else _end(then)))

    def _parse_for(self, start: int) -> A.For:
        self._expect("(")
        var_tok = self._expect("IDENT")
        var_name = var_tok.value
        if var_name.startswith("`"):
            var_name = var_name[1:-1]
        self._expect("in")
        iterable = self.parse_expr(0)
        self._expect(")")
        body = self.parse_expr(0)
        return A.For(var_name, iterable, body, (start, _end(body)))

    def _parse_while(self, start: int) -> A.While:
        self._expect("(")
        cond = self.parse_expr(0)
        self._expect(")")
        body = self.parse_expr(0)
        return A.While(cond, body, (start, _end(body)))

    def _parse_repeat(self, start: int) -> A.Repeat:
        body = self.parse_expr(0)
        return A.Repeat(body, (start, _end(body)))


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


def _start(node: Node) -> int:
    return node.span[0]  # type: ignore[union-attr]


def _end(node: Node) -> int:
    return node.span[1]  # type: ignore[union-attr]


def _with_span(node: Node, span: Span) -> Node:
    """Return a copy of ``node`` with ``span`` replaced. Uses dataclasses.replace."""
    from dataclasses import replace
    return replace(node, span=span)


def _component_name_from(tok: Token, src: str) -> str:
    """Extract a `$name` / `@name` RHS. R accepts identifier or string."""
    if tok.kind == "IDENT":
        name = tok.value
        if name.startswith("`"):
            name = name[1:-1]
        return name
    if tok.kind == "STR":
        return str(tok.cooked)
    raise RParseError(f"expected identifier or string after $/@, got {tok.kind!r}", tok.span, src)
