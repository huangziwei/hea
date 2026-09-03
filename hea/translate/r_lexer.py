"""R lexer — character stream → token stream.

Pure-Python, hand-written. Covers the tidyverse-shaped R sublanguage
the translator supports. Out-of-grammar character sequences raise
:class:`RLexError` with a precise span.

Token kinds are short uppercase strings (``"NUM"``, ``"IDENT"``, ``"<-"`` …)
rather than an enum, so error messages and tests can print them naturally
and the parser's dispatch keys read like the operators they match.

Notable R quirks handled here:

- **Newline as terminator** with continuation suppression. A ``\\n`` is
  emitted as a ``TERM`` token *only* when (a) the bracket-depth is zero and
  (b) the last meaningful token was something that could end an expression
  (literal, identifier, ``)``, ``]``, ``]]``, ``}``, ``break``, ``next``).
  Newlines after binary operators, inside parens/brackets, or directly
  after ``,`` are absorbed as whitespace.
- **Maximal munch** for overlapping operators: ``<-`` beats ``<``, ``<<-``
  beats ``<-``, ``<=`` beats ``<``; ``->>`` beats ``->`` beats ``-``;
  ``::`` and ``:::`` win over ``:``; ``|>`` and ``||`` win over ``|``;
  ``&&`` over ``&``; ``==``/``!=`` over ``=``/``!``.
- **Raw strings** (R 4.0+): ``r"(...)"``, ``r"[...]"``, ``r"{...}"`` plus
  the dash-padded forms ``r"-(...)-"``, ``r"---(...)---"``. Single-quoted
  ``R'(...)'`` form is the same.
- **User infix** ``%anything%`` is tokenized as kind ``%infix%`` with the
  full operator text as the value, including ``%>%`` (which the parser
  routes to the pipe production).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

Span = tuple[int, int]


class RLexError(Exception):
    """Raised when the source contains a character sequence the lexer
    cannot classify. ``span`` points at the offending byte range."""

    def __init__(self, message: str, span: Span, source: str):
        self.span = span
        self.source = source
        super().__init__(
            f"{message} at byte {span[0]}: {source[span[0] : span[1] + 1]!r}"
        )


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    value: str  # original source slice
    span: Span
    cooked: object | None = None


def _is_id_start(c: str) -> bool:
    return c.isalpha() or c == "_" or c == "."


def _is_id_cont(c: str) -> bool:
    return c.isalnum() or c == "_" or c == "."


_KEYWORDS = {
    "if": "if",
    "else": "else",
    "for": "for",
    "while": "while",
    "repeat": "repeat",
    "break": "break",
    "next": "next",
    "in": "in",
    "function": "function",
}

_CONSTANT_LITS = {
    "TRUE": ("BOOL", True),
    "FALSE": ("BOOL", False),
    "NULL": ("NULL", None),
    "NA": ("NA", "NA"),
    "NA_integer_": ("NA", "NA_integer_"),
    "NA_real_": ("NA", "NA_real_"),
    "NA_character_": ("NA", "NA_character_"),
    "NA_complex_": ("NA", "NA_complex_"),
    "Inf": ("INF", None),
    "NaN": ("NAN", None),
}

_EXPR_END_KINDS = frozenset(
    {
        "NUM",
        "INT",
        "COMPLEX",
        "STR",
        "BOOL",
        "NULL",
        "NA",
        "INF",
        "NAN",
        "IDENT",
        ")",
        "]",
        "]]",
        "}",
        "break",
        "next",
    }
)


def tokenize(src: str) -> list[Token]:
    """Tokenize ``src`` into a list of :class:`Token`, ending with EOF."""
    lexer = _Lexer(src)
    return lexer.run()


class _Lexer:
    __slots__ = ("bracket_depth", "i", "last_kind", "n", "src", "tokens")

    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.n = len(src)
        self.tokens: list[Token] = []
        self.bracket_depth = 0
        self.last_kind: str | None = None

    def _emit(self, kind: str, start: int, end: int, *, cooked: object | None = None):
        tok = Token(kind, self.src[start:end], (start, end), cooked)
        self.tokens.append(tok)
        self.last_kind = kind

    def run(self) -> list[Token]:
        while self.i < self.n:
            c = self.src[self.i]

            if c == " " or c == "\t":
                self.i += 1
                continue

            if c == "\r":
                self.i += 1
                continue

            if c == "#":
                while self.i < self.n and self.src[self.i] != "\n":
                    self.i += 1
                continue

            if c == "\n":
                start = self.i
                self.i += 1
                if self.bracket_depth == 0 and self.last_kind in _EXPR_END_KINDS:
                    self._emit("TERM", start, start + 1)
                continue

            if c == ";":
                self._emit("TERM", self.i, self.i + 1)
                self.i += 1
                continue

            if (
                (c == "r" or c == "R")
                and self.i + 1 < self.n
                and self.src[self.i + 1] in ('"', "'")
            ) and self._try_raw_string():
                continue

            if c == '"' or c == "'":
                self._scan_string(c)
                continue

            if c == "`":
                self._scan_backtick_ident()
                continue

            if c.isdigit() or (
                c == "." and self.i + 1 < self.n and self.src[self.i + 1].isdigit()
            ):
                self._scan_number()
                continue

            if _is_id_start(c):
                self._scan_ident()
                continue

            if c == "%":
                self._scan_infix()
                continue

            if self._scan_operator():
                continue

            raise RLexError("unexpected character", (self.i, self.i), self.src)

        self._emit("EOF", self.n, self.n)
        return self.tokens

    def _scan_number(self):
        start = self.i
        is_hex = False
        is_int = False
        is_complex = False

        if (
            self.src[self.i] == "0"
            and self.i + 1 < self.n
            and self.src[self.i + 1] in ("x", "X")
        ):
            is_hex = True
            self.i += 2
            while self.i < self.n and self.src[self.i] in "0123456789abcdefABCDEF":
                self.i += 1
        else:
            while self.i < self.n and self.src[self.i].isdigit():
                self.i += 1
            if self.i < self.n and self.src[self.i] == ".":
                self.i += 1
                while self.i < self.n and self.src[self.i].isdigit():
                    self.i += 1
            if self.i < self.n and self.src[self.i] in ("e", "E"):
                self.i += 1
                if self.i < self.n and self.src[self.i] in ("+", "-"):
                    self.i += 1
                if self.i >= self.n or not self.src[self.i].isdigit():
                    raise RLexError("malformed exponent", (start, self.i), self.src)
                while self.i < self.n and self.src[self.i].isdigit():
                    self.i += 1

        if self.i < self.n and self.src[self.i] == "L":
            is_int = True
            self.i += 1
        elif self.i < self.n and self.src[self.i] == "i":
            is_complex = True
            self.i += 1

        text = self.src[start : self.i]

        if is_complex:
            body = text[:-1]
            value = int(body, 16) if is_hex else float(body)
            self._emit("COMPLEX", start, self.i, cooked=float(value))
        elif is_int:
            body = text[:-1]
            if is_hex:
                value = int(body, 16)
            else:
                try:
                    value = int(body)
                except ValueError:
                    value = int(float(body))
            self._emit("INT", start, self.i, cooked=value)
        else:
            if is_hex:
                value = float(int(text, 16))
            else:
                value = float(text)
            self._emit("NUM", start, self.i, cooked=value)

    def _scan_string(self, quote: str):
        start = self.i
        self.i += 1  # opening quote
        buf: list[str] = []
        while self.i < self.n:
            c = self.src[self.i]
            if c == "\\":
                if self.i + 1 >= self.n:
                    raise RLexError(
                        "unterminated string escape", (start, self.i), self.src
                    )
                esc = self.src[self.i + 1]
                self.i += 2
                buf.append(_decode_escape(esc, self, start))
                continue
            if c == quote:
                self.i += 1  # closing quote
                self._emit("STR", start, self.i, cooked="".join(buf))
                return
            buf.append(c)
            self.i += 1
        raise RLexError("unterminated string", (start, self.i - 1), self.src)

    def _try_raw_string(self) -> bool:
        """Try to scan an R 4.0+ raw string starting at ``self.i``. Returns
        True on success and advances ``self.i``; returns False if this looks
        like the start of an identifier ``r`` / ``R`` instead."""
        save = self.i
        prefix = self.src[self.i]  # 'r' or 'R'
        quote = self.src[self.i + 1]  # '"' or "'"
        j = self.i + 2
        dashes = 0
        while j < self.n and self.src[j] == "-":
            dashes += 1
            j += 1
        if j >= self.n or self.src[j] not in "([{":
            return False
        opener = self.src[j]
        closer = {"(": ")", "[": "]", "{": "}"}[opener]
        j += 1
        body_start = j
        end_pattern = closer + ("-" * dashes) + quote
        end_idx = self.src.find(end_pattern, body_start)
        if end_idx < 0:
            raise RLexError("unterminated raw string", (save, self.n - 1), self.src)
        body = self.src[body_start:end_idx]
        self.i = end_idx + len(end_pattern)
        self._emit("STR", save, self.i, cooked=body)
        _ = prefix  # keep for clarity even if unused
        return True

    def _scan_backtick_ident(self):
        start = self.i
        self.i += 1  # opening backtick
        while self.i < self.n and self.src[self.i] != "`":
            if self.src[self.i] == "\n":
                raise RLexError(
                    "unterminated backtick identifier", (start, self.i), self.src
                )
            self.i += 1
        if self.i >= self.n:
            raise RLexError(
                "unterminated backtick identifier", (start, self.i - 1), self.src
            )
        self.i += 1  # closing backtick
        self._emit("IDENT", start, self.i)

    def _scan_ident(self):
        start = self.i
        self.i += 1
        while self.i < self.n and _is_id_cont(self.src[self.i]):
            self.i += 1
        text = self.src[start : self.i]

        if text in _KEYWORDS:
            self._emit(_KEYWORDS[text], start, self.i)
            return

        if text in _CONSTANT_LITS:
            kind, cooked = _CONSTANT_LITS[text]
            self._emit(kind, start, self.i, cooked=cooked)
            return

        self._emit("IDENT", start, self.i)

    def _scan_infix(self):
        start = self.i
        self.i += 1  # leading %
        while self.i < self.n and self.src[self.i] != "%":
            if self.src[self.i] == "\n":
                raise RLexError(
                    "unterminated %infix% operator", (start, self.i), self.src
                )
            self.i += 1
        if self.i >= self.n:
            raise RLexError(
                "unterminated %infix% operator", (start, self.i - 1), self.src
            )
        self.i += 1  # trailing %
        self._emit("%infix%", start, self.i)

    def _scan_operator(self) -> bool:
        """Try to match a punctuation/operator at ``self.i``. Returns True
        on match. Order matters — longest first within each starting char."""
        src = self.src
        i = self.i

        def emit(kind: str, length: int):
            if kind in ("(", "[", "[["):
                self.bracket_depth += 1
            elif kind in (")", "]", "]]"):
                self.bracket_depth = max(0, self.bracket_depth - 1)
            self._emit(kind, i, i + length)
            self.i = i + length

        c = src[i]

        three = src[i : i + 3]
        if three == "<<-":
            emit("<<-", 3)
            return True
        if three == "->>":
            emit("->>", 3)
            return True
        if three == ":::":
            emit(":::", 3)
            return True

        two = src[i : i + 2]
        if two == "<-":
            emit("<-", 2)
            return True
        if two == "->":
            emit("->", 2)
            return True
        if two == "<=":
            emit("<=", 2)
            return True
        if two == ">=":
            emit(">=", 2)
            return True
        if two == "==":
            emit("==", 2)
            return True
        if two == "!=":
            emit("!=", 2)
            return True
        if two == "&&":
            emit("&&", 2)
            return True
        if two == "||":
            emit("||", 2)
            return True
        if two == "|>":
            emit("|>", 2)
            return True
        if two == "::":
            emit("::", 2)
            return True
        if two == "[[":
            emit("[[", 2)
            return True
        if two == "]]":
            emit("]]", 2)
            return True

        SINGLES = {
            "(": "(",
            ")": ")",
            "[": "[",
            "]": "]",
            "{": "{",
            "}": "}",
            ",": ",",
            "$": "$",
            "@": "@",
            "~": "~",
            "?": "?",
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "^": "^",
            "<": "<",
            ">": ">",
            "=": "=",
            "!": "!",
            "&": "&",
            "|": "|",
            ":": ":",
            "\\": "\\",  # R 4.1+ lambda shorthand
        }
        if c in SINGLES:
            emit(SINGLES[c], 1)
            return True

        return False


_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "`": "`",
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
}


def _decode_escape(esc: str, lex: _Lexer, start: int) -> str:
    """Decode a single backslash escape. ``lex.i`` is positioned *after* the
    two consumed characters (``\\`` + ``esc``). ``\\x{NN}``, ``\\u{NNNN}``,
    ``\\U{NNNNNNNN}``, and bare hex/unicode forms are handled by advancing
    ``lex.i`` further as needed."""
    if esc in _SIMPLE_ESCAPES:
        return _SIMPLE_ESCAPES[esc]
    if esc == "x":
        return _decode_hex_escape(lex, start, max_digits=2)
    if esc == "u":
        return _decode_hex_escape(lex, start, max_digits=4)
    if esc == "U":
        return _decode_hex_escape(lex, start, max_digits=8)
    return esc


def _decode_hex_escape(lex: _Lexer, start: int, *, max_digits: int) -> str:
    """Consume up to ``max_digits`` hex chars (optionally brace-wrapped) and
    return the corresponding Unicode character."""
    braced = False
    if lex.i < lex.n and lex.src[lex.i] == "{":
        braced = True
        lex.i += 1
    digits = []
    while (
        len(digits) < max_digits
        and lex.i < lex.n
        and lex.src[lex.i] in "0123456789abcdefABCDEF"
    ):
        digits.append(lex.src[lex.i])
        lex.i += 1
    if not digits:
        raise RLexError("empty hex escape", (start, lex.i), lex.src)
    if braced:
        if lex.i >= lex.n or lex.src[lex.i] != "}":
            raise RLexError(
                "unterminated brace-form hex escape", (start, lex.i), lex.src
            )
        lex.i += 1
    return chr(int("".join(digits), 16))


def iter_tokens(src: str) -> Iterator[Token]:
    """Convenience iterator over :func:`tokenize` for callers that want
    streaming. Note the lexer is one-shot — this just yields from the list."""
    yield from tokenize(src)
