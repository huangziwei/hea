"""Round-trip tests for the R ↔ Python translator.

Single-direction tests (``test_translate_r.py`` / ``test_translate_py.py``)
verify each translator output in isolation. These tests exercise the
**round-trip contract**: a snippet that goes forward + back should
reproduce the original (modulo cosmetic differences — imports,
library() preambles, whitespace).

Round-trips catch shape drift that's invisible to single-direction
tests: e.g. ``m.summary()`` round-trips to ``summary(m)`` to
``summary(m)`` if R-generic method form isn't restored.

Tests for currently-broken cases are marked ``xfail(strict=True)`` so
they fail-on-pass when the fix lands — at which point the marker
should be removed.
"""

from __future__ import annotations

import re

import pytest

from hea.translate import from_R, to_R

_PY_PREAMBLE = re.compile(
    r"^("
    r"import\s+hea\b"
    r"|import\s+numpy\b"
    r"|from\s+hea(\.[\w.]+)?\s+import\b"
    r")"
)
_R_PREAMBLE = re.compile(r"^library\(")


def _strip_py(src: str) -> str:
    out = [
        line
        for line in src.strip().splitlines()
        if line.strip() and not _PY_PREAMBLE.match(line)
    ]
    return "\n".join(out).strip()


def _strip_r(src: str) -> str:
    out = [
        line
        for line in src.strip().splitlines()
        if line.strip() and not _R_PREAMBLE.match(line)
    ]
    return "\n".join(out).strip()


def _py_r_py(py_src: str) -> str:
    """Python → R → Python; returns the stripped Python body. Uses the
    nested-wrap shape — ``from_R`` unwraps a ``Result`` automatically."""
    return _strip_py(from_R(to_R(py_src)).source)


def _r_py_r(r_src: str) -> str:
    """R → Python → R; returns the stripped R body."""
    return _strip_r(to_R(from_R(r_src)).source)


class TestModelGenericRoundtrip:
    """``m.summary()`` (hea Py) ↔ ``summary(m)`` (R) — round-trip both
    ways. R uses S3 / S4 generic dispatch (``summary(m)``); hea Python
    exposes them as methods on the fitted model (``m.summary()``).

    Canonical form is the method form: starting from ``summary(m)``
    (function form) round-trips to ``m.summary()`` because from_R always
    method-form-ifies known single-arg generics when the first arg is a
    bare identifier. The asymmetry is intentional — method form is the
    hea Python idiom.
    """

    def test_summary_method_py_r_py(self):
        assert _py_r_py("m.summary()") == "m.summary()"

    def test_anova_method_py_r_py(self):
        assert _py_r_py("m.anova()") == "m.anova()"

    def test_coef_method_py_r_py(self):
        assert _py_r_py("m.coef()") == "m.coef()"

    def test_summary_function_form_normalises_to_method(self):
        assert _py_r_py("summary(m)") == "m.summary()"

    def test_anova_two_models_py_r_py(self):
        assert _py_r_py("anova(m1, m2)") == "anova(m1, m2)"


class TestSliceRoundtrip:
    """Forward maps R's 1-based slice positions to hea's 0-based and routes
    the negative (drop) form through the ``drop(...)`` marker; the reverse
    undoes both, so a slice survives a full round-trip."""

    def test_keep_r_py_r(self):
        assert _r_py_r("df |> slice(c(1, 3, 5))") == "df |>\n  slice(c(1, 3, 5))"

    def test_range_r_py_r(self):
        assert _r_py_r("df |> slice(1:3)") == "df |>\n  slice(1:3)"

    def test_last_row_r_py_r(self):
        assert _r_py_r("df |> slice(n())") == "df |>\n  slice(n())"

    def test_drop_negative_r_py_r(self):
        assert _r_py_r("df |> slice(-c(1, 2))") == "df |>\n  slice(-c(1, 2))"
        assert _r_py_r("df |> slice(-(1:2))") == "df |>\n  slice(-(1:2))"

    def test_keep_py_r_py(self):
        assert _py_r_py("df.slice([0, 2, 4])") == "df.slice([0, 2, 4])"

    def test_drop_py_r_py(self):
        assert _py_r_py("df.slice(drop([0, 1]))") == "df.slice(drop([0, 1]))"


class TestRDollarCall:
    """``(obj$method)()`` and ``obj$method()`` in R are calls of a
    function-valued list slot. from_R must reverse to Python attribute
    access ``obj.method(args)`` — NOT subscript ``obj["method"](args)``,
    since the latter doesn't work for hea / polars (string-keyed
    function values aren't a thing on a DataFrame).
    """

    def test_dollar_call_paren_form(self):
        py_out = from_R("(m$summary)()").source
        assert "m.summary()" in py_out

    def test_dollar_call_no_paren(self):
        py_out = from_R("m$summary()").source
        assert "m.summary()" in py_out

    def test_dollar_no_call_stays_subscript(self):
        py_out = from_R("df$col").source
        assert "df['col']" in py_out or 'df["col"]' in py_out


class TestDataLoadRoundtrip:
    """``X = hea.data("X", package="P")`` (Py) must round-trip via the
    explicit ``data("X", package = "P")`` R call.

    Currently ``to_R`` rewrites the Python assignment to a bare
    ``library(P)`` call, dropping the dataset name. The reverse has
    nothing to bind to, so the round-trip loses ``X``.

    Both rdatasets-registered packages (palmerpenguins) and
    bundled-only ones (faraway) hit the same to_R drop, so both xfail
    until to_R emits the explicit ``data(...)`` form.
    """

    def test_faraway_gala(self):
        py_src = "gala = data('gala', package='faraway')\n"
        out = _py_r_py(py_src)
        assert "gala" in out and "data(" in out and "faraway" in out

    def test_faraway_gala_with_model(self):
        py_src = (
            "from hea.models import lm\n"
            "gala = data('gala', package='faraway')\n"
            "m0 = lm('Species ~ Area + Elevation', gala)\n"
        )
        out = _py_r_py(py_src)
        lines = out.splitlines()
        gala_line = next(
            (i for i, ln in enumerate(lines) if "gala" in ln and "data" in ln), -1
        )
        m0_line = next(
            (i for i, ln in enumerate(lines) if ln.lstrip().startswith("m0")), -1
        )
        assert gala_line >= 0, f"gala autoload missing in: {out!r}"
        assert m0_line >= 0, f"m0 binding missing in: {out!r}"
        assert gala_line < m0_line, f"gala must load before m0: {out!r}"

    def test_palmerpenguins_round_trip(self):
        py_src = "penguins = data('penguins', package='palmerpenguins')\n"
        out = _py_r_py(py_src)
        assert "penguins" in out and "palmerpenguins" in out


class TestToRExecuteNonFrameResult:
    """``to_R(..., execute=True)`` on a script whose final value is NOT
    a data.frame (model fit, ``summary.lm``, list, …) used to throw
    ``RuntimeError: R: cannot serialize result``. Now the R driver
    auto-prints the value and the captured stdout is returned as
    :class:`RConsoleOutput` — a ``str`` subclass that renders raw in
    Jupyter / IPython (``repr`` returns the text unquoted, with a
    ``<pre>``-wrapped ``_repr_html_``).
    """

    def test_summary_returns_console_output(self):
        pytest.importorskip("polars")
        import shutil

        if shutil.which("R") is None:
            pytest.skip("R binary not available")
        from hea.translate.inline import RConsoleOutput

        result = to_R(
            "from hea.models import lm\n"
            "gala = data('gala', package='faraway')\n"
            "m0 = lm('Species ~ Area', gala)\n"
            "m0.summary()\n",
            execute=True,
        )
        assert isinstance(result.value, RConsoleOutput)
        assert "\\n" not in repr(result.value)
        assert "Coefficients:" in result.value

    def test_console_output_str_behavior(self):
        from hea.translate.inline import RConsoleOutput

        s = RConsoleOutput("hello\nworld")
        assert len(s) == 11
        assert s.split() == ["hello", "world"]
        assert repr(s) == "hello\nworld"
        assert s._repr_html_() == "<pre>hello\nworld</pre>"

    def test_missing_R_raises_clear_error(self, monkeypatch):
        from hea.translate.runner import RNotFoundError

        monkeypatch.setenv("PATH", "/nonexistent")
        with pytest.raises(RNotFoundError) as excinfo:
            to_R("x = 1\n", execute=True)
        msg = str(excinfo.value)
        assert "R" in msg and "PATH" in msg
        assert "cran.r-project.org" in msg or "package manager" in msg


class TestResultWrapping:
    """``from_R``/``to_R`` accept either a string OR a :class:`Result`
    from a prior call. Lets users (and tests) write the round-trip
    pattern as nested calls — ``to_R(from_R(r_src))`` — instead of
    ``to_R(from_R(r_src).source)``.
    """

    def test_to_R_accepts_from_R_result(self):
        r_src = (
            'data("gala", package = "faraway")\n'
            'm0 <- lm("Species ~ Area", gala)\n'
            "summary(m0)\n"
        )
        out = to_R(from_R(r_src))
        assert out.source.strip() == r_src.strip()

    def test_from_R_accepts_to_R_result(self):
        py_src = (
            "from hea import data\n"
            "from hea.models import lm\n"
            "gala = data('gala', package='faraway')\n"
            "m0 = lm('Species ~ Area', gala)\n"
            "m0.summary()\n"
        )
        out = from_R(to_R(py_src))
        assert out.source.strip() == py_src.strip()

    def test_result_unwrap_preserves_source_only(self):
        from hea.translate.inline import Result

        r_src = "x <- 1"
        synthetic = Result(value="ignored", source=r_src, gaps=[])
        out = from_R(synthetic)
        assert out.source.strip() == "x = 1"


class TestBundledDatasetAutoload:
    """``library(faraway); gala`` (R) must autoload ``gala = hea.data(...)``
    when from_R is called standalone (not as part of a round-trip).

    The autoload covers both rdatasets-registered packages and the
    bundled-CSV tree (datasets/<pkg>/), so users translating R scripts
    that use faraway / gamair / lme4-extras datasets get a runnable
    Python translation.
    """

    def test_faraway_gala_via_library(self):
        r_src = "library(faraway)\nm0 <- lm(Species ~ Area, gala)\n"
        py = from_R(r_src).source
        assert "gala = data('gala', package='faraway')" in py

    def test_faraway_gala_with_dollar_call(self):
        r_src = (
            "library(faraway)\n"
            'm0 <- lm("Species ~ Area + Elevation + Nearest + Scruz + Adjacent", gala)\n'
            "(m0$summary)()\n"
        )
        py = from_R(r_src).source
        assert "gala = data('gala', package='faraway')" in py
        assert "m0.summary()" in py

    def test_no_autoload_when_package_not_loaded(self):
        r_src = "head(penguins)\n"
        py = from_R(r_src).source
        assert "data('penguins'" not in py


class TestUserReportedCase:
    """A full user snippet — Py → R → Py must reproduce the model fit +
    summary call without losing the dataset binding or mangling the method
    call.
    """

    def test_gala_lm_summary(self):
        py_src = (
            "from hea.models import lm\n"
            "gala = data('gala', package='faraway')\n"
            "m0 = lm('Species ~ Area + Elevation + Nearest + Scruz + Adjacent', gala)\n"
            "m0.summary()\n"
        )
        out = _py_r_py(py_src)
        assert "gala" in out
        assert "lm(" in out
        assert "m0.summary()" in out


class TestMixedModelRoundtrip:
    """lme4's ``lmer``/``glmer`` survive R→hea→R: the function name (via the
    ``gmm`` fold and family-based reverse dispatch), the parenthesized
    random-effect bar, and the R family generator name."""

    def test_glmer_poisson_r_py_r(self):
        out = _r_py_r("glmer(y ~ x + (1|g), family=poisson, data=d)")
        assert out.startswith("glmer(")
        assert "(1 | g)" in out
        assert "family = poisson" in out

    def test_lmer_random_slope_r_py_r(self):
        out = _r_py_r("lmer(y ~ x + (1+x|g), data=d)")
        assert out.startswith("lmer(")
        assert "(1 + x | g)" in out
        assert "glmer(" not in out

    def test_glmer_binomial_cbind_r_py_r(self):
        out = _r_py_r("glmer(cbind(s, f) ~ x + (1|g), family=binomial, data=d)")
        assert out.startswith("glmer(")
        assert "cbind(s, f)" in out
        assert "(1 | g)" in out
        assert "family = binomial" in out
        assert "[" not in out
