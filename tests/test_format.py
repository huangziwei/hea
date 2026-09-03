"""Tests for the R-style numeric/p-value formatters in hea.utils.

Each case is anchored to what R's ``format()`` / ``format.pval()`` produce
(the targets of these helpers).
"""

from __future__ import annotations

import math

from hea.utils import format_pval, format_signif, format_signif_jointly


def test_signif_jointly_smaller_drives_decimals():
    est, se = format_signif_jointly([[470.4444], [4.0817]], digits=4)
    assert est == ["470.444"]
    assert se == ["4.082"]


def test_signif_single_element():
    assert format_signif([115.2562], digits=4) == ["115.3"]


def test_signif_switches_to_scientific_below_1e_minus_4():
    out = format_signif([1e-7, 5.0], digits=4)
    assert out[0].endswith("e-07")
    assert out[1] == "5.000"


def test_signif_handles_none_nan_inf():
    out = format_signif([1.5, None, math.nan, math.inf, -math.inf, 0.0], digits=4)
    assert out == ["1.500", "", "NaN", "Inf", "-Inf", "0.000"]


def test_pval_caps_below_machine_eps():
    out = format_pval([0.0], digits=4)
    assert out[0].startswith("<")
    assert out[0].lstrip("<").lstrip() == "2.2e-16"


def test_pval_printcoefmat_style_eps_display():
    out = format_pval([0.0], digits=3)
    assert out[0] == "<2e-16"


def test_pval_mixed_with_eps_and_big():
    out = format_pval([1e-300, 0.000213, 0.5], digits=3)
    assert out[0] == "< 2e-16"
    assert out[1] == "0.000213"
    assert out[2] == "0.500000"


def test_pval_scientific_below_1e_minus_4():
    out = format_pval([3.8e-11], digits=4)
    assert out[0] == "3.800e-11"
