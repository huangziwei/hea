"""Pin behaviour of `hea.ggplot._measure`.

We don't pin pixel-exact widths (font rendering varies across matplotlib
versions). We pin invariants: empty → 0, longer text → wider, bigger
fontsize → bigger size, rotation swaps width/height.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from hea.ggplot import _measure as m


def test_empty_text_is_zero():
    assert m.text_size_in("") == (0.0, 0.0)
    assert m.text_size_in(None) == (0.0, 0.0)


def test_longer_text_is_wider():
    short_w, _ = m.text_size_in("hi")
    long_w, _ = m.text_size_in("hello world this is longer")
    assert long_w > short_w


def test_bigger_fontsize_grows_both_dims():
    w_small, h_small = m.text_size_in("ABC", fontsize=8.0)
    w_big, h_big = m.text_size_in("ABC", fontsize=24.0)
    assert w_big > w_small
    assert h_big > h_small


def test_rotation_swaps_width_and_height():
    w0, h0 = m.text_size_in("hello", rotation=0.0)
    w90, h90 = m.text_size_in("hello", rotation=90.0)
    # Rotated 90°: original width becomes new height (and vice versa),
    # within a small tolerance for italic/descender effects.
    assert w90 == pytest.approx(h0, rel=0.10)
    assert h90 == pytest.approx(w0, rel=0.10)


def test_text_block_zero_for_empty_lines():
    assert m.text_block_size_in([]) == (0.0, 0.0)
    assert m.text_block_size_in(["", None, ""]) == (0.0, 0.0)


def test_text_block_height_scales_with_lines():
    one_w, one_h = m.text_block_size_in(["one"])
    three_w, three_h = m.text_block_size_in(["one", "two", "three"])
    # Three lines should be roughly 3× the height; width is max of widths.
    assert three_h > 2.5 * one_h
    assert three_w >= one_w


def test_max_label_width_picks_widest():
    w = m.max_label_width_in(["a", "abc", "abcdefg", "ab"])
    w_widest, _ = m.text_size_in("abcdefg", fontsize=m.AXIS_TEXT_SIZE_PT)
    assert w == pytest.approx(w_widest)


def test_max_label_width_empty_is_zero():
    assert m.max_label_width_in([]) == 0.0


def test_colorbar_cell_includes_bar_and_ticks():
    w_bar_only = m.colorbar_cell_width_in([])
    w_with_ticks = m.colorbar_cell_width_in(["0.0", "0.5", "1.0"])
    # Empty tick list returns 0 — the cell collapses entirely.
    assert w_bar_only == 0.0
    assert w_with_ticks > m.COLORBAR_BAR_WIDTH_IN
    # Wider ticks → wider cell.
    w_wide = m.colorbar_cell_width_in(["1234567890"])
    assert w_wide > w_with_ticks


def test_legend_cell_zero_for_no_entries():
    assert m.legend_cell_size_in("title", []) == (0.0, 0.0)


def test_legend_cell_grows_with_entries():
    w, h_few = m.legend_cell_size_in(None, ["a", "b"])
    _, h_many = m.legend_cell_size_in(None, ["a", "b", "c", "d", "e", "f"])
    assert h_many > h_few


def test_legend_title_grows_height():
    _, h_no_title = m.legend_cell_size_in(None, ["a", "b"])
    _, h_with_title = m.legend_cell_size_in("Group", ["a", "b"])
    assert h_with_title > h_no_title


def test_strip_height_includes_padding():
    _, raw_h = m.text_size_in("Setosa", fontsize=m.STRIP_TEXT_SIZE_PT)
    cell_h = m.strip_cell_height_in("Setosa")
    assert cell_h > raw_h
    assert cell_h == pytest.approx(raw_h + 2 * m.STRIP_PAD_IN)


def test_strip_width_for_rotated_label():
    cell_w = m.strip_cell_width_in("Setosa")
    # Rotated 90°, the width of the cell is what was originally height.
    _, raw_h = m.text_size_in("Setosa", fontsize=m.STRIP_TEXT_SIZE_PT, rotation=90.0)
    assert cell_w == pytest.approx(raw_h + 2 * m.STRIP_PAD_IN)
