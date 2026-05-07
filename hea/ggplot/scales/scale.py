"""Base :class:`Scale` — abstract enough for both continuous and discrete scales.

A scale is the contract between *data values* and *axis appearance*. For
positional scales (x, y) on a Cartesian coord, the contract is mostly about
ticks: matplotlib autoscales the limits from artist extents, and the scale
contributes break positions + tick labels. Phase 1.1 ships only
:class:`ScaleContinuous`; discrete and non-positional (colour, fill, size,
shape) scales arrive in 1.5 / 1.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Scale:
    aesthetics: tuple[str, ...] = ()
    name: str | None = None
    # ``breaks``: "default" | None (no ticks) | array-like | callable
    breaks: Any = "default"
    # ``labels``: "default" | array-like | callable(breaks) -> list[str]
    labels: Any = "default"
    limits: tuple | None = None
    expand: tuple = (0.05, 0.0)

    def train(self, data) -> None:
        """Update internal state from a column of values. Default: no-op.

        Subclasses override to track ranges (continuous), levels (discrete),
        or whatever else they need at map-time."""

    def map(self, data):
        """Translate raw data values into drawable values. Default: identity.

        Override in non-positional scales (colour, fill, shape, size) where
        the value the geom draws differs from the value in the data."""
        return data

    def apply_to_axis(self, ax, axis: str) -> None:
        """Push limits/breaks/labels onto ``ax``. Default: no-op.

        Only positional scales (``ScaleContinuous`` for now) override this."""


def fmt_number(x: float) -> str:
    """Format a numeric break for display.

    Phase 1.1 form: integer if exact, otherwise ``%g``. Full
    ``scales::label_number`` parity (thousands separators, accuracy=,
    big.mark, etc.) lands in Phase 3 alongside guides.
    """
    if x == int(x):
        return str(int(x))
    return f"{x:g}"


def format_breaks(breaks) -> list[str]:
    """Format a vector of axis breaks the way R's ``format()`` does.

    Per-axis (not per-value): pick fixed vs scientific by comparing the
    maximum width across all breaks; scientific wins only when its max
    is strictly smaller — matching R's default ``scipen = 0``. Without
    this, a small-magnitude axis (e.g. density y at 0..0.001) would
    keep ``0.00025`` while ggplot2 has long since switched to
    ``2.5e-04``.

    Padding rules mirror R: in fixed mode every entry gets enough
    decimals to represent the most-precise break exactly (so ``0`` is
    written as ``0.0000`` alongside ``0.0003``); in scientific mode
    each entry uses just enough mantissa to round-trip and the
    exponent is written ``±NN``.
    """
    import math

    finite = [b for b in breaks
              if b is not None and isinstance(b, (int, float))
              and math.isfinite(float(b))]
    if not finite:
        return [str(b) for b in breaks]

    fixed = _fmt_fixed_vector(finite)
    sci = _fmt_sci_vector(finite)

    use_sci = max(len(s) for s in sci) < max(len(s) for s in fixed)
    return sci if use_sci else fixed


def _fmt_fixed_vector(values) -> list[str]:
    """All values formatted with a common decimal count.

    Mirrors R's vector ``format(scientific=FALSE)`` behaviour: pad each
    entry so the column lines up on the decimal point.
    """
    decimals = max((_decimals_needed(v) for v in values), default=0)
    if decimals == 0:
        out = []
        for v in values:
            iv = int(v)
            out.append(str(iv) if iv == v else f"{v:g}")
        return out
    return [f"{float(v):.{decimals}f}" for v in values]


def _decimals_needed(v: float) -> int:
    """How many decimals are needed to write ``v`` exactly in fixed form."""
    if v == 0 or v == int(v):
        return 0
    s = f"{float(v):.15g}"
    if "e" in s or "E" in s:
        mant, _, exp = s.lower().partition("e")
        mant_dec = len(mant.split(".")[-1]) if "." in mant else 0
        return max(0, mant_dec - int(exp))
    return len(s.split(".")[-1]) if "." in s else 0


def _fmt_sci_vector(values) -> list[str]:
    """Each value in scientific notation, R-style (``3e-04`` / ``1e+07``)."""
    return [_fmt_sci_one(float(v)) for v in values]


def _fmt_sci_one(v: float) -> str:
    if v == 0:
        return "0e+00"
    # Find the smallest mantissa precision that round-trips to v exactly.
    for prec in range(1, 16):
        s = f"{v:.{prec - 1}e}"
        if float(s) == v:
            break
    else:
        s = f"{v:.14e}"
    return _normalise_exponent(s)


def _normalise_exponent(s: str) -> str:
    """Trim mantissa trailing zeros and pad the exponent to two digits.

    ``2.5000e-04`` → ``2.5e-04``; ``3e-4`` → ``3e-04``. Matches R's
    default scientific output width.
    """
    mant, _, exp = s.partition("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    sign = "-" if exp.startswith("-") else "+"
    exp_digits = exp.lstrip("+-").lstrip("0") or "0"
    if len(exp_digits) < 2:
        exp_digits = exp_digits.zfill(2)
    return f"{mant}e{sign}{exp_digits}"
