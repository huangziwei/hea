"""Label formatters — port of R ``scales``'s ``label_*`` family.

Each factory returns a callable that takes a list of break values and
returns formatted strings. Suitable for ``scale_*_continuous(labels=...)``
which already accepts callables.

Currently implemented:

* :func:`label_number` — generic numeric formatter (the base).
* :func:`label_comma` — number with thousand separators.
* :func:`label_dollar` — currency formatter; despite the name (kept for
  R-API parity), works for ANY currency via ``prefix=`` / ``suffix=``.
* :func:`label_currency` — alias of ``label_dollar`` with no default
  prefix (matches R ``scales`` 1.3+).
* :func:`label_percent` — value × 100 with ``%`` suffix.

All match scales' auto-accuracy rule: if all values are integers (or any
exceeds ``largest_with_cents``), no decimals; else 2 decimals. Override
with ``accuracy=``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _coerce_to_list(values):
    """Accept scalar / iterable / numpy array; return a Python list of
    floats with ``None`` preserved for NaN-likes."""
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, Iterable):
        values = [values]
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if math.isnan(f):
            out.append(None)
        else:
            out.append(f)
    return out


def _auto_accuracy(scaled_values, largest_with_cents):
    """scales' auto-precision rule for currency / number formatters:

    * any |x| >= ``largest_with_cents`` → accuracy 1 (no decimals)
    * else if all values are exact integers → accuracy 1
    * else accuracy 0.01 (two decimals)
    """
    finite = [v for v in scaled_values if v is not None]
    if not finite:
        return 1
    arr = np.asarray(finite, dtype=float)
    if np.any(np.abs(arr) >= largest_with_cents):
        return 1
    if np.all(arr == np.floor(arr)):
        return 1
    return 0.01


def _format_number(x, accuracy, big_mark, decimal_mark):
    """Round ``x`` to ``accuracy`` and render with the requested separators."""
    if accuracy < 1:
        ndigits = int(round(-math.log10(accuracy)))
    else:
        ndigits = 0
    rounded = round(x / accuracy) * accuracy
    if ndigits == 0:
        s = f"{int(round(rounded)):,}"
    else:
        s = f"{rounded:,.{ndigits}f}"
    # Swap default separators (',' and '.') for the requested ones.
    # Use a placeholder so the swap is order-independent.
    if big_mark != "," or decimal_mark != ".":
        s = s.replace(",", "\x00").replace(".", decimal_mark).replace("\x00", big_mark)
    return s


def label_number(*, accuracy=None, scale=1, prefix="", suffix="",
                 big_mark=" ", decimal_mark=".", largest_with_cents=1e5):
    """scales::label_number — generic numeric formatter.

    Parameters
    ----------
    accuracy : float, optional
        Round each number to this multiple. ``None`` (default) auto-picks
        per :func:`_auto_accuracy`.
    scale : float, default 1
        Multiplier applied before formatting (e.g. ``1/1000`` to show
        thousands).
    prefix, suffix : str, default ``""``
        Strings prepended / appended to each formatted number.
    big_mark : str, default ``" "``
        Thousand separator. R's ``scales`` defaults to a space here;
        :func:`label_comma` overrides to ``","``.
    decimal_mark : str, default ``"."``
        Decimal separator. Use ``","`` for European convention.
    largest_with_cents : float, default 1e5
        Auto-accuracy threshold — values at or above this skip decimals.
    """
    def fmt(values):
        vs = _coerce_to_list(values)
        scaled = [v * scale if v is not None else None for v in vs]
        acc = accuracy if accuracy is not None else _auto_accuracy(scaled, largest_with_cents)
        out = []
        for v in scaled:
            if v is None:
                out.append("")
                continue
            negative = v < 0
            base = _format_number(abs(v), acc, big_mark, decimal_mark)
            sign = "-" if negative else ""
            out.append(f"{sign}{prefix}{base}{suffix}")
        return out
    return fmt


def label_comma(*, accuracy=None, scale=1, prefix="", suffix="",
                big_mark=",", decimal_mark=".", largest_with_cents=1e5):
    """scales::label_comma — like :func:`label_number` with comma thousand
    separators by default."""
    return label_number(accuracy=accuracy, scale=scale, prefix=prefix,
                        suffix=suffix, big_mark=big_mark,
                        decimal_mark=decimal_mark,
                        largest_with_cents=largest_with_cents)


def label_dollar(*, accuracy=None, scale=1, prefix="$", suffix="",
                 big_mark=",", decimal_mark=".",
                 largest_with_cents=1e5):
    """scales::label_dollar — currency formatter.

    Despite the name (kept for parity with R's ``scales`` package), this
    is a *generic* currency formatter: pass ``prefix="€"`` for euro,
    ``prefix="£"`` for pound, ``prefix="¥"`` for yen, or
    ``suffix=" zł"`` for złoty. R kept the ``dollar`` name for backwards
    compatibility; :func:`label_currency` is the newer alias (no default
    prefix).

    Default behaviour matches scales' ``largest_with_cents = 1e5`` rule:
    integer breaks render without decimals, fractional ones with two,
    and breaks ≥ 100,000 always omit decimals.

    Examples
    --------
    >>> label_dollar()([326, 5000, 18823])
    ['$326', '$5,000', '$18,823']
    >>> label_dollar(scale=1/1000, suffix="K")([1000, 7000, 13000])
    ['$1K', '$7K', '$13K']
    >>> label_dollar(prefix="", suffix="€")([1234.5])
    ['1,234.50€']
    """
    return label_number(accuracy=accuracy, scale=scale, prefix=prefix,
                        suffix=suffix, big_mark=big_mark,
                        decimal_mark=decimal_mark,
                        largest_with_cents=largest_with_cents)


def label_currency(*, accuracy=None, scale=1, prefix="", suffix="",
                   big_mark=",", decimal_mark=".",
                   largest_with_cents=1e5):
    """scales::label_currency — :func:`label_dollar` with no default prefix.

    Use when you want to be explicit about the currency symbol via
    ``prefix=`` / ``suffix=``."""
    return label_number(accuracy=accuracy, scale=scale, prefix=prefix,
                        suffix=suffix, big_mark=big_mark,
                        decimal_mark=decimal_mark,
                        largest_with_cents=largest_with_cents)


def label_percent(*, accuracy=None, scale=100, prefix="", suffix="%",
                  big_mark="", decimal_mark="."):
    """scales::label_percent — value × 100 with ``%`` suffix.

    Default ``scale=100`` matches R: pass values in [0, 1] and get
    ``"5%"`` etc. Pass ``scale=1`` if your inputs are already in percent.
    """
    return label_number(accuracy=accuracy, scale=scale, prefix=prefix,
                        suffix=suffix, big_mark=big_mark,
                        decimal_mark=decimal_mark)
