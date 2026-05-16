"""tidyverse's ``lubridate`` parsers and clock primitives.

* ``today`` / ``now`` — current date / datetime, optional ``tzone=``.
* ``ymd`` / ``mdy`` / ``dmy`` — date parsers.
* ``ymd_hms`` / ``ymd_hm`` / ``mdy_hms`` / ``mdy_hm`` / ``dmy_hms`` /
  ``dmy_hm`` — date+time parsers.

All parsers accept scalars, polars Series, lists / ndarrays, and integer
shorthand (``ymd(20130102)``); return shape mirrors input shape.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def today(tzone: str = ""):
    """lubridate: ``today(tzone="")`` — current date.

    ``tzone=""`` (R default) uses the system local time; pass a tzdata
    name (``"UTC"``, ``"Asia/Tokyo"``) for an explicit zone.
    """
    import datetime as _dt

    if not tzone:
        return _dt.date.today()
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo(tzone)).date()
    except Exception:
        return _dt.date.today()


def _parse_lubridate(value, order: str, with_time: bool, tz: str = ""):
    """Parse a date / datetime according to a lubridate-style order string.

    ``order`` is one of ``"ymd"`` / ``"mdy"`` / ``"dmy"`` (component
    order). Uses ``dateutil.parser`` with the matching ``dayfirst`` /
    ``yearfirst`` hint, which covers stringr's wide format tolerance
    (separators ``-`` / ``/`` / ``.`` / ``,`` / spaces; month names full
    or abbreviated; ordinal suffixes ``1st`` / ``2nd``).

    Works on scalars, polars Series, lists / numpy arrays. Output mirrors
    the input shape: a ``date`` (or ``datetime`` when ``with_time``) for
    scalars; a Python list for list/Series/ndarray inputs.
    """
    import re as _re
    from dateutil import parser as _du

    dayfirst = order == "dmy"
    yearfirst = order == "ymd"

    def _coerce(v):
        # ``ymd(20130102)`` — 8-digit numeric form maps to the canonical string.
        if isinstance(v, (int, float)):
            return str(int(v))
        return v

    def _parse_one(text):
        if text is None:
            return None
        text = _coerce(text)
        if not isinstance(text, str):
            return None
        text = text.strip()
        # Strip English ordinal suffix (``January 31st`` → ``January 31``).
        text = _re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
        try:
            dt = _du.parse(text, dayfirst=dayfirst, yearfirst=yearfirst)
        except (ValueError, _du.ParserError) as e:
            raise ValueError(
                f"{order}{'_hms' if with_time else ''}(): could not parse {text!r}"
            ) from e
        return dt if with_time else dt.date()

    if isinstance(value, pl.Series):
        out = [_parse_one(v) for v in value]
        return pl.Series(value.name, out)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_parse_one(v) for v in value]
    # Scalar return: numpy datetime64 so ``+`` with int/ndarray works the
    # R way (``ymd('2022-01-01') + 5`` → ``2022-01-06``;
    # ``ymd('2022-01-01') + np.array([1, 2])`` → array of two dates).
    # Python's bare ``datetime.date`` rejects ``+ int``.
    scalar = _parse_one(value)
    if scalar is None:
        return None
    unit = "us" if with_time else "D"
    return np.datetime64(scalar, unit)


def ymd(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``ymd()`` — parse year-month-day. Accepts strings,
    integers (``20130102``), polars Series, and lists.
    """
    return _parse_lubridate(value, "ymd", with_time=False, tz=tz)


def mdy(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``mdy()`` — parse month-day-year."""
    return _parse_lubridate(value, "mdy", with_time=False, tz=tz)


def dmy(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``dmy()`` — parse day-month-year."""
    return _parse_lubridate(value, "dmy", with_time=False, tz=tz)


def ymd_hms(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``ymd_hms()`` — parse year-month-day H:M:S."""
    return _parse_lubridate(value, "ymd", with_time=True, tz=tz)


def ymd_hm(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``ymd_hm()`` — parse year-month-day H:M."""
    return _parse_lubridate(value, "ymd", with_time=True, tz=tz)


def mdy_hms(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``mdy_hms()`` — parse month-day-year H:M:S."""
    return _parse_lubridate(value, "mdy", with_time=True, tz=tz)


def mdy_hm(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``mdy_hm()`` — parse month-day-year H:M."""
    return _parse_lubridate(value, "mdy", with_time=True, tz=tz)


def dmy_hms(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``dmy_hms()`` — parse day-month-year H:M:S."""
    return _parse_lubridate(value, "dmy", with_time=True, tz=tz)


def dmy_hm(value, *, tz: str = "", quiet: bool = False, truncated: int = 0):
    """lubridate: ``dmy_hm()`` — parse day-month-year H:M."""
    return _parse_lubridate(value, "dmy", with_time=True, tz=tz)


def now(tzone: str = ""):
    """lubridate: ``now(tzone="")`` — current datetime.

    ``tzone=""`` (R default) returns a naive local-time ``datetime``;
    pass a tzdata name for a tz-aware ``datetime``.
    """
    import datetime as _dt

    if not tzone:
        return _dt.datetime.now()
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo(tzone))
    except Exception:
        return _dt.datetime.now()
