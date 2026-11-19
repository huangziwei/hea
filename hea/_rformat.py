"""R's character representation of a double, as ``as.character`` and
``paste`` produce it.

``coerceToString`` (coerce.c) formats each element on its own with
``R_print.digits = DBL_DIG`` (15): ``formatReal`` picks fixed or scientific
notation and the number of digits from ``scientific`` (format.c), and
``EncodeRealDrop0`` (printutils.c) prints it with trailing zeros dropped.
``scientific`` is the branch R takes where ``long double`` is no wider than
``double``, as on arm64.
"""

from __future__ import annotations

import math

import numpy as np

_TBL = (
    1e00,
    1e01,
    1e02,
    1e03,
    1e04,
    1e05,
    1e06,
    1e07,
    1e08,
    1e09,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18,
    1e19,
    1e20,
    1e21,
    1e22,
)
_KP_MAX = 22
_DBL_DIG = 15
_R_DEC_MIN_EXPONENT = -308


def _scientific(x: float, digits: int) -> tuple[int, int, int, bool]:
    """format.c ``scientific``: ``(neg, kpower, nsig, roundingwidens)`` with
    ``|x| = alpha * 10^kpower``, ``1 <= alpha < 10`` and ``nsig`` the
    significant digits of ``alpha`` rounded to ``digits``."""
    if x == 0.0:
        return 0, 0, 1, False
    neg = 1 if x < 0.0 else 0
    r = -x if neg else x
    if digits >= _DBL_DIG + 1:
        buff = format(r, f"#.{digits - 1}e")
        kpower = int(buff[digits + 2 :])
        i = digits
        while i >= 2:
            if buff[i] != "0":
                break
            i -= 1
        return neg, kpower, i, False
    kp = math.floor(math.log10(r)) - digits + 1
    r_prec = r
    if abs(kp) <= _KP_MAX:
        if kp >= 0:
            r_prec /= _TBL[kp]
        else:
            r_prec *= _TBL[-kp]
    elif kp <= _R_DEC_MIN_EXPONENT:
        r_prec = (r_prec * 1e303) / math.pow(10.0, kp + 303)
    else:
        r_prec /= math.pow(10.0, kp)
    if r_prec < _TBL[digits - 1]:
        r_prec *= 10.0
        kp -= 1
    alpha = float(round(r_prec))
    nsig = digits
    for _ in range(digits):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1
    kpower = kp + digits - 1
    rgt = digits - kpower
    rgt = min(max(rgt, 0), _KP_MAX)
    fuzz = 0.5 / _TBL[rgt]
    roundingwidens = 0 < kpower <= _KP_MAX and r < _TBL[kpower] - fuzz
    return neg, kpower, nsig, roundingwidens


def r_as_character_double(x: float) -> str:
    """``as.character(x)`` for one double ``x`` (``"NaN"`` for every NaN)."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    neg, kpower, nsig, roundingwidens = _scientific(x, _DBL_DIG)
    left = kpower + 1
    if roundingwidens:
        left -= 1
    sleft = neg + (1 if left <= 0 else left)
    rgt = nsig - left
    mxsl = 1 + neg if left < 0 else sleft
    rgt = max(rgt, 0)
    w_f = mxsl + rgt + (rgt != 0)
    e = 2 if (left > 100 or left <= -99) else 1
    d = nsig - 1
    w = neg + (d > 0) + d + 4 + e
    if w_f <= w:
        e = 0
        d = rgt
        w = w_f
    if x == 0.0:
        x = 0.0
    if e:
        s = format(x, f"#{w}.{d}e" if d else f"{w}.{d}e")
    else:
        s = format(x, f"{w}.{d}f")
    dot = s.find(".")
    if dot >= 0:
        replace = dot
        p = dot + 1
        while p < len(s) and "0" <= s[p] <= "9":
            p += 1
            if s[p - 1] != "0":
                replace = p
        if replace != p:
            s = s[:replace] + s[p:]
    return s


def r_as_character(x: np.ndarray) -> list[str]:
    """``as.character`` of a double vector, element by element."""
    return [r_as_character_double(v) for v in np.asarray(x, dtype=float).ravel()]
