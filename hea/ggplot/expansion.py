"""``expansion()`` — fine-grained scale expansion control.

ggplot2's ``expansion(mult, add)`` produces a structured padding spec for
a scale's ``expand`` argument. Each component can be a scalar (symmetric
low/high) or a 2-tuple ``(low, high)`` (asymmetric). The interpretation:

::

    final_low  = data_min - mult_low  * range - add_low
    final_high = data_max + mult_high * range + add_high

Currently :class:`ScaleContinuous` accepts a tuple ``(mult, add)`` for
``expand``; this helper produces a typed :class:`Expansion` that the
renderer can introspect for asymmetric padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Expansion:
    """Padding around a scale's data range.

    ``mult`` — multiplicative padding as a fraction of the data range.
    Scalar applies symmetrically; ``(low, high)`` is asymmetric.

    ``add`` — additive padding in data units. Scalar applies symmetrically;
    ``(low, high)`` is asymmetric.
    """

    mult: Any = 0.05
    add: Any = 0.0

    def split(self) -> tuple[float, float, float, float]:
        """Return ``(mult_low, mult_high, add_low, add_high)`` — always
        4-scalar form, scalars broadcast to (val, val)."""
        m_lo, m_hi = self._pair(self.mult)
        a_lo, a_hi = self._pair(self.add)
        return m_lo, m_hi, a_lo, a_hi

    @staticmethod
    def _pair(v) -> tuple[float, float]:
        if isinstance(v, (int, float)):
            return (float(v), float(v))
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        raise ValueError(
            f"expansion: expected scalar or (low, high) tuple, got {v!r}"
        )


def expansion(mult=0.05, add=0.0) -> Expansion:
    """Build an :class:`Expansion` for the ``expand=`` argument of a scale.

    Examples::

        scale_y_continuous(expand=expansion(mult=0.05))      # symmetric 5%
        scale_x_continuous(expand=expansion(mult=(0, 0.1)))  # 0% low, 10% high
        scale_y_continuous(expand=expansion(add=(0, 1)))     # 1 unit at top
    """
    return Expansion(mult=mult, add=add)
