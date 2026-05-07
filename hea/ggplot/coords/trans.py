"""``coord_trans()`` — display-only axis transform.

ggplot2's distinction: ``scale_x_log10()`` transforms the data BEFORE
stats see it (so smoothers fit in log space); ``coord_trans(x="log10")``
transforms only the *display*, so stats fit in raw space.

Implementation: install the named transform on matplotlib's axis at
render time via the same machinery the existing :class:`Log10Trans` /
:class:`Log2Trans` / :class:`SqrtTrans` / :class:`ReverseTrans` use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cartesian import CoordCartesian


_NAMED_TRANS = {
    "identity": "IdentityTrans",
    "log10": "Log10Trans",
    "log2": "Log2Trans",
    "sqrt": "SqrtTrans",
    "reverse": "ReverseTrans",
}


def _resolve_trans(t):
    if t is None:
        return None
    if isinstance(t, str):
        from ..scales import transformed
        cls = _NAMED_TRANS.get(t)
        if cls is None:
            raise ValueError(
                f"coord_trans: unknown transform {t!r}; "
                f"valid: {sorted(_NAMED_TRANS)}"
            )
        return getattr(transformed, cls)()
    if hasattr(t, "matplotlib_scale"):
        return t
    raise TypeError(
        f"coord_trans: transform must be a string name or Trans instance, "
        f"got {type(t).__name__}"
    )


@dataclass
class CoordTrans(CoordCartesian):
    """Display-only axis transform. Stats still see raw coordinates."""

    x: Any = None  # transform name or Trans instance
    y: Any = None

    def apply_to_axes(self, ax) -> None:
        super().apply_to_axes(ax)
        for axis_name, trans_spec in (("x", self.x), ("y", self.y)):
            t = _resolve_trans(trans_spec)
            if t is None:
                continue
            ms = t.matplotlib_scale()
            if ms is None:
                continue
            scale_name, kwargs = ms
            if axis_name == "x":
                ax.set_xscale(scale_name, **kwargs)
            else:
                ax.set_yscale(scale_name, **kwargs)


def coord_trans(*, x=None, y=None, xlim=None, ylim=None, expand=True):
    """Apply a display-only transform to the x and/or y axis.

    ``x`` / ``y`` accept a string name (``"log10"``, ``"log2"``, ``"sqrt"``,
    ``"reverse"``, ``"identity"``) or a :class:`Trans` instance.
    """
    return CoordTrans(x=x, y=y, xlim=xlim, ylim=ylim, expand=expand)
