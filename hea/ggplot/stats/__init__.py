from .bin import StatBin, stat_bin
from .boxplot import StatBoxplot, stat_boxplot
from .count import StatCount, stat_count
from .density import StatDensity, stat_density
from .identity import StatIdentity, stat_identity
from .smooth import StatSmooth, stat_smooth
from .stat import Stat
from .ydensity import StatYdensity, stat_ydensity


_NAME_TO_STAT = {
    "identity": StatIdentity,
    "bin": StatBin,
    "count": StatCount,
    "density": StatDensity,
    "smooth": StatSmooth,
    "boxplot": StatBoxplot,
    "ydensity": StatYdensity,
}


def resolve_stat(s) -> Stat:
    """Coerce ``s`` to a :class:`Stat` instance.

    Accepts an instance or a string naming one of the built-ins (``"identity"``,
    ``"count"``, ``"bin"``, …). Symmetric to ``positions.resolve_position``."""
    if isinstance(s, Stat):
        return s
    if isinstance(s, str):
        cls = _NAME_TO_STAT.get(s)
        if cls is None:
            raise ValueError(
                f"unknown stat {s!r}; expected one of {sorted(_NAME_TO_STAT)}"
            )
        return cls()
    raise TypeError(f"stat must be a Stat instance or a string, got {type(s).__name__}")


__all__ = [
    "Stat",
    "StatIdentity", "stat_identity",
    "StatBin", "stat_bin",
    "StatCount", "stat_count",
    "StatDensity", "stat_density",
    "StatSmooth", "stat_smooth",
    "StatBoxplot", "stat_boxplot",
    "StatYdensity", "stat_ydensity",
    "resolve_stat",
]
