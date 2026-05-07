from .bin import StatBin, stat_bin
from .boxplot import StatBoxplot, stat_boxplot
from .count import StatCount, stat_count
from .density import StatDensity, stat_density
from .ecdf import StatEcdf, stat_ecdf
from .function import geom_function, stat_function
from .identity import StatIdentity, stat_identity
from .qq import StatQq, StatQqLine, geom_qq, geom_qq_line, stat_qq, stat_qq_line
from .smooth import StatSmooth, stat_smooth
from .stat import Stat
from .sum import StatSum, geom_count, stat_sum
from .summary import StatSummary, stat_summary
from .unique import StatUnique, stat_unique
from .ydensity import StatYdensity, stat_ydensity


_NAME_TO_STAT = {
    "identity": StatIdentity,
    "bin": StatBin,
    "count": StatCount,
    "density": StatDensity,
    "smooth": StatSmooth,
    "boxplot": StatBoxplot,
    "ydensity": StatYdensity,
    "summary": StatSummary,
    "qq": StatQq,
    "qq_line": StatQqLine,
    "ecdf": StatEcdf,
    "unique": StatUnique,
    "sum": StatSum,
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
    "StatSummary", "stat_summary",
    "StatQq", "stat_qq", "geom_qq",
    "StatQqLine", "stat_qq_line", "geom_qq_line",
    "StatEcdf", "stat_ecdf",
    "StatUnique", "stat_unique",
    "StatSum", "stat_sum", "geom_count",
    "stat_function", "geom_function",
    "resolve_stat",
]
