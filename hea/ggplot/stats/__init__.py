from .bin import StatBin, stat_bin
from .bin2d import StatBin2d, StatBinhex, stat_bin_2d, stat_binhex
from .boxplot import StatBoxplot, stat_boxplot
from .count import StatCount, stat_count
from .density import StatDensity, stat_density
from .density_ridges import StatDensityRidges, stat_density_ridges
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
    "bin_2d": StatBin2d,
    "binhex": StatBinhex,
    "count": StatCount,
    "density": StatDensity,
    "density_ridges": StatDensityRidges,
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
    "StatBin",
    "StatBin2d",
    "StatBinhex",
    "StatBoxplot",
    "StatCount",
    "StatDensity",
    "StatDensityRidges",
    "StatEcdf",
    "StatIdentity",
    "StatQq",
    "StatQqLine",
    "StatSmooth",
    "StatSum",
    "StatSummary",
    "StatUnique",
    "StatYdensity",
    "geom_count",
    "geom_function",
    "geom_qq",
    "geom_qq_line",
    "resolve_stat",
    "stat_bin",
    "stat_bin_2d",
    "stat_binhex",
    "stat_boxplot",
    "stat_count",
    "stat_density",
    "stat_density_ridges",
    "stat_ecdf",
    "stat_function",
    "stat_identity",
    "stat_qq",
    "stat_qq_line",
    "stat_smooth",
    "stat_sum",
    "stat_summary",
    "stat_unique",
    "stat_ydensity",
]
