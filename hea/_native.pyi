"""Type stubs for the compiled ``hea._native`` extension (Rust, via PyO3).

Built from ``rust/`` by maturin. Absent on sdist / no-toolchain installs — callers
must degrade to the pure-Python ``hea.R.nmath`` kernels (see ``hea/R/_dispatch.py``).
"""
from numpy.typing import NDArray
import numpy as np

def pnorm(
    x: NDArray[np.float64],
    mu: float = ...,
    sigma: float = ...,
    lower_tail: bool = ...,
    log_p: bool = ...,
) -> NDArray[np.float64]: ...
