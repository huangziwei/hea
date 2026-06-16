"""Statistical models — ``lm``, ``glm``, ``gam``, ``bam``, ``gmm``.

Each model is a port of its R counterpart:

* :func:`lm`   — :func:`stats::lm`   — ordinary least squares
* :func:`glm`  — :func:`stats::glm`  — generalized linear models (Fisher IRLS)
* :func:`gam`  — :func:`mgcv::gam`   — penalized smooth additive models
* :func:`bam`  — :func:`mgcv::bam`   — gam with discrete-covariate speedup
* :func:`gmm`  — :func:`lme4::lmer` / :func:`lme4::glmer` — mixed-effects models
"""

from .bam import bam
from .gam import gam
from .glm import glm
from .lm import lm
from .gmm import gmm

__all__ = ["bam", "gam", "glm", "lm", "gmm"]
