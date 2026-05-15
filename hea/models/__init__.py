"""Statistical models — ``lm``, ``glm``, ``gam``, ``bam``, ``lme``.

Each model is a port of its R counterpart:

* :func:`lm`   — :func:`stats::lm`   — ordinary least squares
* :func:`glm`  — :func:`stats::glm`  — generalized linear models (Fisher IRLS)
* :func:`gam`  — :func:`mgcv::gam`   — penalized smooth additive models
* :func:`bam`  — :func:`mgcv::bam`   — gam with discrete-covariate speedup
* :func:`lme`  — :func:`lme4::lmer` / :func:`lme4::glmer` — mixed-effects models
"""

from .bam import bam
from .gam import gam
from .glm import glm
from .lm import lm
from .lme import lme

__all__ = ["bam", "gam", "glm", "lm", "lme"]
